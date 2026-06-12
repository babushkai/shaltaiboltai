use crate::config::Config;
use crate::images;
use crate::providers::{
    self, ChatEvent, ChatRequest, ImageData, Message, ModelEntry, ProviderKind, ToolCall, Usage,
    UserContent,
};
use crate::session;
use crate::theme::{self, Theme};
use crate::tools;
use ratatui::style::Style;
use ratatui::text::Line;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tui_textarea::TextArea;

/// Hard cap on consecutive model→tool→model rounds for a single user input,
/// so a confused model cannot loop forever.
const MAX_AGENT_TURNS: usize = 30;

/// Per-message cap when flattening history for the compaction summary, so the
/// summary request itself stays small.
const COMPACT_FLATTEN_CAP: usize = 4_000;

/// Cap on project instruction files injected into the system prompt.
const PROJECT_CONTEXT_CAP: usize = 8_000;

/// The slash-command registry: drives the `/` completion menu, `/help`, and
/// dispatch, so the three can never drift apart.
pub struct SlashCommand {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// Argument hint shown in the menu and /help, e.g. `[name]`.
    pub args: Option<&'static str>,
    pub description: &'static str,
}

pub const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "model",
        aliases: &["models"],
        args: Some("[name]"),
        description: "switch model — also Ctrl+P",
    },
    SlashCommand {
        name: "theme",
        aliases: &["themes"],
        args: Some("[name]"),
        description: "choose a color theme (live preview)",
    },
    SlashCommand {
        name: "resume",
        aliases: &["sessions"],
        args: None,
        description: "resume a saved session",
    },
    SlashCommand {
        name: "new",
        aliases: &["clear"],
        args: None,
        description: "start a new session (current one stays saved)",
    },
    SlashCommand {
        name: "compact",
        aliases: &[],
        args: None,
        description: "summarize the conversation to shrink context",
    },
    SlashCommand {
        name: "help",
        aliases: &[],
        args: None,
        description: "show commands and keys",
    },
    SlashCommand {
        name: "quit",
        aliases: &["exit"],
        args: None,
        description: "exit shaltaiboltai",
    },
];

/// Prefix-match commands (name first, then aliases) for the `/` menu.
pub fn match_commands(filter: &str) -> Vec<&'static SlashCommand> {
    let mut by_name: Vec<_> = SLASH_COMMANDS
        .iter()
        .filter(|c| c.name.starts_with(filter))
        .collect();
    let by_alias = SLASH_COMMANDS
        .iter()
        .filter(|c| !c.name.starts_with(filter))
        .filter(|c| c.aliases.iter().any(|a| a.starts_with(filter)));
    by_name.extend(by_alias);
    by_name
}

/// Events delivered to the UI loop from background tasks. `gen` ties an event
/// to the request generation that spawned it; events from a cancelled
/// generation are dropped instead of resurrecting the agent loop.
pub enum AppEvent {
    Chat {
        gen: u64,
        event: ChatEvent,
    },
    ModelsDiscovered(Vec<ModelEntry>),
    ToolFinished {
        gen: u64,
        call: ToolCall,
        content: String,
        is_error: bool,
    },
    CompactionDone {
        session_id: String,
        result: Result<String, String>,
    },
}

#[derive(Debug, PartialEq)]
pub enum Mode {
    Input,
    Streaming,
    RunningTool,
    Approval,
    ModelPicker,
    SessionPicker,
    ThemePicker,
}

/// What the transcript pane renders. Kept separate from the provider history
/// because the display needs entries (tool lines, errors) that are not part of
/// the conversation sent to the model.
#[derive(Clone, Serialize, Deserialize)]
pub enum Entry {
    Banner {
        title: String,
        subtitle: String,
    },
    User(String),
    Assistant(String),
    Tool {
        summary: String,
        result: String,
        is_error: bool,
    },
    Info(String),
    Error(String),
}

pub struct App {
    pub config: Config,
    pub mode: Mode,
    pub should_quit: bool,
    pub compacting: bool,
    pub theme: Theme,
    pub theme_index: usize,
    theme_revert: Option<Theme>,

    pub models: Vec<ModelEntry>,
    pub model: Option<ModelEntry>,
    pub picker_index: usize,
    pub picker_filter: String,

    pub sessions: Vec<session::Meta>,
    pub session_index: usize,
    session_id: String,

    pub transcript: Vec<Entry>,
    /// Bumped on structural transcript changes (clear/replace/pop) and theme
    /// switches so the renderer knows its per-entry cache is stale.
    pub transcript_rev: u64,
    pub history: Vec<Message>,
    pub scroll_from_bottom: usize,
    pub last_usage: Option<Usage>,

    pub textarea: TextArea<'static>,
    input_history: Vec<String>,
    input_history_pos: Option<usize>,
    input_draft: String,
    pub slash_index: usize,
    slash_dismissed: bool,

    // Statusline environment, refreshed at startup and after each turn/tool.
    pub cwd_display: String,
    pub git_branch: Option<String>,

    /// Images staged for the next message: (display name, encoded data).
    pub pending_images: Vec<(String, ImageData)>,

    /// Diff preview for the tool call currently awaiting approval.
    pub approval_preview: Option<Vec<(char, String)>>,

    // Renderer cache, managed by ui::draw.
    pub render_cache: Vec<Vec<Line<'static>>>,
    pub render_cache_width: usize,
    pub render_cache_rev: u64,

    gen: u64,
    streaming_text: String,
    pending_calls: VecDeque<ToolCall>,
    approved_tools: HashSet<String>,
    agent_turns: usize,
    request_task: Option<JoinHandle<()>>,
    tool_task: Option<JoinHandle<()>>,

    tx: UnboundedSender<AppEvent>,
}

impl App {
    pub fn new(config: Config, tx: UnboundedSender<AppEvent>) -> Self {
        let theme = session::load_theme_name()
            .or_else(|| config.theme.clone())
            .and_then(|name| theme::by_name(&name))
            .unwrap_or(theme::DEFAULT);
        let mut app = App {
            config,
            mode: Mode::Input,
            should_quit: false,
            compacting: false,
            theme,
            theme_index: 0,
            theme_revert: None,
            models: Vec::new(),
            model: None,
            picker_index: 0,
            picker_filter: String::new(),
            sessions: Vec::new(),
            session_index: 0,
            session_id: session::new_id(),
            transcript: Vec::new(),
            transcript_rev: 0,
            history: Vec::new(),
            scroll_from_bottom: 0,
            last_usage: None,
            textarea: make_textarea(&theme),
            input_history: session::load_input_history(),
            input_history_pos: None,
            input_draft: String::new(),
            slash_index: 0,
            slash_dismissed: false,
            cwd_display: String::new(),
            git_branch: None,
            pending_images: Vec::new(),
            approval_preview: None,
            render_cache: Vec::new(),
            render_cache_width: 0,
            render_cache_rev: 0,
            gen: 0,
            streaming_text: String::new(),
            pending_calls: VecDeque::new(),
            approved_tools: HashSet::new(),
            agent_turns: 0,
            request_task: None,
            tool_task: None,
            tx,
        };
        app.transcript.push(Entry::Banner {
            title: "shaltaiboltai".into(),
            subtitle: format!(
                "v{} · multi-provider coding agent — Enter send · Alt+Enter newline · Ctrl+P models · /theme · /help",
                env!("CARGO_PKG_VERSION"),
            ),
        });
        app.refresh_environment();
        app.spawn_discovery();
        app
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.mode, Mode::Streaming | Mode::RunningTool) || self.compacting
    }

    /// Cwd and git branch for the statusline. Cheap (one small file read);
    /// refreshed after turns/tools and on a slow idle tick, not per frame.
    pub fn refresh_environment(&mut self) {
        self.cwd_display = std::env::current_dir()
            .map(|p| shorten_path(&p))
            .unwrap_or_default();
        self.git_branch = std::fs::read_to_string(".git/HEAD")
            .ok()
            .and_then(|head| parse_git_head(&head));
    }

    // ---- slash-command menu ----

    /// The active `/` menu filter: input is a single line starting with `/`
    /// and no arguments yet. `None` means the menu is closed.
    pub fn slash_filter(&self) -> Option<String> {
        if self.mode != Mode::Input || self.slash_dismissed {
            return None;
        }
        let lines = self.textarea.lines();
        if lines.len() != 1 {
            return None;
        }
        let rest = lines[0].strip_prefix('/')?;
        if rest.contains(char::is_whitespace) {
            return None;
        }
        Some(rest.to_lowercase())
    }

    pub fn slash_matches(&self) -> Vec<&'static SlashCommand> {
        self.slash_filter()
            .map(|f| match_commands(&f))
            .unwrap_or_default()
    }

    pub fn slash_menu_active(&self) -> bool {
        !self.slash_matches().is_empty()
    }

    pub fn slash_move(&mut self, delta: i64) {
        let len = self.slash_matches().len() as i64;
        if len > 0 {
            self.slash_index = (self.slash_index as i64 + delta).rem_euclid(len) as usize;
        }
    }

    fn selected_slash(&self) -> Option<&'static SlashCommand> {
        let matches = self.slash_matches();
        matches
            .get(self.slash_index.min(matches.len().saturating_sub(1)))
            .copied()
    }

    pub fn complete_selected_slash(&mut self) {
        if let Some(cmd) = self.selected_slash() {
            // Commands that take arguments complete with a trailing space so
            // the user can keep typing.
            let suffix = if cmd.args.is_some() { " " } else { "" };
            self.set_input(&format!("/{}{suffix}", cmd.name));
            self.slash_index = 0;
        }
    }

    pub fn run_selected_slash(&mut self) {
        if let Some(cmd) = self.selected_slash() {
            self.set_input(&format!("/{}", cmd.name));
        }
        self.slash_index = 0;
        self.submit_input();
    }

    pub fn dismiss_slash_menu(&mut self) {
        self.slash_dismissed = true;
    }

    /// Called when the input text changes: reopen a dismissed menu and reset
    /// the selection, mirroring how Claude Code's completion behaves.
    pub fn note_input_changed(&mut self) {
        self.slash_dismissed = false;
        self.slash_index = 0;
    }

    fn spawn_discovery(&self) {
        let config = self.config.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let models = providers::discover_models(config).await;
            let _ = tx.send(AppEvent::ModelsDiscovered(models));
        });
    }

    // ---- background events ----

    pub fn on_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::ModelsDiscovered(models) => {
                if self.model.is_none() {
                    self.model = self
                        .config
                        .default_model
                        .as_ref()
                        .and_then(|want| models.iter().find(|m| &m.id == want).cloned())
                        .or_else(|| models.first().cloned());
                }
                if models.is_empty() {
                    self.transcript.push(Entry::Error(
                        "no models available: set ANTHROPIC_API_KEY / OPENAI_API_KEY or start ollama".into(),
                    ));
                } else {
                    self.transcript.push(Entry::Info(format!(
                        "{} models available across {} provider(s)",
                        models.len(),
                        provider_count(&models),
                    )));
                }
                self.models = models;
            }
            AppEvent::Chat { gen, event } => {
                if gen == self.gen {
                    self.on_chat_event(event);
                }
            }
            AppEvent::ToolFinished {
                gen,
                call,
                content,
                is_error,
            } => {
                if gen == self.gen {
                    self.finish_tool(call, content, is_error);
                }
            }
            AppEvent::CompactionDone { session_id, result } => {
                self.compacting = false;
                // A compaction started in another session must not replace
                // this one's history.
                if session_id == self.session_id {
                    self.finish_compaction(result);
                }
            }
        }
    }

    fn on_chat_event(&mut self, event: ChatEvent) {
        match event {
            ChatEvent::TextDelta(text) => {
                self.streaming_text.push_str(&text);
                // Append to the current assistant block, or start a new one if
                // a sub-agent's tool activity interrupted it.
                match self.transcript.last_mut() {
                    Some(Entry::Assistant(buf)) => buf.push_str(&text),
                    _ => self.transcript.push(Entry::Assistant(text)),
                }
            }
            ChatEvent::ToolActivity { summary, is_error } => {
                // Sub-agent tools have already run inside the CLI; this is
                // display-only and never enters our approval flow.
                if matches!(self.transcript.last(), Some(Entry::Assistant(t)) if t.is_empty()) {
                    self.transcript.pop();
                }
                self.transcript.push(Entry::Tool {
                    summary,
                    result: String::new(),
                    is_error,
                });
            }
            ChatEvent::Completed {
                tool_calls,
                stop_reason,
                usage,
            } => {
                self.request_task = None;
                if usage.is_some() {
                    self.last_usage = usage;
                }
                self.history.push(Message::Assistant {
                    text: std::mem::take(&mut self.streaming_text),
                    tool_calls: tool_calls.clone(),
                });
                if stop_reason.as_deref() == Some("length") {
                    self.transcript.push(Entry::Error(
                        "response was truncated by the output token limit".into(),
                    ));
                }
                if tool_calls.is_empty() {
                    self.end_turn();
                    return;
                }
                self.agent_turns += 1;
                if self.agent_turns > MAX_AGENT_TURNS {
                    self.transcript.push(Entry::Error(format!(
                        "stopped after {MAX_AGENT_TURNS} consecutive tool rounds"
                    )));
                    self.repair_dangling_tool_calls();
                    self.end_turn();
                    return;
                }
                self.pending_calls = tool_calls.into();
                self.advance_tools();
            }
            ChatEvent::Error(message) => {
                self.request_task = None;
                // Keep partial text the user already saw consistent with what
                // the model will see next turn.
                let text = std::mem::take(&mut self.streaming_text);
                if !text.is_empty() {
                    self.history.push(Message::Assistant {
                        text,
                        tool_calls: Vec::new(),
                    });
                } else if matches!(self.transcript.last(), Some(Entry::Assistant(t)) if t.is_empty())
                {
                    self.transcript.pop();
                    self.transcript_rev += 1;
                }
                self.transcript.push(Entry::Error(message));
                self.agent_turns = 0;
                self.mode = Mode::Input;
            }
        }
    }

    /// A user turn finished with a final answer: persist, and compact the
    /// context in the background if it has grown past the threshold.
    fn end_turn(&mut self) {
        self.agent_turns = 0;
        self.mode = Mode::Input;
        self.refresh_environment();
        self.save_session();
        if self.context_over_threshold() && !self.compacting {
            self.transcript.push(Entry::Info(
                "context exceeded threshold — compacting in the background".into(),
            ));
            self.start_compaction();
        }
    }

    /// Process the queue of tool calls returned by the model: pause for
    /// approval where required, execute otherwise, and when the queue is
    /// drained send the results back to the model.
    fn advance_tools(&mut self) {
        self.approval_preview = None;
        match self.pending_calls.front() {
            None => self.start_request(),
            Some(call)
                if tools::requires_approval(call)
                    && !self.approved_tools.contains(call.name.as_str()) =>
            {
                self.approval_preview = tools::approval_preview(call);
                self.mode = Mode::Approval;
            }
            Some(_) => {
                let call = self.pending_calls.pop_front().unwrap();
                self.run_tool(call);
            }
        }
    }

    fn run_tool(&mut self, call: ToolCall) {
        self.mode = Mode::RunningTool;
        let gen = self.gen;
        let tx = self.tx.clone();
        self.tool_task = Some(tokio::spawn(async move {
            let (content, is_error) = tools::execute(&call).await;
            let _ = tx.send(AppEvent::ToolFinished {
                gen,
                call,
                content,
                is_error,
            });
        }));
    }

    fn finish_tool(&mut self, call: ToolCall, content: String, is_error: bool) {
        self.tool_task = None;
        // A command may have switched branches or moved files.
        self.refresh_environment();
        self.transcript.push(Entry::Tool {
            summary: tools::describe(&call),
            result: content.clone(),
            is_error,
        });
        self.history.push(Message::ToolResult {
            call_id: call.id,
            name: call.name,
            content,
            is_error,
        });
        self.advance_tools();
    }

    pub fn approve_pending(&mut self, always: bool) {
        if let Some(call) = self.pending_calls.pop_front() {
            if always {
                self.approved_tools.insert(call.name.clone());
            }
            self.approval_preview = None;
            self.run_tool(call);
        }
    }

    pub fn deny_pending(&mut self) {
        if let Some(call) = self.pending_calls.pop_front() {
            self.approval_preview = None;
            self.finish_tool(call, "User denied this tool call.".into(), true);
        }
    }

    pub fn pending_approval(&self) -> Option<&ToolCall> {
        self.pending_calls.front()
    }

    // ---- user actions ----

    pub fn submit_input(&mut self) {
        let text = self.textarea.lines().join("\n").trim().to_owned();
        if text.is_empty() {
            return;
        }
        // Slash commands stay available while compacting; only chat turns
        // must wait for the new context.
        if self.compacting && !text.starts_with('/') {
            self.transcript.push(Entry::Error(
                "context compaction in progress — try again in a moment".into(),
            ));
            return;
        }
        self.textarea = make_textarea(&self.theme);
        self.remember_input(&text);

        if let Some(command) = text.strip_prefix('/') {
            self.run_slash_command(command);
            return;
        }
        if self.model.is_none() {
            self.transcript.push(Entry::Error(
                "no model selected — configure a provider, then pick one with Ctrl+P".into(),
            ));
            return;
        }
        // Attach staged images plus any image paths referenced in the text
        // (typed or drag-and-dropped onto the terminal).
        let mut images = std::mem::take(&mut self.pending_images);
        for path in images::extract_image_paths(&text) {
            match images::load_image(&path) {
                Ok(attachment) => images.push(attachment),
                Err(e) => self.transcript.push(Entry::Error(format!("{e:#}"))),
            }
        }
        self.transcript.push(Entry::User(text.clone()));
        if !images.is_empty() {
            let names: Vec<&str> = images.iter().map(|(n, _)| n.as_str()).collect();
            self.transcript
                .push(Entry::Info(format!("attached: {}", names.join(", "))));
        }
        let content = if images.is_empty() {
            UserContent::Text(text)
        } else {
            UserContent::Rich {
                text,
                images: images.into_iter().map(|(_, data)| data).collect(),
            }
        };
        self.history.push(Message::User(content));
        self.scroll_from_bottom = 0;
        self.agent_turns = 0;
        self.start_request();
    }

    pub fn paste(&mut self, text: &str) {
        if self.mode != Mode::Input {
            return;
        }
        // Files dragged onto the terminal arrive as a paste of their paths:
        // stage them as attachments instead of cluttering the input.
        let dropped = images::dropped_images(text);
        if !dropped.is_empty() {
            for path in dropped {
                match images::load_image(&path) {
                    Ok((name, data)) => {
                        self.transcript
                            .push(Entry::Info(format!("image staged: {name} — Esc clears")));
                        self.pending_images.push((name, data));
                    }
                    Err(e) => self.transcript.push(Entry::Error(format!("{e:#}"))),
                }
            }
            return;
        }
        self.textarea
            .insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
    }

    // ---- image attachments ----

    /// Ctrl+V: stage an image from the system clipboard for the next message.
    pub fn attach_clipboard_image(&mut self) {
        match images::clipboard_image() {
            Ok(image) => {
                let name = format!("clipboard-{}.png", self.pending_images.len() + 1);
                self.pending_images.push((name, image));
                self.transcript.push(Entry::Info(format!(
                    "image staged from clipboard ({} attached) — Esc clears",
                    self.pending_images.len()
                )));
            }
            Err(e) => self.transcript.push(Entry::Info(format!("{e:#}"))),
        }
    }

    pub fn clear_attachments(&mut self) {
        if !self.pending_images.is_empty() {
            self.pending_images.clear();
            self.transcript
                .push(Entry::Info("attachments cleared".into()));
        }
    }

    // ---- input history (Up/Down recall) ----

    fn remember_input(&mut self, text: &str) {
        self.input_history_pos = None;
        self.input_draft.clear();
        if self.input_history.last().map(String::as_str) != Some(text) {
            self.input_history.push(text.to_owned());
            session::append_input_history(text);
        }
    }

    pub fn input_is_empty(&self) -> bool {
        self.textarea.lines().iter().all(|l| l.is_empty())
    }

    /// Ctrl+U: wipe the whole input and leave history recall, so a subsequent
    /// Up starts from the most recent entry again.
    pub fn clear_input(&mut self) {
        self.textarea = make_textarea(&self.theme);
        self.input_history_pos = None;
        self.input_draft.clear();
    }

    pub fn history_recall_active(&self) -> bool {
        self.input_history_pos.is_some()
    }

    pub fn input_history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let pos = match self.input_history_pos {
            None => {
                self.input_draft = self.textarea.lines().join("\n");
                self.input_history.len() - 1
            }
            Some(0) => 0,
            Some(p) => p - 1,
        };
        self.input_history_pos = Some(pos);
        self.set_input(&self.input_history[pos].clone());
    }

    pub fn input_history_next(&mut self) {
        match self.input_history_pos {
            None => {}
            Some(p) if p + 1 < self.input_history.len() => {
                self.input_history_pos = Some(p + 1);
                self.set_input(&self.input_history[p + 1].clone());
            }
            Some(_) => {
                self.input_history_pos = None;
                let draft = std::mem::take(&mut self.input_draft);
                self.set_input(&draft);
            }
        }
    }

    fn set_input(&mut self, text: &str) {
        self.textarea = make_textarea(&self.theme);
        self.textarea.insert_str(text);
    }

    fn run_slash_command(&mut self, command: &str) {
        let mut parts = command.trim().splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let arg = parts.next().map(str::trim).filter(|a| !a.is_empty());
        let Some(cmd) = SLASH_COMMANDS
            .iter()
            .find(|c| c.name == name || c.aliases.contains(&name))
        else {
            self.transcript.push(Entry::Error(format!(
                "unknown command: /{name} — try /help"
            )));
            return;
        };
        match (cmd.name, arg) {
            ("model", Some(filter)) => self.select_model_by_filter(filter),
            ("model", None) => self.open_picker(),
            ("theme", Some(name)) => self.set_theme_by_name(name),
            ("theme", None) => self.open_themes(),
            ("new", _) => self.reset_session(),
            ("resume", _) => self.open_sessions(),
            ("compact", _) => {
                if self.compacting {
                    self.transcript
                        .push(Entry::Info("compaction already in progress".into()));
                } else if self.history.is_empty() {
                    self.transcript
                        .push(Entry::Info("nothing to compact".into()));
                } else {
                    self.transcript
                        .push(Entry::Info("compacting context…".into()));
                    self.start_compaction();
                }
            }
            ("quit", _) => self.should_quit = true,
            ("help", _) => {
                let commands = SLASH_COMMANDS
                    .iter()
                    .map(|c| {
                        let args = c.args.map(|a| format!(" {a}")).unwrap_or_default();
                        format!("/{}{args} — {}", c.name, c.description)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                self.transcript.push(Entry::Info(format!(
                    "{commands}\nkeys: Ctrl+P models · Alt+Enter newline · Up/Down input history · Ctrl+U clear input · PgUp/PgDn or mouse wheel scroll · Esc cancel · Ctrl+C quit"
                )));
            }
            _ => unreachable!("registry and dispatch are matched"),
        }
    }

    /// `/model <name>`: select directly on a unique match, open the picker
    /// pre-filtered when ambiguous.
    fn select_model_by_filter(&mut self, filter: &str) {
        let needle = filter.to_lowercase();
        let matches: Vec<ModelEntry> = self
            .models
            .iter()
            .filter(|m| m.id.to_lowercase().contains(&needle))
            .cloned()
            .collect();
        let exact = matches.iter().find(|m| m.id.to_lowercase() == needle);
        if let Some(model) = exact.or(if matches.len() == 1 {
            matches.first()
        } else {
            None
        }) {
            self.transcript.push(Entry::Info(format!(
                "model: {} ({})",
                model.id,
                model.provider.label()
            )));
            self.model = Some(model.clone());
        } else if matches.is_empty() {
            self.transcript
                .push(Entry::Error(format!("no model matches \"{filter}\"")));
        } else {
            // Ambiguous: open the picker pre-filtered.
            self.picker_filter = filter.to_owned();
            self.picker_index = 0;
            self.mode = Mode::ModelPicker;
        }
    }

    /// `/theme <name>`: switch and persist directly.
    fn set_theme_by_name(&mut self, name: &str) {
        match theme::by_name(&name.to_lowercase()) {
            Some(picked) => {
                self.theme = picked;
                self.apply_theme();
                session::save_theme_name(picked.name);
                self.transcript
                    .push(Entry::Info(format!("theme: {}", picked.name)));
            }
            None => {
                let names: Vec<&str> = theme::all().iter().map(|t| t.name).collect();
                self.transcript.push(Entry::Error(format!(
                    "unknown theme \"{name}\" — available: {}",
                    names.join(", ")
                )));
            }
        }
    }

    fn start_request(&mut self) {
        let Some(model) = self.model.clone() else {
            self.mode = Mode::Input;
            return;
        };
        self.mode = Mode::Streaming;
        self.streaming_text.clear();
        self.transcript.push(Entry::Assistant(String::new()));

        // Sub-agent providers run their own tool loop, so we don't send ours.
        let tools = if model.provider.is_sub_agent() {
            Vec::new()
        } else {
            tools::definitions()
        };
        let request = ChatRequest {
            model,
            system: system_prompt(),
            messages: self.history.clone(),
            tools,
        };
        let gen = self.gen;
        let config = self.config.clone();
        let tx = self.tx.clone();
        self.request_task = Some(tokio::spawn(async move {
            let (chat_tx, mut chat_rx) = tokio::sync::mpsc::unbounded_channel();
            let stream = tokio::spawn(providers::stream_chat(config, request, chat_tx));
            while let Some(event) = chat_rx.recv().await {
                if tx.send(AppEvent::Chat { gen, event }).is_err() {
                    break;
                }
            }
            let _ = stream.await;
        }));
    }

    pub fn cancel_request(&mut self) {
        // Invalidate in-flight work; late events from old generations are dropped.
        self.gen += 1;
        if let Some(task) = self.request_task.take() {
            task.abort();
        }
        if let Some(task) = self.tool_task.take() {
            task.abort();
        }
        // Keep whatever streamed so far as a valid assistant turn.
        let text = std::mem::take(&mut self.streaming_text);
        if !text.is_empty() {
            self.history.push(Message::Assistant {
                text,
                tool_calls: Vec::new(),
            });
        } else if matches!(self.transcript.last(), Some(Entry::Assistant(t)) if t.is_empty()) {
            self.transcript.pop();
            self.transcript_rev += 1;
        }
        self.pending_calls.clear();
        self.approval_preview = None;
        self.repair_dangling_tool_calls();
        self.transcript.push(Entry::Info("cancelled".into()));
        self.agent_turns = 0;
        self.mode = Mode::Input;
    }

    /// Providers reject an assistant tool call that has no matching result.
    /// After a cancellation mid-round, close any dangling calls with an
    /// explicit "cancelled" result so the next request is valid.
    fn repair_dangling_tool_calls(&mut self) {
        let Some(last_assistant) = self
            .history
            .iter()
            .rposition(|m| matches!(m, Message::Assistant { .. }))
        else {
            return;
        };
        let Message::Assistant { tool_calls, .. } = &self.history[last_assistant] else {
            return;
        };
        let answered: HashSet<String> = self.history[last_assistant + 1..]
            .iter()
            .filter_map(|m| match m {
                Message::ToolResult { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect();
        let missing: Vec<(String, String)> = tool_calls
            .iter()
            .filter(|c| !answered.contains(&c.id))
            .map(|c| (c.id.clone(), c.name.clone()))
            .collect();
        for (call_id, name) in missing {
            self.history.push(Message::ToolResult {
                call_id,
                name,
                content: "Cancelled by user.".into(),
                is_error: true,
            });
        }
    }

    // ---- model picker ----

    pub fn open_picker(&mut self) {
        if self.models.is_empty() {
            self.transcript
                .push(Entry::Error("no models discovered yet".into()));
            return;
        }
        self.picker_index = 0;
        self.picker_filter.clear();
        self.mode = Mode::ModelPicker;
    }

    pub fn filtered_models(&self) -> Vec<&ModelEntry> {
        let needle = self.picker_filter.to_lowercase();
        self.models
            .iter()
            .filter(|m| {
                needle.is_empty()
                    || m.id.to_lowercase().contains(&needle)
                    || m.provider.label().contains(&needle)
            })
            .collect()
    }

    pub fn pick_model(&mut self) {
        let picked = self
            .filtered_models()
            .get(self.picker_index)
            .map(|m| (*m).clone());
        if let Some(model) = picked {
            self.transcript.push(Entry::Info(format!(
                "model: {} ({})",
                model.id,
                model.provider.label()
            )));
            self.model = Some(model);
        }
        self.mode = Mode::Input;
    }

    // ---- theme picker (live preview) ----

    pub fn open_themes(&mut self) {
        self.theme_revert = Some(self.theme);
        self.theme_index = theme::all()
            .iter()
            .position(|t| t.name == self.theme.name)
            .unwrap_or(0);
        self.mode = Mode::ThemePicker;
    }

    pub fn theme_move(&mut self, delta: i64) {
        let themes = theme::all();
        let len = themes.len() as i64;
        let idx = (self.theme_index as i64 + delta).rem_euclid(len) as usize;
        self.theme_index = idx;
        self.theme = themes[idx];
        self.apply_theme();
    }

    pub fn pick_theme(&mut self) {
        self.theme_revert = None;
        session::save_theme_name(self.theme.name);
        self.transcript
            .push(Entry::Info(format!("theme: {}", self.theme.name)));
        self.mode = Mode::Input;
    }

    pub fn revert_theme(&mut self) {
        if let Some(previous) = self.theme_revert.take() {
            self.theme = previous;
            self.apply_theme();
        }
        self.mode = Mode::Input;
    }

    /// Re-style live widgets and invalidate cached rendered lines after a
    /// theme change.
    fn apply_theme(&mut self) {
        let text = self.textarea.lines().join("\n");
        self.textarea = make_textarea(&self.theme);
        self.textarea.insert_str(text);
        self.transcript_rev += 1;
    }

    // ---- sessions ----

    pub fn save_session(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let title = self
            .history
            .iter()
            .find_map(|m| match m {
                Message::User(c) => Some(
                    c.text()
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(64)
                        .collect::<String>(),
                ),
                _ => None,
            })
            .unwrap_or_else(|| "untitled".into());
        let snapshot = session::Session {
            id: self.session_id.clone(),
            title,
            updated_at: session::now_secs(),
            cwd: std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string()),
            model: self.model.clone(),
            history: self.history.clone(),
            transcript: self.transcript.clone(),
        };
        if let Err(e) = session::save(&snapshot) {
            // Persistence is best-effort; never write to stderr from a TUI.
            self.transcript
                .push(Entry::Error(format!("failed to save session: {e:#}")));
        }
    }

    fn reset_session(&mut self) {
        self.save_session();
        self.gen += 1;
        self.session_id = session::new_id();
        self.history.clear();
        self.transcript.clear();
        self.transcript_rev += 1;
        self.agent_turns = 0;
        self.scroll_from_bottom = 0;
        self.last_usage = None;
        self.pending_calls.clear();
        self.approval_preview = None;
        self.transcript
            .push(Entry::Info("started a new session".into()));
    }

    pub fn open_sessions(&mut self) {
        let mut sessions = session::list();
        if sessions.is_empty() {
            self.transcript
                .push(Entry::Info("no saved sessions yet".into()));
            return;
        }
        // Project-scoped ordering: this directory's sessions first (legacy
        // sessions without a cwd count as local), then everything else.
        // The picker badges entries from other directories.
        sessions.sort_by_key(|s| usize::from(session_is_foreign(s)));
        self.sessions = sessions;
        self.session_index = 0;
        self.mode = Mode::SessionPicker;
    }

    pub fn pick_session(&mut self) {
        let Some(meta) = self.sessions.get(self.session_index) else {
            self.mode = Mode::Input;
            return;
        };
        match session::load(&meta.path) {
            Ok(loaded) => {
                self.save_session();
                self.gen += 1;
                self.session_id = loaded.id;
                self.history = loaded.history;
                self.transcript = loaded.transcript;
                self.transcript_rev += 1;
                self.scroll_from_bottom = 0;
                self.agent_turns = 0;
                self.last_usage = None;
                if let Some(saved) = loaded.model {
                    if self
                        .models
                        .iter()
                        .any(|m| m.id == saved.id && m.provider == saved.provider)
                    {
                        self.model = Some(saved);
                    } else {
                        self.transcript.push(Entry::Info(format!(
                            "saved model {} is unavailable — keeping current model",
                            saved.id
                        )));
                    }
                }
                self.transcript.push(Entry::Info("session resumed".into()));
            }
            Err(e) => self
                .transcript
                .push(Entry::Error(format!("failed to load session: {e:#}"))),
        }
        self.mode = Mode::Input;
    }

    // ---- context compaction ----

    pub fn history_chars(&self) -> usize {
        self.history
            .iter()
            .map(|m| match m {
                Message::User(c) => c.text().len() + c.images().len() * 4_000,
                Message::Assistant { text, tool_calls } => {
                    text.len()
                        + tool_calls
                            .iter()
                            .map(|c| c.name.len() + c.arguments.to_string().len())
                            .sum::<usize>()
                }
                Message::ToolResult { content, .. } => content.len(),
            })
            .sum()
    }

    pub fn approx_tokens(&self) -> usize {
        self.history_chars() / 4
    }

    /// How full the context is relative to the compaction threshold, for the
    /// statusline. Uses provider-reported tokens when available.
    pub fn context_percent(&self) -> Option<u8> {
        let threshold = self.effective_compact_threshold().max(1);
        let used = match self.last_usage {
            Some(u) => (u.input_tokens as usize) * 4,
            None => self.history_chars(),
        };
        if used == 0 {
            return None;
        }
        Some(((used * 100 / threshold).min(100)) as u8)
    }

    /// Ollama models are bounded by the configured num_ctx, which is usually
    /// far smaller than the cloud-model threshold — compact well before it.
    fn effective_compact_threshold(&self) -> usize {
        let configured = self.config.compact_threshold_chars;
        match self.model.as_ref().map(|m| m.provider) {
            Some(ProviderKind::Ollama) => configured.min(self.config.ollama_num_ctx * 3),
            _ => configured,
        }
    }

    fn context_over_threshold(&self) -> bool {
        let threshold = self.effective_compact_threshold();
        if self.history_chars() > threshold {
            return true;
        }
        // Prefer the provider-reported context size when we have it.
        self.last_usage
            .is_some_and(|u| u.input_tokens as usize > threshold / 4)
    }

    fn start_compaction(&mut self) {
        let Some(model) = self.model.clone() else {
            return;
        };
        if self.compacting || self.history.is_empty() {
            return;
        }
        self.compacting = true;

        // The history is flattened to plain text so the summary request is
        // valid for every provider regardless of tool-call wire formats.
        let flat = flatten_history(&self.history);
        let request = ChatRequest {
            model,
            system: "You compress coding-assistant conversations into handoff summaries.".into(),
            messages: vec![Message::User(UserContent::Text(format!(
                "Summarize the conversation below so a successor agent can continue seamlessly. \
                 Capture: the user's goals, decisions made, files created or modified and their \
                 current state, commands run with relevant outcomes, and unresolved tasks. \
                 Output only the summary.\n\n<conversation>\n{flat}\n</conversation>"
            )))],
            tools: Vec::new(),
        };
        let session_id = self.session_id.clone();
        let config = self.config.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let (chat_tx, mut chat_rx) = tokio::sync::mpsc::unbounded_channel();
            tokio::spawn(providers::stream_chat(config, request, chat_tx));
            let mut text = String::new();
            let mut error = None;
            while let Some(event) = chat_rx.recv().await {
                match event {
                    ChatEvent::TextDelta(t) => text.push_str(&t),
                    ChatEvent::Error(e) => error = Some(e),
                    ChatEvent::Completed { .. } | ChatEvent::ToolActivity { .. } => {}
                }
            }
            let result = match error {
                Some(e) => Err(e),
                None if text.trim().is_empty() => Err("empty summary".into()),
                None => Ok(text),
            };
            let _ = tx.send(AppEvent::CompactionDone { session_id, result });
        });
    }

    fn finish_compaction(&mut self, result: Result<String, String>) {
        match result {
            Ok(summary) => {
                let before = self.history_chars();
                self.history = vec![Message::User(UserContent::Text(format!(
                    "Context summary of our conversation so far (earlier messages were compacted):\n\n{}",
                    summary.trim()
                )))];
                self.last_usage = None;
                self.transcript.push(Entry::Info(format!(
                    "context compacted: ~{}k → ~{}k chars",
                    before / 1000,
                    self.history_chars() / 1000
                )));
                self.save_session();
            }
            Err(e) => self
                .transcript
                .push(Entry::Error(format!("compaction failed: {e}"))),
        }
    }
}

fn make_textarea(theme: &Theme) -> TextArea<'static> {
    // The block (border + surface background) is restyled every frame by
    // ui::draw_input, since it doubles as the focus indicator.
    let mut textarea = TextArea::default();
    textarea.set_style(Style::new().fg(theme.fg));
    textarea.set_cursor_line_style(Style::default());
    textarea.set_placeholder_text("type a message — Enter send, Alt+Enter newline, /help");
    textarea.set_placeholder_style(Style::new().fg(theme.dim));
    textarea
}

fn flatten_history(history: &[Message]) -> String {
    let cap = |s: &str| -> String {
        if s.chars().count() > COMPACT_FLATTEN_CAP {
            let cut: String = s.chars().take(COMPACT_FLATTEN_CAP).collect();
            format!("{cut}…[truncated]")
        } else {
            s.to_owned()
        }
    };
    let mut flat = String::new();
    for msg in history {
        match msg {
            Message::User(c) => {
                flat.push_str(&format!("[user]\n{}\n", cap(c.text())));
                for _ in c.images() {
                    flat.push_str("[image attached]\n");
                }
                flat.push('\n');
            }
            Message::Assistant { text, tool_calls } => {
                flat.push_str(&format!("[assistant]\n{}\n", cap(text)));
                for call in tool_calls {
                    flat.push_str(&format!(
                        "[assistant called {} with {}]\n",
                        call.name,
                        cap(&call.arguments.to_string())
                    ));
                }
                flat.push('\n');
            }
            Message::ToolResult { name, content, .. } => {
                flat.push_str(&format!("[{} result]\n{}\n\n", name, cap(content)));
            }
        }
    }
    flat
}

/// Whether a saved session belongs to a different working directory (used
/// for picker ordering and badges). Legacy sessions without a recorded cwd
/// count as local.
pub fn session_is_foreign(meta: &session::Meta) -> bool {
    let here = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    meta.cwd.as_ref().is_some_and(|cwd| *cwd != here)
}

/// `~`-abbreviate the home directory and keep at most the last three
/// components so the statusline stays short.
fn shorten_path(path: &std::path::Path) -> String {
    let display = match dirs::home_dir().and_then(|h| path.strip_prefix(&h).ok().map(|p| (h, p))) {
        Some((_, rel)) if rel.as_os_str().is_empty() => "~".to_owned(),
        Some((_, rel)) => format!("~/{}", rel.display()),
        None => path.display().to_string(),
    };
    let parts: Vec<&str> = display.split('/').collect();
    if parts.len() > 4 {
        format!("…/{}", parts[parts.len() - 3..].join("/"))
    } else {
        display
    }
}

/// Branch name from `.git/HEAD` content; `None` for a detached head.
fn parse_git_head(head: &str) -> Option<String> {
    head.trim()
        .strip_prefix("ref: refs/heads/")
        .map(str::to_owned)
}

fn provider_count(models: &[ModelEntry]) -> usize {
    let mut kinds: Vec<_> = models.iter().map(|m| m.provider.label()).collect();
    kinds.sort();
    kinds.dedup();
    kinds.len()
}

fn system_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "?".into());
    let mut prompt = format!(
        "You are shaltaiboltai, an agentic coding assistant running in a terminal. \
         The user's working directory is {cwd} on {}. \
         Use the available tools to read and modify files and run commands when the task calls for it. \
         Prefer edit_file over write_file for existing files, and grep/glob to explore before reading. \
         Prefer small, verifiable steps and report what you did. \
         Format responses in markdown.",
        std::env::consts::OS,
    );
    for name in ["AGENTS.md", "CLAUDE.md"] {
        if let Ok(content) = std::fs::read_to_string(name) {
            let capped: String = content.chars().take(PROJECT_CONTEXT_CAP).collect();
            prompt.push_str(&format!(
                "\n\n# Project instructions (from {name})\n{capped}"
            ));
            break;
        }
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_matching_prefers_names_then_aliases() {
        let names: Vec<_> = match_commands("").iter().map(|c| c.name).collect();
        assert_eq!(names.len(), SLASH_COMMANDS.len());

        let m: Vec<_> = match_commands("mo").iter().map(|c| c.name).collect();
        assert_eq!(m, vec!["model"]);

        // "clear" only matches as an alias of /new.
        let m: Vec<_> = match_commands("cl").iter().map(|c| c.name).collect();
        assert_eq!(m, vec!["new"]);

        assert!(match_commands("zzz").is_empty());
    }

    #[test]
    fn git_head_parsing() {
        assert_eq!(
            parse_git_head("ref: refs/heads/main\n"),
            Some("main".into())
        );
        assert_eq!(
            parse_git_head("ref: refs/heads/feat/x"),
            Some("feat/x".into())
        );
        assert_eq!(parse_git_head("3f2c1a9deadbeef\n"), None);
    }

    #[test]
    fn paths_are_shortened_for_the_statusline() {
        if let Some(home) = dirs::home_dir() {
            assert_eq!(shorten_path(&home), "~");
            let deep = home.join("a/b/c/d/e");
            let s = shorten_path(&deep);
            assert!(s.starts_with("…/"), "{s}");
            assert!(s.ends_with("c/d/e"), "{s}");
        }
        assert_eq!(shorten_path(std::path::Path::new("/tmp/x")), "/tmp/x");
    }
}
