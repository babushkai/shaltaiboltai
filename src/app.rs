use crate::config::Config;
use crate::providers::{self, ChatEvent, ChatRequest, Message, ModelEntry, ToolCall};
use crate::session;
use crate::tools;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tui_textarea::TextArea;

/// Hard cap on consecutive model→tool→model rounds for a single user input,
/// so a confused model cannot loop forever.
const MAX_AGENT_TURNS: usize = 30;

/// Per-message cap when flattening history for the compaction summary, so the
/// summary request itself stays small.
const COMPACT_FLATTEN_CAP: usize = 4_000;

/// Events delivered to the UI loop from background tasks.
pub enum AppEvent {
    Chat(ChatEvent),
    ModelsDiscovered(Vec<ModelEntry>),
    ToolFinished {
        call: ToolCall,
        content: String,
        is_error: bool,
    },
    CompactionDone(Result<String, String>),
}

#[derive(PartialEq)]
pub enum Mode {
    Input,
    Streaming,
    RunningTool,
    Approval,
    ModelPicker,
    SessionPicker,
}

/// What the transcript pane renders. Kept separate from the provider history
/// because the display needs entries (tool lines, errors) that are not part of
/// the conversation sent to the model.
#[derive(Clone, Serialize, Deserialize)]
pub enum Entry {
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

    pub models: Vec<ModelEntry>,
    pub model: Option<ModelEntry>,
    pub picker_index: usize,
    pub picker_filter: String,

    pub sessions: Vec<session::Meta>,
    pub session_index: usize,
    session_id: String,

    pub transcript: Vec<Entry>,
    pub history: Vec<Message>,
    pub scroll_from_bottom: usize,

    pub textarea: TextArea<'static>,

    streaming_text: String,
    pending_calls: VecDeque<ToolCall>,
    auto_approve: bool,
    agent_turns: usize,
    request_task: Option<JoinHandle<()>>,

    tx: UnboundedSender<AppEvent>,
}

impl App {
    pub fn new(config: Config, tx: UnboundedSender<AppEvent>) -> Self {
        let mut app = App {
            config,
            mode: Mode::Input,
            should_quit: false,
            compacting: false,
            models: Vec::new(),
            model: None,
            picker_index: 0,
            picker_filter: String::new(),
            sessions: Vec::new(),
            session_index: 0,
            session_id: session::new_id(),
            transcript: Vec::new(),
            history: Vec::new(),
            scroll_from_bottom: 0,
            textarea: make_textarea(),
            streaming_text: String::new(),
            pending_calls: VecDeque::new(),
            auto_approve: false,
            agent_turns: 0,
            request_task: None,
            tx,
        };
        app.transcript.push(Entry::Info(
            "shaltaiboltai — Enter to send, Alt+Enter for newline, Ctrl+P to pick a model, /help for commands"
                .into(),
        ));
        app.spawn_discovery();
        app
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
            AppEvent::Chat(chat) => self.on_chat_event(chat),
            AppEvent::ToolFinished {
                call,
                content,
                is_error,
            } => {
                self.finish_tool(call, content, is_error);
            }
            AppEvent::CompactionDone(result) => self.finish_compaction(result),
        }
    }

    fn on_chat_event(&mut self, event: ChatEvent) {
        match event {
            ChatEvent::TextDelta(text) => {
                self.streaming_text.push_str(&text);
                if let Some(Entry::Assistant(buf)) = self.transcript.last_mut() {
                    buf.push_str(&text);
                }
            }
            ChatEvent::Completed { tool_calls } => {
                self.request_task = None;
                self.history.push(Message::Assistant {
                    text: std::mem::take(&mut self.streaming_text),
                    tool_calls: tool_calls.clone(),
                });
                if tool_calls.is_empty() {
                    self.end_turn();
                    return;
                }
                self.agent_turns += 1;
                if self.agent_turns > MAX_AGENT_TURNS {
                    self.transcript.push(Entry::Error(format!(
                        "stopped after {MAX_AGENT_TURNS} consecutive tool rounds"
                    )));
                    self.end_turn();
                    return;
                }
                self.pending_calls = tool_calls.into();
                self.advance_tools();
            }
            ChatEvent::Error(message) => {
                self.request_task = None;
                self.streaming_text.clear();
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
        self.save_session();
        if self.history_chars() > self.config.compact_threshold_chars && !self.compacting {
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
        match self.pending_calls.front() {
            None => self.start_request(),
            Some(call) if tools::requires_approval(&call.name) && !self.auto_approve => {
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
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let (content, is_error) = tools::execute(&call).await;
            let _ = tx.send(AppEvent::ToolFinished {
                call,
                content,
                is_error,
            });
        });
    }

    fn finish_tool(&mut self, call: ToolCall, content: String, is_error: bool) {
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

    pub fn approve_pending(&mut self, all: bool) {
        if all {
            self.auto_approve = true;
        }
        if let Some(call) = self.pending_calls.pop_front() {
            self.run_tool(call);
        }
    }

    pub fn deny_pending(&mut self) {
        if let Some(call) = self.pending_calls.pop_front() {
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
        if self.compacting {
            self.transcript.push(Entry::Error(
                "context compaction in progress — try again in a moment".into(),
            ));
            return;
        }
        self.textarea = make_textarea();

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
        self.transcript.push(Entry::User(text.clone()));
        self.history.push(Message::User(text));
        self.scroll_from_bottom = 0;
        self.agent_turns = 0;
        self.start_request();
    }

    pub fn paste(&mut self, text: &str) {
        if self.mode == Mode::Input {
            self.textarea
                .insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
        }
    }

    fn run_slash_command(&mut self, command: &str) {
        match command.trim() {
            "model" | "models" => self.open_picker(),
            "clear" | "new" => self.reset_session(),
            "resume" | "sessions" => self.open_sessions(),
            "compact" => {
                if self.history.is_empty() {
                    self.transcript.push(Entry::Info("nothing to compact".into()));
                } else {
                    self.transcript.push(Entry::Info("compacting context…".into()));
                    self.start_compaction();
                }
            }
            "quit" | "exit" => self.should_quit = true,
            "help" => self.transcript.push(Entry::Info(
                "commands: /model /resume /new /compact /quit — keys: Ctrl+P models, Alt+Enter newline, PgUp/PgDn scroll, Esc cancel, Ctrl+C quit"
                    .into(),
            )),
            other => self
                .transcript
                .push(Entry::Error(format!("unknown command: /{other}"))),
        }
    }

    fn start_request(&mut self) {
        let Some(model) = self.model.clone() else {
            return;
        };
        self.mode = Mode::Streaming;
        self.streaming_text.clear();
        self.transcript.push(Entry::Assistant(String::new()));

        let request = ChatRequest {
            model,
            system: system_prompt(),
            messages: self.history.clone(),
            tools: tools::definitions(),
        };
        let config = self.config.clone();
        let tx = self.tx.clone();
        self.request_task = Some(tokio::spawn(async move {
            let (chat_tx, mut chat_rx) = tokio::sync::mpsc::unbounded_channel();
            let stream = tokio::spawn(providers::stream_chat(config, request, chat_tx));
            while let Some(event) = chat_rx.recv().await {
                if tx.send(AppEvent::Chat(event)).is_err() {
                    break;
                }
            }
            let _ = stream.await;
        }));
    }

    pub fn cancel_request(&mut self) {
        if let Some(task) = self.request_task.take() {
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
        }
        self.pending_calls.clear();
        self.transcript.push(Entry::Info("cancelled".into()));
        self.agent_turns = 0;
        self.mode = Mode::Input;
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

    // ---- sessions ----

    pub fn save_session(&self) {
        if self.history.is_empty() {
            return;
        }
        let title = self
            .history
            .iter()
            .find_map(|m| match m {
                Message::User(t) => Some(
                    t.lines()
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
            model: self.model.clone(),
            history: self.history.clone(),
            transcript: self.transcript.clone(),
        };
        if let Err(e) = session::save(&snapshot) {
            // Persistence is best-effort; the conversation itself still works.
            eprintln!("failed to save session: {e:#}");
        }
    }

    fn reset_session(&mut self) {
        self.save_session();
        self.session_id = session::new_id();
        self.history.clear();
        self.transcript.clear();
        self.agent_turns = 0;
        self.scroll_from_bottom = 0;
        self.transcript
            .push(Entry::Info("started a new session".into()));
    }

    pub fn open_sessions(&mut self) {
        let sessions = session::list();
        if sessions.is_empty() {
            self.transcript
                .push(Entry::Info("no saved sessions yet".into()));
            return;
        }
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
                self.session_id = loaded.id;
                self.history = loaded.history;
                self.transcript = loaded.transcript;
                self.scroll_from_bottom = 0;
                self.agent_turns = 0;
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
                Message::User(t) => t.len(),
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
            messages: vec![Message::User(format!(
                "Summarize the conversation below so a successor agent can continue seamlessly. \
                 Capture: the user's goals, decisions made, files created or modified and their \
                 current state, commands run with relevant outcomes, and unresolved tasks. \
                 Output only the summary.\n\n<conversation>\n{flat}\n</conversation>"
            ))],
            tools: Vec::new(),
        };
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
                    ChatEvent::Completed { .. } => {}
                }
            }
            let result = match error {
                Some(e) => Err(e),
                None if text.trim().is_empty() => Err("empty summary".into()),
                None => Ok(text),
            };
            let _ = tx.send(AppEvent::CompactionDone(result));
        });
    }

    fn finish_compaction(&mut self, result: Result<String, String>) {
        self.compacting = false;
        match result {
            Ok(summary) => {
                let before = self.history_chars();
                self.history = vec![Message::User(format!(
                    "Context summary of our conversation so far (earlier messages were compacted):\n\n{}",
                    summary.trim()
                ))];
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

fn make_textarea() -> TextArea<'static> {
    let mut textarea = TextArea::default();
    textarea.set_cursor_line_style(Style::default());
    textarea.set_placeholder_text("type a message — Enter send, Alt+Enter newline, /help");
    textarea.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().fg(Color::Cyan)),
    );
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
            Message::User(t) => flat.push_str(&format!("[user]\n{}\n\n", cap(t))),
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
    format!(
        "You are shaltaiboltai, an agentic coding assistant running in a terminal. \
         The user's working directory is {cwd} on {}. \
         Use the available tools to read and modify files and run commands when the task calls for it. \
         Prefer small, verifiable steps and report what you did. \
         Format responses in markdown.",
        std::env::consts::OS,
    )
}
