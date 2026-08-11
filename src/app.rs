use crate::config::Config;
use crate::images;
use crate::orchestration::{self, PlannedTask, WorkerOutcome};
use crate::providers::{
    self, ChatEvent, ChatRequest, ImageData, Message, ModelEntry, ProviderKind, RequestPolicy,
    ToolCall, Usage, UserContent,
};
use crate::session;
use crate::theme::{self, Theme};
use crate::tools;
use futures_util::{stream::FuturesUnordered, StreamExt};
use ratatui::style::Style;
use ratatui::text::Line;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{HashSet, VecDeque};
use std::future::Future;
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

/// Reserve space beyond worker evidence for the lead's system prompt, root
/// task, and generated answer/tool calls.
const ORCHESTRATION_SYSTEM_HEADROOM: usize = 8_000;
const MIN_SYNTHESIS_EVIDENCE_CHARS: usize = 4_000;

/// One-turn lookahead keeps memory bounded and makes failure recovery exact:
/// there can never be a second draft to merge with a restored queued prompt.
#[derive(Clone)]
struct QueuedPrompt {
    text: String,
    staged_images: Vec<(String, ImageData)>,
    referenced_images: Vec<(String, ImageData)>,
    model: ModelEntry,
}

struct ActiveRootPrompt {
    prompt: QueuedPrompt,
    history_len: usize,
    transcript_len: usize,
    observed_activity: bool,
}

impl QueuedPrompt {
    fn image_count(&self) -> usize {
        self.staged_images.len() + self.referenced_images.len()
    }

    fn into_images(mut self) -> (String, ModelEntry, Vec<(String, ImageData)>) {
        self.staged_images.append(&mut self.referenced_images);
        (self.text, self.model, self.staged_images)
    }
}

struct RestoredReferences {
    text: String,
    images: Vec<(String, ImageData)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrchestrationPhase {
    Planning,
    Confirming,
    Workers,
    Coordinating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TeamContextFit {
    Fits,
    NeedsCompaction,
    Impossible,
}

struct OrchestrationRun {
    id: u64,
    worker_count: usize,
    planner_model: ModelEntry,
    prompt: Option<QueuedPrompt>,
    tasks: Vec<PlannedTask>,
    outcomes: Vec<WorkerOutcome>,
    entry_indices: Vec<usize>,
    phase: OrchestrationPhase,
}

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
        name: "team",
        aliases: &["orchestrate"],
        args: Some("[2-4|off]"),
        description: "orchestrate the next prompt with read-only workers",
    },
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
        name: "refresh",
        aliases: &["reload"],
        args: None,
        description: "rediscover available models",
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
    ModelsDiscovered {
        models: Vec<ModelEntry>,
        finished: bool,
    },
    ToolFinished {
        gen: u64,
        call: ToolCall,
        content: String,
        is_error: bool,
    },
    CompactionDone {
        session_id: String,
        compaction_gen: u64,
        result: Result<String, String>,
    },
    OrchestrationPlanned {
        run_id: u64,
        result: Result<Vec<PlannedTask>, String>,
    },
    OrchestrationWorkerFinished {
        run_id: u64,
        outcome: WorkerOutcome,
    },
    OrchestrationWorkersFinished {
        run_id: u64,
    },
}

#[derive(Debug, PartialEq)]
pub enum Mode {
    Input,
    Streaming,
    RunningTool,
    Approval,
    OrchestrationConfirm,
    Orchestrating,
    ModelPicker,
    SessionPicker,
    ThemePicker,
    Help,
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
    Agent {
        name: String,
        model: String,
        status: String,
        summary: String,
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
    pub discovering: bool,
    pub theme: Theme,
    pub theme_index: usize,
    theme_revert: Option<Theme>,

    pub models: Vec<ModelEntry>,
    pub model: Option<ModelEntry>,
    /// Prevent a late discovery result from overriding a model the user chose
    /// while provider probes were still running.
    model_selected_explicitly: bool,
    /// Accumulator for the active incremental discovery pass. `models` is the
    /// user-visible snapshot; this is reset on every /refresh.
    discovery_models: Vec<ModelEntry>,
    /// A discovery pass may finish while a model→tool→model turn is active.
    /// Defer automatic selection so one logical turn never crosses providers.
    discovery_reconciliation_deferred: bool,
    pub picker_index: usize,
    pub picker_filter: String,

    pub sessions: Vec<session::Meta>,
    pub session_index: usize,
    session_id: String,

    pub transcript: Vec<Entry>,
    /// Bumped on structural transcript changes (clear/replace/pop) and theme
    /// switches so the renderer knows its per-entry cache is stale.
    pub transcript_rev: u64,
    /// Earliest transcript entry changed in place since the last draw. A
    /// coalesced stream delta followed by tool activity can move that entry
    /// away from the tail, so an index is more reliable than a tail revision.
    pub transcript_dirty_from: Option<usize>,
    pub history: Vec<Message>,
    history_chars_cache: Cell<Option<(usize, usize)>>,
    pub scroll_from_bottom: usize,
    pub last_usage: Option<Usage>,

    pub textarea: TextArea<'static>,
    input_history: Vec<String>,
    input_history_pos: Option<usize>,
    input_draft: String,
    pub slash_index: usize,
    slash_dismissed: bool,
    composer_notice: Option<String>,

    // Statusline environment, refreshed at startup and after each turn/tool.
    pub cwd_display: String,
    pub git_branch: Option<String>,

    /// Images staged for the next message: (display name, encoded data).
    pub pending_images: Vec<(String, ImageData)>,

    /// A single next message captured while the current agent turn runs. It is
    /// deliberately kept out of transcript/history until it is dispatched.
    queued_prompt: Option<QueuedPrompt>,
    /// Frozen bytes for image paths from a queued prompt that was restored.
    /// Editing the text clears these; an unchanged resubmission reuses them.
    restored_references: Option<RestoredReferences>,
    /// Clearing a restored inline attachment suppresses re-resolution while
    /// the text is unchanged; editing the text makes paths eligible again.
    suppressed_reference_text: Option<String>,

    /// Diff preview for the tool call currently awaiting approval.
    pub approval_preview: Option<Vec<(char, String)>>,
    pub approval_scroll: usize,
    /// Approval shortcuts are armed explicitly when type-ahead was available,
    /// so a typed `a`, `y`, or `n` can never decide a newly arrived modal.
    pub approval_focused: bool,

    // Renderer cache, managed by ui::draw.
    pub render_cache: Vec<Vec<Line<'static>>>,
    /// Logical first-line offset for every cached entry. The UI binary-searches
    /// this index instead of walking the whole conversation every frame.
    pub render_cache_starts: Vec<usize>,
    pub render_cache_total_lines: usize,
    pub render_cache_width: usize,
    pub render_cache_rev: u64,

    gen: u64,
    streaming_text: String,
    pending_calls: VecDeque<ToolCall>,
    /// Explicit grants for narrowly scoped paths/searches/commands. These are
    /// intentionally conversation-local and never persisted.
    approved_scopes: HashSet<String>,
    agent_turns: usize,
    active_turn_model: Option<ModelEntry>,
    active_turn_can_promote_queue: bool,
    /// Exact ownership of the submitted root prompt until the provider emits
    /// activity. A launch/model error can then roll it back for lossless retry.
    active_root_prompt: Option<ActiveRootPrompt>,
    /// One-shot worker count armed by `/team`; consumed by the next prompt.
    team_workers: Option<usize>,
    /// The planning/fan-out/coordinator state for one orchestration run.
    orchestration_run: Option<OrchestrationRun>,
    /// Ephemeral, untrusted worker evidence injected into coordinator requests.
    orchestration_context: Option<String>,
    pub orchestration_confirm_focused: bool,
    orchestration_gen: u64,
    compaction_gen: u64,
    request_task: Option<JoinHandle<()>>,
    tool_task: Option<JoinHandle<()>>,
    compaction_task: Option<JoinHandle<()>>,
    orchestration_task: Option<JoinHandle<()>>,

    tx: UnboundedSender<AppEvent>,
}

impl App {
    pub fn new(config: Config, tx: UnboundedSender<AppEvent>) -> Self {
        let theme = session::load_theme_name()
            .or_else(|| config.theme.clone())
            .and_then(|name| theme::by_name(&name))
            .unwrap_or(theme::DEFAULT);
        let models = providers::immediate_models(&config);
        let model = match config.default_model.as_ref() {
            Some(wanted) if providers::cli_model_selector(wanted).is_none() => {
                models.iter().find(|entry| &entry.id == wanted).cloned()
            }
            Some(_) => None,
            None => models.first().cloned(),
        };
        let discovery_models = models.clone();
        let mut app = App {
            config,
            mode: Mode::Input,
            should_quit: false,
            compacting: false,
            discovering: true,
            theme,
            theme_index: 0,
            theme_revert: None,
            models,
            model,
            model_selected_explicitly: false,
            discovery_models,
            discovery_reconciliation_deferred: false,
            picker_index: 0,
            picker_filter: String::new(),
            sessions: Vec::new(),
            session_index: 0,
            session_id: session::new_id(),
            transcript: Vec::new(),
            transcript_rev: 0,
            transcript_dirty_from: None,
            history: Vec::new(),
            history_chars_cache: Cell::new(None),
            scroll_from_bottom: 0,
            last_usage: None,
            textarea: make_textarea(&theme),
            input_history: session::load_input_history(),
            input_history_pos: None,
            input_draft: String::new(),
            slash_index: 0,
            slash_dismissed: false,
            composer_notice: None,
            cwd_display: String::new(),
            git_branch: None,
            pending_images: Vec::new(),
            queued_prompt: None,
            restored_references: None,
            suppressed_reference_text: None,
            approval_preview: None,
            approval_scroll: 0,
            approval_focused: true,
            render_cache: Vec::new(),
            render_cache_starts: Vec::new(),
            render_cache_total_lines: 0,
            render_cache_width: 0,
            render_cache_rev: 0,
            gen: 0,
            streaming_text: String::new(),
            pending_calls: VecDeque::new(),
            approved_scopes: HashSet::new(),
            agent_turns: 0,
            active_turn_model: None,
            active_turn_can_promote_queue: true,
            active_root_prompt: None,
            team_workers: None,
            orchestration_run: None,
            orchestration_context: None,
            orchestration_confirm_focused: false,
            orchestration_gen: 0,
            compaction_gen: 0,
            request_task: None,
            tool_task: None,
            compaction_task: None,
            orchestration_task: None,
            tx,
        };
        app.transcript.push(Entry::Banner {
            title: "Ready to build".into(),
            subtitle: format!(
                "v{} · Describe a change, ask about the code, or type / for commands. F1 opens the keyboard guide.",
                env!("CARGO_PKG_VERSION"),
            ),
        });
        app.refresh_environment();
        app.spawn_discovery();
        app
    }

    pub fn is_busy(&self) -> bool {
        matches!(
            self.mode,
            Mode::Streaming | Mode::RunningTool | Mode::Orchestrating
        ) || self.compacting
            || self.discovering
    }

    /// Cwd and git branch for the statusline. Cheap (one small file read);
    /// refreshed after turns/tools and on a slow idle tick, not per frame.
    pub fn refresh_environment(&mut self) {
        self.cwd_display = std::env::current_dir()
            .map(|p| shorten_path(&p))
            .unwrap_or_default();
        self.git_branch = read_git_branch(std::path::Path::new(".git"));
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
        self.composer_notice = None;
        // Frozen/suppressed inline-reference state is valid only for the exact
        // text it belongs to. Comparing content also makes a no-op Delete or
        // Backspace harmless.
        if self.restored_references.is_some() || self.suppressed_reference_text.is_some() {
            let text = self.textarea.lines().join("\n").trim().to_owned();
            if self
                .restored_references
                .as_ref()
                .is_some_and(|restored| restored.text != text)
            {
                self.restored_references = None;
            }
            if self
                .suppressed_reference_text
                .as_ref()
                .is_some_and(|suppressed| suppressed != &text)
            {
                self.suppressed_reference_text = None;
            }
        }
    }

    /// Whether keyboard and paste events should currently reach the composer.
    /// A captured lookahead prompt locks the editor until it is sent or
    /// restored, which keeps attachment ownership unambiguous.
    pub fn composer_accepts_input(&self) -> bool {
        if self.queued_prompt.is_some() {
            return false;
        }
        match self.mode {
            Mode::Input | Mode::Streaming | Mode::RunningTool => true,
            Mode::Orchestrating => self
                .orchestration_run
                .as_ref()
                .is_some_and(|run| run.phase == OrchestrationPhase::Workers),
            Mode::Approval => !self.approval_focused,
            _ => false,
        }
    }

    pub fn team_workers(&self) -> Option<usize> {
        self.team_workers
    }

    pub fn orchestration_plan(&self) -> &[PlannedTask] {
        self.orchestration_run
            .as_ref()
            .map_or(&[], |run| run.tasks.as_slice())
    }

    pub fn orchestration_planner(&self) -> Option<&ModelEntry> {
        self.orchestration_run
            .as_ref()
            .map(|run| &run.planner_model)
    }

    pub fn orchestration_run_id(&self) -> Option<u64> {
        self.orchestration_run.as_ref().map(|run| run.id)
    }

    pub fn orchestration_status(&self) -> Option<String> {
        let run = self.orchestration_run.as_ref()?;
        Some(match run.phase {
            OrchestrationPhase::Planning => "TEAM · planning".into(),
            OrchestrationPhase::Confirming => "TEAM · plan ready · Tab to review".into(),
            OrchestrationPhase::Workers => format!(
                "TEAM · workers {}/{} · Esc cancel",
                run.outcomes.len(),
                run.tasks.len()
            ),
            OrchestrationPhase::Coordinating => match self.mode {
                Mode::Approval => "TEAM · applying · approval needed".into(),
                Mode::RunningTool => "TEAM · applying changes · Esc cancel".into(),
                _ => "TEAM · synthesizing · Esc cancel".into(),
            },
        })
    }

    pub fn focus_orchestration_confirm(&mut self) {
        if self.mode == Mode::OrchestrationConfirm {
            self.orchestration_confirm_focused = true;
        }
    }

    pub fn toggle_orchestration_confirm_focus(&mut self) {
        if self.mode == Mode::OrchestrationConfirm {
            self.orchestration_confirm_focused = !self.orchestration_confirm_focused;
        }
    }

    pub fn queued_prompt_count(&self) -> usize {
        usize::from(self.queued_prompt.is_some())
    }

    /// Token carried by events for the currently active provider/tool phase.
    /// Each new request or tool execution advances it so delayed events from a
    /// previous round cannot mutate the next phase of the same agent turn.
    pub fn event_generation(&self) -> u64 {
        self.gen
    }

    pub fn queued_image_count(&self) -> usize {
        self.queued_prompt
            .as_ref()
            .map_or(0, QueuedPrompt::image_count)
    }

    pub fn pending_image_count(&self) -> usize {
        let Some(restored) = self.restored_references.as_ref() else {
            return self.pending_images.len();
        };
        let text = self.textarea.lines().join("\n").trim().to_owned();
        self.pending_images.len()
            + if restored.text == text {
                restored.images.len()
            } else {
                0
            }
    }

    pub fn composer_notice(&self) -> Option<&str> {
        self.composer_notice.as_deref()
    }

    pub fn toggle_approval_focus(&mut self) {
        if self.mode != Mode::Approval {
            return;
        }
        self.approval_focused = !(self.approval_focused && self.queued_prompt.is_none());
    }

    pub fn focus_approval(&mut self) {
        if self.mode == Mode::Approval {
            self.approval_focused = true;
        }
    }

    fn spawn_discovery(&mut self) {
        self.discovery_models = providers::immediate_models(&self.config);
        self.discovery_reconciliation_deferred = false;
        let config = self.config.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let batch_tx = tx.clone();
            providers::discover_dynamic_models(&config, |models| {
                let _ = batch_tx.send(AppEvent::ModelsDiscovered {
                    models,
                    finished: false,
                });
            })
            .await;
            let _ = tx.send(AppEvent::ModelsDiscovered {
                models: Vec::new(),
                finished: true,
            });
        });
    }

    pub fn refresh_models(&mut self) {
        if self.discovering {
            self.transcript
                .push(Entry::Info("model discovery is already running".into()));
            return;
        }
        self.discovering = true;
        self.transcript
            .push(Entry::Info("refreshing available models…".into()));
        self.spawn_discovery();
    }

    // ---- background events ----

    pub fn on_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::ModelsDiscovered { models, finished } => {
                for model in models {
                    if !self
                        .discovery_models
                        .iter()
                        .any(|known| known.id == model.id && known.provider == model.provider)
                    {
                        self.discovery_models.push(model);
                    }
                }
                // An explicitly entered CLI selector may not be present in a
                // provider's detected catalog. Preserve it across /refresh as
                // long as that CLI itself is still available.
                if let Some(current) = self.model.as_ref().filter(|current| {
                    self.model_selected_explicitly
                        && current.provider.is_sub_agent()
                        && current.id.contains(':')
                        && self
                            .discovery_models
                            .iter()
                            .any(|model| model.provider == current.provider)
                }) {
                    if !self
                        .discovery_models
                        .iter()
                        .any(|model| model.provider == current.provider && model.id == current.id)
                    {
                        self.discovery_models.push(current.clone());
                    }
                }
                let models = self.discovery_models.clone();
                self.models = models.clone();
                if finished {
                    self.discovering = false;
                    if models.is_empty() {
                        self.transcript.push(Entry::Error(
                            "no models found: configure Anthropic/OpenAI, start Ollama, or sign in to Claude Code/Codex — /refresh retries".into(),
                        ));
                    } else {
                        self.transcript.push(Entry::Info(format!(
                            "{} models available across {} provider(s)",
                            models.len(),
                            provider_count(&models),
                        )));
                    }
                }
                if self.agent_turn_active() {
                    self.discovery_reconciliation_deferred = true;
                } else {
                    self.reconcile_discovered_model(finished);
                }
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
            AppEvent::CompactionDone {
                session_id,
                compaction_gen,
                result,
            } => {
                // A compaction started in another session must not replace
                // this one's history or clear a newer session's busy state.
                if session_id == self.session_id
                    && compaction_gen == self.compaction_gen
                    && self.compacting
                {
                    self.compacting = false;
                    if let Some(task) = self.compaction_task.take() {
                        task.abort();
                    }
                    self.finish_compaction(result);
                }
            }
            AppEvent::OrchestrationPlanned { run_id, result } => {
                if self.orchestration_run.as_ref().is_some_and(|run| {
                    run.id == run_id && run.phase == OrchestrationPhase::Planning
                }) {
                    self.orchestration_task = None;
                    self.finish_orchestration_plan(result);
                }
            }
            AppEvent::OrchestrationWorkerFinished { run_id, outcome } => {
                if self
                    .orchestration_run
                    .as_ref()
                    .is_some_and(|run| run.id == run_id && run.phase == OrchestrationPhase::Workers)
                {
                    self.finish_orchestration_worker(outcome);
                }
            }
            AppEvent::OrchestrationWorkersFinished { run_id } => {
                if self
                    .orchestration_run
                    .as_ref()
                    .is_some_and(|run| run.id == run_id && run.phase == OrchestrationPhase::Workers)
                {
                    self.orchestration_task = None;
                    self.finish_orchestration_workers();
                }
            }
        }
    }

    fn agent_turn_active(&self) -> bool {
        matches!(
            self.mode,
            Mode::Streaming
                | Mode::RunningTool
                | Mode::Approval
                | Mode::OrchestrationConfirm
                | Mode::Orchestrating
        ) || !self.pending_calls.is_empty()
            || self.queued_prompt.is_some()
            || self.orchestration_run.is_some()
    }

    fn reconcile_discovered_model(&mut self, finished: bool) {
        let configured_default = self.config.default_model.as_ref().and_then(|wanted| {
            if providers::cli_model_selector(wanted).is_some() {
                providers::custom_cli_model(wanted, &self.models)
            } else {
                self.models
                    .iter()
                    .find(|model| &model.id == wanted)
                    .cloned()
            }
        });
        if let Some(default) = configured_default.as_ref() {
            if !self
                .models
                .iter()
                .any(|model| model.id == default.id && model.provider == default.provider)
            {
                self.models.push(default.clone());
                self.discovery_models.push(default.clone());
            }
        }
        let current_available = self.model.as_ref().is_some_and(|current| {
            self.models
                .iter()
                .any(|model| model.id == current.id && model.provider == current.provider)
        });
        if !self.model_selected_explicitly {
            if let Some(default) = configured_default {
                self.model = Some(default);
            } else if self.config.default_model.is_some() {
                if finished {
                    let wanted = self.config.default_model.as_deref().unwrap_or("unknown");
                    self.model = None;
                    self.transcript.push(Entry::Error(format!(
                        "configured model {wanted} is unavailable — choose another with Ctrl+P"
                    )));
                }
            } else if self.model.is_none() || (finished && !current_available) {
                self.model = self.models.first().cloned();
            }
        } else if finished && !current_available {
            let unavailable = self
                .model
                .as_ref()
                .map_or_else(|| "saved model".to_owned(), |model| model.id.clone());
            self.model = None;
            self.model_selected_explicitly = false;
            self.transcript.push(Entry::Error(format!(
                "selected model {unavailable} is unavailable — choose another with Ctrl+P"
            )));
        }
    }

    fn apply_deferred_model_reconciliation(&mut self) {
        if self.discovery_reconciliation_deferred {
            self.discovery_reconciliation_deferred = false;
            self.reconcile_discovered_model(!self.discovering);
        }
    }

    fn on_chat_event(&mut self, event: ChatEvent) {
        match event {
            ChatEvent::TextDelta(text) => {
                if !text.is_empty() {
                    if let Some(active) = self.active_root_prompt.as_mut() {
                        active.observed_activity = true;
                    }
                }
                self.streaming_text.push_str(&text);
                // Append to the current assistant block, or start a new one if
                // a sub-agent's tool activity interrupted it.
                match self.transcript.last_mut() {
                    Some(Entry::Assistant(buf)) => buf.push_str(&text),
                    _ => self.transcript.push(Entry::Assistant(text)),
                }
                let changed = self.transcript.len().saturating_sub(1);
                self.transcript_dirty_from = Some(
                    self.transcript_dirty_from
                        .map_or(changed, |earlier| earlier.min(changed)),
                );
            }
            ChatEvent::Notice(message) => {
                let replaced_placeholder =
                    matches!(self.transcript.last(), Some(Entry::Assistant(t)) if t.is_empty());
                if replaced_placeholder {
                    self.transcript.pop();
                }
                self.transcript.push(Entry::Info(message));
                if replaced_placeholder {
                    let changed = self.transcript.len().saturating_sub(1);
                    self.transcript_dirty_from = Some(
                        self.transcript_dirty_from
                            .map_or(changed, |earlier| earlier.min(changed)),
                    );
                }
            }
            ChatEvent::ToolActivity { summary, is_error } => {
                if let Some(active) = self.active_root_prompt.as_mut() {
                    active.observed_activity = true;
                }
                // Sub-agent tools have already run inside the CLI; this is
                // display-only and never enters our approval flow.
                let replaced_placeholder =
                    matches!(self.transcript.last(), Some(Entry::Assistant(t)) if t.is_empty());
                if replaced_placeholder {
                    self.transcript.pop();
                }
                self.transcript.push(Entry::Tool {
                    summary,
                    result: String::new(),
                    is_error,
                });
                if replaced_placeholder {
                    let changed = self.transcript.len().saturating_sub(1);
                    self.transcript_dirty_from = Some(
                        self.transcript_dirty_from
                            .map_or(changed, |earlier| earlier.min(changed)),
                    );
                }
            }
            ChatEvent::Completed {
                tool_calls,
                stop_reason,
                usage,
            } => {
                if let Some(active) = self.active_root_prompt.as_mut() {
                    active.observed_activity = true;
                }
                if let Some(task) = self.request_task.take() {
                    task.abort();
                }
                // Remove the streaming placeholder on the next draw even when
                // no text arrived, without invalidating earlier cached entries.
                if !self.transcript.is_empty() {
                    let changed = self.transcript.len() - 1;
                    self.transcript_dirty_from = Some(
                        self.transcript_dirty_from
                            .map_or(changed, |earlier| earlier.min(changed)),
                    );
                }
                if usage.is_some() {
                    self.last_usage = usage;
                }
                self.history.push(Message::Assistant {
                    text: std::mem::take(&mut self.streaming_text),
                    tool_calls: tool_calls.clone(),
                });
                if !clean_stop_reason(stop_reason.as_deref(), !tool_calls.is_empty()) {
                    self.active_turn_can_promote_queue = false;
                    let message = match stop_reason.as_deref() {
                        Some("length") => "response was truncated by the output token limit".into(),
                        Some(reason) => {
                            format!("response stopped before normal completion: {reason}")
                        }
                        None => "response stream ended before normal completion".into(),
                    };
                    self.transcript.push(Entry::Error(message));
                    if !tool_calls.is_empty() {
                        self.repair_dangling_tool_calls();
                    }
                    self.end_turn();
                    return;
                }
                if tool_calls.is_empty() {
                    self.end_turn();
                    return;
                }
                self.agent_turns += 1;
                if self.agent_turns > MAX_AGENT_TURNS {
                    self.active_turn_can_promote_queue = false;
                    self.transcript.push(Entry::Error(format!(
                        "stopped after {MAX_AGENT_TURNS} consecutive tool rounds"
                    )));
                    self.repair_dangling_tool_calls();
                    self.end_turn();
                    return;
                }
                // The provider request is finished. Tool work gets a distinct
                // phase token, so any late provider Error/Text event is stale.
                self.gen += 1;
                self.pending_calls = tool_calls.into();
                self.advance_tools();
            }
            ChatEvent::Error(message) => {
                let rollback_team_root = self.team_coordinator_has_no_activity();
                if let Some(task) = self.request_task.take() {
                    task.abort();
                }
                if let Some(task) = self.tool_task.take() {
                    task.abort();
                }
                if rollback_team_root {
                    self.fail_orchestration_root(&format!(
                        "team synthesis failed before producing output: {message}"
                    ));
                    return;
                }
                let active_root = self.active_root_prompt.take();
                let composer_untouched = self.input_is_empty()
                    && self.pending_images.is_empty()
                    && self.restored_references.is_none()
                    && self.suppressed_reference_text.is_none();
                let restore_current = self.queued_prompt.is_none()
                    && composer_untouched
                    && active_root
                        .as_ref()
                        .is_some_and(|active| !active.observed_activity);
                if restore_current {
                    let active = active_root.as_ref().expect("checked above");
                    self.history.truncate(active.history_len);
                    self.history_chars_cache.set(None);
                    self.transcript.truncate(active.transcript_len);
                    self.transcript_rev += 1;
                    self.streaming_text.clear();
                } else {
                    // Keep partial text the user already saw consistent with
                    // what the model will see next turn.
                    let text = std::mem::take(&mut self.streaming_text);
                    if !text.is_empty() {
                        self.history.push(Message::Assistant {
                            text,
                            tool_calls: Vec::new(),
                        });
                    } else if matches!(self.transcript.last(), Some(Entry::Assistant(t)) if t.is_empty())
                    {
                        self.transcript.pop();
                        let changed = self.transcript.len();
                        self.transcript_dirty_from = Some(
                            self.transcript_dirty_from
                                .map_or(changed, |earlier| earlier.min(changed)),
                        );
                    }
                }
                self.transcript.push(Entry::Error(message));
                self.pending_calls.clear();
                self.approval_preview = None;
                self.repair_dangling_tool_calls();
                // Fence any already-buffered events from this failed root turn
                // before the restored draft can be submitted again.
                self.gen += 1;
                self.agent_turns = 0;
                self.mode = Mode::Input;
                self.orchestration_run = None;
                self.orchestration_context = None;
                self.active_turn_model = None;
                self.active_turn_can_promote_queue = true;
                self.approval_focused = true;
                if restore_current {
                    self.restore_prompt(
                        active_root.expect("checked above").prompt,
                        "message restored — the provider rejected it before producing output",
                    );
                } else {
                    self.restore_queued_prompt("the previous request failed");
                }
                self.apply_deferred_model_reconciliation();
            }
        }
    }

    /// A user turn finished with a final answer: persist, and compact the
    /// context in the background if it has grown past the threshold.
    fn end_turn(&mut self) {
        // A root turn is closed before another can start. Any provider events
        // already buffered with the old generation are ignored.
        self.gen += 1;
        self.agent_turns = 0;
        self.mode = Mode::Input;
        self.orchestration_run = None;
        self.orchestration_context = None;
        self.active_turn_model = None;
        self.active_root_prompt = None;
        self.refresh_environment();
        let can_promote = std::mem::replace(&mut self.active_turn_can_promote_queue, true);
        if !self.save_session_checked() {
            self.restore_queued_prompt("the completed turn could not be saved");
            self.apply_deferred_model_reconciliation();
            return;
        }
        if !can_promote {
            self.restore_queued_prompt("the previous response did not finish cleanly");
            self.apply_deferred_model_reconciliation();
            return;
        }
        if self.queued_prompt.is_none() {
            self.apply_deferred_model_reconciliation();
        }
        if self.context_over_threshold() && !self.compacting {
            self.transcript.push(Entry::Info(
                "context exceeded threshold — compacting in the background".into(),
            ));
            self.start_compaction();
            if !self.compacting {
                self.restore_queued_prompt("context compaction could not start");
                self.apply_deferred_model_reconciliation();
            }
            return;
        }
        self.dispatch_queued_prompt();
    }

    /// Process the queue of tool calls returned by the model: pause for
    /// approval where required, execute otherwise, and when the queue is
    /// drained send the results back to the model.
    fn advance_tools(&mut self) {
        self.approval_preview = None;
        self.approval_scroll = 0;
        match self.pending_calls.front() {
            None => self.start_request(),
            Some(call)
                if tools::requires_approval(call)
                    && !self.approved_scopes.contains(&tools::approval_scope(call)) =>
            {
                self.approval_preview = tools::approval_preview(call);
                // Always require an explicit Tab before decision shortcuts are
                // armed. This also fences keys buffered just before the modal
                // arrived, even when the one lookahead slot is already full.
                self.approval_focused = false;
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
        self.gen += 1;
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
                self.approved_scopes.insert(tools::approval_scope(&call));
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
        if self.mode != Mode::Input {
            return;
        }
        if self.queued_prompt.is_some() {
            self.composer_notice = Some("next message is already queued".into());
            return;
        }
        let text = self.textarea.lines().join("\n").trim().to_owned();
        if text.is_empty() {
            return;
        }
        // Slash commands stay available while compacting. A normal message is
        // captured as the one-turn lookahead and sent after compaction.
        if self.compacting && !text.starts_with('/') && self.team_workers.is_some() {
            self.composer_notice =
                Some("team mode will start after context compaction — your draft is safe".into());
            return;
        }
        if self.compacting && !text.starts_with('/') {
            self.queue_input();
            return;
        }
        // Keep the user's draft intact while providers are still being found
        // (or when none are configured). Slash commands remain available.
        if !text.starts_with('/') && self.model.is_none() {
            let message = if self.discovering {
                "still discovering models — your draft is safe"
            } else {
                "no model selected — configure a provider or run /refresh, then choose one with Ctrl+P"
            };
            self.transcript.push(Entry::Error(message.into()));
            return;
        }

        if !text.starts_with('/') && self.team_workers.is_some() {
            let worker_count = self.team_workers.expect("checked above");
            if self.pending_image_count() > 0 || !images::extract_image_paths(&text).is_empty() {
                self.transcript.push(Entry::Error(
                    "team mode is text-only for now — attachments and prompt were preserved".into(),
                ));
                return;
            }
            if let Err(error) = orchestration::validate_root_task(&text) {
                self.transcript.push(Entry::Error(format!(
                    "could not start team mode: {error} — prompt preserved"
                )));
                return;
            }
            let model = self
                .model
                .clone()
                .expect("a model was checked before team planning");
            if providers::is_cli_default_model(&model) {
                self.transcript.push(Entry::Error(format!(
                    "team mode needs an explicit {} model so planning and synthesis stay pinned — choose one with /model or Ctrl+P; prompt preserved",
                    model.provider.label()
                )));
                return;
            }
            if orchestration::choose_planner_model(&model, &self.models).is_none()
                || orchestration::choose_worker_models(&model, &self.models, worker_count).len()
                    != worker_count
            {
                self.transcript.push(Entry::Error(
                    "team mode needs at least one workspace-scoped advisory model; Codex is kept as the post-confirmation lead because its read-only CLI sandbox can read outside the workspace — prompt preserved"
                        .into(),
                ));
                return;
            }
            match self.team_context_fit(worker_count, text.chars().count()) {
                TeamContextFit::Fits => {}
                TeamContextFit::NeedsCompaction => {
                    self.transcript.push(Entry::Info(
                        "team run paused — compacting context before fan-out".into(),
                    ));
                    self.start_compaction();
                    if self.compacting {
                        self.composer_notice = Some(
                            "prompt preserved — press Enter again after compaction finishes".into(),
                        );
                    } else {
                        self.transcript.push(Entry::Error(
                            "team context needs compaction, but compaction could not start — prompt preserved"
                                .into(),
                        ));
                    }
                    return;
                }
                TeamContextFit::Impossible => {
                    self.transcript.push(Entry::Error(
                        "team prompt and evidence reserve cannot fit the smallest selected context — shorten the prompt, clear context, or choose a larger-context model; prompt preserved"
                            .into(),
                    ));
                    return;
                }
            }
            let worker_count = self
                .team_workers
                .take()
                .expect("team mode was checked above");
            self.textarea = make_textarea(&self.theme);
            self.remember_input(&text);
            self.start_orchestration_plan(
                QueuedPrompt {
                    text,
                    staged_images: Vec::new(),
                    referenced_images: Vec::new(),
                    model,
                },
                worker_count,
            );
            return;
        }
        self.textarea = make_textarea(&self.theme);
        self.remember_input(&text);

        if let Some(command) = text.strip_prefix('/') {
            self.run_slash_command(command);
            return;
        }

        let model = self
            .model
            .clone()
            .expect("a model was checked before dispatch");
        // Attach staged images plus any image paths referenced in the text
        // (typed or drag-and-dropped onto the terminal).
        let staged_images = std::mem::take(&mut self.pending_images);
        let restored = self.restored_references.take();
        let reuse_restored = restored
            .as_ref()
            .is_some_and(|restored| restored.text == text);
        let mut referenced_images = if reuse_restored {
            restored.map_or_else(Vec::new, |restored| restored.images)
        } else {
            Vec::new()
        };
        let references_suppressed = self
            .suppressed_reference_text
            .take()
            .is_some_and(|suppressed| suppressed == text);
        if !reuse_restored && !references_suppressed {
            for path in images::extract_image_paths(&text) {
                match images::load_image(&path) {
                    Ok(attachment) => referenced_images.push(attachment),
                    Err(e) => self.transcript.push(Entry::Error(format!("{e:#}"))),
                }
            }
        }
        self.dispatch_prompt(QueuedPrompt {
            text,
            staged_images,
            referenced_images,
            model,
        });
    }

    /// Capture the next user turn without touching provider history. This is
    /// intentionally one-slot: after capture the composer locks until the
    /// prompt is dispatched or restored following an abnormal end.
    pub fn queue_input(&mut self) {
        if self.queued_prompt.is_some() {
            self.composer_notice = Some("next message is already queued".into());
            return;
        }
        let text = self.textarea.lines().join("\n").trim().to_owned();
        if text.is_empty() {
            return;
        }
        if text.starts_with('/') {
            self.composer_notice = Some("commands are available after the current turn".into());
            return;
        }
        let Some(model) = self
            .active_turn_model
            .clone()
            .or_else(|| self.model.clone())
        else {
            self.composer_notice = Some("no model available for the next message".into());
            return;
        };

        // Resolve referenced files now, so the queued message owns the exact
        // bytes the user saw when pressing Enter. On failure, keep the draft
        // and originally staged attachments untouched.
        let staged_images = std::mem::take(&mut self.pending_images);
        let mut referenced_images = Vec::new();
        for path in images::extract_image_paths(&text) {
            match images::load_image(&path) {
                Ok(attachment) => referenced_images.push(attachment),
                Err(e) => {
                    self.pending_images = staged_images;
                    self.composer_notice = Some(format!("could not queue message: {e:#}"));
                    return;
                }
            }
        }

        self.textarea = make_textarea(&self.theme);
        self.remember_input(&text);
        self.composer_notice = None;
        self.queued_prompt = Some(QueuedPrompt {
            text,
            staged_images,
            referenced_images,
            model,
        });
    }

    fn start_orchestration_plan(&mut self, prompt: QueuedPrompt, worker_count: usize) {
        debug_assert!(self.orchestration_task.is_none());
        self.orchestration_gen += 1;
        let run_id = self.orchestration_gen;
        let planner_model = orchestration::choose_planner_model(&prompt.model, &self.models)
            .expect("team advisory availability is checked before planning");
        let worker_models =
            orchestration::choose_worker_models(&prompt.model, &self.models, worker_count);
        let request = orchestration::planner_request(
            &planner_model,
            &self.history,
            &prompt.text,
            worker_count,
        )
        .expect("team planner safety is checked before planning");
        self.orchestration_run = Some(OrchestrationRun {
            id: run_id,
            worker_count,
            planner_model,
            prompt: Some(prompt),
            tasks: Vec::new(),
            outcomes: Vec::new(),
            entry_indices: Vec::new(),
            phase: OrchestrationPhase::Planning,
        });
        self.mode = Mode::Orchestrating;
        self.orchestration_confirm_focused = false;
        self.transcript.push(Entry::Info(format!(
            "Shaltaiboltai is planning work for {worker_count} read-only agents…"
        )));

        let config = self.config.clone();
        let tx = self.tx.clone();
        self.orchestration_task = Some(tokio::spawn(async move {
            let result = orchestration::collect_planner_request(config, request)
                .await
                .map_err(|error| error.to_string())
                .and_then(|text| {
                    orchestration::parse_plan(&text, &worker_models, worker_count)
                        .map_err(|error| error.to_string())
                });
            let _ = tx.send(AppEvent::OrchestrationPlanned { run_id, result });
        }));
    }

    fn finish_orchestration_plan(&mut self, result: Result<Vec<PlannedTask>, String>) {
        let Some(run) = self.orchestration_run.as_mut() else {
            return;
        };
        let result = result.and_then(|tasks| {
            orchestration::validate_plan_assignments(&tasks, run.worker_count)?;
            Ok(tasks)
        });
        match result {
            Ok(tasks) => {
                run.tasks = tasks;
                run.phase = OrchestrationPhase::Confirming;
                self.mode = Mode::OrchestrationConfirm;
                self.orchestration_confirm_focused = false;
            }
            Err(error) => {
                let prompt = run.prompt.take();
                self.team_workers = Some(run.worker_count);
                self.orchestration_run = None;
                self.mode = Mode::Input;
                self.transcript
                    .push(Entry::Error(format!("team planning failed: {error}")));
                if let Some(prompt) = prompt {
                    self.restore_prompt(prompt, "team planning failed before any worker started");
                }
                self.apply_deferred_model_reconciliation();
            }
        }
    }

    pub fn confirm_orchestration(&mut self) {
        if self.mode != Mode::OrchestrationConfirm || !self.orchestration_confirm_focused {
            return;
        }
        let Some(mut run) = self.orchestration_run.take() else {
            self.mode = Mode::Input;
            return;
        };
        let Some(prompt) = run.prompt.take() else {
            self.mode = Mode::Input;
            return;
        };
        let task_text = prompt.text.clone();
        let history = self.history.clone();
        self.begin_root_prompt(prompt);

        run.entry_indices = run
            .tasks
            .iter()
            .map(|task| {
                let index = self.transcript.len();
                self.transcript.push(Entry::Agent {
                    name: format!("agent {} · {}", task.id, task.title),
                    model: format!(
                        "{} · {}",
                        task.model.display_id(),
                        task.model.provider.label()
                    ),
                    status: "RUNNING".into(),
                    summary: "reviewing the task in a read-only sandbox…".into(),
                    is_error: false,
                });
                index
            })
            .collect();
        run.phase = OrchestrationPhase::Workers;
        let run_id = run.id;
        let tasks = run.tasks.clone();
        self.orchestration_run = Some(run);
        self.mode = Mode::Orchestrating;
        self.orchestration_confirm_focused = false;

        let config = self.config.clone();
        let tx = self.tx.clone();
        self.orchestration_task = Some(tokio::spawn(async move {
            let mut workers = FuturesUnordered::new();
            for task in tasks {
                let config = config.clone();
                let request = orchestration::worker_request(&history, &task_text, &task);
                workers.push(async move {
                    let result = match request {
                        Ok(request) => orchestration::collect_worker_request(config, request)
                            .await
                            .map_err(|error| error.to_string()),
                        Err(error) => Err(error),
                    };
                    WorkerOutcome {
                        id: task.id,
                        title: task.title,
                        model: task.model,
                        result,
                    }
                });
            }
            while let Some(outcome) = workers.next().await {
                let _ = tx.send(AppEvent::OrchestrationWorkerFinished { run_id, outcome });
            }
            let _ = tx.send(AppEvent::OrchestrationWorkersFinished { run_id });
        }));
    }

    fn finish_orchestration_worker(&mut self, outcome: WorkerOutcome) {
        let Some(run) = self.orchestration_run.as_mut() else {
            return;
        };
        if !run.tasks.iter().any(|task| task.id == outcome.id)
            || run.outcomes.iter().any(|known| known.id == outcome.id)
        {
            return;
        }
        if let Some(index) = outcome
            .id
            .checked_sub(1)
            .and_then(|id| run.entry_indices.get(id))
            .copied()
        {
            if let Some(Entry::Agent {
                status,
                summary,
                is_error,
                ..
            }) = self.transcript.get_mut(index)
            {
                match &outcome.result {
                    Ok(report) => {
                        *status = "DONE".into();
                        *summary = orchestration::report_preview(report);
                    }
                    Err(error) => {
                        *status = "FAILED".into();
                        *summary = orchestration::report_preview(error);
                        *is_error = true;
                    }
                }
                self.transcript_dirty_from = Some(
                    self.transcript_dirty_from
                        .map_or(index, |earlier| earlier.min(index)),
                );
            }
        }
        run.outcomes.push(outcome);
    }

    fn finish_orchestration_workers(&mut self) {
        let (outcomes, task_count) = {
            let Some(run) = self.orchestration_run.as_mut() else {
                return;
            };
            run.outcomes.sort_by_key(|outcome| outcome.id);
            (run.outcomes.clone(), run.tasks.len())
        };
        let successful = outcomes
            .iter()
            .filter(|outcome| outcome.result.is_ok())
            .count();
        if successful == 0 {
            self.fail_orchestration_root(
                "all read-only workers failed; the coordinator was not started",
            );
            return;
        }

        let task = self
            .active_root_prompt
            .as_ref()
            .map(|active| active.prompt.text.as_str())
            .unwrap_or_default();
        let synthesis_budget = self
            .effective_compact_threshold_for(self.active_turn_model.as_ref())
            .saturating_sub(
                self.history_chars()
                    .saturating_add(ORCHESTRATION_SYSTEM_HEADROOM),
            )
            .min(orchestration::MAX_SYNTHESIS_CHARS);
        if synthesis_budget < MIN_SYNTHESIS_EVIDENCE_CHARS {
            self.fail_orchestration_root(
                "team evidence no longer fits the lead model context; compact and retry",
            );
            return;
        }
        self.orchestration_context = Some(orchestration::synthesis_context_with_limit(
            task,
            &outcomes,
            synthesis_budget,
        ));
        if let Some(run) = self.orchestration_run.as_mut() {
            run.phase = OrchestrationPhase::Coordinating;
        }
        self.transcript.push(Entry::Info(if successful == task_count {
            format!(
                "all {successful} agents reported — Shaltaiboltai is synthesizing and applying"
            )
        } else {
            format!(
                "{successful}/{} agents reported — Shaltaiboltai is continuing with partial evidence",
                task_count
            )
        }));
        self.mode = Mode::Input;
        self.start_request();
    }

    fn fail_orchestration_root(&mut self, message: &str) {
        // This path is used both before coordinator launch and for a
        // coordinator that failed before producing canonical output. Own and
        // fence every task here so callers cannot accidentally leave a late
        // event capable of reviving the abandoned root.
        self.gen += 1;
        if let Some(task) = self.request_task.take() {
            task.abort();
        }
        if let Some(task) = self.tool_task.take() {
            task.abort();
        }
        self.streaming_text.clear();
        self.pending_calls.clear();
        self.approval_preview = None;
        self.agent_turns = 0;
        self.approval_focused = true;
        let active = self.active_root_prompt.take();
        let failure_details: Vec<String> = self
            .orchestration_run
            .as_ref()
            .into_iter()
            .flat_map(|run| &run.outcomes)
            .filter_map(|outcome| {
                outcome.result.as_ref().err().map(|error| {
                    format!(
                        "agent {} ({} · {}) failed: {}",
                        outcome.id,
                        outcome.model.display_id(),
                        outcome.model.provider.label(),
                        orchestration::report_preview(error)
                    )
                })
            })
            .collect();
        let successor_in_composer = !self.input_is_empty()
            || !self.pending_images.is_empty()
            || self.restored_references.is_some()
            || self.suppressed_reference_text.is_some();
        let queued_successor = self.queued_prompt.is_some();

        // Workers are advisory and have not produced canonical assistant/tool
        // output. Always remove the abandoned root from provider history,
        // even when a successor draft or queued message must be preserved.
        if let Some(active) = active {
            self.history.truncate(active.history_len);
            self.history_chars_cache.set(None);
            self.transcript.truncate(active.transcript_len);
            self.transcript_rev += 1;
            if !queued_successor && !successor_in_composer {
                self.restore_prompt(active.prompt, message);
            }
        }
        if queued_successor {
            self.restore_queued_prompt(message);
        } else if successor_in_composer {
            self.composer_notice = Some(format!("team run stopped — draft preserved; {message}"));
        }
        self.transcript.push(Entry::Error(message.into()));
        self.transcript
            .extend(failure_details.into_iter().map(Entry::Error));
        self.orchestration_context = None;
        self.orchestration_run = None;
        self.mode = Mode::Input;
        self.active_turn_model = None;
        self.active_turn_can_promote_queue = true;
        self.apply_deferred_model_reconciliation();
    }

    fn team_coordinator_has_no_activity(&self) -> bool {
        self.orchestration_run
            .as_ref()
            .is_some_and(|run| run.phase == OrchestrationPhase::Coordinating)
            && self
                .active_root_prompt
                .as_ref()
                .is_some_and(|active| !active.observed_activity)
    }

    pub fn cancel_orchestration(&mut self) {
        let Some(mut run) = self.orchestration_run.take() else {
            return;
        };
        self.orchestration_gen += 1;
        if let Some(task) = self.orchestration_task.take() {
            task.abort();
        }
        self.orchestration_context = None;
        self.orchestration_confirm_focused = false;
        match run.phase {
            OrchestrationPhase::Planning | OrchestrationPhase::Confirming => {
                self.mode = Mode::Input;
                self.team_workers = Some(run.worker_count);
                if let Some(prompt) = run.prompt.take() {
                    self.restore_prompt(prompt, "team run cancelled before workers started");
                }
                self.transcript
                    .push(Entry::Info("team run cancelled".into()));
                self.apply_deferred_model_reconciliation();
            }
            OrchestrationPhase::Workers => {
                self.orchestration_run = Some(run);
                self.fail_orchestration_root("team run cancelled");
            }
            OrchestrationPhase::Coordinating => {
                self.orchestration_run = Some(run);
                self.cancel_request();
            }
        }
    }

    fn dispatch_prompt(&mut self, prompt: QueuedPrompt) {
        debug_assert_eq!(self.mode, Mode::Input);
        debug_assert!(!self.compacting);
        debug_assert!(self.request_task.is_none());
        debug_assert!(self.tool_task.is_none());
        debug_assert!(self.pending_calls.is_empty());

        self.begin_root_prompt(prompt);
        self.start_request();
    }

    fn begin_root_prompt(&mut self, prompt: QueuedPrompt) {
        debug_assert!(self.request_task.is_none());
        debug_assert!(self.tool_task.is_none());
        debug_assert!(self.pending_calls.is_empty());

        self.active_root_prompt = Some(ActiveRootPrompt {
            prompt: prompt.clone(),
            history_len: self.history.len(),
            transcript_len: self.transcript.len(),
            observed_activity: false,
        });
        let (text, model, images) = prompt.into_images();
        self.active_turn_model = Some(model);
        self.active_turn_can_promote_queue = true;
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
    }

    fn dispatch_queued_prompt(&mut self) {
        if self.mode != Mode::Input
            || self.compacting
            || self.request_task.is_some()
            || self.tool_task.is_some()
            || !self.pending_calls.is_empty()
        {
            return;
        }
        if let Some(prompt) = self.queued_prompt.take() {
            self.dispatch_prompt(prompt);
        }
    }

    fn restore_queued_prompt(&mut self, reason: &str) {
        let Some(prompt) = self.queued_prompt.take() else {
            return;
        };
        self.restore_prompt(prompt, &format!("next message restored — {reason}"));
    }

    fn restore_prompt(&mut self, prompt: QueuedPrompt, notice: &str) {
        let QueuedPrompt {
            text,
            staged_images,
            referenced_images,
            ..
        } = prompt;
        self.set_input(&text);
        self.pending_images = staged_images;
        // Preserve the exact bytes captured from referenced paths. If the user
        // edits the restored text, exact-text matching prevents their reuse.
        self.restored_references = Some(RestoredReferences {
            text,
            images: referenced_images,
        });
        self.suppressed_reference_text = None;
        self.composer_notice = Some(notice.to_owned());
    }

    pub fn paste(&mut self, text: &str) {
        if !self.composer_accepts_input() {
            return;
        }
        // Files dragged onto the terminal arrive as a paste of their paths:
        // stage them as attachments instead of cluttering the input.
        let dropped = images::dropped_images(text);
        if !dropped.is_empty() {
            for path in dropped {
                match images::load_image(&path) {
                    Ok((name, data)) => {
                        self.pending_images.push((name, data));
                        if self.mode == Mode::Input && !self.compacting {
                            self.transcript.push(Entry::Info(
                                "image staged from dropped file — Esc clears".into(),
                            ));
                        }
                    }
                    Err(e) => {
                        if self.mode == Mode::Input && !self.compacting {
                            self.transcript.push(Entry::Error(format!("{e:#}")));
                        } else {
                            self.composer_notice = Some(format!("{e:#}"));
                        }
                    }
                }
            }
            return;
        }
        self.textarea
            .insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
        self.note_input_changed();
    }

    // ---- image attachments ----

    /// Ctrl+V: stage an image from the system clipboard for the next message.
    pub fn attach_clipboard_image(&mut self) {
        if !self.composer_accepts_input() {
            return;
        }
        match images::clipboard_image() {
            Ok(image) => {
                let name = format!("clipboard-{}.png", self.pending_images.len() + 1);
                self.pending_images.push((name, image));
                if self.mode == Mode::Input && !self.compacting {
                    self.transcript.push(Entry::Info(format!(
                        "image staged from clipboard ({} attached) — Esc clears",
                        self.pending_images.len()
                    )));
                }
            }
            Err(e) => {
                if self.mode == Mode::Input && !self.compacting {
                    self.transcript.push(Entry::Info(format!("{e:#}")));
                } else {
                    self.composer_notice = Some(format!("{e:#}"));
                }
            }
        }
    }

    pub fn clear_attachments(&mut self) {
        let text = self.textarea.lines().join("\n").trim().to_owned();
        let had_restored = self
            .restored_references
            .as_ref()
            .is_some_and(|restored| restored.text == text && !restored.images.is_empty());
        if !self.pending_images.is_empty() || had_restored {
            self.pending_images.clear();
            self.restored_references = None;
            if had_restored {
                self.suppressed_reference_text = Some(text);
            }
            if self.mode == Mode::Input && !self.compacting {
                self.transcript
                    .push(Entry::Info("attachments cleared".into()));
            }
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
        self.composer_notice = None;
        self.restored_references = None;
        self.suppressed_reference_text = None;
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
        self.composer_notice = None;
        self.restored_references = None;
        self.suppressed_reference_text = None;
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
            ("team", Some("off")) => {
                self.team_workers = None;
                self.transcript
                    .push(Entry::Info("team mode disabled".into()));
            }
            ("team", count) => {
                let workers = match count {
                    None => orchestration::DEFAULT_WORKERS,
                    Some(raw) => match raw.parse::<usize>() {
                        Ok(value)
                            if (orchestration::MIN_WORKERS..=orchestration::MAX_WORKERS)
                                .contains(&value) =>
                        {
                            value
                        }
                        _ => {
                            self.transcript.push(Entry::Error(format!(
                                "team size must be {}-{}, or /team off",
                                orchestration::MIN_WORKERS,
                                orchestration::MAX_WORKERS
                            )));
                            return;
                        }
                    },
                };
                self.team_workers = Some(workers);
                self.transcript.push(Entry::Info(format!(
                    "team mode armed: the next prompt starts one lead planning call, then asks before launching {workers} read-only workers"
                )));
            }
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
            ("refresh", _) => self.refresh_models(),
            ("quit", _) => self.request_quit(),
            ("help", _) => self.open_help(),
            _ => unreachable!("registry and dispatch are matched"),
        }
    }

    pub fn open_help(&mut self) {
        self.mode = Mode::Help;
    }

    pub fn close_help(&mut self) {
        self.mode = Mode::Input;
        self.dispatch_queued_prompt();
    }

    /// `/model <name>`: select directly on a unique match, open the picker
    /// pre-filtered when ambiguous.
    fn select_model_by_filter(&mut self, filter: &str) {
        // Provider-qualified selectors are exact identities, including
        // unlisted model IDs. Never let substring matching silently turn one
        // explicit CLI model into another.
        if let Some(requested) = providers::cli_model_selector(filter) {
            if let Some(model) = providers::custom_cli_model(filter, &self.models) {
                if !self
                    .models
                    .iter()
                    .any(|known| known.id == model.id && known.provider == model.provider)
                {
                    self.models.push(model.clone());
                    self.discovery_models.push(model.clone());
                }
                self.transcript.push(Entry::Info(format!(
                    "model: {} ({})",
                    model.id,
                    model.provider.label()
                )));
                self.model = Some(model);
                self.model_selected_explicitly = true;
            } else {
                self.transcript.push(Entry::Error(format!(
                    "{} CLI is not available — sign in or run /refresh",
                    requested.provider.label()
                )));
            }
            return;
        }

        let needle = filter.to_lowercase();
        let matches: Vec<ModelEntry> = self
            .models
            .iter()
            .filter(|m| m.id.to_lowercase().contains(&needle))
            .cloned()
            .collect();
        let exact: Vec<&ModelEntry> = matches
            .iter()
            .filter(|model| model.id.to_lowercase() == needle)
            .collect();
        let picked = if exact.len() == 1 {
            exact.first().copied()
        } else if exact.is_empty() && matches.len() == 1 {
            matches.first()
        } else {
            None
        };
        if let Some(model) = picked {
            self.transcript.push(Entry::Info(format!(
                "model: {} ({})",
                model.id,
                model.provider.label()
            )));
            self.model = Some(model.clone());
            self.model_selected_explicitly = true;
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
        let Some(model) = self
            .active_turn_model
            .clone()
            .or_else(|| self.model.clone())
        else {
            self.mode = Mode::Input;
            self.restore_queued_prompt("the selected model is unavailable");
            return;
        };
        debug_assert!(self.request_task.is_none());
        self.mode = Mode::Streaming;
        self.streaming_text.clear();
        self.transcript.push(Entry::Assistant(String::new()));

        let request = self.active_chat_request(model);
        self.gen += 1;
        let gen = self.gen;
        let config = self.config.clone();
        let tx = self.tx.clone();
        self.request_task = Some(tokio::spawn(async move {
            let (chat_tx, chat_rx) = tokio::sync::mpsc::unbounded_channel();
            // Keep the provider future inside this task. Aborting the task now
            // drops HTTP streams and kill-on-drop CLI children as well as the
            // event forwarder, instead of detaching provider work.
            let stream = providers::stream_chat(config, request, chat_tx);
            forward_chat_stream(gen, stream, chat_rx, tx).await;
        }));
    }

    fn active_chat_request(&self, model: ModelEntry) -> ChatRequest {
        // Sub-agent providers run their own tool loop, so we don't send ours.
        let tools = if model.provider.is_sub_agent() {
            Vec::new()
        } else {
            tools::definitions()
        };
        let has_team_context = self.orchestration_context.is_some();
        let request = ChatRequest {
            model,
            system: match self.orchestration_context.as_deref() {
                Some(context) => format!("{}\n\n{context}", system_prompt()),
                None => system_prompt(),
            },
            messages: if has_team_context {
                orchestration::text_only_history(&self.history)
            } else {
                self.history.clone()
            },
            tools,
            policy: RequestPolicy::Interactive,
            force_full_handoff: has_team_context,
        };
        request
    }

    pub fn cancel_request(&mut self) {
        self.cancel_request_inner(true);
    }

    fn cancel_request_inner(&mut self, restore_queue: bool) {
        let rollback_team_root = self.team_coordinator_has_no_activity();
        // Invalidate in-flight work; late events from old generations are dropped.
        self.gen += 1;
        if let Some(task) = self.request_task.take() {
            task.abort();
        }
        if let Some(task) = self.tool_task.take() {
            task.abort();
        }
        if rollback_team_root {
            self.fail_orchestration_root("team synthesis cancelled before producing output");
            if !restore_queue {
                self.queued_prompt = None;
            }
            return;
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
            let changed = self.transcript.len();
            self.transcript_dirty_from = Some(
                self.transcript_dirty_from
                    .map_or(changed, |earlier| earlier.min(changed)),
            );
        }
        self.pending_calls.clear();
        self.approval_preview = None;
        self.repair_dangling_tool_calls();
        self.transcript.push(Entry::Info("cancelled".into()));
        self.agent_turns = 0;
        self.mode = Mode::Input;
        self.orchestration_run = None;
        self.orchestration_context = None;
        self.active_turn_model = None;
        self.active_turn_can_promote_queue = true;
        self.active_root_prompt = None;
        self.approval_focused = true;
        if restore_queue {
            self.restore_queued_prompt("the previous request was cancelled");
        } else {
            self.queued_prompt = None;
        }
        self.apply_deferred_model_reconciliation();
    }

    /// Quit through the same cancellation path as Esc so partial text is
    /// preserved and dangling tool calls are repaired before session save.
    pub fn request_quit(&mut self) {
        if matches!(self.mode, Mode::OrchestrationConfirm | Mode::Orchestrating) {
            self.cancel_orchestration();
            self.composer_notice = Some(
                "team work cancelled and draft preserved — press Ctrl+C again to discard it and quit"
                    .into(),
            );
            return;
        }
        if self.queued_prompt.is_some() {
            if matches!(
                self.mode,
                Mode::Streaming | Mode::RunningTool | Mode::Approval
            ) {
                self.cancel_request_inner(true);
            } else {
                self.cancel_compaction();
                self.mode = Mode::Input;
                self.restore_queued_prompt("quit was paused for confirmation");
            }
            self.composer_notice =
                Some("next message restored — press Ctrl+C again to discard it and quit".into());
            return;
        }
        if matches!(
            self.mode,
            Mode::Streaming | Mode::RunningTool | Mode::Approval
        ) {
            self.cancel_request_inner(false);
        }
        self.cancel_compaction();
        self.queued_prompt = None;
        self.should_quit = true;
    }

    fn cancel_compaction(&mut self) {
        self.compaction_gen += 1;
        if let Some(task) = self.compaction_task.take() {
            task.abort();
        }
        self.compacting = false;
    }

    fn abort_orchestration(&mut self) {
        self.orchestration_gen += 1;
        if let Some(task) = self.orchestration_task.take() {
            task.abort();
        }
        self.orchestration_run = None;
        self.orchestration_context = None;
        self.orchestration_confirm_focused = false;
    }

    pub fn cancel_compaction_request(&mut self) {
        if !self.compacting {
            return;
        }
        self.cancel_compaction();
        self.transcript
            .push(Entry::Info("context compaction cancelled".into()));
        self.restore_queued_prompt("context compaction was cancelled");
        self.apply_deferred_model_reconciliation();
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
                    || m.display_id().to_lowercase().contains(&needle)
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
            self.model_selected_explicitly = true;
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
        self.save_session_checked();
    }

    fn save_session_checked(&mut self) -> bool {
        if let Err(e) = self.persist_session() {
            // Mid-session persistence failures stay inside the restored TUI.
            self.transcript
                .push(Entry::Error(format!("failed to save session: {e:#}")));
            false
        } else {
            true
        }
    }

    /// Final shutdown save. Returning the failure lets `main` restore the
    /// terminal first and then report it on stderr instead of hiding it after
    /// the last frame was already drawn.
    pub fn save_session_for_exit(&self) -> anyhow::Result<()> {
        self.persist_session()
    }

    fn persist_session(&self) -> anyhow::Result<()> {
        if self.history.is_empty() {
            return Ok(());
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
        session::save(&snapshot)
    }

    fn reset_session(&mut self) {
        if let Err(e) = self.persist_session() {
            self.transcript.push(Entry::Error(format!(
                "could not start a new session because the current one was not saved: {e:#}"
            )));
            return;
        }
        self.cancel_compaction();
        self.abort_orchestration();
        self.gen += 1;
        self.session_id = session::new_id();
        self.history.clear();
        self.history_chars_cache.set(None);
        self.transcript.clear();
        self.transcript_rev += 1;
        self.agent_turns = 0;
        self.scroll_from_bottom = 0;
        self.last_usage = None;
        self.pending_calls.clear();
        self.queued_prompt = None;
        self.pending_images.clear();
        self.clear_input();
        self.approved_scopes.clear();
        self.approval_preview = None;
        self.approval_focused = true;
        self.active_turn_model = None;
        self.active_turn_can_promote_queue = true;
        self.active_root_prompt = None;
        self.team_workers = None;
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
                if let Err(e) = self.persist_session() {
                    self.transcript.push(Entry::Error(format!(
                        "could not resume another session because the current one was not saved: {e:#}"
                    )));
                    self.mode = Mode::Input;
                    return;
                }
                self.cancel_compaction();
                self.abort_orchestration();
                self.gen += 1;
                self.session_id = loaded.id;
                self.history = loaded.history;
                self.history_chars_cache.set(None);
                self.transcript = loaded.transcript;
                self.transcript_rev += 1;
                self.scroll_from_bottom = 0;
                self.agent_turns = 0;
                self.last_usage = None;
                self.pending_calls.clear();
                self.queued_prompt = None;
                self.pending_images.clear();
                self.clear_input();
                self.approved_scopes.clear();
                self.approval_preview = None;
                self.approval_focused = true;
                self.active_turn_model = None;
                self.active_turn_can_promote_queue = true;
                self.active_root_prompt = None;
                self.team_workers = None;
                if let Some(saved) = loaded.model {
                    let available = self
                        .models
                        .iter()
                        .find(|model| model.id == saved.id && model.provider == saved.provider)
                        .cloned()
                        .or_else(|| {
                            providers::custom_cli_model(&saved.id, &self.models)
                                .filter(|model| model.provider == saved.provider)
                        });
                    // Discovery is incremental, so a saved provider may not
                    // have published yet. Retain its exact provider+ID
                    // provisionally; final reconciliation validates or clears
                    // it instead of inheriting another session's provider.
                    let resolved = available
                        .clone()
                        .or_else(|| self.discovering.then(|| saved.clone()));
                    if let Some(model) = resolved {
                        if available.is_some()
                            && !self.models.iter().any(|known| {
                                known.id == model.id && known.provider == model.provider
                            })
                        {
                            self.models.push(model.clone());
                            self.discovery_models.push(model.clone());
                        }
                        self.model = Some(model);
                        self.model_selected_explicitly = true;
                    } else {
                        self.model = None;
                        self.model_selected_explicitly = false;
                        self.transcript.push(Entry::Error(format!(
                            "saved model {} is unavailable — choose another with Ctrl+P",
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
        if let Some((len, chars)) = self.history_chars_cache.get() {
            if len == self.history.len() {
                return chars;
            }
        }
        let chars = self
            .history
            .iter()
            .map(|m| match m {
                Message::User(c) => {
                    c.text().len()
                        + c.images()
                            .iter()
                            .map(|image| image.data.len())
                            .sum::<usize>()
                }
                Message::Assistant { text, tool_calls } => {
                    text.len()
                        + tool_calls
                            .iter()
                            .map(|c| c.name.len() + c.arguments.to_string().len())
                            .sum::<usize>()
                }
                Message::ToolResult { content, .. } => content.len(),
            })
            .sum();
        self.history_chars_cache
            .set(Some((self.history.len(), chars)));
        chars
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
        self.effective_compact_threshold_for(self.model.as_ref())
    }

    fn effective_compact_threshold_for(&self, model: Option<&ModelEntry>) -> usize {
        let configured = self.config.compact_threshold_chars;
        match model.map(|model| model.provider) {
            Some(ProviderKind::Ollama) => {
                configured.min(self.config.ollama_num_ctx.saturating_mul(3))
            }
            _ => configured,
        }
    }

    fn team_context_fit(&self, worker_count: usize, root_task_chars: usize) -> TeamContextFit {
        let Some(coordinator) = self.model.as_ref() else {
            return TeamContextFit::Impossible;
        };
        let Some(planner) = orchestration::choose_planner_model(coordinator, &self.models) else {
            return TeamContextFit::Impossible;
        };
        let worker_models =
            orchestration::choose_worker_models(coordinator, &self.models, worker_count);
        if worker_models.len() != worker_count {
            return TeamContextFit::Impossible;
        }
        let smallest_window = std::iter::once(coordinator)
            .chain(std::iter::once(&planner))
            .chain(worker_models.iter())
            .map(|model| self.effective_compact_threshold_for(Some(model)))
            .min()
            .unwrap_or_else(|| self.effective_compact_threshold());
        let reserved = orchestration::MAX_SYNTHESIS_CHARS + ORCHESTRATION_SYSTEM_HEADROOM;
        let Some(available_for_history_and_root) = smallest_window.checked_sub(reserved) else {
            return TeamContextFit::Impossible;
        };
        if root_task_chars > available_for_history_and_root {
            return TeamContextFit::Impossible;
        }
        if self.history_chars().saturating_add(root_task_chars) <= available_for_history_and_root {
            return TeamContextFit::Fits;
        }
        let already_compacted = matches!(
            self.history.as_slice(),
            [Message::User(content)]
                if content
                    .text()
                    .starts_with("Context summary of our conversation so far (earlier messages were compacted):")
        );
        if self.history.is_empty() || already_compacted {
            TeamContextFit::Impossible
        } else {
            TeamContextFit::NeedsCompaction
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
        self.compaction_gen += 1;
        let compaction_gen = self.compaction_gen;

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
            policy: RequestPolicy::ReadOnly,
            force_full_handoff: true,
        };
        let session_id = self.session_id.clone();
        let config = self.config.clone();
        let tx = self.tx.clone();
        self.compaction_task = Some(tokio::spawn(async move {
            let (chat_tx, mut chat_rx) = tokio::sync::mpsc::unbounded_channel();
            // Keep the provider future inside the owned task. Aborting this
            // handle on /new, /resume, or quit now drops HTTP streams and
            // kill-on-drop CLI children instead of leaving billed work behind.
            let stream = providers::stream_chat(config, request, chat_tx);
            let result = collect_compaction_stream(stream, &mut chat_rx).await;
            let _ = tx.send(AppEvent::CompactionDone {
                session_id,
                compaction_gen,
                result,
            });
        }));
    }

    fn finish_compaction(&mut self, result: Result<String, String>) {
        match result {
            Ok(summary) => {
                let before = self.history_chars();
                self.history = vec![Message::User(UserContent::Text(format!(
                    "Context summary of our conversation so far (earlier messages were compacted):\n\n{}",
                    summary.trim()
                )))];
                self.history_chars_cache.set(None);
                self.last_usage = None;
                self.transcript.push(Entry::Info(format!(
                    "context compacted: ~{}k → ~{}k chars",
                    before / 1000,
                    self.history_chars() / 1000
                )));
                if self.save_session_checked() {
                    if self.queued_prompt.is_none() {
                        self.apply_deferred_model_reconciliation();
                    }
                    self.dispatch_queued_prompt();
                } else {
                    self.restore_queued_prompt("the compacted session could not be saved");
                    self.apply_deferred_model_reconciliation();
                }
            }
            Err(e) => {
                self.transcript
                    .push(Entry::Error(format!("compaction failed: {e}")));
                self.restore_queued_prompt("context compaction failed");
                self.apply_deferred_model_reconciliation();
            }
        }
    }
}

async fn collect_compaction_stream<F>(
    stream: F,
    chat_rx: &mut tokio::sync::mpsc::UnboundedReceiver<ChatEvent>,
) -> Result<String, String>
where
    F: Future<Output = ()>,
{
    tokio::pin!(stream);
    let mut text = String::new();
    let mut terminal = None;

    loop {
        tokio::select! {
            _ = &mut stream => {
                while let Ok(event) = chat_rx.try_recv() {
                    stage_compaction_event(event, &mut text, &mut terminal);
                }
                return terminal.unwrap_or_else(|| {
                    Err("summary stream ended before normal completion".into())
                });
            }
            event = chat_rx.recv() => match event {
                Some(event) => stage_compaction_event(event, &mut text, &mut terminal),
                None => {
                    (&mut stream).await;
                    return terminal.unwrap_or_else(|| {
                        Err("summary stream ended before normal completion".into())
                    });
                }
            }
        }
    }
}

fn stage_compaction_event(
    event: ChatEvent,
    text: &mut String,
    terminal: &mut Option<Result<String, String>>,
) {
    // An error reported during provider cleanup wins over an earlier success.
    if terminal.as_ref().is_some_and(Result::is_err) {
        return;
    }
    match event {
        ChatEvent::TextDelta(delta) if terminal.is_none() => {
            text.push_str(&delta);
        }
        ChatEvent::Error(error) => *terminal = Some(Err(error)),
        ChatEvent::Completed {
            tool_calls,
            stop_reason,
            ..
        } if terminal.is_none() => {
            *terminal = Some(
                if clean_stop_reason(stop_reason.as_deref(), !tool_calls.is_empty()) {
                    finish_compaction_text(std::mem::take(text))
                } else {
                    let reason = stop_reason.unwrap_or_else(|| "unknown".into());
                    Err(format!(
                        "summary stopped before normal completion: {reason}"
                    ))
                },
            );
        }
        ChatEvent::TextDelta(_)
        | ChatEvent::Notice(_)
        | ChatEvent::ToolActivity { .. }
        | ChatEvent::Completed { .. } => {}
    }
}

fn finish_compaction_text(text: String) -> Result<String, String> {
    if text.trim().is_empty() {
        Err("empty summary".into())
    } else {
        Ok(text)
    }
}

/// Providers use different normal terminal labels. Unknown/refusal/filter
/// reasons are conservative failures for queue promotion: the next prompt is
/// restored for review instead of being sent automatically.
fn clean_stop_reason(reason: Option<&str>, has_tool_calls: bool) -> bool {
    match reason {
        Some("stop" | "end_turn") => !has_tool_calls,
        Some("tool_use" | "tool_calls" | "function_call") => has_tool_calls,
        None | Some(_) => false,
    }
}

async fn forward_chat_stream<F>(
    gen: u64,
    stream: F,
    mut chat_rx: tokio::sync::mpsc::UnboundedReceiver<ChatEvent>,
    tx: UnboundedSender<AppEvent>,
) where
    F: Future<Output = ()>,
{
    tokio::pin!(stream);
    let mut terminal = None;

    loop {
        tokio::select! {
            _ = &mut stream => break,
            event = chat_rx.recv() => match event {
                Some(event) => {
                    if !forward_or_hold_chat_event(gen, event, &tx, &mut terminal) {
                        return;
                    }
                }
                None => {
                    (&mut stream).await;
                    break;
                }
            }
        }
    }

    // Providers may emit their logical terminal marker before an HTTP body or
    // CLI process has fully closed. Hold that marker until the owned future is
    // done so an immediately queued turn cannot overlap cleanup/session state.
    while let Ok(event) = chat_rx.try_recv() {
        if !forward_or_hold_chat_event(gen, event, &tx, &mut terminal) {
            return;
        }
    }
    let event = terminal
        .unwrap_or_else(|| ChatEvent::Error("provider stream ended before completion".into()));
    let _ = tx.send(AppEvent::Chat { gen, event });
}

fn forward_or_hold_chat_event(
    gen: u64,
    event: ChatEvent,
    tx: &UnboundedSender<AppEvent>,
    terminal: &mut Option<ChatEvent>,
) -> bool {
    match event {
        ChatEvent::Error(_) => *terminal = Some(event),
        ChatEvent::Completed { .. } if !matches!(terminal.as_ref(), Some(ChatEvent::Error(_))) => {
            *terminal = Some(event);
        }
        ChatEvent::TextDelta(_) | ChatEvent::Notice(_) | ChatEvent::ToolActivity { .. }
            if terminal.is_none() =>
        {
            return tx.send(AppEvent::Chat { gen, event }).is_ok();
        }
        ChatEvent::Completed { .. }
        | ChatEvent::TextDelta(_)
        | ChatEvent::Notice(_)
        | ChatEvent::ToolActivity { .. } => {}
    }
    true
}

fn make_textarea(theme: &Theme) -> TextArea<'static> {
    // The block (border + surface background) is restyled every frame by
    // ui::draw_input, since it doubles as the focus indicator.
    let mut textarea = TextArea::default();
    textarea.set_style(Style::new().fg(theme.fg));
    textarea.set_cursor_line_style(Style::default());
    textarea.set_placeholder_text("Describe a change or ask a question…");
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

/// Resolve HEAD from both a normal `.git` directory and the `gitdir:` pointer
/// file used by linked worktrees and submodules.
fn read_git_branch(dot_git: &std::path::Path) -> Option<String> {
    let head = if dot_git.is_dir() {
        dot_git.join("HEAD")
    } else {
        let pointer = std::fs::read_to_string(dot_git).ok()?;
        let target = std::path::PathBuf::from(pointer.trim().strip_prefix("gitdir: ")?);
        let git_dir = if target.is_absolute() {
            target
        } else {
            dot_git
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(target)
        };
        git_dir.join("HEAD")
    };
    std::fs::read_to_string(head)
        .ok()
        .and_then(|head| parse_git_head(&head))
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

    static DATA_DIR_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn offline_config() -> Config {
        Config {
            anthropic_api_key: None,
            openai_api_key: None,
            openai_base_url: "http://127.0.0.1:9".into(),
            ollama_host: "http://127.0.0.1:9".into(),
            default_model: None,
            compact_threshold_chars: 80_000,
            ollama_num_ctx: 16_384,
            theme: None,
            claude_code_bypass_permissions: false,
            codex_full_access: false,
        }
    }

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

    #[tokio::test]
    async fn team_command_arms_one_shot_and_validates_bounds() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);

        app.set_input("/team 4");
        app.submit_input();
        assert_eq!(app.team_workers(), Some(4));

        app.set_input("/team 9");
        app.submit_input();
        assert_eq!(app.team_workers(), Some(4));
        assert!(
            matches!(app.transcript.last(), Some(Entry::Error(error)) if error.contains("2-4"))
        );

        app.set_input("/team off");
        app.submit_input();
        assert_eq!(app.team_workers(), None);
    }

    #[tokio::test]
    async fn team_prompt_with_attachment_is_preserved_without_a_provider_call() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "test-lead".into(),
        });
        app.team_workers = Some(2);
        app.pending_images.push((
            "evidence.png".into(),
            ImageData {
                media_type: "image/png".into(),
                data: "aW1hZ2U=".into(),
            },
        ));
        app.set_input("review this screenshot");

        app.submit_input();

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.team_workers(), Some(2));
        assert_eq!(app.textarea.lines().join("\n"), "review this screenshot");
        assert_eq!(app.pending_images.len(), 1);
        assert!(app.orchestration_run.is_none());
    }

    #[tokio::test]
    async fn oversized_team_prompt_is_preserved_without_being_fanned_out() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "test-lead".into(),
        });
        app.team_workers = Some(2);
        let oversized = "x".repeat(orchestration::MAX_ROOT_TASK_CHARS + 1);
        app.set_input(&oversized);

        app.submit_input();

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.team_workers(), Some(2));
        assert_eq!(app.textarea.lines().join("\n"), oversized);
        assert!(app.orchestration_run.is_none());
        assert!(matches!(
            app.transcript.last(),
            Some(Entry::Error(error)) if error.contains("character limit")
        ));
    }

    #[tokio::test]
    async fn team_mode_rejects_unpinned_cli_defaults_without_consuming_the_draft() {
        for (provider, id) in [
            (ProviderKind::ClaudeCode, "claude-code"),
            (ProviderKind::Codex, "codex"),
        ] {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut app = App::new(offline_config(), tx);
            app.model = Some(ModelEntry {
                provider,
                id: id.into(),
            });
            app.team_workers = Some(2);
            app.set_input("keep this team prompt");

            app.submit_input();

            assert_eq!(app.mode, Mode::Input);
            assert_eq!(app.team_workers(), Some(2));
            assert_eq!(app.textarea.lines().join("\n"), "keep this team prompt");
            assert!(app.orchestration_run.is_none());
            assert!(matches!(
                app.transcript.last(),
                Some(Entry::Error(error)) if error.contains("explicit") && error.contains("pinned")
            ));
        }
    }

    #[tokio::test]
    async fn codex_team_lead_requires_and_uses_a_workspace_scoped_advisory_model() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let codex = ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:gpt-5.6-sol".into(),
        };
        app.model = Some(codex.clone());
        app.models = vec![codex.clone()];
        app.team_workers = Some(2);
        app.set_input("review this repository safely");

        app.submit_input();

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(
            app.textarea.lines().join("\n"),
            "review this repository safely"
        );
        assert!(app.orchestration_run.is_none());
        assert!(matches!(
            app.transcript.last(),
            Some(Entry::Error(error)) if error.contains("workspace-scoped advisory model")
        ));

        let safe = ModelEntry {
            provider: ProviderKind::OpenAi,
            id: "safe-planner".into(),
        };
        app.models.push(safe.clone());
        app.submit_input();

        assert_eq!(app.mode, Mode::Orchestrating);
        assert_eq!(
            app.orchestration_planner()
                .map(|model| (&model.provider, &model.id)),
            Some((&safe.provider, &safe.id))
        );
        assert!(app
            .orchestration_run
            .as_ref()
            .expect("team run")
            .tasks
            .is_empty());
        app.orchestration_task.take().unwrap().abort();
        app.cancel_orchestration();
    }

    #[tokio::test]
    async fn injected_codex_worker_plan_is_rejected_before_confirmation() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::Ollama,
            id: "safe-lead".into(),
        };
        let codex = ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:gpt-5.6-sol".into(),
        };
        app.model = Some(lead.clone());
        app.models = vec![lead.clone(), codex.clone()];
        app.start_orchestration_plan(
            QueuedPrompt {
                text: "preserve this root".into(),
                staged_images: Vec::new(),
                referenced_images: Vec::new(),
                model: lead,
            },
            2,
        );
        let run_id = app.orchestration_run_id().unwrap();
        app.orchestration_task.take().unwrap().abort();

        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![planned_task(1, &codex), planned_task(2, &codex)]),
        });

        assert_eq!(app.mode, Mode::Input);
        assert!(app.orchestration_run.is_none());
        assert_eq!(app.textarea.lines().join("\n"), "preserve this root");
        assert!(app.transcript.iter().any(|entry| {
            matches!(entry, Entry::Error(error) if error.contains("workspace-scoped"))
        }));
    }

    #[tokio::test]
    async fn impossible_team_context_never_starts_a_futile_compaction() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut config = offline_config();
        config.compact_threshold_chars =
            orchestration::MAX_SYNTHESIS_CHARS + ORCHESTRATION_SYSTEM_HEADROOM - 1;
        let mut app = App::new(config, tx);
        let lead = ModelEntry {
            provider: ProviderKind::OpenAi,
            id: "tiny-context".into(),
        };
        app.model = Some(lead.clone());
        app.models = vec![lead];
        app.team_workers = Some(2);
        app.set_input("small root");

        app.submit_input();

        assert_eq!(app.mode, Mode::Input);
        assert!(!app.compacting);
        assert!(app.compaction_task.is_none());
        assert_eq!(app.team_workers(), Some(2));
        assert_eq!(app.textarea.lines().join("\n"), "small root");
        assert!(matches!(
            app.transcript.last(),
            Some(Entry::Error(error)) if error.contains("cannot fit")
        ));
    }

    #[tokio::test]
    async fn pre_worker_cancel_restores_prompt_and_rearms_team_mode() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::Ollama,
            id: "lead".into(),
        };
        app.start_orchestration_plan(
            QueuedPrompt {
                text: "revise and retry".into(),
                staged_images: Vec::new(),
                referenced_images: Vec::new(),
                model: lead,
            },
            3,
        );
        app.cancel_orchestration();

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.team_workers(), Some(3));
        assert_eq!(app.textarea.lines().join("\n"), "revise and retry");
    }

    fn planned_task(id: usize, model: &ModelEntry) -> PlannedTask {
        PlannedTask {
            id,
            title: format!("review {id}"),
            instructions: format!("inspect lens {id}"),
            model: model.clone(),
        }
    }

    fn start_confirmed_test_team(app: &mut App, lead: &ModelEntry, text: &str) -> u64 {
        app.start_orchestration_plan(
            QueuedPrompt {
                text: text.into(),
                staged_images: Vec::new(),
                referenced_images: Vec::new(),
                model: lead.clone(),
            },
            2,
        );
        let run_id = app.orchestration_run_id().unwrap();
        app.orchestration_task.take().unwrap().abort();
        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![planned_task(1, lead), planned_task(2, lead)]),
        });
        app.focus_orchestration_confirm();
        app.confirm_orchestration();
        app.orchestration_task.take().unwrap().abort();
        run_id
    }

    #[tokio::test]
    async fn orchestrated_workers_feed_one_pinned_coordinator_turn() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::Ollama,
            id: "lead".into(),
        };
        app.model = Some(lead.clone());
        app.start_orchestration_plan(
            QueuedPrompt {
                text: "build the feature".into(),
                staged_images: Vec::new(),
                referenced_images: Vec::new(),
                model: lead.clone(),
            },
            2,
        );
        let run_id = app.orchestration_run_id().unwrap();
        app.orchestration_task.take().unwrap().abort();
        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![planned_task(1, &lead), planned_task(2, &lead)]),
        });
        assert_eq!(app.mode, Mode::OrchestrationConfirm);
        app.on_event(AppEvent::ModelsDiscovered {
            models: vec![ModelEntry {
                provider: ProviderKind::OpenAi,
                id: "late-discovery".into(),
            }],
            finished: true,
        });
        assert_eq!(
            app.model.as_ref().map(|model| (&model.provider, &model.id)),
            Some((&lead.provider, &lead.id)),
            "discovery must not replace the pinned lead while a team plan is active"
        );
        app.focus_orchestration_confirm();
        app.confirm_orchestration();
        app.orchestration_task.take().unwrap().abort();

        for id in [2, 1] {
            app.on_event(AppEvent::OrchestrationWorkerFinished {
                run_id,
                outcome: WorkerOutcome {
                    id,
                    title: format!("review {id}"),
                    model: lead.clone(),
                    result: Ok(format!("evidence from {id}")),
                },
            });
        }
        app.on_event(AppEvent::OrchestrationWorkersFinished { run_id });

        assert_eq!(app.mode, Mode::Streaming);
        assert_eq!(
            app.active_turn_model.as_ref().map(|model| &model.id),
            Some(&lead.id)
        );
        let context = app.orchestration_context.as_deref().unwrap();
        assert!(context.find("worker 1").unwrap() < context.find("worker 2").unwrap());
        assert_eq!(
            app.history
                .iter()
                .filter(|message| matches!(message, Message::User(_)))
                .count(),
            1,
            "worker prompts and reports must stay out of canonical history"
        );
        app.cancel_request();
    }

    #[tokio::test]
    async fn first_turn_cli_synthesis_forces_worker_evidence_into_the_handoff() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:test".into(),
        };
        app.history
            .push(Message::User("the first root task".into()));

        let ordinary = app.active_chat_request(lead.clone());
        assert!(!ordinary.force_full_handoff);

        app.orchestration_context = Some("UNTRUSTED WORKER EVIDENCE".into());
        let synthesis = app.active_chat_request(lead);
        assert!(synthesis.force_full_handoff);
        assert!(synthesis.system.contains("UNTRUSTED WORKER EVIDENCE"));
        assert_eq!(synthesis.policy, RequestPolicy::Interactive);
    }

    #[tokio::test]
    async fn team_synthesis_omits_historical_image_payloads() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::OpenAi,
            id: "lead".into(),
        };
        app.history.push(Message::User(UserContent::Rich {
            text: "historical screenshot".into(),
            images: vec![ImageData {
                media_type: "image/png".into(),
                data: "SECRET-BASE64-PAYLOAD".into(),
            }],
        }));
        app.history.push(Message::User("team root".into()));
        app.orchestration_context = Some("UNTRUSTED WORKER EVIDENCE".into());

        let synthesis = app.active_chat_request(lead);
        let serialized = serde_json::to_string(&synthesis.messages).unwrap();
        assert!(!serialized.contains("SECRET-BASE64-PAYLOAD"));
        assert!(serialized.contains("historical image omitted"));
        assert!(serialized.contains("team root"));
    }

    #[tokio::test]
    async fn team_context_budget_uses_the_smallest_participant_and_image_bytes() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::OpenAi,
            id: "cloud-lead".into(),
        };
        let local = ModelEntry {
            provider: ProviderKind::Ollama,
            id: "local-worker".into(),
        };
        app.model = Some(lead.clone());
        app.models = vec![lead.clone(), local];
        app.history.push(Message::User("x".repeat(18_000).into()));

        assert_eq!(
            app.team_context_fit(2, 0),
            TeamContextFit::NeedsCompaction,
            "an Ollama worker's smaller context must gate fan-out"
        );

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut root_heavy = App::new(offline_config(), tx);
        root_heavy.model = Some(lead.clone());
        root_heavy.models = vec![
            lead.clone(),
            ModelEntry {
                provider: ProviderKind::Ollama,
                id: "local-worker".into(),
            },
        ];
        root_heavy
            .history
            .push(Message::User("x".repeat(2_000).into()));
        assert_eq!(
            root_heavy.team_context_fit(2, 16_000),
            TeamContextFit::NeedsCompaction,
            "the root prompt itself must count before fan-out"
        );

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut images = App::new(offline_config(), tx);
        images.model = Some(lead.clone());
        images.models = vec![lead];
        images.history.push(Message::User(UserContent::Rich {
            text: "historical screenshot".into(),
            images: vec![ImageData {
                media_type: "image/png".into(),
                data: "a".repeat(50_000),
            }],
        }));

        assert!(images.history_chars() >= 50_000);
        assert_eq!(
            images.team_context_fit(2, 0),
            TeamContextFit::NeedsCompaction,
            "base64 image bytes must count toward the team context budget"
        );
    }

    #[tokio::test]
    async fn all_worker_failures_restore_the_root_and_fence_late_events() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        let lead = ModelEntry {
            provider: ProviderKind::Ollama,
            id: "lead".into(),
        };
        app.model = Some(lead.clone());
        app.start_orchestration_plan(
            QueuedPrompt {
                text: "do not lose me".into(),
                staged_images: Vec::new(),
                referenced_images: Vec::new(),
                model: lead.clone(),
            },
            2,
        );
        let run_id = app.orchestration_run_id().unwrap();
        app.orchestration_task.take().unwrap().abort();
        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![planned_task(1, &lead), planned_task(2, &lead)]),
        });
        app.focus_orchestration_confirm();
        app.confirm_orchestration();
        app.orchestration_task.take().unwrap().abort();
        for id in [1, 2] {
            app.on_event(AppEvent::OrchestrationWorkerFinished {
                run_id,
                outcome: WorkerOutcome {
                    id,
                    title: format!("review {id}"),
                    model: lead.clone(),
                    result: Err("offline".into()),
                },
            });
        }
        app.on_event(AppEvent::OrchestrationWorkersFinished { run_id });

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.textarea.lines().join("\n"), "do not lose me");
        assert!(app.history.is_empty());
        assert!(app.orchestration_run.is_none());

        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![planned_task(1, &lead), planned_task(2, &lead)]),
        });
        assert_eq!(app.mode, Mode::Input);
        assert!(app.orchestration_run.is_none());
    }

    #[tokio::test]
    async fn cancelled_or_failed_team_root_never_leaks_behind_a_successor() {
        for (queue_successor, all_workers_fail) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut app = App::new(offline_config(), tx);
            let lead = ModelEntry {
                provider: ProviderKind::Ollama,
                id: "lead".into(),
            };
            app.model = Some(lead.clone());
            let run_id = start_confirmed_test_team(&mut app, &lead, "cancel this root");
            app.pending_images.push((
                "successor.png".into(),
                ImageData {
                    media_type: "image/png".into(),
                    data: "c3VjY2Vzc29y".into(),
                },
            ));
            app.textarea.insert_str("preserve the successor");
            if queue_successor {
                app.queue_input();
            }

            if all_workers_fail {
                for id in [1, 2] {
                    app.on_event(AppEvent::OrchestrationWorkerFinished {
                        run_id,
                        outcome: WorkerOutcome {
                            id,
                            title: format!("review {id}"),
                            model: lead.clone(),
                            result: Err(format!("worker {id} unavailable")),
                        },
                    });
                }
                app.on_event(AppEvent::OrchestrationWorkersFinished { run_id });
            } else {
                app.cancel_orchestration();
            }

            assert_eq!(app.mode, Mode::Input);
            assert!(app.history.is_empty(), "cancelled root leaked into history");
            assert_eq!(app.textarea.lines().join("\n"), "preserve the successor");
            assert_eq!(app.pending_images.len(), 1);
            assert_eq!(app.pending_images[0].0, "successor.png");
            assert!(!app
                .transcript
                .iter()
                .any(|entry| matches!(entry, Entry::User(text) if text == "cancel this root")));
            if all_workers_fail {
                assert!(app.transcript.iter().any(|entry| {
                    matches!(entry, Entry::Error(error) if error.contains("worker 1 unavailable"))
                }));
            }
        }
    }

    #[tokio::test]
    async fn coordinator_pre_output_cancel_or_error_removes_root_behind_successor() {
        for (queue_successor, provider_error) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut app = App::new(offline_config(), tx);
            let lead = ModelEntry {
                provider: ProviderKind::Ollama,
                id: "lead".into(),
            };
            app.model = Some(lead.clone());
            let run_id = start_confirmed_test_team(&mut app, &lead, "abandon this root");
            app.pending_images.push((
                "next.png".into(),
                ImageData {
                    media_type: "image/png".into(),
                    data: "bmV4dA==".into(),
                },
            ));
            app.textarea.insert_str("preserve this successor");
            if queue_successor {
                app.queue_input();
            }

            for id in [1, 2] {
                app.on_event(AppEvent::OrchestrationWorkerFinished {
                    run_id,
                    outcome: WorkerOutcome {
                        id,
                        title: format!("review {id}"),
                        model: lead.clone(),
                        result: Ok(format!("evidence {id}")),
                    },
                });
            }
            app.on_event(AppEvent::OrchestrationWorkersFinished { run_id });
            assert_eq!(app.mode, Mode::Streaming);
            assert!(app.team_coordinator_has_no_activity());

            if provider_error {
                app.on_chat_event(ChatEvent::Error("lead unavailable".into()));
            } else {
                app.cancel_request();
            }

            assert_eq!(app.mode, Mode::Input);
            assert!(app.history.is_empty(), "abandoned team root leaked");
            assert_eq!(app.textarea.lines().join("\n"), "preserve this successor");
            assert_eq!(app.pending_images.len(), 1);
            assert_eq!(app.pending_images[0].0, "next.png");
            assert!(!app
                .transcript
                .iter()
                .any(|entry| matches!(entry, Entry::User(text) if text == "abandon this root")));
            assert!(app.orchestration_run.is_none());
            assert!(app.request_task.is_none());
        }
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
    fn linked_worktree_gitdir_resolves_branch() {
        let root = std::env::temp_dir().join(format!(
            "shaltai-gitdir-{}-{}",
            std::process::id(),
            session::now_secs()
        ));
        let git_dir = root.join("actual-git-dir");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(
            git_dir.join("HEAD"),
            "ref: refs/heads/feat/interface-refresh\n",
        )
        .unwrap();
        std::fs::write(root.join(".git"), "gitdir: actual-git-dir\n").unwrap();

        assert_eq!(
            read_git_branch(&root.join(".git")),
            Some("feat/interface-refresh".into())
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn aborting_forwarder_drops_the_provider_future() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (chat_tx, chat_rx) = tokio::sync::mpsc::unbounded_channel();
        let (app_tx, _app_rx) = tokio::sync::mpsc::unbounded_channel();
        let flag = dropped.clone();
        let provider = async move {
            let _guard = DropFlag(flag);
            let _keep_channel_open = chat_tx;
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        };
        let task = tokio::spawn(forward_chat_stream(0, provider, chat_rx, app_tx));
        started_rx.await.unwrap();
        task.abort();
        let _ = task.await;

        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn forwarder_holds_completion_until_provider_cleanup_finishes() {
        let (chat_tx, chat_rx) = tokio::sync::mpsc::unbounded_channel();
        let (app_tx, mut app_rx) = tokio::sync::mpsc::unbounded_channel();
        let (terminal_sent_tx, terminal_sent_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let provider = async move {
            let _ = chat_tx.send(ChatEvent::Completed {
                tool_calls: Vec::new(),
                stop_reason: Some("stop".into()),
                usage: None,
            });
            let _ = terminal_sent_tx.send(());
            let _ = release_rx.await;
        };
        let task = tokio::spawn(forward_chat_stream(7, provider, chat_rx, app_tx));

        terminal_sent_rx.await.unwrap();
        tokio::task::yield_now().await;
        assert!(app_rx.try_recv().is_err());

        release_tx.send(()).unwrap();
        task.await.unwrap();
        assert!(matches!(
            app_rx.recv().await,
            Some(AppEvent::Chat {
                gen: 7,
                event: ChatEvent::Completed { .. }
            })
        ));
    }

    #[tokio::test]
    async fn compaction_requires_a_clean_terminal_marker() {
        let (chat_tx, mut chat_rx) = tokio::sync::mpsc::unbounded_channel();
        let provider = async move {
            let _ = chat_tx.send(ChatEvent::TextDelta("partial summary".into()));
        };
        let result = collect_compaction_stream(provider, &mut chat_rx).await;
        assert!(matches!(result, Err(message) if message.contains("before normal completion")));

        let (chat_tx, mut chat_rx) = tokio::sync::mpsc::unbounded_channel();
        let provider = async move {
            let _ = chat_tx.send(ChatEvent::TextDelta("truncated summary".into()));
            let _ = chat_tx.send(ChatEvent::Completed {
                tool_calls: Vec::new(),
                stop_reason: Some("length".into()),
                usage: None,
            });
        };
        let result = collect_compaction_stream(provider, &mut chat_rx).await;
        assert!(matches!(result, Err(message) if message.contains("length")));
    }

    #[tokio::test]
    async fn cancelled_compaction_rejects_a_buffered_same_session_result() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.history.push(Message::User("current history".into()));
        app.compacting = true;
        let session_id = app.session_id.clone();
        let cancelled_gen = app.compaction_gen;

        app.cancel_compaction_request();
        app.on_event(AppEvent::CompactionDone {
            session_id,
            compaction_gen: cancelled_gen,
            result: Ok("stale summary".into()),
        });

        assert!(!app.compacting);
        assert!(matches!(
            app.history.as_slice(),
            [Message::User(content)] if content.text() == "current history"
        ));
    }

    #[tokio::test]
    async fn new_session_aborts_owned_compaction_work() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let _data_dir_guard = DATA_DIR_ENV_LOCK.lock().await;
        let previous_data_dir = std::env::var_os("SHALTAIBOLTAI_DATA_DIR");
        let data_dir =
            std::env::temp_dir().join(format!("shaltai-compaction-session-{}", session::new_id()));
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &data_dir);

        let dropped = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let flag = dropped.clone();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.compacting = true;
        app.compaction_task = Some(tokio::spawn(async move {
            let _guard = DropFlag(flag);
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        }));
        started_rx.await.unwrap();

        app.reset_session();
        tokio::task::yield_now().await;

        assert!(!app.compacting);
        assert!(app.compaction_task.is_none());
        assert!(dropped.load(Ordering::SeqCst));

        std::fs::remove_dir_all(&data_dir).ok();
        match previous_data_dir {
            Some(path) => std::env::set_var("SHALTAIBOLTAI_DATA_DIR", path),
            None => std::env::remove_var("SHALTAIBOLTAI_DATA_DIR"),
        }
    }

    #[tokio::test]
    async fn session_grants_end_on_new_and_resume() {
        let _data_dir_guard = DATA_DIR_ENV_LOCK.lock().await;
        let previous_data_dir = std::env::var_os("SHALTAIBOLTAI_DATA_DIR");
        let data_dir =
            std::env::temp_dir().join(format!("shaltai-approval-session-{}", session::new_id()));
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &data_dir);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.approved_scopes.insert("run_command\0cargo test".into());
        app.cancel_request();
        assert!(
            app.approved_scopes.contains("run_command\0cargo test"),
            "cancelling work must not revoke a conversation-level grant"
        );
        app.reset_session();
        assert!(app.approved_scopes.is_empty());

        let resumed_path = data_dir.join("resumed.json");
        std::fs::create_dir_all(&data_dir).unwrap();
        let resumed = session::Session {
            id: "resumed".into(),
            title: "resumed session".into(),
            updated_at: session::now_secs(),
            cwd: None,
            model: None,
            history: Vec::new(),
            transcript: Vec::new(),
        };
        std::fs::write(&resumed_path, serde_json::to_vec(&resumed).unwrap()).unwrap();
        app.sessions = vec![session::Meta {
            path: resumed_path,
            title: resumed.title,
            updated_at: resumed.updated_at,
            cwd: None,
        }];
        app.approved_scopes.insert("read_file\0/etc/passwd".into());
        app.pick_session();
        assert!(app.approved_scopes.is_empty());

        let blocked_data_root = data_dir.join("not-a-directory");
        std::fs::write(&blocked_data_root, "occupied").unwrap();
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &blocked_data_root);
        app.history.push(Message::User("must be saved".into()));
        app.approved_scopes.insert("run_command\0cargo test".into());
        app.reset_session();
        assert!(
            app.approved_scopes.contains("run_command\0cargo test"),
            "a blocked session transition must retain the current grants"
        );
        app.pick_session();
        assert!(
            app.approved_scopes.contains("run_command\0cargo test"),
            "a blocked resume must retain the current grants"
        );
        assert!(
            app.save_session_for_exit().is_err(),
            "shutdown must surface a persistence failure"
        );

        std::fs::remove_dir_all(&data_dir).ok();
        match previous_data_dir {
            Some(path) => std::env::set_var("SHALTAIBOLTAI_DATA_DIR", path),
            None => std::env::remove_var("SHALTAIBOLTAI_DATA_DIR"),
        }
    }

    #[tokio::test]
    async fn queued_prompt_waits_for_successful_compaction() {
        let _data_dir_guard = DATA_DIR_ENV_LOCK.lock().await;
        let previous_data_dir = std::env::var_os("SHALTAIBOLTAI_DATA_DIR");
        let data_dir =
            std::env::temp_dir().join(format!("shaltai-queued-compact-{}", session::new_id()));
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &data_dir);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "queue-test".into(),
        });
        app.history.push(Message::User("old context".into()));
        app.compacting = true;
        app.textarea.insert_str("after compaction");
        app.submit_input();
        assert_eq!(app.queued_prompt_count(), 1);
        assert_eq!(app.history.len(), 1);

        app.compacting = false;
        app.finish_compaction(Ok("compressed context".into()));
        assert_eq!(app.mode, Mode::Streaming);
        assert_eq!(app.queued_prompt_count(), 0);
        assert!(matches!(
            app.history.as_slice(),
            [Message::User(summary), Message::User(next)]
                if summary.text().contains("compressed context")
                    && next.text() == "after compaction"
        ));
        app.cancel_request();

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut failed = App::new(offline_config(), tx);
        failed.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "queue-test".into(),
        });
        failed.history.push(Message::User("old context".into()));
        failed.compacting = true;
        failed.textarea.insert_str("do not auto-send");
        failed.submit_input();
        failed.compacting = false;
        failed.finish_compaction(Err("offline".into()));
        assert_eq!(failed.mode, Mode::Input);
        assert_eq!(failed.queued_prompt_count(), 0);
        assert_eq!(failed.textarea.lines().join("\n"), "do not auto-send");
        assert_eq!(failed.history.len(), 1);

        std::fs::remove_dir_all(&data_dir).ok();
        match previous_data_dir {
            Some(path) => std::env::set_var("SHALTAIBOLTAI_DATA_DIR", path),
            None => std::env::remove_var("SHALTAIBOLTAI_DATA_DIR"),
        }
    }

    #[tokio::test]
    async fn save_failure_restores_queued_prompt_without_dispatching() {
        let _data_dir_guard = DATA_DIR_ENV_LOCK.lock().await;
        let previous_data_dir = std::env::var_os("SHALTAIBOLTAI_DATA_DIR");
        let data_dir =
            std::env::temp_dir().join(format!("shaltai-queued-save-{}", session::new_id()));
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &data_dir);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(offline_config(), tx);
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "queue-test".into(),
        });
        app.history.push(Message::User("completed request".into()));
        app.mode = Mode::Streaming;
        app.textarea.insert_str("must remain a draft");
        app.queue_input();

        let blocked_data_root = data_dir.join("not-a-directory");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(&blocked_data_root, "occupied").unwrap();
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &blocked_data_root);
        app.end_turn();

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.queued_prompt_count(), 0);
        assert_eq!(app.textarea.lines().join("\n"), "must remain a draft");
        assert!(app
            .transcript
            .iter()
            .any(|entry| matches!(entry, Entry::Error(text) if text.contains("failed to save"))));

        std::fs::remove_dir_all(&data_dir).ok();
        match previous_data_dir {
            Some(path) => std::env::set_var("SHALTAIBOLTAI_DATA_DIR", path),
            None => std::env::remove_var("SHALTAIBOLTAI_DATA_DIR"),
        }
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
