use crate::config::Config;
use crate::providers::{self, ChatEvent, ChatRequest, Message, ModelEntry, ToolCall};
use crate::tools;
use std::collections::VecDeque;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

/// Hard cap on consecutive model→tool→model rounds for a single user input,
/// so a confused model cannot loop forever.
const MAX_AGENT_TURNS: usize = 30;

/// Events delivered to the UI loop from background tasks.
pub enum AppEvent {
    Chat(ChatEvent),
    ModelsDiscovered(Vec<ModelEntry>),
    ToolFinished {
        call: ToolCall,
        content: String,
        is_error: bool,
    },
}

#[derive(PartialEq)]
pub enum Mode {
    Input,
    Streaming,
    RunningTool,
    Approval,
    ModelPicker,
}

/// What the transcript pane renders. Kept separate from the provider history
/// because the display needs entries (tool lines, errors) that are not part of
/// the conversation sent to the model.
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

    pub models: Vec<ModelEntry>,
    pub model: Option<ModelEntry>,
    pub picker_index: usize,
    pub picker_filter: String,

    pub transcript: Vec<Entry>,
    pub history: Vec<Message>,
    pub scroll_from_bottom: usize,

    pub input: String,
    pub input_cursor: usize,

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
            models: Vec::new(),
            model: None,
            picker_index: 0,
            picker_filter: String::new(),
            transcript: Vec::new(),
            history: Vec::new(),
            scroll_from_bottom: 0,
            input: String::new(),
            input_cursor: 0,
            streaming_text: String::new(),
            pending_calls: VecDeque::new(),
            auto_approve: false,
            agent_turns: 0,
            request_task: None,
            tx,
        };
        app.transcript.push(Entry::Info(
            "shaltaiboltai — Enter to send, Ctrl+P to pick a model, /help for commands".into(),
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
                    self.agent_turns = 0;
                    self.mode = Mode::Input;
                    return;
                }
                self.agent_turns += 1;
                if self.agent_turns > MAX_AGENT_TURNS {
                    self.transcript.push(Entry::Error(format!(
                        "stopped after {MAX_AGENT_TURNS} consecutive tool rounds"
                    )));
                    self.agent_turns = 0;
                    self.mode = Mode::Input;
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
        }
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
        let text = self.input.trim().to_owned();
        if text.is_empty() {
            return;
        }
        self.input.clear();
        self.input_cursor = 0;

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

    fn run_slash_command(&mut self, command: &str) {
        match command.trim() {
            "model" | "models" => self.open_picker(),
            "clear" => {
                self.history.clear();
                self.transcript.clear();
                self.transcript.push(Entry::Info("conversation cleared".into()));
            }
            "quit" | "exit" => self.should_quit = true,
            "help" => self.transcript.push(Entry::Info(
                "commands: /model /clear /quit — keys: Ctrl+P models, PgUp/PgDn scroll, Esc cancel, Ctrl+C quit"
                    .into(),
            )),
            other => self.transcript.push(Entry::Error(format!("unknown command: /{other}"))),
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
         Prefer small, verifiable steps and report what you did.",
        std::env::consts::OS,
    )
}
