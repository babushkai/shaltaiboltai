pub mod anthropic;
pub mod cli_agent;
pub mod ollama;
pub mod openai;
mod sse;

use crate::config::Config;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Ollama,
    /// Claude Code CLI driven as a sub-agent — billed to the user's Claude
    /// subscription, not an API key.
    ClaudeCode,
}

impl ProviderKind {
    pub fn label(&self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenAi => "openai",
            ProviderKind::Ollama => "ollama",
            ProviderKind::ClaudeCode => "claude-code",
        }
    }

    /// Sub-agent providers run their own tool loop, so our tool definitions and
    /// approval flow don't apply to them.
    pub fn is_sub_agent(&self) -> bool {
        matches!(self, ProviderKind::ClaudeCode)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub provider: ProviderKind,
    pub id: String,
}

/// A tool invocation requested by the model. `id` is provider-assigned where
/// available (Anthropic/OpenAI) and synthesized for Ollama.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// A base64-encoded image attached to a user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub media_type: String,
    pub data: String,
}

/// User message content. Untagged so plain-text messages (de)serialize as a
/// bare string — sessions saved before image support still load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Rich {
        text: String,
        images: Vec<ImageData>,
    },
}

impl UserContent {
    pub fn text(&self) -> &str {
        match self {
            UserContent::Text(t) => t,
            UserContent::Rich { text, .. } => text,
        }
    }

    pub fn images(&self) -> &[ImageData] {
        match self {
            UserContent::Text(_) => &[],
            UserContent::Rich { images, .. } => images,
        }
    }
}

impl From<String> for UserContent {
    fn from(text: String) -> Self {
        UserContent::Text(text)
    }
}

impl From<&str> for UserContent {
    fn from(text: &str) -> Self {
        UserContent::Text(text.to_owned())
    }
}

/// Provider-agnostic conversation history. Each provider module converts this
/// into its own wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    User(UserContent),
    Assistant {
        text: String,
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        call_id: String,
        name: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: Value,
}

/// Token counts reported by the provider for one request. `input_tokens`
/// includes cache reads/writes, i.e. it reflects the full context size.
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Events emitted by a streaming chat request. Text streams incrementally;
/// tool calls are accumulated by the provider and delivered complete.
/// `stop_reason` is normalized across providers: `Some("length")` means the
/// response was truncated by the output-token limit.
#[derive(Debug)]
pub enum ChatEvent {
    TextDelta(String),
    /// A tool the model ran itself (sub-agent providers like Claude Code drive
    /// their own tool loop). Display-only — it is not executed by us and does
    /// not enter the approval flow.
    ToolActivity {
        summary: String,
        is_error: bool,
    },
    Completed {
        tool_calls: Vec<ToolCall>,
        stop_reason: Option<String>,
        usage: Option<Usage>,
    },
    Error(String),
}

pub struct ChatRequest {
    pub model: ModelEntry,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
}

pub async fn stream_chat(config: Config, req: ChatRequest, tx: UnboundedSender<ChatEvent>) {
    let result = match req.model.provider {
        ProviderKind::Anthropic => anthropic::stream_chat(&config, &req, &tx).await,
        ProviderKind::OpenAi => openai::stream_chat(&config, &req, &tx).await,
        ProviderKind::Ollama => ollama::stream_chat(&config, &req, &tx).await,
        ProviderKind::ClaudeCode => cli_agent::stream_chat_claude(&config, &req, &tx).await,
    };
    if let Err(e) = result {
        let _ = tx.send(ChatEvent::Error(format!("{e:#}")));
    }
}

/// Discover models from every configured provider. Failures are silent per
/// provider (e.g. Ollama not running) so the rest of the list still works.
pub async fn discover_models(config: Config) -> Vec<ModelEntry> {
    let mut models = Vec::new();

    if config.anthropic_api_key.is_some() {
        for id in [
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ] {
            models.push(ModelEntry {
                provider: ProviderKind::Anthropic,
                id: id.into(),
            });
        }
    }
    if config.openai_api_key.is_some() {
        match openai::list_models(&config).await {
            Ok(ids) => models.extend(ids.into_iter().map(|id| ModelEntry {
                provider: ProviderKind::OpenAi,
                id,
            })),
            Err(_) => {
                for id in ["gpt-5.4", "gpt-5.4-mini"] {
                    models.push(ModelEntry {
                        provider: ProviderKind::OpenAi,
                        id: id.into(),
                    });
                }
            }
        }
    }
    if let Ok(ids) = ollama::list_models(&config).await {
        models.extend(ids.into_iter().map(|id| ModelEntry {
            provider: ProviderKind::Ollama,
            id,
        }));
    }
    // Subscription-backed sub-agents, available when their CLI is installed
    // and logged in.
    if cli_agent::claude_available().await {
        models.push(ModelEntry {
            provider: ProviderKind::ClaudeCode,
            id: "claude-code".into(),
        });
    }

    models
}
