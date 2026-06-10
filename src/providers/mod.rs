pub mod anthropic;
pub mod ollama;
pub mod openai;
mod sse;

use crate::config::Config;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Ollama,
}

impl ProviderKind {
    pub fn label(&self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenAi => "openai",
            ProviderKind::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub provider: ProviderKind,
    pub id: String,
}

/// A tool invocation requested by the model. `id` is provider-assigned where
/// available (Anthropic/OpenAI) and synthesized for Ollama.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Provider-agnostic conversation history. Each provider module converts this
/// into its own wire format.
#[derive(Debug, Clone)]
pub enum Message {
    User(String),
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

/// Events emitted by a streaming chat request. Text streams incrementally;
/// tool calls are accumulated by the provider and delivered complete.
#[derive(Debug)]
pub enum ChatEvent {
    TextDelta(String),
    Completed { tool_calls: Vec<ToolCall> },
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

    models
}
