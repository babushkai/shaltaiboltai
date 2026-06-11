use serde::Deserialize;
use std::path::PathBuf;

/// Resolved runtime configuration. File values are overridden by environment
/// variables so secrets never need to live on disk.
#[derive(Debug, Clone)]
pub struct Config {
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub ollama_host: String,
    pub default_model: Option<String>,
    /// Auto-compact the conversation once its estimated size (in characters)
    /// exceeds this. ~4 chars per token.
    pub compact_threshold_chars: usize,
    /// Context window requested from Ollama (its server-side default is ~4k
    /// regardless of what the model supports).
    pub ollama_num_ctx: usize,
}

pub const DEFAULT_COMPACT_THRESHOLD_CHARS: usize = 80_000;
pub const DEFAULT_OLLAMA_NUM_CTX: usize = 16_384;

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    anthropic_api_key: Option<String>,
    openai_api_key: Option<String>,
    openai_base_url: Option<String>,
    ollama_host: Option<String>,
    default_model: Option<String>,
    compact_threshold_chars: Option<usize>,
    ollama_num_ctx: Option<usize>,
}

impl Config {
    pub fn load() -> Self {
        let file = Self::config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| toml::from_str::<FileConfig>(&s).ok())
            .unwrap_or_default();

        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());

        Config {
            anthropic_api_key: env("ANTHROPIC_API_KEY").or(file.anthropic_api_key),
            openai_api_key: env("OPENAI_API_KEY").or(file.openai_api_key),
            openai_base_url: env("OPENAI_BASE_URL")
                .or(file.openai_base_url)
                .unwrap_or_else(|| "https://api.openai.com/v1".into()),
            ollama_host: env("OLLAMA_HOST")
                .or(file.ollama_host)
                .unwrap_or_else(|| "http://localhost:11434".into()),
            default_model: file.default_model,
            compact_threshold_chars: file
                .compact_threshold_chars
                .unwrap_or(DEFAULT_COMPACT_THRESHOLD_CHARS),
            ollama_num_ctx: file.ollama_num_ctx.unwrap_or(DEFAULT_OLLAMA_NUM_CTX),
        }
    }

    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("shaltaiboltai").join("config.toml"))
    }
}
