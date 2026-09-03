use serde::Deserialize;
use std::fmt;
use std::path::PathBuf;

/// Resolved runtime configuration. File values are overridden by environment
/// variables so secrets never need to live on disk.
#[derive(Clone)]
pub struct Config {
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub openrouter_api_key: Option<String>,
    pub openrouter_base_url: String,
    pub ollama_host: String,
    pub default_model: Option<String>,
    /// Auto-compact the conversation once its estimated size (in characters)
    /// exceeds this. ~4 chars per token.
    pub compact_threshold_chars: usize,
    /// Context window requested from Ollama (its server-side default is ~4k
    /// regardless of what the model supports).
    pub ollama_num_ctx: usize,
    /// Initial theme name; a theme picked at runtime (/theme) takes precedence.
    pub theme: Option<String>,
    /// Freeze decorative motion while keeping state labels and progress text.
    pub reduced_motion: bool,
}

impl fmt::Debug for Config {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Config")
            .field(
                "anthropic_api_key",
                &self.anthropic_api_key.as_ref().map(|_| "[redacted]"),
            )
            .field(
                "openai_api_key",
                &self.openai_api_key.as_ref().map(|_| "[redacted]"),
            )
            .field("openai_base_url", &self.openai_base_url)
            .field(
                "openrouter_api_key",
                &self.openrouter_api_key.as_ref().map(|_| "[redacted]"),
            )
            .field("openrouter_base_url", &self.openrouter_base_url)
            .field("ollama_host", &self.ollama_host)
            .field("default_model", &self.default_model)
            .field("compact_threshold_chars", &self.compact_threshold_chars)
            .field("ollama_num_ctx", &self.ollama_num_ctx)
            .field("theme", &self.theme)
            .field("reduced_motion", &self.reduced_motion)
            .finish()
    }
}

pub const DEFAULT_COMPACT_THRESHOLD_CHARS: usize = 80_000;
pub const DEFAULT_OLLAMA_NUM_CTX: usize = 16_384;

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    anthropic_api_key: Option<String>,
    openai_api_key: Option<String>,
    openai_base_url: Option<String>,
    openrouter_api_key: Option<String>,
    openrouter_base_url: Option<String>,
    ollama_host: Option<String>,
    default_model: Option<String>,
    compact_threshold_chars: Option<usize>,
    ollama_num_ctx: Option<usize>,
    theme: Option<String>,
    reduced_motion: Option<bool>,
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
            openrouter_api_key: env("OPENROUTER_API_KEY").or(file.openrouter_api_key),
            openrouter_base_url: env("OPENROUTER_BASE_URL")
                .or(file.openrouter_base_url)
                .unwrap_or_else(|| "https://openrouter.ai/api/v1".into()),
            ollama_host: env("OLLAMA_HOST")
                .or(file.ollama_host)
                .unwrap_or_else(|| "http://localhost:11434".into()),
            default_model: file.default_model,
            compact_threshold_chars: file
                .compact_threshold_chars
                .unwrap_or(DEFAULT_COMPACT_THRESHOLD_CHARS),
            ollama_num_ctx: file.ollama_num_ctx.unwrap_or(DEFAULT_OLLAMA_NUM_CTX),
            theme: file.theme,
            reduced_motion: env("SHALTAIBOLTAI_REDUCED_MOTION")
                .as_deref()
                .and_then(parse_bool)
                .or(file.reduced_motion)
                .unwrap_or(false),
        }
    }

    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("shaltaiboltai").join("config.toml"))
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_bool, Config, DEFAULT_COMPACT_THRESHOLD_CHARS, DEFAULT_OLLAMA_NUM_CTX};

    #[test]
    fn reduced_motion_environment_values_are_explicit() {
        for enabled in ["1", "true", "TRUE", "yes", "on"] {
            assert_eq!(parse_bool(enabled), Some(true));
        }
        for disabled in ["0", "false", "FALSE", "no", "off"] {
            assert_eq!(parse_bool(disabled), Some(false));
        }
        assert_eq!(parse_bool("sometimes"), None);
    }

    #[test]
    fn debug_output_redacts_provider_secrets() {
        let config = Config {
            anthropic_api_key: Some("anthropic-secret".into()),
            openai_api_key: Some("openai-secret".into()),
            openai_base_url: "https://api.openai.com/v1".into(),
            openrouter_api_key: Some("openrouter-secret".into()),
            openrouter_base_url: "https://openrouter.ai/api/v1".into(),
            ollama_host: "http://localhost:11434".into(),
            default_model: None,
            compact_threshold_chars: DEFAULT_COMPACT_THRESHOLD_CHARS,
            ollama_num_ctx: DEFAULT_OLLAMA_NUM_CTX,
            theme: None,
            reduced_motion: false,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("[redacted]"));
        for secret in ["anthropic-secret", "openai-secret", "openrouter-secret"] {
            assert!(!debug.contains(secret));
        }
    }
}
