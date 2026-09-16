use super::openai;
use super::sse;
use super::{ChatEvent, ChatRequest, Config};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashSet;
use tokio::sync::mpsc::UnboundedSender;

pub const AUTO_MODEL: &str = "openrouter/auto";
const MAX_CATALOG_MODELS: usize = 40;
const MAX_MODEL_ID_CHARS: usize = 256;
const APP_REFERER: &str = "https://github.com/babushkai/shaltaiboltai";
const APP_TITLE: &str = "Shaltaiboltai";

pub async fn stream_chat(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let key = config
        .openrouter_api_key
        .as_deref()
        .context("OPENROUTER_API_KEY is not set")?;

    openai::stream_chat_compatible(
        "OpenRouter",
        key,
        &config.openrouter_base_url,
        Some((APP_REFERER, APP_TITLE)),
        req,
        tx,
    )
    .await
}

/// Fetch a ranked, bounded catalog of text models that explicitly support
/// tools. The automatic router is inserted separately by the caller so it can
/// be shown immediately while this request is in flight.
pub async fn list_models(config: &Config) -> Result<Vec<String>> {
    let key = config
        .openrouter_api_key
        .as_deref()
        .context("OPENROUTER_API_KEY is not set")?;
    let request = reqwest::Client::new()
        .get(format!(
            "{}/models/user",
            config.openrouter_base_url.trim_end_matches('/')
        ))
        .bearer_auth(key)
        .header("HTTP-Referer", APP_REFERER)
        .header("X-OpenRouter-Title", APP_TITLE)
        .query(&[
            ("supported_parameters", "tools"),
            ("output_modalities", "text"),
            ("sort", "intelligence-high-to-low"),
            ("limit", "40"),
        ]);
    let response = sse::send_retrying(request).await?;
    let body: Value = sse::read_json_response(sse::check_status(response).await?).await?;
    parse_model_ids(&body)
}

fn parse_model_ids(body: &Value) -> Result<Vec<String>> {
    let models = body["data"]
        .as_array()
        .context("OpenRouter model catalog is missing its data array")?;
    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    for model in models {
        let Some(id) = model["id"].as_str().filter(|id| valid_model_id(id)) else {
            continue;
        };
        let supports_tools = model["supported_parameters"]
            .as_array()
            .is_some_and(|parameters| parameters.iter().any(|value| value == "tools"));
        let outputs_text = model["architecture"]["output_modalities"]
            .as_array()
            .is_some_and(|modalities| modalities.iter().any(|value| value == "text"));
        // Batch-only aliases are not suitable for an interactive streaming
        // picker even when their base model supports tools.
        if supports_tools
            && outputs_text
            && id != AUTO_MODEL
            && !id.ends_with(":batch")
            && seen.insert(id.to_owned())
        {
            ids.push(id.to_owned());
            if ids.len() == MAX_CATALOG_MODELS {
                break;
            }
        }
    }
    Ok(ids)
}

pub fn valid_model_id(id: &str) -> bool {
    !id.is_empty()
        && id.chars().count() <= MAX_MODEL_ID_CHARS
        && !id.chars().any(|ch| ch.is_control() || ch.is_whitespace())
}

/// OpenRouter-owned routers and `~...` rolling aliases can resolve to a
/// different concrete model between requests. They are useful for solo turns,
/// but cannot satisfy team mode's exact-model confirmation contract.
pub fn is_variable_model(id: &str) -> bool {
    id.starts_with("openrouter/") || id.starts_with('~')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn catalog_keeps_ranked_tool_capable_text_models_only() {
        let body = json!({"data": [
            {
                "id": "vendor/best",
                "supported_parameters": ["tools", "reasoning"],
                "architecture": {"output_modalities": ["text"]}
            },
            {
                "id": "vendor/best:batch",
                "supported_parameters": ["tools"],
                "architecture": {"output_modalities": ["text"]}
            },
            {
                "id": "vendor/no-tools",
                "supported_parameters": ["reasoning"],
                "architecture": {"output_modalities": ["text"]}
            },
            {
                "id": "vendor/image-only",
                "supported_parameters": ["tools"],
                "architecture": {"output_modalities": ["image"]}
            },
            {
                "id": "vendor/best",
                "supported_parameters": ["tools"],
                "architecture": {"output_modalities": ["text"]}
            },
            {
                "id": "vendor/second",
                "supported_parameters": ["tools"],
                "architecture": {"output_modalities": ["text", "image"]}
            }
        ]});

        assert_eq!(
            parse_model_ids(&body).unwrap(),
            vec!["vendor/best", "vendor/second"]
        );
    }

    #[test]
    fn malformed_catalog_and_model_ids_fail_closed() {
        assert!(parse_model_ids(&json!({"models": []})).is_err());
        assert!(!valid_model_id(""));
        assert!(!valid_model_id("vendor/model\nsecret"));
        assert!(!valid_model_id(&"x".repeat(MAX_MODEL_ID_CHARS + 1)));
        assert!(valid_model_id("anthropic/claude-sonnet-4.6"));
    }

    #[test]
    fn variable_routers_are_distinct_from_exact_model_ids() {
        for id in [AUTO_MODEL, "openrouter/free", "~openai/gpt-latest"] {
            assert!(is_variable_model(id), "{id}");
        }
        for id in [
            "openai/gpt-5.6-sol",
            "anthropic/claude-sonnet-5",
            "z-ai/glm-5.2:free",
        ] {
            assert!(!is_variable_model(id), "{id}");
        }
    }
}
