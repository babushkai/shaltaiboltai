use super::sse;
use super::{ChatEvent, ChatRequest, Config, Message, ToolCall, Usage};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

const MAX_STREAM_TOOL_CALLS: usize = 64;
const MAX_TOOL_CALL_ID_BYTES: usize = 1_024;
const MAX_TOOL_NAME_BYTES: usize = 256;
const MAX_TOOL_ARGUMENT_BYTES: usize = 256 * 1_024;
const MAX_PENDING_TOOL_BYTES: usize = 1_024 * 1_024;

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct CompatibleStreamState {
    pending: Vec<PendingToolCall>,
    pending_tool_bytes: usize,
    stop_reason: Option<String>,
    usage: Option<Usage>,
}

impl CompatibleStreamState {
    fn ingest(
        &mut self,
        provider: &str,
        data: &str,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        if data == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(data)
            .with_context(|| format!("{provider} returned malformed streaming JSON"))?;
        if let Some(error) = event.get("error").filter(|error| !error.is_null()) {
            let message = error["message"]
                .as_str()
                .or_else(|| error.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| error.to_string());
            anyhow::bail!("{provider} stream error: {message}");
        }
        if let Some(usage) = event["usage"].as_object() {
            self.usage = Some(Usage {
                input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
                output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
            });
        }
        let Some(choice) = event["choices"]
            .as_array()
            .and_then(|choices| choices.first())
        else {
            return Ok(());
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.stop_reason = Some(reason.to_owned());
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|text| !text.is_empty()) {
            let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                append_tool_fragment(
                    provider,
                    &mut self.pending,
                    &mut self.pending_tool_bytes,
                    call,
                )?;
            }
        }
        Ok(())
    }

    fn finish(self, provider: &str) -> Result<(Vec<ToolCall>, String, Option<Usage>)> {
        let tool_calls = self
            .pending
            .into_iter()
            .map(|pending| {
                let PendingToolCall {
                    id,
                    name,
                    arguments,
                } = pending;
                if id.is_empty() || name.is_empty() {
                    anyhow::bail!("{provider} returned a tool call without an ID or function name");
                }
                let arguments = serde_json::from_str(&arguments).with_context(|| {
                    format!("{provider} returned malformed arguments for tool {name}")
                })?;
                Ok(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stop_reason = self
            .stop_reason
            .with_context(|| format!("{provider} stream ended before a finish reason"))?;
        Ok((tool_calls, stop_reason, self.usage))
    }
}

pub async fn stream_chat(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let key = config
        .openai_api_key
        .as_deref()
        .context("OPENAI_API_KEY is not set")?;

    stream_chat_compatible("OpenAI", key, &config.openai_base_url, None, req, tx).await
}

pub(crate) async fn stream_chat_compatible(
    provider: &str,
    api_key: &str,
    base_url: &str,
    attribution: Option<(&str, &str)>,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let body = request_body(req);

    let mut request = reqwest::Client::new()
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(&body);
    if let Some((referer, title)) = attribution {
        request = request
            .header("HTTP-Referer", referer)
            .header("X-OpenRouter-Title", title);
    }
    let response = sse::check_status(sse::send_retrying(request).await?).await?;

    // Tool call name/arguments arrive as fragments keyed by index.
    let mut state = CompatibleStreamState::default();
    sse::for_each_data(response, |data| state.ingest(provider, data, tx)).await?;
    let (tool_calls, stop_reason, usage) = state.finish(provider)?;

    let _ = tx.send(ChatEvent::Completed {
        tool_calls,
        stop_reason: Some(stop_reason),
        usage,
    });
    Ok(())
}

fn append_tool_fragment(
    provider: &str,
    pending: &mut Vec<PendingToolCall>,
    pending_tool_bytes: &mut usize,
    call: &Value,
) -> Result<()> {
    let index = call["index"]
        .as_u64()
        .with_context(|| format!("{provider} streamed a tool call without an index"))?;
    let index = usize::try_from(index).with_context(|| {
        format!("{provider} streamed a tool call index that does not fit usize")
    })?;
    if index >= MAX_STREAM_TOOL_CALLS {
        anyhow::bail!(
            "{provider} returned tool call index {index}, above the {}-call limit",
            MAX_STREAM_TOOL_CALLS
        );
    }
    while pending.len() <= index {
        pending.push(PendingToolCall::default());
    }

    let PendingToolCall {
        id,
        name,
        arguments,
    } = &mut pending[index];
    let fragments = [
        (id, call["id"].as_str(), MAX_TOOL_CALL_ID_BYTES, "ID"),
        (
            name,
            call["function"]["name"].as_str(),
            MAX_TOOL_NAME_BYTES,
            "function name",
        ),
        (
            arguments,
            call["function"]["arguments"].as_str(),
            MAX_TOOL_ARGUMENT_BYTES,
            "arguments",
        ),
    ];
    for (buffer, fragment, per_field_limit, field) in fragments {
        let Some(fragment) = fragment else {
            continue;
        };
        let field_bytes = buffer
            .len()
            .checked_add(fragment.len())
            .context("streamed tool call size overflow")?;
        let total_bytes = pending_tool_bytes
            .checked_add(fragment.len())
            .context("streamed tool call size overflow")?;
        if field_bytes > per_field_limit || total_bytes > MAX_PENDING_TOOL_BYTES {
            anyhow::bail!("{provider} streamed oversized tool call {field}");
        }
        buffer.push_str(fragment);
        *pending_tool_bytes = total_bytes;
    }
    Ok(())
}

fn request_body(req: &ChatRequest) -> Value {
    let mut body = json!({
        "model": req.model.id,
        "stream": true,
        "stream_options": {"include_usage": true},
        "messages": to_wire_messages(&req.system, &req.messages),
        "tools": req.tools.iter().map(|t| json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        })).collect::<Vec<_>>(),
    });
    if req.tools.is_empty() {
        body.as_object_mut().unwrap().remove("tools");
    }
    body
}

pub async fn list_models(config: &Config) -> Result<Vec<String>> {
    let key = config
        .openai_api_key
        .as_deref()
        .context("OPENAI_API_KEY is not set")?;

    let response = reqwest::Client::new()
        .get(format!(
            "{}/models",
            config.openai_base_url.trim_end_matches('/')
        ))
        .bearer_auth(key);
    let response = sse::send_retrying(response).await?;
    let body: Value = sse::read_json_response(sse::check_status(response).await?).await?;

    let mut ids: Vec<String> = body["data"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .filter(|id| {
            // The /models endpoint also lists embeddings, TTS, etc.; keep chat models.
            (id.starts_with("gpt-") || id.starts_with('o'))
                && !id.contains("embed")
                && !id.contains("audio")
                && !id.contains("tts")
                && !id.contains("image")
                && !id.contains("realtime")
                && !id.contains("transcribe")
                && !id.contains("moderation")
        })
        .collect();
    ids.sort();
    Ok(ids)
}

fn to_wire_messages(system: &str, messages: &[Message]) -> Vec<Value> {
    let mut wire = vec![json!({"role": "system", "content": system})];
    for msg in messages {
        match msg {
            // Plain text stays a bare string for maximum compatibility with
            // OpenAI-compatible servers; images switch to the content-parts form.
            Message::User(content) if content.images().is_empty() => {
                wire.push(json!({"role": "user", "content": content.text()}))
            }
            Message::User(content) => {
                let mut parts = vec![json!({"type": "text", "text": content.text()})];
                for image in content.images() {
                    parts.push(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{};base64,{}", image.media_type, image.data),
                        },
                    }));
                }
                wire.push(json!({"role": "user", "content": parts}));
            }
            Message::Assistant { text, tool_calls } => {
                let mut m = json!({"role": "assistant", "content": text});
                if !tool_calls.is_empty() {
                    m["tool_calls"] = tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                },
                            })
                        })
                        .collect();
                }
                wire.push(m);
            }
            Message::ToolResult {
                call_id, content, ..
            } => {
                wire.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }));
            }
        }
    }
    wire
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{ExecutionPolicy, Workspace};
    use crate::providers::{
        ImageData, ModelEntry, ProviderKind, RequestPolicy, ToolDef, UserContent,
    };

    #[test]
    fn user_images_use_data_url_content_parts() {
        let wire = to_wire_messages(
            "sys",
            &[Message::User(UserContent::Rich {
                text: "what is this?".into(),
                images: vec![ImageData {
                    media_type: "image/png".into(),
                    data: "QUFBQQ==".into(),
                }],
            })],
        );
        let parts = wire[1]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "what is this?");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,QUFBQQ=="
        );
    }

    #[test]
    fn plain_text_users_stay_a_bare_string() {
        let wire = to_wire_messages("sys", &[Message::User("hi".into())]);
        assert_eq!(wire[1]["content"], "hi");
    }

    #[test]
    fn compatible_body_keeps_raw_openrouter_id_and_tool_contract() {
        let workspace = Workspace::new(env!("CARGO_MANIFEST_DIR")).unwrap();
        let request = ChatRequest {
            model: ModelEntry {
                provider: ProviderKind::OpenRouter,
                id: "anthropic/claude-sonnet-4.6".into(),
            },
            system: "system".into(),
            messages: vec![Message::User("inspect".into())],
            tools: vec![ToolDef {
                name: "read_file",
                description: "Read one file",
                schema: json!({"type": "object"}),
            }],
            execution_policy: ExecutionPolicy::new(workspace),
            policy: RequestPolicy::Interactive,
            force_full_handoff: false,
        };

        let body = request_body(&request);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.6");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn compatible_stream_parses_fragmented_tools_and_final_usage() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = CompatibleStreamState::default();
        state
            .ingest(
                "OpenRouter",
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","function":{"name":"read_","arguments":"{\"pa"}}]}}]}"#,
                &tx,
            )
            .unwrap();
        state
            .ingest(
                "OpenRouter",
                r#"{"choices":[{"finish_reason":"tool_calls","delta":{"tool_calls":[{"index":0,"id":"1","function":{"name":"file","arguments":"th\":\"README.md\"}"}}]}}],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#,
                &tx,
            )
            .unwrap();

        let (calls, reason, usage) = state.finish("OpenRouter").unwrap();
        assert_eq!(reason, "tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, json!({"path": "README.md"}));
        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn compatible_stream_rejects_malformed_json_and_error_envelopes() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = CompatibleStreamState::default();
        assert!(state
            .ingest("OpenRouter", "{not-json", &tx)
            .unwrap_err()
            .to_string()
            .contains("malformed streaming JSON"));
        assert!(state
            .ingest(
                "OpenRouter",
                r#"{"error":{"message":"upstream unavailable"}}"#,
                &tx,
            )
            .unwrap_err()
            .to_string()
            .contains("upstream unavailable"));
    }

    #[test]
    fn compatible_stream_bounds_tool_indices_and_fragment_buffers() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = CompatibleStreamState::default();
        let huge_index = json!({
            "choices": [{"delta": {"tool_calls": [{
                "index": MAX_STREAM_TOOL_CALLS,
                "id": "call",
                "function": {"name": "read_file", "arguments": "{}"}
            }]}}]
        });
        assert!(state
            .ingest("OpenRouter", &huge_index.to_string(), &tx)
            .unwrap_err()
            .to_string()
            .contains("above the"));

        let oversized = json!({
            "choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "id": "call",
                "function": {
                    "name": "read_file",
                    "arguments": "x".repeat(MAX_TOOL_ARGUMENT_BYTES + 1)
                }
            }]}}]
        });
        assert!(state
            .ingest("OpenRouter", &oversized.to_string(), &tx)
            .unwrap_err()
            .to_string()
            .contains("oversized tool call arguments"));
    }

    #[test]
    fn compatible_stream_rejects_malformed_tool_arguments_at_completion() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state = CompatibleStreamState::default();
        state
            .ingest(
                "OpenRouter",
                r#"{"choices":[{"finish_reason":"tool_calls","delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{"}}]}}]}"#,
                &tx,
            )
            .unwrap();
        assert!(state
            .finish("OpenRouter")
            .unwrap_err()
            .to_string()
            .contains("malformed arguments"));
    }
}
