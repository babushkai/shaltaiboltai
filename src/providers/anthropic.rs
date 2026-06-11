use super::sse;
use super::{ChatEvent, ChatRequest, Config, Message, ToolCall, Usage};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const MAX_TOKENS: u32 = 32_000;

pub async fn stream_chat(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let key = config
        .anthropic_api_key
        .as_deref()
        .context("ANTHROPIC_API_KEY is not set")?;

    // Cache breakpoints: system, the tool definitions, and the tail of the
    // conversation. In an agent loop each request extends the previous one,
    // so nearly the whole prompt becomes a cache hit.
    let mut body = json!({
        "model": req.model.id,
        "max_tokens": MAX_TOKENS,
        "stream": true,
        "system": [{
            "type": "text",
            "text": req.system,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": to_wire_messages(&req.messages),
    });
    if !req.tools.is_empty() {
        let mut tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.schema,
                })
            })
            .collect();
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
        body["tools"] = Value::Array(tools);
    }

    let request = reqwest::Client::new()
        .post(API_URL)
        .header("x-api-key", key)
        .header("anthropic-version", API_VERSION)
        .json(&body);
    let response = sse::check_status(sse::send_retrying(request).await?).await?;

    // Streaming tool_use inputs arrive as JSON fragments; accumulate per block
    // index and parse once the block closes.
    let mut pending: Vec<(String, String, String)> = Vec::new(); // (id, name, json buffer)
    let mut block_index_to_pending: std::collections::HashMap<u64, usize> = Default::default();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut stop_reason: Option<String> = None;
    let mut usage = Usage::default();

    sse::for_each_data(response, |data| {
        let event: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        match event["type"].as_str().unwrap_or("") {
            "message_start" => {
                let u = &event["message"]["usage"];
                usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0)
                    + u["cache_read_input_tokens"].as_u64().unwrap_or(0)
                    + u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let block = &event["content_block"];
                if block["type"] == "tool_use" {
                    let idx = event["index"].as_u64().unwrap_or(0);
                    block_index_to_pending.insert(idx, pending.len());
                    pending.push((
                        block["id"].as_str().unwrap_or_default().to_owned(),
                        block["name"].as_str().unwrap_or_default().to_owned(),
                        String::new(),
                    ));
                }
            }
            "content_block_delta" => {
                let delta = &event["delta"];
                if let Some(text) = delta["text"].as_str() {
                    let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
                } else if let Some(partial) = delta["partial_json"].as_str() {
                    let idx = event["index"].as_u64().unwrap_or(0);
                    if let Some(&p) = block_index_to_pending.get(&idx) {
                        pending[p].2.push_str(partial);
                    }
                }
            }
            "content_block_stop" => {
                let idx = event["index"].as_u64().unwrap_or(0);
                if let Some(p) = block_index_to_pending.remove(&idx) {
                    let (id, name, buf) = pending[p].clone();
                    let arguments = if buf.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&buf).unwrap_or(json!({}))
                    };
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
            }
            "message_delta" => {
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    // Normalized: truncation is reported as "length" everywhere.
                    stop_reason = Some(if reason == "max_tokens" {
                        "length".into()
                    } else {
                        reason.to_owned()
                    });
                }
                if let Some(out) = event["usage"]["output_tokens"].as_u64() {
                    usage.output_tokens = out;
                }
            }
            "error" => {
                anyhow::bail!("stream error: {}", event["error"]["message"]);
            }
            _ => {}
        }
        Ok(())
    })
    .await?;

    let _ = tx.send(ChatEvent::Completed {
        tool_calls,
        stop_reason,
        usage: Some(usage),
    });
    Ok(())
}

fn to_wire_messages(messages: &[Message]) -> Vec<Value> {
    let mut wire = Vec::new();
    for msg in messages {
        match msg {
            Message::User(content) => {
                let mut blocks = Vec::new();
                // Anthropic recommends images before the text that refers to them.
                for image in content.images() {
                    blocks.push(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.media_type,
                            "data": image.data,
                        },
                    }));
                }
                if !content.text().is_empty() || blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": content.text()}));
                }
                wire.push(json!({"role": "user", "content": blocks}));
            }
            Message::Assistant { text, tool_calls } => {
                let mut content = Vec::new();
                if !text.is_empty() {
                    content.push(json!({"type": "text", "text": text}));
                }
                for tc in tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                wire.push(json!({"role": "assistant", "content": content}));
            }
            Message::ToolResult {
                call_id,
                content,
                is_error,
                ..
            } => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                    "is_error": is_error,
                });
                // Anthropic requires tool results inside a user message;
                // consecutive results merge into one.
                match wire.last_mut() {
                    Some(last) if last["role"] == "user" && last["content"].is_array() => {
                        last["content"].as_array_mut().unwrap().push(block);
                    }
                    _ => wire.push(json!({"role": "user", "content": [block]})),
                }
            }
        }
    }
    // Third cache breakpoint on the conversation tail.
    if let Some(block) = wire
        .last_mut()
        .and_then(|m| m["content"].as_array_mut())
        .and_then(|c| c.last_mut())
    {
        block["cache_control"] = json!({"type": "ephemeral"});
    }
    wire
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{ImageData, UserContent};

    #[test]
    fn user_images_become_image_blocks_before_text() {
        let messages = vec![Message::User(UserContent::Rich {
            text: "what is this?".into(),
            images: vec![ImageData {
                media_type: "image/png".into(),
                data: "QUFBQQ==".into(),
            }],
        })];
        let wire = to_wire_messages(&messages);
        let blocks = wire[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "image");
        assert_eq!(blocks[0]["source"]["media_type"], "image/png");
        assert_eq!(blocks[0]["source"]["data"], "QUFBQQ==");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "what is this?");
    }

    #[test]
    fn plain_text_users_serialize_as_single_text_block() {
        let wire = to_wire_messages(&[Message::User("hi".into())]);
        let blocks = wire[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["text"], "hi");
    }
}
