use super::sse;
use super::{ChatEvent, ChatRequest, Config, Message, ToolCall};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub async fn stream_chat(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let key = config
        .anthropic_api_key
        .as_deref()
        .context("ANTHROPIC_API_KEY is not set")?;

    let mut body = json!({
        "model": req.model.id,
        "max_tokens": 8192,
        "stream": true,
        "system": req.system,
        "messages": to_wire_messages(&req.messages),
        "tools": req.tools.iter().map(|t| json!({
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema,
        })).collect::<Vec<_>>(),
    });
    if req.tools.is_empty() {
        body.as_object_mut().unwrap().remove("tools");
    }

    let response = reqwest::Client::new()
        .post(API_URL)
        .header("x-api-key", key)
        .header("anthropic-version", API_VERSION)
        .json(&body)
        .send()
        .await?;
    let response = sse::check_status(response).await?;

    // Streaming tool_use inputs arrive as JSON fragments; accumulate per block
    // index and parse once the block closes.
    let mut pending: Vec<(String, String, String)> = Vec::new(); // (id, name, json buffer)
    let mut block_index_to_pending: std::collections::HashMap<u64, usize> = Default::default();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    sse::for_each_data(response, |data| {
        let event: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        match event["type"].as_str().unwrap_or("") {
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
            "error" => {
                anyhow::bail!("stream error: {}", event["error"]["message"]);
            }
            _ => {}
        }
        Ok(())
    })
    .await?;

    let _ = tx.send(ChatEvent::Completed { tool_calls });
    Ok(())
}

fn to_wire_messages(messages: &[Message]) -> Vec<Value> {
    let mut wire = Vec::new();
    for msg in messages {
        match msg {
            Message::User(text) => wire.push(json!({"role": "user", "content": text})),
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
    wire
}
