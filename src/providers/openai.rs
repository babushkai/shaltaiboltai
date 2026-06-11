use super::sse;
use super::{ChatEvent, ChatRequest, Config, Message, ToolCall, Usage};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

pub async fn stream_chat(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let key = config
        .openai_api_key
        .as_deref()
        .context("OPENAI_API_KEY is not set")?;

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

    let request = reqwest::Client::new()
        .post(format!("{}/chat/completions", config.openai_base_url))
        .bearer_auth(key)
        .json(&body);
    let response = sse::check_status(sse::send_retrying(request).await?).await?;

    // Tool call name/arguments arrive as fragments keyed by index.
    let mut pending: Vec<(String, String, String)> = Vec::new(); // (id, name, args buffer)
    let mut stop_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;

    sse::for_each_data(response, |data| {
        if data == "[DONE]" {
            return Ok(());
        }
        let event: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        if let Some(u) = event["usage"].as_object() {
            usage = Some(Usage {
                input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
                output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
            });
        }
        let choice = &event["choices"][0];
        if let Some(reason) = choice["finish_reason"].as_str() {
            stop_reason = Some(reason.to_owned());
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
            }
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                let idx = call["index"].as_u64().unwrap_or(0) as usize;
                while pending.len() <= idx {
                    pending.push((String::new(), String::new(), String::new()));
                }
                if let Some(id) = call["id"].as_str() {
                    pending[idx].0.push_str(id);
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    pending[idx].1.push_str(name);
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    pending[idx].2.push_str(args);
                }
            }
        }
        Ok(())
    })
    .await?;

    let tool_calls = pending
        .into_iter()
        .filter(|(_, name, _)| !name.is_empty())
        .map(|(id, name, args)| ToolCall {
            id,
            name,
            arguments: serde_json::from_str(&args).unwrap_or(json!({})),
        })
        .collect();

    let _ = tx.send(ChatEvent::Completed {
        tool_calls,
        stop_reason,
        usage,
    });
    Ok(())
}

pub async fn list_models(config: &Config) -> Result<Vec<String>> {
    let key = config
        .openai_api_key
        .as_deref()
        .context("OPENAI_API_KEY is not set")?;

    let response = reqwest::Client::new()
        .get(format!("{}/models", config.openai_base_url))
        .bearer_auth(key)
        .send()
        .await?;
    let body: Value = sse::check_status(response).await?.json().await?;

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
            Message::User(text) => wire.push(json!({"role": "user", "content": text})),
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
