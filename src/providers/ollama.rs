use super::sse;
use super::{ChatEvent, ChatRequest, Config, Message, ToolCall, Usage};
use anyhow::Result;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

pub async fn stream_chat(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let mut body = json!({
        "model": req.model.id,
        "stream": true,
        "messages": to_wire_messages(&req.system, &req.messages),
        // Ollama's default context window is tiny (~4k); without this, long
        // agent sessions silently lose the system prompt and tools.
        "options": {"num_ctx": config.ollama_num_ctx},
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

    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", config.ollama_host);

    let mut response = sse::send_retrying(client.post(&url).json(&body)).await?;
    // Not every local model supports tool calling; degrade to plain chat
    // rather than failing the whole turn.
    if response.status() == reqwest::StatusCode::BAD_REQUEST {
        let text = response.text().await.unwrap_or_default();
        if text.contains("does not support tools") {
            body.as_object_mut().unwrap().remove("tools");
            response = sse::send_retrying(client.post(&url).json(&body)).await?;
        } else {
            anyhow::bail!("API error 400: {text}");
        }
    }
    let response = sse::check_status(response).await?;

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut call_seq = 0usize;
    let mut stop_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;

    sse::for_each_ndjson(response, |line| {
        let event: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        if let Some(err) = event["error"].as_str() {
            anyhow::bail!("ollama error: {err}");
        }
        let message = &event["message"];
        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
            }
        }
        if let Some(calls) = message["tool_calls"].as_array() {
            for call in calls {
                // Ollama does not assign call ids; synthesize stable ones so
                // results can be matched in our provider-agnostic history.
                call_seq += 1;
                tool_calls.push(ToolCall {
                    id: format!("ollama-call-{call_seq}"),
                    name: call["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: call["function"]["arguments"].clone(),
                });
            }
        }
        if event["done"].as_bool() == Some(true) {
            if let Some(reason) = event["done_reason"].as_str() {
                stop_reason = Some(reason.to_owned());
            }
            usage = Some(Usage {
                input_tokens: event["prompt_eval_count"].as_u64().unwrap_or(0),
                output_tokens: event["eval_count"].as_u64().unwrap_or(0),
            });
        }
        Ok(())
    })
    .await?;

    let _ = tx.send(ChatEvent::Completed {
        tool_calls,
        stop_reason,
        usage,
    });
    Ok(())
}

pub async fn list_models(config: &Config) -> Result<Vec<String>> {
    let response = reqwest::Client::new()
        .get(format!("{}/api/tags", config.ollama_host))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await?;
    let body: Value = sse::check_status(response).await?.json().await?;

    let mut ids: Vec<String> = body["models"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter(|m| {
            // Skip embedding-only models when capabilities are reported.
            m["capabilities"]
                .as_array()
                .is_none_or(|caps| caps.iter().any(|c| c == "completion"))
        })
        .filter_map(|m| m["name"].as_str().map(str::to_owned))
        .collect();
    ids.sort();
    Ok(ids)
}

fn to_wire_messages(system: &str, messages: &[Message]) -> Vec<Value> {
    let mut wire = vec![json!({"role": "system", "content": system})];
    for msg in messages {
        match msg {
            Message::User(content) => {
                let mut m = json!({"role": "user", "content": content.text()});
                if !content.images().is_empty() {
                    // Ollama takes raw base64 strings in a sibling field.
                    m["images"] = content
                        .images()
                        .iter()
                        .map(|i| Value::String(i.data.clone()))
                        .collect();
                }
                wire.push(m);
            }
            Message::Assistant { text, tool_calls } => {
                let mut m = json!({"role": "assistant", "content": text});
                if !tool_calls.is_empty() {
                    m["tool_calls"] = tool_calls
                        .iter()
                        .map(|tc| json!({"function": {"name": tc.name, "arguments": tc.arguments}}))
                        .collect();
                }
                wire.push(m);
            }
            Message::ToolResult { name, content, .. } => {
                wire.push(json!({
                    "role": "tool",
                    "tool_name": name,
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
    use crate::providers::{ImageData, UserContent};

    #[test]
    fn user_images_fill_the_native_images_field() {
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
        assert_eq!(wire[1]["content"], "what is this?");
        assert_eq!(wire[1]["images"][0], "QUFBQQ==");
    }

    #[test]
    fn plain_text_users_have_no_images_field() {
        let wire = to_wire_messages("sys", &[Message::User("hi".into())]);
        assert_eq!(wire[1]["content"], "hi");
        assert!(wire[1].get("images").is_none());
    }
}
