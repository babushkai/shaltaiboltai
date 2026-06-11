//! Sub-agent providers backed by an official CLI (Claude Code) running on the
//! user's subscription. We never see or store a token: the CLI owns its own
//! auth. We spawn it headless, stream its NDJSON events, and adapt them into
//! our provider-agnostic [`ChatEvent`]s. The CLI runs its own tool loop, so our
//! tool definitions and approval flow do not apply here.

use super::{ChatEvent, ChatRequest, Config, Message, Usage};
use anyhow::{Context, Result};
use serde_json::Value;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

/// Whether the `claude` CLI is installed and responds. Cheap version probe with
/// a short timeout so discovery never hangs.
pub async fn claude_available() -> bool {
    let probe = tokio::process::Command::new("claude")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    matches!(
        tokio::time::timeout(std::time::Duration::from_secs(3), probe).await,
        Ok(Ok(status)) if status.success()
    )
}

pub async fn stream_chat_claude(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let Some(prompt) = last_user_text(&req.messages) else {
        anyhow::bail!("no user message to send");
    };
    if has_images(&req.messages) {
        let _ = tx.send(ChatEvent::ToolActivity {
            summary: "note: images are not yet forwarded to the Claude Code provider".into(),
            is_error: false,
        });
    }

    // First turn of a conversation starts fresh; later turns continue the CLI's
    // own session so it keeps full context. Counting our user messages also
    // makes /new (which clears history) naturally start a new CLI session.
    let continue_session = user_message_count(&req.messages) > 1;
    let permission_mode = if config.claude_code_bypass_permissions {
        "bypassPermissions"
    } else {
        "acceptEdits"
    };

    let mut cmd = tokio::process::Command::new("claude");
    cmd.arg("--print")
        .arg(&prompt)
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .arg("--permission-mode")
        .arg(permission_mode);
    if continue_session {
        cmd.arg("--continue");
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd.spawn().context(
        "failed to launch `claude` — install Claude Code and run `claude` once to sign in",
    )?;
    let stdout = child.stdout.take().context("no stdout from claude")?;
    let stderr = child.stderr.take().context("no stderr from claude")?;

    // Drain stderr concurrently so the pipe never blocks the child.
    let stderr_task = tokio::spawn(async move {
        let mut buf = String::new();
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            buf.push_str(&line);
            buf.push('\n');
        }
        buf
    });

    let mut lines = BufReader::new(stdout).lines();
    let mut saw_result = false;
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<Value>(&line) {
            if handle_event(&event, tx) {
                saw_result = true;
            }
        }
    }

    let status = child.wait().await?;
    let stderr = stderr_task.await.unwrap_or_default();
    if !saw_result {
        let detail = stderr.trim();
        if status.success() {
            anyhow::bail!("claude produced no result");
        } else if detail.is_empty() {
            anyhow::bail!("claude exited with {status}");
        } else {
            anyhow::bail!("claude error: {detail}");
        }
    }
    Ok(())
}

/// Translate one Claude Code stream-json event. Returns true when this was the
/// terminal `result` event (so the caller knows the turn completed cleanly).
fn handle_event(event: &Value, tx: &UnboundedSender<ChatEvent>) -> bool {
    match event["type"].as_str().unwrap_or("") {
        // Assistant turn: text blocks stream as deltas, tool_use blocks show as
        // activity. (The CLI executes the tools itself.)
        "assistant" => {
            if let Some(blocks) = event["message"]["content"].as_array() {
                for block in blocks {
                    match block["type"].as_str().unwrap_or("") {
                        "text" => {
                            if let Some(text) = block["text"].as_str() {
                                if !text.is_empty() {
                                    let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
                                }
                            }
                        }
                        "tool_use" => {
                            let _ = tx.send(ChatEvent::ToolActivity {
                                summary: summarize_tool(block),
                                is_error: false,
                            });
                        }
                        _ => {}
                    }
                }
            }
            false
        }
        "result" => {
            if event["is_error"].as_bool() == Some(true) {
                let msg = event["result"]
                    .as_str()
                    .or_else(|| event["error"].as_str())
                    .unwrap_or("claude reported an error");
                let _ = tx.send(ChatEvent::Error(msg.to_owned()));
                return true;
            }
            let usage = event["usage"].as_object().map(|u| Usage {
                input_tokens: u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)
                    + u.get("cache_read_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                    + u.get("cache_creation_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
            });
            let _ = tx.send(ChatEvent::Completed {
                tool_calls: Vec::new(),
                stop_reason: None,
                usage,
            });
            true
        }
        _ => false,
    }
}

/// A short, human-readable line for a tool_use block, e.g. `Bash: cargo test`.
fn summarize_tool(block: &Value) -> String {
    let name = block["name"].as_str().unwrap_or("tool");
    let input = &block["input"];
    let detail = [
        "command",
        "file_path",
        "path",
        "pattern",
        "url",
        "description",
    ]
    .iter()
    .find_map(|key| input[*key].as_str());
    match detail {
        Some(d) => format!("{name}: {}", first_line(d)),
        None => name.to_owned(),
    }
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() > 120 {
        format!("{}…", line.chars().take(120).collect::<String>())
    } else {
        line.to_owned()
    }
}

fn last_user_text(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|m| match m {
        Message::User(c) => Some(c.text().to_owned()),
        _ => None,
    })
}

fn has_images(messages: &[Message]) -> bool {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User(c) => Some(!c.images().is_empty()),
            _ => None,
        })
        .unwrap_or(false)
}

fn user_message_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| matches!(m, Message::User(_)))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::UserContent;
    use serde_json::json;
    use tokio::sync::mpsc::unbounded_channel;

    fn drain(events: &mut tokio::sync::mpsc::UnboundedReceiver<ChatEvent>) -> Vec<ChatEvent> {
        let mut out = Vec::new();
        while let Ok(e) = events.try_recv() {
            out.push(e);
        }
        out
    }

    #[test]
    fn assistant_text_and_tool_use_map_to_events() {
        let (tx, mut rx) = unbounded_channel();
        let event = json!({
            "type": "assistant",
            "message": {"content": [
                {"type": "text", "text": "Reading the file."},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "src/main.rs"}},
                {"type": "tool_use", "name": "Bash", "input": {"command": "cargo test\n--all"}},
            ]},
        });
        assert!(!handle_event(&event, &tx));
        let events = drain(&mut rx);
        assert!(matches!(&events[0], ChatEvent::TextDelta(t) if t == "Reading the file."));
        assert!(
            matches!(&events[1], ChatEvent::ToolActivity { summary, .. } if summary == "Read: src/main.rs")
        );
        assert!(
            matches!(&events[2], ChatEvent::ToolActivity { summary, .. } if summary == "Bash: cargo test")
        );
    }

    #[test]
    fn result_event_completes_with_usage() {
        let (tx, mut rx) = unbounded_channel();
        let event = json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "result": "done",
            "usage": {"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 50},
        });
        assert!(handle_event(&event, &tx));
        let events = drain(&mut rx);
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::Completed {
                usage: Some(u),
                tool_calls,
                ..
            } => {
                assert_eq!(u.input_tokens, 120);
                assert_eq!(u.output_tokens, 50);
                assert!(tool_calls.is_empty());
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn error_result_maps_to_error_event() {
        let (tx, mut rx) = unbounded_channel();
        let event = json!({
            "type": "result",
            "is_error": true,
            "result": "Credit balance is too low",
        });
        assert!(handle_event(&event, &tx));
        assert!(matches!(&drain(&mut rx)[0], ChatEvent::Error(m) if m.contains("Credit balance")));
    }

    #[test]
    fn continuation_is_driven_by_user_message_count() {
        let first = vec![Message::User("hi".into())];
        let later = vec![
            Message::User("hi".into()),
            Message::Assistant {
                text: "hello".into(),
                tool_calls: vec![],
            },
            Message::User("again".into()),
        ];
        assert_eq!(user_message_count(&first), 1);
        assert_eq!(user_message_count(&later), 2);
    }

    #[test]
    fn last_user_text_and_images_read_the_latest_turn() {
        let messages = vec![
            Message::User("old".into()),
            Message::Assistant {
                text: "x".into(),
                tool_calls: vec![],
            },
            Message::User(UserContent::Rich {
                text: "newest".into(),
                images: vec![crate::providers::ImageData {
                    media_type: "image/png".into(),
                    data: "AA==".into(),
                }],
            }),
        ];
        assert_eq!(last_user_text(&messages).as_deref(), Some("newest"));
        assert!(has_images(&messages));
    }
}
