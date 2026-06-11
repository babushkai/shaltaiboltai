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
    drive(cmd, "claude", tx, handle_claude_event).await
}

/// Shared subprocess driver: spawn the CLI, stream its NDJSON stdout through
/// `handle` (which returns true on the turn's terminal event), drain stderr so
/// the pipe never blocks, and surface a useful error if the turn never
/// completed.
async fn drive(
    mut cmd: tokio::process::Command,
    name: &str,
    tx: &UnboundedSender<ChatEvent>,
    handle: impl Fn(&Value, &UnboundedSender<ChatEvent>) -> bool,
) -> Result<()> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to launch `{name}` — is it installed and signed in?"))?;
    let stdout = child.stdout.take().context("no stdout")?;
    let stderr = child.stderr.take().context("no stderr")?;

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
            if handle(&event, tx) {
                saw_result = true;
            }
        }
    }

    let status = child.wait().await?;
    let stderr = stderr_task.await.unwrap_or_default();
    if !saw_result {
        let detail = stderr.trim();
        if status.success() {
            anyhow::bail!("{name} produced no result");
        } else if detail.is_empty() {
            anyhow::bail!("{name} exited with {status}");
        } else {
            anyhow::bail!("{name} error: {detail}");
        }
    }
    Ok(())
}

/// Translate one Claude Code stream-json event. Returns true when this was the
/// terminal `result` event (so the caller knows the turn completed cleanly).
fn handle_claude_event(event: &Value, tx: &UnboundedSender<ChatEvent>) -> bool {
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

// ---- Codex (ChatGPT subscription) ----

pub async fn codex_available() -> bool {
    let probe = tokio::process::Command::new("codex")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    matches!(
        tokio::time::timeout(std::time::Duration::from_secs(3), probe).await,
        Ok(Ok(status)) if status.success()
    )
}

pub async fn stream_chat_codex(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let Some(prompt) = last_user_text(&req.messages) else {
        anyhow::bail!("no user message to send");
    };
    if has_images(&req.messages) {
        let _ = tx.send(ChatEvent::ToolActivity {
            summary: "note: images are not yet forwarded to the Codex provider".into(),
            is_error: false,
        });
    }

    let mut cmd = tokio::process::Command::new("codex");
    cmd.arg("exec");
    // `resume --last` continues the most recent session in this directory; it
    // reuses that session's sandbox, so --sandbox is only set on a fresh run.
    if user_message_count(&req.messages) > 1 {
        cmd.arg("resume").arg("--last");
    } else {
        // workspace-write is OS-sandboxed (no network, confined to the cwd);
        // danger-full-access removes the sandbox entirely.
        let sandbox = if config.codex_full_access {
            "danger-full-access"
        } else {
            "workspace-write"
        };
        cmd.arg("--sandbox").arg(sandbox);
    }
    cmd.arg("--json").arg("--skip-git-repo-check").arg(&prompt);

    drive(cmd, "codex", tx, handle_codex_event).await
}

/// Translate one `codex exec --json` event. Returns true on `turn.completed`
/// (the terminal event).
fn handle_codex_event(event: &Value, tx: &UnboundedSender<ChatEvent>) -> bool {
    match event["type"].as_str().unwrap_or("") {
        "item.completed" | "item.updated" => {
            let item = &event["item"];
            match item["type"].as_str().unwrap_or("") {
                // Only emit finished assistant messages, so item.updated deltas
                // (if any) don't double up with the completed text.
                "agent_message" if event["type"] == "item.completed" => {
                    if let Some(text) = item["text"].as_str() {
                        if !text.is_empty() {
                            let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
                        }
                    }
                }
                "reasoning" | "agent_message" | "todo_list" => {}
                "error" => {
                    let msg = item["message"].as_str().or_else(|| item["text"].as_str());
                    let _ = tx.send(ChatEvent::Error(
                        msg.unwrap_or("codex reported an error").to_owned(),
                    ));
                }
                _ if event["type"] == "item.completed" => {
                    let _ = tx.send(ChatEvent::ToolActivity {
                        summary: summarize_codex_item(item),
                        is_error: item["exit_code"].as_i64().is_some_and(|c| c != 0),
                    });
                }
                _ => {}
            }
            false
        }
        "turn.completed" => {
            // Codex `input_tokens` already includes the cached portion, so it is
            // used as-is (unlike Claude's additive cache fields).
            let usage = event["usage"].as_object().map(|u| Usage {
                input_tokens: u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
                output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
            });
            let _ = tx.send(ChatEvent::Completed {
                tool_calls: Vec::new(),
                stop_reason: None,
                usage,
            });
            true
        }
        "turn.failed" | "error" => {
            let msg = event["error"]["message"]
                .as_str()
                .or_else(|| event["message"].as_str())
                .unwrap_or("codex turn failed");
            let _ = tx.send(ChatEvent::Error(msg.to_owned()));
            true
        }
        _ => false,
    }
}

/// Best-effort one-liner for a non-message Codex item (command_execution,
/// file_change, web_search, mcp_tool_call, …). Defensive about field names
/// since these vary by item type and CLI version.
fn summarize_codex_item(item: &Value) -> String {
    let kind = item["type"].as_str().unwrap_or("activity");
    let detail = ["command", "query", "path", "name", "title", "url"]
        .iter()
        .find_map(|key| item[*key].as_str());
    match detail {
        Some(d) => format!("{kind}: {}", first_line(d)),
        None => match item["changes"].as_array() {
            Some(changes) if !changes.is_empty() => {
                format!("{kind}: {} file(s)", changes.len())
            }
            _ => kind.replace('_', " "),
        },
    }
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
        assert!(!handle_claude_event(&event, &tx));
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
        assert!(handle_claude_event(&event, &tx));
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
        assert!(handle_claude_event(&event, &tx));
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
    fn codex_agent_message_and_completion_map_to_events() {
        let (tx, mut rx) = unbounded_channel();
        assert!(!handle_codex_event(
            &json!({"type": "thread.started", "thread_id": "x"}),
            &tx
        ));
        assert!(!handle_codex_event(
            &json!({"type": "item.completed", "item": {"type": "agent_message", "text": "pong"}}),
            &tx,
        ));
        assert!(handle_codex_event(
            &json!({"type": "turn.completed", "usage": {"input_tokens": 13293, "cached_input_tokens": 2432, "output_tokens": 5}}),
            &tx,
        ));
        let events = drain(&mut rx);
        assert!(matches!(&events[0], ChatEvent::TextDelta(t) if t == "pong"));
        match &events[1] {
            // Codex input_tokens already includes the cached portion: used as-is.
            ChatEvent::Completed { usage: Some(u), .. } => {
                assert_eq!(u.input_tokens, 13293);
                assert_eq!(u.output_tokens, 5);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn codex_tool_items_become_activity_lines() {
        let (tx, mut rx) = unbounded_channel();
        handle_codex_event(
            &json!({"type": "item.completed", "item": {"type": "command_execution", "command": "cargo test\n--all", "exit_code": 0}}),
            &tx,
        );
        handle_codex_event(
            &json!({"type": "item.completed", "item": {"type": "command_execution", "command": "false", "exit_code": 1}}),
            &tx,
        );
        let events = drain(&mut rx);
        assert!(
            matches!(&events[0], ChatEvent::ToolActivity { summary, is_error: false } if summary == "command_execution: cargo test")
        );
        assert!(matches!(
            &events[1],
            ChatEvent::ToolActivity { is_error: true, .. }
        ));
    }

    #[test]
    fn codex_reasoning_items_are_silent() {
        let (tx, mut rx) = unbounded_channel();
        handle_codex_event(
            &json!({"type": "item.completed", "item": {"type": "reasoning", "text": "thinking hard"}}),
            &tx,
        );
        assert!(drain(&mut rx).is_empty());
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
