//! Sub-agent providers backed by an official CLI (Claude Code) running on the
//! user's subscription. We never see or store a token: the CLI owns its own
//! auth. We spawn it headless, stream its NDJSON events, and adapt them into
//! our provider-agnostic [`ChatEvent`]s. The CLI runs its own tool loop, so our
//! tool definitions and approval flow do not apply here.

use super::{ChatEvent, ChatRequest, Config, Message, Usage};
use anyhow::{Context, Result};
use serde_json::Value;
use std::fmt::Write as _;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

/// Whether the `claude` CLI is installed and responds. Cheap version probe with
/// a short timeout so discovery never hangs.
pub async fn claude_available() -> bool {
    command_available("claude").await
}

async fn command_available(name: &str) -> bool {
    let mut command = tokio::process::Command::new(name);
    command
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let probe = command.status();
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
    let Some(prompt) = prompt_for_request(req) else {
        anyhow::bail!("no user message to send");
    };
    if has_images(&req.messages) {
        let _ = tx.send(ChatEvent::ToolActivity {
            summary: "note: images are not yet forwarded to the Claude Code provider".into(),
            is_error: false,
        });
    }

    let cmd = fresh_claude_command(config, &prompt);
    drive(cmd, "claude", tx, handle_claude_event).await
}

fn fresh_claude_command(config: &Config, prompt: &str) -> tokio::process::Command {
    let permission_mode = if config.claude_code_bypass_permissions {
        "bypassPermissions"
    } else {
        "acceptEdits"
    };
    let mut cmd = tokio::process::Command::new("claude");
    cmd.arg("--print")
        .arg("--no-session-persistence")
        .arg(prompt)
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .arg("--permission-mode")
        .arg(permission_mode);
    cmd
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
                stop_reason: Some("stop".into()),
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

/// Build a self-contained prompt for a fresh CLI process. A genuinely first
/// turn stays as terse as the user wrote it; later requests carry the complete
/// provider-agnostic history because no cwd-global CLI session is resumed.
fn prompt_for_request(req: &ChatRequest) -> Option<String> {
    if let [Message::User(content)] = req.messages.as_slice() {
        return Some(content.text().to_owned());
    }
    if !req
        .messages
        .iter()
        .any(|message| matches!(message, Message::User(_)))
    {
        return None;
    }

    let mut prompt = String::from(
        "Continue the coding-assistant conversation below. This is a complete handoff from a \
fresh process; use the supplied history instead of assuming access to an earlier CLI session.\n\n",
    );
    prompt.push_str("## System instructions\n");
    prompt.push_str(&req.system);
    prompt.push_str("\n\n## Conversation history\n");

    for message in &req.messages {
        match message {
            Message::User(content) => {
                prompt.push_str("\n### User\n");
                for (index, image) in content.images().iter().enumerate() {
                    let _ = writeln!(
                        prompt,
                        "[image {}: {}; binary data omitted]",
                        index + 1,
                        image.media_type
                    );
                }
                prompt.push_str(content.text());
                prompt.push('\n');
            }
            Message::Assistant { text, tool_calls } => {
                prompt.push_str("\n### Assistant\n");
                prompt.push_str(text);
                prompt.push('\n');
                for call in tool_calls {
                    let _ = writeln!(
                        prompt,
                        "[tool call: {} (id {})]\n{}",
                        call.name, call.id, call.arguments
                    );
                }
            }
            Message::ToolResult {
                call_id,
                name,
                content,
                is_error,
            } => {
                let status = if *is_error { "error" } else { "success" };
                let _ = writeln!(prompt, "\n### Tool result: {name} (id {call_id}, {status})");
                prompt.push_str(content);
                prompt.push('\n');
            }
        }
    }

    prompt.push_str(
        "\n## Continuation\nContinue from this history and address the latest unresolved user request.",
    );
    Some(prompt)
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

// ---- Codex (ChatGPT subscription) ----

pub async fn codex_available() -> bool {
    command_available("codex").await
}

pub async fn stream_chat_codex(
    config: &Config,
    req: &ChatRequest,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    let Some(prompt) = prompt_for_request(req) else {
        anyhow::bail!("no user message to send");
    };
    if has_images(&req.messages) {
        let _ = tx.send(ChatEvent::ToolActivity {
            summary: "note: images are not yet forwarded to the Codex provider".into(),
            is_error: false,
        });
    }

    let cmd = fresh_codex_command(config, &prompt);
    drive(cmd, "codex", tx, handle_codex_event).await
}

fn fresh_codex_command(config: &Config, prompt: &str) -> tokio::process::Command {
    // Every request starts in a fresh, explicitly sandboxed process. Context is
    // carried in `prompt`, never inferred from another cwd-global CLI session.
    let sandbox = if config.codex_full_access {
        "danger-full-access"
    } else {
        "workspace-write"
    };
    let mut cmd = tokio::process::Command::new("codex");
    cmd.arg("exec")
        .arg("--ephemeral")
        .arg("--sandbox")
        .arg(sandbox)
        .arg("--json")
        .arg("--skip-git-repo-check")
        .arg(prompt);
    cmd
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
                stop_reason: Some("stop".into()),
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
    use crate::providers::{ImageData, ModelEntry, ProviderKind, ToolCall, UserContent};
    use serde_json::json;
    use tokio::sync::mpsc::unbounded_channel;

    fn request(system: &str, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: ModelEntry {
                provider: ProviderKind::Codex,
                id: "codex".into(),
            },
            system: system.into(),
            messages,
            tools: Vec::new(),
        }
    }

    fn config() -> Config {
        Config {
            anthropic_api_key: None,
            openai_api_key: None,
            openai_base_url: "http://127.0.0.1:9".into(),
            ollama_host: "http://127.0.0.1:9".into(),
            default_model: None,
            compact_threshold_chars: 80_000,
            ollama_num_ctx: 16_384,
            theme: None,
            claude_code_bypass_permissions: false,
            codex_full_access: false,
        }
    }

    fn command_args(command: &tokio::process::Command) -> Vec<String> {
        command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

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
    fn first_turn_prompt_stays_plain() {
        let req = request(
            "system context",
            vec![Message::User("fix the tests".into())],
        );
        assert_eq!(prompt_for_request(&req).as_deref(), Some("fix the tests"));
    }

    #[test]
    fn multi_turn_prompt_contains_the_complete_handoff() {
        let req = request(
            "system context",
            vec![
                Message::User(UserContent::Rich {
                    text: "inspect the screenshot".into(),
                    images: vec![ImageData {
                        media_type: "image/png".into(),
                        data: "BASE64-MUST-NOT-BE-IN-PROMPT".into(),
                    }],
                }),
                Message::Assistant {
                    text: "I will inspect it.".into(),
                    tool_calls: vec![ToolCall {
                        id: "call-1".into(),
                        name: "read_file".into(),
                        arguments: json!({"path": "src/main.rs"}),
                    }],
                },
                Message::ToolResult {
                    call_id: "call-1".into(),
                    name: "read_file".into(),
                    content: "fn main() {}".into(),
                    is_error: false,
                },
                Message::User("now fix it".into()),
            ],
        );

        let prompt = prompt_for_request(&req).unwrap();
        for expected in [
            "## System instructions\nsystem context",
            "### User\n[image 1: image/png; binary data omitted]\ninspect the screenshot",
            "### Assistant\nI will inspect it.",
            "[tool call: read_file (id call-1)]\n{\"path\":\"src/main.rs\"}",
            "### Tool result: read_file (id call-1, success)\nfn main() {}",
            "### User\nnow fix it",
        ] {
            assert!(
                prompt.contains(expected),
                "missing {expected:?} in {prompt}"
            );
        }
        assert!(!prompt.contains("BASE64-MUST-NOT-BE-IN-PROMPT"));
    }

    #[test]
    fn cli_commands_always_start_fresh() {
        let config = config();
        let claude_args = command_args(&fresh_claude_command(&config, "prompt"));
        assert!(!claude_args.iter().any(|arg| arg == "--continue"));
        assert!(claude_args
            .iter()
            .any(|arg| arg == "--no-session-persistence"));

        let codex_args = command_args(&fresh_codex_command(&config, "prompt"));
        assert_eq!(codex_args.first().map(String::as_str), Some("exec"));
        assert!(!codex_args
            .iter()
            .any(|arg| arg == "resume" || arg == "--last"));
        assert!(codex_args.iter().any(|arg| arg == "--ephemeral"));
        assert!(codex_args
            .windows(2)
            .any(|args| { args[0] == "--sandbox" && args[1] == "workspace-write" }));
    }

    #[test]
    fn prompt_requires_at_least_one_user_message() {
        let req = request(
            "system context",
            vec![Message::Assistant {
                text: "orphaned".into(),
                tool_calls: vec![],
            }],
        );
        assert!(prompt_for_request(&req).is_none());
    }

    #[test]
    fn image_notice_reads_the_latest_user_turn() {
        let messages = vec![
            Message::User("old".into()),
            Message::Assistant {
                text: "x".into(),
                tool_calls: vec![],
            },
            Message::User(UserContent::Rich {
                text: "newest".into(),
                images: vec![ImageData {
                    media_type: "image/png".into(),
                    data: "AA==".into(),
                }],
            }),
        ];
        assert!(has_images(&messages));
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
}
