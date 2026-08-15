//! Sub-agent providers backed by an official CLI (Claude Code) running on the
//! user's subscription. We never see or store a token: the CLI owns its own
//! auth. We spawn it headless, stream its NDJSON events, and adapt them into
//! our provider-agnostic [`ChatEvent`]s. The CLI runs its own tool loop, so our
//! tool definitions and approval flow do not apply here.

use super::{ChatEvent, ChatRequest, Config, Message, RequestPolicy, Usage};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

const STDERR_CAPTURE_CHARS: usize = 16_000;

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
        let _ = tx.send(ChatEvent::Notice(
            "images are not yet forwarded to the Claude Code provider".into(),
        ));
    }

    let model = cli_model_override(
        &req.model,
        super::ProviderKind::ClaudeCode,
        "claude-code",
        "claude-code:",
    )?;
    let cmd = fresh_claude_command(config, model, req.policy);
    drive(cmd, "claude", &prompt, tx, handle_claude_event).await
}

fn fresh_claude_command(
    config: &Config,
    model: Option<&str>,
    policy: RequestPolicy,
) -> tokio::process::Command {
    let permission_mode = match policy {
        RequestPolicy::ReadOnly => "plan",
        RequestPolicy::Interactive if config.claude_code_bypass_permissions => "bypassPermissions",
        RequestPolicy::Interactive => "acceptEdits",
    };
    let mut cmd = tokio::process::Command::new("claude");
    cmd.arg("--print").arg("--no-session-persistence");
    if policy == RequestPolicy::ReadOnly {
        cmd.arg("--safe-mode");
    }
    if let Some(model) = model {
        cmd.arg("--model").arg(model);
    }
    cmd.arg("--input-format")
        .arg("text")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .arg("--permission-mode")
        .arg(permission_mode);
    if policy == RequestPolicy::ReadOnly {
        cmd.arg("--tools").arg("Read,Glob,Grep");
    }
    cmd
}

/// Shared subprocess driver: spawn the CLI, stream its NDJSON stdout through
/// `handle` (which returns true on the turn's terminal event), drain stderr so
/// the pipe never blocks, and surface a useful error if the turn never
/// completed.
async fn drive(
    mut cmd: tokio::process::Command,
    name: &str,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
    handle: impl Fn(&Value, &UnboundedSender<ChatEvent>) -> bool,
) -> Result<()> {
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to launch `{name}` — is it installed and signed in?"))?;
    let stdout = child.stdout.take().context("no stdout")?;
    let stderr = child.stderr.take().context("no stderr")?;
    let mut stdin = child.stdin.take().context("no stdin")?;

    // Drain stderr concurrently so the pipe never blocks the child.
    let stderr_task = tokio::spawn(async move {
        let mut buf = String::new();
        let mut captured = 0;
        let mut truncated = false;
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let remaining = STDERR_CAPTURE_CHARS.saturating_sub(captured);
            if remaining > 0 {
                let part: String = line.chars().take(remaining).collect();
                captured += part.chars().count();
                buf.push_str(&part);
                buf.push('\n');
                truncated |= part.chars().count() < line.chars().count();
            } else {
                truncated = true;
            }
        }
        if truncated {
            buf.push_str("…[stderr truncated]\n");
        }
        buf
    });

    // Send prompts through stdin instead of argv. This handles prompts that
    // begin with `-`, avoids process-list exposure, and removes ARG_MAX as the
    // conversation handoff grows.
    stdin.write_all(prompt.as_bytes()).await?;
    stdin.shutdown().await?;
    // `ChildStdin::shutdown` flushes pending bytes but does not guarantee that
    // the pipe handle is dropped. Both supported CLIs read stdin to EOF before
    // starting a turn, so keeping this handle alive deadlocks the child while
    // we wait for its stdout.
    drop(stdin);

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
    if !req.force_full_handoff {
        if let [Message::User(content)] = req.messages.as_slice() {
            return Some(content.text().to_owned());
        }
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

/// Exact model slugs advertised by the installed Codex CLI. Its experimental
/// machine-readable catalog is the compatibility boundary; the bundled list
/// remains a fallback when the locally refreshed catalog is unavailable.
pub async fn codex_model_ids() -> Vec<String> {
    let models = run_codex_model_catalog(&["debug", "models"]).await;
    if !models.is_empty() {
        return models;
    }
    run_codex_model_catalog(&["debug", "models", "--bundled"]).await
}

async fn run_codex_model_catalog(args: &[&str]) -> Vec<String> {
    let mut command = tokio::process::Command::new("codex");
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(std::time::Duration::from_secs(3), command.output())
        .await
        .ok()
        .and_then(Result::ok);
    output
        .filter(|output| output.status.success())
        .map_or_else(Vec::new, |output| parse_codex_model_ids(&output.stdout))
}

fn parse_codex_model_ids(bytes: &[u8]) -> Vec<String> {
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return Vec::new();
    };
    let Some(models) = value["models"].as_array() else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    let mut visible: Vec<(u64, usize, String)> = models
        .iter()
        .enumerate()
        .filter_map(|(index, model)| {
            if model["visibility"].as_str() != Some("list") {
                return None;
            }
            let slug = model["slug"].as_str()?.trim();
            if slug.is_empty() || !seen.insert(slug.to_owned()) {
                return None;
            }
            Some((
                model["priority"].as_u64().unwrap_or(u64::MAX),
                index,
                slug.to_owned(),
            ))
        })
        .collect();
    visible.sort_by_key(|(priority, index, _)| (*priority, *index));
    visible.into_iter().map(|(_, _, slug)| slug).collect()
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
        let _ = tx.send(ChatEvent::Notice(
            "images are not yet forwarded to the Codex provider".into(),
        ));
    }

    let model = cli_model_override(&req.model, super::ProviderKind::Codex, "codex", "codex:")?;
    let cmd = fresh_codex_command(config, model, req.policy);
    drive(cmd, "codex", &prompt, tx, handle_codex_event).await
}

fn fresh_codex_command(
    config: &Config,
    model: Option<&str>,
    policy: RequestPolicy,
) -> tokio::process::Command {
    // Every request starts in a fresh, explicitly sandboxed process. Context is
    // carried in `prompt`, never inferred from another cwd-global CLI session.
    let sandbox = match policy {
        RequestPolicy::ReadOnly => "read-only",
        RequestPolicy::Interactive if config.codex_full_access => "danger-full-access",
        RequestPolicy::Interactive => "workspace-write",
    };
    let mut cmd = tokio::process::Command::new("codex");
    cmd.arg("exec").arg("--ephemeral");
    if policy == RequestPolicy::ReadOnly {
        // Worker runs must not inherit normal user configuration (including
        // configured integrations) or user/project execution-policy rules.
        cmd.arg("--ignore-user-config").arg("--ignore-rules");
    }
    cmd.arg("--sandbox").arg(sandbox);
    if let Some(model) = model {
        cmd.arg("--model").arg(model);
    }
    cmd.arg("--json").arg("--skip-git-repo-check").arg("-");
    cmd
}

fn cli_model_override<'a>(
    model: &'a super::ModelEntry,
    expected_provider: super::ProviderKind,
    default_id: &str,
    prefix: &str,
) -> Result<Option<&'a str>> {
    if model.provider != expected_provider {
        anyhow::bail!(
            "model selector {} belongs to {}, not {}",
            model.id,
            model.provider.label(),
            expected_provider.label()
        );
    }
    if model.id == default_id {
        return Ok(None);
    }
    let Some(raw) = model.id.strip_prefix(prefix) else {
        anyhow::bail!(
            "invalid {} model selector: {}",
            expected_provider.label(),
            model.id
        );
    };
    if raw.is_empty() || raw.chars().any(char::is_whitespace) {
        anyhow::bail!(
            "invalid {} model selector: {}",
            expected_provider.label(),
            model.id
        );
    }
    Ok(Some(raw))
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
                    // Codex can emit recoverable warnings (for example, an
                    // unstable-feature warning) as an item-level `error` and
                    // then continue with the assistant response. Only the
                    // top-level `error` / `turn.failed` events are terminal.
                    let _ = tx.send(ChatEvent::Notice(
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
            policy: RequestPolicy::Interactive,
            force_full_handoff: false,
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
            reduced_motion: false,
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
    fn forced_first_turn_handoff_keeps_system_contract_and_user_prompt() {
        let mut req = request(
            "orchestration contract and worker evidence",
            vec![Message::User("fix the tests".into())],
        );
        req.force_full_handoff = true;

        let prompt = prompt_for_request(&req).unwrap();
        assert!(prompt.contains("## System instructions"));
        assert!(prompt.contains("orchestration contract and worker evidence"));
        assert!(prompt.contains("### User\nfix the tests"));
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
        let claude_args = command_args(&fresh_claude_command(
            &config,
            None,
            RequestPolicy::Interactive,
        ));
        assert!(!claude_args.iter().any(|arg| arg == "--continue"));
        assert!(claude_args
            .iter()
            .any(|arg| arg == "--no-session-persistence"));
        assert!(claude_args
            .windows(2)
            .any(|args| args[0] == "--input-format" && args[1] == "text"));
        assert!(!claude_args.iter().any(|arg| arg == "prompt"));

        let codex_args = command_args(&fresh_codex_command(
            &config,
            None,
            RequestPolicy::Interactive,
        ));
        assert_eq!(codex_args.first().map(String::as_str), Some("exec"));
        assert!(!codex_args
            .iter()
            .any(|arg| arg == "resume" || arg == "--last"));
        assert!(codex_args.iter().any(|arg| arg == "--ephemeral"));
        assert!(codex_args
            .windows(2)
            .any(|args| { args[0] == "--sandbox" && args[1] == "workspace-write" }));
        assert_eq!(codex_args.last().map(String::as_str), Some("-"));
    }

    #[test]
    fn explicit_cli_models_are_forwarded_once() {
        let config = config();
        let claude_args = command_args(&fresh_claude_command(
            &config,
            Some("sonnet"),
            RequestPolicy::Interactive,
        ));
        assert_eq!(
            claude_args
                .windows(2)
                .filter(|args| args[0] == "--model" && args[1] == "sonnet")
                .count(),
            1
        );

        let codex_args = command_args(&fresh_codex_command(
            &config,
            Some("gpt-5.6-sol"),
            RequestPolicy::Interactive,
        ));
        assert_eq!(
            codex_args
                .windows(2)
                .filter(|args| args[0] == "--model" && args[1] == "gpt-5.6-sol")
                .count(),
            1
        );
    }

    #[test]
    fn model_selector_validation_rejects_malformed_ids() {
        let malformed = ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:".into(),
        };
        assert!(cli_model_override(&malformed, ProviderKind::Codex, "codex", "codex:").is_err());
        assert!(cli_model_override(
            &malformed,
            ProviderKind::ClaudeCode,
            "claude-code",
            "claude-code:"
        )
        .is_err());
    }

    #[test]
    fn read_only_cli_policy_overrides_elevated_config() {
        let mut config = config();
        config.claude_code_bypass_permissions = true;
        config.codex_full_access = true;

        let claude_args = command_args(&fresh_claude_command(
            &config,
            Some("sonnet"),
            RequestPolicy::ReadOnly,
        ));
        assert!(claude_args.iter().any(|arg| arg == "--safe-mode"));
        assert!(claude_args
            .windows(2)
            .any(|args| args[0] == "--permission-mode" && args[1] == "plan"));
        assert!(claude_args
            .windows(2)
            .any(|args| args[0] == "--tools" && args[1] == "Read,Glob,Grep"));
        assert!(!claude_args
            .iter()
            .any(|arg| arg == "acceptEdits" || arg == "bypassPermissions"));

        let codex_args = command_args(&fresh_codex_command(
            &config,
            Some("gpt-5.6-sol"),
            RequestPolicy::ReadOnly,
        ));
        assert!(codex_args
            .windows(2)
            .any(|args| args[0] == "--sandbox" && args[1] == "read-only"));
        assert!(codex_args.iter().any(|arg| arg == "--ignore-user-config"));
        assert!(codex_args.iter().any(|arg| arg == "--ignore-rules"));
        assert!(!codex_args.iter().any(|arg| arg == "danger-full-access"));
    }

    #[test]
    fn interactive_cli_policy_preserves_configured_permissions() {
        let mut config = config();
        config.claude_code_bypass_permissions = true;
        config.codex_full_access = true;

        let claude_args = command_args(&fresh_claude_command(
            &config,
            None,
            RequestPolicy::Interactive,
        ));
        assert!(!claude_args.iter().any(|arg| arg == "--safe-mode"));
        assert!(!claude_args.iter().any(|arg| arg == "--tools"));
        assert!(claude_args
            .windows(2)
            .any(|args| { args[0] == "--permission-mode" && args[1] == "bypassPermissions" }));

        let codex_args = command_args(&fresh_codex_command(
            &config,
            None,
            RequestPolicy::Interactive,
        ));
        assert!(codex_args
            .windows(2)
            .any(|args| args[0] == "--sandbox" && args[1] == "danger-full-access"));
        assert!(!codex_args.iter().any(|arg| arg == "--ignore-user-config"));
        assert!(!codex_args.iter().any(|arg| arg == "--ignore-rules"));
    }

    #[test]
    fn codex_catalog_filters_hidden_and_duplicate_models_by_priority() {
        let bytes = br#"{"models":[
            {"slug":"later","visibility":"list","priority":20},
            {"slug":"hidden","visibility":"hide","priority":1},
            {"slug":"first","visibility":"list","priority":2},
            {"slug":"first","visibility":"list","priority":3}
        ]}"#;
        assert_eq!(parse_codex_model_ids(bytes), vec!["first", "later"]);
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

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_closes_stdin_before_waiting_for_stdout() {
        let mut command = tokio::process::Command::new("sh");
        command.arg("-c").arg(
            r#"while IFS= read -r line || [ -n "$line" ]; do :; done
printf '%s\n' '{"type":"turn.completed","usage":{}}'"#,
        );
        let (tx, mut rx) = unbounded_channel();

        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            drive(
                command,
                "test-cli",
                "prompt without newline",
                &tx,
                handle_codex_event,
            ),
        )
        .await
        .expect("driver should close stdin so the child can observe EOF")
        .expect("driver should accept the terminal event");

        assert!(matches!(rx.try_recv(), Ok(ChatEvent::Completed { .. })));
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
    fn codex_item_warning_does_not_swallow_the_later_response() {
        let (tx, mut rx) = unbounded_channel();
        assert!(!handle_codex_event(
            &json!({
                "type": "item.completed",
                "item": {
                    "type": "error",
                    "message": "unstable feature warning"
                }
            }),
            &tx,
        ));
        assert!(!handle_codex_event(
            &json!({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Luna replied"}
            }),
            &tx,
        ));
        assert!(handle_codex_event(
            &json!({"type": "turn.completed", "usage": {}}),
            &tx,
        ));

        let events = drain(&mut rx);
        assert!(
            matches!(&events[0], ChatEvent::Notice(message) if message.contains("unstable feature"))
        );
        assert!(matches!(&events[1], ChatEvent::TextDelta(text) if text == "Luna replied"));
        assert!(matches!(&events[2], ChatEvent::Completed { .. }));
        assert!(!events
            .iter()
            .any(|event| matches!(event, ChatEvent::Error(_))));
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
