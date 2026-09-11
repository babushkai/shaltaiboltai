//! Codex app-server protocol primitives shared by the existing isolated
//! advisory path and the dormant persistent transport.
//!
//! The production provider still owns process launch and routing. Keeping the
//! wire contract here makes its fail-closed attestation and event decoding
//! reusable without changing which requests use app-server.

use super::{ChatEvent, Usage};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

pub(super) const VERSION: &str = "0.153.4";
pub(super) const INITIALIZE_ID: i64 = 0;
pub(super) const THREAD_START_ID: i64 = 1;
pub(super) const TURN_START_ID: i64 = 2;
pub(super) const INTERRUPT_ID: i64 = 3;

pub(super) const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_INSTRUCTION_SOURCES: usize = 128;
const MAX_PRE_RESPONSE_NOTIFICATIONS: usize = 256;
const MAX_PRE_RESPONSE_NOTIFICATION_BYTES: usize = 8 * MAX_FRAME_BYTES;

#[derive(Default)]
struct BufferedNotifications {
    messages: VecDeque<Value>,
    encoded_bytes: usize,
}

impl BufferedNotifications {
    fn push(&mut self, message: Value) -> Result<()> {
        if self.messages.len() >= MAX_PRE_RESPONSE_NOTIFICATIONS {
            anyhow::bail!(
                "Codex app-server emitted more than {MAX_PRE_RESPONSE_NOTIFICATIONS} notifications before a response"
            );
        }
        let encoded_bytes = serde_json::to_vec(&message)
            .context("measure buffered Codex app-server notification")?
            .len()
            .saturating_add(1);
        let total = self
            .encoded_bytes
            .checked_add(encoded_bytes)
            .context("Codex app-server notification buffer size overflowed")?;
        if total > MAX_PRE_RESPONSE_NOTIFICATION_BYTES {
            anyhow::bail!(
                "Codex app-server notifications before a response exceeded {MAX_PRE_RESPONSE_NOTIFICATION_BYTES} bytes"
            );
        }
        self.messages.push_back(message);
        self.encoded_bytes = total;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(super) struct Contract {
    pub(super) profile: String,
    pub(super) model: String,
    pub(super) cwd: PathBuf,
    pub(super) workspace_roots: Vec<PathBuf>,
    pub(super) codex_home: PathBuf,
}

pub(super) async fn run_one_turn<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    contract: &Contract,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    send_message(
        stdin,
        &serde_json::json!({
            "id": INITIALIZE_ID,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "shaltaiboltai",
                    "title": "Shaltaiboltai",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {"experimentalApi": true}
            }
        }),
    )
    .await?;
    let initialized = read_response(stdout, stdin, INITIALIZE_ID, tx).await?;
    attest_initialize(&initialized, contract)?;

    send_message(stdin, &serde_json::json!({"method": "initialized"})).await?;
    send_message(
        stdin,
        &serde_json::json!({
            "id": THREAD_START_ID,
            "method": "thread/start",
            "params": {
                "model": contract.model,
                "modelProvider": "openai",
                "allowProviderModelFallback": false,
                "cwd": contract.cwd,
                "runtimeWorkspaceRoots": contract.workspace_roots,
                "approvalPolicy": "never",
                "approvalsReviewer": "user",
                "permissions": contract.profile,
                "ephemeral": true,
                "historyMode": "legacy",
                "environments": [{
                    "environmentId": "local",
                    "cwd": contract.cwd,
                    "runtimeWorkspaceRoots": contract.workspace_roots
                }],
                "dynamicTools": [],
                "selectedCapabilityRoots": [],
                "experimentalRawEvents": false
            }
        }),
    )
    .await?;
    let thread_started = read_response(stdout, stdin, THREAD_START_ID, tx).await?;
    let thread_id = attest_thread(&thread_started, contract)?;

    // This is the first model/cost-bearing operation. Never retry it: an
    // ambiguous transport failure may mean the upstream model already ran.
    send_message(
        stdin,
        &serde_json::json!({
            "id": TURN_START_ID,
            "method": "turn/start",
            "params": {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt, "textElements": []}]
            }
        }),
    )
    .await?;
    let mut buffered_notifications = BufferedNotifications::default();
    let turn_started = read_response_buffering(
        stdout,
        stdin,
        TURN_START_ID,
        tx,
        &mut buffered_notifications,
    )
    .await?;
    let turn = turn_started
        .get("turn")
        .and_then(Value::as_object)
        .context("Codex turn/start response omitted turn")?;
    let turn_id = turn
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .context("Codex turn/start response omitted turn.id")?
        .to_owned();
    if turn.get("status").and_then(Value::as_str) != Some("inProgress") {
        anyhow::bail!("Codex turn/start did not create an in-progress turn");
    }

    consume_turn_with_pending(
        stdout,
        stdin,
        &thread_id,
        &turn_id,
        buffered_notifications.messages,
        tx,
    )
    .await
}

pub(super) async fn send_message<W>(stdin: &mut W, message: &Value) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut encoded = serde_json::to_vec(message).context("serialize Codex app-server message")?;
    if encoded.len() > MAX_FRAME_BYTES {
        anyhow::bail!("Codex app-server request exceeded {MAX_FRAME_BYTES} bytes");
    }
    encoded.push(b'\n');
    stdin.write_all(&encoded).await?;
    stdin.flush().await?;
    Ok(())
}

pub(super) async fn read_message<R>(stdout: &mut BufReader<R>) -> Result<Value>
where
    R: AsyncRead + Unpin,
{
    let mut record = Vec::new();
    loop {
        read_bounded_record(stdout, &mut record)
            .await?
            .context("Codex app-server closed stdout before completing the turn")?;
        if record.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        return serde_json::from_slice(&record).context("invalid JSON from Codex app-server");
    }
}

async fn read_bounded_record<R>(
    reader: &mut BufReader<R>,
    record: &mut Vec<u8>,
) -> Result<Option<()>>
where
    R: AsyncRead + Unpin,
{
    record.clear();
    loop {
        let (consume, complete, overflow, eof) = {
            let available = reader.fill_buf().await?;
            if available.is_empty() {
                (0, false, false, true)
            } else {
                let newline = available.iter().position(|byte| *byte == b'\n');
                let retained = newline.unwrap_or(available.len());
                let overflow = record.len().saturating_add(retained) > MAX_FRAME_BYTES;
                if !overflow {
                    record.extend_from_slice(&available[..retained]);
                }
                (
                    newline.map_or(available.len(), |index| index + 1),
                    newline.is_some(),
                    overflow,
                    false,
                )
            }
        };
        reader.consume(consume);
        if overflow {
            anyhow::bail!("CLI stdout NDJSON record exceeded {MAX_FRAME_BYTES} bytes");
        }
        if complete {
            return Ok(Some(()));
        }
        if eof {
            return Ok((!record.is_empty()).then_some(()));
        }
    }
}

async fn read_response<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    expected_id: i64,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<Value>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    read_response_inner(stdout, stdin, expected_id, tx, None).await
}

async fn read_response_buffering<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    expected_id: i64,
    tx: &UnboundedSender<ChatEvent>,
    buffered_notifications: &mut BufferedNotifications,
) -> Result<Value>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    read_response_inner(stdout, stdin, expected_id, tx, Some(buffered_notifications)).await
}

async fn read_response_inner<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    expected_id: i64,
    tx: &UnboundedSender<ChatEvent>,
    mut buffered_notifications: Option<&mut BufferedNotifications>,
) -> Result<Value>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        let message = read_message(stdout).await?;
        if message.get("method").is_some() {
            if message.get("id").is_some() {
                reject_server_request(stdin, &message).await?;
            } else if let Some(buffer) = buffered_notifications.as_deref_mut() {
                buffer.push(message)?;
            } else {
                emit_notice(&message, tx);
            }
            continue;
        }
        if message.get("id").and_then(Value::as_i64) != Some(expected_id) {
            anyhow::bail!("Codex app-server returned an unexpected response id");
        }
        if let Some(error) = message.get("error") {
            let detail = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown JSON-RPC error");
            anyhow::bail!("Codex app-server request {expected_id} failed: {detail}");
        }
        return message
            .get("result")
            .cloned()
            .context("Codex app-server response omitted result");
    }
}

pub(super) async fn reject_server_request<W>(stdin: &mut W, request: &Value) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let id = request
        .get("id")
        .cloned()
        .context("Codex app-server request omitted id")?;
    send_message(
        stdin,
        &serde_json::json!({
            "id": id,
            "error": {
                "code": -32601,
                "message": "Shaltaiboltai advisory transport does not service server requests"
            }
        }),
    )
    .await
}

pub(super) fn emit_notice(message: &Value, tx: &UnboundedSender<ChatEvent>) {
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let params = &message["params"];
    let text = match method {
        "warning"
        | "guardianWarning"
        | "modelProvider/authRecoveryStarted"
        | "modelProvider/authRecoveryCompleted" => params.get("message").and_then(Value::as_str),
        "configWarning" | "deprecationNotice" => params.get("summary").and_then(Value::as_str),
        _ => None,
    };
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        let _ = tx.send(ChatEvent::Notice(text.to_owned()));
    }
}

pub(super) fn attest_initialize(result: &Value, contract: &Contract) -> Result<()> {
    let user_agent = result
        .get("userAgent")
        .and_then(Value::as_str)
        .context("Codex initialize response omitted userAgent")?;
    let version = user_agent
        .strip_prefix("shaltaiboltai/")
        .and_then(|suffix| suffix.split_whitespace().next())
        .filter(|version| !version.is_empty())
        .context("Codex initialize response had an invalid userAgent")?;
    if version != VERSION {
        anyhow::bail!("unsupported Codex app-server version {version}; expected exactly {VERSION}");
    }
    if result.get("platformFamily").and_then(Value::as_str) != Some(std::env::consts::FAMILY)
        || result.get("platformOs").and_then(Value::as_str) != Some(std::env::consts::OS)
    {
        anyhow::bail!("Codex initialize response platform did not match this process");
    }
    let codex_home = result
        .get("codexHome")
        .and_then(Value::as_str)
        .context("Codex initialize response omitted codexHome")?;
    let expected_codex_home = contract
        .codex_home
        .to_str()
        .context("isolated Codex home is not valid UTF-8")?;
    if codex_home != expected_codex_home {
        anyhow::bail!("Codex initialize response did not attest the exact isolated Codex home");
    }
    Ok(())
}

pub(super) fn attest_thread(result: &Value, contract: &Contract) -> Result<String> {
    let cwd = contract
        .cwd
        .to_str()
        .context("Codex advisory cwd is not valid UTF-8")?;
    let expected_roots = contract
        .workspace_roots
        .iter()
        .map(|root| {
            root.to_str()
                .map(str::to_owned)
                .context("Codex advisory workspace root is not valid UTF-8")
        })
        .collect::<Result<Vec<_>>>()?;
    let actual_roots = result
        .get("runtimeWorkspaceRoots")
        .and_then(Value::as_array)
        .context("Codex thread/start response omitted runtimeWorkspaceRoots")?
        .iter()
        .map(|root| {
            root.as_str()
                .map(str::to_owned)
                .context("Codex returned a non-string runtime workspace root")
        })
        .collect::<Result<Vec<_>>>()?;
    attest_instruction_sources(result, contract)?;

    let thread = result
        .get("thread")
        .and_then(Value::as_object)
        .context("Codex thread/start response omitted thread")?;
    let thread_id = thread
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .context("Codex thread/start response omitted thread.id")?;
    let profile = result
        .get("activePermissionProfile")
        .and_then(Value::as_object)
        .context("Codex did not activate the requested permission profile")?;
    let sandbox = result
        .get("sandbox")
        .and_then(Value::as_object)
        .context("Codex thread/start response omitted sandbox")?;

    let matches_contract = result.get("model").and_then(Value::as_str)
        == Some(contract.model.as_str())
        && result.get("modelProvider").and_then(Value::as_str) == Some("openai")
        && result.get("cwd").and_then(Value::as_str) == Some(cwd)
        && actual_roots == expected_roots
        && result.get("approvalPolicy").and_then(Value::as_str) == Some("never")
        && result.get("approvalsReviewer").and_then(Value::as_str) == Some("user")
        && profile.get("id").and_then(Value::as_str) == Some(contract.profile.as_str())
        && profile.get("extends").is_some_and(Value::is_null)
        && sandbox.get("type").and_then(Value::as_str) == Some("readOnly")
        && sandbox.get("networkAccess").and_then(Value::as_bool) == Some(false)
        && thread.get("cliVersion").and_then(Value::as_str) == Some(VERSION)
        && thread.get("ephemeral").and_then(Value::as_bool) == Some(true)
        && thread.get("path").is_some_and(Value::is_null)
        && thread.get("historyMode").and_then(Value::as_str) == Some("legacy")
        && thread.get("modelProvider").and_then(Value::as_str) == Some("openai")
        && thread.get("model").and_then(Value::as_str) == Some(contract.model.as_str())
        && thread.get("cwd").and_then(Value::as_str) == Some(cwd);
    if !matches_contract {
        anyhow::bail!(
            "Codex thread/start attestation did not match the requested read-only advisory contract"
        );
    }
    Ok(thread_id.to_owned())
}

fn attest_instruction_sources(result: &Value, contract: &Contract) -> Result<()> {
    let sources = result
        .get("instructionSources")
        .and_then(Value::as_array)
        .context("Codex thread/start response omitted instructionSources")?;
    if sources.len() > MAX_INSTRUCTION_SOURCES {
        anyhow::bail!("Codex reported more than {MAX_INSTRUCTION_SOURCES} instruction sources");
    }
    for source in sources {
        let source = source
            .as_str()
            .filter(|source| !source.is_empty())
            .context("Codex returned an invalid instruction source")?;
        let lexical = Path::new(source);
        if !lexical.is_absolute()
            || !contract
                .workspace_roots
                .iter()
                .any(|root| lexical.starts_with(root))
        {
            anyhow::bail!("Codex instruction source is outside the reviewed workspace: {source}");
        }
        let canonical = std::fs::canonicalize(lexical)
            .with_context(|| format!("canonicalize Codex instruction source {source}"))?;
        let metadata = std::fs::metadata(&canonical)
            .with_context(|| format!("inspect Codex instruction source {source}"))?;
        if !metadata.is_file()
            || !contract
                .workspace_roots
                .iter()
                .any(|root| canonical.starts_with(root))
        {
            anyhow::bail!("Codex instruction source escaped the advisory workspace: {source}");
        }
    }
    Ok(())
}

async fn consume_turn_with_pending<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    thread_id: &str,
    turn_id: &str,
    mut pending: VecDeque<Value>,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut usage = None;
    let mut emitted_agent_items = HashSet::new();
    let mut terminal_error = None;
    loop {
        let message = match pending.pop_front() {
            Some(message) => message,
            None => read_message(stdout).await?,
        };
        if message.get("method").is_some() && message.get("id").is_some() {
            reject_server_request(stdin, &message).await?;
            continue;
        }
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            anyhow::bail!("Codex app-server emitted an unexpected response during a turn");
        };
        let params = &message["params"];
        match method {
            "item/completed" if notification_matches(params, thread_id, turn_id) => {
                emit_item(&params["item"], &mut emitted_agent_items, tx);
            }
            "thread/tokenUsage/updated" if notification_matches(params, thread_id, turn_id) => {
                usage = usage_from(params);
            }
            "warning" => {
                let applies = params
                    .get("threadId")
                    .and_then(Value::as_str)
                    .is_none_or(|id| id == thread_id);
                if applies {
                    emit_notice(&message, tx);
                }
            }
            "guardianWarning"
                if params.get("threadId").and_then(Value::as_str) == Some(thread_id) =>
            {
                emit_notice(&message, tx);
            }
            "modelProvider/authRecoveryStarted" | "modelProvider/authRecoveryCompleted"
                if notification_matches(params, thread_id, turn_id) =>
            {
                emit_notice(&message, tx);
            }
            "configWarning" | "deprecationNotice" => {
                emit_notice(&message, tx);
            }
            "error" if notification_matches(params, thread_id, turn_id) => {
                let detail = turn_error_message(&params["error"]);
                if params.get("willRetry").and_then(Value::as_bool) == Some(true) {
                    let _ = tx.send(ChatEvent::Notice(detail));
                } else {
                    terminal_error = Some(detail);
                }
            }
            "model/rerouted" if notification_matches(params, thread_id, turn_id) => {
                let from = params
                    .get("fromModel")
                    .and_then(Value::as_str)
                    .unwrap_or("requested model");
                let to = params
                    .get("toModel")
                    .and_then(Value::as_str)
                    .unwrap_or("another model");
                send_message(
                    stdin,
                    &serde_json::json!({
                        "id": INTERRUPT_ID,
                        "method": "turn/interrupt",
                        "params": {"threadId": thread_id, "turnId": turn_id}
                    }),
                )
                .await?;
                anyhow::bail!(
                    "Codex rerouted the advisory turn from {from} to {to}; exact-model contract failed"
                );
            }
            "turn/completed" => {
                // In the pinned 0.153.4 protocol, turn/completed carries the
                // turn id in params.turn.id (unlike item/usage/error events,
                // which carry params.turnId). Parse a targeted terminal event
                // strictly so malformed protocol cannot be ignored forever.
                let completed_thread_id = params
                    .get("threadId")
                    .and_then(Value::as_str)
                    .context("Codex turn/completed omitted threadId")?;
                if completed_thread_id != thread_id {
                    continue;
                }
                let turn = params
                    .get("turn")
                    .and_then(Value::as_object)
                    .context("Codex turn/completed omitted turn")?;
                let completed_turn_id = turn
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .context("Codex turn/completed omitted turn.id")?;
                if completed_turn_id != turn_id {
                    continue;
                }
                if let Some(items) = turn.get("items").and_then(Value::as_array) {
                    for item in items {
                        if item.get("type").and_then(Value::as_str) == Some("agentMessage") {
                            emit_item(item, &mut emitted_agent_items, tx);
                        }
                    }
                }
                match turn.get("status").and_then(Value::as_str) {
                    Some("completed") if terminal_error.is_none() => {
                        let _ = tx.send(ChatEvent::Completed {
                            tool_calls: Vec::new(),
                            stop_reason: Some("stop".into()),
                            usage,
                        });
                    }
                    Some("failed") => {
                        let error = turn
                            .get("error")
                            .map(turn_error_message)
                            .filter(|message| !message.is_empty())
                            .or(terminal_error)
                            .unwrap_or_else(|| "Codex advisory turn failed".into());
                        let _ = tx.send(ChatEvent::Error(error));
                    }
                    Some("interrupted") => {
                        let _ = tx
                            .send(ChatEvent::Error(terminal_error.unwrap_or_else(|| {
                                "Codex advisory turn was interrupted".into()
                            })));
                    }
                    Some("completed") => {
                        let _ = tx.send(ChatEvent::Error(
                            terminal_error.unwrap_or_else(|| "Codex advisory turn failed".into()),
                        ));
                    }
                    _ => anyhow::bail!("Codex turn/completed had an invalid status"),
                }
                return Ok(());
            }
            _ => {}
        }
    }
}

fn notification_matches(params: &Value, thread_id: &str, turn_id: &str) -> bool {
    params.get("threadId").and_then(Value::as_str) == Some(thread_id)
        && params.get("turnId").and_then(Value::as_str) == Some(turn_id)
}

fn usage_from(params: &Value) -> Option<Usage> {
    let total = params.get("tokenUsage")?.get("total")?;
    Some(Usage {
        // inputTokens already includes cachedInputTokens; do not add it again.
        input_tokens: total.get("inputTokens")?.as_u64()?,
        output_tokens: total.get("outputTokens")?.as_u64()?,
    })
}

fn turn_error_message(error: &Value) -> String {
    error
        .get("additionalDetails")
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty())
        .or_else(|| error.get("message").and_then(Value::as_str))
        .unwrap_or("Codex advisory turn failed")
        .to_owned()
}

fn emit_item(
    item: &Value,
    emitted_agent_items: &mut HashSet<String>,
    tx: &UnboundedSender<ChatEvent>,
) {
    match item.get("type").and_then(Value::as_str).unwrap_or("") {
        "agentMessage" => {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                return;
            };
            if !emitted_agent_items.insert(id.to_owned()) {
                return;
            }
            if let Some(text) = item
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                let _ = tx.send(ChatEvent::TextDelta(text.to_owned()));
            }
        }
        "reasoning" | "plan" | "userMessage" | "hookPrompt" => {}
        _ => {
            let is_error = item
                .get("exitCode")
                .and_then(Value::as_i64)
                .is_some_and(|code| code != 0)
                || matches!(
                    item.get("status").and_then(Value::as_str),
                    Some("failed" | "declined")
                );
            let _ = tx.send(ChatEvent::ToolActivity {
                summary: summarize_item(item),
                is_error,
            });
        }
    }
}

fn summarize_item(item: &Value) -> String {
    let kind = item["type"].as_str().unwrap_or("activity");
    let detail = ["command", "query", "path", "name", "title", "url"]
        .iter()
        .find_map(|key| item[*key].as_str());
    match detail {
        Some(detail) => format!("{kind}: {}", first_line(detail)),
        None => match item["changes"].as_array() {
            Some(changes) if !changes.is_empty() => format!("{kind}: {} file(s)", changes.len()),
            _ => kind.replace('_', " "),
        },
    }
}

fn first_line(value: &str) -> String {
    let line = value.lines().next().unwrap_or("").trim();
    if line.chars().count() > 120 {
        format!("{}…", line.chars().take(120).collect::<String>())
    } else {
        line.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;
    use tokio::io::{duplex, split, AsyncReadExt};
    use tokio::sync::mpsc::unbounded_channel;

    async fn targeted_terminal_error(params: Value) -> String {
        let (client, mut server) = duplex(4 * 1024);
        let (client_read, mut client_write) = split(client);
        let mut client_read = BufReader::new(client_read);
        let (tx, _rx) = unbounded_channel();
        send_message(
            &mut server,
            &json!({"method": "turn/completed", "params": params}),
        )
        .await
        .expect("write malformed terminal notification");

        tokio::time::timeout(
            Duration::from_secs(1),
            consume_turn_with_pending(
                &mut client_read,
                &mut client_write,
                "thread-1",
                "turn-1",
                VecDeque::new(),
                &tx,
            ),
        )
        .await
        .expect("malformed targeted terminal must not hang")
        .expect_err("malformed targeted terminal must fail closed")
        .to_string()
    }

    #[tokio::test]
    async fn malformed_targeted_turn_completion_fails_without_hanging() {
        let missing_thread = targeted_terminal_error(json!({
            "turn": {"id": "turn-1", "status": "completed"}
        }))
        .await;
        assert!(
            missing_thread.contains("omitted threadId"),
            "{missing_thread}"
        );

        let missing_turn = targeted_terminal_error(json!({
            "threadId": "thread-1"
        }))
        .await;
        assert!(missing_turn.contains("omitted turn"), "{missing_turn}");

        let missing_turn_id = targeted_terminal_error(json!({
            "threadId": "thread-1",
            "turn": {"status": "completed"}
        }))
        .await;
        assert!(
            missing_turn_id.contains("omitted turn.id"),
            "{missing_turn_id}"
        );
    }

    #[tokio::test]
    async fn pre_response_turn_events_replay_in_order_and_finish_without_more_input() {
        let (client, mut server) = duplex(64 * 1024);
        let (client_read, mut client_write) = split(client);
        let mut client_read = BufReader::new(client_read);
        let (tx, mut rx) = unbounded_channel();

        for message in [
            json!({
                "method": "error",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "error": {
                        "message": "retrying request",
                        "codexErrorInfo": null,
                        "additionalDetails": null,
                        "misalignment": null
                    },
                    "willRetry": true
                }
            }),
            json!({
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {"type": "agentMessage", "id": "answer-1", "text": "pong"},
                    "completedAtMs": 2
                }
            }),
            json!({
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "tokenUsage": {
                        "total": {
                            "totalTokens": 14,
                            "inputTokens": 9,
                            "cachedInputTokens": 3,
                            "cacheWriteInputTokens": 0,
                            "outputTokens": 5,
                            "reasoningOutputTokens": 0
                        },
                        "last": {
                            "totalTokens": 14,
                            "inputTokens": 9,
                            "cachedInputTokens": 3,
                            "cacheWriteInputTokens": 0,
                            "outputTokens": 5,
                            "reasoningOutputTokens": 0
                        },
                        "modelContextWindow": 128000
                    }
                }
            }),
            json!({
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": "turn-1",
                        "items": [
                            {"type": "agentMessage", "id": "answer-1", "text": "pong"}
                        ],
                        "itemsView": "full",
                        "status": "completed",
                        "error": null,
                        "startedAt": 1,
                        "completedAt": 2,
                        "durationMs": 1000
                    }
                }
            }),
            json!({
                "id": TURN_START_ID,
                "result": {"turn": {
                    "id": "turn-1",
                    "items": [],
                    "itemsView": "notLoaded",
                    "status": "inProgress",
                    "error": null,
                    "startedAt": null,
                    "completedAt": null,
                    "durationMs": null
                }}
            }),
        ] {
            send_message(&mut server, &message)
                .await
                .expect("write pre-response frame");
        }

        let mut buffered = BufferedNotifications::default();
        let response = read_response_buffering(
            &mut client_read,
            &mut client_write,
            TURN_START_ID,
            &tx,
            &mut buffered,
        )
        .await
        .expect("read turn/start response");
        assert_eq!(response["turn"]["id"], "turn-1");
        assert_eq!(buffered.messages.len(), 4);

        tokio::time::timeout(
            Duration::from_secs(1),
            consume_turn_with_pending(
                &mut client_read,
                &mut client_write,
                "thread-1",
                "turn-1",
                buffered.messages,
                &tx,
            ),
        )
        .await
        .expect("buffered terminal must finish without another frame")
        .expect("consume buffered turn");

        let events = std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>();
        assert_eq!(events.len(), 3, "retry notice must be emitted exactly once");
        assert!(matches!(&events[0], ChatEvent::Notice(message) if message == "retrying request"));
        assert!(matches!(&events[1], ChatEvent::TextDelta(text) if text == "pong"));
        match &events[2] {
            ChatEvent::Completed {
                usage: Some(usage), ..
            } => {
                assert_eq!(usage.input_tokens, 9);
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected completed usage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pre_response_model_reroute_is_replayed_and_interrupted() {
        let (client, server) = duplex(16 * 1024);
        let (client_read, mut client_write) = split(client);
        let (server_read, mut server_write) = split(server);
        let mut client_read = BufReader::new(client_read);
        let mut server_read = BufReader::new(server_read);
        let (tx, mut rx) = unbounded_channel();

        send_message(
            &mut server_write,
            &json!({
                "method": "model/rerouted",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "fromModel": "gpt-5.6-sol",
                    "toModel": "gpt-5.5",
                    "reason": "unavailable"
                }
            }),
        )
        .await
        .expect("write reroute notification");
        send_message(
            &mut server_write,
            &json!({
                "id": TURN_START_ID,
                "result": {"turn": {
                    "id": "turn-1",
                    "items": [],
                    "itemsView": "notLoaded",
                    "status": "inProgress",
                    "error": null,
                    "startedAt": null,
                    "completedAt": null,
                    "durationMs": null
                }}
            }),
        )
        .await
        .expect("write turn/start response");

        let mut buffered = BufferedNotifications::default();
        read_response_buffering(
            &mut client_read,
            &mut client_write,
            TURN_START_ID,
            &tx,
            &mut buffered,
        )
        .await
        .expect("read turn/start response");
        let error = consume_turn_with_pending(
            &mut client_read,
            &mut client_write,
            "thread-1",
            "turn-1",
            buffered.messages,
            &tx,
        )
        .await
        .expect_err("reroute must fail the exact-model contract");
        assert!(error.to_string().contains("gpt-5.6-sol to gpt-5.5"));
        assert!(rx.try_recv().is_err());

        let interrupt =
            tokio::time::timeout(Duration::from_secs(1), read_message(&mut server_read))
                .await
                .expect("reroute interrupt must not hang")
                .expect("read reroute interrupt");
        assert_eq!(interrupt["id"], INTERRUPT_ID);
        assert_eq!(interrupt["method"], "turn/interrupt");
        assert_eq!(interrupt["params"]["threadId"], "thread-1");
        assert_eq!(interrupt["params"]["turnId"], "turn-1");
    }

    #[tokio::test]
    async fn foreign_pre_response_model_reroute_does_not_interrupt_or_leak() {
        let (client, server) = duplex(16 * 1024);
        let (client_read, mut client_write) = split(client);
        let (mut server_read, mut server_write) = split(server);
        let mut client_read = BufReader::new(client_read);
        let (tx, mut rx) = unbounded_channel();

        for message in [
            json!({
                "method": "model/rerouted",
                "params": {
                    "threadId": "other-thread",
                    "turnId": "turn-1",
                    "fromModel": "gpt-5.6-sol",
                    "toModel": "gpt-5.5",
                    "reason": "unavailable"
                }
            }),
            json!({
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": "turn-1",
                        "items": [],
                        "itemsView": "notLoaded",
                        "status": "completed",
                        "error": null,
                        "startedAt": 1,
                        "completedAt": 2,
                        "durationMs": 1000
                    }
                }
            }),
            json!({
                "id": TURN_START_ID,
                "result": {"turn": {
                    "id": "turn-1",
                    "items": [],
                    "itemsView": "notLoaded",
                    "status": "inProgress",
                    "error": null,
                    "startedAt": null,
                    "completedAt": null,
                    "durationMs": null
                }}
            }),
        ] {
            send_message(&mut server_write, &message)
                .await
                .expect("write pre-response frame");
        }

        let mut buffered = BufferedNotifications::default();
        read_response_buffering(
            &mut client_read,
            &mut client_write,
            TURN_START_ID,
            &tx,
            &mut buffered,
        )
        .await
        .expect("read turn/start response");
        consume_turn_with_pending(
            &mut client_read,
            &mut client_write,
            "thread-1",
            "turn-1",
            buffered.messages,
            &tx,
        )
        .await
        .expect("foreign reroute must not fail matching turn");
        assert!(matches!(rx.try_recv(), Ok(ChatEvent::Completed { .. })));
        assert!(rx.try_recv().is_err());

        let mut written = [0_u8; 1];
        assert!(
            tokio::time::timeout(Duration::from_millis(20), server_read.read(&mut written))
                .await
                .is_err(),
            "foreign reroute must not send interrupt"
        );
    }

    #[tokio::test]
    async fn pre_response_nonretry_error_wins_over_nominal_completion() {
        let (client, mut server) = duplex(16 * 1024);
        let (client_read, mut client_write) = split(client);
        let mut client_read = BufReader::new(client_read);
        let (tx, mut rx) = unbounded_channel();

        for message in [
            json!({
                "method": "error",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "error": {
                        "message": "terminal transport failure",
                        "codexErrorInfo": null,
                        "additionalDetails": null,
                        "misalignment": null
                    },
                    "willRetry": false
                }
            }),
            json!({
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": "turn-1",
                        "items": [],
                        "itemsView": "notLoaded",
                        "status": "completed",
                        "error": null,
                        "startedAt": 1,
                        "completedAt": 2,
                        "durationMs": 1000
                    }
                }
            }),
            json!({
                "id": TURN_START_ID,
                "result": {"turn": {
                    "id": "turn-1",
                    "items": [],
                    "itemsView": "notLoaded",
                    "status": "inProgress",
                    "error": null,
                    "startedAt": null,
                    "completedAt": null,
                    "durationMs": null
                }}
            }),
        ] {
            send_message(&mut server, &message)
                .await
                .expect("write pre-response frame");
        }

        let mut buffered = BufferedNotifications::default();
        read_response_buffering(
            &mut client_read,
            &mut client_write,
            TURN_START_ID,
            &tx,
            &mut buffered,
        )
        .await
        .expect("read turn/start response");
        consume_turn_with_pending(
            &mut client_read,
            &mut client_write,
            "thread-1",
            "turn-1",
            buffered.messages,
            &tx,
        )
        .await
        .expect("consume terminal error");

        assert!(matches!(
            rx.try_recv(),
            Ok(ChatEvent::Error(message)) if message == "terminal transport failure"
        ));
        assert!(rx.try_recv().is_err(), "must not also emit Completed");
    }

    #[test]
    fn pre_response_notification_buffer_is_count_and_size_bounded() {
        let mut buffered = BufferedNotifications::default();
        for _ in 0..MAX_PRE_RESPONSE_NOTIFICATIONS {
            buffered
                .push(json!({"method": "ignored", "params": {}}))
                .expect("notification within count limit");
        }
        let error = buffered
            .push(json!({"method": "ignored", "params": {}}))
            .expect_err("notification count above limit must fail closed");
        assert!(error.to_string().contains("more than"));

        let mut buffered = BufferedNotifications {
            encoded_bytes: MAX_PRE_RESPONSE_NOTIFICATION_BYTES,
            ..Default::default()
        };
        let error = buffered
            .push(json!({"method": "ignored", "params": {}}))
            .expect_err("notification bytes above limit must fail closed");
        assert!(error.to_string().contains("exceeded"));
    }
}

// The transport remains dormant until the provider and session layers can
// persist thread ownership and uncertain-delivery state.
#[allow(dead_code)]
pub(super) mod persistent;
