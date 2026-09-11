//! Codex app-server protocol primitives shared by isolated one-shot advisory
//! runs and conversation-scoped persistent interactive sessions.
//!
//! Both paths use the same fail-closed initialization/thread attestation and
//! bounded event decoding before a model-bearing turn is accepted.

use super::{ChatEvent, Usage};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
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
const MAX_TURN_ASSISTANT_TEXT_BYTES: usize = 2 * MAX_FRAME_BYTES;
const MAX_TURN_ASSISTANT_TEXT_CHARS: usize = MAX_FRAME_BYTES;
const MAX_TURN_TRACKED_ITEM_IDS: usize = 4_096;
const MAX_TURN_ITEM_ID_BYTES: usize = 4 * 1_024;
const MAX_TURN_ITEM_TYPE_BYTES: usize = 256;
const MAX_TURN_ACTIVITY_ITEMS: usize = 1_024;

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
    pub(super) model: String,
    pub(super) cwd: PathBuf,
    pub(super) workspace_roots: Vec<PathBuf>,
    pub(super) codex_home: PathBuf,
    pub(super) sandbox: ContractSandbox,
    pub(super) developer_instructions: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ContractSandbox {
    AdvisoryProfile {
        profile: String,
    },
    WorkspaceWrite {
        profile: String,
        writable_roots: Vec<PathBuf>,
    },
    DangerFullAccess,
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
    send_message(stdin, &thread_start_request(THREAD_START_ID, contract)?).await?;
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

pub(super) fn thread_start_params(contract: &Contract) -> Result<Value> {
    let cwd = contract
        .cwd
        .to_str()
        .context("Codex app-server cwd is not valid UTF-8")?;
    let workspace_roots = contract
        .workspace_roots
        .iter()
        .map(|root| {
            root.to_str()
                .map(str::to_owned)
                .context("Codex app-server workspace root is not valid UTF-8")
        })
        .collect::<Result<Vec<_>>>()?;
    let mut params = serde_json::json!({
        "model": contract.model,
        "modelProvider": "openai",
        "allowProviderModelFallback": false,
        "cwd": cwd,
        "runtimeWorkspaceRoots": workspace_roots,
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "ephemeral": true,
        "historyMode": "legacy",
        "environments": [{
            "environmentId": "local",
            "cwd": cwd,
            "runtimeWorkspaceRoots": workspace_roots
        }],
        "dynamicTools": [],
        "selectedCapabilityRoots": [],
        "experimentalRawEvents": false
    });
    let params = params
        .as_object_mut()
        .expect("thread/start params are constructed as an object");
    match &contract.sandbox {
        ContractSandbox::AdvisoryProfile { profile } => {
            params.insert("permissions".into(), Value::String(profile.clone()));
        }
        ContractSandbox::WorkspaceWrite { profile, .. } => {
            params.insert("permissions".into(), Value::String(profile.clone()));
        }
        ContractSandbox::DangerFullAccess => {
            params.insert("sandbox".into(), Value::String("danger-full-access".into()));
        }
    }
    if let Some(instructions) = contract
        .developer_instructions
        .as_ref()
        .filter(|instructions| !instructions.is_empty())
    {
        params.insert(
            "developerInstructions".into(),
            Value::String(instructions.clone()),
        );
    }
    Ok(Value::Object(std::mem::take(params)))
}

fn thread_start_request(request_id: i64, contract: &Contract) -> Result<Value> {
    Ok(serde_json::json!({
        "id": request_id,
        "method": "thread/start",
        "params": thread_start_params(contract)?,
    }))
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
                "message": "Shaltaiboltai app-server transport does not service server requests"
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
        .context("Codex app-server cwd is not valid UTF-8")?;
    let expected_roots = contract
        .workspace_roots
        .iter()
        .map(|root| {
            root.to_str()
                .map(str::to_owned)
                .context("Codex reviewed workspace root is not valid UTF-8")
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
    let active_permission_profile = result
        .get("activePermissionProfile")
        .context("Codex thread/start response omitted activePermissionProfile")?;
    let sandbox = result
        .get("sandbox")
        .context("Codex thread/start response omitted sandbox")?;
    let (permission_matches, expected_sandbox) = match &contract.sandbox {
        ContractSandbox::AdvisoryProfile { profile } => {
            let matches = active_permission_profile.as_object().is_some_and(|active| {
                active.get("id").and_then(Value::as_str) == Some(profile.as_str())
                    && active.get("extends").is_some_and(Value::is_null)
            });
            (
                matches,
                serde_json::json!({"type": "readOnly", "networkAccess": false}),
            )
        }
        ContractSandbox::WorkspaceWrite {
            profile,
            writable_roots,
        } => {
            let writable_roots = writable_roots
                .iter()
                .map(|root| {
                    root.to_str()
                        .map(str::to_owned)
                        .context("Codex workspace-write root is not valid UTF-8")
                })
                .collect::<Result<Vec<_>>>()?;
            (
                active_permission_profile.as_object().is_some_and(|active| {
                    active.get("id").and_then(Value::as_str) == Some(profile.as_str())
                        && active.get("extends").is_some_and(Value::is_null)
                }),
                serde_json::json!({
                    "type": "workspaceWrite",
                    "writableRoots": writable_roots,
                    "networkAccess": false,
                    "excludeTmpdirEnvVar": true,
                    "excludeSlashTmp": true
                }),
            )
        }
        ContractSandbox::DangerFullAccess => (
            active_permission_profile.is_null(),
            serde_json::json!({"type": "dangerFullAccess"}),
        ),
    };

    let matches_contract = result.get("model").and_then(Value::as_str)
        == Some(contract.model.as_str())
        && result.get("modelProvider").and_then(Value::as_str) == Some("openai")
        && result.get("cwd").and_then(Value::as_str) == Some(cwd)
        && actual_roots == expected_roots
        && result.get("approvalPolicy").and_then(Value::as_str) == Some("never")
        && result.get("approvalsReviewer").and_then(Value::as_str) == Some("user")
        && permission_matches
        && sandbox == &expected_sandbox
        && thread.get("cliVersion").and_then(Value::as_str) == Some(VERSION)
        && thread.get("ephemeral").and_then(Value::as_bool) == Some(true)
        && thread.get("path").is_some_and(Value::is_null)
        && thread.get("historyMode").and_then(Value::as_str) == Some("legacy")
        && thread.get("modelProvider").and_then(Value::as_str) == Some("openai")
        && thread.get("model").and_then(Value::as_str) == Some(contract.model.as_str())
        && thread.get("cwd").and_then(Value::as_str) == Some(cwd)
        && thread.get("canAcceptDirectInput").and_then(Value::as_bool) == Some(true);
    if !matches_contract {
        anyhow::bail!("Codex thread/start attestation did not match the requested contract");
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
            anyhow::bail!("Codex instruction source escaped the reviewed workspace: {source}");
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
    let mut accumulator = TurnAccumulator::new(thread_id, turn_id);
    loop {
        let message = match pending.pop_front() {
            Some(message) => message,
            None => read_message(stdout).await?,
        };
        if message.get("method").is_some() && message.get("id").is_some() {
            reject_server_request(stdin, &message).await?;
            continue;
        }
        match accumulator.consume_notification(&message, tx)? {
            TurnNotification::Continue => {}
            TurnNotification::ModelRerouted {
                from_model,
                to_model,
            } => {
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
                    "Codex rerouted the turn from {from_model} to {to_model}; exact-model contract failed"
                );
            }
            TurnNotification::Terminal(status) => {
                accumulator.finish_terminal(status.as_str(), tx)?;
                return Ok(());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnTerminalStatus {
    Completed,
    Failed,
    Interrupted,
}

impl TurnTerminalStatus {
    fn parse(status: Option<&str>) -> Result<Self> {
        match status {
            Some("completed") => Ok(Self::Completed),
            Some("failed") => Ok(Self::Failed),
            Some("interrupted") => Ok(Self::Interrupted),
            _ => anyhow::bail!("Codex turn/completed had an invalid status"),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum TurnNotification {
    Continue,
    Terminal(TurnTerminalStatus),
    ModelRerouted {
        from_model: String,
        to_model: String,
    },
}

/// Stateful decoder for the notifications belonging to one Codex turn.
///
/// The persistent transport delivers the raw `Notification` frames before a
/// separate terminal-status event. Keeping finalization separate lets that
/// transport preserve its delivery guarantees while the one-shot path uses
/// this exact same decoder and finalizer.
#[derive(Debug)]
pub(super) struct TurnAccumulator {
    thread_id: String,
    turn_id: String,
    usage: Option<Usage>,
    tracked_item_types: HashMap<String, String>,
    streamed_agent_text: HashMap<String, String>,
    emitted_agent_items: HashSet<String>,
    emitted_activity_items: HashSet<String>,
    terminal_error: Option<String>,
    assistant_text: String,
    assistant_text_chars: usize,
    observed_terminal_status: Option<TurnTerminalStatus>,
    finalized: bool,
}

impl TurnAccumulator {
    pub(super) fn new(thread_id: &str, turn_id: &str) -> Self {
        Self {
            thread_id: thread_id.to_owned(),
            turn_id: turn_id.to_owned(),
            usage: None,
            tracked_item_types: HashMap::new(),
            streamed_agent_text: HashMap::new(),
            emitted_agent_items: HashSet::new(),
            emitted_activity_items: HashSet::new(),
            terminal_error: None,
            assistant_text: String::new(),
            assistant_text_chars: 0,
            observed_terminal_status: None,
            finalized: false,
        }
    }

    pub(super) fn assistant_text(&self) -> &str {
        &self.assistant_text
    }

    pub(super) fn terminal_error(&self) -> Option<&str> {
        self.terminal_error.as_deref()
    }

    pub(super) fn consume_notification(
        &mut self,
        message: &Value,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<TurnNotification> {
        if self.finalized {
            anyhow::bail!("Codex app-server emitted a notification after terminal delivery");
        }
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            anyhow::bail!("Codex app-server emitted an unexpected response during a turn");
        };
        let params = &message["params"];
        match method {
            "item/agentMessage/delta" => {
                self.consume_agent_message_delta(params, tx)?;
            }
            "item/completed" => {
                self.consume_item_completed(params, tx)?;
            }
            "thread/tokenUsage/updated" if self.notification_matches(params) => {
                self.usage = usage_from(params);
            }
            "warning" => {
                let applies = params
                    .get("threadId")
                    .and_then(Value::as_str)
                    .is_none_or(|id| id == self.thread_id.as_str());
                if applies {
                    emit_notice(message, tx);
                }
            }
            "guardianWarning"
                if params.get("threadId").and_then(Value::as_str)
                    == Some(self.thread_id.as_str()) =>
            {
                emit_notice(message, tx);
            }
            "modelProvider/authRecoveryStarted" | "modelProvider/authRecoveryCompleted"
                if self.notification_matches(params) =>
            {
                emit_notice(message, tx);
            }
            "configWarning" | "deprecationNotice" => {
                emit_notice(message, tx);
            }
            "error" if self.notification_matches(params) => {
                let detail = turn_error_message(&params["error"]);
                if params.get("willRetry").and_then(Value::as_bool) == Some(true) {
                    let _ = tx.send(ChatEvent::Notice(detail));
                } else {
                    self.terminal_error = Some(detail);
                }
            }
            "model/rerouted" if self.notification_matches(params) => {
                return Ok(TurnNotification::ModelRerouted {
                    from_model: params
                        .get("fromModel")
                        .and_then(Value::as_str)
                        .unwrap_or("requested model")
                        .to_owned(),
                    to_model: params
                        .get("toModel")
                        .and_then(Value::as_str)
                        .unwrap_or("another model")
                        .to_owned(),
                });
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
                if completed_thread_id != self.thread_id.as_str() {
                    return Ok(TurnNotification::Continue);
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
                if completed_turn_id != self.turn_id.as_str() {
                    return Ok(TurnNotification::Continue);
                }
                if let Some(items) = turn.get("items") {
                    let items = items
                        .as_array()
                        .context("Codex turn/completed had invalid turn.items")?;
                    for item in items {
                        self.consume_item(item, tx, false)?;
                    }
                }
                let status = TurnTerminalStatus::parse(turn.get("status").and_then(Value::as_str))?;
                if status == TurnTerminalStatus::Failed {
                    if let Some(error) = turn
                        .get("error")
                        .map(turn_error_message)
                        .filter(|message| !message.is_empty())
                    {
                        // Preserve the one-shot decoder's precedence: the
                        // turn's terminal error wins over an earlier nonretry
                        // error notification whenever the field is present.
                        self.terminal_error = Some(error);
                    }
                }
                if self
                    .observed_terminal_status
                    .is_some_and(|observed| observed != status)
                {
                    anyhow::bail!("Codex turn/completed changed terminal status");
                }
                self.observed_terminal_status = Some(status);
                return Ok(TurnNotification::Terminal(status));
            }
            _ => {}
        }
        Ok(TurnNotification::Continue)
    }

    /// Finalize from a persistent transport's terminal-status event.
    pub(super) fn finish_terminal(
        &mut self,
        status: &str,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        let status = TurnTerminalStatus::parse(Some(status))?;
        self.finish_terminal_status(status, tx)
    }

    fn finish_terminal_status(
        &mut self,
        status: TurnTerminalStatus,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        if self.finalized {
            anyhow::bail!("Codex turn terminal status was delivered more than once");
        }
        if self
            .observed_terminal_status
            .is_some_and(|observed| observed != status)
        {
            anyhow::bail!(
                "Codex terminal status {} did not match turn/completed status {}",
                status.as_str(),
                self.observed_terminal_status
                    .expect("mismatched observed status checked above")
                    .as_str()
            );
        }
        self.finalized = true;
        match status {
            TurnTerminalStatus::Completed if self.terminal_error.is_none() => {
                let _ = tx.send(ChatEvent::Completed {
                    tool_calls: Vec::new(),
                    stop_reason: Some("stop".into()),
                    usage: self.usage,
                });
            }
            TurnTerminalStatus::Failed => {
                let _ = tx.send(ChatEvent::Error(
                    self.terminal_error
                        .clone()
                        .unwrap_or_else(|| "Codex turn failed".into()),
                ));
            }
            TurnTerminalStatus::Interrupted => {
                let _ = tx.send(ChatEvent::Error(
                    self.terminal_error
                        .clone()
                        .unwrap_or_else(|| "Codex turn was interrupted".into()),
                ));
            }
            TurnTerminalStatus::Completed => {
                let _ = tx.send(ChatEvent::Error(
                    self.terminal_error
                        .clone()
                        .unwrap_or_else(|| "Codex turn failed".into()),
                ));
            }
        }
        Ok(())
    }

    fn notification_matches(&self, params: &Value) -> bool {
        params.get("threadId").and_then(Value::as_str) == Some(self.thread_id.as_str())
            && params.get("turnId").and_then(Value::as_str) == Some(self.turn_id.as_str())
    }

    fn consume_agent_message_delta(
        &mut self,
        params: &Value,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        let params = params
            .as_object()
            .context("Codex item/agentMessage/delta had invalid params")?;
        let thread_id = params
            .get("threadId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("Codex item/agentMessage/delta had invalid threadId")?;
        let turn_id = params
            .get("turnId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("Codex item/agentMessage/delta had invalid turnId")?;
        let item_id = params
            .get("itemId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("Codex item/agentMessage/delta had invalid itemId")?;
        Self::validate_item_id(item_id)?;
        let delta = params
            .get("delta")
            .and_then(Value::as_str)
            .context("Codex item/agentMessage/delta had invalid delta")?;
        if thread_id != self.thread_id || turn_id != self.turn_id {
            return Ok(());
        }

        self.track_item_identity(item_id, "agentMessage")?;
        if self.emitted_agent_items.contains(item_id) {
            anyhow::bail!("Codex emitted an agent-message delta after item completion");
        }
        if delta.is_empty() {
            return Ok(());
        }
        self.append_assistant_text(delta)?;
        self.streamed_agent_text
            .entry(item_id.to_owned())
            .or_default()
            .push_str(delta);
        let _ = tx.send(ChatEvent::TextDelta(delta.to_owned()));
        Ok(())
    }

    fn consume_item_completed(
        &mut self,
        params: &Value,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        let params = params
            .as_object()
            .context("Codex item/completed had invalid params")?;
        let thread_id = params
            .get("threadId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("Codex item/completed had invalid threadId")?;
        let turn_id = params
            .get("turnId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("Codex item/completed had invalid turnId")?;
        let item = params
            .get("item")
            .context("Codex item/completed omitted item")?;
        if thread_id != self.thread_id || turn_id != self.turn_id {
            return Ok(());
        }
        self.consume_item(item, tx, true)
    }

    fn consume_item(
        &mut self,
        item: &Value,
        tx: &UnboundedSender<ChatEvent>,
        forward_activity: bool,
    ) -> Result<()> {
        let object = item
            .as_object()
            .context("Codex completed item was not an object")?;
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .filter(|item_type| !item_type.is_empty())
            .context("Codex completed item had an invalid type")?;
        if item_type.len() > MAX_TURN_ITEM_TYPE_BYTES {
            anyhow::bail!("Codex completed item type exceeded {MAX_TURN_ITEM_TYPE_BYTES} bytes");
        }
        let item_id = object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("Codex completed item had an invalid id")?;
        Self::validate_item_id(item_id)?;
        self.track_item_identity(item_id, item_type)?;

        match item_type {
            "agentMessage" => {
                let text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .context("Codex agentMessage item had invalid text")?;
                self.complete_agent_message(item_id, text, tx)
            }
            "reasoning" | "plan" | "userMessage" | "hookPrompt" => Ok(()),
            _ if forward_activity => self.forward_activity(item_id, item, tx),
            _ => Ok(()),
        }
    }

    fn validate_item_id(item_id: &str) -> Result<()> {
        if item_id.len() > MAX_TURN_ITEM_ID_BYTES {
            anyhow::bail!("Codex turn item id exceeded {MAX_TURN_ITEM_ID_BYTES} bytes");
        }
        Ok(())
    }

    fn track_item_identity(&mut self, item_id: &str, item_type: &str) -> Result<()> {
        if let Some(previous_type) = self.tracked_item_types.get(item_id) {
            if previous_type != item_type {
                anyhow::bail!(
                    "Codex turn item {item_id:?} changed type from {previous_type:?} to {item_type:?}"
                );
            }
            return Ok(());
        }
        if self.tracked_item_types.len() >= MAX_TURN_TRACKED_ITEM_IDS {
            anyhow::bail!(
                "Codex turn exceeded the {MAX_TURN_TRACKED_ITEM_IDS}-item identity limit"
            );
        }
        self.tracked_item_types
            .insert(item_id.to_owned(), item_type.to_owned());
        Ok(())
    }

    fn append_assistant_text(&mut self, text: &str) -> Result<()> {
        let next_bytes = self
            .assistant_text
            .len()
            .checked_add(text.len())
            .context("Codex turn assistant text byte count overflowed")?;
        if next_bytes > MAX_TURN_ASSISTANT_TEXT_BYTES {
            anyhow::bail!(
                "Codex turn assistant text exceeded {MAX_TURN_ASSISTANT_TEXT_BYTES} bytes"
            );
        }
        let text_chars = text.chars().count();
        let next_chars = self
            .assistant_text_chars
            .checked_add(text_chars)
            .context("Codex turn assistant text character count overflowed")?;
        if next_chars > MAX_TURN_ASSISTANT_TEXT_CHARS {
            anyhow::bail!(
                "Codex turn assistant text exceeded {MAX_TURN_ASSISTANT_TEXT_CHARS} characters"
            );
        }
        self.assistant_text.push_str(text);
        self.assistant_text_chars = next_chars;
        Ok(())
    }

    fn complete_agent_message(
        &mut self,
        item_id: &str,
        final_text: &str,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        let streamed_text = self
            .streamed_agent_text
            .get(item_id)
            .map(String::as_str)
            .unwrap_or("");
        if self.emitted_agent_items.contains(item_id) {
            if streamed_text != final_text {
                anyhow::bail!("Codex agentMessage text changed after item completion");
            }
            return Ok(());
        }
        if !final_text.starts_with(streamed_text) {
            anyhow::bail!("Codex agentMessage final text did not start with its streamed text");
        }
        let remaining = final_text[streamed_text.len()..].to_owned();
        if !remaining.is_empty() {
            self.append_assistant_text(&remaining)?;
        }
        self.streamed_agent_text
            .insert(item_id.to_owned(), final_text.to_owned());
        self.emitted_agent_items.insert(item_id.to_owned());
        if !remaining.is_empty() {
            let _ = tx.send(ChatEvent::TextDelta(remaining));
        }
        Ok(())
    }

    fn forward_activity(
        &mut self,
        item_id: &str,
        item: &Value,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        if self.emitted_activity_items.contains(item_id) {
            return Ok(());
        }
        if self.emitted_activity_items.len() >= MAX_TURN_ACTIVITY_ITEMS {
            anyhow::bail!(
                "Codex turn exceeded the {MAX_TURN_ACTIVITY_ITEMS}-activity forwarding limit"
            );
        }
        let is_error = item
            .get("exitCode")
            .and_then(Value::as_i64)
            .is_some_and(|code| code != 0)
            || matches!(
                item.get("status").and_then(Value::as_str),
                Some("failed" | "declined")
            );
        self.emitted_activity_items.insert(item_id.to_owned());
        let _ = tx.send(ChatEvent::ToolActivity {
            summary: summarize_item(item),
            is_error,
        });
        Ok(())
    }
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
        .unwrap_or("Codex turn failed")
        .to_owned()
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

    fn test_contract(sandbox: ContractSandbox) -> Contract {
        Contract {
            model: "gpt-5.6-sol".into(),
            cwd: PathBuf::from("/workspace/project"),
            workspace_roots: vec![PathBuf::from("/workspace/project")],
            codex_home: PathBuf::from("/isolated/codex-home"),
            sandbox,
            developer_instructions: None,
        }
    }

    #[test]
    fn workspace_write_thread_start_uses_permissions_and_omits_sandbox() {
        let contract = test_contract(ContractSandbox::WorkspaceWrite {
            profile: "shaltai-workspace-write".into(),
            writable_roots: vec![PathBuf::from("/workspace/shared")],
        });
        let params = thread_start_params(&contract).expect("build workspace-write thread/start");

        assert_eq!(params["permissions"], "shaltai-workspace-write");
        assert!(
            params.get("sandbox").is_none(),
            "permission profiles and legacy sandbox must never be combined"
        );
    }

    #[test]
    fn workspace_write_attestation_requires_profile_and_exact_effective_sandbox() {
        let contract = test_contract(ContractSandbox::WorkspaceWrite {
            profile: "shaltai-workspace-write".into(),
            writable_roots: vec![PathBuf::from("/workspace/shared")],
        });
        let result = json!({
            "thread": {
                "id": "thread-1",
                "cliVersion": VERSION,
                "ephemeral": true,
                "path": null,
                "historyMode": "legacy",
                "modelProvider": "openai",
                "model": contract.model,
                "cwd": contract.cwd,
                "canAcceptDirectInput": true
            },
            "model": contract.model,
            "modelProvider": "openai",
            "cwd": contract.cwd,
            "runtimeWorkspaceRoots": contract.workspace_roots,
            "instructionSources": [],
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "sandbox": {
                "type": "workspaceWrite",
                "writableRoots": ["/workspace/shared"],
                "networkAccess": false,
                "excludeTmpdirEnvVar": true,
                "excludeSlashTmp": true
            },
            "activePermissionProfile": {
                "id": "shaltai-workspace-write",
                "extends": null
            }
        });
        assert_eq!(
            attest_thread(&result, &contract).expect("attest exact workspace-write contract"),
            "thread-1"
        );

        let mut missing_profile = result.clone();
        missing_profile["activePermissionProfile"] = Value::Null;
        assert!(
            attest_thread(&missing_profile, &contract).is_err(),
            "workspace-write must attest the requested isolated profile"
        );

        let mut inherited_profile = result.clone();
        inherited_profile["activePermissionProfile"]["extends"] = Value::String("base".into());
        assert!(
            attest_thread(&inherited_profile, &contract).is_err(),
            "workspace-write must reject an inherited permission profile"
        );

        let mut wrong_sandbox = result;
        wrong_sandbox["sandbox"]["networkAccess"] = Value::Bool(true);
        assert!(
            attest_thread(&wrong_sandbox, &contract).is_err(),
            "workspace-write effective sandbox must match exactly"
        );
    }

    #[test]
    fn advisory_and_danger_full_access_request_shapes_are_unchanged() {
        let advisory = thread_start_params(&test_contract(ContractSandbox::AdvisoryProfile {
            profile: "shaltai-read-only".into(),
        }))
        .expect("build advisory thread/start");
        assert_eq!(advisory["permissions"], "shaltai-read-only");
        assert!(advisory.get("sandbox").is_none());

        let danger = thread_start_params(&test_contract(ContractSandbox::DangerFullAccess))
            .expect("build danger-full-access thread/start");
        assert_eq!(danger["sandbox"], "danger-full-access");
        assert!(danger.get("permissions").is_none());
    }

    #[test]
    fn turn_accumulator_tracks_text_usage_and_deduplicates_terminal_items() {
        let (tx, mut rx) = unbounded_channel();
        let mut accumulator = TurnAccumulator::new("thread-1", "turn-1");

        let item = json!({
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "agentMessage", "id": "answer-1", "text": "hello"}
            }
        });
        assert_eq!(
            accumulator
                .consume_notification(&item, &tx)
                .expect("consume agent item"),
            TurnNotification::Continue
        );
        accumulator
            .consume_notification(&item, &tx)
            .expect("consume duplicate agent item");
        accumulator
            .consume_notification(
                &json!({
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "tokenUsage": {"total": {"inputTokens": 13, "outputTokens": 8}}
                    }
                }),
                &tx,
            )
            .expect("consume token usage");

        let terminal = accumulator
            .consume_notification(
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {
                            "id": "turn-1",
                            "status": "completed",
                            "error": null,
                            "items": [
                                {"type": "agentMessage", "id": "answer-1", "text": "hello"},
                                {"type": "agentMessage", "id": "answer-2", "text": " world"}
                            ]
                        }
                    }
                }),
                &tx,
            )
            .expect("consume terminal notification");
        assert_eq!(
            terminal,
            TurnNotification::Terminal(TurnTerminalStatus::Completed)
        );
        assert_eq!(accumulator.assistant_text(), "hello world");
        assert_eq!(accumulator.emitted_agent_items.len(), 2);
        let usage = accumulator.usage.expect("usage was accumulated");
        assert_eq!(usage.input_tokens, 13);
        assert_eq!(usage.output_tokens, 8);
        assert_eq!(accumulator.terminal_error(), None);

        accumulator
            .finish_terminal("completed", &tx)
            .expect("finish completed turn");
        let events = std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>();
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], ChatEvent::TextDelta(text) if text == "hello"));
        assert!(matches!(&events[1], ChatEvent::TextDelta(text) if text == " world"));
        assert!(matches!(
            &events[2],
            ChatEvent::Completed {
                usage: Some(Usage {
                    input_tokens: 13,
                    output_tokens: 8
                }),
                ..
            }
        ));
    }

    #[test]
    fn agent_message_deltas_stream_and_completion_emits_only_remaining_suffix() {
        let (tx, mut rx) = unbounded_channel();
        let mut accumulator = TurnAccumulator::new("thread-1", "turn-1");
        for delta in ["hel", "lo"] {
            assert_eq!(
                accumulator
                    .consume_notification(
                        &json!({
                            "method": "item/agentMessage/delta",
                            "params": {
                                "threadId": "thread-1",
                                "turnId": "turn-1",
                                "itemId": "answer-1",
                                "delta": delta
                            }
                        }),
                        &tx,
                    )
                    .expect("consume agent-message delta"),
                TurnNotification::Continue
            );
        }
        let completed = json!({
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "agentMessage",
                    "id": "answer-1",
                    "text": "hello world"
                }
            }
        });
        accumulator
            .consume_notification(&completed, &tx)
            .expect("complete streamed agent message");
        accumulator
            .consume_notification(&completed, &tx)
            .expect("deduplicate repeated completion");

        assert_eq!(accumulator.assistant_text(), "hello world");
        assert!(accumulator.emitted_agent_items.contains("answer-1"));
        let events = std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>();
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], ChatEvent::TextDelta(text) if text == "hel"));
        assert!(matches!(&events[1], ChatEvent::TextDelta(text) if text == "lo"));
        assert!(matches!(&events[2], ChatEvent::TextDelta(text) if text == " world"));
    }

    #[test]
    fn agent_message_completion_rejects_divergent_streamed_text() {
        let (tx, mut rx) = unbounded_channel();
        let mut accumulator = TurnAccumulator::new("thread-1", "turn-1");
        accumulator
            .consume_notification(
                &json!({
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "itemId": "answer-1",
                        "delta": "streamed prefix"
                    }
                }),
                &tx,
            )
            .expect("consume agent-message delta");
        let error = accumulator
            .consume_notification(
                &json!({
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "type": "agentMessage",
                            "id": "answer-1",
                            "text": "different final text"
                        }
                    }
                }),
                &tx,
            )
            .expect_err("divergent authoritative text must fail closed");
        assert!(error.to_string().contains("did not start"));
        assert!(matches!(
            rx.try_recv(),
            Ok(ChatEvent::TextDelta(text)) if text == "streamed prefix"
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn agent_message_notifications_require_valid_ids_text_and_stable_types() {
        let (tx, _rx) = unbounded_channel();

        let mut missing_id = TurnAccumulator::new("thread-1", "turn-1");
        let error = missing_id
            .consume_notification(
                &json!({
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "delta": "text"
                    }
                }),
                &tx,
            )
            .expect_err("delta without itemId must fail closed");
        assert!(error.to_string().contains("invalid itemId"));

        let mut missing_text = TurnAccumulator::new("thread-1", "turn-1");
        let error = missing_text
            .consume_notification(
                &json!({
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "id": "answer-1"}
                    }
                }),
                &tx,
            )
            .expect_err("agent item without text must fail closed");
        assert!(error.to_string().contains("invalid text"));

        let mut changed_type = TurnAccumulator::new("thread-1", "turn-1");
        changed_type
            .consume_notification(
                &json!({
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "itemId": "answer-1",
                        "delta": "text"
                    }
                }),
                &tx,
            )
            .expect("consume valid delta");
        let error = changed_type
            .consume_notification(
                &json!({
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "type": "commandExecution",
                            "id": "answer-1",
                            "command": "true"
                        }
                    }
                }),
                &tx,
            )
            .expect_err("one item id cannot change protocol type");
        assert!(error.to_string().contains("changed type"));

        let mut oversized_id = TurnAccumulator::new("thread-1", "turn-1");
        let error = oversized_id
            .consume_notification(
                &json!({
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "itemId": "x".repeat(MAX_TURN_ITEM_ID_BYTES + 1),
                        "delta": "text"
                    }
                }),
                &tx,
            )
            .expect_err("oversized item id must fail closed");
        assert!(error.to_string().contains("item id exceeded"));
    }

    #[test]
    fn turn_accumulator_enforces_text_identity_and_activity_caps() {
        let (tx, _rx) = unbounded_channel();

        let mut char_limited = TurnAccumulator::new("thread-1", "turn-1");
        char_limited.assistant_text = "x".repeat(MAX_TURN_ASSISTANT_TEXT_CHARS);
        char_limited.assistant_text_chars = MAX_TURN_ASSISTANT_TEXT_CHARS;
        let error = char_limited
            .append_assistant_text("x")
            .expect_err("assistant character cap must be strict");
        assert!(error.to_string().contains("characters"));

        let mut byte_limited = TurnAccumulator::new("thread-1", "turn-1");
        byte_limited.assistant_text = "界".repeat(MAX_TURN_ASSISTANT_TEXT_BYTES / "界".len());
        byte_limited.assistant_text_chars = byte_limited.assistant_text.chars().count();
        let error = byte_limited
            .append_assistant_text("界")
            .expect_err("assistant byte cap must be strict");
        assert!(error.to_string().contains("bytes"));

        let mut identity_limited = TurnAccumulator::new("thread-1", "turn-1");
        identity_limited.tracked_item_types = (0..MAX_TURN_TRACKED_ITEM_IDS)
            .map(|index| (format!("item-{index}"), "reasoning".to_owned()))
            .collect();
        let error = identity_limited
            .track_item_identity("one-too-many", "agentMessage")
            .expect_err("tracked item identity cap must be strict");
        assert!(error.to_string().contains("identity limit"));

        let mut activity_limited = TurnAccumulator::new("thread-1", "turn-1");
        activity_limited.emitted_activity_items = (0..MAX_TURN_ACTIVITY_ITEMS)
            .map(|index| format!("activity-{index}"))
            .collect();
        let error = activity_limited
            .consume_notification(
                &json!({
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "type": "commandExecution",
                            "id": "one-too-many",
                            "command": "true",
                            "status": "completed"
                        }
                    }
                }),
                &tx,
            )
            .expect_err("activity forwarding cap must be strict");
        assert!(error.to_string().contains("activity forwarding limit"));
    }

    #[test]
    fn turn_accumulator_preserves_terminal_error_precedence() {
        let (tx, mut rx) = unbounded_channel();
        let mut accumulator = TurnAccumulator::new("thread-1", "turn-1");
        accumulator
            .consume_notification(
                &json!({
                    "method": "error",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "error": {"message": "earlier failure"},
                        "willRetry": false
                    }
                }),
                &tx,
            )
            .expect("consume nonretry error");
        let terminal = accumulator
            .consume_notification(
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {
                            "id": "turn-1",
                            "status": "failed",
                            "error": {
                                "message": "terminal failure",
                                "additionalDetails": "terminal detail"
                            },
                            "items": []
                        }
                    }
                }),
                &tx,
            )
            .expect("consume failed terminal notification");
        assert_eq!(
            terminal,
            TurnNotification::Terminal(TurnTerminalStatus::Failed)
        );
        assert_eq!(accumulator.terminal_error(), Some("terminal detail"));

        accumulator
            .finish_terminal("failed", &tx)
            .expect("finish failed turn");
        assert!(matches!(
            rx.try_recv(),
            Ok(ChatEvent::Error(message)) if message == "terminal detail"
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn turn_accumulator_rejects_terminal_status_divergence() {
        let (tx, mut rx) = unbounded_channel();
        let mut accumulator = TurnAccumulator::new("thread-1", "turn-1");
        accumulator
            .consume_notification(
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {
                            "id": "turn-1",
                            "status": "completed",
                            "error": null,
                            "items": []
                        }
                    }
                }),
                &tx,
            )
            .expect("consume completed terminal notification");

        let error = accumulator
            .finish_terminal("failed", &tx)
            .expect_err("divergent persistent terminal status must fail closed");
        assert!(error.to_string().contains("did not match"));
        assert!(rx.try_recv().is_err());
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

// Resume/reconnect/steer are fully bounded and regression-tested transport
// capabilities, but the current TUI activation intentionally uses only fresh
// thread start, turn start, cancellation, and shutdown.
#[allow(dead_code)]
pub(super) mod persistent;
