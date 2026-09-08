use super::{attest_initialize, read_message, reject_server_request, send_message, Contract};
use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;

const INITIALIZE_REQUEST_ID: i64 = 0;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(500);

pub(super) type BoxReader = Pin<Box<dyn AsyncRead + Send>>;
pub(super) type BoxWriter = Pin<Box<dyn AsyncWrite + Send>>;

/// Process-specific cleanup owned by a spawned connection. Production
/// implementations must reap the child and terminate its process group on
/// timeout; dropping the implementation must retain kill-on-drop safety.
pub(super) trait ChildControl: Send {
    fn shutdown(&mut self) -> BoxFuture<'_, Result<()>>;
}

pub(super) struct SpawnedAppServer {
    pub(super) reader: BoxReader,
    pub(super) writer: BoxWriter,
    pub(super) child: Box<dyn ChildControl>,
}

/// A factory instance represents one immutable executable/auth/authority
/// identity. Reconnects must launch that same identity, never ambient state.
pub(super) trait AppServerSpawner: Send + Sync {
    fn spawn(&self) -> BoxFuture<'static, Result<SpawnedAppServer>>;
}

#[derive(Debug)]
pub(super) enum SubmissionEvent {
    Started { turn_id: String },
    Notification(Value),
    Terminal { status: String },
    Failed(String),
    DeliveryUncertain(String),
}

pub(super) struct Submission {
    id: String,
    events: mpsc::UnboundedReceiver<SubmissionEvent>,
    command_tx: mpsc::UnboundedSender<Command>,
    cancelled: Arc<AtomicBool>,
    terminal: bool,
}

impl Submission {
    pub(super) fn id(&self) -> &str {
        &self.id
    }

    pub(super) async fn next_event(&mut self) -> Option<SubmissionEvent> {
        let event = self.events.recv().await;
        if matches!(
            event,
            Some(
                SubmissionEvent::Terminal { .. }
                    | SubmissionEvent::Failed(_)
                    | SubmissionEvent::DeliveryUncertain(_)
            )
        ) {
            self.terminal = true;
        }
        event
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if !self.terminal {
            self.cancelled.store(true, Ordering::Release);
            let _ = self.command_tx.send(Command::CancelSubmission {
                submission_id: self.id.clone(),
            });
        }
    }
}

#[derive(Clone)]
pub(super) struct AppServerHandle {
    command_tx: mpsc::UnboundedSender<Command>,
}

impl AppServerHandle {
    pub(super) async fn start_thread(&self, params: Value) -> Result<Value> {
        self.thread_rpc(ThreadRpcKind::Start, params).await
    }

    pub(super) async fn resume_thread(&self, params: Value) -> Result<Value> {
        self.thread_rpc(ThreadRpcKind::Resume, params).await
    }

    async fn thread_rpc(&self, kind: ThreadRpcKind, params: Value) -> Result<Value> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(Command::ThreadRpc {
                kind,
                params,
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("Codex app-server transport stopped"))?;
        reply_rx
            .await
            .context("Codex app-server transport dropped a thread response")?
            .map_err(anyhow::Error::msg)
    }

    /// Queue a model-bearing request and immediately return its cancellation
    /// lease. The lease exists before the actor attempts `turn/start`, so
    /// dropping a caller during the response race still requests interrupt.
    pub(super) fn start_turn(&self, thread_id: String, input: Vec<Value>) -> Result<Submission> {
        if thread_id.is_empty() {
            anyhow::bail!("Codex thread id must not be empty");
        }
        if input.is_empty() {
            anyhow::bail!("Codex turn input must not be empty");
        }
        let submission_id = format!("shaltaiboltai-{:032x}", rand::random::<u128>());
        let cancelled = Arc::new(AtomicBool::new(false));
        let (events_tx, events) = mpsc::unbounded_channel();
        self.command_tx
            .send(Command::StartTurn {
                submission_id: submission_id.clone(),
                thread_id,
                input,
                cancelled: Arc::clone(&cancelled),
                events: events_tx,
            })
            .map_err(|_| anyhow::anyhow!("Codex app-server transport stopped"))?;
        Ok(Submission {
            id: submission_id,
            events,
            command_tx: self.command_tx.clone(),
            cancelled,
            terminal: false,
        })
    }

    pub(super) async fn reconnect(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(Command::Reconnect { reply: reply_tx })
            .map_err(|_| anyhow::anyhow!("Codex app-server transport stopped"))?;
        reply_rx
            .await
            .context("Codex app-server transport dropped reconnect result")?
            .map_err(anyhow::Error::msg)
    }
}

pub(super) struct PersistentAppServer {
    handle: AppServerHandle,
    task: Option<JoinHandle<Result<()>>>,
}

impl PersistentAppServer {
    pub(super) async fn connect(
        spawner: Arc<dyn AppServerSpawner>,
        contract: Contract,
    ) -> Result<Self> {
        let connection = connect_once(&spawner, &contract).await?;
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let handle = AppServerHandle {
            command_tx: command_tx.clone(),
        };
        let task = tokio::spawn(
            Actor {
                spawner,
                contract,
                connection: Some(connection),
                next_request_id: 1,
                pending: HashMap::new(),
                active: None,
                shutdown: None,
            }
            .run(command_rx),
        );
        Ok(Self {
            handle,
            task: Some(task),
        })
    }

    pub(super) fn handle(&self) -> AppServerHandle {
        self.handle.clone()
    }

    pub(super) async fn shutdown(mut self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.handle
            .command_tx
            .send(Command::Shutdown {
                reply: Some(reply_tx),
            })
            .map_err(|_| anyhow::anyhow!("Codex app-server transport stopped"))?;
        reply_rx
            .await
            .context("Codex app-server transport dropped shutdown result")?
            .map_err(anyhow::Error::msg)?;
        if let Some(task) = self.task.take() {
            task.await
                .context("Codex app-server transport task panicked")??;
        }
        Ok(())
    }
}

impl Drop for PersistentAppServer {
    fn drop(&mut self) {
        if self.task.is_some() {
            let _ = self
                .handle
                .command_tx
                .send(Command::Shutdown { reply: None });
        }
    }
}

enum Command {
    ThreadRpc {
        kind: ThreadRpcKind,
        params: Value,
        reply: oneshot::Sender<std::result::Result<Value, String>>,
    },
    StartTurn {
        submission_id: String,
        thread_id: String,
        input: Vec<Value>,
        cancelled: Arc<AtomicBool>,
        events: mpsc::UnboundedSender<SubmissionEvent>,
    },
    CancelSubmission {
        submission_id: String,
    },
    Reconnect {
        reply: oneshot::Sender<std::result::Result<(), String>>,
    },
    Shutdown {
        reply: Option<oneshot::Sender<std::result::Result<(), String>>>,
    },
}

#[derive(Clone, Copy)]
enum ThreadRpcKind {
    Start,
    Resume,
}

impl ThreadRpcKind {
    fn method(self) -> &'static str {
        match self {
            Self::Start => "thread/start",
            Self::Resume => "thread/resume",
        }
    }
}

struct Connection {
    reader: BufReader<BoxReader>,
    writer: BoxWriter,
    child: Box<dyn ChildControl>,
}

enum PendingRequest {
    ThreadRpc(oneshot::Sender<std::result::Result<Value, String>>),
    TurnStart { submission_id: String },
    Interrupt { submission_id: String },
}

struct ActiveSubmission {
    submission_id: String,
    thread_id: String,
    turn_id: Option<String>,
    start_acknowledged: bool,
    cancel_requested: bool,
    interrupt_request_id: Option<i64>,
    interrupt_acknowledged: bool,
    terminal_status: Option<String>,
    events: mpsc::UnboundedSender<SubmissionEvent>,
}

struct ShutdownState {
    reply: Option<oneshot::Sender<std::result::Result<(), String>>>,
    deadline: Instant,
}

struct Actor {
    spawner: Arc<dyn AppServerSpawner>,
    contract: Contract,
    connection: Option<Connection>,
    next_request_id: i64,
    pending: HashMap<i64, PendingRequest>,
    active: Option<ActiveSubmission>,
    shutdown: Option<ShutdownState>,
}

enum ActorInput {
    Command(Option<Command>),
    Frame(Result<Value>),
    ShutdownDeadline,
}

impl Actor {
    async fn run(mut self, mut commands: mpsc::UnboundedReceiver<Command>) -> Result<()> {
        loop {
            let input = if let Some(connection) = self.connection.as_mut() {
                let shutdown_deadline = self
                    .shutdown
                    .as_ref()
                    .map(|shutdown| shutdown.deadline)
                    .unwrap_or_else(Instant::now);
                tokio::select! {
                    command = commands.recv() => ActorInput::Command(command),
                    frame = read_message(&mut connection.reader) => ActorInput::Frame(frame),
                    _ = tokio::time::sleep_until(shutdown_deadline), if self.shutdown.is_some() => {
                        ActorInput::ShutdownDeadline
                    }
                }
            } else if let Some(shutdown) = &self.shutdown {
                let deadline = shutdown.deadline;
                tokio::select! {
                    command = commands.recv() => ActorInput::Command(command),
                    _ = tokio::time::sleep_until(deadline) => ActorInput::ShutdownDeadline,
                }
            } else {
                ActorInput::Command(commands.recv().await)
            };

            match input {
                ActorInput::Command(Some(command)) => self.handle_command(command).await,
                ActorInput::Command(None) => {
                    self.begin_shutdown(None).await;
                }
                ActorInput::Frame(Ok(frame)) => self.handle_frame(frame).await,
                ActorInput::Frame(Err(error)) => {
                    self.connection_lost(format!("Codex app-server connection failed: {error:#}"))
                        .await;
                }
                ActorInput::ShutdownDeadline => {
                    self.connection_lost(
                        "Codex app-server shutdown timed out; delivery is uncertain".into(),
                    )
                    .await;
                }
            }

            if self.shutdown.is_some() && self.active.is_none() {
                self.finish_shutdown().await;
                return Ok(());
            }
        }
    }

    async fn handle_command(&mut self, command: Command) {
        match command {
            Command::ThreadRpc {
                kind,
                params,
                reply,
            } => self.start_thread_rpc(kind, params, reply).await,
            Command::StartTurn {
                submission_id,
                thread_id,
                input,
                cancelled,
                events,
            } => {
                self.start_turn(submission_id, thread_id, input, cancelled, events)
                    .await;
            }
            Command::CancelSubmission { submission_id } => {
                self.cancel_submission(&submission_id).await;
            }
            Command::Reconnect { reply } => {
                if self.connection.is_some() {
                    let _ = reply.send(Ok(()));
                } else if self.active.is_some() {
                    let _ = reply.send(Err(
                        "cannot reconnect while a Codex delivery is uncertain".into()
                    ));
                } else {
                    let result = connect_once(&self.spawner, &self.contract)
                        .await
                        .map_err(|error| format!("{error:#}"));
                    if let Ok(connection) = result {
                        self.connection = Some(connection);
                        self.next_request_id = 1;
                        let _ = reply.send(Ok(()));
                    } else if let Err(error) = result {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Shutdown { reply } => self.begin_shutdown(reply).await,
        }
    }

    async fn start_thread_rpc(
        &mut self,
        kind: ThreadRpcKind,
        params: Value,
        reply: oneshot::Sender<std::result::Result<Value, String>>,
    ) {
        if self.shutdown.is_some() {
            let _ = reply.send(Err("Codex app-server transport is shutting down".into()));
            return;
        }
        if self.active.is_some() {
            let _ = reply.send(Err("Codex app-server transport has an active turn".into()));
            return;
        }
        let Some(connection) = self.connection.as_mut() else {
            let _ = reply.send(Err(
                "Codex app-server is disconnected; reconnect before resuming a thread".into(),
            ));
            return;
        };
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let request = serde_json::json!({
            "id": request_id,
            "method": kind.method(),
            "params": params,
        });
        if let Err(error) = send_message(&mut connection.writer, &request).await {
            let detail = format!("Codex app-server request failed: {error:#}");
            let _ = reply.send(Err(detail.clone()));
            self.connection_lost(detail).await;
            return;
        }
        self.pending
            .insert(request_id, PendingRequest::ThreadRpc(reply));
    }

    async fn start_turn(
        &mut self,
        submission_id: String,
        thread_id: String,
        input: Vec<Value>,
        cancelled: Arc<AtomicBool>,
        events: mpsc::UnboundedSender<SubmissionEvent>,
    ) {
        if self.shutdown.is_some() {
            let _ = events.send(SubmissionEvent::Failed(
                "Codex app-server transport is shutting down".into(),
            ));
            return;
        }
        if self.active.is_some() {
            let _ = events.send(SubmissionEvent::Failed(
                "Codex app-server transport already has an active turn".into(),
            ));
            return;
        }
        if cancelled.load(Ordering::Acquire) {
            let _ = events.send(SubmissionEvent::Failed(
                "Codex turn was cancelled before delivery".into(),
            ));
            return;
        }
        let Some(connection) = self.connection.as_mut() else {
            let _ = events.send(SubmissionEvent::Failed(
                "Codex app-server is disconnected; resume the thread before starting a turn".into(),
            ));
            return;
        };

        let request_id = self.next_request_id;
        self.next_request_id += 1;
        self.active = Some(ActiveSubmission {
            submission_id: submission_id.clone(),
            thread_id: thread_id.clone(),
            turn_id: None,
            start_acknowledged: false,
            cancel_requested: false,
            interrupt_request_id: None,
            interrupt_acknowledged: false,
            terminal_status: None,
            events,
        });
        let request = serde_json::json!({
            "id": request_id,
            "method": "turn/start",
            "params": {
                "threadId": thread_id,
                "clientUserMessageId": submission_id,
                "input": input,
            }
        });
        if let Err(error) = send_message(&mut connection.writer, &request).await {
            self.connection_lost(format!(
                "Codex turn delivery is uncertain after the write failed: {error:#}"
            ))
            .await;
            return;
        }
        self.pending
            .insert(request_id, PendingRequest::TurnStart { submission_id });
    }

    async fn cancel_submission(&mut self, submission_id: &str) {
        let Some(mut active) = self.active.take() else {
            return;
        };
        if active.submission_id != submission_id {
            self.active = Some(active);
            return;
        }
        active.cancel_requested = true;
        let interrupt_failure = if active.terminal_status.is_none() {
            self.send_interrupt_if_ready(&mut active).await
        } else {
            None
        };
        self.active = Some(active);
        if let Some(detail) = interrupt_failure {
            self.connection_lost(detail).await;
            return;
        }
        self.finish_active_if_ready();
    }

    async fn send_interrupt_if_ready(&mut self, active: &mut ActiveSubmission) -> Option<String> {
        if active.interrupt_request_id.is_some() {
            return None;
        }
        let turn_id = active.turn_id.as_deref()?;
        let Some(connection) = self.connection.as_mut() else {
            return Some("Codex turn interrupt could not use a disconnected transport".into());
        };
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let request = serde_json::json!({
            "id": request_id,
            "method": "turn/interrupt",
            "params": {
                "threadId": active.thread_id,
                "turnId": turn_id,
            }
        });
        if let Err(error) = send_message(&mut connection.writer, &request).await {
            return Some(format!(
                "Codex turn interrupt failed; delivery is uncertain: {error:#}"
            ));
        }
        active.interrupt_request_id = Some(request_id);
        self.pending.insert(
            request_id,
            PendingRequest::Interrupt {
                submission_id: active.submission_id.clone(),
            },
        );
        None
    }

    async fn handle_frame(&mut self, frame: Value) {
        if frame.get("method").is_some() {
            if frame.get("id").is_some() {
                let failed = match self.connection.as_mut() {
                    Some(connection) => reject_server_request(&mut connection.writer, &frame)
                        .await
                        .err(),
                    None => None,
                };
                if let Some(error) = failed {
                    self.connection_lost(format!(
                        "failed to reject Codex app-server request: {error:#}"
                    ))
                    .await;
                }
            } else {
                self.handle_notification(frame).await;
            }
            return;
        }

        let Some(request_id) = frame.get("id").and_then(Value::as_i64) else {
            self.connection_lost("Codex app-server response omitted a numeric id".into())
                .await;
            return;
        };
        let Some(pending) = self.pending.remove(&request_id) else {
            self.connection_lost(format!(
                "Codex app-server returned unknown response id {request_id}"
            ))
            .await;
            return;
        };
        match pending {
            PendingRequest::ThreadRpc(reply) => {
                let _ = reply.send(response_result(&frame, request_id));
            }
            PendingRequest::TurnStart { submission_id } => {
                self.handle_turn_start_response(&submission_id, &frame, request_id)
                    .await;
            }
            PendingRequest::Interrupt { submission_id } => {
                self.handle_interrupt_response(&submission_id, &frame, request_id)
                    .await;
            }
        }
    }

    async fn handle_turn_start_response(
        &mut self,
        submission_id: &str,
        frame: &Value,
        request_id: i64,
    ) {
        let result = response_result(frame, request_id).and_then(|result| {
            let turn = result
                .get("turn")
                .and_then(Value::as_object)
                .ok_or_else(|| "Codex turn/start response omitted turn".to_owned())?;
            let turn_id = turn
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| "Codex turn/start response omitted turn.id".to_owned())?;
            if turn.get("status").and_then(Value::as_str) != Some("inProgress") {
                return Err("Codex turn/start did not create an in-progress turn".to_owned());
            }
            Ok(turn_id.to_owned())
        });

        let Some(mut active) = self.active.take() else {
            self.connection_lost(
                "Codex turn/start response arrived without an active submission".into(),
            )
            .await;
            return;
        };
        if active.submission_id != submission_id {
            self.active = Some(active);
            self.connection_lost("Codex turn/start response crossed submissions".into())
                .await;
            return;
        }
        match result {
            Ok(turn_id) => {
                active.turn_id = Some(turn_id.clone());
                active.start_acknowledged = true;
                let _ = active.events.send(SubmissionEvent::Started { turn_id });
                let interrupt_failure =
                    if active.cancel_requested && active.terminal_status.is_none() {
                        self.send_interrupt_if_ready(&mut active).await
                    } else {
                        None
                    };
                self.active = Some(active);
                if let Some(detail) = interrupt_failure {
                    self.connection_lost(detail).await;
                    return;
                }
                self.finish_active_if_ready();
            }
            Err(error) => {
                let _ = active.events.send(SubmissionEvent::Failed(error));
            }
        }
    }

    async fn handle_interrupt_response(
        &mut self,
        submission_id: &str,
        frame: &Value,
        request_id: i64,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        if active.submission_id != submission_id {
            return;
        }
        match response_result(frame, request_id) {
            Ok(_) => active.interrupt_acknowledged = true,
            Err(error) => {
                self.connection_lost(format!("Codex rejected turn interrupt: {error}"))
                    .await;
                return;
            }
        }
        self.finish_active_if_ready();
    }

    async fn handle_notification(&mut self, frame: Value) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let params = &frame["params"];
        if params
            .get("threadId")
            .and_then(Value::as_str)
            .is_some_and(|thread_id| thread_id != active.thread_id)
        {
            return;
        }
        if let (Some(expected), Some(actual)) = (
            active.turn_id.as_deref(),
            params.get("turnId").and_then(Value::as_str),
        ) {
            if actual != expected {
                return;
            }
        }
        let _ = active
            .events
            .send(SubmissionEvent::Notification(frame.clone()));

        if frame.get("method").and_then(Value::as_str) != Some("turn/completed") {
            return;
        }
        let Some(turn) = params.get("turn").and_then(Value::as_object) else {
            self.connection_lost("Codex turn/completed omitted turn".into())
                .await;
            return;
        };
        let Some(turn_id) = turn
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        else {
            self.connection_lost("Codex turn/completed omitted turn.id".into())
                .await;
            return;
        };
        if active
            .turn_id
            .as_deref()
            .is_some_and(|expected| expected != turn_id)
        {
            return;
        }
        active.turn_id.get_or_insert_with(|| turn_id.to_owned());
        let Some(status) = turn.get("status").and_then(Value::as_str) else {
            self.connection_lost("Codex turn/completed omitted status".into())
                .await;
            return;
        };
        active.terminal_status = Some(status.to_owned());
        self.finish_active_if_ready();
    }

    fn finish_active_if_ready(&mut self) {
        let ready = self.active.as_ref().is_some_and(|active| {
            active.start_acknowledged
                && active.terminal_status.is_some()
                && (active.interrupt_request_id.is_none() || active.interrupt_acknowledged)
        });
        if !ready {
            return;
        }
        let active = self.active.take().expect("active checked above");
        let status = active
            .terminal_status
            .expect("terminal status checked above");
        let _ = active.events.send(SubmissionEvent::Terminal { status });
    }

    async fn begin_shutdown(
        &mut self,
        reply: Option<oneshot::Sender<std::result::Result<(), String>>>,
    ) {
        if self.shutdown.is_some() {
            if let Some(reply) = reply {
                let _ = reply.send(Err(
                    "Codex app-server transport is already shutting down".into()
                ));
            }
            return;
        }
        self.shutdown = Some(ShutdownState {
            reply,
            deadline: Instant::now() + SHUTDOWN_TIMEOUT,
        });
        if let Some(submission_id) = self
            .active
            .as_ref()
            .map(|active| active.submission_id.clone())
        {
            self.cancel_submission(&submission_id).await;
        }
    }

    async fn finish_shutdown(&mut self) {
        if let Some(connection) = self.connection.take() {
            close_connection(connection).await;
        }
        let result = if self.pending.is_empty() {
            Ok(())
        } else {
            Err("Codex app-server stopped with pending protocol requests".into())
        };
        self.fail_pending("Codex app-server transport shut down");
        if let Some(shutdown) = self.shutdown.take() {
            if let Some(reply) = shutdown.reply {
                let _ = reply.send(result);
            }
        }
    }

    async fn connection_lost(&mut self, detail: String) {
        if let Some(active) = self.active.take() {
            let _ = active
                .events
                .send(SubmissionEvent::DeliveryUncertain(detail.clone()));
        }
        self.fail_pending(&detail);
        if let Some(connection) = self.connection.take() {
            close_connection(connection).await;
        }
    }

    fn fail_pending(&mut self, detail: &str) {
        for (_, pending) in self.pending.drain() {
            if let PendingRequest::ThreadRpc(reply) = pending {
                let _ = reply.send(Err(detail.to_owned()));
            }
        }
    }
}

fn response_result(frame: &Value, request_id: i64) -> std::result::Result<Value, String> {
    if let Some(error) = frame.get("error") {
        let detail = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown JSON-RPC error");
        return Err(format!(
            "Codex app-server request {request_id} failed: {detail}"
        ));
    }
    frame
        .get("result")
        .cloned()
        .ok_or_else(|| "Codex app-server response omitted result".into())
}

async fn connect_once(
    spawner: &Arc<dyn AppServerSpawner>,
    contract: &Contract,
) -> Result<Connection> {
    let spawned = tokio::time::timeout(CONNECT_TIMEOUT, spawner.spawn())
        .await
        .context("timed out launching Codex app-server")??;
    let mut connection = Connection {
        reader: BufReader::new(spawned.reader),
        writer: spawned.writer,
        child: spawned.child,
    };
    let initialized = tokio::time::timeout(
        CONNECT_TIMEOUT,
        initialize_connection(&mut connection, contract),
    )
    .await
    .context("timed out initializing Codex app-server")
    .and_then(|result| result);
    if let Err(error) = initialized {
        close_connection(connection).await;
        return Err(error);
    }
    Ok(connection)
}

async fn initialize_connection(connection: &mut Connection, contract: &Contract) -> Result<()> {
    send_message(
        &mut connection.writer,
        &serde_json::json!({
            "id": INITIALIZE_REQUEST_ID,
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
    loop {
        let frame = read_message(&mut connection.reader).await?;
        if frame.get("method").is_some() {
            if frame.get("id").is_some() {
                reject_server_request(&mut connection.writer, &frame).await?;
            }
            continue;
        }
        if frame.get("id").and_then(Value::as_i64) != Some(INITIALIZE_REQUEST_ID) {
            anyhow::bail!("Codex app-server returned an unexpected initialize response id");
        }
        let result = response_result(&frame, INITIALIZE_REQUEST_ID).map_err(anyhow::Error::msg)?;
        attest_initialize(&result, contract)?;
        break;
    }
    send_message(
        &mut connection.writer,
        &serde_json::json!({"method": "initialized"}),
    )
    .await
}

async fn close_connection(mut connection: Connection) {
    let _ = connection.writer.shutdown().await;
    let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, connection.child.shutdown()).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tokio::io::{duplex, split, DuplexStream};
    use tokio::sync::oneshot as test_oneshot;

    struct FakeChild {
        shutdowns: Arc<AtomicUsize>,
    }

    impl ChildControl for FakeChild {
        fn shutdown(&mut self) -> BoxFuture<'_, Result<()>> {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        }
    }

    struct FakeSpawner {
        streams: Mutex<VecDeque<DuplexStream>>,
        spawns: Arc<AtomicUsize>,
        shutdowns: Arc<AtomicUsize>,
    }

    impl AppServerSpawner for FakeSpawner {
        fn spawn(&self) -> BoxFuture<'static, Result<SpawnedAppServer>> {
            let stream = self.streams.lock().expect("fake spawner lock").pop_front();
            let spawns = Arc::clone(&self.spawns);
            let shutdowns = Arc::clone(&self.shutdowns);
            async move {
                let stream = stream.context("fake app-server has no connection")?;
                spawns.fetch_add(1, Ordering::SeqCst);
                let (reader, writer) = split(stream);
                Ok(SpawnedAppServer {
                    reader: Box::pin(reader),
                    writer: Box::pin(writer),
                    child: Box::new(FakeChild { shutdowns }),
                })
            }
            .boxed()
        }
    }

    fn fixture(
        client_streams: Vec<DuplexStream>,
    ) -> (
        Arc<FakeSpawner>,
        Contract,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let spawns = Arc::new(AtomicUsize::new(0));
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let spawner = Arc::new(FakeSpawner {
            streams: Mutex::new(client_streams.into()),
            spawns: Arc::clone(&spawns),
            shutdowns: Arc::clone(&shutdowns),
        });
        let contract = Contract {
            profile: "unused-by-transport".into(),
            model: "unused-by-transport".into(),
            cwd: std::env::current_dir().expect("test cwd"),
            workspace_roots: Vec::new(),
            codex_home: "/tmp/fake-persistent-codex-home".into(),
        };
        (spawner, contract, spawns, shutdowns)
    }

    async fn serve_initialize(stream: &mut DuplexStream, codex_home: &str) {
        let mut request = BufReader::new(stream);
        let initialize = read_message(&mut request)
            .await
            .expect("initialize request");
        assert_eq!(initialize["method"], "initialize");
        assert_eq!(initialize["id"], INITIALIZE_REQUEST_ID);
        send_message(
            request.get_mut(),
            &json!({
                "id": INITIALIZE_REQUEST_ID,
                "result": {
                    "userAgent": format!("shaltaiboltai/{} (test)", super::super::VERSION),
                    "codexHome": codex_home,
                    "platformFamily": std::env::consts::FAMILY,
                    "platformOs": std::env::consts::OS,
                }
            }),
        )
        .await
        .expect("initialize response");
        let initialized = read_message(&mut request)
            .await
            .expect("initialized notification");
        assert_eq!(initialized["method"], "initialized");
    }

    async fn next_request(reader: &mut BufReader<&mut DuplexStream>, method: &str) -> Value {
        let request = read_message(reader).await.expect("protocol request");
        assert_eq!(request["method"], method);
        request
    }

    #[tokio::test]
    async fn reuses_one_initialized_connection_for_two_turns() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, spawns, shutdowns) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);

            let thread = next_request(&mut reader, "thread/start").await;
            send_message(
                reader.get_mut(),
                &json!({"id": thread["id"], "result": {"thread": {"id": "thread-1"}}}),
            )
            .await
            .expect("thread response");

            for index in 1..=2 {
                let turn = next_request(&mut reader, "turn/start").await;
                assert!(turn["params"]["clientUserMessageId"]
                    .as_str()
                    .is_some_and(|id| id.starts_with("shaltaiboltai-")));
                let turn_id = format!("turn-{index}");
                send_message(
                    reader.get_mut(),
                    &json!({
                        "id": turn["id"],
                        "result": {"turn": {"id": turn_id, "status": "inProgress"}}
                    }),
                )
                .await
                .expect("turn response");
                send_message(
                    reader.get_mut(),
                    &json!({
                        "method": "turn/completed",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": turn_id,
                            "turn": {"id": turn_id, "status": "completed", "items": []}
                        }
                    }),
                )
                .await
                .expect("turn completion");
            }
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let thread = handle
            .start_thread(json!({"ephemeral": true}))
            .await
            .expect("start thread");
        assert_eq!(thread["thread"]["id"], "thread-1");
        for expected in ["turn-1", "turn-2"] {
            let mut submission = handle
                .start_turn(
                    "thread-1".into(),
                    vec![json!({"type": "text", "text": expected, "textElements": []})],
                )
                .expect("start turn");
            assert!(matches!(
                submission.next_event().await,
                Some(SubmissionEvent::Started { turn_id }) if turn_id == expected
            ));
            assert!(matches!(
                submission.next_event().await,
                Some(SubmissionEvent::Notification(frame))
                    if frame["method"] == "turn/completed"
            ));
            assert!(matches!(
                submission.next_event().await,
                Some(SubmissionEvent::Terminal { status }) if status == "completed"
            ));
        }
        runtime.shutdown().await.expect("shutdown transport");
        fake.await.expect("fake app-server");
        assert_eq!(spawns.load(Ordering::SeqCst), 1);
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn correlates_out_of_order_thread_responses_and_rejects_server_requests() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let first = read_message(&mut reader).await.expect("first request");
            let second = read_message(&mut reader).await.expect("second request");
            send_message(
                reader.get_mut(),
                &json!({
                    "id": "approval-1",
                    "method": "item/commandExecution/requestApproval",
                    "params": {}
                }),
            )
            .await
            .expect("server request");
            let rejection = read_message(&mut reader).await.expect("request rejection");
            assert_eq!(rejection["id"], "approval-1");
            assert_eq!(rejection["error"]["code"], -32601);
            send_message(
                reader.get_mut(),
                &json!({"id": second["id"], "result": {"marker": second["method"]}}),
            )
            .await
            .expect("second response");
            send_message(
                reader.get_mut(),
                &json!({"id": first["id"], "result": {"marker": first["method"]}}),
            )
            .await
            .expect("first response");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let (started, resumed) = tokio::join!(
            handle.start_thread(json!({})),
            handle.resume_thread(json!({"threadId": "thread-1"}))
        );
        assert_eq!(started.expect("start response")["marker"], "thread/start");
        assert_eq!(resumed.expect("resume response")["marker"], "thread/resume");
        runtime.shutdown().await.expect("shutdown transport");
        fake.await.expect("fake app-server");
    }

    #[tokio::test]
    async fn cancellation_before_turn_response_interrupts_exact_returned_turn() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let (turn_seen_tx, turn_seen_rx) = test_oneshot::channel();
        let (release_response_tx, release_response_rx) = test_oneshot::channel();
        let (interrupt_seen_tx, interrupt_seen_rx) = test_oneshot::channel();
        let (release_terminal_tx, release_terminal_rx) = test_oneshot::channel();
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            turn_seen_tx.send(()).expect("signal turn write");
            release_response_rx.await.expect("release turn response");
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-cancel", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");
            let interrupt = next_request(&mut reader, "turn/interrupt").await;
            assert_eq!(interrupt["params"]["threadId"], "thread-1");
            assert_eq!(interrupt["params"]["turnId"], "turn-cancel");
            send_message(
                reader.get_mut(),
                &json!({"id": interrupt["id"], "result": {}}),
            )
            .await
            .expect("interrupt response");
            interrupt_seen_tx.send(()).expect("signal interrupt");
            release_terminal_rx.await.expect("release terminal event");
            send_message(
                reader.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-cancel",
                        "turn": {"id": "turn-cancel", "status": "interrupted", "items": []}
                    }
                }),
            )
            .await
            .expect("interrupted completion");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "stop"})],
            )
            .expect("queue turn");
        turn_seen_rx.await.expect("turn write observed");
        drop(submission);
        release_response_tx.send(()).expect("release turn response");
        interrupt_seen_rx.await.expect("interrupt observed");
        let mut overlapping = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "must wait"})],
            )
            .expect("queue overlapping turn");
        assert!(matches!(
            overlapping.next_event().await,
            Some(SubmissionEvent::Failed(message)) if message.contains("active turn")
        ));
        release_terminal_tx
            .send(())
            .expect("release terminal event");
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn reconnects_an_idle_closed_connection_with_a_fresh_initialize() {
        let (client_one, mut server_one) = duplex(64 * 1024);
        let (client_two, mut server_two) = duplex(64 * 1024);
        let (spawner, contract, spawns, shutdowns) = fixture(vec![client_one, client_two]);
        let first = tokio::spawn(async move {
            serve_initialize(&mut server_one, "/tmp/fake-persistent-codex-home").await;
        });
        let second = tokio::spawn(async move {
            serve_initialize(&mut server_two, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server_two);
            let resumed = next_request(&mut reader, "thread/resume").await;
            send_message(
                reader.get_mut(),
                &json!({"id": resumed["id"], "result": {"thread": {"id": "thread-1"}}}),
            )
            .await
            .expect("resume response");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        first.await.expect("first fake app-server");
        tokio::time::timeout(Duration::from_secs(1), async {
            while shutdowns.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("actor should reap the idle closed connection");

        let handle = runtime.handle();
        handle.reconnect().await.expect("reconnect transport");
        let resumed = handle
            .resume_thread(json!({"threadId": "thread-1"}))
            .await
            .expect("resume thread after reconnect");
        assert_eq!(resumed["thread"]["id"], "thread-1");
        runtime.shutdown().await.expect("shutdown transport");
        second.await.expect("second fake app-server");
        assert_eq!(spawns.load(Ordering::SeqCst), 2);
        assert_eq!(shutdowns.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn eof_after_turn_write_is_uncertain_and_never_retried() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, spawns, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let _turn = next_request(&mut reader, "turn/start").await;
            drop(reader);
            drop(server);
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "once"})],
            )
            .expect("queue turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("connection failed")
        ));
        let mut refused = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "twice"})],
            )
            .expect("queue refused turn");
        assert!(matches!(
            refused.next_event().await,
            Some(SubmissionEvent::Failed(message)) if message.contains("disconnected")
        ));
        assert_eq!(spawns.load(Ordering::SeqCst), 1);
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }
}
