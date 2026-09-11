use super::{
    attest_initialize, read_message, reject_server_request, send_message, BufferedNotifications,
    Contract, MAX_FRAME_BYTES,
};
use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, Notify};
use tokio::task::JoinHandle;
use tokio::time::Instant;

const INITIALIZE_REQUEST_ID: i64 = 0;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(5);
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(500);
const COMMAND_QUEUE_CAPACITY: usize = 16;
const SHUTDOWN_QUEUE_CAPACITY: usize = 1;
const FRAME_QUEUE_CAPACITY: usize = 32;
const SUBMISSION_EVENT_QUEUE_CAPACITY: usize = 64;
const SUBMISSION_TERMINAL_EVENT_RESERVE: usize = 2;
const MAX_IN_FLIGHT_STEERS: usize = 1;
const MAX_IDLE_WAITERS: usize = 16;
const MAX_PENDING_THREAD_RPCS: usize = 8;
const MAX_IDENTIFIER_BYTES: usize = 4 * 1024;
const MAX_COMMAND_PAYLOAD_BYTES: usize = MAX_FRAME_BYTES - 16 * 1024;

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
    events: mpsc::Receiver<SubmissionEvent>,
    handle: AppServerHandle,
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

    /// Add user input to this submission's active Codex turn. The actor owns
    /// the exact turn id, so callers cannot accidentally steer a stale turn.
    pub(super) async fn steer(&self, input: Vec<Value>) -> Result<String> {
        self.handle.steer_turn(self.id.clone(), input).await
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if !self.terminal {
            self.cancelled.store(true, Ordering::Release);
            self.handle.cancel_notify.notify_one();
        }
    }
}

#[derive(Clone)]
pub(super) struct AppServerHandle {
    command_tx: mpsc::Sender<Command>,
    cancel_notify: Arc<Notify>,
    steer_admission: Arc<AtomicBool>,
}

struct SteerAdmission {
    occupied: Arc<AtomicBool>,
}

impl SteerAdmission {
    fn acquire(occupied: &Arc<AtomicBool>) -> Result<Self> {
        occupied
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                anyhow::anyhow!("Codex turn already has an admitted steer; queue it sequentially")
            })?;
        Ok(Self {
            occupied: Arc::clone(occupied),
        })
    }
}

impl Drop for SteerAdmission {
    fn drop(&mut self) {
        self.occupied.store(false, Ordering::Release);
    }
}

impl AppServerHandle {
    fn try_send_command(&self, command: Command) -> Result<()> {
        self.command_tx
            .try_send(command)
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    anyhow::anyhow!("Codex app-server command queue is full")
                }
                mpsc::error::TrySendError::Closed(_) => {
                    anyhow::anyhow!("Codex app-server transport stopped")
                }
            })
    }

    pub(super) async fn start_thread(&self, params: Value) -> Result<Value> {
        self.thread_rpc(ThreadRpcKind::Start, params).await
    }

    pub(super) async fn resume_thread(&self, params: Value) -> Result<Value> {
        self.thread_rpc(ThreadRpcKind::Resume, params).await
    }

    async fn thread_rpc(&self, kind: ThreadRpcKind, params: Value) -> Result<Value> {
        validate_command_payload("Codex thread request", &params)?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.try_send_command(Command::ThreadRpc {
            kind,
            params,
            reply: reply_tx,
        })?;
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
        validate_identifier("Codex thread id", &thread_id)?;
        if input.is_empty() {
            anyhow::bail!("Codex turn input must not be empty");
        }
        validate_command_payload("Codex turn input", &input)?;
        let submission_id = format!("shaltaiboltai-{:032x}", rand::random::<u128>());
        let cancelled = Arc::new(AtomicBool::new(false));
        let (events_tx, events) = mpsc::channel(SUBMISSION_EVENT_QUEUE_CAPACITY);
        self.try_send_command(Command::StartTurn {
            submission_id: submission_id.clone(),
            thread_id,
            input,
            cancelled: Arc::clone(&cancelled),
            events: events_tx,
        })?;
        Ok(Submission {
            id: submission_id,
            events,
            handle: self.clone(),
            cancelled,
            terminal: false,
        })
    }

    pub(super) async fn steer_turn(
        &self,
        submission_id: String,
        input: Vec<Value>,
    ) -> Result<String> {
        if submission_id.is_empty() {
            anyhow::bail!("Codex submission id must not be empty");
        }
        validate_identifier("Codex submission id", &submission_id)?;
        if input.is_empty() {
            anyhow::bail!("Codex steer input must not be empty");
        }
        validate_command_payload("Codex steer input", &input)?;
        let admission = SteerAdmission::acquire(&self.steer_admission)?;
        let client_user_message_id = format!("shaltaiboltai-{:032x}", rand::random::<u128>());
        let (reply_tx, reply_rx) = oneshot::channel();
        self.try_send_command(Command::SteerTurn {
            submission_id,
            client_user_message_id,
            input,
            admission,
            reply: reply_tx,
        })?;
        reply_rx
            .await
            .context("Codex app-server transport dropped a steer response")?
            .map_err(anyhow::Error::msg)
    }

    pub(super) async fn reconnect(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.try_send_command(Command::Reconnect { reply: reply_tx })?;
        reply_rx
            .await
            .context("Codex app-server transport dropped reconnect result")?
            .map_err(anyhow::Error::msg)
    }

    /// Wait until an acknowledged cancellation reaches a terminal event. New
    /// turns use this barrier so an immediately retried prompt cannot overlap
    /// the still-running turn it replaced.
    pub(super) async fn wait_idle(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.try_send_command(Command::WaitIdle { reply: reply_tx })?;
        reply_rx
            .await
            .context("Codex app-server transport dropped the idle barrier")?
            .map_err(anyhow::Error::msg)
    }
}

pub(super) struct PersistentAppServer {
    handle: AppServerHandle,
    shutdown_tx: mpsc::Sender<ShutdownCommand>,
    task: Option<JoinHandle<Result<()>>>,
}

impl PersistentAppServer {
    pub(super) async fn connect(
        spawner: Arc<dyn AppServerSpawner>,
        contract: Contract,
    ) -> Result<Self> {
        let connection = connect_once(&spawner, &contract).await?;
        let (command_tx, command_rx) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(SHUTDOWN_QUEUE_CAPACITY);
        let cancel_notify = Arc::new(Notify::new());
        let handle = AppServerHandle {
            command_tx: command_tx.clone(),
            cancel_notify: Arc::clone(&cancel_notify),
            steer_admission: Arc::new(AtomicBool::new(false)),
        };
        let task = tokio::spawn(
            Actor {
                spawner,
                contract,
                connection: Some(connection),
                next_request_id: 1,
                pending: HashMap::new(),
                active: None,
                idle_waiters: Vec::new(),
                shutdown: None,
                cancel_notify,
            }
            .run(command_rx, shutdown_rx),
        );
        Ok(Self {
            handle,
            shutdown_tx,
            task: Some(task),
        })
    }

    pub(super) fn handle(&self) -> AppServerHandle {
        self.handle.clone()
    }

    pub(super) async fn shutdown(mut self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let request_result = self
            .shutdown_tx
            .try_send(ShutdownCommand {
                reply: Some(reply_tx),
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    anyhow::anyhow!("Codex app-server shutdown is already queued")
                }
                mpsc::error::TrySendError::Closed(_) => {
                    anyhow::anyhow!("Codex app-server transport stopped")
                }
            });
        let protocol_result = match request_result {
            Ok(()) => match reply_rx.await {
                Ok(result) => result.map_err(anyhow::Error::msg),
                Err(_) => Err(anyhow::anyhow!(
                    "Codex app-server transport dropped shutdown result"
                )),
            },
            Err(error) => Err(error),
        };
        let actor_result = if let Some(task) = self.task.take() {
            task.await
                .context("Codex app-server transport task panicked")?
        } else {
            Ok(())
        };
        protocol_result?;
        actor_result
    }
}

impl Drop for PersistentAppServer {
    fn drop(&mut self) {
        if self.task.is_some() {
            let _ = self.shutdown_tx.try_send(ShutdownCommand { reply: None });
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
        events: mpsc::Sender<SubmissionEvent>,
    },
    SteerTurn {
        submission_id: String,
        client_user_message_id: String,
        input: Vec<Value>,
        admission: SteerAdmission,
        reply: oneshot::Sender<std::result::Result<String, String>>,
    },
    Reconnect {
        reply: oneshot::Sender<std::result::Result<(), String>>,
    },
    WaitIdle {
        reply: oneshot::Sender<std::result::Result<(), String>>,
    },
}

struct ShutdownCommand {
    reply: Option<oneshot::Sender<std::result::Result<(), String>>>,
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

struct InitializingConnection {
    reader: BufReader<BoxReader>,
    writer: BoxWriter,
    child: Box<dyn ChildControl>,
}

struct Connection {
    frames: mpsc::Receiver<Result<Value>>,
    writer: BoxWriter,
    child: Box<dyn ChildControl>,
    reader_task: JoinHandle<()>,
}

enum PendingRequest {
    ThreadRpc(oneshot::Sender<std::result::Result<Value, String>>),
    TurnStart {
        submission_id: String,
    },
    Steer {
        submission_id: String,
        _admission: SteerAdmission,
        reply: oneshot::Sender<std::result::Result<String, String>>,
    },
    Interrupt {
        submission_id: String,
    },
}

struct QueuedSteer {
    client_user_message_id: String,
    input: Vec<Value>,
    admission: SteerAdmission,
    reply: oneshot::Sender<std::result::Result<String, String>>,
}

enum TurnStartResponse {
    Started(String),
    Rejected(String),
    ProtocolViolation(String),
}

enum RpcResponse<'a> {
    Success(&'a Value),
    Rejected(String),
    ProtocolViolation(String),
}

struct ActiveSubmission {
    submission_id: String,
    thread_id: String,
    cancelled: Arc<AtomicBool>,
    turn_id: Option<String>,
    start_acknowledged: bool,
    cancel_requested: bool,
    interrupt_request_id: Option<i64>,
    interrupt_acknowledged: bool,
    cancel_deadline: Option<Instant>,
    queued_steers: VecDeque<QueuedSteer>,
    pending_steers: usize,
    pre_start_notifications: BufferedNotifications,
    terminal_status: Option<String>,
    events: mpsc::Sender<SubmissionEvent>,
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
    idle_waiters: Vec<oneshot::Sender<std::result::Result<(), String>>>,
    shutdown: Option<ShutdownState>,
    cancel_notify: Arc<Notify>,
}

enum ActorInput {
    Shutdown(Option<ShutdownCommand>),
    CancellationNotice,
    Command(Option<Command>),
    ChannelsClosed,
    Frame(Result<Value>),
    CancelDeadline,
    ShutdownDeadline,
}

impl Actor {
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        mut shutdowns: mpsc::Receiver<ShutdownCommand>,
    ) -> Result<()> {
        let mut commands_open = true;
        let mut shutdowns_open = true;
        let cancel_notify = Arc::clone(&self.cancel_notify);
        loop {
            let now = Instant::now();
            let cancel_deadline = self
                .active
                .as_ref()
                .and_then(|active| active.cancel_deadline);
            let shutdown_deadline = self.shutdown.as_ref().map(|shutdown| shutdown.deadline);
            let expired_deadline = if cancel_deadline.is_some_and(|deadline| deadline <= now) {
                Some(ActorInput::CancelDeadline)
            } else if shutdown_deadline.is_some_and(|deadline| deadline <= now) {
                Some(ActorInput::ShutdownDeadline)
            } else {
                None
            };
            let input = if let Some(expired_deadline) = expired_deadline {
                expired_deadline
            } else {
                let priority_shutdown = if shutdowns_open {
                    match shutdowns.try_recv() {
                        Ok(shutdown) => Some(shutdown),
                        Err(mpsc::error::TryRecvError::Empty) => None,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            shutdowns_open = false;
                            None
                        }
                    }
                } else {
                    None
                };
                if let Some(shutdown) = priority_shutdown {
                    ActorInput::Shutdown(Some(shutdown))
                } else if !commands_open && !shutdowns_open && self.shutdown.is_none() {
                    ActorInput::ChannelsClosed
                } else if self.active.as_ref().is_some_and(|active| {
                    active.cancelled.load(Ordering::Acquire) && !active.cancel_requested
                }) {
                    ActorInput::CancellationNotice
                } else if let Some(connection) = self.connection.as_mut() {
                    let cancel_deadline = cancel_deadline.unwrap_or(now);
                    let shutdown_deadline = shutdown_deadline.unwrap_or(now);
                    tokio::select! {
                        shutdown = shutdowns.recv(), if shutdowns_open => ActorInput::Shutdown(shutdown),
                        _ = cancel_notify.notified() => ActorInput::CancellationNotice,
                        command = commands.recv(), if commands_open => ActorInput::Command(command),
                        frame = connection.frames.recv() => ActorInput::Frame(
                            frame.unwrap_or_else(|| {
                                Err(anyhow::anyhow!("Codex app-server frame reader stopped"))
                            })
                        ),
                        _ = tokio::time::sleep_until(cancel_deadline),
                            if self.active.as_ref().is_some_and(|active| active.cancel_deadline.is_some()) => {
                            ActorInput::CancelDeadline
                        }
                        _ = tokio::time::sleep_until(shutdown_deadline), if self.shutdown.is_some() => {
                            ActorInput::ShutdownDeadline
                        }
                    }
                } else if let Some(shutdown) = &self.shutdown {
                    let deadline = shutdown.deadline;
                    tokio::select! {
                        queued_shutdown = shutdowns.recv(), if shutdowns_open => ActorInput::Shutdown(queued_shutdown),
                        _ = cancel_notify.notified() => ActorInput::CancellationNotice,
                        command = commands.recv(), if commands_open => ActorInput::Command(command),
                        _ = tokio::time::sleep_until(deadline) => ActorInput::ShutdownDeadline,
                    }
                } else {
                    tokio::select! {
                        shutdown = shutdowns.recv(), if shutdowns_open => ActorInput::Shutdown(shutdown),
                        _ = cancel_notify.notified() => ActorInput::CancellationNotice,
                        command = commands.recv(), if commands_open => ActorInput::Command(command),
                    }
                }
            };

            match input {
                ActorInput::Shutdown(Some(shutdown)) => {
                    self.begin_shutdown(shutdown.reply).await;
                }
                ActorInput::Shutdown(None) => {
                    shutdowns_open = false;
                }
                ActorInput::CancellationNotice => {
                    let cancelled_submission = self.active.as_ref().and_then(|active| {
                        (active.cancelled.load(Ordering::Acquire) && !active.cancel_requested)
                            .then(|| active.submission_id.clone())
                    });
                    if let Some(submission_id) = cancelled_submission {
                        self.cancel_submission(&submission_id).await;
                    }
                }
                ActorInput::Command(Some(command)) => self.handle_command(command).await,
                ActorInput::Command(None) => {
                    commands_open = false;
                }
                ActorInput::ChannelsClosed => {
                    self.begin_shutdown(None).await;
                }
                ActorInput::Frame(Ok(frame)) => self.handle_frame(frame).await,
                ActorInput::Frame(Err(error)) => {
                    self.connection_lost(format!("Codex app-server connection failed: {error:#}"))
                        .await;
                }
                ActorInput::CancelDeadline => {
                    self.connection_lost(
                        "Codex turn cancellation timed out; delivery is uncertain".into(),
                    )
                    .await;
                }
                ActorInput::ShutdownDeadline => {
                    self.connection_lost(
                        "Codex app-server shutdown timed out; delivery is uncertain".into(),
                    )
                    .await;
                }
            }

            if !commands_open && !shutdowns_open && self.shutdown.is_none() {
                self.begin_shutdown(None).await;
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
            Command::SteerTurn {
                submission_id,
                client_user_message_id,
                input,
                admission,
                reply,
            } => {
                self.steer_turn(
                    submission_id,
                    client_user_message_id,
                    input,
                    admission,
                    reply,
                )
                .await;
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
            Command::WaitIdle { reply } => {
                if self.shutdown.is_some() {
                    let _ = reply.send(Err("Codex app-server transport is shutting down".into()));
                } else if self.active.is_some() {
                    if self.idle_waiters.len() >= MAX_IDLE_WAITERS {
                        let _ = reply.send(Err("too many Codex idle waiters".into()));
                    } else {
                        self.idle_waiters.push(reply);
                    }
                } else if self.connection.is_some() {
                    let _ = reply.send(Ok(()));
                } else {
                    let _ = reply.send(Err("Codex app-server transport is disconnected".into()));
                }
            }
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
        if self
            .pending
            .values()
            .filter(|pending| matches!(pending, PendingRequest::ThreadRpc(_)))
            .count()
            >= MAX_PENDING_THREAD_RPCS
        {
            let _ = reply.send(Err("too many pending Codex thread requests".into()));
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
        if let Err(error) = send_message_bounded(&mut connection.writer, &request).await {
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
        events: mpsc::Sender<SubmissionEvent>,
    ) {
        if self.shutdown.is_some() {
            let _ = events.try_send(SubmissionEvent::Failed(
                "Codex app-server transport is shutting down".into(),
            ));
            return;
        }
        if self.active.is_some() {
            let _ = events.try_send(SubmissionEvent::Failed(
                "Codex app-server transport already has an active turn".into(),
            ));
            return;
        }
        if cancelled.load(Ordering::Acquire) {
            let _ = events.try_send(SubmissionEvent::Failed(
                "Codex turn was cancelled before delivery".into(),
            ));
            return;
        }
        let Some(connection) = self.connection.as_mut() else {
            let _ = events.try_send(SubmissionEvent::Failed(
                "Codex app-server is disconnected; resume the thread before starting a turn".into(),
            ));
            return;
        };

        let request_id = self.next_request_id;
        self.next_request_id += 1;
        self.active = Some(ActiveSubmission {
            submission_id: submission_id.clone(),
            thread_id: thread_id.clone(),
            cancelled,
            turn_id: None,
            start_acknowledged: false,
            cancel_requested: false,
            interrupt_request_id: None,
            interrupt_acknowledged: false,
            cancel_deadline: None,
            queued_steers: VecDeque::new(),
            pending_steers: 0,
            pre_start_notifications: BufferedNotifications::default(),
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
        if let Err(error) = send_message_bounded(&mut connection.writer, &request).await {
            self.connection_lost(format!(
                "Codex turn delivery is uncertain after the write failed: {error:#}"
            ))
            .await;
            return;
        }
        self.pending
            .insert(request_id, PendingRequest::TurnStart { submission_id });
    }

    async fn steer_turn(
        &mut self,
        submission_id: String,
        client_user_message_id: String,
        input: Vec<Value>,
        admission: SteerAdmission,
        reply: oneshot::Sender<std::result::Result<String, String>>,
    ) {
        if self.shutdown.is_some() {
            let _ = reply.send(Err("Codex app-server transport is shutting down".into()));
            return;
        }
        let Some(mut active) = self.active.take() else {
            let _ = reply.send(Err("no active Codex turn to steer".into()));
            return;
        };
        if active.submission_id != submission_id {
            self.active = Some(active);
            let _ = reply.send(Err("Codex steer targeted a stale submission".into()));
            return;
        }
        if active.cancel_requested || active.terminal_status.is_some() {
            self.active = Some(active);
            let _ = reply.send(Err("Codex turn is no longer steerable".into()));
            return;
        }
        if active
            .queued_steers
            .len()
            .saturating_add(active.pending_steers)
            >= MAX_IN_FLIGHT_STEERS
        {
            self.active = Some(active);
            let _ = reply.send(Err(
                "Codex turn already has an in-flight steer; queue it sequentially".into(),
            ));
            return;
        }

        let steer = QueuedSteer {
            client_user_message_id,
            input,
            admission,
            reply,
        };
        let failure = if active.turn_id.is_some() {
            self.send_steer(&mut active, steer).await
        } else {
            active.queued_steers.push_back(steer);
            None
        };
        self.active = Some(active);
        if let Some(detail) = failure {
            self.connection_lost(detail).await;
        }
    }

    async fn send_steer(
        &mut self,
        active: &mut ActiveSubmission,
        steer: QueuedSteer,
    ) -> Option<String> {
        let turn_id = active
            .turn_id
            .as_deref()
            .expect("steers are sent only after a turn id is known");
        let Some(connection) = self.connection.as_mut() else {
            let detail = "Codex steer could not use a disconnected transport".to_owned();
            let _ = steer.reply.send(Err(detail.clone()));
            return Some(detail);
        };
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let request = serde_json::json!({
            "id": request_id,
            "method": "turn/steer",
            "params": {
                "threadId": active.thread_id,
                "clientUserMessageId": &steer.client_user_message_id,
                "input": &steer.input,
                "expectedTurnId": turn_id,
            }
        });
        if let Err(error) = send_message_bounded(&mut connection.writer, &request).await {
            let detail =
                format!("Codex steer delivery is uncertain after the write failed: {error:#}");
            let _ = steer.reply.send(Err(detail.clone()));
            return Some(detail);
        }
        active.pending_steers += 1;
        self.pending.insert(
            request_id,
            PendingRequest::Steer {
                submission_id: active.submission_id.clone(),
                _admission: steer.admission,
                reply: steer.reply,
            },
        );
        None
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
        active.cancel_deadline = Some(Instant::now() + CANCEL_TIMEOUT);
        for queued in active.queued_steers.drain(..) {
            let _ = queued
                .reply
                .send(Err("Codex steer was cancelled before delivery".into()));
        }
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
        if let Err(error) = send_message_bounded(&mut connection.writer, &request).await {
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
        let has_method = frame.get("method").is_some();
        if has_method && (frame.get("result").is_some() || frame.get("error").is_some()) {
            self.connection_lost(
                "Codex app-server emitted a hybrid request/response envelope".into(),
            )
            .await;
            return;
        }
        if has_method {
            if frame
                .get("method")
                .and_then(Value::as_str)
                .is_none_or(|method| method.is_empty())
            {
                self.connection_lost(
                    "Codex app-server emitted an envelope with an invalid method".into(),
                )
                .await;
                return;
            }
            if frame.get("id").is_some() {
                let failed = match self.connection.as_mut() {
                    Some(connection) => tokio::time::timeout(
                        WRITE_TIMEOUT,
                        reject_server_request(&mut connection.writer, &frame),
                    )
                    .await
                    .map_err(|_| anyhow::anyhow!("timed out rejecting Codex server request"))
                    .and_then(|result| result)
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
                self.handle_thread_rpc_response(reply, &frame, request_id)
                    .await;
            }
            PendingRequest::TurnStart { submission_id } => {
                self.handle_turn_start_response(&submission_id, &frame, request_id)
                    .await;
            }
            PendingRequest::Steer {
                submission_id,
                _admission: admission,
                reply,
            } => {
                // Make the next steer admissible before waking this caller. On
                // the multi-thread runtime, a reply receiver can otherwise win
                // the scheduling race against this match arm's final drop.
                drop(admission);
                self.handle_steer_response(&submission_id, reply, &frame, request_id)
                    .await;
            }
            PendingRequest::Interrupt { submission_id } => {
                self.handle_interrupt_response(&submission_id, &frame, request_id)
                    .await;
            }
        }
    }

    async fn handle_thread_rpc_response(
        &mut self,
        reply: oneshot::Sender<std::result::Result<Value, String>>,
        frame: &Value,
        request_id: i64,
    ) {
        match classify_rpc_response(frame, request_id, "thread RPC") {
            RpcResponse::Success(result) => {
                let _ = reply.send(Ok(result.clone()));
            }
            RpcResponse::Rejected(error) => {
                let _ = reply.send(Err(error));
            }
            RpcResponse::ProtocolViolation(detail) => {
                let _ = reply.send(Err(detail.clone()));
                self.connection_lost(format!("Codex thread RPC protocol violation: {detail}"))
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
        let result = parse_turn_start_response(frame, request_id);

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
            TurnStartResponse::Started(turn_id) => {
                active.turn_id = Some(turn_id.clone());
                active.start_acknowledged = true;
                let _ = active.events.try_send(SubmissionEvent::Started { turn_id });
                let buffered = std::mem::take(&mut active.pre_start_notifications).messages;
                self.active = Some(active);
                for notification in buffered {
                    self.handle_notification(notification).await;
                    if self.active.is_none() || self.connection.is_none() {
                        return;
                    }
                }
                let Some(mut active) = self.active.take() else {
                    return;
                };
                let mut delivery_failure = None;
                if active.terminal_status.is_some() {
                    for queued in active.queued_steers.drain(..) {
                        let _ = queued.reply.send(Err(
                            "Codex turn completed before the queued steer could be delivered"
                                .into(),
                        ));
                    }
                } else if active.cancel_requested {
                    for queued in active.queued_steers.drain(..) {
                        let _ = queued
                            .reply
                            .send(Err("Codex steer was cancelled before delivery".into()));
                    }
                } else {
                    while let Some(steer) = active.queued_steers.pop_front() {
                        if let Some(detail) = self.send_steer(&mut active, steer).await {
                            delivery_failure = Some(detail);
                            break;
                        }
                    }
                }
                let interrupt_failure = if delivery_failure.is_none()
                    && active.cancel_requested
                    && active.terminal_status.is_none()
                {
                    self.send_interrupt_if_ready(&mut active).await
                } else {
                    None
                };
                self.active = Some(active);
                if let Some(detail) = delivery_failure.or(interrupt_failure) {
                    self.connection_lost(detail).await;
                    return;
                }
                self.finish_active_if_ready();
            }
            TurnStartResponse::Rejected(error) => {
                for queued in active.queued_steers.drain(..) {
                    let _ = queued.reply.send(Err(error.clone()));
                }
                let _ = active.events.try_send(SubmissionEvent::Failed(error));
                for waiter in self.idle_waiters.drain(..) {
                    let _ = waiter.send(Ok(()));
                }
            }
            TurnStartResponse::ProtocolViolation(detail) => {
                self.active = Some(active);
                self.connection_lost(format!(
                    "Codex turn/start protocol violation; delivery is uncertain: {detail}"
                ))
                .await;
            }
        }
    }

    async fn handle_steer_response(
        &mut self,
        submission_id: &str,
        reply: oneshot::Sender<std::result::Result<String, String>>,
        frame: &Value,
        request_id: i64,
    ) {
        let Some(active) = self.active.as_mut() else {
            let _ = reply.send(Err(
                "Codex steer response arrived without an active submission".into(),
            ));
            self.connection_lost(
                "Codex steer response arrived without an active submission".into(),
            )
            .await;
            return;
        };
        if active.submission_id != submission_id {
            let _ = reply.send(Err("Codex steer response crossed submissions".into()));
            self.connection_lost("Codex steer response crossed submissions".into())
                .await;
            return;
        }
        if active.pending_steers == 0 {
            let detail = "Codex turn/steer response had no pending steer".to_owned();
            let _ = reply.send(Err(detail.clone()));
            self.connection_lost(detail).await;
            return;
        }
        active.pending_steers -= 1;
        let result = match classify_rpc_response(frame, request_id, "turn/steer") {
            RpcResponse::Rejected(error) => {
                if let Some(actual_turn_id) = steer_mismatch_actual_turn(frame) {
                    let expected_turn_id = active.turn_id.as_deref().unwrap_or("<missing>");
                    let detail = format!(
                        "Codex turn/steer state diverged: expected {expected_turn_id:?}, server reported {actual_turn_id:?}; delivery is uncertain"
                    );
                    let _ = reply.send(Err(error));
                    self.connection_lost(detail).await;
                    return;
                }
                let _ = reply.send(Err(error));
                self.finish_active_if_ready();
                return;
            }
            RpcResponse::ProtocolViolation(detail) => {
                let _ = reply.send(Err(detail.clone()));
                self.connection_lost(format!(
                    "Codex turn/steer protocol violation; delivery is uncertain: {detail}"
                ))
                .await;
                return;
            }
            RpcResponse::Success(result) => result,
        };
        let validated = result
            .get("turnId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| "Codex turn/steer response omitted turnId".to_owned())
            .and_then(|turn_id| {
                if active.turn_id.as_deref() != Some(turn_id) {
                    return Err("Codex turn/steer response changed the active turn id".into());
                }
                Ok(turn_id.to_owned())
            });
        match validated {
            Ok(turn_id) => {
                let _ = reply.send(Ok(turn_id));
            }
            Err(detail) => {
                let _ = reply.send(Err(detail.clone()));
                self.connection_lost(format!(
                    "Codex turn/steer protocol violation; delivery is uncertain: {detail}"
                ))
                .await;
                return;
            }
        }
        self.finish_active_if_ready();
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
        match classify_rpc_response(frame, request_id, "turn/interrupt") {
            RpcResponse::Success(result) if result.is_object() => {
                active.interrupt_acknowledged = true;
            }
            RpcResponse::Success(_) => {
                self.connection_lost(
                    "Codex turn/interrupt protocol violation: response result was not an object"
                        .into(),
                )
                .await;
                return;
            }
            RpcResponse::Rejected(error) => {
                self.connection_lost(format!("Codex rejected turn interrupt: {error}"))
                    .await;
                return;
            }
            RpcResponse::ProtocolViolation(detail) => {
                self.connection_lost(format!("Codex turn/interrupt protocol violation: {detail}"))
                    .await;
                return;
            }
        }
        self.finish_active_if_ready();
    }

    async fn handle_notification(&mut self, frame: Value) {
        let Some(mut active) = self.active.take() else {
            return;
        };
        if !active.start_acknowledged {
            let buffered = active.pre_start_notifications.push(frame);
            self.active = Some(active);
            if let Err(error) = buffered {
                self.connection_lost(format!(
                    "Codex app-server pre-response notification buffer failed: {error:#}"
                ))
                .await;
            }
            return;
        }
        let routed = route_notification(&mut active, frame);
        self.active = Some(active);
        if let Err(error) = routed {
            self.connection_lost(format!(
                "Codex app-server emitted an invalid turn notification: {error:#}"
            ))
            .await;
            return;
        }
        self.finish_active_if_ready();
    }

    fn finish_active_if_ready(&mut self) {
        let ready = self.active.as_ref().is_some_and(|active| {
            active.start_acknowledged
                && active.terminal_status.is_some()
                && active.queued_steers.is_empty()
                && active.pending_steers == 0
                && (active.interrupt_request_id.is_none() || active.interrupt_acknowledged)
        });
        if !ready {
            return;
        }
        let active = self.active.take().expect("active checked above");
        let status = active
            .terminal_status
            .expect("terminal status checked above");
        let _ = active.events.try_send(SubmissionEvent::Terminal { status });
        for waiter in self.idle_waiters.drain(..) {
            let _ = waiter.send(Ok(()));
        }
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
        self.fail_idle_waiters("Codex app-server transport shut down");
        if let Some(shutdown) = self.shutdown.take() {
            if let Some(reply) = shutdown.reply {
                let _ = reply.send(result);
            }
        }
    }

    async fn connection_lost(&mut self, detail: String) {
        if let Some(mut active) = self.active.take() {
            for queued in active.queued_steers.drain(..) {
                let _ = queued.reply.send(Err(detail.clone()));
            }
            let _ = active
                .events
                .try_send(SubmissionEvent::DeliveryUncertain(detail.clone()));
        }
        self.fail_pending(&detail);
        self.fail_idle_waiters(&detail);
        if let Some(connection) = self.connection.take() {
            close_connection(connection).await;
        }
    }

    fn fail_pending(&mut self, detail: &str) {
        for (_, pending) in self.pending.drain() {
            match pending {
                PendingRequest::ThreadRpc(reply) => {
                    let _ = reply.send(Err(detail.to_owned()));
                }
                PendingRequest::Steer { reply, .. } => {
                    let _ = reply.send(Err(detail.to_owned()));
                }
                PendingRequest::TurnStart { .. } | PendingRequest::Interrupt { .. } => {}
            }
        }
    }

    fn fail_idle_waiters(&mut self, detail: &str) {
        for waiter in self.idle_waiters.drain(..) {
            let _ = waiter.send(Err(detail.to_owned()));
        }
    }
}

fn route_notification(active: &mut ActiveSubmission, frame: Value) -> Result<()> {
    let method = frame
        .get("method")
        .and_then(Value::as_str)
        .context("notification omitted method")?;
    let params = frame
        .get("params")
        .and_then(Value::as_object)
        .context("notification omitted params")?;
    if params
        .get("threadId")
        .and_then(Value::as_str)
        .is_some_and(|thread_id| thread_id != active.thread_id)
    {
        return Ok(());
    }
    let expected_turn_id = active
        .turn_id
        .as_deref()
        .context("active submission omitted its acknowledged turn id")?;
    let notification_turn_id = params.get("turnId").and_then(Value::as_str).or_else(|| {
        matches!(method, "turn/started" | "turn/completed")
            .then(|| params.get("turn")?.get("id")?.as_str())
            .flatten()
    });
    if notification_turn_id.is_some_and(|turn_id| turn_id != expected_turn_id) {
        return Ok(());
    }

    if method == "turn/completed" {
        let thread_id = params
            .get("threadId")
            .and_then(Value::as_str)
            .context("turn/completed omitted threadId")?;
        if thread_id != active.thread_id {
            return Ok(());
        }
        let turn = params
            .get("turn")
            .and_then(Value::as_object)
            .context("turn/completed omitted turn")?;
        let turn_id = turn
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .context("turn/completed omitted turn.id")?;
        if turn_id != expected_turn_id {
            return Ok(());
        }
        let status = turn
            .get("status")
            .and_then(Value::as_str)
            .context("turn/completed omitted status")?;
        if !matches!(status, "completed" | "failed" | "interrupted") {
            anyhow::bail!("turn/completed reported non-terminal status {status:?}");
        }
        active.terminal_status = Some(status.to_owned());
    }
    let required_capacity = if method == "turn/completed" {
        SUBMISSION_TERMINAL_EVENT_RESERVE
    } else {
        SUBMISSION_TERMINAL_EVENT_RESERVE + 1
    };
    if active.events.capacity() < required_capacity {
        anyhow::bail!("submission notification queue exceeded its bounded capacity");
    }
    match active.events.try_send(SubmissionEvent::Notification(frame)) {
        Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => Ok(()),
        Err(mpsc::error::TrySendError::Full(_)) => {
            anyhow::bail!("submission notification queue exceeded its bounded capacity")
        }
    }
}

fn parse_turn_start_response(frame: &Value, request_id: i64) -> TurnStartResponse {
    match classify_rpc_response(frame, request_id, "turn/start") {
        RpcResponse::Success(result) => {
            let parsed = (|| {
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
            })();
            match parsed {
                Ok(turn_id) => TurnStartResponse::Started(turn_id),
                Err(error) => TurnStartResponse::ProtocolViolation(error),
            }
        }
        RpcResponse::Rejected(error) => TurnStartResponse::Rejected(error),
        RpcResponse::ProtocolViolation(error) => TurnStartResponse::ProtocolViolation(error),
    }
}

fn classify_rpc_response<'a>(frame: &'a Value, request_id: i64, method: &str) -> RpcResponse<'a> {
    match (frame.get("result"), frame.get("error")) {
        (Some(result), None) => RpcResponse::Success(result),
        (None, Some(error)) => {
            let Some(error) = error.as_object() else {
                return RpcResponse::ProtocolViolation(format!(
                    "Codex {method} response contained a non-object error"
                ));
            };
            if error.get("code").and_then(Value::as_i64).is_none() {
                return RpcResponse::ProtocolViolation(format!(
                    "Codex {method} response error omitted numeric code"
                ));
            }
            let Some(message) = error
                .get("message")
                .and_then(Value::as_str)
                .filter(|message| !message.is_empty())
            else {
                return RpcResponse::ProtocolViolation(format!(
                    "Codex {method} response error omitted message"
                ));
            };
            RpcResponse::Rejected(format!(
                "Codex app-server request {request_id} failed: {message}"
            ))
        }
        (Some(_), Some(_)) => RpcResponse::ProtocolViolation(format!(
            "Codex {method} response contained both result and error"
        )),
        (None, None) => RpcResponse::ProtocolViolation(format!(
            "Codex {method} response omitted both result and error"
        )),
    }
}

fn steer_mismatch_actual_turn(frame: &Value) -> Option<&str> {
    const PREFIX: &str = "expected active turn id `";
    const SEPARATOR: &str = "` but found `";
    let message = frame.get("error")?.get("message")?.as_str()?;
    let (_, actual) = message.strip_prefix(PREFIX)?.split_once(SEPARATOR)?;
    actual
        .strip_suffix('`')
        .filter(|turn_id| !turn_id.is_empty())
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

fn validate_command_payload<T>(label: &str, payload: &T) -> Result<()>
where
    T: Serialize,
{
    let encoded_bytes = serde_json::to_vec(payload)
        .with_context(|| format!("serialize {label} for size validation"))?
        .len();
    if encoded_bytes > MAX_COMMAND_PAYLOAD_BYTES {
        anyhow::bail!(
            "{label} exceeded the {MAX_COMMAND_PAYLOAD_BYTES}-byte transport payload limit"
        );
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if value.len() > MAX_IDENTIFIER_BYTES {
        anyhow::bail!("{label} exceeded the {MAX_IDENTIFIER_BYTES}-byte transport limit");
    }
    Ok(())
}

async fn connect_once(
    spawner: &Arc<dyn AppServerSpawner>,
    contract: &Contract,
) -> Result<Connection> {
    let spawned = tokio::time::timeout(CONNECT_TIMEOUT, spawner.spawn())
        .await
        .context("timed out launching Codex app-server")??;
    let mut connection = InitializingConnection {
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
        close_initializing_connection(connection).await;
        return Err(error);
    }
    let InitializingConnection {
        reader,
        writer,
        child,
    } = connection;
    let (frames, reader_task) = spawn_frame_reader(reader);
    Ok(Connection {
        frames,
        writer,
        child,
        reader_task,
    })
}

async fn initialize_connection(
    connection: &mut InitializingConnection,
    contract: &Contract,
) -> Result<()> {
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

fn spawn_frame_reader(
    mut reader: BufReader<BoxReader>,
) -> (mpsc::Receiver<Result<Value>>, JoinHandle<()>) {
    let (frames_tx, frames) = mpsc::channel(FRAME_QUEUE_CAPACITY);
    let reader_task = tokio::spawn(async move {
        loop {
            let frame = read_message(&mut reader).await;
            let terminal = frame.is_err();
            if frames_tx.send(frame).await.is_err() || terminal {
                break;
            }
        }
    });
    (frames, reader_task)
}

async fn send_message_bounded(writer: &mut BoxWriter, message: &Value) -> Result<()> {
    tokio::time::timeout(WRITE_TIMEOUT, send_message(writer, message))
        .await
        .context("timed out writing to Codex app-server")?
}

async fn close_initializing_connection(mut connection: InitializingConnection) {
    let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, connection.writer.shutdown()).await;
    let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, connection.child.shutdown()).await;
}

async fn close_connection(mut connection: Connection) {
    connection.reader_task.abort();
    let _ = connection.reader_task.await;
    let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, connection.writer.shutdown()).await;
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
    async fn bounds_queued_commands_and_payload_bytes() {
        let (command_tx, _commands) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let handle = AppServerHandle {
            command_tx,
            cancel_notify: Arc::new(Notify::new()),
            steer_admission: Arc::new(AtomicBool::new(false)),
        };
        let oversized_id = handle
            .start_turn(
                "x".repeat(MAX_IDENTIFIER_BYTES + 1),
                vec![json!({"type": "text", "text": "bounded"})],
            )
            .err()
            .expect("oversized identifier must fail before enqueue");
        assert!(oversized_id.to_string().contains("transport limit"));
        let mut queued = Vec::new();
        for index in 0..COMMAND_QUEUE_CAPACITY {
            queued.push(
                handle
                    .start_turn(
                        format!("thread-{index}"),
                        vec![json!({"type": "text", "text": "bounded"})],
                    )
                    .expect("command within queue capacity"),
            );
        }
        let full = handle
            .start_turn(
                "thread-overflow".into(),
                vec![json!({"type": "text", "text": "overflow"})],
            )
            .err()
            .expect("command queue must reject overflow");
        assert!(full.to_string().contains("queue is full"));

        let thread_rpc_full = handle
            .start_thread(json!({"model": "test"}))
            .await
            .expect_err("thread RPC must not wait outside a full queue");
        assert!(thread_rpc_full.to_string().contains("queue is full"));
        let reconnect_full = handle
            .reconnect()
            .await
            .expect_err("reconnect must not wait outside a full queue");
        assert!(reconnect_full.to_string().contains("queue is full"));
        let idle_full = handle
            .wait_idle()
            .await
            .expect_err("idle barrier must not wait outside a full queue");
        assert!(idle_full.to_string().contains("queue is full"));

        let oversized = handle
            .steer_turn(
                "submission-1".into(),
                vec![json!({
                    "type": "text",
                    "text": "x".repeat(MAX_COMMAND_PAYLOAD_BYTES),
                })],
            )
            .await
            .expect_err("oversized steer must fail before enqueue");
        assert!(oversized.to_string().contains("payload limit"));
        drop(queued);
    }

    #[test]
    fn terminal_notification_uses_the_exact_reserved_capacity() {
        let (events, mut event_rx) = mpsc::channel(SUBMISSION_EVENT_QUEUE_CAPACITY);
        for index in 0..SUBMISSION_EVENT_QUEUE_CAPACITY - SUBMISSION_TERMINAL_EVENT_RESERVE {
            events
                .try_send(SubmissionEvent::Notification(json!({"index": index})))
                .expect("fill ordinary notification capacity");
        }
        let mut active = ActiveSubmission {
            submission_id: "submission-1".into(),
            thread_id: "thread-1".into(),
            cancelled: Arc::new(AtomicBool::new(false)),
            turn_id: Some("turn-1".into()),
            start_acknowledged: true,
            cancel_requested: false,
            interrupt_request_id: None,
            interrupt_acknowledged: false,
            cancel_deadline: None,
            queued_steers: VecDeque::new(),
            pending_steers: 0,
            pre_start_notifications: BufferedNotifications::default(),
            terminal_status: None,
            events,
        };
        let ordinary = json!({
            "method": "item/completed",
            "params": {"threadId": "thread-1", "turnId": "turn-1"}
        });
        assert!(route_notification(&mut active, ordinary).is_err());
        assert_eq!(active.events.capacity(), SUBMISSION_TERMINAL_EVENT_RESERVE);

        route_notification(
            &mut active,
            json!({
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {"id": "turn-1", "status": "completed", "items": []}
                }
            }),
        )
        .expect("completion must use one reserved slot");
        let status = active
            .terminal_status
            .take()
            .expect("completion records terminal status");
        active
            .events
            .try_send(SubmissionEvent::Terminal { status })
            .expect("synthesized terminal must use the final reserved slot");
        assert_eq!(active.events.capacity(), 0);
        assert!(event_rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn admits_only_one_steer_end_to_end() {
        let (command_tx, mut commands) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let handle = AppServerHandle {
            command_tx,
            cancel_notify: Arc::new(Notify::new()),
            steer_admission: Arc::new(AtomicBool::new(false)),
        };
        let first = handle.steer_turn(
            "submission-1".into(),
            vec![json!({"type": "text", "text": "first"})],
        );
        tokio::pin!(first);
        assert!(first.as_mut().now_or_never().is_none());
        let second = handle
            .steer_turn(
                "submission-1".into(),
                vec![json!({"type": "text", "text": "second"})],
            )
            .await
            .expect_err("second steer must not wait while retaining its payload");
        assert!(second.to_string().contains("admitted steer"));
        drop(commands.recv().await.expect("first steer command"));
        let first_error = first
            .await
            .expect_err("dropping admitted command drops its response");
        assert!(first_error.to_string().contains("dropped a steer response"));
    }

    #[tokio::test]
    async fn expired_deadline_does_not_drop_a_queued_shutdown() {
        let (spawner, contract, _, _) = fixture(Vec::new());
        let (command_tx, command_rx) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(SHUTDOWN_QUEUE_CAPACITY);
        let (events, mut event_rx) = mpsc::channel(SUBMISSION_EVENT_QUEUE_CAPACITY);
        let (shutdown_reply, shutdown_result) = oneshot::channel();
        shutdown_tx
            .try_send(ShutdownCommand {
                reply: Some(shutdown_reply),
            })
            .expect("queue shutdown");
        let cancel_notify = Arc::new(Notify::new());
        let actor = tokio::spawn(
            Actor {
                spawner,
                contract,
                connection: None,
                next_request_id: 1,
                pending: HashMap::new(),
                active: Some(ActiveSubmission {
                    submission_id: "submission-1".into(),
                    thread_id: "thread-1".into(),
                    cancelled: Arc::new(AtomicBool::new(false)),
                    turn_id: Some("turn-1".into()),
                    start_acknowledged: true,
                    cancel_requested: true,
                    interrupt_request_id: None,
                    interrupt_acknowledged: false,
                    cancel_deadline: Some(Instant::now()),
                    queued_steers: VecDeque::new(),
                    pending_steers: 0,
                    pre_start_notifications: BufferedNotifications::default(),
                    terminal_status: None,
                    events,
                }),
                idle_waiters: Vec::new(),
                shutdown: None,
                cancel_notify,
            }
            .run(command_rx, shutdown_rx),
        );

        assert!(matches!(
            event_rx.recv().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("cancellation timed out")
        ));
        tokio::time::timeout(Duration::from_secs(1), shutdown_result)
            .await
            .expect("queued shutdown must not be lost behind expired deadline")
            .expect("shutdown reply sender")
            .expect("shutdown result");
        drop(command_tx);
        drop(shutdown_tx);
        actor
            .await
            .expect("actor task")
            .expect("actor completed cleanly");
    }

    #[tokio::test]
    async fn completed_turn_cancellation_bounds_a_missing_steer_response() {
        let (spawner, contract, _, _) = fixture(Vec::new());
        let (events, mut event_rx) = mpsc::channel(SUBMISSION_EVENT_QUEUE_CAPACITY);
        let steer_occupied = Arc::new(AtomicBool::new(false));
        let admission = SteerAdmission::acquire(&steer_occupied).expect("admit pending steer");
        let (steer_reply, steer_result) = oneshot::channel();
        let mut pending = HashMap::new();
        pending.insert(
            7,
            PendingRequest::Steer {
                submission_id: "submission-1".into(),
                _admission: admission,
                reply: steer_reply,
            },
        );
        let cancel_notify = Arc::new(Notify::new());
        let mut actor = Actor {
            spawner,
            contract,
            connection: None,
            next_request_id: 8,
            pending,
            active: Some(ActiveSubmission {
                submission_id: "submission-1".into(),
                thread_id: "thread-1".into(),
                cancelled: Arc::new(AtomicBool::new(true)),
                turn_id: Some("turn-1".into()),
                start_acknowledged: true,
                cancel_requested: false,
                interrupt_request_id: None,
                interrupt_acknowledged: false,
                cancel_deadline: None,
                queued_steers: VecDeque::new(),
                pending_steers: 1,
                pre_start_notifications: BufferedNotifications::default(),
                terminal_status: Some("completed".into()),
                events,
            }),
            idle_waiters: Vec::new(),
            shutdown: None,
            cancel_notify,
        };

        actor.cancel_submission("submission-1").await;
        let deadline = actor
            .active
            .as_ref()
            .and_then(|active| active.cancel_deadline)
            .expect("cancellation must bound an unresolved steer response");
        assert!(deadline > Instant::now());
        actor
            .active
            .as_mut()
            .expect("active submission")
            .cancel_deadline = Some(Instant::now());

        let (command_tx, command_rx) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(SHUTDOWN_QUEUE_CAPACITY);
        let (shutdown_reply, shutdown_result) = oneshot::channel();
        shutdown_tx
            .try_send(ShutdownCommand {
                reply: Some(shutdown_reply),
            })
            .expect("queue shutdown");
        let actor = tokio::spawn(actor.run(command_rx, shutdown_rx));

        assert!(matches!(
            event_rx.recv().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("cancellation timed out")
        ));
        assert!(steer_result
            .await
            .expect("pending steer reply")
            .expect_err("missing steer response must fail")
            .contains("cancellation timed out"));
        shutdown_result
            .await
            .expect("shutdown reply sender")
            .expect("shutdown result");
        drop(command_tx);
        drop(shutdown_tx);
        actor
            .await
            .expect("actor task")
            .expect("actor completed cleanly");
        assert!(!steer_occupied.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn closed_command_and_shutdown_channels_shutdown_an_idle_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, shutdowns) = fixture(vec![client]);
        let (release_server, hold_server) = test_oneshot::channel();
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            hold_server.await.expect("hold idle fake server");
        });
        let connection = connect_once(
            &(Arc::clone(&spawner) as Arc<dyn AppServerSpawner>),
            &contract,
        )
        .await
        .expect("connect actor fixture");
        let (command_tx, command_rx) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(SHUTDOWN_QUEUE_CAPACITY);
        drop(command_tx);
        drop(shutdown_tx);
        let cancel_notify = Arc::new(Notify::new());
        let actor = tokio::spawn(
            Actor {
                spawner,
                contract,
                connection: Some(connection),
                next_request_id: 1,
                pending: HashMap::new(),
                active: None,
                idle_waiters: Vec::new(),
                shutdown: None,
                cancel_notify,
            }
            .run(command_rx, shutdown_rx),
        );
        tokio::time::timeout(Duration::from_secs(1), actor)
            .await
            .expect("closed channels must not strand idle actor")
            .expect("actor task")
            .expect("actor completed cleanly");
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        release_server.send(()).expect("release fake server");
        fake.await.expect("fake app-server");
    }

    #[tokio::test]
    async fn closed_channels_continue_polling_an_active_turn_during_shutdown() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, shutdowns) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let interrupt = next_request(&mut reader, "turn/interrupt").await;
            assert_eq!(interrupt["params"]["threadId"], "thread-1");
            assert_eq!(interrupt["params"]["turnId"], "turn-1");
            send_message(
                reader.get_mut(),
                &json!({"id": interrupt["id"], "result": {}}),
            )
            .await
            .expect("interrupt response");
            send_message(
                reader.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-1", "status": "interrupted", "items": []}
                    }
                }),
            )
            .await
            .expect("turn completion");
        });
        let connection = connect_once(
            &(Arc::clone(&spawner) as Arc<dyn AppServerSpawner>),
            &contract,
        )
        .await
        .expect("connect actor fixture");
        let (command_tx, command_rx) = mpsc::channel(COMMAND_QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(SHUTDOWN_QUEUE_CAPACITY);
        drop(command_tx);
        drop(shutdown_tx);
        let (events, mut event_rx) = mpsc::channel(SUBMISSION_EVENT_QUEUE_CAPACITY);
        let cancel_notify = Arc::new(Notify::new());
        let actor = tokio::spawn(
            Actor {
                spawner,
                contract,
                connection: Some(connection),
                next_request_id: 1,
                pending: HashMap::new(),
                active: Some(ActiveSubmission {
                    submission_id: "submission-1".into(),
                    thread_id: "thread-1".into(),
                    cancelled: Arc::new(AtomicBool::new(false)),
                    turn_id: Some("turn-1".into()),
                    start_acknowledged: true,
                    cancel_requested: false,
                    interrupt_request_id: None,
                    interrupt_acknowledged: false,
                    cancel_deadline: None,
                    queued_steers: VecDeque::new(),
                    pending_steers: 0,
                    pre_start_notifications: BufferedNotifications::default(),
                    terminal_status: None,
                    events,
                }),
                idle_waiters: Vec::new(),
                shutdown: None,
                cancel_notify,
            }
            .run(command_rx, shutdown_rx),
        );

        tokio::time::timeout(Duration::from_secs(1), actor)
            .await
            .expect("closed channels must keep polling the active turn")
            .expect("actor task")
            .expect("actor completed cleanly");
        assert!(matches!(
            event_rx.recv().await,
            Some(SubmissionEvent::Notification(frame)) if frame["method"] == "turn/completed"
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(SubmissionEvent::Terminal { status }) if status == "interrupted"
        ));
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        fake.await.expect("fake app-server");
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
    async fn actor_commands_do_not_corrupt_a_split_response_frame() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let (partial_written_tx, partial_written_rx) = test_oneshot::channel();
        let (release_remainder_tx, release_remainder_rx) = test_oneshot::channel();
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            let response = serde_json::to_vec(&json!({
                "id": turn["id"],
                "result": {"turn": {"id": "turn-split", "status": "inProgress"}}
            }))
            .expect("encode split response");
            let split = response.len() / 2;
            reader
                .get_mut()
                .write_all(&response[..split])
                .await
                .expect("write first response fragment");
            reader
                .get_mut()
                .flush()
                .await
                .expect("flush first response fragment");
            partial_written_tx
                .send(())
                .expect("signal first response fragment");
            release_remainder_rx
                .await
                .expect("release second response fragment");
            reader
                .get_mut()
                .write_all(&response[split..])
                .await
                .expect("write second response fragment");
            reader
                .get_mut()
                .write_all(b"\n")
                .await
                .expect("terminate split response");
            reader
                .get_mut()
                .flush()
                .await
                .expect("flush split response");

            let steer = next_request(&mut reader, "turn/steer").await;
            send_message(
                reader.get_mut(),
                &json!({"id": steer["id"], "result": {"turnId": "turn-split"}}),
            )
            .await
            .expect("steer response");
            send_message(
                reader.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-split", "status": "completed", "items": []}
                    }
                }),
            )
            .await
            .expect("turn completion");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "split", "textElements": []})],
            )
            .expect("start turn");
        partial_written_rx
            .await
            .expect("first response fragment observed");
        tokio::time::sleep(Duration::from_millis(25)).await;

        let steer = handle.steer_turn(
            submission.id().to_owned(),
            vec![json!({"type": "text", "text": "while split", "textElements": []})],
        );
        tokio::pin!(steer);
        assert!(steer.as_mut().now_or_never().is_none());
        handle
            .reconnect()
            .await
            .expect("actor remains responsive during partial frame");
        release_remainder_tx
            .send(())
            .expect("release second response fragment");

        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-split"
        ));
        assert_eq!(
            steer.await.expect("steer accepted after split response"),
            "turn-split"
        );
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Notification(frame)) if frame["method"] == "turn/completed"
        ));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Terminal { status }) if status == "completed"
        ));
        runtime.shutdown().await.expect("shutdown transport");
        fake.await.expect("fake app-server");
    }

    #[tokio::test]
    async fn queued_steer_uses_acknowledged_turn_and_replays_only_matching_notifications() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            for (turn_id, text) in [("turn-old", "stale"), ("turn-live", "current")] {
                send_message(
                    reader.get_mut(),
                    &json!({
                        "method": "item/completed",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": turn_id,
                            "item": {"type": "agentMessage", "text": text}
                        }
                    }),
                )
                .await
                .expect("pre-response item");
            }
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-live", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");

            let steer = next_request(&mut reader, "turn/steer").await;
            assert_eq!(steer["params"]["expectedTurnId"], "turn-live");
            assert_eq!(steer["params"]["input"][0]["text"], "clarify");
            assert!(steer["params"]["clientUserMessageId"]
                .as_str()
                .is_some_and(|id| id.starts_with("shaltaiboltai-")));
            send_message(
                reader.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-live", "status": "completed", "items": []}
                    }
                }),
            )
            .await
            .expect("turn completion");
            send_message(
                reader.get_mut(),
                &json!({"id": steer["id"], "result": {"turnId": "turn-live"}}),
            )
            .await
            .expect("steer response");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "start", "textElements": []})],
            )
            .expect("start turn");
        let steer_handle = handle.clone();
        let submission_id = submission.id().to_owned();
        let steer = tokio::spawn(async move {
            steer_handle
                .steer_turn(
                    submission_id,
                    vec![json!({"type": "text", "text": "clarify", "textElements": []})],
                )
                .await
        });

        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-live"
        ));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Notification(frame))
                if frame["params"]["item"]["text"] == "current"
        ));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Notification(frame)) if frame["method"] == "turn/completed"
        ));
        assert_eq!(
            steer.await.expect("steer task").expect("steer accepted"),
            "turn-live"
        );
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Terminal { status }) if status == "completed"
        ));
        runtime.shutdown().await.expect("shutdown transport");
        fake.await.expect("fake app-server");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn successful_steer_releases_admission_before_waking_the_caller() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-live", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");
            for expected in ["first", "second"] {
                let steer = next_request(&mut reader, "turn/steer").await;
                assert_eq!(steer["params"]["input"][0]["text"], expected);
                send_message(
                    reader.get_mut(),
                    &json!({"id": steer["id"], "result": {"turnId": "turn-live"}}),
                )
                .await
                .expect("steer response");
            }
            send_message(
                reader.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-live", "status": "completed", "items": []}
                    }
                }),
            )
            .await
            .expect("turn completion");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let mut submission = runtime
            .handle()
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "start", "textElements": []})],
            )
            .expect("start turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-live"
        ));
        for text in ["first", "second"] {
            assert_eq!(
                submission
                    .steer(vec![json!({
                        "type": "text",
                        "text": text,
                        "textElements": []
                    })])
                    .await
                    .expect("sequential steer"),
                "turn-live"
            );
        }
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Notification(frame)) if frame["method"] == "turn/completed"
        ));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Terminal { status }) if status == "completed"
        ));
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn malformed_successful_steer_response_invalidates_the_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-live", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");
            let steer = next_request(&mut reader, "turn/steer").await;
            send_message(
                reader.get_mut(),
                &json!({"id": steer["id"], "result": {"turnId": "turn-other"}}),
            )
            .await
            .expect("malformed successful steer response");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "start", "textElements": []})],
            )
            .expect("start turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-live"
        ));
        let error = submission
            .steer(vec![json!({
                "type": "text",
                "text": "malformed",
                "textElements": []
            })])
            .await
            .expect_err("changed turn id must fail closed");
        assert!(error.to_string().contains("changed the active turn id"));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("protocol violation")
        ));
        let mut refused = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "must not send"})],
            )
            .expect("queue refused turn");
        assert!(matches!(
            refused.next_event().await,
            Some(SubmissionEvent::Failed(message)) if message.contains("disconnected")
        ));
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn malformed_steer_envelope_invalidates_the_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-live", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");
            let steer = next_request(&mut reader, "turn/steer").await;
            send_message(reader.get_mut(), &json!({"id": steer["id"]}))
                .await
                .expect("malformed steer envelope");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let mut submission = runtime
            .handle()
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "start", "textElements": []})],
            )
            .expect("start turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-live"
        ));
        let error = submission
            .steer(vec![json!({
                "type": "text",
                "text": "malformed",
                "textElements": []
            })])
            .await
            .expect_err("malformed response envelope must fail closed");
        assert!(error.to_string().contains("omitted both result and error"));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("protocol violation")
        ));
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn hybrid_request_response_envelope_invalidates_the_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let (release_server, hold_server) = test_oneshot::channel();
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let thread = next_request(&mut reader, "thread/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": thread["id"],
                    "method": "thread/started",
                    "result": {"thread": {"id": "thread-1"}}
                }),
            )
            .await
            .expect("hybrid envelope");
            hold_server.await.expect("hold hybrid server open");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.handle().start_thread(json!({"model": "test"})),
        )
        .await;
        release_server.send(()).expect("release fake server");
        fake.await.expect("fake app-server");
        let error = response
            .expect("hybrid envelope must not strand its pending RPC")
            .expect_err("hybrid envelope must fail closed");
        assert!(error.to_string().contains("hybrid request/response"));
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn malformed_successful_start_response_invalidates_the_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-invalid", "status": "completed"}}
                }),
            )
            .await
            .expect("malformed successful start response");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "start", "textElements": []})],
            )
            .expect("start turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("turn/start protocol violation")
                    && message.contains("in-progress")
        ));
        let idle_error = handle
            .wait_idle()
            .await
            .expect_err("invalid start disconnects the actor");
        assert!(idle_error.to_string().contains("disconnected"));
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn steer_turn_id_divergence_invalidates_the_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-live", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");
            let steer = next_request(&mut reader, "turn/steer").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": steer["id"],
                    "error": {
                        "code": -32600,
                        "message": "expected active turn id `turn-live` but found `turn-other`"
                    }
                }),
            )
            .await
            .expect("steer mismatch response");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let mut submission = runtime
            .handle()
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "start", "textElements": []})],
            )
            .expect("start turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-live"
        ));
        let error = submission
            .steer(vec![json!({
                "type": "text",
                "text": "diverge",
                "textElements": []
            })])
            .await
            .expect_err("turn-id divergence must fail closed");
        assert!(error.to_string().contains("expected active turn id"));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("state diverged") && message.contains("turn-other")
        ));
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn completion_before_start_ack_never_delivers_a_queued_steer() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-fast", "status": "completed", "items": []}
                    }
                }),
            )
            .await
            .expect("early completion");
            tokio::task::yield_now().await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-fast", "status": "inProgress"}}
                }),
            )
            .await
            .expect("turn response");
            assert!(
                tokio::time::timeout(Duration::from_millis(100), read_message(&mut reader))
                    .await
                    .is_err()
            );
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "fast", "textElements": []})],
            )
            .expect("start turn");
        let steer_handle = handle.clone();
        let submission_id = submission.id().to_owned();
        let steer = tokio::spawn(async move {
            steer_handle
                .steer_turn(
                    submission_id,
                    vec![json!({"type": "text", "text": "too late", "textElements": []})],
                )
                .await
        });

        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-fast"
        ));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Notification(frame)) if frame["method"] == "turn/completed"
        ));
        let error = steer
            .await
            .expect("steer task")
            .expect_err("completed turn must reject queued steer");
        assert!(error.to_string().contains("completed"));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Terminal { status }) if status == "completed"
        ));
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
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
        let idle_handle = handle.clone();
        let idle = tokio::spawn(async move { idle_handle.wait_idle().await });
        tokio::task::yield_now().await;
        assert!(
            !idle.is_finished(),
            "idle barrier released before terminal event"
        );
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
        idle.await
            .expect("idle barrier task")
            .expect("terminal event releases idle barrier");
        fake.await.expect("fake app-server");
        runtime.shutdown().await.expect("shutdown transport");
    }

    #[tokio::test]
    async fn rejected_start_fails_queued_steer_and_releases_idle_waiters() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let (turn_seen_tx, turn_seen_rx) = test_oneshot::channel();
        let (release_rejection_tx, release_rejection_rx) = test_oneshot::channel();
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            turn_seen_tx.send(()).expect("signal turn request");
            release_rejection_rx.await.expect("release rejection");
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "error": {"code": -32600, "message": "turn rejected"}
                }),
            )
            .await
            .expect("turn rejection");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let handle = runtime.handle();
        let mut submission = handle
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "reject", "textElements": []})],
            )
            .expect("start turn");
        turn_seen_rx.await.expect("turn request observed");

        let steer = handle.steer_turn(
            submission.id().to_owned(),
            vec![json!({"type": "text", "text": "queued", "textElements": []})],
        );
        tokio::pin!(steer);
        assert!(steer.as_mut().now_or_never().is_none());
        let idle = handle.wait_idle();
        tokio::pin!(idle);
        assert!(idle.as_mut().now_or_never().is_none());
        handle
            .reconnect()
            .await
            .expect("command barrier before rejection");
        release_rejection_tx.send(()).expect("release rejection");

        let steer_error = steer.await.expect_err("rejected start must fail steer");
        assert!(steer_error.to_string().contains("turn rejected"));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Failed(message)) if message.contains("turn rejected")
        ));
        idle.await.expect("rejected start leaves actor idle");
        runtime.shutdown().await.expect("shutdown transport");
        fake.await.expect("fake app-server");
    }

    #[tokio::test]
    async fn non_terminal_completion_status_invalidates_the_connection() {
        let (client, mut server) = duplex(64 * 1024);
        let (spawner, contract, _, _) = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            serve_initialize(&mut server, "/tmp/fake-persistent-codex-home").await;
            let mut reader = BufReader::new(&mut server);
            let turn = next_request(&mut reader, "turn/start").await;
            send_message(
                reader.get_mut(),
                &json!({
                    "id": turn["id"],
                    "result": {"turn": {"id": "turn-live", "status": "inProgress"}}
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
                        "turn": {"id": "turn-live", "status": "inProgress", "items": []}
                    }
                }),
            )
            .await
            .expect("invalid completion");
        });

        let runtime = PersistentAppServer::connect(spawner, contract)
            .await
            .expect("connect transport");
        let mut submission = runtime
            .handle()
            .start_turn(
                "thread-1".into(),
                vec![json!({"type": "text", "text": "status", "textElements": []})],
            )
            .expect("start turn");
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::Started { turn_id }) if turn_id == "turn-live"
        ));
        assert!(matches!(
            submission.next_event().await,
            Some(SubmissionEvent::DeliveryUncertain(message))
                if message.contains("non-terminal status")
        ));
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
