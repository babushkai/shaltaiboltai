use super::{
    cli_model_override, has_images, persistent_codex_app_server,
    persistent_handoff_prompt_for_request, stream_chat_codex,
};
use crate::config::Config;
use crate::policy::ExecutionPolicy;
use crate::providers::codex_app_server::persistent::{
    AppServerHandle, AppServerSpawner, PersistentAppServer, Submission, SubmissionEvent,
};
use crate::providers::codex_app_server::{
    attest_thread, thread_start_params, Contract, TurnAccumulator, TurnNotification,
};
use crate::providers::{ChatEvent, ChatRequest, Message, ProviderKind, RequestPolicy, UserContent};
use anyhow::{Context, Result};
use futures_util::future::{BoxFuture, Shared};
use futures_util::FutureExt;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc::UnboundedSender, Mutex};

const MAX_CONTINUITY_ID_BYTES: usize = 4 * 1024;

type SpawnerFactory = dyn Fn(String, ExecutionPolicy, String) -> Result<(Arc<dyn AppServerSpawner>, Contract)>
    + Send
    + Sync;
type CleanupResult = std::result::Result<(), Arc<str>>;
type CleanupFuture = Shared<BoxFuture<'static, CleanupResult>>;

/// Conversation-scoped Codex app-server ownership. An operation gate
/// serializes lifecycle changes, while an active submission is consumed
/// without holding the state mutex.
#[derive(Clone)]
pub struct CodexRuntime {
    operation_gate: Arc<Mutex<()>>,
    inner: Arc<Mutex<RuntimeState>>,
    factory: Arc<SpawnerFactory>,
}

impl Default for CodexRuntime {
    fn default() -> Self {
        Self {
            operation_gate: Arc::new(Mutex::new(())),
            inner: Arc::new(Mutex::new(RuntimeState::default())),
            factory: Arc::new(|model, execution_policy, system| {
                persistent_codex_app_server(&model, &execution_policy, Some(system))
            }),
        }
    }
}

#[derive(Default)]
struct RuntimeState {
    generation: u64,
    server: Option<RuntimeServer>,
    cleanup: Option<PendingCleanup>,
    cleanup_failure: Option<Arc<str>>,
    closed: bool,
}

struct PendingCleanup {
    generation: u64,
    future: CleanupFuture,
}

/// A connected process is installed immediately with `thread: None`. This
/// preparing state retains kill/reap ownership if its caller is cancelled
/// while thread/start is awaiting a response.
struct RuntimeServer {
    server: PersistentAppServer,
    handle: AppServerHandle,
    contract: Contract,
    thread: Option<RuntimeThread>,
}

struct RuntimeThread {
    thread_id: String,
    binding: ThreadBinding,
    dirty: bool,
}

struct ThreadBinding {
    continuity_id: String,
    model: String,
    execution_policy: ExecutionPolicy,
    system: String,
    history_checkpoint: Vec<u8>,
}

impl ThreadBinding {
    fn matches(
        &self,
        continuity_id: &str,
        model: &str,
        execution_policy: &ExecutionPolicy,
        system: &str,
        history_checkpoint: &[u8],
    ) -> bool {
        self.continuity_id == continuity_id
            && self.model == model
            && &self.execution_policy == execution_policy
            && self.system == system
            && self.history_checkpoint == history_checkpoint
    }
}

struct ActiveTurn {
    submission: Submission,
    generation: u64,
    thread_id: String,
    request_messages: Vec<Message>,
}

enum ConsumeOutcome {
    Clean { assistant_text: String },
    Invalid,
    Rejected(String),
    Poison(String),
    DeliveryUncertain(String),
}

impl CodexRuntime {
    pub async fn stream_chat(
        &self,
        config: &Config,
        req: &ChatRequest,
        tx: &UnboundedSender<ChatEvent>,
    ) -> Result<()> {
        let _operation = self.operation_gate.lock().await;
        self.finish_pending_cleanup().await?;
        self.ensure_open().await?;
        if tx.is_closed() {
            anyhow::bail!("Codex chat event receiver closed before turn delivery");
        }
        let model = cli_model_override(&req.model, ProviderKind::Codex, "codex", "codex:")?;
        let Some(model) = model else {
            // The bare selector deliberately retains the CLI's own model
            // choice and the existing one-process-per-turn behavior.
            self.retire_for_stateless().await?;
            return stream_chat_codex(config, req, tx).await;
        };
        if req.policy != RequestPolicy::Interactive
            || req.continuity_id.is_none()
            || req.force_full_handoff
        {
            self.retire_for_stateless().await?;
            return stream_chat_codex(config, req, tx).await;
        }

        if has_images(&req.messages) {
            tx.send(ChatEvent::Notice(
                "images are not yet forwarded to the Codex provider".into(),
            ))
            .map_err(|_| {
                anyhow::anyhow!("Codex chat event receiver closed before turn delivery")
            })?;
        }

        let active = self.prepare_turn(req, model).await?;
        let generation = active.generation;
        let request_messages = active.request_messages.clone();
        match consume_turn(active, tx).await {
            ConsumeOutcome::Clean { assistant_text } if !tx.is_closed() => {
                let mut completed_messages = request_messages;
                completed_messages.push(Message::Assistant {
                    text: assistant_text,
                    tool_calls: Vec::new(),
                });
                let checkpoint = history_checkpoint(&completed_messages)?;
                self.commit_clean(generation, checkpoint).await;
                Ok(())
            }
            ConsumeOutcome::Clean { .. } | ConsumeOutcome::Invalid => Ok(()),
            ConsumeOutcome::Rejected(detail) => anyhow::bail!("{detail}"),
            ConsumeOutcome::Poison(detail) => match self.poison(generation).await {
                Ok(()) => anyhow::bail!("{detail}"),
                Err(cleanup_error) => anyhow::bail!(
                    "{detail}; additionally, Codex app-server cleanup failed: {cleanup_error:#}"
                ),
            },
            ConsumeOutcome::DeliveryUncertain(detail) => match self.poison(generation).await {
                Ok(()) => anyhow::bail!("{detail}"),
                Err(cleanup_error) => anyhow::bail!(
                    "{detail}; additionally, Codex app-server cleanup failed: {cleanup_error:#}"
                ),
            },
        }
    }

    async fn prepare_turn(&self, req: &ChatRequest, model: &str) -> Result<ActiveTurn> {
        let continuity_id = req
            .continuity_id
            .as_deref()
            .context("persistent Codex request omitted continuity identity")?;
        if continuity_id.is_empty() || continuity_id.len() > MAX_CONTINUITY_ID_BYTES {
            anyhow::bail!(
                "Codex continuity identity must contain 1..={MAX_CONTINUITY_ID_BYTES} bytes"
            );
        }
        let (history_prefix, latest_user) = latest_user_turn(&req.messages)?;
        let prefix_checkpoint = history_checkpoint(history_prefix)?;

        // A dropped caller retains its dirty lease until the transport has
        // delivered/acknowledged cancellation. Never select or replace a
        // thread while that work is still active.
        if self.wait_idle().await.is_err() {
            self.retire_server()
                .await
                .context("retire disconnected Codex app-server")?;
        }

        let reuse_thread = {
            let state = self.inner.lock().await;
            state.server.as_ref().is_some_and(|server| {
                server.thread.as_ref().is_some_and(|thread| {
                    latest_user.images().is_empty()
                        && !thread.dirty
                        && thread.binding.matches(
                            continuity_id,
                            model,
                            &req.execution_policy,
                            &req.system,
                            &prefix_checkpoint,
                        )
                })
            })
        };

        if !reuse_thread {
            // A process owns every native thread it starts. Replacing the
            // process bounds loaded/subscribed thread state when a session,
            // history, model, image handoff, or dirty lease prevents reuse.
            self.retire_server()
                .await
                .context("retire superseded Codex app-server")?;
            self.connect_server(model, req).await?;
            self.start_fresh_thread(continuity_id, model, req, prefix_checkpoint.clone())
                .await?;
        }

        let input_text = if reuse_thread {
            latest_user.text().to_owned()
        } else {
            persistent_handoff_prompt_for_request(req).context("no user message to send")?
        };
        let input = text_input(input_text);

        // Dirty is a lease acquired before the model-bearing turn/start is
        // admitted. Only a clean terminal result with the same generation can
        // release it.
        let (generation, handle, thread_id) = {
            let mut state = self.inner.lock().await;
            state.generation = state.generation.wrapping_add(1);
            let generation = state.generation;
            let server = state
                .server
                .as_mut()
                .context("persistent Codex server was not initialized")?;
            let thread = server
                .thread
                .as_mut()
                .context("persistent Codex thread was not initialized")?;
            thread.dirty = true;
            (generation, server.handle.clone(), thread.thread_id.clone())
        };
        let submission = handle.start_turn(thread_id.clone(), input)?;

        Ok(ActiveTurn {
            submission,
            generation,
            thread_id,
            request_messages: req.messages.clone(),
        })
    }

    async fn start_fresh_thread(
        &self,
        continuity_id: &str,
        model: &str,
        req: &ChatRequest,
        history_checkpoint: Vec<u8>,
    ) -> Result<()> {
        let (handle, contract) = {
            let mut state = self.inner.lock().await;
            state.generation = state.generation.wrapping_add(1);
            let server = state
                .server
                .as_mut()
                .context("persistent Codex process was not initialized")?;
            let mut contract = server.contract.clone();
            contract.model = model.to_owned();
            contract.developer_instructions = Some(req.system.clone());
            server.contract = contract.clone();
            server.thread = None;
            (server.handle.clone(), contract)
        };
        let params = thread_start_params(&contract)?;
        let result = handle.start_thread(params).await?;
        let thread_id = match attest_thread(&result, &contract) {
            Ok(thread_id) => thread_id,
            Err(error) => {
                if let Err(cleanup_error) = self.retire_server().await {
                    return Err(error).context(format!(
                        "Codex fresh-thread attestation failed; additionally, app-server cleanup failed: {cleanup_error:#}"
                    ));
                }
                return Err(error).context("Codex fresh-thread attestation failed");
            }
        };
        let mut state = self
            .inner
            .try_lock()
            .context("Codex runtime state was unexpectedly busy after thread creation")?;
        let server = state
            .server
            .as_mut()
            .context("persistent Codex process disappeared during thread creation")?;
        server.thread = Some(RuntimeThread {
            thread_id,
            binding: ThreadBinding {
                continuity_id: continuity_id.to_owned(),
                model: model.to_owned(),
                execution_policy: req.execution_policy.clone(),
                system: req.system.clone(),
                history_checkpoint,
            },
            dirty: false,
        });
        Ok(())
    }

    async fn connect_server(&self, model: &str, req: &ChatRequest) -> Result<()> {
        let (spawner, contract) = (self.factory)(
            model.to_owned(),
            req.execution_policy.clone(),
            req.system.clone(),
        )?;
        let server = PersistentAppServer::connect(spawner, contract.clone()).await?;
        let handle = server.handle();
        // The operation gate excludes every other runtime state transition, so
        // installation must not add a cancellation point after the connected
        // actor has been created.
        let mut state = self
            .inner
            .try_lock()
            .context("Codex runtime state was unexpectedly busy after app-server connect")?;
        state.generation = state.generation.wrapping_add(1);
        state.server = Some(RuntimeServer {
            server,
            handle,
            contract,
            thread: None,
        });
        Ok(())
    }

    async fn commit_clean(&self, generation: u64, checkpoint: Vec<u8>) {
        let mut state = self.inner.lock().await;
        if state.generation != generation {
            return;
        }
        if let Some(thread) = state
            .server
            .as_mut()
            .and_then(|server| server.thread.as_mut())
        {
            thread.binding.history_checkpoint = checkpoint;
            thread.dirty = false;
        }
    }

    async fn poison(&self, generation: u64) -> Result<()> {
        if self.inner.lock().await.generation != generation {
            return Ok(());
        }
        self.retire_server().await
    }

    async fn wait_idle(&self) -> Result<()> {
        let handle = self
            .inner
            .lock()
            .await
            .server
            .as_ref()
            .map(|server| server.handle.clone());
        match handle {
            Some(handle) => handle.wait_idle().await,
            None => Ok(()),
        }
    }

    async fn retire_server(&self) -> Result<()> {
        {
            let mut state = self.inner.lock().await;
            if state.cleanup.is_none() {
                if let Some(server) = state.server.take() {
                    state.generation = state.generation.wrapping_add(1);
                    let generation = state.generation;
                    let future = async move {
                        server
                            .server
                            .shutdown()
                            .await
                            .map_err(|error| Arc::<str>::from(format!("{error:#}")))
                    }
                    .boxed()
                    .shared();
                    state.cleanup = Some(PendingCleanup { generation, future });
                }
            }
        }
        self.finish_pending_cleanup().await
    }

    async fn retire_for_stateless(&self) -> Result<()> {
        let _ = self.wait_idle().await;
        self.retire_server().await
    }

    /// Finish cancellation and retire any native Codex process before a
    /// stateless/non-Codex request is allowed to begin. The gate is released
    /// before that other provider performs model work, preserving team-worker
    /// concurrency.
    pub async fn prepare_stateless_boundary(&self) -> Result<()> {
        let _operation = self.operation_gate.lock().await;
        self.finish_pending_cleanup().await?;
        self.ensure_open().await?;
        self.retire_for_stateless().await
    }

    async fn ensure_open(&self) -> Result<()> {
        let state = self.inner.lock().await;
        if let Some(error) = &state.cleanup_failure {
            anyhow::bail!("Codex app-server cleanup previously failed: {error}");
        }
        if state.closed {
            anyhow::bail!("Codex runtime is shut down");
        }
        Ok(())
    }

    /// Await a shared cleanup future without moving its only handle out of
    /// runtime state. If the caller is aborted, the next request or final
    /// shutdown resumes the same future and still observes its result.
    async fn finish_pending_cleanup(&self) -> Result<()> {
        let pending = {
            let state = self.inner.lock().await;
            state
                .cleanup
                .as_ref()
                .map(|cleanup| (cleanup.generation, cleanup.future.clone()))
        };
        let Some((generation, future)) = pending else {
            let state = self.inner.lock().await;
            if let Some(error) = &state.cleanup_failure {
                anyhow::bail!("Codex app-server cleanup previously failed: {error}");
            }
            return Ok(());
        };

        let result = future.await;
        let mut state = self.inner.lock().await;
        if state
            .cleanup
            .as_ref()
            .is_some_and(|cleanup| cleanup.generation == generation)
        {
            state.cleanup = None;
            if let Err(error) = &result {
                state.cleanup_failure = Some(Arc::clone(error));
            }
        }
        result.map_err(|error| anyhow::anyhow!("{error}"))
    }

    /// Idempotently stop and reap the currently owned app-server. The operation
    /// gate makes concurrent shutdown callers share the same completion
    /// barrier instead of returning while a prior reap runs.
    pub async fn shutdown(&self) -> Result<()> {
        let _operation = self.operation_gate.lock().await;
        self.finish_pending_cleanup().await?;
        {
            let mut state = self.inner.lock().await;
            state.closed = true;
        }
        // Always attempt retirement, including on repeated calls. A previous
        // shutdown future may have been aborted after setting `closed` but
        // before it installed the shared cleanup barrier.
        self.retire_server().await
    }

    #[cfg(test)]
    fn with_factory(factory: Arc<SpawnerFactory>) -> Self {
        Self {
            operation_gate: Arc::new(Mutex::new(())),
            inner: Arc::new(Mutex::new(RuntimeState::default())),
            factory,
        }
    }
}

async fn consume_turn(mut active: ActiveTurn, tx: &UnboundedSender<ChatEvent>) -> ConsumeOutcome {
    let mut accumulator = None;
    while let Some(event) = active.submission.next_event().await {
        match event {
            SubmissionEvent::Started { turn_id } => {
                if accumulator.is_some() {
                    return ConsumeOutcome::Poison(
                        "Codex app-server delivered turn/start more than once".into(),
                    );
                }
                accumulator = Some(TurnAccumulator::new(&active.thread_id, &turn_id));
            }
            SubmissionEvent::Notification(notification) => {
                let Some(accumulator) = accumulator.as_mut() else {
                    return ConsumeOutcome::Poison(
                        "Codex notification arrived before turn/start".into(),
                    );
                };
                let decoded = match accumulator.consume_notification(&notification, tx) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        return ConsumeOutcome::Poison(format!(
                            "invalid Codex turn notification: {error:#}"
                        ));
                    }
                };
                match decoded {
                    TurnNotification::Continue | TurnNotification::Terminal(_) => {}
                    TurnNotification::ModelRerouted {
                        from_model,
                        to_model,
                    } => {
                        return ConsumeOutcome::Rejected(format!(
                            "Codex rerouted the turn from {from_model} to {to_model}; exact-model contract failed"
                        ));
                    }
                }
            }
            SubmissionEvent::Terminal { status } => {
                let Some(accumulator) = accumulator.as_mut() else {
                    return ConsumeOutcome::Poison(
                        "Codex terminal event arrived before turn/start".into(),
                    );
                };
                let clean = status == "completed" && accumulator.terminal_error().is_none();
                if let Err(error) = accumulator.finish_terminal(&status, tx) {
                    return ConsumeOutcome::Poison(format!(
                        "invalid Codex terminal event: {error:#}"
                    ));
                }
                return if clean {
                    ConsumeOutcome::Clean {
                        assistant_text: accumulator.assistant_text().to_owned(),
                    }
                } else {
                    ConsumeOutcome::Invalid
                };
            }
            SubmissionEvent::Failed(detail) => return ConsumeOutcome::Rejected(detail),
            SubmissionEvent::DeliveryUncertain(detail) => {
                return ConsumeOutcome::DeliveryUncertain(detail);
            }
        }
    }
    ConsumeOutcome::Poison("Codex app-server submission ended before a terminal event".into())
}

fn latest_user_turn(messages: &[Message]) -> Result<(&[Message], &UserContent)> {
    let (latest, prefix) = messages.split_last().context("no user message to send")?;
    let Message::User(content) = latest else {
        anyhow::bail!("latest Codex history entry is not a user message");
    };
    Ok((prefix, content))
}

fn history_checkpoint(messages: &[Message]) -> Result<Vec<u8>> {
    serde_json::to_vec(messages).context("serialize Codex history checkpoint")
}

fn text_input(text: String) -> Vec<Value> {
    vec![serde_json::json!({
        "type": "text",
        "text": text,
        "textElements": []
    })]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_COMPACT_THRESHOLD_CHARS, DEFAULT_OLLAMA_NUM_CTX};
    use crate::policy::{ApprovalPolicy, SandboxMode, Workspace};
    use crate::providers::codex_app_server::persistent::{
        BoxReader, BoxWriter, ChildControl, SpawnedAppServer,
    };
    use crate::providers::codex_app_server::{
        read_message, send_message, ContractSandbox, VERSION,
    };
    use crate::providers::{ImageData, ModelEntry, ToolCall};
    use futures_util::future::BoxFuture;
    use futures_util::FutureExt;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;
    use tokio::io::{duplex, split, BufReader, DuplexStream};
    use tokio::sync::{mpsc, oneshot};

    const FAKE_CODEX_HOME: &str = "/tmp/shaltaiboltai-fake-codex-runtime-home";

    struct FakeChild {
        shutdowns: Arc<AtomicUsize>,
        shutdown_gate: Option<Arc<ShutdownGate>>,
    }

    impl ChildControl for FakeChild {
        fn shutdown(&mut self) -> BoxFuture<'_, Result<()>> {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            let shutdown_gate = self.shutdown_gate.clone();
            async move {
                if let Some(gate) = shutdown_gate {
                    gate.started.store(true, Ordering::SeqCst);
                    gate.started_notify.notify_waiters();
                    while !gate.released.load(Ordering::SeqCst) {
                        gate.release_notify.notified().await;
                    }
                }
                Ok(())
            }
            .boxed()
        }
    }

    #[derive(Default)]
    struct ShutdownGate {
        started: AtomicBool,
        released: AtomicBool,
        started_notify: tokio::sync::Notify,
        release_notify: tokio::sync::Notify,
    }

    impl ShutdownGate {
        async fn wait_started(&self) {
            while !self.started.load(Ordering::SeqCst) {
                self.started_notify.notified().await;
            }
        }

        fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release_notify.notify_waiters();
        }
    }

    struct FakeSpawner {
        streams: StdMutex<VecDeque<DuplexStream>>,
        spawns: Arc<AtomicUsize>,
        shutdowns: Arc<AtomicUsize>,
        shutdown_gate: Option<Arc<ShutdownGate>>,
    }

    impl AppServerSpawner for FakeSpawner {
        fn spawn(&self) -> BoxFuture<'static, Result<SpawnedAppServer>> {
            let stream = self.streams.lock().expect("fake spawner lock").pop_front();
            let spawns = Arc::clone(&self.spawns);
            let shutdowns = Arc::clone(&self.shutdowns);
            let shutdown_gate = self.shutdown_gate.clone();
            async move {
                let stream = stream.context("fake app-server has no connection")?;
                spawns.fetch_add(1, Ordering::SeqCst);
                let (reader, writer) = split(stream);
                let reader: BoxReader = Box::pin(reader);
                let writer: BoxWriter = Box::pin(writer);
                Ok(SpawnedAppServer {
                    reader,
                    writer,
                    child: Box::new(FakeChild {
                        shutdowns,
                        shutdown_gate,
                    }),
                })
            }
            .boxed()
        }
    }

    struct RuntimeFixture {
        runtime: CodexRuntime,
        spawns: Arc<AtomicUsize>,
        shutdowns: Arc<AtomicUsize>,
    }

    fn fixture(client_streams: Vec<DuplexStream>) -> RuntimeFixture {
        fixture_with_shutdown_gate(client_streams, None)
    }

    fn fixture_with_shutdown_gate(
        client_streams: Vec<DuplexStream>,
        shutdown_gate: Option<Arc<ShutdownGate>>,
    ) -> RuntimeFixture {
        let spawns = Arc::new(AtomicUsize::new(0));
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let spawner: Arc<dyn AppServerSpawner> = Arc::new(FakeSpawner {
            streams: StdMutex::new(client_streams.into()),
            spawns: Arc::clone(&spawns),
            shutdowns: Arc::clone(&shutdowns),
            shutdown_gate,
        });
        let factory: Arc<SpawnerFactory> = Arc::new(move |model, policy, system| {
            Ok((
                Arc::clone(&spawner),
                Contract {
                    model,
                    cwd: policy.workspace().cwd().to_path_buf(),
                    workspace_roots: policy.effective_user_visible_roots().to_vec(),
                    codex_home: PathBuf::from(FAKE_CODEX_HOME),
                    sandbox: ContractSandbox::DangerFullAccess,
                    developer_instructions: Some(system),
                },
            ))
        });
        RuntimeFixture {
            runtime: CodexRuntime::with_factory(factory),
            spawns,
            shutdowns,
        }
    }

    fn config() -> Config {
        Config {
            anthropic_api_key: None,
            openai_api_key: None,
            openai_base_url: "https://api.openai.com/v1".into(),
            openrouter_api_key: None,
            openrouter_base_url: "https://openrouter.ai/api/v1".into(),
            ollama_host: "http://localhost:11434".into(),
            default_model: None,
            compact_threshold_chars: DEFAULT_COMPACT_THRESHOLD_CHARS,
            ollama_num_ctx: DEFAULT_OLLAMA_NUM_CTX,
            theme: None,
            reduced_motion: false,
        }
    }

    fn execution_policy() -> ExecutionPolicy {
        let cwd = std::env::current_dir().expect("test current directory");
        ExecutionPolicy::from_parts(
            Workspace::new(cwd).expect("canonical test workspace"),
            SandboxMode::DangerFullAccess,
            ApprovalPolicy::Never,
        )
    }

    fn request(policy: &ExecutionPolicy, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: ModelEntry {
                provider: ProviderKind::Codex,
                id: "codex:gpt-test".into(),
            },
            continuity_id: Some("conversation-1".into()),
            system: "stay exact".into(),
            messages,
            tools: Vec::new(),
            execution_policy: policy.clone(),
            policy: RequestPolicy::Interactive,
            force_full_handoff: false,
        }
    }

    fn assistant(text: &str) -> Message {
        Message::Assistant {
            text: text.into(),
            tool_calls: Vec::<ToolCall>::new(),
        }
    }

    async fn initialize_server(stream: DuplexStream) -> BufReader<DuplexStream> {
        let mut server = BufReader::new(stream);
        let initialize = read_message(&mut server).await.expect("initialize request");
        assert_eq!(initialize["method"], "initialize");
        send_message(
            server.get_mut(),
            &json!({
                "id": initialize["id"],
                "result": {
                    "userAgent": format!("shaltaiboltai/{VERSION} test"),
                    "platformFamily": std::env::consts::FAMILY,
                    "platformOs": std::env::consts::OS,
                    "codexHome": FAKE_CODEX_HOME,
                }
            }),
        )
        .await
        .expect("initialize response");
        let initialized = read_message(&mut server)
            .await
            .expect("initialized notification");
        assert_eq!(initialized["method"], "initialized");
        server
    }

    async fn accept_thread(server: &mut BufReader<DuplexStream>, thread_id: &str) -> Value {
        let request = read_message(server).await.expect("thread/start request");
        assert_eq!(request["method"], "thread/start");
        let params = &request["params"];
        assert_eq!(params["developerInstructions"], "stay exact");
        let model = params["model"].clone();
        let cwd = params["cwd"].clone();
        let workspace_roots = params["runtimeWorkspaceRoots"].clone();
        send_message(
            server.get_mut(),
            &json!({
                "id": request["id"],
                "result": {
                    "thread": {
                        "id": thread_id,
                        "cliVersion": VERSION,
                        "ephemeral": true,
                        "path": null,
                        "historyMode": "legacy",
                        "modelProvider": "openai",
                        "model": model,
                        "cwd": cwd,
                        "canAcceptDirectInput": true,
                    },
                    "model": model,
                    "modelProvider": "openai",
                    "cwd": cwd,
                    "runtimeWorkspaceRoots": workspace_roots,
                    "instructionSources": [],
                    "approvalPolicy": "never",
                    "approvalsReviewer": "user",
                    "sandbox": {"type": "dangerFullAccess"},
                    "activePermissionProfile": null,
                }
            }),
        )
        .await
        .expect("thread/start response");
        request
    }

    async fn complete_turn(
        server: &mut BufReader<DuplexStream>,
        expected_thread_id: &str,
        turn_id: &str,
        assistant_text: &str,
    ) -> Value {
        let request = read_message(server).await.expect("turn/start request");
        assert_eq!(request["method"], "turn/start");
        assert_eq!(request["params"]["threadId"], expected_thread_id);
        send_message(
            server.get_mut(),
            &json!({
                "id": request["id"],
                "result": {"turn": {"id": turn_id, "status": "inProgress"}}
            }),
        )
        .await
        .expect("turn/start response");
        send_message(
            server.get_mut(),
            &json!({
                "method": "turn/completed",
                "params": {
                    "threadId": expected_thread_id,
                    "turn": {
                        "id": turn_id,
                        "status": "completed",
                        "items": [{
                            "id": format!("message-{turn_id}"),
                            "type": "agentMessage",
                            "text": assistant_text,
                        }]
                    }
                }
            }),
        )
        .await
        .expect("turn completion");
        request
    }

    async fn run_turn(runtime: &CodexRuntime, req: &ChatRequest) -> Result<Vec<ChatEvent>> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let result = runtime.stream_chat(&config(), req, &tx).await;
        drop(tx);
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        result.map(|()| events)
    }

    fn streamed_text(events: &[ChatEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ChatEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn clean_two_turn_history_reuses_one_process_and_thread() {
        let (client, server) = duplex(64 * 1024);
        let fixture = fixture(vec![client]);
        let fake = tokio::spawn(async move {
            let mut server = initialize_server(server).await;
            let first_thread = accept_thread(&mut server, "thread-1").await;
            assert_eq!(first_thread["params"]["model"], "gpt-test");
            let first = complete_turn(&mut server, "thread-1", "turn-1", "one").await;
            let first_input = first["params"]["input"][0]["text"]
                .as_str()
                .expect("first input text");
            assert!(!first_input.contains("stay exact"));
            assert!(!first_input.contains("## System instructions"));
            assert!(first_input.contains("### User\nfirst"));

            let second = complete_turn(&mut server, "thread-1", "turn-2", "two").await;
            assert_eq!(second["params"]["input"][0]["text"], "second");
            assert!(read_message(&mut server).await.is_err());
        });

        let policy = execution_policy();
        let first = request(&policy, vec![Message::User("first".into())]);
        assert_eq!(
            streamed_text(&run_turn(&fixture.runtime, &first).await.unwrap()),
            "one"
        );
        let second = request(
            &policy,
            vec![
                Message::User("first".into()),
                assistant("one"),
                Message::User("second".into()),
            ],
        );
        assert_eq!(
            streamed_text(&run_turn(&fixture.runtime, &second).await.unwrap()),
            "two"
        );
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 1);
        fixture.runtime.shutdown().await.unwrap();
        fixture.runtime.shutdown().await.unwrap();
        assert_eq!(fixture.shutdowns.load(Ordering::SeqCst), 1);
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn history_mismatch_starts_a_fresh_attested_thread() {
        let (first_client, first_server) = duplex(64 * 1024);
        let (second_client, second_server) = duplex(64 * 1024);
        let fixture = fixture(vec![first_client, second_client]);
        let first_fake = tokio::spawn(async move {
            let mut server = initialize_server(first_server).await;
            accept_thread(&mut server, "thread-1").await;
            complete_turn(&mut server, "thread-1", "turn-1", "one").await;
            assert!(read_message(&mut server).await.is_err());
        });
        let second_fake = tokio::spawn(async move {
            let mut server = initialize_server(second_server).await;
            accept_thread(&mut server, "thread-2").await;
            let second = complete_turn(&mut server, "thread-2", "turn-2", "reset").await;
            let input = second["params"]["input"][0]["text"]
                .as_str()
                .expect("replacement input text");
            assert!(input.contains("### Assistant\nchanged"));
            assert!(input.contains("### User\nsecond"));
            assert!(read_message(&mut server).await.is_err());
        });

        let policy = execution_policy();
        run_turn(
            &fixture.runtime,
            &request(&policy, vec![Message::User("first".into())]),
        )
        .await
        .unwrap();
        run_turn(
            &fixture.runtime,
            &request(
                &policy,
                vec![
                    Message::User("first".into()),
                    assistant("changed"),
                    Message::User("second".into()),
                ],
            ),
        )
        .await
        .unwrap();
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 2);
        assert_eq!(fixture.shutdowns.load(Ordering::SeqCst), 1);
        fixture.runtime.shutdown().await.unwrap();
        assert_eq!(fixture.shutdowns.load(Ordering::SeqCst), 2);
        first_fake.await.unwrap();
        second_fake.await.unwrap();
    }

    #[tokio::test]
    async fn dropped_turn_stays_dirty_and_forces_a_fresh_thread() {
        let (first_client, first_server) = duplex(64 * 1024);
        let (second_client, second_server) = duplex(64 * 1024);
        let fixture = fixture(vec![first_client, second_client]);
        let (turn_seen_tx, turn_seen_rx) = oneshot::channel();
        let first_fake = tokio::spawn(async move {
            let mut server = initialize_server(first_server).await;
            accept_thread(&mut server, "thread-1").await;
            let first = read_message(&mut server).await.expect("first turn/start");
            assert_eq!(first["method"], "turn/start");
            send_message(
                server.get_mut(),
                &json!({
                    "id": first["id"],
                    "result": {"turn": {"id": "turn-1", "status": "inProgress"}}
                }),
            )
            .await
            .unwrap();
            turn_seen_tx.send(()).unwrap();

            let interrupt = read_message(&mut server).await.expect("turn/interrupt");
            assert_eq!(interrupt["method"], "turn/interrupt");
            send_message(
                server.get_mut(),
                &json!({"id": interrupt["id"], "result": {}}),
            )
            .await
            .unwrap();
            send_message(
                server.get_mut(),
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-1", "status": "interrupted", "items": []}
                    }
                }),
            )
            .await
            .unwrap();
            assert!(read_message(&mut server).await.is_err());
        });
        let second_fake = tokio::spawn(async move {
            let mut server = initialize_server(second_server).await;
            accept_thread(&mut server, "thread-2").await;
            complete_turn(&mut server, "thread-2", "turn-2", "fresh").await;
            assert!(read_message(&mut server).await.is_err());
        });

        let policy = execution_policy();
        let runtime = fixture.runtime.clone();
        let first = request(&policy, vec![Message::User("first".into())]);
        let active = tokio::spawn(async move {
            let (tx, _rx) = mpsc::unbounded_channel();
            runtime.stream_chat(&config(), &first, &tx).await
        });
        turn_seen_rx.await.unwrap();
        active.abort();
        assert!(active.await.unwrap_err().is_cancelled());

        let retry = request(&policy, vec![Message::User("retry".into())]);
        assert_eq!(
            streamed_text(&run_turn(&fixture.runtime, &retry).await.unwrap()),
            "fresh"
        );
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 2);
        assert_eq!(fixture.shutdowns.load(Ordering::SeqCst), 1);
        fixture.runtime.shutdown().await.unwrap();
        assert_eq!(fixture.shutdowns.load(Ordering::SeqCst), 2);
        first_fake.await.unwrap();
        second_fake.await.unwrap();
    }

    #[tokio::test]
    async fn uncertain_delivery_drops_the_server_without_retrying_the_turn() {
        let (client_one, server_one) = duplex(64 * 1024);
        let (client_two, server_two) = duplex(64 * 1024);
        let fixture = fixture(vec![client_one, client_two]);
        let first_fake = tokio::spawn(async move {
            let mut server = initialize_server(server_one).await;
            accept_thread(&mut server, "thread-1").await;
            let request = read_message(&mut server).await.expect("turn/start");
            assert_eq!(request["method"], "turn/start");
            // EOF after the model-bearing write makes delivery ambiguous.
        });
        let second_fake = tokio::spawn(async move {
            let mut server = initialize_server(server_two).await;
            accept_thread(&mut server, "thread-2").await;
            complete_turn(&mut server, "thread-2", "turn-2", "recovered").await;
            assert!(read_message(&mut server).await.is_err());
        });

        let policy = execution_policy();
        let first = run_turn(
            &fixture.runtime,
            &request(&policy, vec![Message::User("uncertain".into())]),
        )
        .await
        .expect_err("ambiguous delivery must fail");
        assert!(
            first
                .to_string()
                .contains("closed stdout before completing"),
            "{first:#}"
        );
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 1);

        let recovered = run_turn(
            &fixture.runtime,
            &request(&policy, vec![Message::User("retry explicitly".into())]),
        )
        .await
        .unwrap();
        assert_eq!(streamed_text(&recovered), "recovered");
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 2);
        fixture.runtime.shutdown().await.unwrap();
        first_fake.await.unwrap();
        second_fake.await.unwrap();
    }

    #[tokio::test]
    async fn image_turn_replaces_the_process_and_uses_a_bounded_text_handoff() {
        let (first_client, first_server) = duplex(64 * 1024);
        let (second_client, second_server) = duplex(64 * 1024);
        let fixture = fixture(vec![first_client, second_client]);
        let first_fake = tokio::spawn(async move {
            let mut server = initialize_server(first_server).await;
            accept_thread(&mut server, "thread-1").await;
            complete_turn(&mut server, "thread-1", "turn-1", "one").await;
            assert!(read_message(&mut server).await.is_err());
        });
        let second_fake = tokio::spawn(async move {
            let mut server = initialize_server(second_server).await;
            accept_thread(&mut server, "thread-2").await;
            let turn = complete_turn(&mut server, "thread-2", "turn-2", "two").await;
            let input = turn["params"]["input"][0]["text"]
                .as_str()
                .expect("image handoff text");
            assert!(input.contains("[image 1: image/png; binary data omitted]"));
            assert!(input.contains("### User"));
            assert!(input.contains("inspect this"));
            assert!(!input.contains("not-forwarded-base64"));
            assert!(read_message(&mut server).await.is_err());
        });

        let policy = execution_policy();
        run_turn(
            &fixture.runtime,
            &request(&policy, vec![Message::User("first".into())]),
        )
        .await
        .unwrap();
        let events = run_turn(
            &fixture.runtime,
            &request(
                &policy,
                vec![
                    Message::User("first".into()),
                    assistant("one"),
                    Message::User(UserContent::Rich {
                        text: "inspect this".into(),
                        images: vec![ImageData {
                            media_type: "image/png".into(),
                            data: "not-forwarded-base64".into(),
                        }],
                    }),
                ],
            ),
        )
        .await
        .unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            ChatEvent::Notice(message) if message.contains("not yet forwarded")
        )));
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 2);
        fixture.runtime.shutdown().await.unwrap();
        first_fake.await.unwrap();
        second_fake.await.unwrap();
    }

    #[tokio::test]
    async fn closed_event_receiver_prevents_process_and_model_work() {
        let fixture = fixture(Vec::new());
        let policy = execution_policy();
        let request = request(&policy, vec![Message::User("must not run".into())]);
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);

        let error = fixture
            .runtime
            .stream_chat(&config(), &request, &tx)
            .await
            .expect_err("closed receiver must fail before provider work");
        assert!(error.to_string().contains("receiver closed"));
        assert_eq!(fixture.spawns.load(Ordering::SeqCst), 0);
        fixture.runtime.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn aborted_shutdown_is_resumed_and_reaped_by_the_next_shutdown() {
        let (client, server) = duplex(64 * 1024);
        let gate = Arc::new(ShutdownGate::default());
        let fixture = fixture_with_shutdown_gate(vec![client], Some(Arc::clone(&gate)));
        let fake = tokio::spawn(async move {
            let mut server = initialize_server(server).await;
            accept_thread(&mut server, "thread-1").await;
            complete_turn(&mut server, "thread-1", "turn-1", "done").await;
            assert!(read_message(&mut server).await.is_err());
        });
        let policy = execution_policy();
        run_turn(
            &fixture.runtime,
            &request(&policy, vec![Message::User("first".into())]),
        )
        .await
        .unwrap();

        let runtime = fixture.runtime.clone();
        let first_shutdown = tokio::spawn(async move { runtime.shutdown().await });
        tokio::time::timeout(std::time::Duration::from_secs(1), gate.wait_started())
            .await
            .expect("child cleanup must start");
        first_shutdown.abort();
        assert!(first_shutdown.await.unwrap_err().is_cancelled());
        gate.release();

        fixture.runtime.shutdown().await.unwrap();
        assert_eq!(fixture.shutdowns.load(Ordering::SeqCst), 1);
        fake.await.unwrap();
    }
}
