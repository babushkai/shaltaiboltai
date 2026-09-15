//! Sub-agent providers backed by an official CLI (Claude Code) running on the
//! user's subscription. We never see or store a token: the CLI owns its own
//! auth. We spawn it headless, stream its NDJSON events, and adapt them into
//! our provider-agnostic [`ChatEvent`]s. The CLI runs its own tool loop, so our
//! tool definitions and approval UI do not apply inside the child. Every child
//! therefore receives an explicit, fail-closed snapshot of app authority.

use super::{ChatEvent, ChatRequest, Config, Message, RequestPolicy, Usage};
use crate::policy::{ApprovalPolicy, ExecutionPolicy, SandboxMode};
use anyhow::{Context, Result};
use serde_json::Value;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

const STDERR_CAPTURE_BYTES: usize = 16_000;
const MAX_NDJSON_RECORD_BYTES: usize = 1024 * 1024;
const CLAUDE_READ_ONLY_TOOLS: &str = "Read,Glob,Grep";
const CODEX_CONSTRAINED_CONFIG: [&str; 13] = [
    r#"model_provider="openai""#,
    "mcp_servers={}",
    r#"web_search="disabled""#,
    "features.hooks=false",
    "features.plugins=false",
    "features.apps=false",
    "features.enable_mcp_apps=false",
    "features.browser_use=false",
    "features.browser_use_external=false",
    "features.browser_use_full_cdp_access=false",
    "features.in_app_browser=false",
    "features.computer_use=false",
    "features.image_generation=false",
];

#[derive(Clone, Copy)]
enum CliExecutable {
    Claude,
    Codex,
}

impl CliExecutable {
    const fn name(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
        }
    }

    const fn file_name(self) -> &'static str {
        #[cfg(windows)]
        match self {
            Self::Claude => "claude.exe",
            Self::Codex => "codex.exe",
        }
        #[cfg(not(windows))]
        self.name()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedCliCandidate {
    /// Absolute spelling found in PATH. Keep it so a symlink located in a
    /// writable workspace is rejected even when its target is outside it.
    lexical: PathBuf,
    canonical: PathBuf,
    identity: ExecutableIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecutableIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_seconds: i64,
    #[cfg(unix)]
    change_nanoseconds: i64,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    link_count: u64,
    length: u64,
}

static CLAUDE_EXECUTABLES: OnceLock<Vec<ResolvedCliCandidate>> = OnceLock::new();
static CODEX_EXECUTABLES: OnceLock<Vec<ResolvedCliCandidate>> = OnceLock::new();

struct CliLaunch {
    command: tokio::process::Command,
    candidate: &'static ResolvedCliCandidate,
}

#[cfg(unix)]
struct ProcessGroupGuard {
    pgid: libc::pid_t,
    armed: bool,
}

#[cfg(unix)]
impl ProcessGroupGuard {
    fn new(child_id: u32) -> Result<Self> {
        let pgid = libc::pid_t::try_from(child_id).context("child process id exceeds pid_t")?;
        Ok(Self { pgid, armed: true })
    }

    fn terminate(&mut self) -> std::io::Result<()> {
        if !self.armed {
            return Ok(());
        }
        // SAFETY: the spawned CLI is configured as its own process-group
        // leader. A negative pid therefore targets only that owned group.
        if unsafe { libc::kill(-self.pgid, libc::SIGKILL) } == 0 {
            self.armed = false;
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            self.armed = false;
            Ok(())
        } else {
            Err(error)
        }
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

/// Snapshot every absolute, canonical CLI candidate once for this process.
/// PATH entries relative to cwd are ignored. Discovery only inspects metadata;
/// the actual execution policy chooses a trusted candidate immediately before
/// the first and only launch.
fn resolved_cli_candidates(cli: CliExecutable) -> &'static [ResolvedCliCandidate] {
    let cache = match cli {
        CliExecutable::Claude => &CLAUDE_EXECUTABLES,
        CliExecutable::Codex => &CODEX_EXECUTABLES,
    };
    cache.get_or_init(|| resolve_cli_candidates_from_environment(cli))
}

fn resolve_cli_candidates_from_environment(cli: CliExecutable) -> Vec<ResolvedCliCandidate> {
    std::env::var_os("PATH").map_or_else(Vec::new, |path| {
        resolve_cli_candidates_from_path(cli.file_name(), &path)
    })
}

fn resolve_cli_candidates_from_path(file_name: &str, path: &OsStr) -> Vec<ResolvedCliCandidate> {
    let mut resolved = Vec::new();
    for directory in std::env::split_paths(path) {
        if !directory.is_absolute() {
            continue;
        }
        let lexical = directory.join(file_name);
        let Ok(canonical) = std::fs::canonicalize(&lexical) else {
            continue;
        };
        let Some(identity) = executable_identity(&canonical) else {
            continue;
        };
        resolved.push(ResolvedCliCandidate {
            lexical,
            canonical,
            identity,
        });
    }
    resolved
}

fn executable_identity(path: &Path) -> Option<ExecutableIdentity> {
    let Ok(metadata) = std::fs::metadata(path) else {
        return None;
    };
    if !metadata.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.mode() & 0o111 == 0 {
            return None;
        }
        Some(ExecutableIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
            change_seconds: metadata.ctime(),
            change_nanoseconds: metadata.ctime_nsec(),
            mode: metadata.mode(),
            link_count: metadata.nlink(),
            length: metadata.len(),
        })
    }
    #[cfg(not(unix))]
    {
        // Windows needs a stable file-handle identity plus content binding to
        // close replacement races. Until that is implemented, CLI providers
        // are intentionally unavailable instead of trusting size/mtime.
        let _ = metadata;
        None
    }
}

fn candidate_is_unchanged(candidate: &ResolvedCliCandidate) -> bool {
    #[cfg(unix)]
    if candidate.identity.link_count != 1 {
        return false;
    }
    std::fs::canonicalize(&candidate.canonical).is_ok_and(|current| current == candidate.canonical)
        && executable_identity(&candidate.canonical).is_some_and(|identity| {
            #[cfg(unix)]
            if identity.link_count != 1 {
                return false;
            }
            identity == candidate.identity
        })
}

fn cli_candidate_for_policy(
    cli: CliExecutable,
    execution_policy: &ExecutionPolicy,
) -> Result<&'static ResolvedCliCandidate> {
    let writable_roots = model_writable_roots(execution_policy)?;
    select_cli_candidate(cli.name(), resolved_cli_candidates(cli), &writable_roots)
}

fn model_writable_roots(execution_policy: &ExecutionPolicy) -> Result<Vec<PathBuf>> {
    let mut roots = execution_policy.effective_user_visible_roots().to_vec();
    #[cfg(unix)]
    add_cli_trust_root(&mut roots, Path::new("/tmp"))?;
    if let Some(tmpdir) = std::env::var_os("TMPDIR").filter(|value| !value.is_empty()) {
        add_cli_trust_root(&mut roots, Path::new(&tmpdir))?;
    }
    Ok(roots)
}

fn add_cli_trust_root(roots: &mut Vec<PathBuf>, path: &Path) -> Result<()> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("canonicalize model-writable root {}", path.display()))?;
    if !std::fs::metadata(&canonical)
        .with_context(|| format!("inspect model-writable root {}", canonical.display()))?
        .is_dir()
    {
        anyhow::bail!(
            "model-writable root is not a directory: {}",
            canonical.display()
        );
    }
    if !roots.contains(&canonical) {
        roots.push(canonical);
    }
    Ok(())
}

#[cfg(test)]
fn select_cli_executable<'a>(
    name: &str,
    candidates: &'a [ResolvedCliCandidate],
    workspace_roots: &[PathBuf],
) -> Result<&'a Path> {
    Ok(select_cli_candidate(name, candidates, workspace_roots)?
        .canonical
        .as_path())
}

fn select_cli_candidate<'a>(
    name: &str,
    candidates: &'a [ResolvedCliCandidate],
    workspace_roots: &[PathBuf],
) -> Result<&'a ResolvedCliCandidate> {
    let Some(candidate) = candidates.first() else {
        anyhow::bail!("`{name}` was not found at an executable absolute PATH entry");
    };
    if !candidate_is_unchanged(candidate) {
        anyhow::bail!("refusing to launch `{name}` because it changed after discovery");
    }
    if workspace_roots
        .iter()
        .any(|root| candidate.lexical.starts_with(root) || candidate.canonical.starts_with(root))
    {
        anyhow::bail!("refusing to launch `{name}` from a writable workspace root");
    }
    Ok(candidate)
}

/// Discovery is deliberately non-executing. The actual workspace may be
/// selected later with `-C`, so trust is enforced only when a turn has captured
/// its immutable execution policy.
pub async fn claude_available() -> bool {
    !resolved_cli_candidates(CliExecutable::Claude).is_empty()
}

pub async fn stream_chat_claude(
    _config: &Config,
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
    let cmd = fresh_claude_command(model, req.policy, &req.execution_policy)?;
    drive_cli(cmd, "claude", &prompt, tx, handle_claude_event).await
}

fn fresh_claude_command(
    model: Option<&str>,
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
) -> Result<CliLaunch> {
    let effective_sandbox = effective_sandbox(request_policy, execution_policy);
    if effective_sandbox == SandboxMode::DangerFullAccess {
        reject_unmediated_untrusted_writes(request_policy, execution_policy, "Claude Code")?;
    }
    let candidate = cli_candidate_for_policy(CliExecutable::Claude, execution_policy)?;
    Ok(CliLaunch {
        command: build_claude_command(
            &candidate.canonical,
            model,
            request_policy,
            execution_policy,
        ),
        candidate,
    })
}

fn build_claude_command(
    executable: &Path,
    model: Option<&str>,
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
) -> tokio::process::Command {
    let effective_sandbox = effective_sandbox(request_policy, execution_policy);
    // Claude has no OS-enforced path deny that can preserve `.git`, `.agents`,
    // and `.codex`. Both constrained app modes therefore map to an advisory
    // plan session; only explicit Full Access receives mutation tools.
    let constrained = effective_sandbox != SandboxMode::DangerFullAccess;
    let permission_mode = if constrained {
        "plan"
    } else {
        "bypassPermissions"
    };
    let mut cmd = tokio::process::Command::new(executable);
    cmd.current_dir(execution_policy.workspace().cwd());
    cmd.arg("--print").arg("--no-session-persistence");
    if constrained {
        // Constrained children must not gain tools or authority from project or
        // user customizations. With no `--mcp-config`, strict MCP mode also
        // makes the empty MCP set explicit.
        cmd.arg("--safe-mode").arg("--strict-mcp-config");
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
    if constrained {
        cmd.arg("--tools").arg(CLAUDE_READ_ONLY_TOOLS);
    }
    cmd
}

fn effective_sandbox(
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
) -> SandboxMode {
    match request_policy {
        RequestPolicy::ReadOnly => SandboxMode::ReadOnly,
        RequestPolicy::Interactive => execution_policy.sandbox_mode(),
    }
}

fn reject_unmediated_untrusted_writes(
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
    provider: &str,
) -> Result<()> {
    if request_policy == RequestPolicy::Interactive
        && execution_policy.approval_policy() == ApprovalPolicy::Untrusted
        && execution_policy.sandbox_mode() != SandboxMode::ReadOnly
    {
        anyhow::bail!(
            "{provider} cannot run with untrusted write approvals because its headless mode \
             cannot surface inner approval prompts; select Read Only or Ask for approval"
        );
    }
    Ok(())
}

fn additional_workspace_roots(
    execution_policy: &ExecutionPolicy,
) -> impl Iterator<Item = &std::path::Path> {
    let cwd = execution_policy.workspace().cwd();
    execution_policy
        .effective_user_visible_roots()
        .iter()
        .map(std::path::PathBuf::as_path)
        .filter(move |root| *root != cwd)
}

async fn drain_stderr_bounded<R>(mut reader: R) -> String
where
    R: AsyncRead + Unpin,
{
    let mut captured = Vec::with_capacity(STDERR_CAPTURE_BYTES);
    let mut buffer = [0_u8; 8 * 1024];
    let mut truncated = false;
    while let Ok(read) = reader.read(&mut buffer).await {
        if read == 0 {
            break;
        }
        let retained = read.min(STDERR_CAPTURE_BYTES.saturating_sub(captured.len()));
        captured.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    let mut rendered = String::from_utf8_lossy(&captured).into_owned();
    if truncated {
        if !rendered.ends_with('\n') && !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str("…[stderr truncated]\n");
    }
    rendered
}

async fn finish_stderr_drain(mut task: tokio::task::JoinHandle<String>) -> String {
    match tokio::time::timeout(std::time::Duration::from_secs(1), &mut task).await {
        Ok(Ok(stderr)) => stderr,
        Ok(Err(_)) => String::new(),
        Err(_) => {
            task.abort();
            let _ = task.await;
            "…[stderr drain timed out]\n".into()
        }
    }
}

/// Fill `record` with one NDJSON frame without ever retaining more than the
/// configured record limit. `BufReader::lines` cannot be used here because a
/// hostile newline-free child can make it allocate until process exhaustion.
async fn read_bounded_ndjson_record<R>(
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
                let overflow = record.len().saturating_add(retained) > MAX_NDJSON_RECORD_BYTES;
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
            anyhow::bail!("CLI stdout NDJSON record exceeded {MAX_NDJSON_RECORD_BYTES} bytes");
        }
        if complete {
            return Ok(Some(()));
        }
        if eof {
            return Ok((!record.is_empty()).then_some(()));
        }
    }
}

/// Shared subprocess driver: spawn the CLI, stream its NDJSON stdout through
/// `handle` (which returns true on the turn's terminal event), drain stderr so
/// the pipe never blocks, and surface a useful error if the turn never
/// completed.
async fn drive_cli(
    launch: CliLaunch,
    name: &str,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
    handle: impl Fn(&Value, &UnboundedSender<ChatEvent>) -> bool,
) -> Result<()> {
    drive_inner(
        launch.command,
        Some(launch.candidate),
        name,
        prompt,
        tx,
        handle,
    )
    .await
}

#[cfg(test)]
async fn drive(
    cmd: tokio::process::Command,
    name: &str,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
    handle: impl Fn(&Value, &UnboundedSender<ChatEvent>) -> bool,
) -> Result<()> {
    drive_inner(cmd, None, name, prompt, tx, handle).await
}

async fn drive_inner(
    mut cmd: tokio::process::Command,
    expected_candidate: Option<&ResolvedCliCandidate>,
    name: &str,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
    handle: impl Fn(&Value, &UnboundedSender<ChatEvent>) -> bool,
) -> Result<()> {
    #[cfg(unix)]
    cmd.as_std_mut().process_group(0);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if expected_candidate.is_some_and(|candidate| !candidate_is_unchanged(candidate)) {
        anyhow::bail!("refusing to launch `{name}` because it changed after discovery");
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to launch `{name}` — is it installed and signed in?"))?;
    #[cfg(unix)]
    let mut process_group = ProcessGroupGuard::new(
        child
            .id()
            .context("spawned CLI did not expose a process id")?,
    )?;
    let stdout = child.stdout.take().context("no stdout")?;
    let stderr = child.stderr.take().context("no stderr")?;
    let mut stdin = child.stdin.take().context("no stdin")?;

    // Drain every stderr byte so the child cannot block, but retain only a
    // fixed diagnostic prefix even when it never emits a newline.
    let stderr_task = tokio::spawn(drain_stderr_bounded(stderr));

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

    let mut stdout = BufReader::new(stdout);
    let mut record = Vec::new();
    let mut saw_result = false;
    loop {
        let frame = match read_bounded_ndjson_record(&mut stdout, &mut record).await {
            Ok(frame) => frame,
            Err(error) => {
                drop(stdout);
                #[cfg(unix)]
                let cleanup = async {
                    let termination = process_group
                        .terminate()
                        .context("failed to terminate CLI descendants");
                    if termination.is_err() {
                        let _ = child.kill().await;
                    }
                    let reaped = child.wait().await.context("failed to reap CLI").map(|_| ());
                    termination.and(reaped)
                }
                .await;
                #[cfg(not(unix))]
                let cleanup = async {
                    let _ = child.kill().await;
                    child.wait().await.context("failed to reap CLI")?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                let _ = finish_stderr_drain(stderr_task).await;
                cleanup?;
                return Err(error);
            }
        };
        let Some(()) = frame else {
            break;
        };
        if record.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        if let Ok(event) = serde_json::from_slice::<Value>(&record) {
            if handle(&event, tx) {
                saw_result = true;
                break;
            }
        }
    }
    drop(stdout);

    let status = if saw_result {
        match tokio::time::timeout(std::time::Duration::from_millis(250), child.wait()).await {
            Ok(status) => status?,
            Err(_) => {
                #[cfg(unix)]
                process_group
                    .terminate()
                    .context("failed to terminate CLI descendants")?;
                #[cfg(not(unix))]
                child.kill().await.context("failed to terminate CLI")?;
                child.wait().await?
            }
        }
    } else {
        child.wait().await?
    };
    #[cfg(unix)]
    process_group
        .terminate()
        .context("failed to terminate CLI descendants")?;
    let stderr = finish_stderr_drain(stderr_task).await;
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
    !resolved_cli_candidates(CliExecutable::Codex).is_empty()
}

/// Model discovery must not execute a PATH candidate before the turn's `-C`
/// workspace and execution policy are known. The stable bare Codex selector is
/// still advertised by the caller; explicit model IDs remain accepted.
pub async fn codex_model_ids() -> Vec<String> {
    Vec::new()
}

pub async fn stream_chat_codex(
    _config: &Config,
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
    let cmd = fresh_codex_command(model, req.policy, &req.execution_policy)?;
    drive_cli(cmd, "codex", &prompt, tx, handle_codex_event).await
}

fn fresh_codex_command(
    model: Option<&str>,
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
) -> Result<CliLaunch> {
    reject_unmediated_untrusted_writes(request_policy, execution_policy, "Codex")?;
    let candidate = cli_candidate_for_policy(CliExecutable::Codex, execution_policy)?;
    Ok(CliLaunch {
        command: build_codex_command(
            &candidate.canonical,
            model,
            request_policy,
            execution_policy,
        ),
        candidate,
    })
}

fn build_codex_command(
    executable: &Path,
    model: Option<&str>,
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
) -> tokio::process::Command {
    // Every request starts in a fresh, explicitly sandboxed process. Context is
    // carried in `prompt`, never inferred from another cwd-global CLI session.
    let sandbox = effective_sandbox(request_policy, execution_policy);
    let mut cmd = tokio::process::Command::new(executable);
    cmd.current_dir(execution_policy.workspace().cwd());
    cmd.arg("exec").arg("--ephemeral");
    if sandbox != SandboxMode::DangerFullAccess {
        // Constrained runs must not inherit user integrations or execution
        // rules. Project config is still discovered by Codex, so reset every
        // ambient integration surface explicitly and require recognized config
        // keys. An older incompatible CLI then fails closed at startup.
        cmd.arg("--ignore-user-config")
            .arg("--ignore-rules")
            .arg("--strict-config");
        for constraint in CODEX_CONSTRAINED_CONFIG {
            cmd.arg("-c").arg(constraint);
        }
    }
    cmd.arg("--sandbox").arg(sandbox.to_string());
    // `codex exec` cannot route its inner approval requests to this app. Make
    // its fail-closed headless default explicit so ambient configuration can
    // never turn those requests into automatic approvals.
    cmd.arg("-c").arg(r#"approval_policy="never""#);
    if sandbox == SandboxMode::WorkspaceWrite {
        // A user's Codex config may broaden workspace mode with network,
        // legacy roots, or writable temp directories. Reset those inherited
        // knobs before adding only the canonical roots in this snapshot.
        for constraint in [
            "sandbox_workspace_write.network_access=false",
            "sandbox_workspace_write.writable_roots=[]",
            "sandbox_workspace_write.exclude_tmpdir_env_var=true",
            "sandbox_workspace_write.exclude_slash_tmp=true",
        ] {
            cmd.arg("-c").arg(constraint);
        }
        for root in additional_workspace_roots(execution_policy) {
            cmd.arg("--add-dir").arg(root);
        }
    }
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
    use crate::policy::Workspace;
    use crate::providers::{ImageData, ModelEntry, ProviderKind, ToolCall, UserContent};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::sync::mpsc::unbounded_channel;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    struct PolicyFixture {
        base: PathBuf,
        extra: PathBuf,
        policy: ExecutionPolicy,
    }

    impl PolicyFixture {
        fn new(sandbox: SandboxMode, approval: ApprovalPolicy) -> Self {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let base = std::env::temp_dir().join(format!(
                "shaltaiboltai-provider-policy-{}-{sequence}",
                std::process::id()
            ));
            let cwd = base.join("workspace");
            let extra = base.join("extra");
            std::fs::create_dir_all(&cwd).expect("create test workspace");
            std::fs::create_dir_all(&extra).expect("create additional test root");
            let workspace =
                Workspace::from_roots(&cwd, [&extra]).expect("canonical test workspace");
            let extra = std::fs::canonicalize(extra).expect("canonical additional test root");
            Self {
                base,
                extra,
                policy: ExecutionPolicy::from_parts(workspace, sandbox, approval),
            }
        }

        fn cwd(&self) -> &Path {
            self.policy.workspace().cwd()
        }
    }

    impl Drop for PolicyFixture {
        fn drop(&mut self) {
            let safe = self
                .base
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("shaltaiboltai-provider-policy-"));
            if safe {
                let _ = std::fs::remove_dir_all(&self.base);
            }
        }
    }

    fn default_execution_policy() -> ExecutionPolicy {
        let cwd = std::env::current_dir().expect("test current directory");
        ExecutionPolicy::new(Workspace::new(cwd).expect("canonical current directory"))
    }

    fn test_cli_executable(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("shaltaiboltai-provider-test-cli")
            .join(name)
    }

    fn test_claude_command(
        model: Option<&str>,
        request_policy: RequestPolicy,
        execution_policy: &ExecutionPolicy,
    ) -> Result<tokio::process::Command> {
        if effective_sandbox(request_policy, execution_policy) == SandboxMode::DangerFullAccess {
            reject_unmediated_untrusted_writes(request_policy, execution_policy, "Claude Code")?;
        }
        Ok(build_claude_command(
            &test_cli_executable("claude"),
            model,
            request_policy,
            execution_policy,
        ))
    }

    fn test_codex_command(
        model: Option<&str>,
        request_policy: RequestPolicy,
        execution_policy: &ExecutionPolicy,
    ) -> Result<tokio::process::Command> {
        reject_unmediated_untrusted_writes(request_policy, execution_policy, "Codex")?;
        Ok(build_codex_command(
            &test_cli_executable("codex"),
            model,
            request_policy,
            execution_policy,
        ))
    }

    fn create_fake_executable(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("fake executable parent"))
            .expect("create fake executable directory");
        std::fs::write(path, b"#!/bin/sh\nexit 99\n").expect("write fake executable");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .expect("mark fake executable executable");
        }
    }

    fn request(system: &str, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: ModelEntry {
                provider: ProviderKind::Codex,
                id: "codex".into(),
            },
            system: system.into(),
            messages,
            tools: Vec::new(),
            execution_policy: default_execution_policy(),
            policy: RequestPolicy::Interactive,
            force_full_handoff: false,
        }
    }

    fn command_args(command: &tokio::process::Command) -> Vec<String> {
        command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn command_cwd(command: &tokio::process::Command) -> Option<&Path> {
        command.as_std().get_current_dir()
    }

    fn has_arg_pair(args: &[String], flag: &str, value: &str) -> bool {
        args.windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    }

    fn values_for<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
        args.windows(2)
            .filter(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
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
    fn cli_resolution_ignores_relative_path_entries() {
        let fixture = PolicyFixture::new(SandboxMode::WorkspaceWrite, ApprovalPolicy::OnRequest);
        let trusted_bin = fixture.base.join("trusted-bin");
        let trusted = trusted_bin.join("codex");
        create_fake_executable(&trusted);
        let path =
            std::env::join_paths([PathBuf::from("."), trusted_bin]).expect("join synthetic PATH");

        let candidates = resolve_cli_candidates_from_path("codex", &path);
        assert_eq!(candidates.len(), 1, "relative PATH entries must be skipped");
        assert_eq!(
            select_cli_executable(
                "codex",
                &candidates,
                fixture.policy.effective_user_visible_roots(),
            )
            .expect("trusted absolute candidate"),
            std::fs::canonicalize(trusted).unwrap()
        );
    }

    #[test]
    fn first_workspace_path_candidate_fails_without_falling_through() {
        let fixture = PolicyFixture::new(SandboxMode::DangerFullAccess, ApprovalPolicy::Never);
        let workspace_bin = fixture.cwd().join("bin");
        let workspace_cli = workspace_bin.join("claude");
        create_fake_executable(&workspace_cli);
        let trusted_bin = fixture.base.join("trusted-bin");
        create_fake_executable(&trusted_bin.join("claude"));
        let path = std::env::join_paths([workspace_bin, trusted_bin]).expect("join synthetic PATH");

        let candidates = resolve_cli_candidates_from_path("claude", &path);
        assert_eq!(candidates.len(), 2);
        let error = select_cli_executable(
            "claude",
            &candidates,
            fixture.policy.effective_user_visible_roots(),
        )
        .expect_err("a first-hit workspace executable must fail closed");
        assert!(error.to_string().contains("writable workspace root"));
    }

    #[test]
    fn cli_candidate_under_model_writable_temp_root_is_rejected() {
        let fixture = PolicyFixture::new(SandboxMode::DangerFullAccess, ApprovalPolicy::Never);
        let temp_bin = fixture.base.join("temp-bin");
        let temp_cli = temp_bin.join("codex");
        create_fake_executable(&temp_cli);
        let path = std::env::join_paths([temp_bin]).expect("join synthetic PATH");
        let candidates = resolve_cli_candidates_from_path("codex", &path);
        let writable_roots = model_writable_roots(&fixture.policy).expect("model writable roots");

        assert_eq!(candidates.len(), 1);
        assert!(select_cli_executable("codex", &candidates, &writable_roots).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn workspace_symlink_to_trusted_cli_is_rejected_by_lexical_path() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest);
        let trusted = fixture.base.join("trusted-bin/claude");
        create_fake_executable(&trusted);
        let workspace_bin = fixture.cwd().join("bin");
        std::fs::create_dir_all(&workspace_bin).expect("create workspace bin");
        std::os::unix::fs::symlink(&trusted, workspace_bin.join("claude"))
            .expect("create workspace CLI symlink");
        let path = std::env::join_paths([workspace_bin]).expect("join synthetic PATH");

        let candidates = resolve_cli_candidates_from_path("claude", &path);
        assert_eq!(candidates.len(), 1);
        assert_ne!(candidates[0].lexical, candidates[0].canonical);
        assert!(select_cli_executable(
            "claude",
            &candidates,
            fixture.policy.effective_user_visible_roots(),
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn hard_link_alias_invalidates_a_resolved_cli_candidate() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest);
        let trusted_bin = fixture.base.join("trusted-bin");
        let trusted = trusted_bin.join("codex");
        create_fake_executable(&trusted);
        let path = std::env::join_paths([trusted_bin]).expect("join synthetic PATH");
        let candidates = resolve_cli_candidates_from_path("codex", &path);
        assert_eq!(candidates.len(), 1);

        let alias = fixture.cwd().join("codex-hard-link");
        std::fs::hard_link(&trusted, &alias).expect("create workspace hard-link alias");
        std::fs::write(&alias, b"#!/bin/sh\nprintf compromised\n")
            .expect("mutate CLI through workspace hard-link");
        std::fs::remove_file(&alias).expect("remove workspace hard-link alias");
        let error = select_cli_executable(
            "codex",
            &candidates,
            fixture.policy.effective_user_visible_roots(),
        )
        .expect_err("a hard-link alias must invalidate the cached executable identity");
        assert!(error.to_string().contains("changed after discovery"));

        std::fs::hard_link(&trusted, &alias).expect("recreate workspace hard-link alias");
        let aliased = resolve_cli_candidates_from_path("codex", &path);
        assert_eq!(aliased.len(), 1, "first executable PATH hit is retained");
        assert!(select_cli_executable(
            "codex",
            &aliased,
            fixture.policy.effective_user_visible_roots(),
        )
        .is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_revalidates_cli_identity_immediately_before_spawn() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest);
        let trusted_bin = fixture.base.join("trusted-bin");
        let trusted = trusted_bin.join("claude");
        create_fake_executable(&trusted);
        let path = std::env::join_paths([trusted_bin]).expect("join synthetic PATH");
        let candidates = resolve_cli_candidates_from_path("claude", &path);
        let candidate = candidates.first().expect("resolved fake CLI");
        let command = tokio::process::Command::new(&candidate.canonical);

        let alias = fixture.cwd().join("claude-hard-link");
        std::fs::hard_link(&trusted, &alias).expect("create workspace hard-link alias");
        std::fs::write(&alias, b"#!/bin/sh\nprintf compromised\n")
            .expect("mutate CLI through workspace hard-link");
        std::fs::remove_file(alias).expect("remove workspace hard-link alias");
        let (tx, _rx) = unbounded_channel();

        let error = drive_inner(
            command,
            Some(candidate),
            "claude",
            "prompt",
            &tx,
            handle_claude_event,
        )
        .await
        .expect_err("driver must revalidate before spawn");
        assert!(error.to_string().contains("changed after discovery"));
    }

    #[tokio::test]
    async fn codex_model_discovery_never_executes_a_path_candidate() {
        assert!(codex_model_ids().await.is_empty());
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
    fn cli_commands_start_fresh_in_the_captured_workspace() {
        let fixture = PolicyFixture::new(SandboxMode::WorkspaceWrite, ApprovalPolicy::OnRequest);
        let claude = test_claude_command(None, RequestPolicy::Interactive, &fixture.policy)
            .expect("workspace-write Claude command");
        let claude_args = command_args(&claude);
        assert_eq!(command_cwd(&claude), Some(fixture.cwd()));
        assert!(!claude_args.iter().any(|arg| arg == "--continue"));
        assert!(claude_args
            .iter()
            .any(|arg| arg == "--no-session-persistence"));
        assert!(has_arg_pair(&claude_args, "--input-format", "text"));
        assert!(has_arg_pair(&claude_args, "--permission-mode", "plan"));
        assert!(claude_args.iter().any(|arg| arg == "--safe-mode"));
        assert!(claude_args.iter().any(|arg| arg == "--strict-mcp-config"));
        assert!(has_arg_pair(
            &claude_args,
            "--tools",
            CLAUDE_READ_ONLY_TOOLS
        ));
        let extra = fixture.extra.to_string_lossy();
        assert!(values_for(&claude_args, "--add-dir").is_empty());
        assert!(!claude_args.iter().any(|arg| arg == "prompt"));

        let codex = test_codex_command(None, RequestPolicy::Interactive, &fixture.policy)
            .expect("workspace-write Codex command");
        let codex_args = command_args(&codex);
        assert_eq!(command_cwd(&codex), Some(fixture.cwd()));
        assert_eq!(codex_args.first().map(String::as_str), Some("exec"));
        assert!(!codex_args
            .iter()
            .any(|arg| arg == "resume" || arg == "--last"));
        assert!(codex_args.iter().any(|arg| arg == "--ephemeral"));
        assert!(codex_args.iter().any(|arg| arg == "--ignore-user-config"));
        assert!(codex_args.iter().any(|arg| arg == "--ignore-rules"));
        assert!(codex_args.iter().any(|arg| arg == "--strict-config"));
        assert!(has_arg_pair(&codex_args, "--sandbox", "workspace-write"));
        assert!(has_arg_pair(
            &codex_args,
            "-c",
            r#"approval_policy="never""#
        ));
        for constraint in CODEX_CONSTRAINED_CONFIG {
            assert!(has_arg_pair(&codex_args, "-c", constraint));
        }
        for constraint in [
            "sandbox_workspace_write.network_access=false",
            "sandbox_workspace_write.writable_roots=[]",
            "sandbox_workspace_write.exclude_tmpdir_env_var=true",
            "sandbox_workspace_write.exclude_slash_tmp=true",
        ] {
            assert!(has_arg_pair(&codex_args, "-c", constraint));
        }
        assert_eq!(values_for(&codex_args, "--add-dir"), vec![extra.as_ref()]);
        assert_eq!(codex_args.last().map(String::as_str), Some("-"));
    }

    #[test]
    fn explicit_cli_models_are_forwarded_once() {
        let policy = default_execution_policy();
        let claude_args = command_args(
            &test_claude_command(Some("sonnet"), RequestPolicy::Interactive, &policy)
                .expect("Claude command"),
        );
        assert_eq!(
            claude_args
                .windows(2)
                .filter(|args| args[0] == "--model" && args[1] == "sonnet")
                .count(),
            1
        );

        let codex_args = command_args(
            &test_codex_command(Some("gpt-5.6-sol"), RequestPolicy::Interactive, &policy)
                .expect("Codex command"),
        );
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
    fn advisory_policy_forces_read_only_over_full_execution_authority() {
        let fixture = PolicyFixture::new(SandboxMode::DangerFullAccess, ApprovalPolicy::Never);
        let claude = test_claude_command(Some("sonnet"), RequestPolicy::ReadOnly, &fixture.policy)
            .expect("read-only Claude advisory command");
        let claude_args = command_args(&claude);
        assert_eq!(command_cwd(&claude), Some(fixture.cwd()));
        assert!(claude_args.iter().any(|arg| arg == "--safe-mode"));
        assert!(claude_args.iter().any(|arg| arg == "--strict-mcp-config"));
        assert!(has_arg_pair(&claude_args, "--permission-mode", "plan"));
        assert!(has_arg_pair(
            &claude_args,
            "--tools",
            CLAUDE_READ_ONLY_TOOLS
        ));
        assert!(!claude_args
            .iter()
            .any(|arg| arg == "acceptEdits" || arg == "bypassPermissions"));
        assert!(values_for(&claude_args, "--add-dir").is_empty());

        let codex = test_codex_command(
            Some("gpt-5.6-sol"),
            RequestPolicy::ReadOnly,
            &fixture.policy,
        )
        .expect("read-only Codex advisory command");
        let codex_args = command_args(&codex);
        assert_eq!(command_cwd(&codex), Some(fixture.cwd()));
        assert!(has_arg_pair(&codex_args, "--sandbox", "read-only"));
        assert!(codex_args.iter().any(|arg| arg == "--ignore-user-config"));
        assert!(codex_args.iter().any(|arg| arg == "--ignore-rules"));
        assert!(codex_args.iter().any(|arg| arg == "--strict-config"));
        for constraint in CODEX_CONSTRAINED_CONFIG {
            assert!(has_arg_pair(&codex_args, "-c", constraint));
        }
        assert!(!codex_args.iter().any(|arg| arg == "danger-full-access"));
        assert!(values_for(&codex_args, "--add-dir").is_empty());
    }

    #[test]
    fn interactive_cli_modes_follow_the_execution_policy_exactly() {
        for (sandbox, claude_mode) in [
            (SandboxMode::ReadOnly, "plan"),
            (SandboxMode::WorkspaceWrite, "plan"),
            (SandboxMode::DangerFullAccess, "bypassPermissions"),
        ] {
            let approval = if sandbox == SandboxMode::DangerFullAccess {
                ApprovalPolicy::Never
            } else {
                ApprovalPolicy::OnRequest
            };
            let fixture = PolicyFixture::new(sandbox, approval);
            let claude = test_claude_command(None, RequestPolicy::Interactive, &fixture.policy)
                .expect("Claude command for policy mode");
            let claude_args = command_args(&claude);
            assert_eq!(command_cwd(&claude), Some(fixture.cwd()));
            assert!(has_arg_pair(&claude_args, "--permission-mode", claude_mode));
            let constrained = sandbox != SandboxMode::DangerFullAccess;
            assert_eq!(
                claude_args.iter().any(|arg| arg == "--safe-mode"),
                constrained
            );
            assert_eq!(
                claude_args.iter().any(|arg| arg == "--strict-mcp-config"),
                constrained
            );
            let expected_claude_tools = match sandbox {
                SandboxMode::ReadOnly | SandboxMode::WorkspaceWrite => {
                    vec![CLAUDE_READ_ONLY_TOOLS]
                }
                SandboxMode::DangerFullAccess => Vec::new(),
            };
            assert_eq!(values_for(&claude_args, "--tools"), expected_claude_tools);

            let codex = test_codex_command(None, RequestPolicy::Interactive, &fixture.policy)
                .expect("Codex command for policy mode");
            let codex_args = command_args(&codex);
            assert_eq!(command_cwd(&codex), Some(fixture.cwd()));
            assert!(has_arg_pair(&codex_args, "--sandbox", &sandbox.to_string()));
            assert!(has_arg_pair(
                &codex_args,
                "-c",
                r#"approval_policy="never""#
            ));
            assert_eq!(
                codex_args.iter().any(|arg| arg == "--ignore-user-config"),
                constrained
            );
            assert_eq!(
                codex_args.iter().any(|arg| arg == "--ignore-rules"),
                constrained
            );
            assert_eq!(
                codex_args.iter().any(|arg| arg == "--strict-config"),
                constrained
            );
            for constraint in CODEX_CONSTRAINED_CONFIG {
                assert_eq!(has_arg_pair(&codex_args, "-c", constraint), constrained);
            }

            let extra = fixture.extra.to_string_lossy();
            let expected_extra = if sandbox == SandboxMode::WorkspaceWrite {
                vec![extra.as_ref()]
            } else {
                Vec::new()
            };
            assert!(values_for(&claude_args, "--add-dir").is_empty());
            assert_eq!(values_for(&codex_args, "--add-dir"), expected_extra);

            for constraint in [
                "sandbox_workspace_write.network_access=false",
                "sandbox_workspace_write.writable_roots=[]",
                "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                "sandbox_workspace_write.exclude_slash_tmp=true",
            ] {
                assert_eq!(
                    has_arg_pair(&codex_args, "-c", constraint),
                    sandbox == SandboxMode::WorkspaceWrite
                );
            }

            assert!(!codex_args
                .iter()
                .any(|arg| arg == "--dangerously-bypass-approvals-and-sandbox"));
        }
    }

    #[test]
    fn untrusted_writes_fail_closed_in_headless_providers() {
        let workspace = PolicyFixture::new(SandboxMode::WorkspaceWrite, ApprovalPolicy::Untrusted);
        let claude = test_claude_command(None, RequestPolicy::Interactive, &workspace.policy)
            .expect("Claude workspace mode is narrowed to advisory-only");
        let claude_args = command_args(&claude);
        assert!(has_arg_pair(&claude_args, "--permission-mode", "plan"));
        assert!(has_arg_pair(
            &claude_args,
            "--tools",
            CLAUDE_READ_ONLY_TOOLS
        ));
        let codex = test_codex_command(None, RequestPolicy::Interactive, &workspace.policy)
            .expect_err("Codex must not silently accept untrusted writes");
        assert!(codex.to_string().contains("cannot surface inner approval"));

        let full = PolicyFixture::new(SandboxMode::DangerFullAccess, ApprovalPolicy::Untrusted);
        let claude = test_claude_command(None, RequestPolicy::Interactive, &full.policy)
            .expect_err("Claude must not silently accept untrusted Full Access");
        assert!(claude.to_string().contains("cannot surface inner approval"));
        let codex = test_codex_command(None, RequestPolicy::Interactive, &full.policy)
            .expect_err("Codex must not silently accept untrusted Full Access");
        assert!(codex.to_string().contains("cannot surface inner approval"));
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
        let mut command = tokio::process::Command::new("/bin/sh");
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

    #[cfg(unix)]
    #[tokio::test]
    async fn newline_free_multi_megabyte_stderr_is_drained_with_bounded_capture() {
        let mut command = tokio::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("/usr/bin/yes x | /usr/bin/head -c 4194304 | /usr/bin/tr -d '\\n' >&2; exit 7");
        let (tx, _rx) = unbounded_channel();

        let error = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drive(command, "fake-cli", "prompt", &tx, handle_codex_event),
        )
        .await
        .expect("bounded stderr drain should not hang")
        .expect_err("fake CLI intentionally exits without a result");
        let rendered = error.to_string();
        assert!(rendered.contains("stderr truncated"));
        assert!(rendered.len() <= STDERR_CAPTURE_BYTES + 128);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn newline_free_oversized_stdout_record_fails_and_reaps_the_cli_group() {
        let mut command = tokio::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("/usr/bin/yes x | /usr/bin/head -c 4194304 | /usr/bin/tr -d '\\n'; sleep 30");
        let (tx, _rx) = unbounded_channel();

        let error = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drive(command, "fake-cli", "prompt", &tx, handle_codex_event),
        )
        .await
        .expect("oversized stdout must terminate rather than hang")
        .expect_err("oversized NDJSON must fail the turn");
        assert!(error.to_string().contains("NDJSON record exceeded"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn terminal_event_terminates_descendants_holding_output_pipes() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest);
        let marker = fixture.base.join("terminal-descendant-leaked");
        let mut command = tokio::process::Command::new("/bin/sh");
        command
            .env("SHALTAIBOLTAI_TEST_MARKER", &marker)
            .arg("-c")
            .arg(
                r#"(sleep 1; printf leaked > "$SHALTAIBOLTAI_TEST_MARKER") &
printf '%s\n' '{"type":"turn.completed","usage":{}}'
sleep 30"#,
            );
        let (tx, _rx) = unbounded_channel();

        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            drive(command, "fake-codex", "prompt", &tx, handle_codex_event),
        )
        .await
        .expect("terminal event must not wait for inherited output pipes")
        .expect("terminal event should complete the fake turn");
        tokio::time::sleep(std::time::Duration::from_millis(1_200)).await;
        assert!(
            !marker.exists(),
            "background CLI descendant escaped cleanup"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_drive_terminates_background_descendants() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest);
        let marker = fixture.base.join("cancelled-descendant-leaked");
        let mut command = tokio::process::Command::new("/bin/sh");
        command
            .env("SHALTAIBOLTAI_TEST_MARKER", &marker)
            .arg("-c")
            .arg(
                r#"(sleep 1; printf leaked > "$SHALTAIBOLTAI_TEST_MARKER") &
printf '%s\n' '{"type":"ready"}'
sleep 30"#,
            );
        let (tx, mut rx) = unbounded_channel();
        let task = tokio::spawn(async move {
            drive(command, "fake-codex", "prompt", &tx, |event, tx| {
                let _ = tx.send(ChatEvent::Notice(
                    event["type"].as_str().unwrap_or("unknown").to_owned(),
                ));
                false
            })
            .await
        });

        let ready = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("fake CLI should start")
            .expect("fake CLI readiness event");
        assert!(matches!(ready, ChatEvent::Notice(message) if message == "ready"));
        task.abort();
        let _ = task.await;
        tokio::time::sleep(std::time::Duration::from_millis(1_200)).await;
        assert!(!marker.exists(), "cancelled CLI descendant escaped cleanup");
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
