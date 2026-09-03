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
use std::collections::HashSet;
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
const MAX_CODEX_MODEL_CACHE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_CODEX_MODELS: usize = 64;
const MAX_CODEX_MODEL_ID_CHARS: usize = 256;
const CODEX_APP_SERVER_VERSION: &str = "0.152.1";
const CODEX_APP_SERVER_INITIALIZE_ID: i64 = 0;
const CODEX_APP_SERVER_THREAD_START_ID: i64 = 1;
const CODEX_APP_SERVER_TURN_START_ID: i64 = 2;
const CODEX_APP_SERVER_INTERRUPT_ID: i64 = 3;
const MAX_CODEX_INSTRUCTION_SOURCES: usize = 128;
const CODEX_ADVISORY_HOME_PREFIX: &str = "shaltaiboltai-codex-advisory-";
const CLAUDE_READ_ONLY_TOOLS: &str = "Read,Glob,Grep";
// Current list-visible models in the bundled upstream Codex catalog. These are
// the fallback when the installed CLI has not populated its local model cache.
// Custom `codex:<id>` selectors remain accepted when Codex adds new models.
const CODEX_CURATED_MODELS: &[&str] = &[
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.2",
];
const CODEX_CONSTRAINED_CONFIG: &[&str] = &[
    r#"model_provider="openai""#,
    // Pin both first-party endpoints so user-level configuration cannot route
    // authenticated advisory traffic to another origin. Empty openai_base_url
    // selects Codex's compiled-in OpenAI Responses endpoint.
    r#"openai_base_url="""#,
    r#"chatgpt_base_url="https://chatgpt.com/backend-api/""#,
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
    // Memories are model-visible capability roots. They are useful in a normal
    // Codex session but would widen a team worker beyond the reviewed project.
    "features.memories=false",
    // A Shaltaiboltai team already owns fan-out. Nested Codex agents would
    // escape its exact-model/call-count contract and multiply cost.
    "agents.enabled=false",
    "features.multi_agent=false",
    "features.multi_agent_v2=false",
    // Keep secrets and ambient customizations out of untrusted model-visible
    // shell state. The prompt carries the deliberate project instructions.
    r#"shell_environment_policy.inherit="core""#,
    "shell_environment_policy.ignore_default_excludes=false",
    "allow_login_shell=false",
    "features.shell_snapshot=false",
    "skills.include_instructions=false",
    "skills.bundled.enabled=false",
    "features.skill_search=false",
    "features.skill_mcp_dependency_install=false",
];
const CODEX_ADVISORY_PROFILE_PREFIX: &str = "shaltaiboltai-advisory-";
const CODEX_ADVISORY_FILESYSTEM: &str =
    r#"filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read"}}"#;

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

struct CodexAppServerLaunch {
    command: tokio::process::Command,
    candidate: &'static ResolvedCliCandidate,
    contract: CodexAppServerContract,
    _isolated_home: IsolatedCodexHome,
}

#[derive(Debug, Clone)]
struct CodexAppServerContract {
    profile: String,
    model: String,
    cwd: PathBuf,
    workspace_roots: Vec<PathBuf>,
    codex_home: PathBuf,
}

struct IsolatedCodexHome {
    path: PathBuf,
    temp_root: PathBuf,
    source_auth: PathBuf,
    #[cfg(unix)]
    source_identity: CodexAuthIdentity,
}

#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CodexAuthIdentity {
    device: u64,
    inode: u64,
    change_seconds: i64,
    change_nanoseconds: i64,
    mode: u32,
    owner: u32,
    link_count: u64,
    length: u64,
}

impl IsolatedCodexHome {
    fn path(&self) -> &Path {
        &self.path
    }

    fn source_auth(&self) -> &Path {
        &self.source_auth
    }

    #[cfg(unix)]
    fn auth_is_unchanged(&self) -> bool {
        let auth_link = self.path.join("auth.json");
        std::fs::symlink_metadata(&auth_link)
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
            && std::fs::canonicalize(&auth_link).is_ok_and(|target| target == self.source_auth)
            && codex_auth_identity(&self.source_auth)
                .is_some_and(|identity| identity == self.source_identity)
    }

    #[cfg(not(unix))]
    fn auth_is_unchanged(&self) -> bool {
        false
    }
}

impl Drop for IsolatedCodexHome {
    fn drop(&mut self) {
        let safe_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix(CODEX_ADVISORY_HOME_PREFIX))
            .is_some_and(|suffix| {
                suffix.len() == 32 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
            });
        if !safe_name || self.path.parent() != Some(self.temp_root.as_path()) {
            return;
        }
        match std::fs::symlink_metadata(&self.path) {
            Ok(metadata) if metadata.file_type().is_dir() => {
                let _ = std::fs::remove_dir_all(&self.path);
            }
            Ok(_) => {
                // Never recurse through a replacement symlink. Removing this
                // exact, randomly named entry is the narrow cleanup fallback.
                let _ = std::fs::remove_file(&self.path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {}
        }
    }
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
    for root in ["/tmp", "/var/tmp"] {
        add_cli_trust_root(&mut roots, Path::new(root))?;
    }
    if let Some(tmpdir) = std::env::var_os("TMPDIR").filter(|value| !value.is_empty()) {
        add_cli_trust_root(&mut roots, Path::new(&tmpdir))?;
    }
    Ok(roots)
}

fn configured_codex_home() -> Result<PathBuf> {
    let codex_home = std::env::var_os("CODEX_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".codex")))
        .context("cannot locate Codex home for subscription authentication")?;
    if !codex_home.is_absolute() {
        anyhow::bail!("CODEX_HOME must be absolute for read-only Codex advisory runs");
    }
    Ok(codex_home)
}

#[cfg(unix)]
fn codex_auth_identity(path: &Path) -> Option<CodexAuthIdentity> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::metadata(path).ok()?;
    metadata.is_file().then(|| CodexAuthIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        change_seconds: metadata.ctime(),
        change_nanoseconds: metadata.ctime_nsec(),
        mode: metadata.mode(),
        owner: metadata.uid(),
        link_count: metadata.nlink(),
        length: metadata.len(),
    })
}

#[cfg(unix)]
fn validated_codex_auth_file(
    source_home: &Path,
    forbidden_roots: &[PathBuf],
) -> Result<(PathBuf, CodexAuthIdentity)> {
    use std::os::unix::fs::OpenOptionsExt;

    if !source_home.is_absolute() {
        anyhow::bail!("Codex home must be absolute");
    }
    let lexical_auth = source_home.join("auth.json");
    let link_metadata = std::fs::symlink_metadata(&lexical_auth).with_context(|| {
        format!(
            "inspect Codex authentication file {}",
            lexical_auth.display()
        )
    })?;
    if link_metadata.file_type().is_symlink() || !link_metadata.file_type().is_file() {
        anyhow::bail!(
            "Codex authentication file must be a regular, non-symlink file: {}",
            lexical_auth.display()
        );
    }

    // O_NOFOLLOW binds the precondition check to the non-symlink final path.
    // Write access is required because Codex may rotate and persist OAuth
    // tokens while this short-lived app-server owns the symlink.
    let checked_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&lexical_auth)
        .with_context(|| {
            format!(
                "open Codex authentication file read/write without following symlinks: {}",
                lexical_auth.display()
            )
        })?;
    let opened_metadata = checked_file
        .metadata()
        .context("inspect opened Codex authentication file")?;
    let identity = codex_auth_identity_from_metadata(&opened_metadata)
        .context("Codex authentication file is not regular")?;
    // SAFETY: geteuid has no preconditions and does not mutate process state.
    let effective_uid = unsafe { libc::geteuid() };
    if identity.owner != effective_uid
        || identity.mode & 0o077 != 0
        || identity.mode & 0o600 != 0o600
        || identity.link_count != 1
    {
        anyhow::bail!(
            "Codex authentication file must be owner-only, owner-readable/writable, and have exactly one hard link: {}",
            lexical_auth.display()
        );
    }

    let canonical_auth = std::fs::canonicalize(&lexical_auth).with_context(|| {
        format!(
            "canonicalize Codex authentication file {}",
            lexical_auth.display()
        )
    })?;
    if forbidden_roots
        .iter()
        .any(|root| lexical_auth.starts_with(root) || canonical_auth.starts_with(root))
    {
        anyhow::bail!(
            "refusing Codex authentication file under a model-visible workspace or temporary root: {}",
            lexical_auth.display()
        );
    }
    if codex_auth_identity(&canonical_auth).as_ref() != Some(&identity) {
        anyhow::bail!("Codex authentication file changed while it was being validated");
    }
    Ok((canonical_auth, identity))
}

#[cfg(unix)]
fn codex_auth_identity_from_metadata(metadata: &std::fs::Metadata) -> Option<CodexAuthIdentity> {
    use std::os::unix::fs::MetadataExt;

    metadata.is_file().then(|| CodexAuthIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        change_seconds: metadata.ctime(),
        change_nanoseconds: metadata.ctime_nsec(),
        mode: metadata.mode(),
        owner: metadata.uid(),
        link_count: metadata.nlink(),
        length: metadata.len(),
    })
}

#[cfg(unix)]
fn create_isolated_codex_home_in(
    source_home: &Path,
    temp_root: &Path,
    auth_forbidden_roots: &[PathBuf],
    model_visible_roots: &[PathBuf],
) -> Result<IsolatedCodexHome> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};

    let (source_auth, source_identity) =
        validated_codex_auth_file(source_home, auth_forbidden_roots)?;
    let temp_root = std::fs::canonicalize(temp_root)
        .with_context(|| format!("canonicalize temporary root {}", temp_root.display()))?;
    if !std::fs::metadata(&temp_root)
        .with_context(|| format!("inspect temporary root {}", temp_root.display()))?
        .is_dir()
    {
        anyhow::bail!("temporary root is not a directory: {}", temp_root.display());
    }

    for _ in 0..16 {
        let candidate = temp_root.join(format!(
            "{CODEX_ADVISORY_HOME_PREFIX}{:032x}",
            rand::random::<u128>()
        ));
        if model_visible_roots
            .iter()
            .any(|root| candidate.starts_with(root))
        {
            anyhow::bail!(
                "temporary Codex home would be visible inside an advisory workspace: {}",
                candidate.display()
            );
        }

        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        match builder.create(&candidate) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create isolated Codex home {}", candidate.display())
                });
            }
        }

        // Install the cleanup guard immediately after the exclusive create so
        // every later validation failure removes only this exact directory.
        let isolated = IsolatedCodexHome {
            path: candidate,
            temp_root: temp_root.clone(),
            source_auth: source_auth.clone(),
            source_identity: source_identity.clone(),
        };
        let canonical_candidate = std::fs::canonicalize(&isolated.path).with_context(|| {
            format!(
                "canonicalize isolated Codex home {}",
                isolated.path.display()
            )
        })?;
        if canonical_candidate != isolated.path
            || isolated.path.parent() != Some(temp_root.as_path())
        {
            anyhow::bail!("isolated Codex home escaped its temporary root");
        }

        // Reassert and verify 0700 before placing the authentication symlink.
        std::fs::set_permissions(&isolated.path, std::fs::Permissions::from_mode(0o700))
            .context("set isolated Codex home permissions")?;
        let home_metadata =
            std::fs::symlink_metadata(&isolated.path).context("inspect isolated Codex home")?;
        // SAFETY: geteuid has no preconditions and does not mutate process state.
        let effective_uid = unsafe { libc::geteuid() };
        if !home_metadata.file_type().is_dir()
            || home_metadata.mode() & 0o777 != 0o700
            || home_metadata.uid() != effective_uid
        {
            anyhow::bail!("isolated Codex home is not a private owner-only directory");
        }

        std::os::unix::fs::symlink(&source_auth, isolated.path.join("auth.json"))
            .context("link Codex subscription authentication into isolated home")?;
        if !isolated.auth_is_unchanged() {
            anyhow::bail!("Codex authentication file changed while creating isolated home");
        }
        return Ok(isolated);
    }
    anyhow::bail!("could not allocate a unique isolated Codex home")
}

#[cfg(unix)]
fn create_isolated_codex_home(execution_policy: &ExecutionPolicy) -> Result<IsolatedCodexHome> {
    let source_home = configured_codex_home()?;
    let auth_forbidden_roots = model_writable_roots(execution_policy)?;
    create_isolated_codex_home_in(
        &source_home,
        &std::env::temp_dir(),
        &auth_forbidden_roots,
        execution_policy.effective_user_visible_roots(),
    )
}

#[cfg(not(unix))]
fn create_isolated_codex_home(_execution_policy: &ExecutionPolicy) -> Result<IsolatedCodexHome> {
    anyhow::bail!("read-only Codex advisory transport requires Unix auth-file isolation")
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

/// Drive one Codex app-server process for a single attested advisory turn. The
/// initialize and thread/start requests are local protocol setup only; the
/// model is not invoked until `run_codex_app_server_protocol` sends turn/start
/// after every profile, model, workspace, and version check succeeds.
async fn drive_codex_app_server(
    mut launch: CodexAppServerLaunch,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()> {
    #[cfg(unix)]
    launch.command.as_std_mut().process_group(0);
    launch
        .command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if !candidate_is_unchanged(launch.candidate) {
        anyhow::bail!("refusing to launch `codex` because it changed after discovery");
    }
    if !launch._isolated_home.auth_is_unchanged() {
        anyhow::bail!(
            "refusing to launch `codex` because subscription authentication changed after validation"
        );
    }

    let mut child = launch.command.spawn().context(
        "failed to launch `codex app-server` — is Codex 0.152.1 installed and signed in?",
    )?;
    #[cfg(unix)]
    let mut process_group = ProcessGroupGuard::new(
        child
            .id()
            .context("spawned Codex app-server did not expose a process id")?,
    )?;
    let stdout = child
        .stdout
        .take()
        .context("Codex app-server has no stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("Codex app-server has no stderr")?;
    let mut stdin = child
        .stdin
        .take()
        .context("Codex app-server has no stdin")?;
    let stderr_task = tokio::spawn(drain_stderr_bounded(stderr));
    let mut stdout = BufReader::new(stdout);

    let protocol_result =
        run_codex_app_server_protocol(&mut stdout, &mut stdin, &launch.contract, prompt, tx).await;

    // App-server has no shutdown RPC. EOF is its documented single-client
    // stdio shutdown, with the owned process group as a bounded fallback.
    let _ = stdin.shutdown().await;
    drop(stdin);
    drop(stdout);
    let status =
        match tokio::time::timeout(std::time::Duration::from_millis(250), child.wait()).await {
            Ok(status) => status?,
            Err(_) => {
                #[cfg(unix)]
                process_group
                    .terminate()
                    .context("failed to terminate Codex app-server descendants")?;
                #[cfg(not(unix))]
                child
                    .kill()
                    .await
                    .context("failed to terminate Codex app-server")?;
                child.wait().await?
            }
        };
    #[cfg(unix)]
    process_group
        .terminate()
        .context("failed to terminate Codex app-server descendants")?;
    let stderr = finish_stderr_drain(stderr_task).await;

    match protocol_result {
        Ok(()) => Ok(()),
        Err(error) => {
            let detail = stderr.trim();
            if detail.is_empty() {
                Err(error).with_context(|| format!("Codex app-server exited with {status}"))
            } else {
                Err(error).with_context(|| format!("Codex app-server: {detail}"))
            }
        }
    }
}

async fn run_codex_app_server_protocol<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    contract: &CodexAppServerContract,
    prompt: &str,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    send_codex_app_server_message(
        stdin,
        &serde_json::json!({
            "id": CODEX_APP_SERVER_INITIALIZE_ID,
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
    let initialized =
        read_codex_app_server_response(stdout, stdin, CODEX_APP_SERVER_INITIALIZE_ID, tx).await?;
    attest_codex_app_server_initialize(&initialized, contract)?;

    send_codex_app_server_message(stdin, &serde_json::json!({"method": "initialized"})).await?;
    send_codex_app_server_message(
        stdin,
        &serde_json::json!({
            "id": CODEX_APP_SERVER_THREAD_START_ID,
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
    let thread_started =
        read_codex_app_server_response(stdout, stdin, CODEX_APP_SERVER_THREAD_START_ID, tx).await?;
    let thread_id = attest_codex_app_server_thread(&thread_started, contract)?;

    // This is the first model/cost-bearing operation. Never retry it: an
    // ambiguous transport failure may mean the upstream model already ran.
    send_codex_app_server_message(
        stdin,
        &serde_json::json!({
            "id": CODEX_APP_SERVER_TURN_START_ID,
            "method": "turn/start",
            "params": {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt, "textElements": []}]
            }
        }),
    )
    .await?;
    let turn_started =
        read_codex_app_server_response(stdout, stdin, CODEX_APP_SERVER_TURN_START_ID, tx).await?;
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

    consume_codex_app_server_turn(stdout, stdin, &thread_id, &turn_id, tx).await
}

async fn send_codex_app_server_message<W>(stdin: &mut W, message: &Value) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut encoded = serde_json::to_vec(message).context("serialize Codex app-server message")?;
    if encoded.len() > MAX_NDJSON_RECORD_BYTES {
        anyhow::bail!("Codex app-server request exceeded {MAX_NDJSON_RECORD_BYTES} bytes");
    }
    encoded.push(b'\n');
    stdin.write_all(&encoded).await?;
    stdin.flush().await?;
    Ok(())
}

async fn read_codex_app_server_message<R>(stdout: &mut BufReader<R>) -> Result<Value>
where
    R: AsyncRead + Unpin,
{
    let mut record = Vec::new();
    loop {
        read_bounded_ndjson_record(stdout, &mut record)
            .await?
            .context("Codex app-server closed stdout before completing the turn")?;
        if record.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        return serde_json::from_slice(&record).context("invalid JSON from Codex app-server");
    }
}

async fn read_codex_app_server_response<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    expected_id: i64,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<Value>
where
    R: AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    loop {
        let message = read_codex_app_server_message(stdout).await?;
        if message.get("method").is_some() {
            if message.get("id").is_some() {
                reject_codex_app_server_request(stdin, &message).await?;
            } else {
                emit_codex_app_server_notice(&message, tx);
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

async fn reject_codex_app_server_request<W>(stdin: &mut W, request: &Value) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let id = request
        .get("id")
        .cloned()
        .context("Codex app-server request omitted id")?;
    send_codex_app_server_message(
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

fn emit_codex_app_server_notice(message: &Value, tx: &UnboundedSender<ChatEvent>) {
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let params = &message["params"];
    let text = match method {
        "warning" | "configWarning" | "guardianWarning" | "authRecovery" => {
            params.get("message").and_then(Value::as_str)
        }
        "deprecationNotice" => params.get("summary").and_then(Value::as_str),
        _ => None,
    };
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        let _ = tx.send(ChatEvent::Notice(text.to_owned()));
    }
}

fn attest_codex_app_server_initialize(
    result: &Value,
    contract: &CodexAppServerContract,
) -> Result<()> {
    let user_agent = result
        .get("userAgent")
        .and_then(Value::as_str)
        .context("Codex initialize response omitted userAgent")?;
    let version = user_agent
        .strip_prefix("shaltaiboltai/")
        .and_then(|suffix| suffix.split_whitespace().next())
        .filter(|version| !version.is_empty())
        .context("Codex initialize response had an invalid userAgent")?;
    if version != CODEX_APP_SERVER_VERSION {
        anyhow::bail!(
            "unsupported Codex app-server version {version}; expected exactly {CODEX_APP_SERVER_VERSION}"
        );
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

fn attest_codex_app_server_thread(
    result: &Value,
    contract: &CodexAppServerContract,
) -> Result<String> {
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
    attest_codex_instruction_sources(result, contract)?;

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
        && thread.get("cliVersion").and_then(Value::as_str) == Some(CODEX_APP_SERVER_VERSION)
        && thread.get("ephemeral").and_then(Value::as_bool) == Some(true)
        && thread.get("path").is_some_and(Value::is_null)
        && thread.get("historyMode").and_then(Value::as_str) == Some("legacy")
        && thread.get("modelProvider").and_then(Value::as_str) == Some("openai")
        && thread.get("cwd").and_then(Value::as_str) == Some(cwd);
    if !matches_contract {
        anyhow::bail!(
            "Codex thread/start attestation did not match the requested read-only advisory contract"
        );
    }
    Ok(thread_id.to_owned())
}

fn attest_codex_instruction_sources(
    result: &Value,
    contract: &CodexAppServerContract,
) -> Result<()> {
    let sources = result
        .get("instructionSources")
        .and_then(Value::as_array)
        .context("Codex thread/start response omitted instructionSources")?;
    if sources.len() > MAX_CODEX_INSTRUCTION_SOURCES {
        anyhow::bail!(
            "Codex reported more than {MAX_CODEX_INSTRUCTION_SOURCES} instruction sources"
        );
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

async fn consume_codex_app_server_turn<R, W>(
    stdout: &mut BufReader<R>,
    stdin: &mut W,
    thread_id: &str,
    turn_id: &str,
    tx: &UnboundedSender<ChatEvent>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut usage = None;
    let mut emitted_agent_items = HashSet::new();
    let mut terminal_error = None;
    loop {
        let message = read_codex_app_server_message(stdout).await?;
        if message.get("method").is_some() && message.get("id").is_some() {
            reject_codex_app_server_request(stdin, &message).await?;
            continue;
        }
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            anyhow::bail!("Codex app-server emitted an unexpected response during a turn");
        };
        let params = &message["params"];
        match method {
            "item/completed" if codex_notification_matches(params, thread_id, turn_id) => {
                emit_codex_app_server_item(&params["item"], &mut emitted_agent_items, tx);
            }
            "thread/tokenUsage/updated"
                if codex_notification_matches(params, thread_id, turn_id) =>
            {
                usage = codex_app_server_usage(params);
            }
            "warning" => {
                let applies = params
                    .get("threadId")
                    .and_then(Value::as_str)
                    .is_none_or(|id| id == thread_id);
                if applies {
                    emit_codex_app_server_notice(&message, tx);
                }
            }
            "configWarning" | "guardianWarning" | "authRecovery" | "deprecationNotice" => {
                emit_codex_app_server_notice(&message, tx);
            }
            "error" if codex_notification_matches(params, thread_id, turn_id) => {
                let detail = codex_turn_error_message(&params["error"]);
                if params.get("willRetry").and_then(Value::as_bool) == Some(true) {
                    let _ = tx.send(ChatEvent::Notice(detail));
                } else {
                    terminal_error = Some(detail);
                }
            }
            "model/rerouted" if codex_notification_matches(params, thread_id, turn_id) => {
                let from = params
                    .get("fromModel")
                    .and_then(Value::as_str)
                    .unwrap_or("requested model");
                let to = params
                    .get("toModel")
                    .and_then(Value::as_str)
                    .unwrap_or("another model");
                send_codex_app_server_message(
                    stdin,
                    &serde_json::json!({
                        "id": CODEX_APP_SERVER_INTERRUPT_ID,
                        "method": "turn/interrupt",
                        "params": {"threadId": thread_id, "turnId": turn_id}
                    }),
                )
                .await?;
                anyhow::bail!(
                    "Codex rerouted the advisory turn from {from} to {to}; exact-model contract failed"
                );
            }
            "turn/completed" if codex_notification_matches(params, thread_id, turn_id) => {
                let turn = params
                    .get("turn")
                    .and_then(Value::as_object)
                    .context("Codex turn/completed omitted turn")?;
                if turn.get("id").and_then(Value::as_str) != Some(turn_id) {
                    continue;
                }
                if let Some(items) = turn.get("items").and_then(Value::as_array) {
                    for item in items {
                        if item.get("type").and_then(Value::as_str) == Some("agentMessage") {
                            emit_codex_app_server_item(item, &mut emitted_agent_items, tx);
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
                            .map(codex_turn_error_message)
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

fn codex_notification_matches(params: &Value, thread_id: &str, turn_id: &str) -> bool {
    params.get("threadId").and_then(Value::as_str) == Some(thread_id)
        && params.get("turnId").and_then(Value::as_str) == Some(turn_id)
}

fn codex_app_server_usage(params: &Value) -> Option<Usage> {
    let total = params.get("tokenUsage")?.get("total")?;
    Some(Usage {
        // inputTokens already includes cachedInputTokens; do not add it again.
        input_tokens: total.get("inputTokens")?.as_u64()?,
        output_tokens: total.get("outputTokens")?.as_u64()?,
    })
}

fn codex_turn_error_message(error: &Value) -> String {
    error
        .get("additionalDetails")
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty())
        .or_else(|| error.get("message").and_then(Value::as_str))
        .unwrap_or("Codex advisory turn failed")
        .to_owned()
}

fn emit_codex_app_server_item(
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
                summary: summarize_codex_item(item),
                is_error,
            });
        }
    }
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
/// workspace and execution policy are known. Publish the curated upstream
/// catalog instead; the Codex CLI remains authoritative about account access.
pub async fn codex_model_ids() -> Vec<String> {
    if let Some(models) = read_codex_model_cache().await {
        return models;
    }
    curated_codex_model_ids()
}

fn curated_codex_model_ids() -> Vec<String> {
    CODEX_CURATED_MODELS
        .iter()
        .map(|model| (*model).to_owned())
        .collect()
}

async fn read_codex_model_cache() -> Option<Vec<String>> {
    let codex_home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".codex")))?;
    let path = codex_home.join("models_cache.json");
    let path_metadata = tokio::fs::symlink_metadata(&path).await.ok()?;
    if !path_metadata.file_type().is_file() || path_metadata.len() > MAX_CODEX_MODEL_CACHE_BYTES {
        return None;
    }
    let file = tokio::fs::File::open(path).await.ok()?;
    let opened_metadata = file.metadata().await.ok()?;
    if !opened_metadata.is_file() || opened_metadata.len() > MAX_CODEX_MODEL_CACHE_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
    file.take(MAX_CODEX_MODEL_CACHE_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .ok()?;
    if bytes.len() as u64 > MAX_CODEX_MODEL_CACHE_BYTES {
        return None;
    }
    parse_codex_model_cache(&bytes)
}

fn parse_codex_model_cache(bytes: &[u8]) -> Option<Vec<String>> {
    let body: Value = serde_json::from_slice(bytes).ok()?;
    let rows = body["models"].as_array()?;
    let mut seen = HashSet::new();
    let models: Vec<String> = rows
        .iter()
        .filter(|row| row["visibility"].as_str() == Some("list"))
        .filter_map(|row| row["slug"].as_str())
        .filter(|id| {
            !id.is_empty()
                && id.chars().count() <= MAX_CODEX_MODEL_ID_CHARS
                && !id.chars().any(|ch| ch.is_control() || ch.is_whitespace())
        })
        .filter(|id| seen.insert((*id).to_owned()))
        .take(MAX_CODEX_MODELS)
        .map(str::to_owned)
        .collect();
    (!models.is_empty()).then_some(models)
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
    if req.policy == RequestPolicy::ReadOnly {
        let model = model.context(
            "read-only Codex advisory runs require an explicit `codex:<model>` selection",
        )?;
        let launch = fresh_codex_app_server_launch(model, &req.execution_policy)?;
        drive_codex_app_server(launch, &prompt, tx).await
    } else {
        let cmd = fresh_codex_command(model, req.policy, &req.execution_policy)?;
        drive_cli(cmd, "codex", &prompt, tx, handle_codex_event).await
    }
}

fn fresh_codex_app_server_launch(
    model: &str,
    execution_policy: &ExecutionPolicy,
) -> Result<CodexAppServerLaunch> {
    let candidate = cli_candidate_for_policy(CliExecutable::Codex, execution_policy)?;
    let profile = fresh_codex_advisory_profile();
    let cwd = execution_policy.workspace().cwd().to_path_buf();
    let workspace_roots = execution_policy.effective_user_visible_roots().to_vec();
    if cwd.to_str().is_none() || workspace_roots.iter().any(|root| root.to_str().is_none()) {
        anyhow::bail!("Codex app-server workspace paths must be valid UTF-8");
    }
    let isolated_home = create_isolated_codex_home(execution_policy)?;
    if isolated_home.path().to_str().is_none() {
        anyhow::bail!("isolated Codex home path must be valid UTF-8");
    }
    let command = build_codex_app_server_command(
        &candidate.canonical,
        &profile,
        execution_policy,
        isolated_home.path(),
        isolated_home.source_auth(),
    )?;
    Ok(CodexAppServerLaunch {
        command,
        candidate,
        contract: CodexAppServerContract {
            profile,
            model: model.to_owned(),
            cwd,
            workspace_roots,
            codex_home: isolated_home.path().to_path_buf(),
        },
        _isolated_home: isolated_home,
    })
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
        )?,
        candidate,
    })
}

fn build_codex_command(
    executable: &Path,
    model: Option<&str>,
    request_policy: RequestPolicy,
    execution_policy: &ExecutionPolicy,
) -> Result<tokio::process::Command> {
    // Every request starts in a fresh, explicitly sandboxed process. Context is
    // carried in `prompt`, never inferred from another cwd-global CLI session.
    let sandbox = effective_sandbox(request_policy, execution_policy);
    let mut cmd = tokio::process::Command::new(executable);
    cmd.current_dir(execution_policy.workspace().cwd());
    cmd.arg("exec")
        .arg("--ephemeral")
        .arg("-C")
        .arg(execution_policy.workspace().cwd());
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
    // `codex exec` cannot route its inner approval requests to this app. Make
    // its fail-closed headless default explicit so ambient configuration can
    // never turn those requests into automatic approvals.
    cmd.arg("-c").arg(r#"approval_policy="never""#);
    if sandbox == SandboxMode::ReadOnly {
        // Legacy `--sandbox read-only` blocks writes but grants broad disk
        // reads. A named permission profile defaults to deny, adds only the
        // runtime workspace roots and minimal executable/library reads, and
        // keeps network off. Passing `--sandbox` here would disable this
        // profile and is therefore intentionally forbidden.
        // A fresh identifier prevents an identically named profile in a
        // project or managed config layer from recursively merging authority
        // into this one. Session-flag `default_permissions` selects profile
        // syntax before lower config layers in supported Codex releases.
        let profile = fresh_codex_advisory_profile();
        cmd.arg("-c")
            .arg(format!(r#"default_permissions="{profile}""#));
        cmd.arg("-c")
            .arg(format!("permissions.{profile}.{CODEX_ADVISORY_FILESYSTEM}"));
        cmd.arg("-c")
            .arg(format!("permissions.{profile}.network={{enabled=false}}"));
        cmd.arg("-c")
            .arg(codex_advisory_roots_config(&profile, execution_policy)?);
    } else {
        cmd.arg("--sandbox").arg(sandbox.to_string());
    }
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
    Ok(cmd)
}

fn build_codex_app_server_command(
    executable: &Path,
    profile: &str,
    execution_policy: &ExecutionPolicy,
    isolated_codex_home: &Path,
    source_auth: &Path,
) -> Result<tokio::process::Command> {
    let mut cmd = tokio::process::Command::new(executable);
    cmd.current_dir(execution_policy.workspace().cwd());
    cmd.env("CODEX_HOME", isolated_codex_home);
    cmd.arg("app-server").arg("--stdio").arg("--strict-config");
    for constraint in CODEX_CONSTRAINED_CONFIG {
        cmd.arg("-c").arg(constraint);
    }
    // An isolated CODEX_HOME has a different keyring namespace. Force the
    // file backend so the owner-only auth.json symlink remains authoritative
    // and OAuth token refreshes persist to the user's original file.
    cmd.arg("-c").arg(r#"cli_auth_credentials_store="file""#);
    cmd.arg("-c").arg(r#"approval_policy="never""#);
    cmd.arg("-c")
        .arg(format!(r#"default_permissions="{profile}""#));
    // Keep both authentication files unreadable to model tools even on
    // platforms whose minimal runtime allowance includes a broad scratch
    // directory. Denying the whole isolated home would also hide Codex's
    // CODEX_HOME/tmp/arg0 runtime helper on Linux.
    cmd.arg("-c")
        .arg(codex_advisory_app_server_filesystem_config(
            profile,
            executable,
            isolated_codex_home,
            source_auth,
        )?);
    cmd.arg("-c")
        .arg(format!("permissions.{profile}.network={{enabled=false}}"));
    cmd.arg("-c")
        .arg(codex_advisory_roots_config(profile, execution_policy)?);
    Ok(cmd)
}

fn fresh_codex_advisory_profile() -> String {
    format!(
        "{CODEX_ADVISORY_PROFILE_PREFIX}{:032x}",
        rand::random::<u128>()
    )
}

fn codex_advisory_app_server_filesystem_config(
    profile: &str,
    executable: &Path,
    isolated_codex_home: &Path,
    source_auth: &Path,
) -> Result<String> {
    let isolated_auth_glob = codex_advisory_auth_deny_glob(&isolated_codex_home.join("auth.json"))?;
    let source_auth_glob = codex_advisory_auth_deny_glob(source_auth)?;
    let executable = executable
        .to_str()
        .with_context(|| format!("Codex executable path is not valid UTF-8: {executable:?}"))?;
    let source_auth = source_auth
        .to_str()
        .with_context(|| format!("Codex auth path is not valid UTF-8: {source_auth:?}"))?;
    Ok(format!(
        "permissions.{profile}.filesystem={{glob_scan_max_depth=2,\":root\"=\"deny\",\":minimal\"=\"read\",\":workspace_roots\"={{\".\"=\"read\"}},{}=\"read\",{}=\"deny\",{}=\"deny\",{}=\"deny\"}}",
        toml::Value::String(executable.to_owned()),
        toml::Value::String(source_auth.to_owned()),
        toml::Value::String(isolated_auth_glob),
        toml::Value::String(source_auth_glob)
    ))
}

fn codex_advisory_auth_deny_glob(auth: &Path) -> Result<String> {
    if auth.file_name() != Some(OsStr::new("auth.json")) {
        anyhow::bail!("Codex authentication filename must be exactly `auth.json`");
    }
    let parent_path = auth
        .parent()
        .context("Codex authentication path must have a parent directory")?;
    let parent = parent_path
        .to_str()
        .with_context(|| format!("Codex auth parent path is not valid UTF-8: {parent_path:?}"))?;
    if parent
        .chars()
        .any(|ch| matches!(ch, '*' | '?' | '[' | ']' | '{' | '}' | '\\'))
    {
        anyhow::bail!(
            "Codex auth parent path contains glob syntax and cannot be safely isolated: {parent}"
        );
    }
    // The bracket expression matches the exact final character while making
    // this a deny-glob. Codex 0.152.1 emits glob denies after its unconditional
    // macOS minimal-runtime allowances, unlike a literal path deny.
    parent_path
        .join("auth.jso[n]")
        .to_str()
        .map(str::to_owned)
        .context("Codex auth deny glob is not valid UTF-8")
}

fn codex_advisory_roots_config(
    profile: &str,
    execution_policy: &ExecutionPolicy,
) -> Result<String> {
    let cwd = execution_policy.workspace().cwd();
    let entries = execution_policy
        .effective_user_visible_roots()
        .iter()
        .map(PathBuf::as_path)
        .filter(|root| *root != cwd)
        .map(|root| {
            let root = root
                .to_str()
                .with_context(|| format!("Codex advisory root is not valid UTF-8: {root:?}"))?;
            Ok(format!("{}=true", toml::Value::String(root.to_owned())))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "permissions.{profile}.workspace_roots={{{}}}",
        entries.join(",")
    ))
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
        build_codex_command(
            &test_cli_executable("codex"),
            model,
            request_policy,
            execution_policy,
        )
    }

    fn test_codex_app_server_command(
        execution_policy: &ExecutionPolicy,
    ) -> Result<(tokio::process::Command, String)> {
        let profile = fresh_codex_advisory_profile();
        let isolated_home = test_cli_executable("isolated-codex-home");
        let source_auth = test_cli_executable("source-codex-home").join("auth.json");
        let command = build_codex_app_server_command(
            &test_cli_executable("codex"),
            &profile,
            execution_policy,
            &isolated_home,
            &source_auth,
        )?;
        Ok((command, profile))
    }

    fn test_codex_app_server_contract(fixture: &PolicyFixture) -> CodexAppServerContract {
        CodexAppServerContract {
            profile: fresh_codex_advisory_profile(),
            model: "gpt-5.6-sol".into(),
            cwd: fixture.cwd().to_path_buf(),
            workspace_roots: fixture.policy.effective_user_visible_roots().to_vec(),
            codex_home: PathBuf::from("/tmp/fake-codex-home"),
        }
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

    fn command_env_path(command: &tokio::process::Command, key: &str) -> Option<PathBuf> {
        command
            .as_std()
            .get_envs()
            .find(|(name, _)| *name == OsStr::new(key))
            .and_then(|(_, value)| value.map(PathBuf::from))
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

    fn advisory_profile_from_args(args: &[String]) -> &str {
        values_for(args, "-c")
            .into_iter()
            .find_map(|value| {
                value
                    .strip_prefix(r#"default_permissions=""#)
                    .and_then(|value| value.strip_suffix('"'))
            })
            .expect("Codex advisory profile selection")
    }

    fn assert_fresh_advisory_profile(profile: &str) {
        let suffix = profile
            .strip_prefix(CODEX_ADVISORY_PROFILE_PREFIX)
            .expect("Shaltaiboltai profile prefix");
        assert_eq!(suffix.len(), 32);
        assert!(suffix.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    fn drain(events: &mut tokio::sync::mpsc::UnboundedReceiver<ChatEvent>) -> Vec<ChatEvent> {
        let mut out = Vec::new();
        while let Ok(e) = events.try_recv() {
            out.push(e);
        }
        out
    }

    async fn fake_app_server_request<R>(
        reader: &mut BufReader<R>,
        expected_method: &str,
    ) -> Result<Value>
    where
        R: AsyncRead + Unpin,
    {
        let request = read_codex_app_server_message(reader).await?;
        assert_eq!(
            request.get("method").and_then(Value::as_str),
            Some(expected_method)
        );
        Ok(request)
    }

    async fn fake_app_server_initialize<R, W>(
        reader: &mut BufReader<R>,
        writer: &mut W,
        version: &str,
        codex_home: &Path,
    ) -> Result<()>
    where
        R: AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin,
    {
        let initialize = fake_app_server_request(reader, "initialize").await?;
        assert_eq!(initialize["id"], CODEX_APP_SERVER_INITIALIZE_ID);
        assert_eq!(
            initialize["params"]["capabilities"]["experimentalApi"],
            true
        );
        send_codex_app_server_message(
            writer,
            &json!({
                "id": CODEX_APP_SERVER_INITIALIZE_ID,
                "result": {
                    "userAgent": format!("shaltaiboltai/{version} (test)"),
                    "codexHome": codex_home,
                    "platformFamily": std::env::consts::FAMILY,
                    "platformOs": std::env::consts::OS
                }
            }),
        )
        .await?;
        let initialized = fake_app_server_request(reader, "initialized").await?;
        assert!(initialized.get("id").is_none());
        Ok(())
    }

    fn fake_attested_thread_result(contract: &CodexAppServerContract, thread_id: &str) -> Value {
        json!({
            "thread": {
                "id": thread_id,
                "cliVersion": CODEX_APP_SERVER_VERSION,
                "ephemeral": true,
                "path": null,
                "historyMode": "legacy",
                "modelProvider": "openai",
                "cwd": contract.cwd
            },
            "model": contract.model,
            "modelProvider": "openai",
            "cwd": contract.cwd,
            "runtimeWorkspaceRoots": contract.workspace_roots,
            "instructionSources": [],
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "sandbox": {"type": "readOnly", "networkAccess": false},
            "activePermissionProfile": {"id": contract.profile, "extends": null}
        })
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
    async fn codex_model_discovery_returns_a_bounded_nonexecuting_catalog() {
        let models = codex_model_ids().await;
        assert!(!models.is_empty());
        assert!(models.len() <= MAX_CODEX_MODELS);
        assert!(models
            .iter()
            .all(|model| !model.contains(char::is_whitespace)));
    }

    #[test]
    fn codex_model_cache_keeps_only_bounded_list_visible_rows() {
        let oversized = "x".repeat(MAX_CODEX_MODEL_ID_CHARS + 1);
        let body = serde_json::json!({"models": [
            {"slug": "gpt-best", "visibility": "list"},
            {"slug": "gpt-hidden", "visibility": "hide"},
            {"slug": "gpt-best", "visibility": "list"},
            {"slug": "bad model", "visibility": "list"},
            {"slug": oversized, "visibility": "list"},
            {"slug": "gpt-next", "visibility": "list"}
        ]});
        assert_eq!(
            parse_codex_model_cache(body.to_string().as_bytes()).unwrap(),
            vec!["gpt-best", "gpt-next"]
        );
        assert!(parse_codex_model_cache(br#"{"models": []}"#).is_none());
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

        let (codex, profile) = test_codex_app_server_command(&fixture.policy)
            .expect("read-only Codex advisory command");
        let codex_args = command_args(&codex);
        assert_eq!(command_cwd(&codex), Some(fixture.cwd()));
        assert_eq!(codex_args.first().map(String::as_str), Some("app-server"));
        assert!(codex_args.iter().any(|arg| arg == "--stdio"));
        assert!(codex_args.iter().any(|arg| arg == "--strict-config"));
        assert_eq!(
            command_env_path(&codex, "CODEX_HOME"),
            Some(test_cli_executable("isolated-codex-home"))
        );
        assert!(values_for(&codex_args, "--sandbox").is_empty());
        assert!(!codex_args
            .iter()
            .any(|arg| arg == "exec" || arg == "--json"));
        assert!(!codex_args.iter().any(|arg| arg == "--ignore-user-config"));
        assert!(!codex_args.iter().any(|arg| arg == "--ignore-rules"));
        assert!(values_for(&codex_args, "--model").is_empty());
        assert!(values_for(&codex_args, "-C").is_empty());
        for constraint in CODEX_CONSTRAINED_CONFIG {
            assert!(has_arg_pair(&codex_args, "-c", constraint));
        }
        assert!(has_arg_pair(
            &codex_args,
            "-c",
            r#"cli_auth_credentials_store="file""#
        ));
        assert_eq!(advisory_profile_from_args(&codex_args), profile);
        assert_fresh_advisory_profile(&profile);
        let filesystem = codex_advisory_app_server_filesystem_config(
            &profile,
            &test_cli_executable(CliExecutable::Codex.file_name()),
            &test_cli_executable("isolated-codex-home"),
            &test_cli_executable("source-codex-home").join("auth.json"),
        )
        .unwrap();
        assert!(has_arg_pair(&codex_args, "-c", &filesystem));
        assert!(has_arg_pair(
            &codex_args,
            "-c",
            &format!("permissions.{profile}.network={{enabled=false}}")
        ));
        let roots = codex_advisory_roots_config(&profile, &fixture.policy).unwrap();
        assert!(has_arg_pair(&codex_args, "-c", &roots));
        assert!(!codex_args.iter().any(|arg| arg == "danger-full-access"));
        assert!(values_for(&codex_args, "--add-dir").is_empty());
    }

    #[test]
    fn advisory_profile_name_is_unique_per_codex_child() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);
        let (first, first_profile) =
            test_codex_app_server_command(&fixture.policy).expect("first read-only Codex command");
        let (second, second_profile) =
            test_codex_app_server_command(&fixture.policy).expect("second read-only Codex command");
        assert_eq!(
            advisory_profile_from_args(&command_args(&first)),
            first_profile
        );
        assert_eq!(
            advisory_profile_from_args(&command_args(&second)),
            second_profile
        );
        assert_fresh_advisory_profile(&first_profile);
        assert_fresh_advisory_profile(&second_profile);
        assert_ne!(first_profile, second_profile);
    }

    #[test]
    fn app_server_filesystem_override_is_one_parseable_table() {
        let profile = "shaltaiboltai-advisory-test";
        let executable = Path::new("/opt/Codex 0.152.1/bin/codex");
        let isolated_home = Path::new("/private/tmp/isolated.home");
        let source_auth = Path::new("/var/tmp/source.home/auth.json");
        let override_value = codex_advisory_app_server_filesystem_config(
            profile,
            executable,
            isolated_home,
            source_auth,
        )
        .expect("filesystem override");

        let prefix = format!("permissions.{profile}.filesystem=");
        let inline_table = override_value
            .strip_prefix(&prefix)
            .expect("one filesystem-table override");
        assert!(!override_value.contains(&format!("permissions.{profile}.filesystem.")));
        let parsed: toml::Value = toml::from_str(&format!("value={inline_table}"))
            .expect("valid TOML inline filesystem table");
        let table = parsed["value"].as_table().expect("filesystem table");
        assert_eq!(table[executable.to_str().unwrap()].as_str(), Some("read"));
        assert!(table.get(isolated_home.to_str().unwrap()).is_none());
        assert_eq!(table[source_auth.to_str().unwrap()].as_str(), Some("deny"));
        assert_eq!(table["glob_scan_max_depth"].as_integer(), Some(2));
        let isolated_auth = isolated_home.join("auth.json");
        assert_eq!(
            table[&codex_advisory_auth_deny_glob(&isolated_auth).unwrap()].as_str(),
            Some("deny")
        );
        assert_eq!(
            table[&codex_advisory_auth_deny_glob(source_auth).unwrap()].as_str(),
            Some("deny")
        );
        assert!(codex_advisory_auth_deny_glob(Path::new("/tmp/unsafe[glob]/auth.json")).is_err());
        assert!(codex_advisory_auth_deny_glob(Path::new("/tmp/auth.json.bak")).is_err());
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
            if sandbox == SandboxMode::ReadOnly {
                assert!(values_for(&codex_args, "--sandbox").is_empty());
                let profile = advisory_profile_from_args(&codex_args);
                assert_fresh_advisory_profile(profile);
                assert!(has_arg_pair(
                    &codex_args,
                    "-c",
                    &format!("permissions.{profile}.{CODEX_ADVISORY_FILESYSTEM}")
                ));
                let roots = codex_advisory_roots_config(profile, &fixture.policy).unwrap();
                assert!(has_arg_pair(&codex_args, "-c", &roots));
            } else {
                assert!(has_arg_pair(&codex_args, "--sandbox", &sandbox.to_string()));
            }
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
    fn create_test_codex_auth(home: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        std::fs::create_dir_all(home).expect("create test Codex home");
        let auth = home.join("auth.json");
        std::fs::write(&auth, br#"{"tokens":{"access_token":"test"}}"#)
            .expect("write test Codex auth");
        std::fs::set_permissions(&auth, std::fs::Permissions::from_mode(0o600))
            .expect("make test Codex auth owner-only");
        auth
    }

    #[cfg(unix)]
    #[test]
    fn codex_app_server_isolated_home_is_private_linked_and_cleaned() {
        use std::os::unix::fs::MetadataExt;

        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);
        let source_home = fixture.base.join("source-codex-home");
        let source_auth = create_test_codex_auth(&source_home);
        let temp_root = fixture.base.join("private-temp");
        std::fs::create_dir(&temp_root).expect("create private temp root");

        let isolated = create_isolated_codex_home_in(
            &source_home,
            &temp_root,
            &[],
            fixture.policy.effective_user_visible_roots(),
        )
        .expect("create isolated Codex home");
        let isolated_path = isolated.path().to_path_buf();
        let metadata = std::fs::symlink_metadata(&isolated_path).expect("inspect isolated home");
        assert!(metadata.file_type().is_dir());
        assert_eq!(metadata.mode() & 0o777, 0o700);
        // SAFETY: geteuid has no preconditions and does not mutate process state.
        assert_eq!(metadata.uid(), unsafe { libc::geteuid() });
        let auth_link = isolated_path.join("auth.json");
        assert!(std::fs::symlink_metadata(&auth_link)
            .expect("inspect isolated auth link")
            .file_type()
            .is_symlink());
        assert_eq!(
            std::fs::canonicalize(&auth_link).expect("resolve isolated auth link"),
            std::fs::canonicalize(&source_auth).expect("resolve source auth")
        );
        assert!(isolated.auth_is_unchanged());

        drop(isolated);
        assert!(!isolated_path.exists(), "isolated home must be cleaned up");
        assert_eq!(
            std::fs::read_to_string(source_auth).expect("source auth survives cleanup"),
            r#"{"tokens":{"access_token":"test"}}"#
        );
    }

    #[cfg(unix)]
    #[test]
    fn codex_app_server_rejects_unsafe_auth_preconditions() {
        use std::os::unix::fs::PermissionsExt;

        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);

        let symlink_home = fixture.base.join("symlink-auth-home");
        std::fs::create_dir(&symlink_home).expect("create symlink auth home");
        let target = fixture.base.join("auth-target.json");
        std::fs::write(&target, "{}").expect("write symlink target");
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600))
            .expect("make symlink target private");
        std::os::unix::fs::symlink(&target, symlink_home.join("auth.json"))
            .expect("create auth symlink");
        assert!(validated_codex_auth_file(&symlink_home, &[])
            .expect_err("auth symlinks must fail closed")
            .to_string()
            .contains("regular, non-symlink"));

        let directory_home = fixture.base.join("directory-auth-home");
        std::fs::create_dir_all(directory_home.join("auth.json"))
            .expect("create directory at auth path");
        assert!(validated_codex_auth_file(&directory_home, &[]).is_err());

        let broad_mode_home = fixture.base.join("broad-mode-auth-home");
        let broad_mode_auth = create_test_codex_auth(&broad_mode_home);
        std::fs::set_permissions(&broad_mode_auth, std::fs::Permissions::from_mode(0o640))
            .expect("broaden auth permissions");
        assert!(validated_codex_auth_file(&broad_mode_home, &[])
            .expect_err("group-readable auth must fail closed")
            .to_string()
            .contains("owner-only"));

        let hard_link_home = fixture.base.join("hard-link-auth-home");
        let hard_link_auth = create_test_codex_auth(&hard_link_home);
        std::fs::hard_link(&hard_link_auth, fixture.base.join("auth-hard-link"))
            .expect("create auth hard link");
        assert!(validated_codex_auth_file(&hard_link_home, &[])
            .expect_err("hard-linked auth must fail closed")
            .to_string()
            .contains("exactly one hard link"));

        let rooted_home = fixture.base.join("rooted-auth-home");
        create_test_codex_auth(&rooted_home);
        let forbidden = vec![std::fs::canonicalize(&fixture.base).expect("canonical fixture root")];
        assert!(validated_codex_auth_file(&rooted_home, &forbidden)
            .expect_err("auth under a model-visible or temp root must fail closed")
            .to_string()
            .contains("model-visible workspace or temporary root"));
    }

    #[tokio::test]
    async fn codex_app_server_attests_before_one_model_turn() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);
        let contract = test_codex_app_server_contract(&fixture);
        let server_contract = contract.clone();
        let (client, server) = tokio::io::duplex(64 * 1024);
        let (client_read, mut client_write) = tokio::io::split(client);
        let (server_read, mut server_write) = tokio::io::split(server);
        let mut client_read = BufReader::new(client_read);
        let mut server_read = BufReader::new(server_read);
        let (tx, mut rx) = unbounded_channel();

        let fake = tokio::spawn(async move {
            fake_app_server_initialize(
                &mut server_read,
                &mut server_write,
                CODEX_APP_SERVER_VERSION,
                &server_contract.codex_home,
            )
            .await?;

            let thread_start = fake_app_server_request(&mut server_read, "thread/start").await?;
            let params = &thread_start["params"];
            assert_eq!(params["model"], server_contract.model);
            assert_eq!(params["modelProvider"], "openai");
            assert_eq!(params["allowProviderModelFallback"], false);
            assert_eq!(params["cwd"], json!(&server_contract.cwd));
            assert_eq!(
                params["runtimeWorkspaceRoots"],
                json!(&server_contract.workspace_roots)
            );
            assert_eq!(params["approvalPolicy"], "never");
            assert_eq!(params["approvalsReviewer"], "user");
            assert_eq!(params["permissions"], server_contract.profile);
            assert_eq!(params["ephemeral"], true);
            assert_eq!(params["historyMode"], "legacy");
            assert_eq!(
                params["environments"],
                json!([{
                    "environmentId": "local",
                    "cwd": server_contract.cwd,
                    "runtimeWorkspaceRoots": server_contract.workspace_roots
                }])
            );
            assert_eq!(params["dynamicTools"], json!([]));
            assert_eq!(params["selectedCapabilityRoots"], json!([]));
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "id": CODEX_APP_SERVER_THREAD_START_ID,
                    "result": fake_attested_thread_result(&server_contract, "thread-1")
                }),
            )
            .await?;

            let turn_start = fake_app_server_request(&mut server_read, "turn/start").await?;
            assert_eq!(turn_start["id"], CODEX_APP_SERVER_TURN_START_ID);
            assert_eq!(turn_start["params"]["threadId"], "thread-1");
            assert_eq!(turn_start["params"]["input"][0]["type"], "text");
            assert_eq!(turn_start["params"]["input"][0]["text"], "inspect only");
            assert_eq!(turn_start["params"]["input"][0]["textElements"], json!([]));
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "id": CODEX_APP_SERVER_TURN_START_ID,
                    "result": {"turn": {"id": "turn-1", "status": "inProgress"}}
                }),
            )
            .await?;

            // Wrongly correlated events cannot leak another thread's output or usage.
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "method": "item/completed",
                    "params": {
                        "threadId": "other-thread",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "id": "wrong", "text": "leak"}
                    }
                }),
            )
            .await?;
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "id": "approval-1",
                    "method": "item/commandExecution/requestApproval",
                    "params": {"threadId": "thread-1", "turnId": "turn-1"}
                }),
            )
            .await?;
            let rejection = read_codex_app_server_message(&mut server_read).await?;
            assert_eq!(rejection["id"], "approval-1");
            assert_eq!(rejection["error"]["code"], -32601);

            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "method": "warning",
                    "params": {"threadId": "thread-1", "message": "safe warning"}
                }),
            )
            .await?;
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {"type": "agentMessage", "id": "answer-1", "text": "pong"}
                    }
                }),
            )
            .await?;
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "tokenUsage": {"total": {
                            "totalTokens": 13298,
                            "inputTokens": 13293,
                            "cachedInputTokens": 2432,
                            "cacheWriteInputTokens": 0,
                            "outputTokens": 5,
                            "reasoningOutputTokens": 0
                        }}
                    }
                }),
            )
            .await?;
            send_codex_app_server_message(
                &mut server_write,
                &json!({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "turn": {
                            "id": "turn-1",
                            "status": "completed",
                            "error": null,
                            "items": [
                                {"type": "agentMessage", "id": "answer-1", "text": "pong"}
                            ]
                        }
                    }
                }),
            )
            .await?;
            Ok::<_, anyhow::Error>(())
        });

        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_codex_app_server_protocol(
                &mut client_read,
                &mut client_write,
                &contract,
                "inspect only",
                &tx,
            ),
        )
        .await
        .expect("fake app-server protocol should not hang")
        .expect("attested app-server turn");
        drop(client_write);
        fake.await
            .expect("fake app-server task")
            .expect("fake app-server exchange");

        let events = drain(&mut rx);
        assert!(matches!(&events[0], ChatEvent::Notice(message) if message == "safe warning"));
        assert!(matches!(&events[1], ChatEvent::TextDelta(text) if text == "pong"));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ChatEvent::TextDelta(_)))
                .count(),
            1,
            "turn summary must not duplicate the completed item"
        );
        match &events[2] {
            ChatEvent::Completed {
                usage: Some(usage), ..
            } => {
                assert_eq!(usage.input_tokens, 13293);
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected completed usage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn codex_app_server_attestation_failure_never_starts_a_turn() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);
        let contract = test_codex_app_server_contract(&fixture);
        let server_contract = contract.clone();
        let (client, server) = tokio::io::duplex(32 * 1024);
        let (client_read, mut client_write) = tokio::io::split(client);
        let (server_read, mut server_write) = tokio::io::split(server);
        let mut client_read = BufReader::new(client_read);
        let mut server_read = BufReader::new(server_read);
        let (tx, mut rx) = unbounded_channel();

        let fake = tokio::spawn(async move {
            fake_app_server_initialize(
                &mut server_read,
                &mut server_write,
                CODEX_APP_SERVER_VERSION,
                &server_contract.codex_home,
            )
            .await?;
            let _thread_start = fake_app_server_request(&mut server_read, "thread/start").await?;
            let mut result = fake_attested_thread_result(&server_contract, "thread-1");
            result["activePermissionProfile"]["id"] = json!("ambient-broader-profile");
            send_codex_app_server_message(
                &mut server_write,
                &json!({"id": CODEX_APP_SERVER_THREAD_START_ID, "result": result}),
            )
            .await?;

            let next = read_codex_app_server_message(&mut server_read).await;
            Ok::<_, anyhow::Error>(next.ok())
        });

        let error = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_codex_app_server_protocol(
                &mut client_read,
                &mut client_write,
                &contract,
                "must not be billed",
                &tx,
            ),
        )
        .await
        .expect("failed attestation should not hang")
        .expect_err("mismatched permission profile must fail closed");
        assert!(error.to_string().contains("attestation"));
        drop(client_write);
        drop(client_read);
        assert!(fake
            .await
            .expect("fake app-server task")
            .expect("fake app-server exchange")
            .is_none());
        assert!(drain(&mut rx).is_empty());
    }

    #[test]
    fn codex_app_server_rejects_unsupported_versions_and_contract_drift() {
        let initialize = json!({
            "userAgent": "shaltaiboltai/0.153.0 (test)",
            "codexHome": "/tmp/fake-codex-home",
            "platformFamily": std::env::consts::FAMILY,
            "platformOs": std::env::consts::OS
        });
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);
        let contract = test_codex_app_server_contract(&fixture);
        let error = attest_codex_app_server_initialize(&initialize, &contract)
            .expect_err("unreviewed app-server versions must fail closed");
        assert!(error.to_string().contains("expected exactly 0.152.1"));

        let wrong_home = json!({
            "userAgent": format!("shaltaiboltai/{CODEX_APP_SERVER_VERSION} (test)"),
            "codexHome": "/tmp/not-the-isolated-home",
            "platformFamily": std::env::consts::FAMILY,
            "platformOs": std::env::consts::OS
        });
        assert!(attest_codex_app_server_initialize(&wrong_home, &contract)
            .expect_err("Codex must attest the exact isolated home")
            .to_string()
            .contains("exact isolated Codex home"));

        let mut result = fake_attested_thread_result(&contract, "thread-1");
        assert_eq!(
            attest_codex_app_server_thread(&result, &contract).unwrap(),
            "thread-1"
        );
        result["model"] = json!("wrong-model");
        assert!(attest_codex_app_server_thread(&result, &contract).is_err());
        result["model"] = json!(contract.model);
        result["runtimeWorkspaceRoots"] = json!([contract.cwd]);
        assert!(attest_codex_app_server_thread(&result, &contract).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn codex_app_server_rejects_instruction_source_symlink_escape() {
        let fixture = PolicyFixture::new(SandboxMode::ReadOnly, ApprovalPolicy::Never);
        let contract = test_codex_app_server_contract(&fixture);
        let outside = fixture.base.join("outside-agents.md");
        std::fs::write(&outside, "outside instructions").expect("write outside instructions");
        let lexical = fixture.cwd().join("AGENTS.md");
        std::os::unix::fs::symlink(&outside, &lexical).expect("create instruction symlink");
        let mut result = fake_attested_thread_result(&contract, "thread-1");
        result["instructionSources"] = json!([lexical]);

        let error = attest_codex_app_server_thread(&result, &contract)
            .expect_err("instruction symlink escape must fail before turn/start");
        assert!(error.to_string().contains("escaped the advisory workspace"));
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
