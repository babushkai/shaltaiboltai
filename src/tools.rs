use crate::policy::{
    ApprovalPolicy, BoundaryAction, ExecutionPolicy, GrantBinding, PathClassification, SandboxMode,
};
use crate::providers::{ToolCall, ToolDef};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_READ_FILE_BYTES: usize = MAX_OUTPUT_BYTES;
const MAX_DIRECTORY_ENTRIES: usize = 1_000;
const MAX_SEARCH_RESULTS: usize = 200;
const MAX_SEARCH_FILE_BYTES: u64 = 1024 * 1024;
const MAX_SEARCH_VISITED_FILES: usize = 20_000;
const MAX_SEARCH_DURATION: Duration = Duration::from_secs(10);
const SEARCH_OUTPUT_LIMIT_MARKER: &str = "… search stopped at 32 KiB output limit";
const MAX_DIFF_PREVIEW_LINES: usize = 40;
const MAX_EDITABLE_FILE_BYTES: usize = 256 * 1024;
const MAX_DIFF_INPUT_LINES: usize = 10_000;
const MAX_DIFF_DURATION: Duration = Duration::from_millis(25);

#[derive(Clone, Copy)]
struct SearchLimits {
    max_visited_files: usize,
    max_duration: Duration,
}

const DEFAULT_SEARCH_LIMITS: SearchLimits = SearchLimits {
    max_visited_files: MAX_SEARCH_VISITED_FILES,
    max_duration: MAX_SEARCH_DURATION,
};

#[derive(Clone, Copy)]
enum SearchStop {
    Cancelled,
    Duration,
    VisitedFiles,
}

impl SearchStop {
    fn message(self, limits: SearchLimits) -> String {
        match self {
            Self::Cancelled => "cancelled".into(),
            Self::Duration => format!(
                "reached the {}-second time limit",
                limits.max_duration.as_secs()
            ),
            Self::VisitedFiles => {
                format!("reached the {}-file visit limit", limits.max_visited_files)
            }
        }
    }
}

/// Dropping the async search future cannot abort a running `spawn_blocking`
/// closure, so signal it explicitly. Walkers check this flag between entries
/// and while scanning lines, allowing Esc/worker timeout to stop detached CPU
/// and filesystem work promptly.
struct CancelSearchOnDrop(Arc<AtomicBool>);

impl Drop for CancelSearchOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
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
        // SAFETY: `pgid` came from the child we just spawned after configuring
        // it as its own process-group leader. A negative pid targets only that
        // group. ESRCH means the group has already exited.
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

pub fn definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "read_file",
            description: "Read a UTF-8 text file and return its contents.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, absolute or relative to the working directory."}
                },
                "required": ["path"]
            }),
        },
        ToolDef {
            name: "write_file",
            description: "Write content to a file, creating it (and parent directories) if needed, overwriting if it exists. For changes to an existing file prefer edit_file.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Full file content to write."}
                },
                "required": ["path", "content"]
            }),
        },
        ToolDef {
            name: "edit_file",
            description: "Replace an exact string in a file. old_string must match exactly once unless replace_all is true; include surrounding lines to make it unique.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "old_string": {"type": "string", "description": "Exact text to find."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."}
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolDef {
            name: "list_directory",
            description: "List the entries of a directory.",
            schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to the working directory."}
                }
            }),
        },
        ToolDef {
            name: "grep",
            description: "Search file contents with a regular expression, recursively. Respects .gitignore and skips hidden/binary files. Returns path:line:text matches.",
            schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "Directory to search under. Defaults to the working directory."}
                },
                "required": ["pattern"]
            }),
        },
        ToolDef {
            name: "glob",
            description: "Find files by name with a glob pattern (e.g. **/*.rs). Respects .gitignore.",
            schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern matched against paths relative to the search root."},
                    "path": {"type": "string", "description": "Directory to search under. Defaults to the working directory."}
                },
                "required": ["pattern"]
            }),
        },
        ToolDef {
            name: "run_command",
            description: "Run a shell command in the working directory and return stdout/stderr. 60 second timeout.",
            schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "sandbox_permissions": {
                        "type": "string",
                        "enum": ["use_default", "require_escalated"],
                        "description": "Use the configured sandbox, or explicitly request execution outside it. Defaults to use_default."
                    },
                    "justification": {
                        "type": "string",
                        "description": "Optional concise reason shown when elevated execution is requested."
                    }
                },
                "required": ["command"]
            }),
        },
    ]
}

/// Whether a model-initiated tool can execute immediately, must be presented
/// for a narrowly scoped user decision, or is forbidden by active policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolDecision {
    Allow,
    Ask,
    Deny,
}

/// A fully resolved, policy-bound assessment. Callers may use `scope` to bind
/// a [`GrantBinding`], but execution always recomputes this assessment so a
/// retargeted symlink or changed policy invalidates the earlier decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolAssessment {
    decision: ToolDecision,
    scope: Vec<u8>,
    canonical_target: Option<PathBuf>,
    reason: Option<String>,
    requires_escalation: bool,
}

impl ToolAssessment {
    pub const fn decision(&self) -> ToolDecision {
        self.decision
    }

    pub fn scope(&self) -> &[u8] {
        &self.scope
    }

    pub fn canonical_target(&self) -> Option<&Path> {
        self.canonical_target.as_deref()
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// True only for `run_command` calls that explicitly requested execution
    /// outside the configured sandbox. An approval for an untrusted default
    /// command deliberately leaves this false and keeps the command sandboxed.
    pub const fn requires_escalation(&self) -> bool {
        self.requires_escalation
    }
}

/// Explicit authority supplied to the execution boundary. Session approvals
/// remain tied to the exact scope, authority instance, policy fingerprint, and
/// policy generation recorded by [`ExecutionPolicy::bind_grant`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ToolAuthorization {
    #[default]
    Default,
    Approved(GrantBinding),
}

impl ToolAuthorization {
    pub fn approved(policy: &ExecutionPolicy, assessment: &ToolAssessment) -> Self {
        Self::Approved(policy.bind_grant(assessment.scope.clone()))
    }
}

/// Requested process authority for `run_command`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SandboxPermission {
    #[default]
    UseDefault,
    RequireEscalated,
}

impl SandboxPermission {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UseDefault => "use_default",
            Self::RequireEscalated => "require_escalated",
        }
    }
}

impl FromStr for SandboxPermission {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "use_default" => Ok(Self::UseDefault),
            "require_escalated" => Ok(Self::RequireEscalated),
            _ => Err(format!("invalid sandbox_permissions `{value}`")),
        }
    }
}

/// Assess a call solely against explicit runtime authority. No decision in
/// this module consults the process working directory.
pub fn assess(policy: &ExecutionPolicy, call: &ToolCall) -> ToolAssessment {
    let args = &call.arguments;
    match call.name.as_str() {
        "read_file" => {
            let Some(path) = required_string(args, "path") else {
                return denied(policy, call, "missing required argument: path");
            };
            assess_path(policy, call, path, false, None)
        }
        "write_file" => {
            let Some(path) = required_string(args, "path") else {
                return denied(policy, call, "missing required argument: path");
            };
            if required_string(args, "content").is_none() {
                return denied(policy, call, "missing required argument: content");
            }
            assess_path(policy, call, path, true, None)
        }
        "edit_file" => {
            let Some(path) = required_string(args, "path") else {
                return denied(policy, call, "missing required argument: path");
            };
            if required_string(args, "old_string").is_none() {
                return denied(policy, call, "missing required argument: old_string");
            }
            if required_string(args, "new_string").is_none() {
                return denied(policy, call, "missing required argument: new_string");
            }
            if args
                .get("replace_all")
                .is_some_and(|value| !value.is_boolean())
            {
                return denied(policy, call, "replace_all must be a boolean");
            }
            assess_path(policy, call, path, true, None)
        }
        "list_directory" => {
            let Some(path) = optional_path(args) else {
                return denied(policy, call, "path must be a string");
            };
            assess_path(policy, call, path, false, None)
        }
        "grep" => {
            let Some(pattern) = required_string(args, "pattern") else {
                return denied(policy, call, "missing required argument: pattern");
            };
            if let Err(error) = regex::Regex::new(pattern) {
                return denied(policy, call, format!("invalid regex: {error}"));
            }
            let Some(path) = optional_path(args) else {
                return denied(policy, call, "path must be a string");
            };
            assess_path(policy, call, path, false, Some(pattern))
        }
        "glob" => {
            let Some(pattern) = required_string(args, "pattern") else {
                return denied(policy, call, "missing required argument: pattern");
            };
            if let Err(error) = globset::GlobBuilder::new(pattern)
                .literal_separator(false)
                .build()
            {
                return denied(policy, call, format!("invalid glob pattern: {error}"));
            }
            let Some(path) = optional_path(args) else {
                return denied(policy, call, "path must be a string");
            };
            assess_path(policy, call, path, false, Some(pattern))
        }
        "run_command" => assess_command(policy, call),
        _ => denied(policy, call, format!("unknown tool: {}", call.name)),
    }
}

fn assess_path(
    policy: &ExecutionPolicy,
    call: &ToolCall,
    raw_path: &str,
    write: bool,
    search_pattern: Option<&str>,
) -> ToolAssessment {
    let classification = if write {
        policy.classify_write(raw_path)
    } else {
        policy.classify_read(raw_path)
    };
    let classification = match classification {
        Ok(classification) => classification,
        Err(error) => return denied(policy, call, error.to_string()),
    };
    let scope = path_scope(policy, call, &classification, search_pattern);
    ToolAssessment {
        decision: decision_for(classification.action()),
        scope,
        canonical_target: Some(classification.target().to_path_buf()),
        reason: decision_reason(classification.action(), write),
        requires_escalation: false,
    }
}

fn assess_command(policy: &ExecutionPolicy, call: &ToolCall) -> ToolAssessment {
    let Some(command) = required_string(&call.arguments, "command") else {
        return denied(policy, call, "missing required argument: command");
    };
    if call
        .arguments
        .get("justification")
        .is_some_and(|value| !value.is_string())
    {
        return denied(policy, call, "justification must be a string");
    }
    let permission = match call.arguments.get("sandbox_permissions") {
        None => SandboxPermission::UseDefault,
        Some(Value::String(value)) => match value.parse() {
            Ok(permission) => permission,
            Err(error) => return denied(policy, call, error),
        },
        Some(_) => return denied(policy, call, "sandbox_permissions must be a string"),
    };

    let decision = match permission {
        SandboxPermission::UseDefault if policy.approval_policy() == ApprovalPolicy::Untrusted => {
            ToolDecision::Ask
        }
        SandboxPermission::UseDefault => ToolDecision::Allow,
        SandboxPermission::RequireEscalated
            if policy.sandbox_mode() == SandboxMode::DangerFullAccess =>
        {
            ToolDecision::Allow
        }
        SandboxPermission::RequireEscalated
            if policy.approval_policy() == ApprovalPolicy::Never =>
        {
            ToolDecision::Deny
        }
        SandboxPermission::RequireEscalated => ToolDecision::Ask,
    };
    let reason = match decision {
        ToolDecision::Allow => None,
        ToolDecision::Ask if permission == SandboxPermission::RequireEscalated => {
            Some("command requests execution outside the configured sandbox".into())
        }
        ToolDecision::Ask => Some("untrusted policy requires approval for shell commands".into()),
        ToolDecision::Deny => {
            Some("approval policy `never` forbids execution outside the configured sandbox".into())
        }
    };
    let fingerprint = policy.fingerprint().to_string();
    ToolAssessment {
        decision,
        scope: encoded_scope([
            b"policy".as_slice(),
            fingerprint.as_bytes(),
            b"run_command".as_slice(),
            permission.as_str().as_bytes(),
            command.as_bytes(),
        ]),
        canonical_target: None,
        reason,
        requires_escalation: permission == SandboxPermission::RequireEscalated,
    }
}

fn path_scope(
    policy: &ExecutionPolicy,
    call: &ToolCall,
    classification: &PathClassification,
    search_pattern: Option<&str>,
) -> Vec<u8> {
    let fingerprint = policy.fingerprint().to_string();
    let mut fields = vec![
        b"policy".as_slice(),
        fingerprint.as_bytes(),
        call.name.as_bytes(),
        classification.target().as_os_str().as_encoded_bytes(),
    ];
    if let Some(pattern) = search_pattern {
        fields.push(pattern.as_bytes());
    }
    encoded_scope(fields)
}

/// Length-prefix every field so scopes are unambiguous and preserve exact
/// OS-native path bytes. In particular, never use `Path::display()` here:
/// distinct non-UTF-8 paths can have the same lossy rendering.
fn encoded_scope<'a>(fields: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let mut scope = Vec::new();
    for field in fields {
        scope.extend_from_slice(&(field.len() as u64).to_be_bytes());
        scope.extend_from_slice(field);
    }
    scope
}

fn decision_for(action: BoundaryAction) -> ToolDecision {
    match action {
        BoundaryAction::Allow => ToolDecision::Allow,
        BoundaryAction::RequiresApproval => ToolDecision::Ask,
        BoundaryAction::Deny => ToolDecision::Deny,
    }
}

fn decision_reason(action: BoundaryAction, write: bool) -> Option<String> {
    match action {
        BoundaryAction::Allow => None,
        BoundaryAction::RequiresApproval if write => {
            Some("write crosses the active sandbox boundary".into())
        }
        BoundaryAction::RequiresApproval => Some("read crosses the active sandbox boundary".into()),
        BoundaryAction::Deny => Some("active policy forbids this filesystem access".into()),
    }
}

fn denied(policy: &ExecutionPolicy, call: &ToolCall, reason: impl Into<String>) -> ToolAssessment {
    let serialized =
        serde_json::to_string(&call.arguments).unwrap_or_else(|_| call.arguments.to_string());
    let fingerprint = policy.fingerprint().to_string();
    ToolAssessment {
        decision: ToolDecision::Deny,
        scope: encoded_scope([
            b"policy".as_slice(),
            fingerprint.as_bytes(),
            call.name.as_bytes(),
            b"invalid".as_slice(),
            serialized.as_bytes(),
        ]),
        canonical_target: None,
        reason: Some(reason.into()),
        requires_escalation: false,
    }
}

fn required_string<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn optional_path(args: &Value) -> Option<&str> {
    match args.get("path") {
        None => Some("."),
        Some(Value::String(path)) => Some(path),
        Some(_) => None,
    }
}

/// Human wording paired with [`ToolAssessment::scope`] in the approval footer.
pub fn approval_scope_label(call: &ToolCall) -> &'static str {
    match call.name.as_str() {
        "read_file" | "write_file" | "edit_file" | "list_directory" => "this path",
        "grep" | "glob" => "this search",
        "run_command" => "this exact command",
        _ => "these exact arguments",
    }
}

/// One-line human-readable summary shown in the approval prompt and transcript.
pub fn describe(call: &ToolCall) -> String {
    let arg = |k: &str| call.arguments[k].as_str().unwrap_or("?");
    match call.name.as_str() {
        "read_file" => format!("read_file {}", arg("path")),
        "write_file" => format!(
            "write_file {} ({} bytes)",
            arg("path"),
            call.arguments["content"].as_str().map_or(0, str::len)
        ),
        "edit_file" => format!("edit_file {}", arg("path")),
        "list_directory" => {
            format!(
                "list_directory {}",
                call.arguments["path"].as_str().unwrap_or(".")
            )
        }
        "grep" => format!(
            "grep /{}/ in {}",
            arg("pattern"),
            call.arguments["path"].as_str().unwrap_or(".")
        ),
        "glob" => format!(
            "glob {} in {}",
            arg("pattern"),
            call.arguments["path"].as_str().unwrap_or(".")
        ),
        "run_command" => format!("run_command: {}", arg("command")),
        other => format!("{other} {}", call.arguments),
    }
}

/// Diff preview for the approval dialog: what the file change would do.
/// Tags: '+' insert, '-' delete, ' ' context, '@' hunk header, '!' problem.
pub fn approval_preview(
    assessment: &ToolAssessment,
    call: &ToolCall,
) -> Option<Vec<(char, String)>> {
    let path = assessment.canonical_target()?;
    match call.name.as_str() {
        "write_file" => {
            let old = match read_change_file_bounded(path, true) {
                Ok(old) => old,
                Err(error) => {
                    return Some(vec![(
                        '!',
                        format!(
                            "preview unavailable for {}: {error:#}; approval may overwrite the existing target without a diff",
                            path.display()
                        ),
                    )]);
                }
            };
            let new = call.arguments["content"].as_str()?;
            Some(match diff_lines_bounded(&old, new) {
                Ok(lines) => lines,
                Err(error) => vec![(
                    '!',
                    format!(
                        "preview unavailable for {}: {error:#}; approval may overwrite the target without a diff",
                        path.display()
                    ),
                )],
            })
        }
        "edit_file" => {
            let old = match read_change_file_bounded(path, false) {
                Ok(s) => s,
                Err(e) => {
                    return Some(vec![('!', format!("cannot read {}: {e}", path.display()))]);
                }
            };
            match apply_edit_bounded(
                &old,
                call.arguments["old_string"].as_str()?,
                call.arguments["new_string"].as_str()?,
                call.arguments["replace_all"].as_bool().unwrap_or(false),
            ) {
                Ok(new) => Some(match diff_lines_bounded(&old, &new) {
                    Ok(lines) => lines,
                    Err(error) => vec![('!', format!("preview unavailable: {error:#}"))],
                }),
                Err(e) => Some(vec![('!', format!("{e:#}"))]),
            }
        }
        _ => None,
    }
}

fn read_change_file_bounded(path: &Path, missing_is_empty: bool) -> Result<String> {
    let initial_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if missing_is_empty && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(String::new());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    };
    if !initial_metadata.file_type().is_file() {
        anyhow::bail!("{} is not a regular file", path.display());
    }
    if initial_metadata.len() > MAX_EDITABLE_FILE_BYTES as u64 {
        anyhow::bail!(
            "{} is {} bytes; editable files are limited to {} bytes",
            path.display(),
            initial_metadata.len(),
            MAX_EDITABLE_FILE_BYTES
        );
    }

    let (file, opened_metadata) = open_regular_file_nonblocking(path)?;
    if opened_metadata.len() > MAX_EDITABLE_FILE_BYTES as u64 {
        anyhow::bail!(
            "{} is {} bytes; editable files are limited to {} bytes",
            path.display(),
            opened_metadata.len(),
            MAX_EDITABLE_FILE_BYTES
        );
    }

    let mut bytes =
        Vec::with_capacity((opened_metadata.len() as usize).min(MAX_EDITABLE_FILE_BYTES));
    file.take((MAX_EDITABLE_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() > MAX_EDITABLE_FILE_BYTES {
        anyhow::bail!(
            "{} exceeds the {}-byte editable-file limit",
            path.display(),
            MAX_EDITABLE_FILE_BYTES
        );
    }
    String::from_utf8(bytes).with_context(|| {
        format!(
            "{} is not UTF-8 text; binary files cannot be previewed or edited safely",
            path.display()
        )
    })
}

fn open_regular_file_nonblocking(path: &Path) -> Result<(std::fs::File, std::fs::Metadata)> {
    let initial_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if !initial_metadata.file_type().is_file() {
        anyhow::bail!("{} is not a regular file", path.display());
    }

    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let opened_metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened file {}", path.display()))?;
    if !opened_metadata.file_type().is_file() {
        anyhow::bail!("{} is not a regular file", path.display());
    }
    Ok((file, opened_metadata))
}

fn write_regular_file_nonblocking(
    path: &Path,
    content: &[u8],
    reject_hard_links: bool,
) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            anyhow::bail!("{} is not a regular file", path.display());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    }

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to open {} for writing", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened file {}", path.display()))?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("{} is not a regular file", path.display());
    }
    #[cfg(unix)]
    if reject_hard_links {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() > 1 {
            anyhow::bail!(
                "refusing to overwrite {} because it has multiple hard links under constrained authority",
                path.display()
            );
        }
    }
    #[cfg(windows)]
    if reject_hard_links {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
        };

        let mut information = BY_HANDLE_FILE_INFORMATION::default();
        // SAFETY: `file` owns a valid, open Windows file handle and
        // `information` remains writable for the duration of the call.
        let inspected =
            unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), &mut information) };
        if inspected == 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| {
                format!(
                    "failed to inspect hard-link count for opened file {}",
                    path.display()
                )
            });
        }
        if information.nNumberOfLinks > 1 {
            anyhow::bail!(
                "refusing to overwrite {} because it has multiple hard links under constrained authority",
                path.display()
            );
        }
    }
    #[cfg(not(any(unix, windows)))]
    if reject_hard_links {
        anyhow::bail!(
            "refusing to overwrite {} because this platform cannot verify hard-link count under constrained authority",
            path.display()
        );
    }
    file.set_len(0)
        .with_context(|| format!("failed to truncate {}", path.display()))?;
    file.write_all(content)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))
}

fn diff_lines_bounded(old: &str, new: &str) -> Result<Vec<(char, String)>> {
    for (label, text) in [("existing", old), ("replacement", new)] {
        if text.len() > MAX_EDITABLE_FILE_BYTES {
            anyhow::bail!(
                "{label} content is {} bytes; diff inputs are limited to {} bytes",
                text.len(),
                MAX_EDITABLE_FILE_BYTES
            );
        }
        if text.lines().take(MAX_DIFF_INPUT_LINES + 1).count() > MAX_DIFF_INPUT_LINES {
            anyhow::bail!("{label} content exceeds the {MAX_DIFF_INPUT_LINES}-line diff limit");
        }
    }
    Ok(diff_lines(old, new))
}

fn diff_lines(old: &str, new: &str) -> Vec<(char, String)> {
    let mut configuration = similar::TextDiff::configure();
    configuration.timeout(MAX_DIFF_DURATION);
    let diff = configuration.diff_lines(old, new);
    let mut out = Vec::new();
    for hunk in diff.unified_diff().context_radius(2).iter_hunks() {
        out.push(('@', hunk.header().to_string()));
        for change in hunk.iter_changes() {
            let tag = match change.tag() {
                similar::ChangeTag::Insert => '+',
                similar::ChangeTag::Delete => '-',
                similar::ChangeTag::Equal => ' ',
            };
            out.push((tag, change.value().trim_end_matches('\n').to_owned()));
            if out.len() >= MAX_DIFF_PREVIEW_LINES {
                out.push(('@', "… diff truncated".into()));
                return out;
            }
        }
    }
    if out.is_empty() {
        out.push((' ', "(no changes)".into()));
    }
    out
}

/// Execute one call under explicit runtime authority. The call is reassessed
/// immediately, and only the canonical target from that fresh assessment is
/// used for filesystem I/O. A grant can satisfy `Ask`; it can never override
/// `Deny`.
pub async fn execute(
    policy: &ExecutionPolicy,
    call: &ToolCall,
    authorization: &ToolAuthorization,
) -> (String, bool) {
    match run(policy, call, authorization).await {
        Ok(output) => (truncate(output), false),
        Err(e) => (format!("{e:#}"), true),
    }
}

async fn run(
    policy: &ExecutionPolicy,
    call: &ToolCall,
    authorization: &ToolAuthorization,
) -> Result<String> {
    // This is intentionally independent from the assessment used to render an
    // approval dialog. It closes policy-generation and symlink-retarget races.
    let assessment = assess(policy, call);
    let approved = authorize(policy, &assessment, authorization)?;
    let args = &call.arguments;
    match call.name.as_str() {
        "read_file" => {
            let path = assessed_target(&assessment)?;
            read_file_bounded(path).await
        }
        "write_file" => {
            let path = assessed_target(&assessment)?;
            let content = str_arg(args, "content")?;
            let reject_hard_links = policy.sandbox_mode() != SandboxMode::DangerFullAccess;
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await?;
                }
            }
            let write_path = path.to_path_buf();
            let write_content = content.as_bytes().to_vec();
            tokio::task::spawn_blocking(move || {
                write_regular_file_nonblocking(&write_path, &write_content, reject_hard_links)
            })
            .await
            .context("regular-file writer task failed")??;
            Ok(format!(
                "wrote {} bytes to {}",
                content.len(),
                path.display()
            ))
        }
        "edit_file" => {
            let path = assessed_target(&assessment)?;
            let reject_hard_links = policy.sandbox_mode() != SandboxMode::DangerFullAccess;
            let editable_path = path.to_path_buf();
            let content = tokio::task::spawn_blocking(move || {
                read_change_file_bounded(&editable_path, false)
            })
            .await
            .context("editable-file reader task failed")??;
            let updated = apply_edit_bounded(
                &content,
                str_arg(args, "old_string")?,
                str_arg(args, "new_string")?,
                args["replace_all"].as_bool().unwrap_or(false),
            )?;
            let write_path = path.to_path_buf();
            tokio::task::spawn_blocking(move || {
                write_regular_file_nonblocking(&write_path, updated.as_bytes(), reject_hard_links)
            })
            .await
            .context("regular-file writer task failed")??;
            Ok(format!("edited {}", path.display()))
        }
        "list_directory" => {
            let path = assessed_target(&assessment)?;
            list_directory_bounded(path, MAX_DIRECTORY_ENTRIES).await
        }
        "grep" => {
            let pattern = str_arg(args, "pattern")?.to_owned();
            let root = assessed_target(&assessment)?.to_path_buf();
            run_cancellable_search(move |cancelled| {
                grep_files_with_limits(&pattern, root.as_path(), &cancelled, DEFAULT_SEARCH_LIMITS)
            })
            .await
        }
        "glob" => {
            let pattern = str_arg(args, "pattern")?.to_owned();
            let root = assessed_target(&assessment)?.to_path_buf();
            run_cancellable_search(move |cancelled| {
                glob_files_with_limits(&pattern, root.as_path(), &cancelled, DEFAULT_SEARCH_LIMITS)
            })
            .await
        }
        "run_command" => {
            let command = str_arg(args, "command")?;
            // Approval for an untrusted *default* command is not escalation:
            // the process still goes through the constrained backend.
            let approved_escalation = assessment.requires_escalation() && approved;
            let prepared =
                crate::sandbox::prepare_shell_command(policy, command, approved_escalation)?;
            let (command, cleanup) = prepared.into_tokio_parts();
            let result = run_command_bounded(command).await;
            cleanup.cleanup()?;
            result
        }
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

fn authorize(
    policy: &ExecutionPolicy,
    assessment: &ToolAssessment,
    authorization: &ToolAuthorization,
) -> Result<bool> {
    match assessment.decision() {
        ToolDecision::Deny => anyhow::bail!(
            "tool denied by policy: {}",
            assessment.reason().unwrap_or("access is forbidden")
        ),
        ToolDecision::Allow => Ok(false),
        ToolDecision::Ask => match authorization {
            ToolAuthorization::Approved(grant)
                if policy.accepts_grant(grant, assessment.scope()) =>
            {
                Ok(true)
            }
            ToolAuthorization::Approved(_) => {
                anyhow::bail!("tool approval is stale or does not match this exact request")
            }
            ToolAuthorization::Default => anyhow::bail!(
                "tool requires approval: {}",
                assessment.reason().unwrap_or("approval required")
            ),
        },
    }
}

fn assessed_target(assessment: &ToolAssessment) -> Result<&Path> {
    assessment
        .canonical_target()
        .context("tool assessment has no canonical target")
}

struct BoundedBytes {
    bytes: Vec<u8>,
    truncated: bool,
}

async fn drain_bounded(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    max_bytes: usize,
) -> std::io::Result<BoundedBytes> {
    let mut output = Vec::with_capacity(max_bytes.min(8 * 1024));
    let mut truncated = false;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        let remaining = max_bytes.saturating_sub(output.len());
        let retained = remaining.min(read);
        output.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    Ok(BoundedBytes {
        bytes: output,
        truncated,
    })
}

async fn run_command_bounded(mut command: tokio::process::Command) -> Result<String> {
    #[cfg(unix)]
    command.as_std_mut().process_group(0);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().context("failed to start command")?;
    #[cfg(unix)]
    let mut process_group = ProcessGroupGuard::new(
        child
            .id()
            .context("spawned command did not expose a process id")?,
    )?;
    let stdout = child.stdout.take().context("failed to capture stdout")?;
    let stderr = child.stderr.take().context("failed to capture stderr")?;
    let stdout_task = tokio::spawn(drain_bounded(stdout, MAX_OUTPUT_BYTES));
    let stderr_task = tokio::spawn(drain_bounded(stderr, MAX_OUTPUT_BYTES));
    let execution = async {
        let status = child.wait().await.context("failed to wait for command")?;
        #[cfg(unix)]
        process_group
            .terminate()
            .context("failed to terminate command descendants")?;
        let stdout = stdout_task
            .await
            .context("stdout drain task failed")?
            .context("failed to read command stdout")?;
        let stderr = stderr_task
            .await
            .context("stderr drain task failed")?
            .context("failed to read command stderr")?;
        Ok::<_, anyhow::Error>((status, stdout, stderr))
    };
    let (status, stdout, stderr) = tokio::time::timeout(COMMAND_TIMEOUT, execution)
        .await
        .context("command timed out after 60s")??;

    let mut result = String::from_utf8_lossy(&stdout.bytes).into_owned();
    if !stderr.bytes.is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&String::from_utf8_lossy(&stderr.bytes));
    }
    if stdout.truncated || stderr.truncated {
        result.push_str("\n[output truncated]");
    }
    let result = truncate(result);
    if !status.success() {
        anyhow::bail!("exit status {status}\n{result}");
    }
    Ok(if result.trim().is_empty() {
        "(no output)".into()
    } else {
        result
    })
}

/// Read only the prefix that can be returned to the model, plus one byte to
/// detect overflow. This bounds regular files and special streams alike rather
/// than allocating the entire input before the shared output truncation runs.
async fn read_file_bounded(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || read_file_bounded_sync(&path))
        .await
        .context("bounded file reader task failed")?
}

fn read_file_bounded_sync(path: &Path) -> Result<String> {
    let (file, _) = open_regular_file_nonblocking(path)?;
    let mut bytes = Vec::with_capacity(MAX_READ_FILE_BYTES + 1);
    file.take((MAX_READ_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let overflow = bytes.len() > MAX_READ_FILE_BYTES;
    if overflow {
        bytes.truncate(MAX_READ_FILE_BYTES);
    }
    let mut text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) if overflow && error.utf8_error().error_len().is_none() => {
            // A valid multi-byte scalar may cross the bounded prefix. Keep the
            // complete UTF-8 portion; malformed bytes earlier still fail as
            // they did when the whole file was read.
            let valid_up_to = error.utf8_error().valid_up_to();
            let mut bytes = error.into_bytes();
            bytes.truncate(valid_up_to);
            String::from_utf8(bytes).expect("validated UTF-8 prefix")
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read {} as UTF-8", path.display()));
        }
    };
    if overflow {
        // The public execution path applies the same output cap and marker as
        // every other tool. Appending here ensures a split UTF-8 scalar cannot
        // make an oversized input look complete after its prefix is shortened.
        text.push_str("\n[output truncated]");
    }
    Ok(text)
}

async fn list_directory_bounded(path: &Path, max_entries: usize) -> Result<String> {
    let mut entries = tokio::fs::read_dir(path)
        .await
        .with_context(|| format!("failed to list {}", path.display()))?;
    let mut names = Vec::with_capacity(max_entries.min(256));
    let mut overflow = false;
    while let Some(entry) = entries.next_entry().await? {
        if names.len() >= max_entries {
            overflow = true;
            break;
        }
        let suffix = if entry.file_type().await?.is_dir() {
            "/"
        } else {
            ""
        };
        names.push(format!("{}{suffix}", entry.file_name().to_string_lossy()));
    }
    names.sort();
    if overflow {
        names.push(format!("… stopped at {max_entries} entries"));
    }
    Ok(names.join("\n"))
}

async fn run_cancellable_search(
    search: impl FnOnce(Arc<AtomicBool>) -> Result<String> + Send + 'static,
) -> Result<String> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelSearchOnDrop(cancelled.clone());
    tokio::task::spawn_blocking(move || search(cancelled)).await?
}

fn apply_edit(content: &str, old: &str, new: &str, replace_all: bool) -> Result<String> {
    let matches = content.matches(old).count();
    match matches {
        0 => anyhow::bail!("old_string not found in file"),
        1 => Ok(content.replacen(old, new, 1)),
        n if replace_all => {
            let _ = n;
            Ok(content.replace(old, new))
        }
        n => anyhow::bail!(
            "old_string matches {n} times — include more surrounding context to make it unique, or set replace_all"
        ),
    }
}

fn apply_edit_bounded(content: &str, old: &str, new: &str, replace_all: bool) -> Result<String> {
    for (label, text) in [("file", content), ("old_string", old), ("new_string", new)] {
        if text.len() > MAX_EDITABLE_FILE_BYTES {
            anyhow::bail!(
                "{label} is {} bytes; edits are limited to {} bytes",
                text.len(),
                MAX_EDITABLE_FILE_BYTES
            );
        }
    }

    let matches = content.matches(old).count();
    let replacements = match matches {
        1 => 1,
        count if count > 1 && replace_all => count,
        _ => 0,
    };
    if replacements > 0 {
        let removed = old
            .len()
            .checked_mul(replacements)
            .context("edit size overflow")?;
        let inserted = new
            .len()
            .checked_mul(replacements)
            .context("edit size overflow")?;
        let updated_len = content
            .len()
            .checked_sub(removed)
            .and_then(|len| len.checked_add(inserted))
            .context("edit size overflow")?;
        if updated_len > MAX_EDITABLE_FILE_BYTES {
            anyhow::bail!(
                "edited file would be {updated_len} bytes; edits are limited to {MAX_EDITABLE_FILE_BYTES} bytes"
            );
        }
    }
    apply_edit(content, old, new, replace_all)
}

struct BoundedSearchOutput {
    text: String,
    byte_limited: bool,
}

impl BoundedSearchOutput {
    fn new() -> Self {
        Self {
            text: String::with_capacity(MAX_OUTPUT_BYTES),
            byte_limited: false,
        }
    }

    fn push_parts(&mut self, parts: &[&str]) -> bool {
        if self.byte_limited {
            return false;
        }
        let separator_len = usize::from(!self.text.is_empty());
        let payload_limit = MAX_OUTPUT_BYTES - SEARCH_OUTPUT_LIMIT_MARKER.len() - 1;
        let available = payload_limit
            .saturating_sub(self.text.len())
            .saturating_sub(separator_len);
        let required = parts
            .iter()
            .fold(0usize, |total, part| total.saturating_add(part.len()));
        if required <= available {
            if separator_len == 1 {
                self.text.push('\n');
            }
            for part in parts {
                self.text.push_str(part);
            }
            return true;
        }

        if available > 0 {
            if separator_len == 1 {
                self.text.push('\n');
            }
            let mut remaining = available;
            for part in parts {
                if remaining == 0 {
                    break;
                }
                let mut end = remaining.min(part.len());
                while !part.is_char_boundary(end) {
                    end -= 1;
                }
                self.text.push_str(&part[..end]);
                remaining -= end;
            }
        }
        if !self.text.is_empty() {
            self.text.push('\n');
        }
        self.text.push_str(SEARCH_OUTPUT_LIMIT_MARKER);
        self.byte_limited = true;
        false
    }

    fn finish(mut self, empty: &str, stopped: Option<SearchStop>, limits: SearchLimits) -> String {
        if self.text.is_empty() {
            self.push_parts(&[empty]);
        }
        if !self.byte_limited {
            if let Some(reason) = stopped {
                let message = format!("… search stopped: {}", reason.message(limits));
                self.push_parts(&[&message]);
            }
        }
        self.text
    }
}

fn grep_files_with_limits(
    pattern: &str,
    root: &Path,
    cancelled: &AtomicBool,
    limits: SearchLimits,
) -> Result<String> {
    let re = regex::Regex::new(pattern).context("invalid regex")?;
    let mut out = BoundedSearchOutput::new();
    let mut matches = 0usize;
    let started = Instant::now();
    let mut visited_files = 0usize;
    let mut stopped = None;

    'walk: for entry in ignore::WalkBuilder::new(root).build().flatten() {
        if let Some(reason) = search_stop(cancelled, started, limits) {
            stopped = Some(reason);
            break;
        }
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if visited_files >= limits.max_visited_files {
            stopped = Some(SearchStop::VisitedFiles);
            break;
        }
        visited_files += 1;
        if entry
            .metadata()
            .map_or(true, |m| m.len() > MAX_SEARCH_FILE_BYTES)
        {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue; // binary or unreadable
        };
        for (no, line) in content.lines().enumerate() {
            if let Some(reason) = search_stop(cancelled, started, limits) {
                stopped = Some(reason);
                break 'walk;
            }
            if re.is_match(line) {
                let path = entry.path().display().to_string();
                let line_number = (no + 1).to_string();
                if !out.push_parts(&[
                    path.as_str(),
                    ":",
                    line_number.as_str(),
                    ":",
                    line.trim_end(),
                ]) {
                    return Ok(out.finish("no matches", None, limits));
                }
                matches += 1;
                if matches >= MAX_SEARCH_RESULTS {
                    let message = format!("… stopped at {MAX_SEARCH_RESULTS} matches");
                    out.push_parts(&[&message]);
                    return Ok(out.finish("no matches", None, limits));
                }
            }
        }
    }
    Ok(out.finish("no matches", stopped, limits))
}

fn glob_files_with_limits(
    pattern: &str,
    root: &Path,
    cancelled: &AtomicBool,
    limits: SearchLimits,
) -> Result<String> {
    let glob = globset::GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .context("invalid glob pattern")?
        .compile_matcher();
    let mut out = Vec::new();
    let started = Instant::now();
    let mut visited_files = 0usize;
    let mut stopped = None;

    for entry in ignore::WalkBuilder::new(root).build().flatten() {
        if let Some(reason) = search_stop(cancelled, started, limits) {
            stopped = Some(reason);
            break;
        }
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if visited_files >= limits.max_visited_files {
            stopped = Some(SearchStop::VisitedFiles);
            break;
        }
        visited_files += 1;
        let relative = entry.path().strip_prefix(root).unwrap_or(entry.path());
        if glob.is_match(relative) || glob.is_match(entry.path()) {
            out.push(entry.path().display().to_string());
            if out.len() >= MAX_SEARCH_RESULTS {
                out.sort();
                out.push(format!("… stopped at {MAX_SEARCH_RESULTS} files"));
                return Ok(out.join("\n"));
            }
        }
    }
    out.sort();
    Ok(finish_search(out, "no files matched", stopped, limits))
}

fn search_stop(
    cancelled: &AtomicBool,
    started: Instant,
    limits: SearchLimits,
) -> Option<SearchStop> {
    if cancelled.load(Ordering::Acquire) {
        Some(SearchStop::Cancelled)
    } else if started.elapsed() >= limits.max_duration {
        Some(SearchStop::Duration)
    } else {
        None
    }
}

fn finish_search(
    lines: Vec<String>,
    empty: &str,
    stopped: Option<SearchStop>,
    limits: SearchLimits,
) -> String {
    let mut output = BoundedSearchOutput::new();
    for line in &lines {
        if !output.push_parts(&[line]) {
            break;
        }
    }
    output.finish(empty, stopped, limits)
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args[key]
        .as_str()
        .with_context(|| format!("missing required argument: {key}"))
}

fn truncate(mut s: String) -> String {
    if s.len() > MAX_OUTPUT_BYTES {
        let mut cut = MAX_OUTPUT_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("\n[output truncated]");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{ApprovalPolicy, SandboxMode, Workspace};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, AtomicOrdering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "shaltaiboltai-tools-{label}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create isolated test directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let is_ours = self
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("shaltaiboltai-tools-"));
            if is_ours {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "t".into(),
            name: name.into(),
            arguments: args,
        }
    }

    fn policy(
        root: &Path,
        sandbox_mode: SandboxMode,
        approval_policy: ApprovalPolicy,
    ) -> ExecutionPolicy {
        ExecutionPolicy::from_parts(
            Workspace::new(root).expect("canonical test workspace"),
            sandbox_mode,
            approval_policy,
        )
    }

    fn default_policy(root: &Path) -> ExecutionPolicy {
        policy(root, SandboxMode::WorkspaceWrite, ApprovalPolicy::OnRequest)
    }

    fn decision(policy: &ExecutionPolicy, name: &str, args: Value) -> ToolDecision {
        assess(policy, &call(name, args)).decision()
    }

    #[test]
    fn run_command_schema_exposes_typed_sandbox_request() {
        let definition = definitions()
            .into_iter()
            .find(|definition| definition.name == "run_command")
            .expect("run_command definition");
        assert_eq!(
            definition.schema["properties"]["sandbox_permissions"]["enum"],
            json!(["use_default", "require_escalated"])
        );
        assert_eq!(
            definition.schema["properties"]["justification"]["type"],
            "string"
        );
    }

    #[test]
    fn sandbox_permission_parsing_is_exact_and_defaults_are_explicit() {
        assert_eq!(
            "use_default".parse::<SandboxPermission>(),
            Ok(SandboxPermission::UseDefault)
        );
        assert_eq!(
            "require_escalated".parse::<SandboxPermission>(),
            Ok(SandboxPermission::RequireEscalated)
        );
        assert!("danger-full-access".parse::<SandboxPermission>().is_err());

        let root = TestDirectory::new("sandbox-parse");
        let policy = default_policy(root.path());
        assert_eq!(
            decision(&policy, "run_command", json!({"command": "pwd"})),
            ToolDecision::Allow
        );
        for arguments in [
            json!({"command": "pwd", "sandbox_permissions": "unknown"}),
            json!({"command": "pwd", "sandbox_permissions": 7}),
            json!({"command": "pwd", "justification": false}),
        ] {
            assert_eq!(
                decision(&policy, "run_command", arguments),
                ToolDecision::Deny
            );
        }
    }

    #[tokio::test]
    async fn read_file_reads_only_a_bounded_utf8_prefix() {
        let root = TestDirectory::new("bounded-read");
        let path = root.path().join("large.txt");
        let content = format!("{}界unreachable-tail", "x".repeat(MAX_READ_FILE_BYTES - 1));
        fs::write(&path, content).unwrap();
        let policy = default_policy(root.path());

        let (output, is_error) = execute(
            &policy,
            &call("read_file", json!({"path": "large.txt"})),
            &ToolAuthorization::Default,
        )
        .await;

        assert!(!is_error, "{output}");
        assert!(output.ends_with("[output truncated]"));
        assert!(!output.contains("unreachable-tail"));
        assert!(output.len() <= MAX_OUTPUT_BYTES + "\n[output truncated]".len());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn read_file_rejects_fifo_without_waiting_for_a_writer() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let root = TestDirectory::new("fifo-read");
        let fifo = root.path().join("input.fifo");
        let fifo_bytes = CString::new(fifo.as_os_str().as_bytes()).expect("fifo path without NUL");
        // SAFETY: `fifo_bytes` is a valid NUL-terminated path owned for the
        // duration of the call, and the fixture path does not exist yet.
        assert_eq!(unsafe { libc::mkfifo(fifo_bytes.as_ptr(), 0o600) }, 0);
        let policy = default_policy(root.path());

        let (output, is_error) = tokio::time::timeout(
            Duration::from_secs(1),
            execute(
                &policy,
                &call("read_file", json!({"path": "input.fifo"})),
                &ToolAuthorization::Default,
            ),
        )
        .await
        .expect("FIFO read must not block");
        assert!(is_error);
        assert!(output.contains("not a regular file"), "{output}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_file_rejects_fifo_without_waiting_for_a_reader() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let root = TestDirectory::new("fifo-write");
        let fifo = root.path().join("output.fifo");
        let fifo_bytes = CString::new(fifo.as_os_str().as_bytes()).expect("fifo path without NUL");
        // SAFETY: `fifo_bytes` is a valid NUL-terminated path owned for the
        // duration of the call, and the fixture path does not exist yet.
        assert_eq!(unsafe { libc::mkfifo(fifo_bytes.as_ptr(), 0o600) }, 0);
        let policy = default_policy(root.path());

        let (output, is_error) = tokio::time::timeout(
            Duration::from_secs(1),
            execute(
                &policy,
                &call(
                    "write_file",
                    json!({"path": "output.fifo", "content": "must not block"}),
                ),
                &ToolAuthorization::Default,
            ),
        )
        .await
        .expect("FIFO write must not block");
        assert!(is_error);
        assert!(output.contains("not a regular file"), "{output}");
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn constrained_writers_reject_hard_links_to_protected_metadata() {
        let root = TestDirectory::new("protected-hard-links");
        for name in [".git", ".agents", ".codex"] {
            fs::create_dir(root.path().join(name)).unwrap();
            fs::write(root.path().join(name).join("config"), "protected\n").unwrap();
        }
        let policy = default_policy(root.path());

        for (index, name) in [".git", ".agents", ".codex"].into_iter().enumerate() {
            let protected = root.path().join(name).join("config");
            let alias_name = format!("innocent-{index}.txt");
            let alias = root.path().join(&alias_name);
            fs::hard_link(&protected, &alias).unwrap();
            let write = call(
                "write_file",
                json!({"path": &alias_name, "content": "overwritten\n"}),
            );
            assert_eq!(assess(&policy, &write).decision(), ToolDecision::Allow);
            let (output, is_error) = execute(&policy, &write, &ToolAuthorization::Default).await;
            assert!(is_error);
            assert!(output.contains("multiple hard links"), "{output}");
            assert_eq!(fs::read_to_string(&protected).unwrap(), "protected\n");

            let edit = call(
                "edit_file",
                json!({"path": &alias_name, "old_string": "protected", "new_string": "changed"}),
            );
            let (output, is_error) = execute(&policy, &edit, &ToolAuthorization::Default).await;
            assert!(is_error);
            assert!(output.contains("multiple hard links"), "{output}");
            assert_eq!(fs::read_to_string(&protected).unwrap(), "protected\n");
        }
    }

    #[tokio::test]
    async fn directory_listing_stops_at_the_entry_limit() {
        let root = TestDirectory::new("bounded-list");
        for name in ["a", "b", "c"] {
            fs::write(root.path().join(name), name).unwrap();
        }

        let output = list_directory_bounded(root.path(), 2).await.unwrap();

        assert_eq!(output.lines().count(), 3);
        assert!(output.ends_with("… stopped at 2 entries"));
    }

    #[test]
    fn recursive_searches_stop_at_file_and_time_budgets() {
        let root = TestDirectory::new("bounded-search");
        fs::write(root.path().join("one.txt"), "needle").unwrap();
        let cancelled = AtomicBool::new(false);

        let normal =
            grep_files_with_limits("needle", root.path(), &cancelled, DEFAULT_SEARCH_LIMITS)
                .unwrap();
        assert!(normal.contains("one.txt:1:needle"));
        assert!(!normal.contains("search stopped"));

        let normal_glob =
            glob_files_with_limits("**/*.txt", root.path(), &cancelled, DEFAULT_SEARCH_LIMITS)
                .unwrap();
        assert!(normal_glob.contains("one.txt"));
        assert!(!normal_glob.contains("search stopped"));

        let file_limited = grep_files_with_limits(
            "needle",
            root.path(),
            &cancelled,
            SearchLimits {
                max_visited_files: 0,
                max_duration: Duration::from_secs(1),
            },
        )
        .unwrap();
        assert!(file_limited.contains("0-file visit limit"));

        let time_limited = glob_files_with_limits(
            "**/*.txt",
            root.path(),
            &cancelled,
            SearchLimits {
                max_visited_files: 10,
                max_duration: Duration::ZERO,
            },
        )
        .unwrap();
        assert!(time_limited.contains("0-second time limit"));
    }

    #[test]
    fn grep_truncates_a_matching_megabyte_line_while_collecting() {
        let root = TestDirectory::new("bounded-grep-line");
        let mut content = String::from("needle:");
        content.push_str(&"x".repeat(MAX_SEARCH_FILE_BYTES as usize - content.len()));
        fs::write(root.path().join("huge.txt"), content).unwrap();
        let cancelled = AtomicBool::new(false);

        let output =
            grep_files_with_limits("needle", root.path(), &cancelled, DEFAULT_SEARCH_LIMITS)
                .unwrap();
        assert!(output.len() <= MAX_OUTPUT_BYTES);
        assert!(output.contains(SEARCH_OUTPUT_LIMIT_MARKER));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_search_future_stops_its_blocking_work() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(run_cancellable_search(move |cancelled| {
            let _ = started_tx.send(());
            let started = Instant::now();
            while !cancelled.load(Ordering::Acquire) && started.elapsed() < Duration::from_secs(2) {
                std::thread::yield_now();
            }
            let stopped_cooperatively = cancelled.load(Ordering::Acquire);
            let _ = stopped_tx.send(stopped_cooperatively);
            Ok("stopped".into())
        }));

        tokio::time::timeout(Duration::from_secs(1), started_rx)
            .await
            .expect("blocking search did not start")
            .expect("start signal dropped");
        task.abort();
        let _ = task.await;
        let stopped_cooperatively = tokio::time::timeout(Duration::from_secs(1), stopped_rx)
            .await
            .expect("blocking search ignored cancellation")
            .expect("stop signal dropped");
        assert!(stopped_cooperatively);
    }

    #[test]
    fn default_policy_allows_in_sandbox_calls_and_asks_at_boundaries() {
        let root = TestDirectory::new("default-matrix");
        let outside = TestDirectory::new("default-matrix-outside");
        fs::create_dir(root.path().join(".git")).unwrap();
        let policy = default_policy(root.path());

        for (name, arguments) in [
            ("read_file", json!({"path": "file"})),
            ("read_file", json!({"path": outside.path().join("secret")})),
            ("list_directory", json!({})),
            ("grep", json!({"pattern": "x"})),
            ("glob", json!({"pattern": "**/*.rs"})),
            ("write_file", json!({"path": "file", "content": "x"})),
            ("run_command", json!({"command": "pwd"})),
        ] {
            assert_eq!(
                decision(&policy, name, arguments),
                ToolDecision::Allow,
                "{name} should stay inside default authority"
            );
        }
        assert_eq!(
            decision(
                &policy,
                "write_file",
                json!({"path": ".git/config", "content": "x"})
            ),
            ToolDecision::Ask
        );
        assert_eq!(
            decision(
                &policy,
                "write_file",
                json!({"path": outside.path().join("file"), "content": "x"})
            ),
            ToolDecision::Ask
        );
        let escalated = assess(
            &policy,
            &call(
                "run_command",
                json!({"command": "pwd", "sandbox_permissions": "require_escalated"}),
            ),
        );
        assert_eq!(escalated.decision(), ToolDecision::Ask);
        assert!(escalated.requires_escalation());
    }

    #[test]
    fn untrusted_asks_for_mutations_and_default_shell_without_escalating() {
        let root = TestDirectory::new("untrusted-matrix");
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        assert_eq!(
            decision(&policy, "read_file", json!({"path": "file"})),
            ToolDecision::Allow
        );
        assert_eq!(
            decision(
                &policy,
                "write_file",
                json!({"path": "file", "content": "x"})
            ),
            ToolDecision::Ask
        );
        let command = assess(&policy, &call("run_command", json!({"command": "pwd"})));
        assert_eq!(command.decision(), ToolDecision::Ask);
        assert!(!command.requires_escalation());
    }

    #[test]
    fn never_denies_boundary_crossings_without_prompting() {
        let root = TestDirectory::new("never-matrix");
        let outside = TestDirectory::new("never-matrix-outside");
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Never,
        );
        assert_eq!(
            decision(
                &policy,
                "write_file",
                json!({"path": "inside", "content": "x"})
            ),
            ToolDecision::Allow
        );
        for arguments in [
            json!({"path": ".git/config", "content": "x"}),
            json!({"path": outside.path().join("file"), "content": "x"}),
        ] {
            assert_eq!(
                decision(&policy, "write_file", arguments),
                ToolDecision::Deny
            );
        }
        assert_eq!(
            decision(
                &policy,
                "run_command",
                json!({"command": "pwd", "sandbox_permissions": "require_escalated"})
            ),
            ToolDecision::Deny
        );
        assert_eq!(
            decision(&policy, "run_command", json!({"command": "pwd"})),
            ToolDecision::Allow
        );
    }

    #[test]
    fn read_only_and_full_access_have_distinct_authority() {
        let root = TestDirectory::new("mode-matrix");
        let read_only = policy(
            root.path(),
            SandboxMode::ReadOnly,
            ApprovalPolicy::OnRequest,
        );
        assert_eq!(
            decision(
                &read_only,
                "write_file",
                json!({"path": "file", "content": "x"})
            ),
            ToolDecision::Ask
        );
        assert_eq!(
            decision(&read_only, "run_command", json!({"command": "pwd"})),
            ToolDecision::Allow
        );

        let full_access = policy(
            root.path(),
            SandboxMode::DangerFullAccess,
            ApprovalPolicy::Never,
        );
        assert_eq!(
            decision(
                &full_access,
                "write_file",
                json!({"path": "/tmp/full-access-target", "content": "x"})
            ),
            ToolDecision::Allow
        );
        assert_eq!(
            decision(
                &full_access,
                "run_command",
                json!({"command": "pwd", "sandbox_permissions": "require_escalated"})
            ),
            ToolDecision::Allow
        );
    }

    #[test]
    fn malformed_calls_fail_closed_before_execution() {
        let root = TestDirectory::new("malformed");
        let policy = default_policy(root.path());
        for tool in [
            call("read_file", json!({})),
            call("write_file", json!({"path": "x"})),
            call("edit_file", json!({"path": "x", "old_string": "a"})),
            call("list_directory", json!({"path": 42})),
            call("grep", json!({"pattern": "["})),
            call("glob", json!({"pattern": "["})),
            call("run_command", json!({})),
            call("unknown", json!({})),
        ] {
            assert_eq!(assess(&policy, &tool).decision(), ToolDecision::Deny);
        }
    }

    #[test]
    fn approval_scopes_bind_policy_canonical_target_search_and_command() {
        let root = TestDirectory::new("scopes");
        let mut policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        let first_path = call("write_file", json!({"path": "one.txt", "content": "same"}));
        let second_path = call("write_file", json!({"path": "two.txt", "content": "same"}));
        let same_path_new_content = call(
            "write_file",
            json!({"path": "one.txt", "content": "updated"}),
        );
        assert_ne!(
            assess(&policy, &first_path).scope(),
            assess(&policy, &second_path).scope()
        );
        assert_eq!(
            assess(&policy, &first_path).scope(),
            assess(&policy, &same_path_new_content).scope()
        );

        let first_search = call("grep", json!({"path": ".", "pattern": "one"}));
        let second_search = call("grep", json!({"path": ".", "pattern": "two"}));
        assert_ne!(
            assess(&policy, &first_search).scope(),
            assess(&policy, &second_search).scope()
        );
        let first_command = call("run_command", json!({"command": "cargo test"}));
        let second_command = call("run_command", json!({"command": "cargo publish"}));
        assert_ne!(
            assess(&policy, &first_command).scope(),
            assess(&policy, &second_command).scope()
        );
        let prior_scope = assess(&policy, &first_path).scope().to_owned();
        assert!(policy.update(SandboxMode::ReadOnly, ApprovalPolicy::Untrusted));
        assert_ne!(prior_scope, assess(&policy, &first_path).scope());
    }

    #[tokio::test]
    async fn explicit_policy_cwd_controls_relative_file_io() {
        let root = TestDirectory::new("explicit-cwd");
        let policy = default_policy(root.path());
        let tool = call(
            "write_file",
            json!({"path": "nested/result.txt", "content": "inside"}),
        );
        let assessment = assess(&policy, &tool);
        assert_eq!(assessment.decision(), ToolDecision::Allow);
        assert_eq!(
            assessment.canonical_target(),
            Some(policy.workspace().cwd().join("nested/result.txt").as_path())
        );
        let (output, is_error) = execute(&policy, &tool, &ToolAuthorization::Default).await;
        assert!(!is_error, "{output}");
        assert_eq!(
            fs::read_to_string(root.path().join("nested/result.txt")).unwrap(),
            "inside"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn approved_scope_cannot_follow_a_retargeted_symlink() {
        use std::os::unix::fs::symlink;

        let root = TestDirectory::new("retarget-workspace");
        let inside = root.path().join("inside");
        let outside = TestDirectory::new("retarget-outside");
        fs::create_dir(&inside).unwrap();
        let link = root.path().join("target");
        symlink(&inside, &link).unwrap();
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        let tool = call(
            "write_file",
            json!({"path": "target/result", "content": "secret"}),
        );
        let original = assess(&policy, &tool);
        assert_eq!(original.decision(), ToolDecision::Ask);
        let authorization = ToolAuthorization::approved(&policy, &original);

        fs::remove_file(&link).unwrap();
        symlink(outside.path(), &link).unwrap();
        let retargeted = assess(&policy, &tool);
        assert_ne!(original.scope(), retargeted.scope());
        let (output, is_error) = execute(&policy, &tool, &authorization).await;
        assert!(is_error);
        assert!(output.contains("stale or does not match"), "{output}");
        assert!(!outside.path().join("result").exists());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn approval_scopes_preserve_non_utf8_path_identity() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::fs::symlink;

        let root = TestDirectory::new("non-utf8-scope-workspace");
        let targets = TestDirectory::new("non-utf8-scope-targets");
        let first = targets
            .path()
            .join(OsString::from_vec(b"outside-\x80".to_vec()));
        let second = targets
            .path()
            .join(OsString::from_vec(b"outside-\x81".to_vec()));
        fs::create_dir(&first).unwrap();
        fs::create_dir(&second).unwrap();
        let link = root.path().join("target");
        symlink(&first, &link).unwrap();
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        let tool = call(
            "write_file",
            json!({"path": "target/result", "content": "secret"}),
        );
        let original = assess(&policy, &tool);
        let authorization = ToolAuthorization::approved(&policy, &original);

        fs::remove_file(&link).unwrap();
        symlink(&second, &link).unwrap();
        let retargeted = assess(&policy, &tool);
        assert_eq!(
            original.canonical_target().unwrap().to_string_lossy(),
            retargeted.canonical_target().unwrap().to_string_lossy(),
            "the regression requires paths that collide when rendered lossily"
        );
        assert_ne!(original.scope(), retargeted.scope());

        let (output, is_error) = execute(&policy, &tool, &authorization).await;
        assert!(is_error);
        assert!(output.contains("stale or does not match"), "{output}");
        assert!(!first.join("result").exists());
        assert!(!second.join("result").exists());
    }

    #[cfg(unix)]
    #[test]
    fn scope_encoding_does_not_collapse_lossy_non_utf8_paths() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let first = PathBuf::from(OsString::from_vec(b"outside-\x80".to_vec()));
        let second = PathBuf::from(OsString::from_vec(b"outside-\x81".to_vec()));
        assert_eq!(first.to_string_lossy(), second.to_string_lossy());
        assert_ne!(
            encoded_scope([first.as_os_str().as_encoded_bytes()]),
            encoded_scope([second.as_os_str().as_encoded_bytes()])
        );
    }

    #[test]
    fn apply_edit_enforces_unique_match() {
        assert_eq!(apply_edit("a b a", "b", "c", false).unwrap(), "a c a");
        assert!(apply_edit("a b a", "a", "c", false).is_err());
        assert_eq!(apply_edit("a b a", "a", "c", true).unwrap(), "c b c");
        assert!(apply_edit("a b a", "z", "c", false).is_err());
    }

    #[tokio::test]
    async fn exact_approval_executes_but_never_policy_cannot_be_bypassed() {
        let root = TestDirectory::new("approved-write-workspace");
        let outside = TestDirectory::new("approved-write-outside");
        let default = default_policy(root.path());
        let target = outside.path().join("result");
        let tool = call(
            "write_file",
            json!({"path": &target, "content": "approved"}),
        );
        let assessment = assess(&default, &tool);
        assert_eq!(assessment.decision(), ToolDecision::Ask);

        let (output, is_error) = execute(&default, &tool, &ToolAuthorization::Default).await;
        assert!(is_error);
        assert!(output.contains("requires approval"));
        assert!(!target.exists());

        let authorization = ToolAuthorization::approved(&default, &assessment);
        let (output, is_error) = execute(&default, &tool, &authorization).await;
        assert!(!is_error, "{output}");
        assert_eq!(fs::read_to_string(&target).unwrap(), "approved");

        fs::remove_file(&target).unwrap();
        let never = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Never,
        );
        let denied = assess(&never, &tool);
        assert_eq!(denied.decision(), ToolDecision::Deny);
        let forged = ToolAuthorization::approved(&never, &denied);
        let (output, is_error) = execute(&never, &tool, &forged).await;
        assert!(is_error);
        assert!(output.contains("denied by policy"), "{output}");
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn policy_change_invalidates_previously_bound_approval() {
        let root = TestDirectory::new("stale-policy");
        let mut policy = default_policy(root.path());
        let tool = call(
            "write_file",
            json!({"path": ".git/config", "content": "unsafe"}),
        );
        let assessment = assess(&policy, &tool);
        let authorization = ToolAuthorization::approved(&policy, &assessment);
        assert!(policy.update(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest));
        let (output, is_error) = execute(&policy, &tool, &authorization).await;
        assert!(is_error);
        assert!(output.contains("stale or does not match"), "{output}");
        assert!(!root.path().join(".git/config").exists());
    }

    #[tokio::test]
    async fn edit_file_round_trip_and_preview_use_the_policy_target() {
        let root = TestDirectory::new("edit");
        let path = root.path().join("file.txt");
        fs::write(&path, "hello world\n").unwrap();
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        let tool = call(
            "edit_file",
            json!({"path": "file.txt", "old_string": "world", "new_string": "rust"}),
        );
        let assessment = assess(&policy, &tool);
        let preview = approval_preview(&assessment, &tool).expect("canonical preview");
        assert!(preview
            .iter()
            .any(|(tag, line)| *tag == '-' && line == "hello world"));
        assert!(preview
            .iter()
            .any(|(tag, line)| *tag == '+' && line == "hello rust"));
        let assessment = assess(&policy, &tool);
        let authorization = ToolAuthorization::approved(&policy, &assessment);
        let (out, is_error) = execute(&policy, &tool, &authorization).await;
        assert!(!is_error, "{out}");
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello rust\n");
    }

    #[tokio::test]
    async fn edit_and_approval_preview_reject_large_or_binary_files() {
        let root = TestDirectory::new("bounded-change-preview");
        let large = root.path().join("large.txt");
        let binary = root.path().join("binary.dat");
        fs::write(&large, vec![b'x'; MAX_EDITABLE_FILE_BYTES + 1]).unwrap();
        fs::write(&binary, [0xff, 0xfe, 0xfd]).unwrap();
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );

        for path in ["large.txt", "binary.dat"] {
            let write = call(
                "write_file",
                json!({"path": path, "content": "replacement"}),
            );
            let assessment = assess(&policy, &write);
            let preview = approval_preview(&assessment, &write).expect("write preview warning");
            assert_eq!(preview[0].0, '!');
            assert!(preview[0].1.contains("preview unavailable"), "{preview:?}");

            let edit = call(
                "edit_file",
                json!({"path": path, "old_string": "x", "new_string": "y"}),
            );
            let assessment = assess(&policy, &edit);
            let authorization = ToolAuthorization::approved(&policy, &assessment);
            let (output, is_error) = execute(&policy, &edit, &authorization).await;
            assert!(is_error);
            assert!(
                output.contains("editable files are limited") || output.contains("not UTF-8 text"),
                "{output}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn approval_preview_rejects_special_files_without_opening_them() {
        use std::os::unix::net::UnixListener;

        let root = TestDirectory::new("special-change-preview");
        let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, AtomicOrdering::Relaxed);
        let socket_path = PathBuf::from("/tmp")
            .join(format!("sb-preview-{}-{sequence}.sock", std::process::id()));
        let listener = UnixListener::bind(&socket_path).expect("bind unix socket");
        let policy = policy(
            root.path(),
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        let write = call(
            "write_file",
            json!({"path": &socket_path, "content": "replacement"}),
        );
        let assessment = assess(&policy, &write);
        let preview = approval_preview(&assessment, &write).expect("special-file warning");
        assert_eq!(preview[0].0, '!');
        assert!(preview[0].1.contains("not a regular file"), "{preview:?}");
        drop(listener);
        fs::remove_file(socket_path).expect("remove unix socket");
    }

    #[tokio::test]
    async fn danger_full_access_command_uses_explicit_workspace_cwd() {
        let root = TestDirectory::new("command-cwd");
        let policy = policy(
            root.path(),
            SandboxMode::DangerFullAccess,
            ApprovalPolicy::Never,
        );
        let (output, is_error) = execute(
            &policy,
            &call("run_command", json!({"command": "pwd"})),
            &ToolAuthorization::Default,
        )
        .await;
        assert!(!is_error, "{output}");
        assert_eq!(output.trim(), policy.workspace().cwd().to_string_lossy());
    }

    #[tokio::test]
    async fn command_streaming_is_memory_bounded_and_drained() {
        let mut command = tokio::process::Command::new("/bin/sh");
        command.args([
            "-c",
            "i=0; while [ \"$i\" -lt 40000 ]; do printf x; i=$((i + 1)); done",
        ]);
        let output = run_command_bounded(command).await.unwrap();
        assert!(output.ends_with("[output truncated]"));
        assert!(output.len() <= MAX_OUTPUT_BYTES + "\n[output truncated]".len());
    }

    #[tokio::test]
    async fn commands_receive_closed_stdin() {
        let mut command = tokio::process::Command::new("/bin/sh");
        command.args([
            "-c",
            "if IFS= read -r value; then printf 'unexpected:%s' \"$value\"; else printf eof; fi",
        ]);
        let output = run_command_bounded(command).await.unwrap();
        assert_eq!(output, "eof");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn command_completion_terminates_background_descendants() {
        let root = TestDirectory::new("command-process-group");
        let marker = root.path().join("leaked");
        let mut command = tokio::process::Command::new("/bin/sh");
        command.env("SHALTAIBOLTAI_TEST_MARKER", &marker).args([
            "-c",
            "(sleep 1; printf leaked > \"$SHALTAIBOLTAI_TEST_MARKER\") >/dev/null 2>&1 & printf done",
        ]);

        let output = run_command_bounded(command).await.unwrap();
        assert_eq!(output, "done");
        tokio::time::sleep(Duration::from_millis(1_200)).await;
        assert!(!marker.exists(), "background descendant escaped cleanup");
    }

    #[test]
    fn diff_preview_shows_changes() {
        let lines = diff_lines("a\nb\nc\n", "a\nB\nc\n");
        assert!(lines.iter().any(|(t, l)| *t == '-' && l == "b"));
        assert!(lines.iter().any(|(t, l)| *t == '+' && l == "B"));
    }

    #[test]
    fn adversarial_diff_preview_obeys_a_short_deadline() {
        let old = (0..MAX_DIFF_INPUT_LINES)
            .map(|index| format!("old-{index}\n"))
            .collect::<String>();
        let new = (0..MAX_DIFF_INPUT_LINES)
            .map(|index| format!("new-{index}\n"))
            .collect::<String>();
        let started = Instant::now();
        let lines = diff_lines_bounded(&old, &new).expect("bounded adversarial diff");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(!lines.is_empty());
        assert!(lines.len() <= MAX_DIFF_PREVIEW_LINES + 1);
    }
}
