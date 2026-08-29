//! OS-enforced shell sandbox preparation.
//!
//! Policy and approvals are deliberately separate concerns: this module only
//! turns an already selected [`ExecutionPolicy`] into a process boundary. A
//! constrained policy never degrades to an unsandboxed shell when its backend
//! is unavailable.

use crate::policy::{protected_metadata_paths_for_roots, ExecutionPolicy, SandboxMode};
#[cfg(target_os = "linux")]
use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

const SHELL: &str = "/bin/sh";
const MACOS_SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
#[cfg(target_os = "linux")]
const LINUX_BWRAP_CANDIDATES: [&str; 2] = ["/usr/bin/bwrap", "/bin/bwrap"];
const PROTECTED_METADATA_NAMES: [&str; 3] = [".git", ".agents", ".codex"];
const SEATBELT_BASE_POLICY: &str = include_str!("../assets/sandbox/seatbelt_base.sbpl");

/// Marker inherited by descendants so diagnostics can identify the boundary.
pub const SANDBOX_ENV_VAR: &str = "SHALTAIBOLTAI_SANDBOX";
pub const LINUX_SECCOMP_EXEC_OPTION: &str = "--__sandbox-seccomp-exec";

/// Trusted process boundary used by a prepared command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxBackend {
    MacosSeatbelt,
    LinuxBubblewrap { executable: BubblewrapExecutable },
    Unsandboxed,
}

impl SandboxBackend {
    pub const fn marker(self) -> &'static str {
        match self {
            Self::MacosSeatbelt => "seatbelt",
            Self::LinuxBubblewrap { .. } => "bubblewrap",
            Self::Unsandboxed => "unsandboxed",
        }
    }

    /// Detect a trusted, absolute backend for the current target.
    ///
    /// No `PATH` lookup is performed. This matters because commands may run in
    /// a workspace whose contents are controlled by the model.
    pub fn detect() -> Result<Self, SandboxError> {
        #[cfg(target_os = "macos")]
        {
            require_executable(Path::new(MACOS_SANDBOX_EXEC))?;
            Ok(Self::MacosSeatbelt)
        }

        #[cfg(target_os = "linux")]
        {
            for executable in BubblewrapExecutable::ALL {
                if is_executable(executable.path()) {
                    return Ok(Self::LinuxBubblewrap { executable });
                }
            }
            Err(SandboxError::BackendUnavailable {
                backend: "bubblewrap",
                searched: LINUX_BWRAP_CANDIDATES.to_vec(),
            })
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Err(SandboxError::UnsupportedPlatform(
                std::env::consts::OS.to_owned(),
            ))
        }
    }
}

/// Trusted absolute Bubblewrap locations, in production preference order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BubblewrapExecutable {
    UsrBin,
    Bin,
}

impl BubblewrapExecutable {
    #[cfg(target_os = "linux")]
    const ALL: [Self; 2] = [Self::UsrBin, Self::Bin];

    pub fn path(self) -> &'static Path {
        match self {
            Self::UsrBin => Path::new("/usr/bin/bwrap"),
            Self::Bin => Path::new("/bin/bwrap"),
        }
    }
}

/// A deterministic process description. It is intentionally inspectable so
/// callers and tests can audit every argument before spawning it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandSpec {
    program: PathBuf,
    args: Vec<OsString>,
    cwd: PathBuf,
    env: Vec<(OsString, OsString)>,
    backend: SandboxBackend,
    synthetic_mount_targets: Vec<PathBuf>,
}

impl CommandSpec {
    pub fn program(&self) -> &Path {
        &self.program
    }

    pub fn args(&self) -> &[OsString] {
        &self.args
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn env(&self) -> &[(OsString, OsString)] {
        &self.env
    }

    pub const fn backend(&self) -> SandboxBackend {
        self.backend
    }

    pub fn synthetic_mount_targets(&self) -> &[PathBuf] {
        &self.synthetic_mount_targets
    }

    /// Convert the audited description to Tokio without changing its program,
    /// arguments, working directory, or environment additions.
    pub fn into_tokio_command(self) -> tokio::process::Command {
        let mut command = tokio::process::Command::new(self.program);
        command.args(self.args);
        command.current_dir(self.cwd);
        command.envs(self.env);
        command.kill_on_drop(true);
        command
    }
}

/// An executable shell command plus any short-lived host mount targets needed
/// to express a missing read-only path to Bubblewrap.
pub struct PreparedShellCommand {
    spec: CommandSpec,
    cleanup: ShellCleanupGuard,
}

impl PreparedShellCommand {
    /// Keep the cleanup guard alive until the spawned process has terminated.
    pub fn into_tokio_parts(self) -> (tokio::process::Command, ShellCleanupGuard) {
        (self.spec.into_tokio_command(), self.cleanup)
    }
}

/// Identity-bound cleanup for synthetic Bubblewrap mount targets.
///
/// The guard removes only empty directories that this process created and
/// whose device/inode identity has not changed. It is safe to drop on spawn
/// failure, timeout, cancellation, or ordinary completion.
pub struct ShellCleanupGuard {
    #[cfg(target_os = "linux")]
    targets: Vec<SyntheticMountTarget>,
}

impl ShellCleanupGuard {
    fn empty() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            targets: Vec::new(),
        }
    }

    /// Perform cleanup now and surface evidence of concurrent replacement.
    pub fn cleanup(self) -> Result<(), SandboxError> {
        #[cfg(target_os = "linux")]
        {
            let mut guard = self;
            let targets = std::mem::take(&mut guard.targets);
            cleanup_synthetic_mount_targets(&targets)
        }

        #[cfg(not(target_os = "linux"))]
        Ok(())
    }
}

impl Drop for ShellCleanupGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            let targets = std::mem::take(&mut self.targets);
            let _ = cleanup_synthetic_mount_targets(&targets);
        }
    }
}

#[cfg(target_os = "linux")]
struct SyntheticMountTarget {
    path: PathBuf,
    device: u64,
    inode: u64,
}

/// Prepare a shell command using the current platform's trusted backend.
///
/// `approved_escalation` must be `true` only after the caller has obtained the
/// approval required by the active policy. An approved escalation is explicit
/// authority to cross the sandbox boundary; it is never inferred here.
pub fn prepare_shell_command(
    policy: &ExecutionPolicy,
    command: &str,
    approved_escalation: bool,
) -> Result<PreparedShellCommand, SandboxError> {
    if approved_escalation || policy.sandbox_mode() == SandboxMode::DangerFullAccess {
        let spec = prepare_shell_command_for_backend(
            policy,
            command,
            approved_escalation,
            SandboxBackend::Unsandboxed,
        )?;
        return Ok(PreparedShellCommand {
            spec,
            cleanup: ShellCleanupGuard::empty(),
        });
    }

    let backend = SandboxBackend::detect()?;
    let spec = prepare_shell_command_for_backend(policy, command, approved_escalation, backend)?;
    let cleanup = materialize_synthetic_mount_targets(&spec)?;
    Ok(PreparedShellCommand { spec, cleanup })
}

/// Prepare the production Linux boundary with an explicitly supplied copy of
/// this package's application binary.
///
/// Cargo integration tests run from their own harness executable, so
/// `current_exe()` cannot exercise the application's hidden seccomp child
/// stage. This debug-only entry point lets the integration suite supply
/// `CARGO_BIN_EXE_shaltaiboltai` while retaining the production Bubblewrap
/// detection, mount materialization, and cleanup path.
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub fn prepare_shell_command_for_linux_integration_test(
    policy: &ExecutionPolicy,
    command: &str,
    application_binary: &Path,
) -> Result<PreparedShellCommand, SandboxError> {
    if policy.sandbox_mode() == SandboxMode::DangerFullAccess {
        return Err(SandboxError::UntrustedSandboxHelper(
            application_binary.to_path_buf(),
        ));
    }
    if !application_binary.is_absolute() || !is_executable(application_binary) {
        return Err(SandboxError::UntrustedSandboxHelper(
            application_binary.to_path_buf(),
        ));
    }
    let backend = SandboxBackend::detect()?;
    let SandboxBackend::LinuxBubblewrap { executable } = backend else {
        return Err(SandboxError::UnsupportedPlatform(
            std::env::consts::OS.to_owned(),
        ));
    };
    let spec = bubblewrap_spec_with_executable_and_helper(
        policy,
        command,
        policy.workspace().cwd().to_path_buf(),
        executable.path(),
        executable,
        application_binary,
    )?;
    let cleanup = materialize_synthetic_mount_targets(&spec)?;
    Ok(PreparedShellCommand { spec, cleanup })
}

/// Pure command construction for auditing and tests.
///
/// Passing [`SandboxBackend::Unsandboxed`] cannot bypass a constrained policy;
/// an explicit approved escalation is still required.
pub fn prepare_shell_command_for_backend(
    policy: &ExecutionPolicy,
    command: &str,
    approved_escalation: bool,
    backend: SandboxBackend,
) -> Result<CommandSpec, SandboxError> {
    let cwd = policy.workspace().cwd().to_path_buf();
    if approved_escalation || policy.sandbox_mode() == SandboxMode::DangerFullAccess {
        return Ok(raw_shell_spec(
            command,
            cwd,
            if approved_escalation {
                "escalated"
            } else {
                "danger-full-access"
            },
        ));
    }

    match backend {
        SandboxBackend::MacosSeatbelt => seatbelt_spec(policy, command, cwd),
        SandboxBackend::LinuxBubblewrap { executable } => {
            bubblewrap_spec(policy, command, cwd, executable)
        }
        SandboxBackend::Unsandboxed => Err(SandboxError::UnsandboxedConstrainedPolicy),
    }
}

fn raw_shell_spec(command: &str, cwd: PathBuf, marker: &str) -> CommandSpec {
    CommandSpec {
        program: PathBuf::from(SHELL),
        args: vec![OsString::from("-c"), OsString::from(command)],
        cwd,
        env: marker_env(marker),
        backend: SandboxBackend::Unsandboxed,
        synthetic_mount_targets: Vec::new(),
    }
}

fn seatbelt_spec(
    policy: &ExecutionPolicy,
    command: &str,
    cwd: PathBuf,
) -> Result<CommandSpec, SandboxError> {
    let (profile, definitions) = seatbelt_profile(policy)?;
    let mut args = vec![OsString::from("-p"), OsString::from(profile)];
    args.extend(
        definitions
            .iter()
            .map(|(name, path)| seatbelt_definition_arg(name, path)),
    );
    args.extend([
        OsString::from("--"),
        OsString::from(SHELL),
        OsString::from("-c"),
        OsString::from(command),
    ]);
    Ok(CommandSpec {
        program: PathBuf::from(MACOS_SANDBOX_EXEC),
        args,
        cwd,
        env: marker_env(SandboxBackend::MacosSeatbelt.marker()),
        backend: SandboxBackend::MacosSeatbelt,
        synthetic_mount_targets: Vec::new(),
    })
}

fn seatbelt_profile(
    policy: &ExecutionPolicy,
) -> Result<(String, Vec<(String, PathBuf)>), SandboxError> {
    let mut profile = String::with_capacity(SEATBELT_BASE_POLICY.len() + 2_048);
    profile.push_str(SEATBELT_BASE_POLICY);
    profile.push_str("\n; Legacy Codex-compatible constrained modes can read the full disk.\n");
    profile.push_str("(allow file-read*)\n");
    let mut definitions = Vec::new();

    if policy.sandbox_mode() == SandboxMode::WorkspaceWrite {
        let writable_roots = writable_roots(policy)?;
        profile.push_str("; Writes are limited to the workspace and temporary roots.\n");
        profile.push_str("(allow file-write*\n");
        for (index, root) in writable_roots.iter().enumerate() {
            let name = format!("WRITABLE_ROOT_{index}");
            profile.push_str(&format!("  (subpath (param \"{name}\"))\n"));
            definitions.push((name, root.clone()));
        }
        profile.push_str(")\n");

        // Denies are global so overlapping roots (for example, a workspace in
        // TMPDIR) cannot reopen protected metadata through a broader allow.
        for (index, protected) in seatbelt_protected_metadata_paths(&writable_roots)
            .into_iter()
            .enumerate()
        {
            let name = format!("PROTECTED_METADATA_{index}");
            profile.push_str(&format!(
                "(deny file-write* (literal (param \"{name}\")) (subpath (param \"{name}\")))\n"
            ));
            definitions.push((name, protected));
        }

        // Moving a writable root would relocate its protected descendants past
        // the pathname rules. Keep the authority anchors themselves immutable.
        for (index, root) in policy.effective_user_visible_roots().iter().enumerate() {
            let name = format!("USER_ROOT_{index}");
            profile.push_str(&format!(
                "(deny file-write-unlink (require-all (literal (param \"{name}\")) (vnode-type DIRECTORY)))\n"
            ));
            definitions.push((name, root.clone()));
        }
    }

    Ok((profile, definitions))
}

/// Seatbelt path filters preserve caller spelling even on a case-insensitive
/// volume. Enumerate the finite ASCII casings of the reserved top-level names
/// so `.GIT`, `.Agents`, and similar aliases cannot pass a lowercase literal
/// deny. Linked-worktree and symlink targets still come from the shared policy
/// expansion.
fn seatbelt_protected_metadata_paths(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = protected_metadata_paths_for_roots(roots);
    for root in roots {
        for protected in PROTECTED_METADATA_NAMES {
            for alias in ascii_case_variants(protected) {
                let path = root.join(alias);
                if !paths.contains(&path) {
                    paths.push(path);
                }
            }
        }
    }
    paths
}

fn ascii_case_variants(value: &str) -> Vec<String> {
    let mut variants = vec![String::new()];
    for character in value.chars() {
        if character.is_ascii_alphabetic() {
            let mut expanded = Vec::with_capacity(variants.len() * 2);
            for prefix in variants {
                let mut lowercase = prefix.clone();
                lowercase.push(character.to_ascii_lowercase());
                expanded.push(lowercase);

                let mut uppercase = prefix;
                uppercase.push(character.to_ascii_uppercase());
                expanded.push(uppercase);
            }
            variants = expanded;
        } else {
            for prefix in &mut variants {
                prefix.push(character);
            }
        }
    }
    variants
}

fn seatbelt_definition_arg(name: &str, value: &Path) -> OsString {
    let mut argument = OsString::from(format!("-D{name}="));
    argument.push(value.as_os_str());
    argument
}

fn bubblewrap_spec(
    policy: &ExecutionPolicy,
    command: &str,
    cwd: PathBuf,
    executable: BubblewrapExecutable,
) -> Result<CommandSpec, SandboxError> {
    bubblewrap_spec_with_executable(policy, command, cwd, executable.path(), executable)
}

fn bubblewrap_spec_with_executable(
    policy: &ExecutionPolicy,
    command: &str,
    cwd: PathBuf,
    executable: &Path,
    trusted_executable: BubblewrapExecutable,
) -> Result<CommandSpec, SandboxError> {
    let inner_executable = std::env::current_exe().map_err(SandboxError::CurrentExecutable)?;
    bubblewrap_spec_with_executable_and_helper(
        policy,
        command,
        cwd,
        executable,
        trusted_executable,
        &inner_executable,
    )
}

fn bubblewrap_spec_with_executable_and_helper(
    policy: &ExecutionPolicy,
    command: &str,
    cwd: PathBuf,
    executable: &Path,
    trusted_executable: BubblewrapExecutable,
    inner_executable: &Path,
) -> Result<CommandSpec, SandboxError> {
    let mut args = os_args(&[
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]);

    if policy.sandbox_mode() == SandboxMode::WorkspaceWrite {
        let writable_roots = writable_roots(policy)?;
        ensure_helper_is_not_writable(inner_executable, &writable_roots)?;
        for root in &writable_roots {
            push_path_option(&mut args, "--bind", root, root);
        }

        // Re-bind protected metadata after every writable root. Missing names
        // use a temporary empty mount target so ordinary repositories do not
        // fail merely because `.agents` or `.codex` has never been created.
        // Symlinked boundaries remain unenforceable and fail closed.
        let mut synthetic_mount_targets = Vec::new();
        for protected in protected_metadata_paths_for_roots(&writable_roots) {
            match std::fs::symlink_metadata(&protected) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(SandboxError::UnenforceableProtectedPath(protected));
                }
                Ok(_) => push_path_option(&mut args, "--ro-bind", &protected, &protected),
                Err(source) if source.kind() == io::ErrorKind::NotFound => {
                    args.push(OsString::from("--perms"));
                    args.push(OsString::from("555"));
                    args.push(OsString::from("--tmpfs"));
                    args.push(protected.as_os_str().to_os_string());
                    args.push(OsString::from("--remount-ro"));
                    args.push(protected.as_os_str().to_os_string());
                    synthetic_mount_targets.push(protected);
                }
                Err(source) => {
                    return Err(SandboxError::PathInspection {
                        path: protected,
                        source,
                    });
                }
            }
        }

        args.push(OsString::from("--chdir"));
        args.push(cwd.as_os_str().to_os_string());
        args.push(OsString::from("--setenv"));
        args.push(OsString::from(SANDBOX_ENV_VAR));
        let backend = SandboxBackend::LinuxBubblewrap {
            executable: trusted_executable,
        };
        args.push(OsString::from(backend.marker()));
        args.extend([
            OsString::from("--"),
            inner_executable.as_os_str().to_os_string(),
            {
                let mut argument = OsString::from(LINUX_SECCOMP_EXEC_OPTION);
                argument.push("=");
                argument.push(command);
                argument
            },
        ]);

        return Ok(CommandSpec {
            program: executable.to_path_buf(),
            args,
            cwd,
            env: marker_env(backend.marker()),
            backend,
            synthetic_mount_targets,
        });
    }

    args.push(OsString::from("--chdir"));
    args.push(cwd.as_os_str().to_os_string());
    args.push(OsString::from("--setenv"));
    args.push(OsString::from(SANDBOX_ENV_VAR));
    let backend = SandboxBackend::LinuxBubblewrap {
        executable: trusted_executable,
    };
    args.push(OsString::from(backend.marker()));
    args.extend([
        OsString::from("--"),
        inner_executable.as_os_str().to_os_string(),
        {
            let mut argument = OsString::from(LINUX_SECCOMP_EXEC_OPTION);
            argument.push("=");
            argument.push(command);
            argument
        },
    ]);

    Ok(CommandSpec {
        program: executable.to_path_buf(),
        args,
        cwd,
        env: marker_env(backend.marker()),
        backend,
        synthetic_mount_targets: Vec::new(),
    })
}

fn ensure_helper_is_not_writable(
    helper: &Path,
    writable_roots: &[PathBuf],
) -> Result<(), SandboxError> {
    if !helper.is_absolute() || !is_executable(helper) {
        return Err(SandboxError::UntrustedSandboxHelper(helper.to_path_buf()));
    }
    let canonical = std::fs::canonicalize(helper)
        .map_err(|_| SandboxError::UntrustedSandboxHelper(helper.to_path_buf()))?;
    if writable_roots
        .iter()
        .any(|root| helper.starts_with(root) || canonical.starts_with(root))
    {
        return Err(SandboxError::UntrustedSandboxHelper(helper.to_path_buf()));
    }
    Ok(())
}

fn materialize_synthetic_mount_targets(
    spec: &CommandSpec,
) -> Result<ShellCleanupGuard, SandboxError> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};

        let mut guard = ShellCleanupGuard::empty();
        for path in spec.synthetic_mount_targets() {
            let mut builder = std::fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(path) {
                Ok(()) => {
                    let metadata = match std::fs::symlink_metadata(path) {
                        Ok(metadata) => metadata,
                        Err(source) => {
                            let _ = std::fs::remove_dir(path);
                            return Err(SandboxError::PathInspection {
                                path: path.clone(),
                                source,
                            });
                        }
                    };
                    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
                        let _ = std::fs::remove_dir(path);
                        return Err(SandboxError::UnenforceableProtectedPath(path.clone()));
                    }
                    guard.targets.push(SyntheticMountTarget {
                        path: path.clone(),
                        device: metadata.dev(),
                        inode: metadata.ino(),
                    });
                }
                Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                    let metadata = std::fs::symlink_metadata(path).map_err(|source| {
                        SandboxError::PathInspection {
                            path: path.clone(),
                            source,
                        }
                    })?;
                    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
                        return Err(SandboxError::UnenforceableProtectedPath(path.clone()));
                    }
                }
                Err(source) => {
                    return Err(SandboxError::SyntheticMountCreation {
                        path: path.clone(),
                        source,
                    });
                }
            }
        }
        Ok(guard)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = spec;
        Ok(ShellCleanupGuard::empty())
    }
}

#[cfg(target_os = "linux")]
fn cleanup_synthetic_mount_targets(targets: &[SyntheticMountTarget]) -> Result<(), SandboxError> {
    use std::os::unix::fs::MetadataExt;

    let mut first_error = None;
    for target in targets.iter().rev() {
        let result = (|| {
            let metadata = match std::fs::symlink_metadata(&target.path) {
                Ok(metadata) => metadata,
                Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(source) => {
                    return Err(SandboxError::PathInspection {
                        path: target.path.clone(),
                        source,
                    });
                }
            };
            if !metadata.file_type().is_dir()
                || metadata.file_type().is_symlink()
                || metadata.dev() != target.device
                || metadata.ino() != target.inode
            {
                return Err(SandboxError::SyntheticMountInterference(
                    target.path.clone(),
                ));
            }
            match std::fs::remove_dir(&target.path) {
                Ok(()) => Ok(()),
                Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(source) if source.kind() == io::ErrorKind::DirectoryNotEmpty => Err(
                    SandboxError::SyntheticMountInterference(target.path.clone()),
                ),
                Err(source) => Err(SandboxError::SyntheticMountCleanup {
                    path: target.path.clone(),
                    source,
                }),
            }
        })();
        if first_error.is_none() {
            first_error = result.err();
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// Inner Linux child stage. Bubblewrap must establish the mount and network
/// namespaces before `no_new_privs` and seccomp are applied, because some
/// system Bubblewrap installations rely on setuid setup. This process then
/// replaces itself with the requested shell so every descendant inherits the
/// network syscall filter.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn x32_abi_guard() -> [seccompiler::sock_filter; 4] {
    use seccompiler::sock_filter;

    const X32_SYSCALL_BIT: u32 = 0x4000_0000;
    const SECCOMP_DATA_NR_OFFSET: u32 = 0;

    let statement = |code: u32, k: u32| sock_filter {
        code: code as u16,
        jt: 0,
        jf: 0,
        k,
    };

    [
        statement(
            libc::BPF_LD | libc::BPF_W | libc::BPF_ABS,
            SECCOMP_DATA_NR_OFFSET,
        ),
        sock_filter {
            code: (libc::BPF_JMP | libc::BPF_JSET | libc::BPF_K) as u16,
            jt: 0,
            jf: 1,
            k: X32_SYSCALL_BIT,
        },
        statement(
            libc::BPF_RET | libc::BPF_K,
            libc::SECCOMP_RET_ERRNO | libc::EPERM as u32,
        ),
        statement(libc::BPF_RET | libc::BPF_K, libc::SECCOMP_RET_ALLOW),
    ]
}

#[cfg(target_os = "linux")]
pub fn exec_linux_seccomp_shell(command: &OsStr) -> Result<(), SandboxError> {
    use seccompiler::{
        apply_filter, BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition,
        SeccompFilter, SeccompRule, TargetArch,
    };
    use std::os::unix::process::CommandExt;

    fn sandbox_error(error: impl fmt::Display) -> SandboxError {
        SandboxError::LinuxSeccomp(error.to_string())
    }

    let no_new_privileges = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if no_new_privileges != 0 {
        return Err(SandboxError::LinuxSeccomp(
            io::Error::last_os_error().to_string(),
        ));
    }

    fn deny(rules: &mut BTreeMap<i64, Vec<SeccompRule>>, syscall: i64) {
        rules.insert(syscall, Vec::new());
    }

    let mut rules = BTreeMap::new();
    for syscall in [
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_socket,
        libc::SYS_connect,
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_bind,
        libc::SYS_listen,
        libc::SYS_getpeername,
        libc::SYS_getsockname,
        libc::SYS_shutdown,
        libc::SYS_sendto,
        libc::SYS_sendmsg,
        libc::SYS_sendmmsg,
        libc::SYS_recvfrom,
        libc::SYS_recvmsg,
        libc::SYS_recvmmsg,
        libc::SYS_getsockopt,
        libc::SYS_setsockopt,
    ] {
        deny(&mut rules, syscall);
    }

    // Standalone sockets and every operation that can address a host-visible
    // socket are denied. Process-local Unix socket pairs remain available
    // through read/write for ordinary build tools; message-oriented APIs stay
    // blocked so descriptor passing cannot widen the boundary.
    let deny_non_unix = SeccompRule::new(vec![SeccompCondition::new(
        0,
        SeccompCmpArgLen::Dword,
        SeccompCmpOp::Ne,
        libc::AF_UNIX as u64,
    )
    .map_err(sandbox_error)?])
    .map_err(sandbox_error)?;
    rules.insert(libc::SYS_socketpair, vec![deny_non_unix]);

    let architecture = if cfg!(target_arch = "x86_64") {
        TargetArch::x86_64
    } else if cfg!(target_arch = "aarch64") {
        TargetArch::aarch64
    } else {
        return Err(SandboxError::LinuxSeccomp(format!(
            "unsupported Linux architecture {}",
            std::env::consts::ARCH
        )));
    };
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM as u32),
        architecture,
    )
    .map_err(sandbox_error)?;
    let program: BpfProgram = filter.try_into().map_err(sandbox_error)?;

    // x32 shares AUDIT_ARCH_X86_64 but sets a high syscall-number bit. The
    // seccompiler architecture check alone therefore cannot distinguish it
    // from the native ABI. Reject the entire alternate ABI before installing
    // the syscall-specific filter so raw x32 calls cannot bypass the denylist.
    #[cfg(target_arch = "x86_64")]
    apply_filter(&x32_abi_guard()).map_err(sandbox_error)?;
    apply_filter(&program).map_err(sandbox_error)?;

    let error = std::process::Command::new(SHELL)
        .arg("-c")
        .arg(command)
        .exec();
    Err(SandboxError::ChildExec(error))
}

fn os_args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

fn push_path_option(args: &mut Vec<OsString>, option: &str, source: &Path, target: &Path) {
    args.push(OsString::from(option));
    args.push(source.as_os_str().to_os_string());
    args.push(target.as_os_str().to_os_string());
}

fn writable_roots(policy: &ExecutionPolicy) -> Result<Vec<PathBuf>, SandboxError> {
    let mut roots = policy.effective_user_visible_roots().to_vec();
    add_canonical_directory(&mut roots, Path::new("/tmp"))?;
    if let Some(tmpdir) = std::env::var_os("TMPDIR").filter(|value| !value.is_empty()) {
        add_canonical_directory(&mut roots, Path::new(&tmpdir))?;
    }
    Ok(roots)
}

fn add_canonical_directory(roots: &mut Vec<PathBuf>, path: &Path) -> Result<(), SandboxError> {
    let canonical = std::fs::canonicalize(path).map_err(|source| SandboxError::PathInspection {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata =
        std::fs::metadata(&canonical).map_err(|source| SandboxError::PathInspection {
            path: canonical.clone(),
            source,
        })?;
    if !metadata.is_dir() {
        return Err(SandboxError::NotDirectory(canonical));
    }
    if !roots.contains(&canonical) {
        roots.push(canonical);
    }
    Ok(())
}

fn marker_env(value: &str) -> Vec<(OsString, OsString)> {
    vec![(OsString::from(SANDBOX_ENV_VAR), OsString::from(value))]
}

#[cfg(target_os = "macos")]
fn require_executable(path: &Path) -> Result<(), SandboxError> {
    if is_executable(path) {
        Ok(())
    } else {
        Err(SandboxError::BackendUnavailable {
            backend: "Seatbelt",
            searched: vec![MACOS_SANDBOX_EXEC],
        })
    }
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }

    #[cfg(not(unix))]
    {
        true
    }
}

#[derive(Debug)]
pub enum SandboxError {
    BackendUnavailable {
        backend: &'static str,
        searched: Vec<&'static str>,
    },
    UnsupportedPlatform(String),
    UnsandboxedConstrainedPolicy,
    UnenforceableProtectedPath(PathBuf),
    PathInspection {
        path: PathBuf,
        source: io::Error,
    },
    SyntheticMountCreation {
        path: PathBuf,
        source: io::Error,
    },
    SyntheticMountCleanup {
        path: PathBuf,
        source: io::Error,
    },
    SyntheticMountInterference(PathBuf),
    NotDirectory(PathBuf),
    CurrentExecutable(io::Error),
    UntrustedSandboxHelper(PathBuf),
    #[cfg(target_os = "linux")]
    LinuxSeccomp(String),
    #[cfg(target_os = "linux")]
    ChildExec(io::Error),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable { backend, searched } => write!(
                formatter,
                "{backend} sandbox backend is unavailable at trusted path(s): {}",
                searched.join(", ")
            ),
            Self::UnsupportedPlatform(platform) => {
                write!(
                    formatter,
                    "no shell sandbox backend is available for {platform}"
                )
            }
            Self::UnsandboxedConstrainedPolicy => formatter
                .write_str("refusing to run a constrained policy without an OS sandbox backend"),
            Self::UnenforceableProtectedPath(path) => write!(
                formatter,
                "cannot enforce protected metadata path {} with Bubblewrap; it must exist and must not be a symlink",
                path.display()
            ),
            Self::PathInspection { path, source } => {
                write!(
                    formatter,
                    "cannot inspect sandbox path {}: {source}",
                    path.display()
                )
            }
            Self::SyntheticMountCreation { path, source } => write!(
                formatter,
                "cannot create temporary Bubblewrap mount target {}: {source}",
                path.display()
            ),
            Self::SyntheticMountCleanup { path, source } => write!(
                formatter,
                "cannot clean temporary Bubblewrap mount target {}: {source}",
                path.display()
            ),
            Self::SyntheticMountInterference(path) => write!(
                formatter,
                "temporary Bubblewrap mount target {} changed concurrently; it was preserved",
                path.display()
            ),
            Self::NotDirectory(path) => {
                write!(
                    formatter,
                    "sandbox writable root {} is not a directory",
                    path.display()
                )
            }
            Self::CurrentExecutable(source) => {
                write!(formatter, "cannot resolve the sandbox helper executable: {source}")
            }
            Self::UntrustedSandboxHelper(path) => write!(
                formatter,
                "sandbox helper must be an absolute executable outside every sandbox-writable root: {}",
                path.display()
            ),
            #[cfg(target_os = "linux")]
            Self::LinuxSeccomp(error) => {
                write!(formatter, "cannot install the Linux network seccomp filter: {error}")
            }
            #[cfg(target_os = "linux")]
            Self::ChildExec(source) => {
                write!(formatter, "cannot start the sandboxed shell: {source}")
            }
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PathInspection { source, .. }
            | Self::SyntheticMountCreation { source, .. }
            | Self::SyntheticMountCleanup { source, .. }
            | Self::CurrentExecutable(source) => Some(source),
            #[cfg(target_os = "linux")]
            Self::ChildExec(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{ApprovalPolicy, Workspace};
    use std::ffi::OsStr;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "shaltaiboltai-sandbox-{label}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create isolated sandbox test directory");
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
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with("shaltaiboltai-sandbox-"));
            if is_ours {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }

    #[cfg(target_os = "macos")]
    struct OutsideFile {
        path: PathBuf,
    }

    #[cfg(target_os = "macos")]
    impl OutsideFile {
        fn new(label: &str) -> Option<Self> {
            let home = std::env::var_os("HOME").filter(|value| !value.is_empty())?;
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = PathBuf::from(home).join(format!(
                ".shaltaiboltai-sandbox-{label}-{}-{sequence}",
                std::process::id()
            ));
            if path.exists() {
                return None;
            }
            Some(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    #[cfg(target_os = "macos")]
    impl Drop for OutsideFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn policy_fixture(mode: SandboxMode) -> (TestDirectory, ExecutionPolicy) {
        let directory = TestDirectory::new(mode.to_string().as_str());
        for name in PROTECTED_METADATA_NAMES {
            fs::create_dir(directory.path().join(name)).expect("create protected metadata");
        }
        let workspace = Workspace::new(directory.path()).expect("canonical workspace");
        (
            directory,
            ExecutionPolicy::from_parts(workspace, mode, ApprovalPolicy::OnRequest),
        )
    }

    fn strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn constrained_policy_rejects_unsandboxed_backend() {
        let (_directory, policy) = policy_fixture(SandboxMode::ReadOnly);
        let error =
            prepare_shell_command_for_backend(&policy, "true", false, SandboxBackend::Unsandboxed)
                .expect_err("constrained command must fail closed");
        assert!(matches!(error, SandboxError::UnsandboxedConstrainedPolicy));
    }

    #[test]
    fn explicit_escalation_and_full_access_are_raw_shells() {
        let (_read_only_directory, read_only) = policy_fixture(SandboxMode::ReadOnly);
        let escalated = prepare_shell_command_for_backend(
            &read_only,
            "printf escalated",
            true,
            SandboxBackend::Unsandboxed,
        )
        .expect("approved escalation");
        assert_eq!(escalated.program(), Path::new(SHELL));
        assert_eq!(strings(escalated.args()), ["-c", "printf escalated"]);
        assert_eq!(escalated.backend(), SandboxBackend::Unsandboxed);
        assert_eq!(
            escalated.env(),
            &[(OsString::from(SANDBOX_ENV_VAR), OsString::from("escalated"))]
        );

        let (_full_directory, full) = policy_fixture(SandboxMode::DangerFullAccess);
        let direct = prepare_shell_command_for_backend(
            &full,
            "printf full",
            false,
            SandboxBackend::MacosSeatbelt,
        )
        .expect("full access");
        assert_eq!(direct.program(), Path::new(SHELL));
        assert_eq!(strings(direct.args()), ["-c", "printf full"]);
    }

    #[test]
    fn seatbelt_read_only_is_closed_default_full_read_and_no_write_allowlist() {
        let (_directory, policy) = policy_fixture(SandboxMode::ReadOnly);
        let spec =
            prepare_shell_command_for_backend(&policy, "pwd", false, SandboxBackend::MacosSeatbelt)
                .expect("Seatbelt spec");
        assert_eq!(spec.program(), Path::new(MACOS_SANDBOX_EXEC));
        assert_eq!(spec.cwd(), policy.workspace().cwd());
        assert_eq!(spec.backend(), SandboxBackend::MacosSeatbelt);
        let args = strings(spec.args());
        let profile = &args[1];
        assert!(profile.contains("(deny default)"));
        assert!(profile.contains("(allow file-read*)"));
        assert!(!profile.contains("Writes are limited"));
        assert!(!profile.contains("(allow network-outbound"));
        assert_eq!(&args[args.len() - 4..], ["--", SHELL, "-c", "pwd"]);
    }

    #[test]
    fn seatbelt_workspace_write_uses_params_and_global_metadata_denies() {
        let (_directory, policy) = policy_fixture(SandboxMode::WorkspaceWrite);
        let spec = prepare_shell_command_for_backend(
            &policy,
            "touch output",
            false,
            SandboxBackend::MacosSeatbelt,
        )
        .expect("Seatbelt spec");
        let args = strings(spec.args());
        let profile = &args[1];
        assert!(profile.contains("(subpath (param \"WRITABLE_ROOT_0\"))"));
        assert!(profile.contains("(deny file-write* (literal (param \"PROTECTED_METADATA_0\"))"));
        assert!(profile.contains("(deny file-write-unlink"));
        assert!(args.iter().any(|arg| {
            arg == &format!("-DWRITABLE_ROOT_0={}", policy.workspace().cwd().display())
        }));
        for name in PROTECTED_METADATA_NAMES {
            assert!(args.iter().any(|arg| arg.ends_with(name)));
        }
        for alias in [".GIT", ".Agents", ".CODEX"] {
            assert!(
                args.iter().any(|arg| arg.ends_with(alias)),
                "Seatbelt must deny alternate-case alias {alias}"
            );
        }
    }

    #[test]
    fn bubblewrap_read_only_has_full_read_namespaces_and_no_writable_bind() {
        let (directory, policy) = policy_fixture(SandboxMode::ReadOnly);
        let spec = bubblewrap_spec_with_executable(
            &policy,
            "pwd",
            directory.path().to_path_buf(),
            Path::new("/trusted/bwrap"),
            BubblewrapExecutable::UsrBin,
        )
        .expect("bubblewrap spec");
        let args = strings(spec.args());
        assert_eq!(spec.program(), Path::new("/trusted/bwrap"));
        assert!(args
            .windows(3)
            .any(|window| window == ["--ro-bind", "/", "/"]));
        assert!(args.iter().any(|arg| arg == "--unshare-user"));
        assert!(args.iter().any(|arg| arg == "--unshare-pid"));
        assert!(args.iter().any(|arg| arg == "--unshare-net"));
        assert!(!args.iter().any(|arg| arg == "--bind"));
        assert_eq!(args[args.len() - 3], "--");
        assert_eq!(
            Path::new(&args[args.len() - 2]),
            std::env::current_exe().expect("current test executable")
        );
        assert_eq!(
            args.last().map(String::as_str),
            Some("--__sandbox-seccomp-exec=pwd")
        );
    }

    #[test]
    fn bubblewrap_workspace_write_rebinds_roots_and_metadata_after_writable_mounts() {
        let (directory, policy) = policy_fixture(SandboxMode::WorkspaceWrite);
        let spec = bubblewrap_spec_with_executable(
            &policy,
            "touch output",
            directory.path().to_path_buf(),
            Path::new("/trusted/bwrap"),
            BubblewrapExecutable::UsrBin,
        )
        .expect("bubblewrap spec");
        let args = strings(spec.args());
        let workspace = policy.workspace().cwd().to_string_lossy();
        assert!(args
            .windows(3)
            .any(|window| window == ["--bind", workspace.as_ref(), workspace.as_ref()]));
        for name in PROTECTED_METADATA_NAMES {
            let protected = policy
                .workspace()
                .cwd()
                .join(name)
                .to_string_lossy()
                .into_owned();
            assert!(args
                .windows(3)
                .any(|window| window == ["--ro-bind", protected.as_str(), protected.as_str()]));
        }
        let last_bind = args.iter().rposition(|arg| arg == "--bind").unwrap();
        let first_protection = args.iter().position(|arg| arg == "--ro-bind").unwrap();
        let git_path = policy
            .workspace()
            .cwd()
            .join(".git")
            .to_string_lossy()
            .into_owned();
        let metadata_protection = args
            .iter()
            .enumerate()
            .skip(first_protection + 1)
            .find_map(|(index, arg)| (arg == &git_path).then_some(index - 1))
            .expect("metadata protection");
        assert!(metadata_protection > last_bind);
        assert_eq!(
            args.last().map(String::as_str),
            Some("--__sandbox-seccomp-exec=touch output")
        );
    }

    #[cfg(unix)]
    #[test]
    fn bubblewrap_rejects_a_seccomp_helper_inside_a_writable_root() {
        use std::os::unix::fs::PermissionsExt;

        let directory = TestDirectory::new("writable-seccomp-helper");
        let helper = directory.path().join("replaceable-helper");
        fs::write(&helper, "#!/bin/sh\nexit 0\n").expect("write fake helper");
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o755))
            .expect("make fake helper executable");
        let policy =
            ExecutionPolicy::new(Workspace::new(directory.path()).expect("canonical workspace"));

        let error = bubblewrap_spec_with_executable_and_helper(
            &policy,
            "true",
            policy.workspace().cwd().to_path_buf(),
            Path::new("/trusted/bwrap"),
            BubblewrapExecutable::UsrBin,
            &helper,
        )
        .expect_err("a model-writable helper must fail closed");
        assert!(matches!(
            error,
            SandboxError::UntrustedSandboxHelper(path) if path == helper
        ));
    }

    #[test]
    fn linked_worktree_git_directories_are_read_only_in_both_sandboxes() {
        let directory = TestDirectory::new("linked-worktree-metadata");
        let git_dir = directory.path().join("metadata/worktrees/linked");
        let common_dir = directory.path().join("metadata/common");
        fs::create_dir_all(&git_dir).expect("create linked git directory");
        fs::create_dir_all(common_dir.join("refs")).expect("create common git directory");
        fs::write(
            directory.path().join(".git"),
            format!("gitdir: {}\n", git_dir.display()),
        )
        .expect("write gitdir pointer");
        fs::write(git_dir.join("commondir"), "../../common\n").expect("write commondir pointer");
        let policy = ExecutionPolicy::new(
            Workspace::new(directory.path()).expect("canonical linked workspace"),
        );

        let bubblewrap = bubblewrap_spec_with_executable(
            &policy,
            "true",
            policy.workspace().cwd().to_path_buf(),
            Path::new("/trusted/bwrap"),
            BubblewrapExecutable::UsrBin,
        )
        .expect("Bubblewrap linked-worktree spec");
        let bubblewrap_args = strings(bubblewrap.args());
        let seatbelt = prepare_shell_command_for_backend(
            &policy,
            "true",
            false,
            SandboxBackend::MacosSeatbelt,
        )
        .expect("Seatbelt linked-worktree spec");
        let seatbelt_args = strings(seatbelt.args());

        for protected in [
            fs::canonicalize(&git_dir).unwrap(),
            fs::canonicalize(&common_dir).unwrap(),
        ] {
            let rendered = protected.to_string_lossy();
            assert!(bubblewrap_args
                .windows(3)
                .any(|window| { window == ["--ro-bind", rendered.as_ref(), rendered.as_ref()] }));
            assert!(seatbelt_args
                .iter()
                .any(|argument| argument.ends_with(rendered.as_ref())));
        }
    }

    #[test]
    fn bubblewrap_masks_missing_protected_metadata_with_synthetic_read_only_mounts() {
        let directory = TestDirectory::new("missing-metadata");
        let policy =
            ExecutionPolicy::new(Workspace::new(directory.path()).expect("canonical workspace"));
        let spec = bubblewrap_spec_with_executable(
            &policy,
            "true",
            policy.workspace().cwd().to_path_buf(),
            Path::new("/trusted/bwrap"),
            BubblewrapExecutable::UsrBin,
        )
        .expect("missing metadata gets a synthetic boundary");
        let args = strings(spec.args());
        for name in PROTECTED_METADATA_NAMES {
            let protected = policy.workspace().cwd().join(name);
            let rendered = protected.to_string_lossy();
            assert!(spec.synthetic_mount_targets().contains(&protected));
            assert!(args.windows(6).any(|window| {
                window
                    == [
                        "--perms",
                        "555",
                        "--tmpfs",
                        rendered.as_ref(),
                        "--remount-ro",
                        rendered.as_ref(),
                    ]
            }));
        }
    }

    #[cfg(unix)]
    #[test]
    fn bubblewrap_fails_closed_for_symlinked_protected_metadata() {
        let directory = TestDirectory::new("symlinked-metadata");
        let target = directory.path().join("metadata-target");
        fs::create_dir(&target).expect("metadata target");
        std::os::unix::fs::symlink(&target, directory.path().join(".git"))
            .expect("protected symlink");
        let policy =
            ExecutionPolicy::new(Workspace::new(directory.path()).expect("canonical workspace"));
        let error = bubblewrap_spec_with_executable(
            &policy,
            "true",
            policy.workspace().cwd().to_path_buf(),
            Path::new("/trusted/bwrap"),
            BubblewrapExecutable::UsrBin,
        )
        .expect_err("symlinked metadata boundary must fail closed");
        assert!(matches!(
            error,
            SandboxError::UnenforceableProtectedPath(path)
                if path == policy.workspace().cwd().join(".git")
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn synthetic_mount_targets_are_created_and_cleaned_by_exact_identity() {
        let directory = TestDirectory::new("synthetic-cleanup");
        let mut spec = raw_shell_spec("true", directory.path().to_path_buf(), "test");
        spec.synthetic_mount_targets = PROTECTED_METADATA_NAMES
            .iter()
            .map(|name| directory.path().join(name))
            .collect();
        let guard = materialize_synthetic_mount_targets(&spec).expect("materialize targets");
        for name in PROTECTED_METADATA_NAMES {
            assert!(directory.path().join(name).is_dir());
        }
        guard.cleanup().expect("identity-bound cleanup");
        for name in PROTECTED_METADATA_NAMES {
            assert!(!directory.path().join(name).exists());
        }
    }

    #[test]
    fn command_spec_conversion_preserves_program_arguments_cwd_and_marker() {
        let (_directory, policy) = policy_fixture(SandboxMode::DangerFullAccess);
        let spec = prepare_shell_command_for_backend(
            &policy,
            "printf ok",
            false,
            SandboxBackend::Unsandboxed,
        )
        .expect("raw command");
        let expected = spec.clone();
        let command = spec.into_tokio_command();
        let std = command.as_std();
        assert_eq!(std.get_program(), expected.program());
        assert_eq!(std.get_args().collect::<Vec<_>>(), expected.args());
        assert_eq!(std.get_current_dir(), Some(policy.workspace().cwd()));
        assert!(std.get_envs().any(|(key, value)| {
            key == OsStr::new(SANDBOX_ENV_VAR) && value == Some(OsStr::new("danger-full-access"))
        }));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn seatbelt_smoke_enforces_read_write_metadata_and_network_boundaries() {
        if !is_executable(Path::new(MACOS_SANDBOX_EXEC)) || !seatbelt_can_nest() {
            return;
        }
        let Some(outside) = OutsideFile::new("seatbelt-outside") else {
            return;
        };

        let (read_only_directory, read_only) = policy_fixture(SandboxMode::ReadOnly);
        let denied = run_sync(
            prepare_shell_command_for_backend(
                &read_only,
                "printf denied > read-only.txt",
                false,
                SandboxBackend::MacosSeatbelt,
            )
            .expect("read-only spec"),
        );
        assert!(!denied.status.success());
        assert!(!read_only_directory.path().join("read-only.txt").exists());

        let (workspace_directory, workspace) = policy_fixture(SandboxMode::WorkspaceWrite);
        let allowed = run_sync(
            prepare_shell_command_for_backend(
                &workspace,
                "printf allowed > workspace.txt",
                false,
                SandboxBackend::MacosSeatbelt,
            )
            .expect("workspace spec"),
        );
        assert!(allowed.status.success(), "{allowed:?}");
        assert_eq!(
            fs::read_to_string(workspace_directory.path().join("workspace.txt")).unwrap(),
            "allowed"
        );

        let outside_command = format!("printf denied > '{}'", outside.path().display());
        let denied = run_sync(
            prepare_shell_command_for_backend(
                &workspace,
                &outside_command,
                false,
                SandboxBackend::MacosSeatbelt,
            )
            .expect("outside-write spec"),
        );
        assert!(!denied.status.success());

        let protected = run_sync(
            prepare_shell_command_for_backend(
                &workspace,
                "printf denied > .git/config",
                false,
                SandboxBackend::MacosSeatbelt,
            )
            .expect("metadata spec"),
        );
        assert!(!protected.status.success());

        use std::os::unix::fs::MetadataExt;
        let lowercase = fs::metadata(workspace_directory.path().join(".git")).unwrap();
        if let Ok(uppercase) = fs::metadata(workspace_directory.path().join(".GIT")) {
            if lowercase.dev() == uppercase.dev() && lowercase.ino() == uppercase.ino() {
                let alternate_case = run_sync(
                    prepare_shell_command_for_backend(
                        &workspace,
                        "printf denied > .GIT/config",
                        false,
                        SandboxBackend::MacosSeatbelt,
                    )
                    .expect("alternate-case metadata spec"),
                );
                assert!(!alternate_case.status.success());
                assert!(!workspace_directory.path().join(".git/config").exists());
            }
        }

        let network = run_sync(
            prepare_shell_command_for_backend(
                &workspace,
                "/usr/bin/curl --connect-timeout 1 --silent https://example.com >/dev/null",
                false,
                SandboxBackend::MacosSeatbelt,
            )
            .expect("network spec"),
        );
        assert!(!network.status.success());
    }

    #[cfg(target_os = "macos")]
    fn seatbelt_can_nest() -> bool {
        let output = std::process::Command::new(MACOS_SANDBOX_EXEC)
            .args(["-p", "(version 1) (allow default)", "--", "/usr/bin/true"])
            .output();
        output.is_ok_and(|output| output.status.success())
    }

    #[cfg(target_os = "macos")]
    fn run_sync(spec: CommandSpec) -> std::process::Output {
        let mut command = std::process::Command::new(spec.program);
        command.args(spec.args);
        command.current_dir(spec.cwd);
        command.envs(spec.env);
        command.output().expect("run sandbox smoke command")
    }
}
