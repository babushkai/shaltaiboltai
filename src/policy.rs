//! Typed execution authority shared by CLI, TUI, and tool enforcement.
//!
//! The types in this module describe authority; they do not implement an OS
//! sandbox. Callers must enforce the resulting classifications at the tool and
//! process boundary.

use std::collections::hash_map::DefaultHasher;
use std::ffi::OsStr;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

const PROTECTED_METADATA_NAMES: [&str; 3] = [".git", ".agents", ".codex"];
const MAX_GIT_POINTER_BYTES: u64 = 16 * 1024;
static NEXT_AUTHORITY_ID: AtomicU64 = AtomicU64::new(1);

/// Filesystem authority applied to model-initiated tools and commands.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SandboxMode {
    ReadOnly,
    #[default]
    WorkspaceWrite,
    DangerFullAccess,
}

impl SandboxMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "Read Only",
            Self::WorkspaceWrite => "Workspace",
            Self::DangerFullAccess => "Full Access",
        }
    }
}

impl fmt::Display for SandboxMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        })
    }
}

impl FromStr for SandboxMode {
    type Err = ParsePolicyValueError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "read-only" => Ok(Self::ReadOnly),
            "workspace-write" => Ok(Self::WorkspaceWrite),
            "danger-full-access" => Ok(Self::DangerFullAccess),
            _ => Err(ParsePolicyValueError::new("sandbox mode", value)),
        }
    }
}

/// Determines when a boundary crossing may be presented to the user.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ApprovalPolicy {
    Untrusted,
    #[default]
    OnRequest,
    Never,
}

impl ApprovalPolicy {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Untrusted => "Untrusted",
            Self::OnRequest => "Ask for approval",
            Self::Never => "Never",
        }
    }

    const fn boundary_action(self) -> BoundaryAction {
        match self {
            Self::Untrusted | Self::OnRequest => BoundaryAction::RequiresApproval,
            Self::Never => BoundaryAction::Deny,
        }
    }
}

impl fmt::Display for ApprovalPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Untrusted => "untrusted",
            Self::OnRequest => "on-request",
            Self::Never => "never",
        })
    }
}

impl FromStr for ApprovalPolicy {
    type Err = ParsePolicyValueError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "untrusted" => Ok(Self::Untrusted),
            "on-request" => Ok(Self::OnRequest),
            "never" => Ok(Self::Never),
            _ => Err(ParsePolicyValueError::new("approval policy", value)),
        }
    }
}

/// User-facing combinations shown by `/permissions`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum PermissionPreset {
    ReadOnly,
    #[default]
    AskForApproval,
    FullAccess,
}

impl PermissionPreset {
    pub const ALL: [Self; 3] = [Self::ReadOnly, Self::AskForApproval, Self::FullAccess];

    pub const fn id(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::AskForApproval => "auto",
            Self::FullAccess => "full-access",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "Read Only",
            Self::AskForApproval => "Ask for approval",
            Self::FullAccess => "Full Access",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::ReadOnly => {
                "Read files. Ask before edits or network access."
            }
            Self::AskForApproval => {
                "Edit and run sandboxed commands in the workspace. Ask before network or outside writes."
            }
            Self::FullAccess => {
                "Edit any file and use the network without asking. This can expose or delete data."
            }
        }
    }

    pub const fn sandbox_mode(self) -> SandboxMode {
        match self {
            Self::ReadOnly => SandboxMode::ReadOnly,
            Self::AskForApproval => SandboxMode::WorkspaceWrite,
            Self::FullAccess => SandboxMode::DangerFullAccess,
        }
    }

    pub const fn approval_policy(self) -> ApprovalPolicy {
        match self {
            Self::ReadOnly | Self::AskForApproval => ApprovalPolicy::OnRequest,
            Self::FullAccess => ApprovalPolicy::Never,
        }
    }
}

impl fmt::Display for PermissionPreset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.id())
    }
}

impl FromStr for PermissionPreset {
    type Err = ParsePolicyValueError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "read-only" => Ok(Self::ReadOnly),
            "auto" | "ask-for-approval" | "default" => Ok(Self::AskForApproval),
            "full-access" => Ok(Self::FullAccess),
            _ => Err(ParsePolicyValueError::new("permission preset", value)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsePolicyValueError {
    kind: &'static str,
    value: String,
}

impl ParsePolicyValueError {
    fn new(kind: &'static str, value: &str) -> Self {
        Self {
            kind,
            value: value.to_owned(),
        }
    }

    pub const fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParsePolicyValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid {} `{}`", self.kind, self.value)
    }
}

impl std::error::Error for ParsePolicyValueError {}

/// Canonical workspace roots. Construction is the only time roots can change.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Workspace {
    cwd: PathBuf,
    roots: Vec<PathBuf>,
}

impl Workspace {
    /// Canonicalize `cwd` and create a workspace containing only that root.
    pub fn new(cwd: impl AsRef<Path>) -> Result<Self, PolicyPathError> {
        let cwd = canonical_directory(cwd.as_ref(), PathOperation::WorkspaceRoot)?;
        Ok(Self {
            roots: vec![cwd.clone()],
            cwd,
        })
    }

    /// Canonicalize all additional roots relative to `cwd`.
    pub fn from_roots<I, P>(cwd: impl AsRef<Path>, roots: I) -> Result<Self, PolicyPathError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self::new(cwd)?.with_extra_roots(roots)
    }

    /// Return a new workspace with one additional canonical root.
    pub fn with_extra_root(mut self, root: impl AsRef<Path>) -> Result<Self, PolicyPathError> {
        let input = root.as_ref();
        let absolute = if input.is_absolute() {
            input.to_path_buf()
        } else {
            self.cwd.join(input)
        };
        let root = canonical_directory(&absolute, PathOperation::AdditionalRoot)?;
        if !self.roots.contains(&root) {
            self.roots.push(root);
        }
        Ok(self)
    }

    /// Return a new workspace with repeatable additional roots, preserving
    /// first-seen order after canonical-path deduplication.
    pub fn with_extra_roots<I, P>(mut self, roots: I) -> Result<Self, PolicyPathError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        for root in roots {
            self = self.with_extra_root(root)?;
        }
        Ok(self)
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Canonical roots shown to the user and considered part of the workspace.
    pub fn effective_user_visible_roots(&self) -> &[PathBuf] {
        &self.roots
    }

    fn contains(&self, target: &Path) -> bool {
        self.roots.iter().any(|root| target.starts_with(root))
    }

    fn contains_protected_metadata(&self, target: &Path) -> bool {
        self.roots
            .iter()
            .any(|root| contains_logical_protected_metadata(root, target))
            || protected_metadata_paths_for_roots(&self.roots)
                .iter()
                .any(|protected| target.starts_with(protected))
    }
}

fn contains_logical_protected_metadata(root: &Path, target: &Path) -> bool {
    let Ok(relative) = target.strip_prefix(root) else {
        return false;
    };
    let Some(std::path::Component::Normal(name)) = relative.components().next() else {
        return false;
    };
    is_protected_metadata_name(name)
}

fn is_protected_metadata_name(name: &OsStr) -> bool {
    #[cfg(any(target_os = "macos", windows))]
    {
        name.to_str().is_some_and(|name| {
            PROTECTED_METADATA_NAMES
                .iter()
                .any(|protected| name.eq_ignore_ascii_case(protected))
        })
    }

    #[cfg(not(any(target_os = "macos", windows)))]
    {
        PROTECTED_METADATA_NAMES
            .iter()
            .any(|protected| name == OsStr::new(protected))
    }
}

/// Expand protected project metadata to its effective filesystem locations.
///
/// Besides top-level `.git`, `.agents`, and `.codex`, this includes worktree
/// `gitdir:` pointers, their optional `commondir`, symlink targets, and a bare
/// repository root. Both direct tools and OS sandboxes use this exact expansion
/// so their authority decisions cannot diverge.
pub(crate) fn protected_metadata_paths_for_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for root in roots {
        if looks_like_bare_git_repository(root) {
            push_unique_path(&mut paths, root.clone());
        }

        for name in PROTECTED_METADATA_NAMES {
            let logical = root.join(name);
            push_unique_path(&mut paths, logical.clone());
            if let Ok(resolved) = std::fs::canonicalize(&logical) {
                push_unique_path(&mut paths, resolved);
            }
        }

        let dot_git = root.join(".git");
        if dot_git.is_file() {
            if let Some(git_dir) = resolve_git_pointer(&dot_git) {
                push_unique_path(&mut paths, git_dir.clone());
                if let Some(common_dir) = resolve_common_git_dir(&git_dir) {
                    push_unique_path(&mut paths, common_dir);
                }
            }
        }
    }
    paths
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}

fn looks_like_bare_git_repository(root: &Path) -> bool {
    root.join("HEAD").is_file() && root.join("objects").is_dir() && root.join("refs").is_dir()
}

fn resolve_git_pointer(dot_git: &Path) -> Option<PathBuf> {
    let contents = read_small_metadata_file(dot_git)?;
    let raw = contents.trim().strip_prefix("gitdir:")?.trim();
    if raw.is_empty() || raw.contains(['\r', '\n']) {
        return None;
    }
    canonical_directory_from_pointer(dot_git.parent()?, raw)
}

fn resolve_common_git_dir(git_dir: &Path) -> Option<PathBuf> {
    let pointer = git_dir.join("commondir");
    if !pointer.is_file() {
        return None;
    }
    let raw = read_small_metadata_file(&pointer)?;
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(['\r', '\n']) {
        return None;
    }
    canonical_directory_from_pointer(git_dir, raw)
}

fn canonical_directory_from_pointer(base: &Path, raw: &str) -> Option<PathBuf> {
    let raw = Path::new(raw);
    let path = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        base.join(raw)
    };
    let canonical = std::fs::canonicalize(path).ok()?;
    canonical.is_dir().then_some(canonical)
}

fn read_small_metadata_file(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > MAX_GIT_POINTER_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize + 1);
    std::fs::File::open(path)
        .ok()?
        .take(MAX_GIT_POINTER_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_GIT_POINTER_BYTES {
        return None;
    }
    String::from_utf8(bytes).ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathOperation {
    WorkspaceRoot,
    AdditionalRoot,
    TargetResolution,
}

impl fmt::Display for PathOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::WorkspaceRoot => "workspace root",
            Self::AdditionalRoot => "additional workspace root",
            Self::TargetResolution => "target",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyPathErrorKind {
    NotFound,
    NotDirectory,
    DanglingSymlink,
    Inaccessible,
    InvalidPath,
}

/// A fail-closed path validation or resolution error.
#[derive(Debug)]
pub struct PolicyPathError {
    operation: PathOperation,
    kind: PolicyPathErrorKind,
    path: PathBuf,
    source: Option<io::Error>,
}

impl PolicyPathError {
    fn new(
        operation: PathOperation,
        kind: PolicyPathErrorKind,
        path: PathBuf,
        source: Option<io::Error>,
    ) -> Self {
        Self {
            operation,
            kind,
            path,
            source,
        }
    }

    pub const fn operation(&self) -> PathOperation {
        self.operation
    }

    pub const fn kind(&self) -> PolicyPathErrorKind {
        self.kind
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl fmt::Display for PolicyPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self.kind {
            PolicyPathErrorKind::NotFound => "does not exist",
            PolicyPathErrorKind::NotDirectory => "is not a directory",
            PolicyPathErrorKind::DanglingSymlink => "contains a dangling symlink",
            PolicyPathErrorKind::Inaccessible => "cannot be safely resolved",
            PolicyPathErrorKind::InvalidPath => "has no resolvable filesystem ancestor",
        };
        write!(
            formatter,
            "{} {} {reason}",
            self.operation,
            self.path.display()
        )
    }
}

impl std::error::Error for PolicyPathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

fn canonical_directory(path: &Path, operation: PathOperation) -> Result<PathBuf, PolicyPathError> {
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        let kind = if error.kind() == io::ErrorKind::NotFound {
            PolicyPathErrorKind::NotFound
        } else {
            PolicyPathErrorKind::Inaccessible
        };
        PolicyPathError::new(operation, kind, path.to_path_buf(), Some(error))
    })?;
    let metadata = std::fs::metadata(&canonical).map_err(|error| {
        PolicyPathError::new(
            operation,
            PolicyPathErrorKind::Inaccessible,
            canonical.clone(),
            Some(error),
        )
    })?;
    if !metadata.is_dir() {
        return Err(PolicyPathError::new(
            operation,
            PolicyPathErrorKind::NotDirectory,
            canonical,
            None,
        ));
    }
    Ok(canonical)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessKind {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PathBoundary {
    Workspace,
    ProtectedMetadata,
    OutsideWorkspace,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BoundaryAction {
    Allow,
    RequiresApproval,
    Deny,
}

/// Resolved target plus the authority decision for one access kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathClassification {
    access: AccessKind,
    target: PathBuf,
    boundary: PathBoundary,
    action: BoundaryAction,
}

impl PathClassification {
    pub const fn access(&self) -> AccessKind {
        self.access
    }

    pub fn target(&self) -> &Path {
        &self.target
    }

    pub const fn boundary(&self) -> PathBoundary {
        self.boundary
    }

    pub const fn action(&self) -> BoundaryAction {
        self.action
    }
}

/// Exact policy content used when binding a session approval.
///
/// Equality compares the full typed policy, while `Display` provides a compact
/// diagnostic identifier rather than an authentication token.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PolicyFingerprint {
    sandbox_mode: SandboxMode,
    approval_policy: ApprovalPolicy,
    network_access: bool,
    roots: Vec<PathBuf>,
}

impl fmt::Display for PolicyFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        write!(formatter, "{:016x}", hasher.finish())
    }
}

/// A session grant bound to one authority instance, generation, and scope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrantBinding {
    authority_id: u64,
    generation: u64,
    fingerprint: PolicyFingerprint,
    scope: Vec<u8>,
}

impl GrantBinding {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn fingerprint(&self) -> &PolicyFingerprint {
        &self.fingerprint
    }

    pub fn scope(&self) -> &[u8] {
        &self.scope
    }
}

/// Active runtime authority. Policy changes invalidate previously bound grants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPolicy {
    workspace: Workspace,
    sandbox_mode: SandboxMode,
    approval_policy: ApprovalPolicy,
    generation: u64,
    authority_id: u64,
}

impl ExecutionPolicy {
    /// Construct the production default: workspace-write, on-request approval,
    /// and restricted network access.
    pub fn new(workspace: Workspace) -> Self {
        Self::from_parts(workspace, SandboxMode::default(), ApprovalPolicy::default())
    }

    pub fn from_parts(
        workspace: Workspace,
        sandbox_mode: SandboxMode,
        approval_policy: ApprovalPolicy,
    ) -> Self {
        Self {
            workspace,
            sandbox_mode,
            approval_policy,
            generation: 1,
            authority_id: next_authority_id(),
        }
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub const fn sandbox_mode(&self) -> SandboxMode {
        self.sandbox_mode
    }

    pub const fn approval_policy(&self) -> ApprovalPolicy {
        self.approval_policy
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Constrained modes never receive network authority from this policy.
    pub const fn network_access_enabled(&self) -> bool {
        matches!(self.sandbox_mode, SandboxMode::DangerFullAccess)
    }

    pub const fn network_status_label(&self) -> &'static str {
        if self.network_access_enabled() {
            "enabled"
        } else {
            "restricted"
        }
    }

    pub fn effective_user_visible_roots(&self) -> &[PathBuf] {
        self.workspace.effective_user_visible_roots()
    }

    pub fn matching_preset(&self) -> Option<PermissionPreset> {
        PermissionPreset::ALL.into_iter().find(|preset| {
            self.sandbox_mode == preset.sandbox_mode()
                && self.approval_policy == preset.approval_policy()
        })
    }

    /// Label used in compact permission surfaces.
    pub fn status_label(&self) -> &'static str {
        self.matching_preset()
            .map_or("Custom permissions", PermissionPreset::label)
    }

    pub const fn permission_status_label(&self) -> &'static str {
        self.sandbox_mode.label()
    }

    pub const fn approval_status_label(&self) -> &'static str {
        self.approval_policy.label()
    }

    /// Apply a built-in preset, incrementing the generation only if authority
    /// actually changed. The return value reports whether grants were invalidated.
    pub fn apply_preset(&mut self, preset: PermissionPreset) -> bool {
        self.update(preset.sandbox_mode(), preset.approval_policy())
    }

    /// Apply custom CLI settings as one atomic policy generation.
    pub fn update(&mut self, sandbox_mode: SandboxMode, approval_policy: ApprovalPolicy) -> bool {
        if self.sandbox_mode == sandbox_mode && self.approval_policy == approval_policy {
            return false;
        }
        self.sandbox_mode = sandbox_mode;
        self.approval_policy = approval_policy;
        self.generation = self.generation.wrapping_add(1);
        true
    }

    /// Resolve an existing target or a prospective target beneath its nearest
    /// existing ancestor. Present-but-unresolvable entries fail closed.
    pub fn canonical_target(&self, target: impl AsRef<Path>) -> Result<PathBuf, PolicyPathError> {
        resolve_target(self.workspace.cwd(), target.as_ref())
    }

    pub fn is_protected_metadata(&self, target: impl AsRef<Path>) -> Result<bool, PolicyPathError> {
        let target = self.canonical_target(target)?;
        Ok(self.workspace.contains_protected_metadata(&target))
    }

    pub fn classify_read(
        &self,
        target: impl AsRef<Path>,
    ) -> Result<PathClassification, PolicyPathError> {
        self.classify(target.as_ref(), AccessKind::Read)
    }

    pub fn classify_write(
        &self,
        target: impl AsRef<Path>,
    ) -> Result<PathClassification, PolicyPathError> {
        self.classify(target.as_ref(), AccessKind::Write)
    }

    fn classify(
        &self,
        target: &Path,
        access: AccessKind,
    ) -> Result<PathClassification, PolicyPathError> {
        let target = self.canonical_target(target)?;
        let boundary = if self.workspace.contains_protected_metadata(&target) {
            PathBoundary::ProtectedMetadata
        } else if self.workspace.contains(&target) {
            PathBoundary::Workspace
        } else {
            PathBoundary::OutsideWorkspace
        };
        let action = self.action_for(access, boundary);
        Ok(PathClassification {
            access,
            target,
            boundary,
            action,
        })
    }

    fn action_for(&self, access: AccessKind, boundary: PathBoundary) -> BoundaryAction {
        if access == AccessKind::Write && self.approval_policy == ApprovalPolicy::Untrusted {
            return BoundaryAction::RequiresApproval;
        }

        if self.sandbox_mode == SandboxMode::DangerFullAccess {
            return BoundaryAction::Allow;
        }

        match (access, boundary, self.sandbox_mode) {
            (AccessKind::Read, _, _)
            | (AccessKind::Write, PathBoundary::Workspace, SandboxMode::WorkspaceWrite) => {
                BoundaryAction::Allow
            }
            (AccessKind::Write, PathBoundary::OutsideWorkspace, _)
            | (AccessKind::Write, PathBoundary::ProtectedMetadata, _)
            | (AccessKind::Write, PathBoundary::Workspace, SandboxMode::ReadOnly) => {
                self.approval_policy.boundary_action()
            }
            (_, _, SandboxMode::DangerFullAccess) => BoundaryAction::Allow,
        }
    }

    pub fn fingerprint(&self) -> PolicyFingerprint {
        PolicyFingerprint {
            sandbox_mode: self.sandbox_mode,
            approval_policy: self.approval_policy,
            network_access: self.network_access_enabled(),
            roots: self.effective_user_visible_roots().to_vec(),
        }
    }

    pub fn bind_grant(&self, scope: impl Into<Vec<u8>>) -> GrantBinding {
        GrantBinding {
            authority_id: self.authority_id,
            generation: self.generation,
            fingerprint: self.fingerprint(),
            scope: scope.into(),
        }
    }

    pub fn accepts_grant(&self, binding: &GrantBinding, scope: &[u8]) -> bool {
        binding.authority_id == self.authority_id
            && binding.generation == self.generation
            && binding.fingerprint == self.fingerprint()
            && binding.scope == scope
    }
}

fn next_authority_id() -> u64 {
    let id = NEXT_AUTHORITY_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        NEXT_AUTHORITY_ID.fetch_add(1, Ordering::Relaxed)
    } else {
        id
    }
}

fn resolve_target(cwd: &Path, target: &Path) -> Result<PathBuf, PolicyPathError> {
    let absolute = if target.as_os_str().is_empty() {
        cwd.to_path_buf()
    } else if target.is_absolute() {
        target.to_path_buf()
    } else {
        cwd.join(target)
    };
    let mut resolved = PathBuf::new();

    // Resolve one component at a time. Existing components are canonicalized
    // immediately, so `link/..` follows the link before applying `..` just as
    // the kernel does. Missing components remain lexical, allowing prospective
    // writes, but later parent components are applied before classification.
    for component in absolute.components() {
        match component {
            std::path::Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            std::path::Component::RootDir => resolved.push(std::path::MAIN_SEPARATOR.to_string()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            std::path::Component::Normal(name) => {
                let candidate = resolved.join(name);
                match std::fs::canonicalize(&candidate) {
                    Ok(canonical) => resolved = canonical,
                    Err(canonical_error) => match std::fs::symlink_metadata(&candidate) {
                        Ok(metadata) => {
                            let kind = if metadata.file_type().is_symlink() {
                                PolicyPathErrorKind::DanglingSymlink
                            } else {
                                PolicyPathErrorKind::Inaccessible
                            };
                            return Err(PolicyPathError::new(
                                PathOperation::TargetResolution,
                                kind,
                                candidate,
                                Some(canonical_error),
                            ));
                        }
                        Err(metadata_error) if metadata_error.kind() == io::ErrorKind::NotFound => {
                            resolved.push(name);
                        }
                        Err(metadata_error) => {
                            return Err(PolicyPathError::new(
                                PathOperation::TargetResolution,
                                PolicyPathErrorKind::Inaccessible,
                                candidate,
                                Some(metadata_error),
                            ));
                        }
                    },
                }
            }
        }
    }

    if resolved.as_os_str().is_empty() {
        Err(PolicyPathError::new(
            PathOperation::TargetResolution,
            PolicyPathErrorKind::InvalidPath,
            absolute,
            None,
        ))
    } else {
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "shaltaiboltai-policy-{label}-{}-{sequence}",
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
            let expected_prefix = "shaltaiboltai-policy-";
            let is_ours = self
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(expected_prefix));
            if is_ours {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }

    fn workspace_fixture(label: &str) -> (TestDirectory, Workspace) {
        let directory = TestDirectory::new(label);
        let workspace = Workspace::new(directory.path()).expect("valid workspace");
        (directory, workspace)
    }

    #[test]
    fn policy_values_parse_and_display_with_upstream_spellings() {
        assert_eq!(SandboxMode::default(), SandboxMode::WorkspaceWrite);
        assert_eq!(ApprovalPolicy::default(), ApprovalPolicy::OnRequest);
        assert_eq!(
            PermissionPreset::default(),
            PermissionPreset::AskForApproval
        );

        for (text, value) in [
            ("read-only", SandboxMode::ReadOnly),
            ("workspace-write", SandboxMode::WorkspaceWrite),
            ("danger-full-access", SandboxMode::DangerFullAccess),
        ] {
            assert_eq!(text.parse::<SandboxMode>(), Ok(value));
            assert_eq!(value.to_string(), text);
        }
        for (text, value) in [
            ("untrusted", ApprovalPolicy::Untrusted),
            ("on-request", ApprovalPolicy::OnRequest),
            ("never", ApprovalPolicy::Never),
        ] {
            assert_eq!(text.parse::<ApprovalPolicy>(), Ok(value));
            assert_eq!(value.to_string(), text);
        }
        assert_eq!("auto".parse(), Ok(PermissionPreset::AskForApproval));
        assert_eq!("default".parse(), Ok(PermissionPreset::AskForApproval));
        assert!("on-failure".parse::<ApprovalPolicy>().is_err());
        assert!("everything".parse::<SandboxMode>().is_err());
    }

    #[test]
    fn presets_have_exact_user_facing_copy_and_authority() {
        assert_eq!(PermissionPreset::ReadOnly.label(), "Read Only");
        assert_eq!(
            PermissionPreset::ReadOnly.description(),
            "Read files. Ask before edits or network access."
        );
        assert_eq!(PermissionPreset::AskForApproval.label(), "Ask for approval");
        assert_eq!(
            PermissionPreset::AskForApproval.description(),
            "Edit and run sandboxed commands in the workspace. Ask before network or outside writes."
        );
        assert_eq!(PermissionPreset::FullAccess.label(), "Full Access");
        assert_eq!(
            PermissionPreset::FullAccess.description(),
            "Edit any file and use the network without asking. This can expose or delete data."
        );
        assert_eq!(
            PermissionPreset::ReadOnly.sandbox_mode(),
            SandboxMode::ReadOnly
        );
        assert_eq!(
            PermissionPreset::AskForApproval.approval_policy(),
            ApprovalPolicy::OnRequest
        );
        assert_eq!(
            PermissionPreset::FullAccess.approval_policy(),
            ApprovalPolicy::Never
        );
    }

    #[test]
    fn workspace_canonicalizes_relative_extra_roots_and_deduplicates() {
        let directory = TestDirectory::new("roots");
        fs::create_dir(directory.path().join("extra")).expect("create extra root");
        let workspace = Workspace::from_roots(
            directory.path(),
            [PathBuf::from("extra"), directory.path().join("extra")],
        )
        .expect("canonical workspace");
        let expected_cwd = fs::canonicalize(directory.path()).expect("canonical cwd");
        let expected_extra =
            fs::canonicalize(directory.path().join("extra")).expect("canonical extra root");
        assert_eq!(workspace.cwd(), expected_cwd);
        assert_eq!(
            workspace.effective_user_visible_roots(),
            &[expected_cwd, expected_extra]
        );
    }

    #[test]
    fn workspace_rejects_missing_and_non_directory_roots() {
        let directory = TestDirectory::new("bad-roots");
        let missing = Workspace::new(directory.path().join("missing"))
            .expect_err("missing cwd must fail closed");
        assert_eq!(missing.kind(), PolicyPathErrorKind::NotFound);

        let file = directory.path().join("file");
        fs::write(&file, "not a directory").expect("create file");
        let workspace = Workspace::new(directory.path()).expect("valid cwd");
        let error = workspace
            .with_extra_root(&file)
            .expect_err("file root must fail closed");
        assert_eq!(error.kind(), PolicyPathErrorKind::NotDirectory);
    }

    #[test]
    fn prospective_targets_keep_missing_suffixes() {
        let (directory, workspace) = workspace_fixture("missing-suffix");
        fs::create_dir(directory.path().join("existing")).expect("create parent");
        let policy = ExecutionPolicy::new(workspace);
        let resolved = policy
            .canonical_target("existing/new/deep/file.rs")
            .expect("resolve prospective target");
        assert_eq!(
            resolved,
            fs::canonicalize(directory.path().join("existing"))
                .expect("canonical parent")
                .join("new/deep/file.rs")
        );
    }

    #[test]
    fn prospective_parent_components_cannot_escape_policy_boundaries() {
        let (_directory, workspace) = workspace_fixture("missing-parent-components");
        let policy = ExecutionPolicy::new(workspace);

        let escaped = policy
            .classify_write("missing/../../escaped.txt")
            .expect("resolve prospective escape");
        assert_eq!(escaped.boundary(), PathBoundary::OutsideWorkspace);
        assert_eq!(escaped.action(), BoundaryAction::RequiresApproval);
        assert_eq!(
            escaped.target(),
            policy
                .workspace()
                .cwd()
                .parent()
                .expect("workspace has a parent")
                .join("escaped.txt")
        );

        let protected = policy
            .classify_write("missing/../.git/config")
            .expect("resolve protected metadata alias");
        assert_eq!(protected.boundary(), PathBoundary::ProtectedMetadata);
        assert_eq!(protected.action(), BoundaryAction::RequiresApproval);
        assert_eq!(
            protected.target(),
            policy.workspace().cwd().join(".git/config")
        );
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_targets_fail_closed() {
        use std::os::unix::fs::symlink;

        let (directory, workspace) = workspace_fixture("dangling");
        symlink("missing-destination", directory.path().join("dangling"))
            .expect("create dangling symlink");
        let policy = ExecutionPolicy::new(workspace);
        for target in ["dangling", "dangling/child"] {
            let error = policy
                .canonical_target(target)
                .expect_err("dangling symlink must fail closed");
            assert_eq!(error.kind(), PolicyPathErrorKind::DanglingSymlink);
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_targets_are_classified_by_their_real_location() {
        use std::os::unix::fs::symlink;

        let (workspace_directory, workspace) = workspace_fixture("symlink-workspace");
        let outside = TestDirectory::new("symlink-outside");
        symlink(outside.path(), workspace_directory.path().join("escape"))
            .expect("create escape symlink");
        let policy = ExecutionPolicy::new(workspace);
        let read = policy
            .classify_read("escape/prospective.txt")
            .expect("resolve symlinked target");
        assert_eq!(read.boundary(), PathBoundary::OutsideWorkspace);
        assert_eq!(read.action(), BoundaryAction::Allow);
        assert!(read
            .target()
            .starts_with(fs::canonicalize(outside.path()).expect("canonical outside root")));
    }

    #[test]
    fn protected_metadata_is_top_level_per_workspace_root() {
        let (directory, workspace) = workspace_fixture("protected");
        fs::create_dir(directory.path().join("src")).expect("create source directory");
        let policy = ExecutionPolicy::new(workspace);
        for target in [".git/config", ".agents/policy.toml", ".codex/config.toml"] {
            assert!(policy
                .is_protected_metadata(target)
                .expect("classify target"));
            let write = policy.classify_write(target).expect("classify write");
            assert_eq!(write.boundary(), PathBoundary::ProtectedMetadata);
            assert_eq!(write.action(), BoundaryAction::RequiresApproval);
        }
        assert!(!policy
            .is_protected_metadata("src/.git/config")
            .expect("classify nested metadata"));
    }

    #[cfg(any(target_os = "macos", windows))]
    #[test]
    fn protected_metadata_names_are_ascii_case_insensitive() {
        let (_directory, workspace) = workspace_fixture("protected-case-fold");
        let policy = ExecutionPolicy::new(workspace);

        for target in [".GIT/config", ".Agents/policy.toml", ".CODEX/config.toml"] {
            let write = policy
                .classify_write(target)
                .expect("classify alternate-case metadata");
            assert_eq!(write.boundary(), PathBoundary::ProtectedMetadata);
            assert_eq!(write.action(), BoundaryAction::RequiresApproval);
        }
    }

    #[test]
    fn git_pointer_and_common_directory_are_protected_at_their_real_locations() {
        let workspace_directory = TestDirectory::new("linked-worktree");
        let git_storage = TestDirectory::new("linked-git-storage");
        let git_dir = git_storage.path().join("worktrees/linked");
        let common_dir = git_storage.path().join("common");
        fs::create_dir_all(&git_dir).expect("create linked git directory");
        fs::create_dir_all(common_dir.join("refs")).expect("create common git directory");
        fs::write(
            workspace_directory.path().join(".git"),
            format!("gitdir: {}\n", git_dir.display()),
        )
        .expect("write gitdir pointer");
        fs::write(git_dir.join("commondir"), "../../common\n").expect("write commondir pointer");

        let policy = ExecutionPolicy::new(
            Workspace::new(workspace_directory.path()).expect("linked workspace"),
        );
        for target in [git_dir.join("config"), common_dir.join("refs/heads/main")] {
            let write = policy
                .classify_write(&target)
                .expect("classify git metadata");
            assert_eq!(write.boundary(), PathBoundary::ProtectedMetadata);
            assert_eq!(write.action(), BoundaryAction::RequiresApproval);
        }
    }

    #[test]
    fn bare_repository_root_is_entirely_protected() {
        let bare = TestDirectory::new("bare-repository");
        fs::write(bare.path().join("HEAD"), "ref: refs/heads/main\n").expect("write HEAD");
        fs::create_dir(bare.path().join("objects")).expect("create objects");
        fs::create_dir(bare.path().join("refs")).expect("create refs");
        let policy = ExecutionPolicy::new(Workspace::new(bare.path()).expect("bare workspace"));

        for target in ["config", "refs/heads/main", "objects/pack/new.pack"] {
            let write = policy
                .classify_write(target)
                .expect("classify bare metadata");
            assert_eq!(write.boundary(), PathBoundary::ProtectedMetadata);
            assert_eq!(write.action(), BoundaryAction::RequiresApproval);
        }
    }

    #[test]
    fn default_policy_allows_workspace_writes_and_restricts_boundaries() {
        let (directory, workspace) = workspace_fixture("default");
        let outside = TestDirectory::new("default-outside");
        let policy = ExecutionPolicy::new(workspace);
        assert_eq!(policy.sandbox_mode(), SandboxMode::WorkspaceWrite);
        assert_eq!(policy.approval_policy(), ApprovalPolicy::OnRequest);
        assert!(!policy.network_access_enabled());
        assert_eq!(policy.network_status_label(), "restricted");
        assert_eq!(policy.status_label(), "Ask for approval");
        assert_eq!(policy.permission_status_label(), "Workspace");
        assert_eq!(policy.approval_status_label(), "Ask for approval");
        assert_eq!(
            policy
                .classify_read(directory.path().join("src/lib.rs"))
                .expect("classify workspace read")
                .action(),
            BoundaryAction::Allow
        );
        assert_eq!(
            policy
                .classify_write(directory.path().join("src/lib.rs"))
                .expect("classify workspace write")
                .action(),
            BoundaryAction::Allow
        );
        assert_eq!(
            policy
                .classify_write(outside.path().join("file"))
                .expect("classify outside write")
                .action(),
            BoundaryAction::RequiresApproval
        );
    }

    #[test]
    fn read_only_and_never_deny_write_escalation() {
        let (_directory, workspace) = workspace_fixture("read-only-never");
        let policy =
            ExecutionPolicy::from_parts(workspace, SandboxMode::ReadOnly, ApprovalPolicy::Never);
        assert_eq!(policy.status_label(), "Custom permissions");
        assert_eq!(
            policy
                .classify_write("new-file")
                .expect("classify write")
                .action(),
            BoundaryAction::Deny
        );
        assert!(!policy.network_access_enabled());
    }

    #[test]
    fn untrusted_requires_approval_for_in_workspace_writes() {
        let (_directory, workspace) = workspace_fixture("untrusted");
        let policy = ExecutionPolicy::from_parts(
            workspace,
            SandboxMode::WorkspaceWrite,
            ApprovalPolicy::Untrusted,
        );
        assert_eq!(
            policy
                .classify_write("src/lib.rs")
                .expect("classify untrusted write")
                .action(),
            BoundaryAction::RequiresApproval
        );
        assert_eq!(
            policy
                .classify_read("src/lib.rs")
                .expect("classify untrusted read")
                .action(),
            BoundaryAction::Allow
        );
    }

    #[test]
    fn full_access_enables_network_and_all_filesystem_boundaries() {
        let (_directory, workspace) = workspace_fixture("full-access");
        let outside = TestDirectory::new("full-access-outside");
        let mut policy = ExecutionPolicy::new(workspace);
        assert!(policy.apply_preset(PermissionPreset::FullAccess));
        assert!(policy.network_access_enabled());
        assert_eq!(policy.network_status_label(), "enabled");
        assert_eq!(policy.status_label(), "Full Access");
        for target in [
            outside.path().join("file"),
            policy.workspace().cwd().join(".git/config"),
        ] {
            assert_eq!(
                policy
                    .classify_write(target)
                    .expect("classify unrestricted write")
                    .action(),
                BoundaryAction::Allow
            );
        }
    }

    #[test]
    fn policy_generation_changes_only_with_effective_authority() {
        let (_directory, workspace) = workspace_fixture("generation");
        let mut policy = ExecutionPolicy::new(workspace);
        assert_eq!(policy.generation(), 1);
        assert!(!policy.apply_preset(PermissionPreset::AskForApproval));
        assert_eq!(policy.generation(), 1);
        assert!(policy.apply_preset(PermissionPreset::ReadOnly));
        assert_eq!(policy.generation(), 2);
        assert!(!policy.update(SandboxMode::ReadOnly, ApprovalPolicy::OnRequest));
        assert_eq!(policy.generation(), 2);
    }

    #[test]
    fn grants_bind_authority_generation_fingerprint_and_scope() {
        let (_directory, workspace) = workspace_fixture("grant");
        let mut policy = ExecutionPolicy::new(workspace.clone());
        let grant = policy.bind_grant(b"run_command\0cargo test".to_vec());
        assert!(policy.accepts_grant(&grant, b"run_command\0cargo test"));
        assert!(!policy.accepts_grant(&grant, b"run_command\0cargo build"));

        let other_authority = ExecutionPolicy::new(workspace);
        assert_eq!(policy.fingerprint(), other_authority.fingerprint());
        assert!(!other_authority.accepts_grant(&grant, grant.scope()));

        policy.apply_preset(PermissionPreset::ReadOnly);
        assert!(!policy.accepts_grant(&grant, grant.scope()));
        assert_ne!(grant.fingerprint(), &policy.fingerprint());
        assert_eq!(grant.generation(), 1);
        assert_eq!(grant.fingerprint().to_string().len(), 16);
    }
}
