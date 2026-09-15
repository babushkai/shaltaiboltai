#![cfg(target_os = "linux")]

use shaltaiboltai::policy::{ApprovalPolicy, ExecutionPolicy, SandboxMode, Workspace};
use shaltaiboltai::sandbox::{
    prepare_shell_command_for_linux_integration_test, PreparedShellCommand,
};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "shaltaiboltai-linux-sandbox-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create isolated Linux sandbox fixture");
        Self { path }
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let is_ours = self
            .path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| name.starts_with("shaltaiboltai-linux-sandbox-"));
        if is_ours {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

struct OutsideFile {
    path: PathBuf,
}

impl OutsideFile {
    fn new() -> Self {
        let home = std::env::var_os("HOME").expect("Linux integration test requires HOME");
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = PathBuf::from(home).join(format!(
            ".shaltaiboltai-linux-sandbox-outside-{}-{sequence}",
            std::process::id()
        ));
        assert!(!path.exists(), "outside fixture must start absent");
        Self { path }
    }
}

impl Drop for OutsideFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn shell_quote(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

async fn run(prepared: PreparedShellCommand) -> std::process::Output {
    let (mut command, cleanup) = prepared.into_tokio_parts();
    let output = command
        .output()
        .await
        .expect("spawn the production Bubblewrap command");
    cleanup
        .cleanup()
        .expect("clean synthetic protected-metadata targets");
    output
}

fn prepare(
    policy: &ExecutionPolicy,
    command: &str,
    application_binary: &Path,
) -> PreparedShellCommand {
    prepare_shell_command_for_linux_integration_test(policy, command, application_binary)
        .expect("prepare the production Linux sandbox")
}

#[tokio::test]
async fn production_linux_boundary_enforces_read_write_metadata_network_and_cleanup() {
    let application_binary = Path::new(env!("CARGO_BIN_EXE_shaltaiboltai"));
    assert!(application_binary.is_absolute());
    assert!(application_binary.is_file());

    let read_only_directory = TestDirectory::new("read-only");
    let read_only = ExecutionPolicy::from_parts(
        Workspace::new(&read_only_directory.path).expect("canonical read-only workspace"),
        SandboxMode::ReadOnly,
        ApprovalPolicy::OnRequest,
    );
    let denied = run(prepare(
        &read_only,
        "printf denied > read-only.txt",
        application_binary,
    ))
    .await;
    assert!(!denied.status.success());
    assert!(!read_only_directory.path.join("read-only.txt").exists());

    let workspace_directory = TestDirectory::new("workspace-write");
    let workspace = ExecutionPolicy::new(
        Workspace::new(&workspace_directory.path).expect("canonical writable workspace"),
    );
    let allowed = run(prepare(
        &workspace,
        "printf '%s' \"$SHALTAIBOLTAI_SANDBOX\" > marker.txt",
        application_binary,
    ))
    .await;
    assert!(allowed.status.success(), "{allowed:?}");
    assert_eq!(
        std::fs::read_to_string(workspace_directory.path.join("marker.txt"))
            .expect("read sandbox marker"),
        "bubblewrap"
    );

    let outside = OutsideFile::new();
    let outside_command = format!("printf denied > {}", shell_quote(&outside.path));
    let denied = run(prepare(&workspace, &outside_command, application_binary)).await;
    assert!(!denied.status.success());
    assert!(!outside.path.exists());

    let protected = run(prepare(
        &workspace,
        "printf denied > .git/config",
        application_binary,
    ))
    .await;
    assert!(!protected.status.success());

    let curl = Path::new("/usr/bin/curl");
    assert!(curl.is_file(), "Linux CI must provide /usr/bin/curl");
    let network = run(prepare(
        &workspace,
        "/usr/bin/curl --connect-timeout 1 --silent https://example.com >/dev/null",
        application_binary,
    ))
    .await;
    assert!(!network.status.success());

    for name in [".git", ".agents", ".codex"] {
        assert!(
            !workspace_directory.path.join(name).exists(),
            "synthetic {name} mount target must be cleaned"
        );
    }

    let python = Path::new("/usr/bin/python3");
    assert!(python.is_file(), "Linux CI must provide /usr/bin/python3");
    let socket_path = workspace_directory.path.join("host.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket_path)
        .expect("bind host filesystem-backed Unix socket");
    let unix_connect = format!(
        "/usr/bin/python3 -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])' {}",
        shell_quote(&socket_path)
    );
    let blocked_connect = run(prepare(&workspace, &unix_connect, application_binary)).await;
    assert!(
        !blocked_connect.status.success(),
        "seccomp must block connects to host Unix sockets: {blocked_connect:?}"
    );
    drop(listener);
    std::fs::remove_file(&socket_path).expect("remove host Unix socket");

    let blocked_socket_syscalls = run(prepare(
        &workspace,
        "/usr/bin/python3 -c 'import ctypes,errno,platform; libc=ctypes.CDLL(None,use_errno=True); socket_nr,connect_nr,sendmsg_nr={\"x86_64\":(41,42,46),\"aarch64\":(198,203,211)}[platform.machine()]; socket_result=libc.syscall(socket_nr,2,1,0); socket_errno=ctypes.get_errno(); connect_result=libc.syscall(connect_nr,-1,0,0); connect_errno=ctypes.get_errno(); sendmsg_result=libc.syscall(sendmsg_nr,-1,0,0); sendmsg_errno=ctypes.get_errno(); raise SystemExit(0 if (socket_result,socket_errno,connect_result,connect_errno,sendmsg_result,sendmsg_errno)==(-1,errno.EPERM,-1,errno.EPERM,-1,errno.EPERM) else 1)'",
        application_binary,
    ))
    .await;
    assert!(
        blocked_socket_syscalls.status.success(),
        "seccomp must reject socket, connect, and sendmsg with EPERM: {blocked_socket_syscalls:?}"
    );

    #[cfg(target_arch = "x86_64")]
    {
        let x32 = run(prepare(
            &workspace,
            "/usr/bin/python3 -c 'import ctypes,errno; libc=ctypes.CDLL(None,use_errno=True); result=libc.syscall(0x40000000 | 39); raise SystemExit(0 if result == -1 and ctypes.get_errno() == errno.EPERM else 1)'",
            application_binary,
        ))
        .await;
        assert!(
            x32.status.success(),
            "seccomp must reject x32 syscall numbers before ABI dispatch: {x32:?}"
        );
    }

    let local_socketpair = run(prepare(
        &workspace,
        "/usr/bin/python3 -c 'import os,socket; a,b=socket.socketpair(); os.write(a.fileno(),b\"ok\"); assert os.read(b.fileno(),2)==b\"ok\"; print(\"socketpair-ok\")'",
        application_binary,
    ))
    .await;
    assert!(local_socketpair.status.success(), "{local_socketpair:?}");
    assert!(String::from_utf8_lossy(&local_socketpair.stdout).contains("socketpair-ok"));

    let git_dir = workspace_directory.path.join("metadata/worktrees/linked");
    let common_dir = workspace_directory.path.join("metadata/common");
    std::fs::create_dir_all(&git_dir).expect("create linked git directory");
    std::fs::create_dir_all(common_dir.join("refs")).expect("create common git directory");
    std::fs::write(
        workspace_directory.path.join(".git"),
        format!("gitdir: {}\n", git_dir.display()),
    )
    .expect("write gitdir pointer");
    std::fs::write(git_dir.join("commondir"), "../../common\n").expect("write commondir pointer");
    std::fs::write(git_dir.join("config"), "original\n").expect("write protected config");
    let linked_git_write = format!("printf changed > {}", shell_quote(&git_dir.join("config")));
    let linked_git_denied = run(prepare(&workspace, &linked_git_write, application_binary)).await;
    assert!(!linked_git_denied.status.success(), "{linked_git_denied:?}");
    assert_eq!(
        std::fs::read_to_string(git_dir.join("config")).unwrap(),
        "original\n"
    );
}
