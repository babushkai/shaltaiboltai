#[cfg(target_os = "linux")]
use shaltaiboltai::sandbox;
use shaltaiboltai::{app, cli, config, images, mascot, policy, ui};

use app::{App, AppEvent, Mode, PermissionOverlay};
use config::Config;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use futures_util::StreamExt;
use policy::{ApprovalPolicy, ExecutionPolicy, SandboxMode, Workspace};
use std::path::{Path, PathBuf};

const HELP: &str = "\
shaltaiboltai — a multi-provider agentic coding TUI

USAGE:
    shaltaiboltai [OPTIONS] [PROMPT]

OPTIONS:
    -m, --model <MODEL>               Override the configured model
    -C, --cd <DIR>                    Set the working directory
        --add-dir <DIR>               Add a writable workspace root (repeatable)
    -s, --sandbox <MODE>              read-only | workspace-write | danger-full-access
    -a, --ask-for-approval <POLICY>   on-request | never
    -i, --image <PATH,...>            Attach startup images (repeatable)
        --full-auto                   workspace-write with on-request approval
        --dangerously-bypass-approvals-and-sandbox
                                      Full disk/network access without approvals
        --no-alt-screen               Keep terminal scrollback visible
    -h, --help                        Print this help and exit
    -V, --version                     Print version and exit

With no options it launches the interactive TUI. Configure providers via
ANTHROPIC_API_KEY / OPENAI_API_KEY / a running Ollama, or a logged-in
`claude` / `codex` CLI for subscription use on Unix. See the README for details.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApprovalDecision {
    ApproveOnce,
    AllowForSession,
    Deny,
}

fn approval_decision(key: &KeyEvent) -> Option<ApprovalDecision> {
    if key.code == KeyCode::Esc {
        return Some(ApprovalDecision::Deny);
    }
    if !key.modifiers.is_empty() {
        return None;
    }
    match key.code {
        KeyCode::Char('y') => Some(ApprovalDecision::ApproveOnce),
        KeyCode::Char('a') => Some(ApprovalDecision::AllowForSession),
        KeyCode::Char('n') => Some(ApprovalDecision::Deny),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Handle non-interactive flags before touching the terminal, so the binary
    // behaves like a normal CLI in pipes and scripts.
    let options = match cli::parse_args(std::env::args_os().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}\n\n{HELP}");
            std::process::exit(2);
        }
    };
    if options.sandbox_seccomp_command.is_some() {
        #[cfg(target_os = "linux")]
        {
            let command = options
                .sandbox_seccomp_command
                .as_deref()
                .expect("checked above");
            sandbox::exec_linux_seccomp_shell(command)?;
            unreachable!("successful sandbox helper execution replaces the process");
        }
        #[cfg(not(target_os = "linux"))]
        anyhow::bail!("the internal seccomp child stage is only available on Linux");
    }
    if options.help {
        println!("{HELP}");
        return Ok(());
    }
    if options.version {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let launch_cwd = std::env::current_dir()?;
    let (execution_policy, warning) = resolve_launch_policy(&options, &launch_cwd)?;
    if let Some(warning) = warning {
        eprintln!("warning: {warning}");
    }
    let startup_images = load_startup_images(&options.images, execution_policy.workspace().cwd())?;
    let mut launch_config = Config::load();
    if let Some(model) = &options.model {
        launch_config.default_model = Some(model.clone());
    }
    let no_alt_screen = options.no_alt_screen;
    let mut terminal = if no_alt_screen {
        let height = crossterm::terminal::size()?.1.max(1);
        ratatui::init_with_options(ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(height),
        })
    } else {
        ratatui::init()
    };
    // Detect direct Ghostty/Kitty sessions after alternate-screen setup and
    // before EventStream starts. Unsupported/error paths keep the deterministic
    // half-block mascot without a blocking terminal capability query.
    let native_mascot = mascot::NativeMascot::detect().ok().flatten();
    let _ = execute!(std::io::stdout(), EnableBracketedPaste, EnableMouseCapture);
    let result = run(
        &mut terminal,
        native_mascot.as_ref(),
        launch_config,
        execution_policy,
        startup_images,
        options.prompt,
    )
    .await;
    // Remove Kitty placeholder cells before leaving the alternate screen.
    // The terminal then releases every virtual placement owned by this UI.
    if let Some(native_mascot) = &native_mascot {
        let _ = native_mascot.clear();
    }
    if !no_alt_screen {
        let _ = terminal.clear();
    }
    let _ = execute!(
        std::io::stdout(),
        DisableMouseCapture,
        DisableBracketedPaste
    );
    if no_alt_screen {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = terminal.show_cursor();
    } else {
        ratatui::restore();
    }
    result
}

fn resolve_launch_policy(
    options: &cli::LaunchOptions,
    launch_cwd: &Path,
) -> anyhow::Result<(ExecutionPolicy, Option<String>)> {
    let selected_cwd = options.cwd.as_ref().map_or_else(
        || launch_cwd.to_path_buf(),
        |cwd| {
            if cwd.is_absolute() {
                cwd.clone()
            } else {
                launch_cwd.join(cwd)
            }
        },
    );
    let (sandbox, approval) = if options.dangerously_bypass_approvals_and_sandbox {
        (SandboxMode::DangerFullAccess, ApprovalPolicy::Never)
    } else if options.full_auto {
        (SandboxMode::WorkspaceWrite, ApprovalPolicy::OnRequest)
    } else {
        let sandbox = match options.sandbox_mode {
            Some(cli::SandboxMode::ReadOnly) => SandboxMode::ReadOnly,
            Some(cli::SandboxMode::WorkspaceWrite) => SandboxMode::WorkspaceWrite,
            Some(cli::SandboxMode::DangerFullAccess) => SandboxMode::DangerFullAccess,
            None => SandboxMode::WorkspaceWrite,
        };
        let approval = match options.approval_policy {
            Some(cli::ApprovalPolicy::OnRequest) | None => ApprovalPolicy::OnRequest,
            Some(cli::ApprovalPolicy::Never) => ApprovalPolicy::Never,
        };
        (sandbox, approval)
    };
    let ignore_additional_dirs =
        !options.additional_writable_dirs.is_empty() && sandbox == SandboxMode::ReadOnly;
    let additional_dirs = if ignore_additional_dirs {
        &[][..]
    } else {
        options.additional_writable_dirs.as_slice()
    };
    let workspace = Workspace::from_roots(&selected_cwd, additional_dirs)?;
    let warning = ignore_additional_dirs
        .then(|| "--add-dir is ignored because the selected sandbox is read-only".to_owned());
    Ok((
        ExecutionPolicy::from_parts(workspace, sandbox, approval),
        warning,
    ))
}

fn load_startup_images(
    paths: &[PathBuf],
    cwd: &Path,
) -> anyhow::Result<Vec<(String, shaltaiboltai::providers::ImageData)>> {
    paths
        .iter()
        .map(|path| {
            let path = if path.is_absolute() {
                path.clone()
            } else {
                cwd.join(path)
            };
            images::load_image(&path).map_err(|error| anyhow::anyhow!("{error:#}"))
        })
        .collect()
}

async fn run(
    terminal: &mut ratatui::DefaultTerminal,
    native_mascot: Option<&mascot::NativeMascot>,
    config: Config,
    execution_policy: ExecutionPolicy,
    startup_images: Vec<(String, shaltaiboltai::providers::ImageData)>,
    initial_prompt: Option<String>,
) -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AppEvent>();
    let mut app = App::with_policy(config, execution_policy, tx);
    if !startup_images.is_empty() {
        let names = startup_images
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        app.pending_images.extend(startup_images);
        app.transcript
            .push(app::Entry::Info(format!("attached: {names}")));
    }
    let mut pending_initial_prompt = initial_prompt;
    if let Some(prompt) = &pending_initial_prompt {
        app.textarea.insert_str(prompt);
    }
    let mut term_events = EventStream::new();
    let mut animation = tokio::time::interval(std::time::Duration::from_millis(120));
    animation.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut environment = tokio::time::interval(std::time::Duration::from_secs(2));
    environment.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Tokio intervals fire immediately once. Consume that initial pulse so a
    // new working state starts at the first authored mascot pose.
    animation.tick().await;
    environment.tick().await;

    while !app.should_quit {
        submit_initial_prompt_when_ready(&mut app, &mut pending_initial_prompt);
        terminal.draw(|frame| {
            if let Some(native_mascot) = native_mascot {
                ui::draw_with_native_mascot(frame, &mut app, native_mascot);
            } else {
                ui::draw(frame, &mut app);
            }
        })?;

        tokio::select! {
            Some(event) = rx.recv() => {
                app.on_event(event);
                // Coalesce a bounded burst into one redraw, then yield back to
                // terminal input. An unbounded high-rate stream must not starve
                // Esc/Ctrl+C or type-ahead key handling.
                for _ in 0..255 {
                    let Ok(event) = rx.try_recv() else {
                        break;
                    };
                    app.on_event(event);
                }
            }
            Some(Ok(event)) = term_events.next() => match event {
                Event::Key(key) if key.kind == KeyEventKind::Press => handle_key(&mut app, key),
                Event::Mouse(mouse) => handle_mouse(&mut app, mouse),
                Event::Paste(text) => app.paste(&text),
                _ => {}
            },
            // A persistent clock keeps the mascot moving even during dense
            // provider streams; recreating sleeps on every event can starve it.
            _ = animation.tick(), if app.needs_animation() => app.advance_animation(),
            // Idle: pick up external changes (e.g. a branch switch in another
            // terminal) for the statusline.
            _ = environment.tick(), if !app.is_busy() => {
                app.refresh_environment();
            }
        }
    }
    app.save_session_for_exit()?;
    Ok(())
}

fn submit_initial_prompt_when_ready(app: &mut App, pending: &mut Option<String>) {
    let Some(expected) = pending.as_deref() else {
        return;
    };
    let current = app.textarea.lines().join("\n");
    if current != expected {
        *pending = None;
        return;
    }
    if expected.trim().is_empty() {
        *pending = None;
        return;
    }
    if expected.trim_start().starts_with('/') || app.model.is_some() {
        app.submit_input();
        *pending = None;
    }
}

fn handle_key(app: &mut App, key: KeyEvent) {
    // Global bindings, regardless of mode.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        app.request_quit();
        return;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::End {
        app.scroll_from_bottom = 0;
        return;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Home {
        app.scroll_from_bottom = usize::MAX;
        return;
    }
    if key.code == KeyCode::Esc
        && app.compacting
        && app.mode == Mode::Input
        && !app.slash_menu_active()
    {
        app.cancel_compaction_request();
        return;
    }
    if key.code == KeyCode::F(1)
        && app.mode == Mode::Input
        && !app.compacting
        && app.composer_accepts_input()
    {
        app.open_help();
        return;
    }
    if app.permission_overlay.is_some() {
        handle_permission_key(app, key);
        return;
    }
    match app.mode {
        Mode::Input => handle_input_key(app, key),
        Mode::Streaming | Mode::RunningTool => handle_active_key(app, key),
        Mode::Approval => handle_approval_key(app, key),
        Mode::OrchestrationConfirm => handle_orchestration_confirm_key(app, key),
        Mode::Orchestrating => handle_orchestrating_key(app, key),
        Mode::ModelPicker => handle_model_picker_key(app, key),
        Mode::SessionPicker => handle_session_picker_key(app, key),
        Mode::ThemePicker => match key.code {
            KeyCode::Esc => app.revert_theme(),
            KeyCode::Enter => app.pick_theme(),
            KeyCode::Up => app.theme_move(-1),
            KeyCode::Down => app.theme_move(1),
            _ => {}
        },
        Mode::Help => match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::F(1) => app.close_help(),
            _ => {}
        },
    }
}

fn handle_orchestration_confirm_key(app: &mut App, key: KeyEvent) {
    if key.code == KeyCode::Esc || (key.modifiers.is_empty() && key.code == KeyCode::Char('n')) {
        app.cancel_orchestration();
        return;
    }
    if !key.modifiers.is_empty() {
        return;
    }
    match key.code {
        KeyCode::Tab => app.toggle_orchestration_confirm_focus(),
        KeyCode::Enter | KeyCode::Char('y') if app.orchestration_confirm_focused => {
            app.confirm_orchestration();
        }
        _ => {}
    }
}

fn handle_orchestrating_key(app: &mut App, key: KeyEvent) {
    if key.code == KeyCode::Esc {
        app.cancel_orchestration();
        return;
    }
    if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) {
        handle_scroll_key(app, key);
        return;
    }
    // Only the concurrent worker phase exposes the existing one-slot
    // lookahead composer. Planning and coordination keep it locked so a failed
    // root prompt can still be restored without merging two drafts.
    if app.composer_accepts_input() {
        handle_lookahead_composer_key(app, key);
    }
}

fn handle_active_key(app: &mut App, key: KeyEvent) {
    if key.code == KeyCode::Esc {
        app.cancel_request();
        return;
    }
    if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) {
        handle_scroll_key(app, key);
        return;
    }
    if key.code == KeyCode::Enter && app.submit_active_local_command() {
        return;
    }
    handle_lookahead_composer_key(app, key);
}

fn handle_permission_key(app: &mut App, key: KeyEvent) {
    match app.permission_overlay {
        Some(PermissionOverlay::Picker) => match key.code {
            KeyCode::Esc => app.close_permissions(),
            KeyCode::Enter => app.select_permission(),
            KeyCode::Up => app.permission_move(-1),
            KeyCode::Down => app.permission_move(1),
            _ => {}
        },
        Some(PermissionOverlay::FullAccessConfirm) => match key.code {
            KeyCode::Esc => app.cancel_full_access_confirmation(),
            KeyCode::Up | KeyCode::Left => app.move_full_access_confirmation(-1),
            KeyCode::Down | KeyCode::Right => app.move_full_access_confirmation(1),
            KeyCode::Enter => app.activate_full_access_confirmation(),
            _ => {}
        },
        None => {}
    }
}

fn handle_approval_key(app: &mut App, key: KeyEvent) {
    if !app.approval_focused {
        match key.code {
            // The first Esc only leaves type-ahead focus. A second Esc, now in
            // approval focus, performs the existing deny action.
            KeyCode::Tab | KeyCode::Esc => app.focus_approval(),
            KeyCode::PageUp => app.approval_scroll = app.approval_scroll.saturating_sub(8),
            KeyCode::PageDown => app.approval_scroll = app.approval_scroll.saturating_add(8),
            _ => handle_lookahead_composer_key(app, key),
        }
        return;
    }

    if key.code == KeyCode::Tab {
        app.toggle_approval_focus();
        return;
    }
    match approval_decision(&key) {
        Some(ApprovalDecision::ApproveOnce) => app.approve_pending(false),
        Some(ApprovalDecision::AllowForSession) => app.approve_pending(true),
        Some(ApprovalDecision::Deny) => app.deny_pending(),
        None => match key.code {
            KeyCode::Up => app.approval_scroll = app.approval_scroll.saturating_sub(1),
            KeyCode::Down => app.approval_scroll = app.approval_scroll.saturating_add(1),
            KeyCode::PageUp => app.approval_scroll = app.approval_scroll.saturating_sub(8),
            KeyCode::PageDown => app.approval_scroll = app.approval_scroll.saturating_add(8),
            KeyCode::Home => app.approval_scroll = 0,
            KeyCode::End => app.approval_scroll = usize::MAX,
            _ => {}
        },
    }
}

fn handle_lookahead_composer_key(app: &mut App, key: KeyEvent) {
    if !app.composer_accepts_input() {
        return;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('v') => {
                app.attach_clipboard_image();
                return;
            }
            KeyCode::Char('x') => {
                app.clear_attachments();
                return;
            }
            KeyCode::Char('u') => {
                app.clear_input();
                app.note_input_changed();
                return;
            }
            _ => {}
        }
    }
    match key.code {
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
            app.textarea.insert_newline();
            app.note_input_changed();
        }
        KeyCode::Enter => app.queue_input(),
        _ => {
            app.textarea.input(key);
            if matches!(
                key.code,
                KeyCode::Char(_) | KeyCode::Backspace | KeyCode::Delete
            ) {
                app.note_input_changed();
            }
        }
    }
}

fn handle_input_key(app: &mut App, key: KeyEvent) {
    if !app.composer_accepts_input() {
        if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) {
            handle_scroll_key(app, key);
        }
        return;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('p') => {
                app.open_picker();
                return;
            }
            KeyCode::Char('v') => {
                app.attach_clipboard_image();
                return;
            }
            KeyCode::Char('x') => {
                app.clear_attachments();
                return;
            }
            // Shell-style line kill; clears the whole input rather than
            // tui-textarea's default delete-to-line-head.
            KeyCode::Char('u') => {
                app.clear_input();
                app.note_input_changed();
                return;
            }
            _ => {}
        }
    }
    let menu = app.slash_menu_active();
    match key.code {
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
            app.textarea.insert_newline();
            app.note_input_changed();
        }
        // The `/` completion menu captures navigation while open.
        KeyCode::Enter if menu => app.run_selected_slash(),
        KeyCode::Tab if menu => app.complete_selected_slash(),
        KeyCode::Up if menu => app.slash_move(-1),
        KeyCode::Down if menu => app.slash_move(1),
        KeyCode::Esc if menu => app.dismiss_slash_menu(),
        KeyCode::Esc => app.clear_attachments(),
        KeyCode::Enter => app.submit_input(),
        // Shell-style prompt recall when the input is empty (or while already
        // navigating history); otherwise Up/Down move the cursor in the editor.
        KeyCode::Up if app.input_is_empty() || app.history_recall_active() => {
            app.input_history_prev();
        }
        KeyCode::Down if app.history_recall_active() => {
            app.input_history_next();
        }
        KeyCode::PageUp | KeyCode::PageDown => handle_scroll_key(app, key),
        _ => {
            app.textarea.input(key);
            if matches!(
                key.code,
                KeyCode::Char(_) | KeyCode::Backspace | KeyCode::Delete
            ) {
                app.note_input_changed();
            }
        }
    }
}

/// Wheel/trackpad scrolling, in any mode. Mouse capture trades away the
/// terminal's native click-drag selection — hold Shift (Linux/Windows) or
/// Option (macOS) to select text while the TUI is running.
///
/// One line per event: trackpads emit a dense, velocity-scaled event stream,
/// so larger steps multiply the speed and feel chunky rather than faster.
fn handle_mouse(app: &mut App, mouse: MouseEvent) {
    if app.permission_overlay == Some(PermissionOverlay::Picker) {
        match mouse.kind {
            MouseEventKind::ScrollUp => app.permission_move(-1),
            MouseEventKind::ScrollDown => app.permission_move(1),
            _ => {}
        }
        return;
    }
    if app.mode == Mode::Approval {
        match mouse.kind {
            MouseEventKind::ScrollUp => app.approval_scroll = app.approval_scroll.saturating_sub(1),
            MouseEventKind::ScrollDown => {
                app.approval_scroll = app.approval_scroll.saturating_add(1)
            }
            _ => {}
        }
        return;
    }
    match mouse.kind {
        MouseEventKind::ScrollUp => app.scroll_from_bottom += 1,
        MouseEventKind::ScrollDown => {
            app.scroll_from_bottom = app.scroll_from_bottom.saturating_sub(1)
        }
        _ => {}
    }
}

fn handle_scroll_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::PageUp => app.scroll_from_bottom += 10,
        KeyCode::PageDown => app.scroll_from_bottom = app.scroll_from_bottom.saturating_sub(10),
        _ => {}
    }
}

fn handle_model_picker_key(app: &mut App, key: KeyEvent) {
    let count = app.filtered_models().len();
    match key.code {
        KeyCode::Esc => app.mode = Mode::Input,
        KeyCode::Enter => app.pick_model(),
        KeyCode::Up => app.picker_index = app.picker_index.saturating_sub(1),
        KeyCode::Down => {
            if count > 0 {
                app.picker_index = (app.picker_index + 1).min(count - 1);
            }
        }
        KeyCode::Backspace => {
            app.picker_filter.pop();
            app.picker_index = 0;
        }
        KeyCode::Char(c) => {
            app.picker_filter.push(c);
            app.picker_index = 0;
        }
        _ => {}
    }
}

fn handle_session_picker_key(app: &mut App, key: KeyEvent) {
    let count = app.sessions.len();
    match key.code {
        KeyCode::Esc => app.mode = Mode::Input,
        KeyCode::Enter => app.pick_session(),
        KeyCode::Up => app.session_index = app.session_index.saturating_sub(1),
        KeyCode::Down if count > 0 => {
            app.session_index = (app.session_index + 1).min(count - 1);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shaltaiboltai::orchestration::PlannedTask;
    use shaltaiboltai::providers::{ChatEvent, ImageData, ModelEntry, ProviderKind, ToolCall};
    use tokio::sync::mpsc::unbounded_channel;

    fn test_app() -> App {
        let data_dir = std::env::temp_dir().join(format!("shaltai-main-{}", std::process::id()));
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", data_dir);
        let config = Config {
            anthropic_api_key: None,
            openai_api_key: None,
            openai_base_url: "http://127.0.0.1:9".into(),
            ollama_host: "http://127.0.0.1:9".into(),
            default_model: None,
            compact_threshold_chars: 80_000,
            ollama_num_ctx: 16_384,
            theme: None,
            reduced_motion: false,
        };
        let (tx, _rx) = unbounded_channel();
        let mut app = App::new(config, tx);
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "key-test".into(),
        });
        app
    }

    fn show_test_team_confirmation(app: &mut App) {
        app.textarea.insert_str("/team 2");
        app.submit_input();
        app.textarea.insert_str("coordinate this safely");
        app.submit_input();
        let run_id = app
            .orchestration_run_id()
            .expect("team planning should own a run id");
        let model = app.model.clone().expect("test model");
        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![
                PlannedTask {
                    id: 1,
                    title: "inspect state".into(),
                    instructions: "read relevant files".into(),
                    model: model.clone(),
                },
                PlannedTask {
                    id: 2,
                    title: "review risks".into(),
                    instructions: "identify safety gaps".into(),
                    model,
                },
            ]),
        });
        assert_eq!(app.mode, Mode::OrchestrationConfirm);
    }

    #[test]
    fn approval_shortcuts_reject_modified_keys() {
        assert_eq!(
            approval_decision(&KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL)),
            None
        );
        assert_eq!(
            approval_decision(&KeyEvent::new(KeyCode::Char('y'), KeyModifiers::ALT)),
            None
        );
        assert_eq!(
            approval_decision(&KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE)),
            Some(ApprovalDecision::AllowForSession)
        );
        assert_eq!(
            approval_decision(&KeyEvent::new(KeyCode::Esc, KeyModifiers::CONTROL)),
            Some(ApprovalDecision::Deny)
        );
    }

    #[test]
    fn read_only_additional_directories_are_ignored_with_a_warning() {
        let workspace = std::env::temp_dir().join(format!(
            "shaltai-read-only-cli-{}",
            shaltaiboltai::session::new_id()
        ));
        std::fs::create_dir_all(&workspace).expect("test workspace");
        let options = cli::parse_args(["--sandbox", "read-only", "--add-dir", "missing-directory"])
            .expect("valid CLI options");

        let (policy, warning) =
            resolve_launch_policy(&options, &workspace).expect("read-only launch policy");

        assert_eq!(policy.sandbox_mode(), SandboxMode::ReadOnly);
        assert_eq!(policy.workspace().effective_user_visible_roots().len(), 1);
        assert!(warning
            .as_deref()
            .is_some_and(|warning| warning.contains("--add-dir") && warning.contains("ignored")));
        std::fs::remove_dir_all(workspace).ok();
    }

    #[tokio::test]
    async fn positional_prompt_waits_for_a_model_then_submits_once() {
        let mut app = test_app();
        app.model = None;
        app.textarea.insert_str("inspect this workspace");
        let mut pending = Some("inspect this workspace".to_owned());

        submit_initial_prompt_when_ready(&mut app, &mut pending);
        assert!(pending.is_some());
        assert_eq!(app.textarea.lines().join("\n"), "inspect this workspace");

        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "startup-prompt-test".into(),
        });
        submit_initial_prompt_when_ready(&mut app, &mut pending);
        assert!(pending.is_none());
        assert!(app.input_is_empty());
        assert!(app.history.iter().any(|message| matches!(
            message,
            shaltaiboltai::providers::Message::User(content)
                if content.text() == "inspect this workspace"
        )));
        app.cancel_request();
    }

    #[tokio::test]
    async fn team_plan_cannot_start_until_tab_arms_confirmation() {
        let mut app = test_app();
        show_test_team_confirmation(&mut app);

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
        );
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.mode, Mode::OrchestrationConfirm);
        assert!(!app.orchestration_confirm_focused);

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert!(app.orchestration_confirm_focused);
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.mode, Mode::Orchestrating);

        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(app.mode, Mode::Input);
    }

    #[tokio::test]
    async fn unfocused_team_plan_can_always_be_cancelled_safely() {
        for key in [KeyCode::Char('n'), KeyCode::Esc] {
            let mut app = test_app();
            show_test_team_confirmation(&mut app);
            handle_key(&mut app, KeyEvent::new(key, KeyModifiers::NONE));

            assert_eq!(app.mode, Mode::Input);
            assert_eq!(app.textarea.lines().join("\n"), "coordinate this safely");
        }
    }

    #[tokio::test]
    async fn active_turn_keys_edit_and_queue_the_next_message() {
        let mut app = test_app();
        app.mode = Mode::Streaming;
        for ch in "next request".chars() {
            handle_key(
                &mut app,
                KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
            );
        }
        assert_eq!(app.textarea.lines().join("\n"), "next request");

        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.queued_prompt_count(), 1);
        assert!(app.input_is_empty());
        assert_eq!(app.mode, Mode::Streaming);
    }

    #[tokio::test]
    async fn first_quit_restores_a_queued_message_and_its_attachments() {
        let mut app = test_app();
        app.mode = Mode::Streaming;
        app.pending_images.push((
            "queued.png".into(),
            ImageData {
                media_type: "image/png".into(),
                data: "cXVldWVk".into(),
            },
        ));
        app.textarea.insert_str("do not lose this");
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        );
        assert!(!app.should_quit);
        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.textarea.lines().join("\n"), "do not lose this");
        assert_eq!(app.pending_image_count(), 1);
        assert!(app
            .composer_notice()
            .is_some_and(|notice| notice.contains("Ctrl+C again")));

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        );
        assert!(app.should_quit);
    }

    #[tokio::test]
    async fn active_composer_can_clear_an_attachment_without_cancelling() {
        let mut app = test_app();
        app.mode = Mode::Streaming;
        app.pending_images.push((
            "wrong.png".into(),
            ImageData {
                media_type: "image/png".into(),
                data: "d3Jvbmc=".into(),
            },
        ));
        app.textarea.insert_str("keep this text");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
        );

        assert_eq!(app.mode, Mode::Streaming);
        assert_eq!(app.pending_image_count(), 0);
        assert_eq!(app.textarea.lines().join("\n"), "keep this text");
    }

    #[tokio::test]
    async fn occupied_compaction_lookahead_locks_a_second_draft() {
        let mut app = test_app();
        app.compacting = true;
        for ch in "after compaction".chars() {
            handle_key(
                &mut app,
                KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
            );
        }
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.queued_prompt_count(), 1);
        assert!(app.input_is_empty());

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
        );
        assert!(app.input_is_empty());

        handle_key(&mut app, KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE));
        assert_eq!(app.mode, Mode::Input, "Help must not strand the queue");
    }

    #[tokio::test]
    async fn overlay_escape_closes_before_compaction_is_cancelled() {
        let mut app = test_app();
        app.compacting = true;
        app.open_help();

        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(app.mode, Mode::Input);
        assert!(app.compacting);

        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.compacting);
    }

    #[tokio::test]
    async fn escape_cancels_compaction_and_restores_its_lookahead() {
        let mut app = test_app();
        app.compacting = true;
        app.textarea.insert_str("after compaction");
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.queued_prompt_count(), 1);

        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.compacting);
        assert_eq!(app.queued_prompt_count(), 0);
        assert_eq!(app.textarea.lines().join("\n"), "after compaction");

        app.compacting = true;
        app.clear_input();
        app.textarea.insert_str("unsent draft");
        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.compacting);
        assert_eq!(app.textarea.lines().join("\n"), "unsent draft");
    }

    #[tokio::test]
    async fn newly_arrived_approval_cannot_consume_typed_decision_letters() {
        let mut app = test_app();
        app.policy.apply_preset(policy::PermissionPreset::ReadOnly);
        app.mode = Mode::Streaming;
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
        );
        app.on_event(AppEvent::Chat {
            gen: 0,
            event: ChatEvent::Completed {
                tool_calls: vec![ToolCall {
                    id: "focus-call".into(),
                    name: "write_file".into(),
                    arguments: serde_json::json!({"path": "x.txt", "content": "x"}),
                }],
                stop_reason: Some("tool_calls".into()),
                usage: None,
            },
        });
        assert_eq!(app.mode, Mode::Approval);
        assert!(!app.approval_focused);

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
        );
        assert_eq!(app.textarea.lines().join("\n"), "ay");
        assert!(app.pending_approval().is_some());

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
        );
        assert!(app.pending_approval().is_none());
        app.cancel_request();
    }

    #[tokio::test]
    async fn escape_denies_an_approval_without_releasing_the_queued_prompt() {
        let mut app = test_app();
        app.policy.apply_preset(policy::PermissionPreset::ReadOnly);
        app.mode = Mode::Streaming;
        app.textarea.insert_str("next request");
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        app.on_event(AppEvent::Chat {
            gen: 0,
            event: ChatEvent::Completed {
                tool_calls: vec![ToolCall {
                    id: "queued-focus-call".into(),
                    name: "write_file".into(),
                    arguments: serde_json::json!({"path": "x.txt", "content": "x"}),
                }],
                stop_reason: Some("tool_calls".into()),
                usage: None,
            },
        });
        assert!(!app.approval_focused);

        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.approval_focused);
        assert!(app.pending_approval().is_some());
        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.pending_approval().is_none());
        assert_eq!(app.queued_prompt_count(), 1);
        assert_eq!(app.mode, Mode::Streaming);
        app.cancel_request();
    }

    #[tokio::test]
    async fn full_access_requires_an_explicit_destructive_selection() {
        let mut app = test_app();
        app.open_permissions();

        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            app.permission_overlay,
            Some(PermissionOverlay::FullAccessConfirm)
        );
        assert!(!app.full_access_enable_selected);

        // Enter activates the preselected safe action and cannot silently
        // broaden authority.
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.permission_overlay, Some(PermissionOverlay::Picker));
        assert_ne!(
            app.policy.matching_preset(),
            Some(policy::PermissionPreset::FullAccess)
        );

        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert!(app.full_access_enable_selected);
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            app.policy.matching_preset(),
            Some(policy::PermissionPreset::FullAccess)
        );
        assert!(app.permission_overlay.is_none());
    }
}
