use shaltaiboltai::{app, config, ui};

use app::{App, AppEvent, Mode};
use config::Config;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use futures_util::StreamExt;

const HELP: &str = "\
shaltaiboltai — a multi-provider agentic coding TUI

USAGE:
    shaltaiboltai [OPTIONS]

OPTIONS:
    -h, --help       Print this help and exit
    -V, --version    Print version and exit

With no options it launches the interactive TUI. Configure providers via
ANTHROPIC_API_KEY / OPENAI_API_KEY / a running Ollama, or a logged-in
`claude` / `codex` CLI for subscription use. See the README for details.";

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{HELP}");
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if let Some(unknown) = args.first() {
        eprintln!("error: unrecognized argument `{unknown}`\n\n{HELP}");
        std::process::exit(2);
    }

    let mut terminal = ratatui::init();
    let _ = execute!(std::io::stdout(), EnableBracketedPaste, EnableMouseCapture);
    let result = run(&mut terminal).await;
    let _ = execute!(
        std::io::stdout(),
        DisableMouseCapture,
        DisableBracketedPaste
    );
    ratatui::restore();
    result
}

async fn run(terminal: &mut ratatui::DefaultTerminal) -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AppEvent>();
    let mut app = App::new(Config::load(), tx);
    let mut term_events = EventStream::new();

    while !app.should_quit {
        terminal.draw(|frame| ui::draw(frame, &mut app))?;

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
            // Keep the status-bar spinner animating while the agent works.
            _ = tokio::time::sleep(std::time::Duration::from_millis(120)), if app.is_busy() => {}
            // Idle: pick up external changes (e.g. a branch switch in another
            // terminal) for the statusline.
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)), if !app.is_busy() => {
                app.refresh_environment();
            }
        }
    }
    app.save_session_for_exit()?;
    Ok(())
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
    match app.mode {
        Mode::Input => handle_input_key(app, key),
        Mode::Streaming | Mode::RunningTool => handle_active_key(app, key),
        Mode::Approval => handle_approval_key(app, key),
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

fn handle_active_key(app: &mut App, key: KeyEvent) {
    if key.code == KeyCode::Esc {
        app.cancel_request();
        return;
    }
    if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) {
        handle_scroll_key(app, key);
        return;
    }
    handle_lookahead_composer_key(app, key);
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
            claude_code_bypass_permissions: false,
            codex_full_access: false,
        };
        let (tx, _rx) = unbounded_channel();
        let mut app = App::new(config, tx);
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "key-test".into(),
        });
        app
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
}
