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
                // Coalesce bursts (e.g. stream deltas) into a single redraw.
                while let Ok(event) = rx.try_recv() {
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
    if key.code == KeyCode::F(1) && app.mode == Mode::Input {
        app.open_help();
        return;
    }
    match app.mode {
        Mode::Input => handle_input_key(app, key),
        Mode::Streaming | Mode::RunningTool => {
            if key.code == KeyCode::Esc {
                app.cancel_request();
            } else {
                handle_scroll_key(app, key);
            }
        }
        Mode::Approval => match approval_decision(&key) {
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
        },
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

fn handle_input_key(app: &mut App, key: KeyEvent) {
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
}
