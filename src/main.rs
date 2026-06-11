use shaltaiboltai::{app, config, ui};

use app::{App, AppEvent, Mode};
use config::Config;
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEvent,
    KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let _ = execute!(std::io::stdout(), EnableBracketedPaste);
    let result = run(&mut terminal).await;
    let _ = execute!(std::io::stdout(), DisableBracketedPaste);
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
    app.save_session();
    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) {
    // Global bindings, regardless of mode.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        app.should_quit = true;
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
        Mode::Approval => match key.code {
            KeyCode::Char('y') | KeyCode::Enter => app.approve_pending(false),
            KeyCode::Char('a') => app.approve_pending(true),
            KeyCode::Char('n') | KeyCode::Esc => app.deny_pending(),
            _ => handle_scroll_key(app, key),
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
        KeyCode::Down => {
            if count > 0 {
                app.session_index = (app.session_index + 1).min(count - 1);
            }
        }
        _ => {}
    }
}
