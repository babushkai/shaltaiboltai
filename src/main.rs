use shaltaiboltai::{app, config, ui};

use app::{App, AppEvent, Mode};
use config::Config;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = run(&mut terminal).await;
    ratatui::restore();
    result
}

async fn run(terminal: &mut ratatui::DefaultTerminal) -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AppEvent>();
    let mut app = App::new(Config::load(), tx);
    let mut term_events = EventStream::new();

    while !app.should_quit {
        terminal.draw(|frame| ui::draw(frame, &app))?;

        tokio::select! {
            Some(event) = rx.recv() => app.on_event(event),
            Some(Ok(event)) = term_events.next() => {
                if let Event::Key(key) = event {
                    if key.kind == crossterm::event::KeyEventKind::Press {
                        handle_key(&mut app, key);
                    }
                }
            }
        }
    }
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
        Mode::ModelPicker => handle_picker_key(app, key),
    }
}

fn handle_input_key(app: &mut App, key: KeyEvent) {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('p') => app.open_picker(),
            KeyCode::Char('u') => {
                app.input.clear();
                app.input_cursor = 0;
            }
            _ => {}
        }
        return;
    }
    match key.code {
        KeyCode::Enter => app.submit_input(),
        KeyCode::Char(c) => {
            let at = byte_index(&app.input, app.input_cursor);
            app.input.insert(at, c);
            app.input_cursor += 1;
        }
        KeyCode::Backspace => {
            if app.input_cursor > 0 {
                app.input_cursor -= 1;
                let at = byte_index(&app.input, app.input_cursor);
                app.input.remove(at);
            }
        }
        KeyCode::Left => app.input_cursor = app.input_cursor.saturating_sub(1),
        KeyCode::Right => app.input_cursor = (app.input_cursor + 1).min(app.input.chars().count()),
        KeyCode::Home => app.input_cursor = 0,
        KeyCode::End => app.input_cursor = app.input.chars().count(),
        _ => handle_scroll_key(app, key),
    }
}

fn handle_scroll_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::PageUp => app.scroll_from_bottom += 10,
        KeyCode::PageDown => app.scroll_from_bottom = app.scroll_from_bottom.saturating_sub(10),
        _ => {}
    }
}

fn handle_picker_key(app: &mut App, key: KeyEvent) {
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

/// Convert a char-based cursor position to a byte index for String edits.
fn byte_index(s: &str, char_pos: usize) -> usize {
    s.char_indices()
        .nth(char_pos)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}
