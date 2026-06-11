use ratatui::backend::TestBackend;
use ratatui::style::Color;
use ratatui::Terminal;
use shaltaiboltai::app::App;
use shaltaiboltai::config::Config;
use shaltaiboltai::{theme, ui};
use tokio::sync::mpsc::unbounded_channel;

/// Tests must never read or write the user's real data dir (persisted theme,
/// sessions, input history).
fn isolate_data_dir() {
    let tmp = std::env::temp_dir().join(format!("shaltai-render-{}", std::process::id()));
    std::env::set_var("SHALTAIBOLTAI_DATA_DIR", tmp);
}

fn offline_config() -> Config {
    Config {
        anthropic_api_key: None,
        openai_api_key: None,
        openai_base_url: "http://127.0.0.1:9".into(),
        ollama_host: "http://127.0.0.1:9".into(),
        default_model: None,
        compact_threshold_chars: 80_000,
        ollama_num_ctx: 16_384,
        theme: None,
    }
}

#[tokio::test]
async fn renders_themed_frame() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let buffer = terminal.backend().buffer().clone();

    // Rounded border corner of the transcript block.
    assert_eq!(buffer[(0, 0)].symbol(), "╭");
    // Default theme (mocha) background is painted.
    assert_eq!(app.theme.name, theme::DEFAULT.name);
    assert_eq!(buffer[(0, 0)].bg, theme::DEFAULT.bg.unwrap());
    // Title with the diamond brand mark is present.
    let top_row: String = (0..80)
        .map(|x| buffer[(x, 0)].symbol().to_owned())
        .collect();
    assert!(top_row.contains("◆ shaltaiboltai"), "{top_row}");
}

#[tokio::test]
async fn theme_switch_restyles_the_frame() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    app.open_themes();
    // Walk to a different theme and confirm the painted background follows.
    let start = app.theme.name;
    app.theme_move(1);
    assert_ne!(app.theme.name, start);

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    assert_eq!(buffer[(0, 0)].bg, app.theme.bg.unwrap());

    // Esc must restore the original theme.
    app.revert_theme();
    assert_eq!(app.theme.name, start);
}

#[tokio::test]
async fn terminal_theme_keeps_default_background() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.theme = theme::TERMINAL;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    assert_eq!(buffer[(0, 0)].bg, Color::Reset);
}
