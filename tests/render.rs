use ratatui::backend::TestBackend;
use ratatui::style::Color;
use ratatui::Terminal;
use shaltaiboltai::app::{App, AppEvent, Entry, Mode};
use shaltaiboltai::config::Config;
use shaltaiboltai::providers::{ChatEvent, ImageData, ModelEntry, ProviderKind, ToolCall};
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
        claude_code_bypass_permissions: false,
        codex_full_access: false,
    }
}

fn screen(terminal: &Terminal<TestBackend>) -> String {
    let area = terminal.backend().buffer().area;
    let buffer = terminal.backend().buffer();
    (0..area.height)
        .map(|y| {
            (0..area.width)
                .map(|x| buffer[(x, y)].symbol().to_owned())
                .collect::<String>()
                + "\n"
        })
        .collect()
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
async fn slash_input_opens_the_command_menu() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    app.textarea.insert_str("/th");
    assert!(app.slash_menu_active());

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    let screen: String = (0..24)
        .map(|y| {
            (0..80)
                .map(|x| buffer[(x, y)].symbol().to_owned())
                .collect::<String>()
                + "\n"
        })
        .collect();
    assert!(screen.contains("/theme"), "{screen}");
    assert!(screen.contains("color theme"), "{screen}");

    // Tab completes the highlighted command into the input, with a trailing
    // space because /theme takes an argument.
    app.complete_selected_slash();
    assert_eq!(app.textarea.lines().join(""), "/theme ");
}

#[tokio::test]
async fn model_picker_distinguishes_cli_defaults_aliases_and_exact_models() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.models = vec![
        ModelEntry {
            provider: ProviderKind::ClaudeCode,
            id: "claude-code".into(),
        },
        ModelEntry {
            provider: ProviderKind::ClaudeCode,
            id: "claude-code:sonnet".into(),
        },
        ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:gpt-5.6-sol".into(),
        },
    ];
    app.open_picker();
    let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();

    terminal.draw(|frame| ui::draw(frame, &mut app)).unwrap();
    let rendered = screen(&terminal);

    assert!(rendered.contains("claude-code  CLI default"), "{rendered}");
    assert!(
        rendered.contains("claude-code  sonnet · latest alias · subscription sub-agent"),
        "{rendered}"
    );
    assert!(
        rendered.contains("codex        gpt-5.6-sol · subscription sub-agent"),
        "{rendered}"
    );
}

#[tokio::test]
async fn statusline_shows_cwd_and_branch() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    // The test process runs inside the repo, so both should be present.
    assert!(!app.cwd_display.is_empty());
    let mut terminal = Terminal::new(TestBackend::new(120, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    // Layout bottom-up: input is 3 rows (1 line + borders), status is the
    // single row above it: 24 - 3 - 1 = 20.
    let status_row: String = (0..120)
        .map(|x| buffer[(x, 20)].symbol().to_owned())
        .collect();
    assert!(
        status_row.contains(app.cwd_display.as_str()),
        "{status_row}"
    );
    if let Some(branch) = &app.git_branch {
        assert!(status_row.contains(branch.as_str()), "{status_row}");
    }
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

#[tokio::test]
async fn help_is_a_responsive_overlay_instead_of_transcript_noise() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.textarea.insert_str("/help");
    app.submit_input();
    assert_eq!(app.mode, Mode::Help);
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("keyboard guide"), "{rendered}");
    assert!(rendered.contains("restore queued, then quit"), "{rendered}");
    assert!(rendered.contains("F1 · Enter · Esc close"), "{rendered}");
}

#[tokio::test]
async fn help_height_boundary_keeps_the_queue_safe_quit_binding() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.open_help();
    // This yields exactly 17 inner rows: one too short for the detailed guide.
    let mut terminal = Terminal::new(TestBackend::new(80, 21)).unwrap();

    terminal.draw(|frame| ui::draw(frame, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("queue-safe quit"), "{rendered}");
}

#[tokio::test]
async fn long_approval_keeps_actions_visible_and_scrolls_its_preview() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    let content = (0..80)
        .map(|line| format!("approval line {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Completed {
            tool_calls: vec![ToolCall {
                id: "approval".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({
                    "path": ".approval-preview-test-do-not-write",
                    "content": content,
                }),
            }],
            stop_reason: Some("tool_calls".into()),
            usage: None,
        },
    });
    assert_eq!(app.mode, Mode::Approval);
    app.focus_approval();
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let first = screen(&terminal);
    assert!(first.contains("review tool request"), "{first}");
    assert!(first.contains("approve"), "{first}");
    assert!(first.contains("deny"), "{first}");
    assert!(first.contains("lines 1–"), "{first}");

    app.approval_scroll = usize::MAX;
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let last = screen(&terminal);
    assert!(app.approval_scroll > 0);
    assert!(last.contains("diff truncated"), "{last}");
    assert!(last.contains("approve"), "{last}");
    assert!(last.contains("deny"), "{last}");

    let mut narrow = Terminal::new(TestBackend::new(24, 8)).unwrap();
    narrow.draw(|f| ui::draw(f, &mut app)).unwrap();
    let narrow_screen = screen(&narrow);
    assert!(narrow_screen.contains("y yes"), "{narrow_screen}");
    assert!(narrow_screen.contains("n no"), "{narrow_screen}");
    assert!(narrow_screen.contains("a this path"), "{narrow_screen}");
}

#[tokio::test]
async fn coalesced_delta_then_tool_activity_refreshes_the_assistant_entry() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.transcript = vec![Entry::Assistant(String::new())];
    app.transcript_rev += 1;
    app.mode = Mode::Streaming;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();

    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::TextDelta("coalesced text is visible".into()),
    });
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::ToolActivity {
            summary: "inspected src/ui.rs".into(),
            is_error: false,
        },
    });
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();

    let rendered = screen(&terminal);
    assert!(rendered.contains("coalesced text is visible"), "{rendered}");
    assert_eq!(app.render_cache.len(), app.transcript.len());
    assert!(app.transcript_dirty_from.is_none());
}

#[tokio::test]
async fn tool_first_activity_replaces_the_cached_thinking_placeholder() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.transcript = vec![Entry::Assistant(String::new())];
    app.transcript_rev += 1;
    app.mode = Mode::Streaming;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();

    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::ToolActivity {
            summary: "running the first tool".into(),
            is_error: false,
        },
    });
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();

    let rendered = screen(&terminal);
    assert!(rendered.contains("running the first tool"), "{rendered}");
    assert!(!rendered.contains("thinking…"), "{rendered}");
}

#[tokio::test]
async fn conversation_rail_labels_people_and_tool_state() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.transcript = vec![
        Entry::User("Improve the interface hierarchy".into()),
        Entry::Assistant("**Done.** The hierarchy is clearer.".into()),
        Entry::Tool {
            summary: "checked the rendered interface".into(),
            result: "all assertions passed".into(),
            is_error: false,
        },
    ];
    app.transcript_rev += 1;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("YOU"), "{rendered}");
    assert!(rendered.contains("ASSISTANT"), "{rendered}");
    assert!(rendered.contains("DONE"), "{rendered}");
    assert!(rendered.contains("TOOL · 1 output line"), "{rendered}");
}

#[tokio::test]
async fn narrow_terminal_preserves_conversation_status_and_composer() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    let mut terminal = Terminal::new(TestBackend::new(32, 10)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("shaltaiboltai"), "{rendered}");
    assert!(rendered.contains("discovering"), "{rendered}");
    assert!(rendered.contains("compose"), "{rendered}");
}

#[tokio::test]
async fn active_composer_explains_and_confirms_one_turn_lookahead() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.model = Some(ModelEntry {
        provider: ProviderKind::Ollama,
        id: "queue-test".into(),
    });
    app.mode = Mode::Streaming;
    app.textarea.insert_str("run these checks next");
    let mut terminal = Terminal::new(TestBackend::new(80, 18)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let composing = screen(&terminal);
    assert!(composing.contains("next message"), "{composing}");
    assert!(composing.contains("Enter queue"), "{composing}");

    app.queue_input();
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let queued = screen(&terminal);
    assert!(queued.contains("next message queued"), "{queued}");
    assert!(queued.contains("waiting for current"), "{queued}");

    let mut narrow = Terminal::new(TestBackend::new(24, 8)).unwrap();
    narrow.draw(|f| ui::draw(f, &mut app)).unwrap();
    let narrow = screen(&narrow);
    assert!(narrow.contains("queued"), "{narrow}");

    let (tx, _rx) = unbounded_channel();
    let mut with_images = App::new(offline_config(), tx);
    with_images.model = Some(ModelEntry {
        provider: ProviderKind::Ollama,
        id: "queue-test".into(),
    });
    with_images.mode = Mode::Streaming;
    for name in ["one.png", "two.png"] {
        with_images.pending_images.push((
            name.into(),
            ImageData {
                media_type: "image/png".into(),
                data: "aW1hZ2U=".into(),
            },
        ));
    }
    with_images.textarea.insert_str("inspect these");
    with_images.queue_input();
    let mut narrow = Terminal::new(TestBackend::new(24, 8)).unwrap();
    narrow
        .draw(|frame| ui::draw(frame, &mut with_images))
        .unwrap();
    let narrow = screen(&narrow);
    assert!(narrow.contains("2 images"), "{narrow}");
}

#[tokio::test]
async fn approval_arrives_with_composer_focus_and_explicit_review_hint() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.mode = Mode::Streaming;
    app.textarea.insert_str("typed y remains text");
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Completed {
            tool_calls: vec![ToolCall {
                id: "focus-render".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({"path": "x.txt", "content": "x"}),
            }],
            stop_reason: Some("tool_calls".into()),
            usage: None,
        },
    });
    let mut terminal = Terminal::new(TestBackend::new(80, 18)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let rendered = screen(&terminal);

    assert!(rendered.contains("composer focus"), "{rendered}");
    assert!(rendered.contains("Tab review"), "{rendered}");
    assert!(rendered.contains("typed y remains text"), "{rendered}");
}

#[tokio::test]
async fn tall_draft_keeps_the_approval_review_escape_hatch_visible() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.mode = Mode::Streaming;
    app.textarea.insert_str(
        (1..=8)
            .map(|line| format!("draft line {line}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Completed {
            tool_calls: vec![ToolCall {
                id: "tall-draft-approval".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({"path": "x.txt", "content": "x"}),
            }],
            stop_reason: Some("tool_calls".into()),
            usage: None,
        },
    });

    let mut terminal = Terminal::new(TestBackend::new(32, 10)).unwrap();
    terminal.draw(|frame| ui::draw(frame, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("Tab"), "{rendered}");
    assert!(rendered.contains("draft line 8"), "{rendered}");

    let mut tiny = Terminal::new(TestBackend::new(24, 8)).unwrap();
    tiny.draw(|frame| ui::draw(frame, &mut app)).unwrap();
    let rendered = screen(&tiny);
    assert!(rendered.contains("Tab"), "{rendered}");
    assert!(rendered.contains("draft line 8"), "{rendered}");
}

#[tokio::test]
async fn scrolled_transcript_exposes_a_jump_to_latest_affordance() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.transcript = (0..40)
        .map(|index| Entry::Info(format!("event {index}")))
        .collect();
    app.transcript_rev += 1;
    app.scroll_from_bottom = usize::MAX;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("lines from latest"), "{rendered}");
    assert!(rendered.contains("Ctrl+End jump"), "{rendered}");
    assert_eq!(app.render_cache_starts.len(), app.transcript.len());
}

#[tokio::test]
async fn scrolled_transcript_stays_anchored_when_tail_reflows() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.transcript = (0..40)
        .map(|index| Entry::Info(format!("anchored event {index}")))
        .chain(std::iter::once(Entry::Info("short tail".into())))
        .collect();
    app.transcript_rev += 1;
    let tail = app.transcript.len() - 1;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    app.scroll_from_bottom = 10;
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let transcript_rows = |terminal: &Terminal<TestBackend>| {
        let buffer = terminal.backend().buffer();
        (1..19)
            .map(|y| {
                (1..79)
                    .map(|x| buffer[(x, y)].symbol().to_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
    };
    let anchored = transcript_rows(&terminal);
    let original_offset = app.scroll_from_bottom;

    app.transcript[tail] = Entry::Info("growing tail ".repeat(120));
    app.transcript_dirty_from = Some(tail);
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    assert_eq!(transcript_rows(&terminal), anchored);
    assert!(app.scroll_from_bottom > original_offset);

    app.transcript[tail] = Entry::Info("short tail".into());
    app.transcript_dirty_from = Some(tail);
    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    assert_eq!(transcript_rows(&terminal), anchored);
    assert_eq!(app.scroll_from_bottom, original_offset);
}

#[tokio::test]
async fn error_and_cancel_replacements_keep_scrolled_content_anchored() {
    isolate_data_dir();
    let transcript_rows = |terminal: &Terminal<TestBackend>| {
        let buffer = terminal.backend().buffer();
        (1..19)
            .map(|y| {
                (1..79)
                    .map(|x| buffer[(x, y)].symbol().to_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
    };

    let (tx, _rx) = unbounded_channel();
    let mut errored = App::new(offline_config(), tx);
    errored.discovering = false;
    errored.transcript = (0..40)
        .map(|index| Entry::Info(format!("error anchor {index}")))
        .chain(std::iter::once(Entry::Assistant(String::new())))
        .collect();
    errored.transcript_rev += 1;
    errored.mode = Mode::Streaming;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut errored)).unwrap();
    errored.scroll_from_bottom = 10;
    terminal.draw(|f| ui::draw(f, &mut errored)).unwrap();
    let anchored = transcript_rows(&terminal);
    errored.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Error("provider failed with details ".repeat(80)),
    });
    terminal.draw(|f| ui::draw(f, &mut errored)).unwrap();
    assert_eq!(transcript_rows(&terminal), anchored);

    let (tx, _rx) = unbounded_channel();
    let mut cancelled = App::new(offline_config(), tx);
    cancelled.discovering = false;
    cancelled.transcript = (0..40)
        .map(|index| Entry::Info(format!("cancel anchor {index}")))
        .chain(std::iter::once(Entry::Assistant(String::new())))
        .collect();
    cancelled.transcript_rev += 1;
    cancelled.mode = Mode::Streaming;
    terminal.draw(|f| ui::draw(f, &mut cancelled)).unwrap();
    cancelled.scroll_from_bottom = 10;
    terminal.draw(|f| ui::draw(f, &mut cancelled)).unwrap();
    let anchored = transcript_rows(&terminal);
    cancelled.cancel_request();
    terminal.draw(|f| ui::draw(f, &mut cancelled)).unwrap();
    assert_eq!(transcript_rows(&terminal), anchored);
}

#[tokio::test]
async fn long_approval_material_wraps_to_reachable_visual_rows() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut command_app = App::new(offline_config(), tx);
    command_app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Completed {
            tool_calls: vec![ToolCall {
                id: "long-command".into(),
                name: "run_command".into(),
                arguments: serde_json::json!({
                    "command": format!("echo start; {} echo APPROVAL_TAIL_§", "echo segment; ".repeat(40)),
                }),
            }],
            stop_reason: Some("tool_calls".into()),
            usage: None,
        },
    });
    command_app.focus_approval();
    let mut terminal = Terminal::new(TestBackend::new(52, 16)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut command_app)).unwrap();
    command_app.approval_scroll = usize::MAX;
    terminal.draw(|f| ui::draw(f, &mut command_app)).unwrap();
    let command_screen = screen(&terminal);
    assert!(command_screen.contains('§'), "{command_screen}");
    assert!(
        command_screen.contains("this exact command"),
        "{command_screen}"
    );

    let (tx, _rx) = unbounded_channel();
    let mut diff_app = App::new(offline_config(), tx);
    diff_app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Completed {
            tool_calls: vec![ToolCall {
                id: "wide-diff".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({
                    "path": ".approval-wide-diff-test-do-not-write",
                    "content": format!("{}§\n", "界".repeat(80)),
                }),
            }],
            stop_reason: Some("tool_calls".into()),
            usage: None,
        },
    });
    diff_app.focus_approval();
    let mut terminal = Terminal::new(TestBackend::new(42, 14)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut diff_app)).unwrap();
    diff_app.approval_scroll = usize::MAX;
    terminal.draw(|f| ui::draw(f, &mut diff_app)).unwrap();
    let diff_screen = screen(&terminal);
    assert!(diff_screen.contains('§'), "{diff_screen}");
}

#[tokio::test]
async fn narrow_help_prioritizes_safety_bindings_without_overflow() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.open_help();
    let mut terminal = Terminal::new(TestBackend::new(32, 10)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let rendered = screen(&terminal);
    assert!(rendered.contains("cancel / deny"), "{rendered}");
    assert!(rendered.contains("approval"), "{rendered}");
    assert!(rendered.contains("queue-safe quit"), "{rendered}");
}

#[tokio::test]
async fn light_theme_tool_badge_uses_a_readable_on_color() {
    isolate_data_dir();
    let (tx, _rx) = unbounded_channel();
    let mut app = App::new(offline_config(), tx);
    app.discovering = false;
    app.theme = theme::LATTE;
    app.transcript = vec![Entry::Tool {
        summary: "checked contrast".into(),
        result: "ok".into(),
        is_error: false,
    }];
    app.transcript_rev += 1;
    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();

    terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    let buffer = terminal.backend().buffer();
    let done_cell = (0..buffer.area.height)
        .flat_map(|y| (0..buffer.area.width).map(move |x| (x, y)))
        .map(|position| &buffer[position])
        .find(|cell| cell.symbol() == "D" && cell.bg == theme::LATTE.success)
        .expect("DONE badge should be rendered on the success color");
    assert_eq!(done_cell.fg, Color::Black);
    assert!(
        (0..buffer.area.height)
            .flat_map(|y| (0..buffer.area.width).map(move |x| (x, y)))
            .map(|position| &buffer[position])
            .all(|cell| !(cell.bg == theme::LATTE.surface.unwrap()
                && cell.fg == theme::LATTE.success)),
        "low-contrast success text should fall back on Latte surfaces"
    );

    let (tx, _rx) = unbounded_channel();
    let mut approval = App::new(offline_config(), tx);
    approval.theme = theme::LATTE;
    approval.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Completed {
            tool_calls: vec![ToolCall {
                id: "latte-approval".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({
                    "path": ".latte-contrast-test-do-not-write",
                    "content": "contrast check",
                }),
            }],
            stop_reason: Some("tool_calls".into()),
            usage: None,
        },
    });
    let mut terminal = Terminal::new(TestBackend::new(60, 16)).unwrap();
    terminal.draw(|f| ui::draw(f, &mut approval)).unwrap();
    let buffer = terminal.backend().buffer();
    assert!(screen(&terminal).contains("review tool request"));
    assert!(
        (0..buffer.area.height)
            .flat_map(|y| (0..buffer.area.width).map(move |x| (x, y)))
            .map(|position| &buffer[position])
            .all(|cell| !(cell.bg == theme::LATTE.surface.unwrap()
                && cell.fg == theme::LATTE.warning)),
        "low-contrast warning text should fall back on Latte surfaces"
    );
}
