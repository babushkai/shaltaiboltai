use shaltaiboltai::app::{App, AppEvent, Mode};
use shaltaiboltai::config::Config;
use shaltaiboltai::providers::{ChatEvent, Message, ToolCall};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

/// Config that can't reach any provider, so background discovery fails fast
/// and tests never touch the network meaningfully.
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

fn test_app() -> (App, UnboundedReceiver<AppEvent>) {
    // Never touch the user's real data dir (theme, sessions, input history).
    let tmp = std::env::temp_dir().join(format!("shaltai-sm-{}", std::process::id()));
    std::env::set_var("SHALTAIBOLTAI_DATA_DIR", tmp);
    let (tx, rx) = unbounded_channel();
    (App::new(offline_config(), tx), rx)
}

fn write_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "write_file".into(),
        arguments: serde_json::json!({"path": "x.txt", "content": "hi"}),
    }
}

fn completed(tool_calls: Vec<ToolCall>) -> ChatEvent {
    ChatEvent::Completed {
        tool_calls,
        stop_reason: None,
        usage: None,
    }
}

#[tokio::test]
async fn stale_tool_events_after_cancel_do_not_resume_the_loop() {
    let (mut app, _rx) = test_app();

    // Model requests a mutating tool → approval gate.
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: completed(vec![write_call("c1")]),
    });
    assert_eq!(app.mode, Mode::Approval);
    assert!(app.pending_approval().is_some());

    app.cancel_request();
    assert_eq!(app.mode, Mode::Input);
    // The dangling tool_use must be closed so the next request is valid.
    assert!(matches!(
        app.history.last(),
        Some(Message::ToolResult { call_id, is_error: true, .. }) if call_id == "c1"
    ));

    // A tool result from the cancelled generation arrives late: it must be
    // dropped, not appended, and must not restart a request.
    let len = app.history.len();
    app.on_event(AppEvent::ToolFinished {
        gen: 0,
        call: write_call("c1"),
        content: "done".into(),
        is_error: false,
    });
    assert_eq!(app.history.len(), len);
    assert_eq!(app.mode, Mode::Input);
}

#[tokio::test]
async fn denied_tool_calls_record_an_error_result() {
    let (mut app, _rx) = test_app();

    app.on_event(AppEvent::Chat {
        gen: 0,
        event: completed(vec![write_call("c1")]),
    });
    assert_eq!(app.mode, Mode::Approval);

    app.deny_pending();
    let denial = app.history.iter().find(
        |m| matches!(m, Message::ToolResult { call_id, is_error: true, .. } if call_id == "c1"),
    );
    assert!(
        denial.is_some(),
        "denial should be recorded as an error tool result"
    );
    // No model configured → the follow-up request cannot start; we must land
    // back in input mode rather than a stuck state.
    assert_eq!(app.mode, Mode::Input);
}

#[tokio::test]
async fn mid_stream_errors_keep_partial_text_in_history() {
    let (mut app, _rx) = test_app();

    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::TextDelta("partial answer".into()),
    });
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: ChatEvent::Error("connection reset".into()),
    });

    assert!(
        app.history.iter().any(|m| matches!(
            m,
            Message::Assistant { text, .. } if text == "partial answer"
        )),
        "text the user saw must stay in the conversation"
    );
    assert_eq!(app.mode, Mode::Input);
}

#[tokio::test]
async fn compaction_result_for_an_old_session_is_discarded() {
    let (mut app, _rx) = test_app();

    app.history.push(Message::User("current work".into()));
    let before = app.history.len();

    app.on_event(AppEvent::CompactionDone {
        session_id: "some-old-session".into(),
        result: Ok("summary of an older conversation".into()),
    });

    assert_eq!(app.history.len(), before, "history must not be replaced");
    assert!(!app.compacting);
}

#[tokio::test]
async fn slash_theme_with_argument_switches_directly() {
    let (mut app, _rx) = test_app();
    app.textarea.insert_str("/theme nord");
    app.submit_input();
    assert_eq!(app.theme.name, "nord");

    // Unknown names error and keep the current theme.
    app.textarea.insert_str("/theme nonexistent");
    app.submit_input();
    assert_eq!(app.theme.name, "nord");
}

#[tokio::test]
async fn slash_model_with_argument_selects_or_prefilters() {
    use shaltaiboltai::providers::{ModelEntry, ProviderKind};
    let (mut app, _rx) = test_app();
    app.models = vec![
        ModelEntry {
            provider: ProviderKind::Ollama,
            id: "qwen3.5:latest".into(),
        },
        ModelEntry {
            provider: ProviderKind::Ollama,
            id: "gpt-oss:20b-cloud".into(),
        },
    ];

    // Unique substring match selects directly.
    app.textarea.insert_str("/model qwen");
    app.submit_input();
    assert_eq!(
        app.model.as_ref().map(|m| m.id.as_str()),
        Some("qwen3.5:latest")
    );
    assert_eq!(app.mode, Mode::Input);

    // Ambiguous match opens the picker pre-filtered.
    app.models.push(ModelEntry {
        provider: ProviderKind::Ollama,
        id: "qwen2:7b".into(),
    });
    app.textarea.insert_str("/model qwen");
    app.submit_input();
    assert_eq!(app.mode, Mode::ModelPicker);
    assert_eq!(app.picker_filter, "qwen");
}

#[tokio::test]
async fn session_picker_orders_current_project_first() {
    use shaltaiboltai::session;
    let (mut app, _rx) = test_app();
    let here = std::env::current_dir().unwrap().display().to_string();
    for (id, title, cwd) in [
        ("scope-other", "other project", "/somewhere/else".to_owned()),
        ("scope-here", "this project", here),
    ] {
        session::save(&session::Session {
            id: id.into(),
            title: title.into(),
            updated_at: session::now_secs(),
            cwd: Some(cwd),
            model: None,
            history: vec![Message::User("x".into())],
            transcript: Vec::new(),
        })
        .unwrap();
    }

    app.open_sessions();
    assert_eq!(app.mode, Mode::SessionPicker);
    let titles: Vec<&str> = app.sessions.iter().map(|s| s.title.as_str()).collect();
    let here_pos = titles.iter().position(|t| *t == "this project").unwrap();
    let other_pos = titles.iter().position(|t| *t == "other project").unwrap();
    assert!(here_pos < other_pos, "{titles:?}");
}

#[tokio::test]
async fn image_paths_in_the_message_become_attachments() {
    use shaltaiboltai::providers::UserContent;
    let (mut app, _rx) = test_app();
    let img = std::env::temp_dir().join(format!("shaltai-sm-img-{}.png", std::process::id()));
    std::fs::write(&img, b"fake").unwrap();

    // No model configured: the request won't start, but the history entry is
    // still built — which is what we're asserting on.
    app.model = Some(shaltaiboltai::providers::ModelEntry {
        provider: shaltaiboltai::providers::ProviderKind::Ollama,
        id: "test".into(),
    });
    app.textarea
        .insert_str(format!("describe {}", img.display()));
    app.submit_input();

    let Some(Message::User(content)) = app.history.iter().find(|m| matches!(m, Message::User(_)))
    else {
        panic!("user message missing");
    };
    assert!(matches!(content, UserContent::Rich { .. }));
    assert_eq!(content.images().len(), 1);
    assert_eq!(content.images()[0].media_type, "image/png");
    assert!(content.text().contains("describe"));
    std::fs::remove_file(img).ok();
}

#[tokio::test]
async fn dropping_a_file_onto_the_terminal_stages_it() {
    let (mut app, _rx) = test_app();
    let img = std::env::temp_dir().join(format!("shaltai-drop-{}.png", std::process::id()));
    std::fs::write(&img, b"fake").unwrap();

    // A drag-and-drop arrives as a paste event containing only the path.
    app.paste(&img.display().to_string());
    assert_eq!(app.pending_images.len(), 1);
    assert!(app.input_is_empty(), "the path must not land in the input");

    // Ordinary pasted text still goes into the editor.
    app.paste("hello world");
    assert!(!app.input_is_empty());

    std::fs::remove_file(img).ok();
}

#[tokio::test]
async fn clear_input_wipes_the_text_and_exits_history_recall() {
    let (mut app, _rx) = test_app();

    app.textarea.insert_str("first prompt");
    app.submit_input(); // remembered into input history (no model → no request)
    app.input_history_prev();
    assert!(!app.input_is_empty());
    assert!(app.history_recall_active());

    app.clear_input();
    assert!(app.input_is_empty());
    assert!(!app.history_recall_active());
}
