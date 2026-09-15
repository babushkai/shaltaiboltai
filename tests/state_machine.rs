use shaltaiboltai::app::{App, AppEvent, Mode};
use shaltaiboltai::config::Config;
use shaltaiboltai::policy::{ExecutionPolicy, PermissionPreset, Workspace};
use shaltaiboltai::providers::{
    ChatEvent, ImageData, Message, ModelEntry, ProviderKind, ToolCall, UserContent,
};
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
        reduced_motion: false,
    }
}

fn test_app() -> (App, UnboundedReceiver<AppEvent>) {
    test_app_with_config(offline_config())
}

fn test_app_with_config(config: Config) -> (App, UnboundedReceiver<AppEvent>) {
    // Never touch the user's real data dir (theme, sessions, input history).
    let tmp = std::env::temp_dir().join(format!("shaltai-sm-{}", std::process::id()));
    std::env::set_var("SHALTAIBOLTAI_DATA_DIR", tmp);
    let (tx, rx) = unbounded_channel();
    (App::new(config, tx), rx)
}

fn write_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "write_file".into(),
        arguments: serde_json::json!({"path": "x.txt", "content": "hi"}),
    }
}

fn completed(tool_calls: Vec<ToolCall>) -> ChatEvent {
    let stop_reason = if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    ChatEvent::Completed {
        tool_calls,
        stop_reason: Some(stop_reason.into()),
        usage: None,
    }
}

fn enable_test_model(app: &mut App) {
    app.model = Some(ModelEntry {
        provider: ProviderKind::Ollama,
        id: "queue-test".into(),
    });
}

fn require_write_approval(app: &mut App) {
    app.policy.apply_preset(PermissionPreset::ReadOnly);
}

#[tokio::test]
async fn next_prompt_waits_outside_history_then_dispatches_exactly_once() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    assert_eq!(app.mode, Mode::Streaming);
    let first_gen = app.event_generation();

    app.textarea.insert_str("second request");
    app.queue_input();
    assert_eq!(app.queued_prompt_count(), 1);
    assert!(app.input_is_empty());
    assert_eq!(
        app.history
            .iter()
            .filter(|message| matches!(message, Message::User(_)))
            .count(),
        1,
        "queued input must not enter provider history early"
    );
    assert!(!app.transcript.iter().any(
        |entry| matches!(entry, shaltaiboltai::app::Entry::User(text) if text == "second request")
    ));

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: completed(Vec::new()),
    });

    assert_eq!(app.queued_prompt_count(), 0);
    assert_eq!(app.mode, Mode::Streaming);
    assert!(matches!(
        app.history.as_slice(),
        [
            Message::User(first),
            Message::Assistant { .. },
            Message::User(second)
        ] if first.text() == "first request" && second.text() == "second request"
    ));

    // The queued root turn owns a fresh generation. Buffered events from the
    // completed request cannot corrupt or terminate it.
    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Error("late old error".into()),
    });
    assert_eq!(app.mode, Mode::Streaming);
    assert!(!app.transcript.iter().any(
        |entry| matches!(entry, shaltaiboltai::app::Entry::Error(text) if text == "late old error")
    ));
    app.cancel_request();
}

#[tokio::test]
async fn error_and_cancel_restore_queued_text_and_attachments() {
    for cancel in [false, true] {
        let (mut app, _rx) = test_app();
        enable_test_model(&mut app);
        app.textarea.insert_str("first request");
        app.submit_input();
        let first_gen = app.event_generation();
        app.pending_images.push((
            "queued.png".into(),
            ImageData {
                media_type: "image/png".into(),
                data: "cXVldWVk".into(),
            },
        ));
        app.textarea.insert_str("keep this next request");
        app.queue_input();

        if cancel {
            app.cancel_request();
        } else {
            app.on_event(AppEvent::Chat {
                gen: first_gen,
                event: ChatEvent::Error("provider failed".into()),
            });
        }

        assert_eq!(app.mode, Mode::Input);
        assert_eq!(app.queued_prompt_count(), 0);
        assert_eq!(app.textarea.lines().join("\n"), "keep this next request");
        assert_eq!(app.pending_images.len(), 1);
        assert_eq!(app.pending_images[0].0, "queued.png");
        assert_eq!(
            app.history
                .iter()
                .filter(|message| matches!(message, Message::User(_)))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn provider_rejection_before_activity_restores_the_current_prompt() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.pending_images.push((
        "retry.png".into(),
        ImageData {
            media_type: "image/png".into(),
            data: "cmV0cnk=".into(),
        },
    ));
    app.textarea.insert_str("retry this exact request");
    app.submit_input();
    let gen = app.event_generation();

    app.on_event(AppEvent::Chat {
        gen,
        event: ChatEvent::Notice("images are not forwarded by this CLI".into()),
    });
    app.on_event(AppEvent::Chat {
        gen,
        event: ChatEvent::Error("unknown model".into()),
    });

    assert_eq!(app.mode, Mode::Input);
    assert_eq!(app.textarea.lines().join("\n"), "retry this exact request");
    assert_eq!(app.pending_images.len(), 1);
    assert!(
        app.history.is_empty(),
        "the rejected user turn must roll back"
    );
    assert!(app
        .composer_notice()
        .is_some_and(|notice| notice.contains("message restored")));
}

#[tokio::test]
async fn early_provider_failure_never_overwrites_a_successive_draft() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let gen = app.event_generation();

    app.textarea.insert_str("draft for later");
    app.pending_images.push((
        "later.png".into(),
        ImageData {
            media_type: "image/png".into(),
            data: "bGF0ZXI=".into(),
        },
    ));
    app.on_event(AppEvent::Chat {
        gen,
        event: ChatEvent::Error("unknown model".into()),
    });

    assert_eq!(app.textarea.lines().join("\n"), "draft for later");
    assert_eq!(app.pending_images[0].0, "later.png");
    assert!(matches!(
        app.history.first(),
        Some(Message::User(content)) if content.text() == "first request"
    ));
}

#[tokio::test]
async fn restored_queue_reuses_frozen_referenced_image_bytes() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();

    let image_path =
        std::env::temp_dir().join(format!("shaltai-queued-ref-{}.png", std::process::id()));
    std::fs::write(&image_path, b"original image bytes").unwrap();
    app.textarea
        .insert_str(format!("inspect {}", image_path.display()));
    app.queue_input();
    std::fs::remove_file(&image_path).unwrap();

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Error("provider failed".into()),
    });
    assert_eq!(
        app.pending_image_count(),
        1,
        "the frozen referenced image must remain visible after restore"
    );
    app.submit_input();

    assert!(matches!(
        app.history.last(),
        Some(Message::User(UserContent::Rich { images, .. })) if images.len() == 1
    ));
    assert!(!app.transcript.iter().any(
        |entry| matches!(entry, shaltaiboltai::app::Entry::Error(text) if text.contains("does not exist"))
    ));
    app.cancel_request();
}

#[tokio::test]
async fn clearing_a_restored_reference_prevents_hidden_resubmission() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();

    let image_path =
        std::env::temp_dir().join(format!("shaltai-queued-clear-{}.png", std::process::id()));
    std::fs::write(&image_path, b"original image bytes").unwrap();
    app.textarea
        .insert_str(format!("inspect {}", image_path.display()));
    app.queue_input();
    std::fs::remove_file(&image_path).unwrap();
    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Error("provider failed".into()),
    });

    assert_eq!(app.pending_image_count(), 1);
    app.clear_attachments();
    assert_eq!(app.pending_image_count(), 0);
    app.submit_input();
    assert!(matches!(
        app.history.last(),
        Some(Message::User(UserContent::Text(text))) if text.contains("inspect")
    ));
    app.cancel_request();
}

#[tokio::test]
async fn editing_a_restored_reference_permanently_invalidates_frozen_bytes() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();

    let image_path =
        std::env::temp_dir().join(format!("shaltai-queued-edit-{}.png", std::process::id()));
    std::fs::write(&image_path, b"original image bytes").unwrap();
    let original = format!("inspect {}", image_path.display());
    app.textarea.insert_str(&original);
    app.queue_input();
    std::fs::remove_file(&image_path).unwrap();
    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Error("provider failed".into()),
    });
    assert_eq!(app.pending_image_count(), 1);

    // A Delete/Backspace key can be delivered without changing text. The
    // content comparison in note_input_changed must keep frozen bytes intact.
    app.note_input_changed();
    assert_eq!(app.pending_image_count(), 1);

    app.textarea.insert_str(" edited");
    app.note_input_changed();
    assert_eq!(
        app.pending_image_count(),
        0,
        "the first text mutation must discard frozen referenced bytes"
    );
    app.clear_attachments();
    app.clear_input();
    app.textarea.insert_str(&original);
    app.note_input_changed();
    app.submit_input();

    assert!(matches!(
        app.history.last(),
        Some(Message::User(UserContent::Text(text))) if text == &original
    ));
    app.cancel_request();
}

#[tokio::test]
async fn truncated_turn_restores_instead_of_auto_sending_queue() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();
    app.textarea.insert_str("review before retrying");
    app.queue_input();

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Completed {
            tool_calls: Vec::new(),
            stop_reason: Some("length".into()),
            usage: None,
        },
    });

    assert_eq!(app.mode, Mode::Input);
    assert_eq!(app.queued_prompt_count(), 0);
    assert_eq!(app.textarea.lines().join("\n"), "review before retrying");
}

#[tokio::test]
async fn filtered_completion_restores_instead_of_auto_sending_queue() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();
    app.textarea.insert_str("review after filtering");
    app.queue_input();

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Completed {
            tool_calls: Vec::new(),
            stop_reason: Some("content_filter".into()),
            usage: None,
        },
    });

    assert_eq!(app.mode, Mode::Input);
    assert_eq!(app.queued_prompt_count(), 0);
    assert_eq!(app.textarea.lines().join("\n"), "review after filtering");
    assert!(app.transcript.iter().any(
        |entry| matches!(entry, shaltaiboltai::app::Entry::Error(text) if text.contains("content_filter"))
    ));
}

#[tokio::test]
async fn missing_terminal_reason_restores_instead_of_panicking_or_sending() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();
    app.textarea.insert_str("review after premature eof");
    app.queue_input();

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Completed {
            tool_calls: Vec::new(),
            stop_reason: None,
            usage: None,
        },
    });

    assert_eq!(app.mode, Mode::Input);
    assert_eq!(app.queued_prompt_count(), 0);
    assert_eq!(
        app.textarea.lines().join("\n"),
        "review after premature eof"
    );
    assert!(app.transcript.iter().any(|entry| {
        matches!(entry, shaltaiboltai::app::Entry::Error(text) if text.contains("ended before normal completion"))
    }));
}

#[tokio::test]
async fn inconsistent_stop_with_tool_calls_never_executes_or_promotes() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();
    app.textarea.insert_str("review malformed tool response");
    app.queue_input();

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: ChatEvent::Completed {
            tool_calls: vec![write_call("malformed-tool")],
            stop_reason: Some("stop".into()),
            usage: None,
        },
    });

    assert_eq!(app.mode, Mode::Input);
    assert!(app.pending_approval().is_none());
    assert_eq!(
        app.textarea.lines().join("\n"),
        "review malformed tool response"
    );
    assert!(app.history.iter().any(|message| matches!(
        message,
        Message::ToolResult {
            call_id,
            is_error: true,
            ..
        } if call_id == "malformed-tool"
    )));
}

#[tokio::test]
async fn queued_prompt_stays_behind_the_complete_tool_loop() {
    let (mut app, _rx) = test_app();
    require_write_approval(&mut app);
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    let first_gen = app.event_generation();
    app.textarea.insert_str("after all tools");
    app.queue_input();

    app.on_event(AppEvent::Chat {
        gen: first_gen,
        event: completed(vec![write_call("queued-tool")]),
    });
    assert_eq!(app.mode, Mode::Approval);
    assert_eq!(app.queued_prompt_count(), 1);
    assert!(!app.approval_focused);

    app.deny_pending();
    assert_eq!(app.mode, Mode::Streaming);
    assert_eq!(app.queued_prompt_count(), 1);
    let follow_up_gen = app.event_generation();
    app.on_event(AppEvent::Chat {
        gen: follow_up_gen,
        event: completed(Vec::new()),
    });

    let tool_result = app
        .history
        .iter()
        .position(|message| matches!(message, Message::ToolResult { call_id, .. } if call_id == "queued-tool"))
        .unwrap();
    let queued_user = app
        .history
        .iter()
        .position(|message| matches!(message, Message::User(UserContent::Text(text)) if text == "after all tools"))
        .unwrap();
    assert!(tool_result < queued_user);
    assert_eq!(app.mode, Mode::Streaming);
    app.cancel_request();
}

#[tokio::test]
async fn late_provider_events_cannot_interrupt_the_tool_phase() {
    let (mut app, _rx) = test_app();
    require_write_approval(&mut app);
    enable_test_model(&mut app);
    app.textarea.insert_str("request with a tool");
    app.submit_input();
    let provider_gen = app.event_generation();

    app.on_event(AppEvent::Chat {
        gen: provider_gen,
        event: completed(vec![write_call("phase-fence")]),
    });
    assert_eq!(app.mode, Mode::Approval);
    assert!(app.pending_approval().is_some());

    app.on_event(AppEvent::Chat {
        gen: provider_gen,
        event: ChatEvent::Error("late error from finished request".into()),
    });
    assert_eq!(app.mode, Mode::Approval);
    assert!(app.pending_approval().is_some());
    assert!(!app.transcript.iter().any(|entry| {
        matches!(entry, shaltaiboltai::app::Entry::Error(text) if text.contains("late error"))
    }));
    app.cancel_request();
}

#[tokio::test]
async fn one_lookahead_slot_locks_further_input_and_busy_commands() {
    let (mut app, _rx) = test_app();
    enable_test_model(&mut app);
    app.textarea.insert_str("first request");
    app.submit_input();
    app.textarea.insert_str("one queued request");
    app.queue_input();

    assert!(!app.composer_accepts_input());
    app.paste("must be ignored");
    app.queue_input();
    assert!(app.input_is_empty());
    assert!(app
        .composer_notice()
        .is_some_and(|notice| notice.contains("already queued")));
    app.cancel_request();

    // Slash commands typed during a new active turn remain drafts; they can
    // never become a delayed /new, /resume, or /quit side effect.
    app.clear_input();
    app.textarea.insert_str("restart active request");
    app.submit_input();
    app.textarea.insert_str("/new");
    app.queue_input();
    assert_eq!(app.queued_prompt_count(), 0);
    assert_eq!(app.textarea.lines().join("\n"), "/new");
    assert!(app
        .composer_notice()
        .is_some_and(|notice| notice.contains("commands are available")));
    app.cancel_request();
}

#[tokio::test]
async fn stale_tool_events_after_cancel_do_not_resume_the_loop() {
    let (mut app, _rx) = test_app();
    require_write_approval(&mut app);

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
    require_write_approval(&mut app);

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

#[cfg(unix)]
#[tokio::test]
async fn approval_cannot_rebind_to_a_retargeted_symlink() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!(
        "shaltai-approval-retarget-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    let workspace = root.join("workspace");
    let first_target = root.join("first-target");
    let second_target = root.join("second-target");
    std::fs::create_dir_all(&workspace).expect("create workspace");
    std::fs::create_dir_all(&first_target).expect("create first target");
    std::fs::create_dir_all(&second_target).expect("create second target");
    let link = workspace.join("escape");
    symlink(&first_target, &link).expect("create first symlink");

    let policy = ExecutionPolicy::new(Workspace::new(&workspace).expect("valid workspace"));
    let (tx, _rx) = unbounded_channel();
    let mut app = App::with_policy(offline_config(), policy, tx);
    let call = ToolCall {
        id: "retargeted-approval".into(),
        name: "write_file".into(),
        arguments: serde_json::json!({"path": "escape/file.txt", "content": "secret"}),
    };
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: completed(vec![call]),
    });
    assert_eq!(app.mode, Mode::Approval);

    std::fs::remove_file(&link).expect("remove first symlink");
    symlink(&second_target, &link).expect("retarget symlink");
    app.approve_pending(true);

    assert_eq!(app.mode, Mode::Approval);
    assert!(app.pending_approval().is_some());
    assert!(!first_target.join("file.txt").exists());
    assert!(!second_target.join("file.txt").exists());

    app.deny_pending();
    std::fs::remove_dir_all(root).ok();
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
    app.compacting = true;

    app.on_event(AppEvent::CompactionDone {
        session_id: "some-old-session".into(),
        compaction_gen: 0,
        result: Ok("summary of an older conversation".into()),
    });

    assert_eq!(app.history.len(), before, "history must not be replaced");
    assert!(
        app.compacting,
        "an old completion must not clear the current session's busy state"
    );
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
async fn slash_model_accepts_explicit_cli_selectors() {
    let (mut app, _rx) = test_app();
    app.models = vec![
        ModelEntry {
            provider: ProviderKind::ClaudeCode,
            id: "claude-code".into(),
        },
        ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex".into(),
        },
    ];

    app.textarea.insert_str("/model claude-code:claude-fable-5");
    app.submit_input();
    assert_eq!(
        app.model
            .as_ref()
            .map(|model| (model.provider, model.id.as_str())),
        Some((ProviderKind::ClaudeCode, "claude-code:claude-fable-5"))
    );

    app.textarea.insert_str("/model codex:gpt-future");
    app.submit_input();
    assert_eq!(
        app.model
            .as_ref()
            .map(|model| (model.provider, model.id.as_str())),
        Some((ProviderKind::Codex, "codex:gpt-future"))
    );
    assert!(app
        .models
        .iter()
        .any(|model| model.id == "codex:gpt-future"));

    app.textarea.insert_str("/model codex:default");
    app.submit_input();
    assert_eq!(
        app.model.as_ref().map(|model| model.id.as_str()),
        Some("codex")
    );

    app.models.push(ModelEntry {
        provider: ProviderKind::Codex,
        id: "codex:gpt-5.6-sol".into(),
    });
    app.textarea.insert_str("/model codex:gpt-5.6");
    app.submit_input();
    assert_eq!(
        app.model.as_ref().map(|model| model.id.as_str()),
        Some("codex:gpt-5.6"),
        "a qualified custom selector must not fuzzy-match a listed model"
    );
}

#[tokio::test]
async fn configured_cli_selector_materializes_after_provider_discovery() {
    let mut config = offline_config();
    config.default_model = Some("codex:gpt-future".into());
    let (mut app, _rx) = test_app_with_config(config);
    assert!(app.model.is_none());

    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![
            ModelEntry {
                provider: ProviderKind::Ollama,
                id: "codex:gpt-future".into(),
            },
            ModelEntry {
                provider: ProviderKind::Codex,
                id: "codex".into(),
            },
        ],
        finished: true,
    });

    assert_eq!(
        app.model
            .as_ref()
            .map(|model| (model.provider, model.id.as_str())),
        Some((ProviderKind::Codex, "codex:gpt-future"))
    );
    assert!(app
        .models
        .iter()
        .any(|model| model.id == "codex:gpt-future"));
}

#[tokio::test]
async fn unavailable_configured_cli_model_never_falls_back_across_providers() {
    let mut config = offline_config();
    config.anthropic_api_key = Some("test-key".into());
    config.default_model = Some("codex:gpt-unavailable".into());
    let (mut app, _rx) = test_app_with_config(config);
    assert!(app.model.is_none());

    app.on_event(AppEvent::ModelsDiscovered {
        models: Vec::new(),
        finished: true,
    });

    assert!(app.model.is_none());
    assert!(app.transcript.iter().any(|entry| matches!(
        entry,
        shaltaiboltai::app::Entry::Error(message)
            if message.contains("codex:gpt-unavailable") && message.contains("unavailable")
    )));
}

#[tokio::test]
async fn unavailable_explicit_cli_model_never_falls_back_across_providers() {
    let (mut app, _rx) = test_app();
    app.models = vec![ModelEntry {
        provider: ProviderKind::Codex,
        id: "codex".into(),
    }];
    app.textarea.insert_str("/model codex");
    app.submit_input();

    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![ModelEntry {
            provider: ProviderKind::Anthropic,
            id: "claude-sonnet-4-6".into(),
        }],
        finished: true,
    });

    assert!(app.model.is_none());
    assert!(app.transcript.iter().any(|entry| matches!(
        entry,
        shaltaiboltai::app::Entry::Error(message)
            if message.contains("selected model codex is unavailable")
    )));
}

#[tokio::test]
async fn bare_cli_default_config_is_bound_to_its_provider() {
    let mut config = offline_config();
    config.default_model = Some("codex".into());
    let (mut app, _rx) = test_app_with_config(config);

    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![
            ModelEntry {
                provider: ProviderKind::Ollama,
                id: "codex".into(),
            },
            ModelEntry {
                provider: ProviderKind::Codex,
                id: "codex".into(),
            },
        ],
        finished: true,
    });

    assert_eq!(
        app.model.as_ref().map(|model| model.provider),
        Some(ProviderKind::Codex)
    );
}

#[tokio::test]
async fn custom_cli_selector_round_trips_through_a_saved_session() {
    use shaltaiboltai::session;

    let (mut app, _rx) = test_app();
    let id = format!("cli-selector-session-{}", std::process::id());
    let title = format!("custom cli selector {}", std::process::id());
    session::save(&session::Session {
        id,
        title: title.clone(),
        updated_at: session::now_secs(),
        cwd: std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string()),
        model: Some(ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:gpt-future".into(),
        }),
        history: vec![Message::User("continue later".into())],
        transcript: Vec::new(),
    })
    .unwrap();

    app.open_sessions();
    app.session_index = app
        .sessions
        .iter()
        .position(|session| session.title == title)
        .unwrap();
    app.pick_session();

    assert_eq!(
        app.model
            .as_ref()
            .map(|model| (model.provider, model.id.as_str())),
        Some((ProviderKind::Codex, "codex:gpt-future"))
    );

    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex".into(),
        }],
        finished: true,
    });
    assert_eq!(
        app.model.as_ref().map(|model| model.id.as_str()),
        Some("codex:gpt-future"),
        "later discovery must retain the provisionally resumed selector"
    );
}

#[tokio::test]
async fn legacy_cli_defaults_resume_before_discovery() {
    use shaltaiboltai::session;

    for (provider, selector) in [
        (ProviderKind::ClaudeCode, "claude-code"),
        (ProviderKind::Codex, "codex"),
    ] {
        let (mut app, _rx) = test_app();
        let title = format!("legacy {selector} {}", std::process::id());
        session::save(&session::Session {
            id: format!("legacy-{selector}-{}", std::process::id()),
            title: title.clone(),
            updated_at: session::now_secs(),
            cwd: std::env::current_dir()
                .ok()
                .map(|path| path.display().to_string()),
            model: Some(ModelEntry {
                provider,
                id: selector.into(),
            }),
            history: vec![Message::User("continue later".into())],
            transcript: Vec::new(),
        })
        .unwrap();

        app.open_sessions();
        app.session_index = app
            .sessions
            .iter()
            .position(|session| session.title == title)
            .unwrap();
        app.pick_session();
        assert_eq!(
            app.model
                .as_ref()
                .map(|model| (model.provider, model.id.as_str())),
            Some((provider, selector))
        );

        app.on_event(AppEvent::ModelsDiscovered {
            models: vec![ModelEntry {
                provider,
                id: selector.into(),
            }],
            finished: true,
        });
        assert_eq!(
            app.model.as_ref().map(|model| model.id.as_str()),
            Some(selector)
        );
    }
}

#[tokio::test]
async fn api_and_local_models_resume_before_their_discovery_batch() {
    use shaltaiboltai::session;

    for (provider, selector) in [
        (ProviderKind::OpenAi, "gpt-saved"),
        (ProviderKind::Ollama, "qwen-saved:latest"),
    ] {
        let (mut app, _rx) = test_app();
        let title = format!("early {selector} {}", std::process::id());
        session::save(&session::Session {
            id: format!(
                "early-{}-{}",
                selector.replace(':', "-"),
                std::process::id()
            ),
            title: title.clone(),
            updated_at: session::now_secs(),
            cwd: std::env::current_dir()
                .ok()
                .map(|path| path.display().to_string()),
            model: Some(ModelEntry {
                provider,
                id: selector.into(),
            }),
            history: vec![Message::User("continue later".into())],
            transcript: Vec::new(),
        })
        .unwrap();

        app.open_sessions();
        app.session_index = app
            .sessions
            .iter()
            .position(|session| session.title == title)
            .unwrap();
        app.pick_session();
        assert_eq!(
            app.model
                .as_ref()
                .map(|model| (model.provider, model.id.as_str())),
            Some((provider, selector))
        );

        app.on_event(AppEvent::ModelsDiscovered {
            models: vec![ModelEntry {
                provider,
                id: selector.into(),
            }],
            finished: true,
        });
        assert_eq!(
            app.model
                .as_ref()
                .map(|model| (model.provider, model.id.as_str())),
            Some((provider, selector))
        );
    }
}

#[tokio::test]
async fn unavailable_saved_model_never_inherits_the_previous_sessions_provider() {
    use shaltaiboltai::session;

    let (mut app, _rx) = test_app();
    app.discovering = false;
    app.models = vec![ModelEntry {
        provider: ProviderKind::Anthropic,
        id: "claude-sonnet-4-6".into(),
    }];
    app.model = app.models.first().cloned();
    let title = format!("unavailable saved model {}", std::process::id());
    session::save(&session::Session {
        id: format!("unavailable-saved-model-{}", std::process::id()),
        title: title.clone(),
        updated_at: session::now_secs(),
        cwd: std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string()),
        model: Some(ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex:gpt-unavailable".into(),
        }),
        history: vec![Message::User("continue later".into())],
        transcript: Vec::new(),
    })
    .unwrap();

    app.open_sessions();
    app.session_index = app
        .sessions
        .iter()
        .position(|session| session.title == title)
        .unwrap();
    app.pick_session();

    assert!(app.model.is_none());
    assert!(app.transcript.iter().any(|entry| matches!(
        entry,
        shaltaiboltai::app::Entry::Error(message)
            if message.contains("saved model codex:gpt-unavailable is unavailable")
    )));
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
    app.model = Some(shaltaiboltai::providers::ModelEntry {
        provider: shaltaiboltai::providers::ProviderKind::Ollama,
        id: "test".into(),
    });

    app.textarea.insert_str("first prompt");
    app.submit_input(); // successfully submitted prompts enter recall history
    app.input_history_prev();
    assert!(!app.input_is_empty());
    assert!(app.history_recall_active());
    app.pending_images.push((
        "keep.png".into(),
        ImageData {
            media_type: "image/png".into(),
            data: "a2VlcA==".into(),
        },
    ));

    app.clear_input();
    assert!(app.input_is_empty());
    assert!(!app.history_recall_active());
    assert_eq!(
        app.pending_images.len(),
        1,
        "Ctrl+U clears text, not independently staged images"
    );
}

#[tokio::test]
async fn draft_survives_model_discovery() {
    let (mut app, _rx) = test_app();
    app.discovering = true;
    app.textarea
        .insert_str("keep this carefully written prompt");

    app.submit_input();

    assert_eq!(
        app.textarea.lines().join("\n"),
        "keep this carefully written prompt"
    );
    assert!(app.history.is_empty());
    assert!(app
        .transcript
        .iter()
        .any(|entry| matches!(entry, shaltaiboltai::app::Entry::Error(text) if text.contains("draft is safe"))));
}

#[tokio::test]
async fn refreshed_models_replace_an_unavailable_selection() {
    use shaltaiboltai::providers::{ModelEntry, ProviderKind};
    let (mut app, _rx) = test_app();
    app.model = Some(ModelEntry {
        provider: ProviderKind::Ollama,
        id: "removed-model".into(),
    });
    app.discovering = true;

    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex".into(),
        }],
        finished: true,
    });

    assert!(!app.discovering);
    assert_eq!(
        app.model.as_ref().map(|model| model.id.as_str()),
        Some("codex")
    );
}

#[tokio::test]
async fn dynamic_default_replaces_only_the_provisional_model() {
    use shaltaiboltai::providers::{ModelEntry, ProviderKind};

    let mut config = offline_config();
    config.anthropic_api_key = Some("test-key".into());
    config.default_model = Some("codex".into());
    let (mut app, _rx) = test_app_with_config(config.clone());
    assert!(
        app.model.is_none(),
        "an undiscovered explicit default must win"
    );
    assert!(
        app.models
            .iter()
            .all(|model| model.provider == ProviderKind::Anthropic),
        "static providers should be selectable immediately"
    );

    let discovered = vec![
        app.models.first().cloned().unwrap(),
        ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex".into(),
        },
    ];
    app.on_event(AppEvent::ModelsDiscovered {
        models: discovered.clone(),
        finished: true,
    });
    assert_eq!(
        app.model.as_ref().map(|model| model.id.as_str()),
        Some("codex")
    );

    let (mut explicitly_selected, _rx) = test_app_with_config(config);
    explicitly_selected.models = discovered.clone();
    explicitly_selected.open_picker();
    explicitly_selected.pick_model();
    let picked = explicitly_selected.model.clone();
    explicitly_selected.on_event(AppEvent::ModelsDiscovered {
        models: discovered,
        finished: true,
    });
    assert_eq!(
        explicitly_selected
            .model
            .as_ref()
            .map(|model| (model.id.as_str(), model.provider)),
        picked
            .as_ref()
            .map(|model| (model.id.as_str(), model.provider))
    );
}

#[tokio::test]
async fn first_dynamic_provider_is_selectable_before_discovery_finishes() {
    use shaltaiboltai::providers::{ModelEntry, ProviderKind};

    let (mut app, _rx) = test_app();
    assert!(app.discovering);
    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex".into(),
        }],
        finished: false,
    });

    assert!(app.discovering);
    assert_eq!(
        app.model.as_ref().map(|model| model.id.as_str()),
        Some("codex")
    );
    assert!(app.models.iter().any(|model| model.id == "codex"));

    app.on_event(AppEvent::ModelsDiscovered {
        models: Vec::new(),
        finished: true,
    });
    assert!(!app.discovering);
}

#[tokio::test]
async fn discovery_never_switches_provider_mid_agent_turn() {
    use shaltaiboltai::providers::{ModelEntry, ProviderKind};

    let mut config = offline_config();
    config.anthropic_api_key = Some("test-key".into());
    config.default_model = Some("codex".into());
    let (mut app, _rx) = test_app_with_config(config);
    require_write_approval(&mut app);
    let anthropic = app.models.first().cloned().unwrap();
    app.model = Some(anthropic.clone());
    app.mode = Mode::Streaming;
    app.transcript
        .push(shaltaiboltai::app::Entry::Assistant(String::new()));

    app.on_event(AppEvent::ModelsDiscovered {
        models: vec![ModelEntry {
            provider: ProviderKind::Codex,
            id: "codex".into(),
        }],
        finished: false,
    });
    assert_eq!(
        app.model.as_ref().map(|model| model.provider),
        Some(ProviderKind::Anthropic)
    );

    app.on_event(AppEvent::Chat {
        gen: 0,
        event: completed(vec![write_call("same-turn-tool")]),
    });
    assert_eq!(app.mode, Mode::Approval);
    app.deny_pending();
    assert_eq!(app.mode, Mode::Streaming);
    assert_eq!(
        app.model.as_ref().map(|model| model.provider),
        Some(ProviderKind::Anthropic),
        "the tool follow-up must stay on the turn's original provider"
    );

    app.cancel_request();
    assert_eq!(
        app.model.as_ref().map(|model| model.provider),
        Some(ProviderKind::Codex),
        "the discovered default may apply once the turn ends"
    );
}

#[tokio::test]
async fn quitting_during_approval_repairs_the_tool_turn() {
    let (mut app, _rx) = test_app();
    require_write_approval(&mut app);
    app.on_event(AppEvent::Chat {
        gen: 0,
        event: completed(vec![write_call("quit-call")]),
    });
    assert_eq!(app.mode, Mode::Approval);

    app.request_quit();

    assert!(app.should_quit);
    assert_eq!(app.mode, Mode::Input);
    assert!(app.history.iter().any(|message| matches!(
        message,
        Message::ToolResult { call_id, is_error: true, .. } if call_id == "quit-call"
    )));
}
