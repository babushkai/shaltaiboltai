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
    }
}

fn test_app() -> (App, UnboundedReceiver<AppEvent>) {
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
