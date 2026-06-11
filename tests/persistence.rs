use shaltaiboltai::app::Entry;
use shaltaiboltai::providers::{Message, ModelEntry, ProviderKind, ToolCall};
use shaltaiboltai::session::{self, Session};
use std::path::PathBuf;

/// Both tests share one process-wide data dir and run in parallel, so they
/// assert only on their own files and never wipe the directory.
fn isolated_data_dir() -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("shaltaiboltai-test-{}", std::process::id()));
    std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &tmp);
    tmp
}

#[test]
fn session_round_trip() {
    isolated_data_dir();

    let original = Session {
        id: "test-1".into(),
        title: "fix the build".into(),
        updated_at: session::now_secs(),
        cwd: Some("/tmp/project".into()),
        model: Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "qwen3.5:latest".into(),
        }),
        history: vec![
            Message::User("fix the build".into()),
            Message::Assistant {
                text: "Looking.".into(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "run_command".into(),
                    arguments: serde_json::json!({"command": "cargo build"}),
                }],
            },
            Message::ToolResult {
                call_id: "c1".into(),
                name: "run_command".into(),
                content: "ok".into(),
                is_error: false,
            },
        ],
        transcript: vec![
            Entry::User("fix the build".into()),
            Entry::Tool {
                summary: "run_command: cargo build".into(),
                result: "ok".into(),
                is_error: false,
            },
        ],
    };

    session::save(&original).expect("save");
    let listed = session::list();
    let meta = listed
        .iter()
        .find(|m| m.title == "fix the build")
        .expect("saved session should be listed");
    assert_eq!(meta.cwd.as_deref(), Some("/tmp/project"));

    let loaded = session::load(&meta.path).expect("load");
    assert_eq!(loaded.id, "test-1");
    assert_eq!(loaded.history.len(), 3);
    assert!(
        matches!(&loaded.history[1], Message::Assistant { tool_calls, .. } if tool_calls.len() == 1)
    );
    assert_eq!(loaded.transcript.len(), 2);
}

#[test]
fn legacy_sessions_without_cwd_still_load() {
    let tmp = isolated_data_dir();
    let dir = tmp.join("sessions");
    std::fs::create_dir_all(&dir).unwrap();
    // A file saved by a version that predates the cwd field.
    let path = dir.join("legacy-1.json");
    std::fs::write(
        &path,
        r#"{"id":"legacy-1","title":"old session","updated_at":1,"model":null,"history":[{"User":"plain old text"}],"transcript":[]}"#,
    )
    .unwrap();

    let loaded = session::load(&path).expect("legacy file should deserialize");
    assert_eq!(loaded.cwd, None);
    // Pre-image user messages were bare strings; they must still round-trip.
    assert!(matches!(
        &loaded.history[0],
        Message::User(c) if c.text() == "plain old text" && c.images().is_empty()
    ));
    assert!(session::list().iter().any(|m| m.title == "old session"));
}
