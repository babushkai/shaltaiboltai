use shaltaiboltai::app::Entry;
use shaltaiboltai::providers::{Message, ModelEntry, ProviderKind, ToolCall};
use shaltaiboltai::session::{self, Session};

#[test]
fn session_round_trip() {
    let tmp = std::env::temp_dir().join(format!("shaltaiboltai-test-{}", std::process::id()));
    std::env::set_var("SHALTAIBOLTAI_DATA_DIR", &tmp);

    let original = Session {
        id: "test-1".into(),
        title: "fix the build".into(),
        updated_at: session::now_secs(),
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
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].title, "fix the build");

    let loaded = session::load(&listed[0].path).expect("load");
    assert_eq!(loaded.id, "test-1");
    assert_eq!(loaded.history.len(), 3);
    assert!(
        matches!(&loaded.history[1], Message::Assistant { tool_calls, .. } if tool_calls.len() == 1)
    );
    assert_eq!(loaded.transcript.len(), 2);

    std::fs::remove_dir_all(&tmp).ok();
}
