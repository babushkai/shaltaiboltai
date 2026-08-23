//! Live end-to-end check of a sub-agent CLI provider through the real
//! provider pipeline (spawn → stream → ChatEvent), without the TUI.
//! Usage: `cargo run --example cli_probe -- claude-code[:model]|codex[:model]`
//! This bills the corresponding subscription one trivial read-only turn.

use shaltaiboltai::config::Config;
use shaltaiboltai::providers::{
    self, ChatEvent, ChatRequest, Message, ModelEntry, ProviderKind, RequestPolicy,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "codex".into());
    let (provider, model_id) = match which.as_str() {
        "claude" => (ProviderKind::ClaudeCode, "claude-code".to_owned()),
        "claude-code" | "codex" => (
            if which == "codex" {
                ProviderKind::Codex
            } else {
                ProviderKind::ClaudeCode
            },
            which.clone(),
        ),
        selector if selector.starts_with("claude-code:") => {
            (ProviderKind::ClaudeCode, selector.to_owned())
        }
        selector if selector.starts_with("codex:") => (ProviderKind::Codex, selector.to_owned()),
        other => anyhow::bail!("unknown provider {other}; use claude-code or codex"),
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req = ChatRequest {
        model: ModelEntry {
            provider,
            id: model_id,
        },
        system: String::new(),
        messages: vec![Message::User(
            "Reply with exactly the word: pong. Do not use any tools.".into(),
        )],
        tools: Vec::new(),
        policy: RequestPolicy::Interactive,
        force_full_handoff: false,
    };
    tokio::spawn(providers::stream_chat(Config::load(), req, tx));

    println!("-- {which} --");
    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            ChatEvent::TextDelta(t) => {
                print!("{t}");
                text.push_str(&t);
            }
            ChatEvent::Notice(message) => println!("[note] {message}"),
            ChatEvent::ToolActivity { summary, .. } => println!("[activity] {summary}"),
            ChatEvent::Completed { usage, .. } => {
                println!("\n[completed] usage={usage:?}");
                break;
            }
            ChatEvent::Error(e) => anyhow::bail!("error: {e}"),
        }
    }
    anyhow::ensure!(!text.trim().is_empty(), "no text received");
    println!("-- ok: streamed {} chars --", text.len());
    Ok(())
}
