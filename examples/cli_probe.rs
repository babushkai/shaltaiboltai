//! Live end-to-end check of a sub-agent CLI provider through the real
//! provider pipeline (spawn → stream → ChatEvent), without the TUI.
//! Usage: `cargo run --example cli_probe -- claude-code|codex`
//! This bills the corresponding subscription one trivial read-only turn.

use shaltaiboltai::config::Config;
use shaltaiboltai::providers::{self, ChatEvent, ChatRequest, Message, ModelEntry, ProviderKind};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "codex".into());
    let provider = match which.as_str() {
        "claude-code" | "claude" => ProviderKind::ClaudeCode,
        "codex" => ProviderKind::Codex,
        other => anyhow::bail!("unknown provider {other}; use claude-code or codex"),
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req = ChatRequest {
        model: ModelEntry {
            provider,
            id: which.clone(),
        },
        system: String::new(),
        messages: vec![Message::User(
            "Reply with exactly the word: pong. Do not use any tools.".into(),
        )],
        tools: Vec::new(),
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
