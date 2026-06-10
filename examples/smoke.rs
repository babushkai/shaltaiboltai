//! Non-interactive smoke test of the provider layer:
//! `cargo run --example smoke [model_id]`
//! Discovers models, then runs one agentic round (with a tool nudge) against
//! the chosen model, auto-approving tool calls.

use shaltaiboltai::config::Config;
use shaltaiboltai::providers::{self, ChatEvent, ChatRequest, Message};
use shaltaiboltai::tools;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load();
    let models = providers::discover_models(config.clone()).await;
    println!("discovered {} models:", models.len());
    for m in &models {
        println!("  {:<10} {}", m.provider.label(), m.id);
    }

    let want = std::env::args().nth(1);
    let Some(model) = models
        .iter()
        .find(|m| want.as_deref().is_none_or(|w| m.id == w))
        .cloned()
    else {
        println!("no matching model; done.");
        return Ok(());
    };
    println!("\n-- chatting with {} --", model.id);

    let mut history = vec![Message::User(
        "List the files in the current directory using the list_directory tool, then summarize in one sentence.".into(),
    )];

    for round in 0..4 {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let req = ChatRequest {
            model: model.clone(),
            system: "You are a test agent. Use tools when asked.".into(),
            messages: history.clone(),
            tools: tools::definitions(),
        };
        tokio::spawn(providers::stream_chat(config.clone(), req, tx));

        let mut text = String::new();
        let mut calls = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                ChatEvent::TextDelta(t) => {
                    print!("{t}");
                    text.push_str(&t);
                }
                ChatEvent::Completed { tool_calls } => calls = tool_calls,
                ChatEvent::Error(e) => anyhow::bail!("provider error: {e}"),
            }
        }
        println!();
        history.push(Message::Assistant {
            text,
            tool_calls: calls.clone(),
        });
        if calls.is_empty() {
            println!("-- round {round}: final answer, smoke test passed --");
            return Ok(());
        }
        for call in calls {
            println!("[tool] {}", tools::describe(&call));
            let (content, is_error) = tools::execute(&call).await;
            println!(
                "[result, err={is_error}] {}",
                content.lines().next().unwrap_or("")
            );
            history.push(Message::ToolResult {
                call_id: call.id,
                name: call.name,
                content,
                is_error,
            });
        }
    }
    anyhow::bail!("model never produced a final answer in 4 rounds")
}
