# shaltaiboltai

A Claude Code-style agentic coding TUI in Rust. Chat with a model, let it read/write files and run shell commands (with approval), and switch between providers — Anthropic, OpenAI (or any OpenAI-compatible endpoint), and local Ollama — mid-conversation.

## Run

```sh
cargo run --release
```

Providers are auto-discovered at startup:

| Provider | Enabled by | Models |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Claude (Fable, Opus, Sonnet, Haiku) |
| OpenAI | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | fetched from `/v1/models` |
| Ollama | running locally (`OLLAMA_HOST`, default `http://localhost:11434`) | fetched from `/api/tags` |

No keys needed for Ollama — if it's running, its models just show up. Models without tool support automatically fall back to plain chat.

## Keys & commands

| Key | Action |
|---|---|
| `Enter` | send message |
| `Alt+Enter` | insert newline (multi-line input; pasting multi-line text also works) |
| `Ctrl+P` or `/model` | model picker (type to filter, `Enter` to select) |
| `Esc` | cancel an in-flight response |
| `y` / `a` / `n` | approve / approve-all / deny a tool call |
| `PgUp` / `PgDn` | scroll transcript |
| `/resume` | pick a saved session to continue |
| `/new` or `/clear` | start a new session (the old one stays saved) |
| `/compact` | summarize the conversation to shrink context |
| `Ctrl+C` or `/quit` | exit |

Assistant responses render markdown (headings, bold/italic, lists, blockquotes, fenced code).

## Sessions & compaction

Conversations auto-save after every completed turn to `~/Library/Application Support/shaltaiboltai/sessions/` (or `$SHALTAIBOLTAI_DATA_DIR/sessions`); resume any of them with `/resume`. When the context grows past a threshold (`compact_threshold_chars`, default 80,000 chars ≈ 20k tokens) the conversation is summarized in the background by the current model and replaced with the summary, so long sessions keep working on small-context local models too. `/compact` triggers it manually; the status bar shows the live context size.

## Tools

The agent has four tools: `read_file`, `list_directory` (auto-approved), `write_file`, `run_command` (require approval; `a` auto-approves for the session). Commands time out after 60s and output is capped at 32 KB.

## Config (optional)

`~/.config/shaltaiboltai/config.toml` — environment variables take precedence:

```toml
default_model = "qwen3.5:latest"
# compact_threshold_chars = 80000  # auto-compact context beyond this size
# anthropic_api_key = "sk-ant-..."
# openai_api_key = "sk-..."
# openai_base_url = "https://api.openai.com/v1"   # any OpenAI-compatible server
# ollama_host = "http://localhost:11434"
```

## Development

`cargo run --example smoke [model_id]` exercises the provider layer end-to-end (discovery → streaming → tool call → result → final answer) without the TUI.

Architecture: `src/providers/` speaks each API natively over reqwest (SSE for Anthropic/OpenAI, NDJSON for Ollama) and normalizes everything to one `Message`/`ToolCall`/`ChatEvent` model; `src/app.rs` owns the agent loop and approval state machine; `src/ui.rs` is pure rendering.
