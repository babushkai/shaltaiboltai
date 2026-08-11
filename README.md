# shaltaiboltai

> [!NOTE]
> This project is built along with Claude Fable, which is now regulated by US government. 

A Claude Code-style agentic coding TUI in Rust. Chat with a model, let it read/write files and run shell commands (with approval), and switch between providers — Anthropic, OpenAI (or any OpenAI-compatible endpoint), and local Ollama — mid-conversation.

## Install

Prebuilt binary (macOS arm64/x86_64, Linux x86_64), no Rust toolchain needed:

```sh
curl -fsSL https://github.com/babushkai/shaltaiboltai/releases/latest/download/install.sh | sh
```

It installs to `~/.local/bin`; override with `SHALTAI_INSTALL_DIR`, or pin a tag with `SHALTAI_VERSION=v0.1.0`. Then run `shaltaiboltai`.

From source (any platform with a Rust toolchain):

```sh
cargo install --git https://github.com/babushkai/shaltaiboltai --locked
```

Or clone and `cargo run --release`.

Providers are auto-discovered at startup:

| Provider | Enabled by | Models |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Claude (Fable, Opus, Sonnet, Haiku) |
| OpenAI | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | fetched from `/v1/models` |
| Ollama | running locally (`OLLAMA_HOST`, default `http://localhost:11434`) | fetched from `/api/tags` |
| Claude Code | the `claude` CLI installed and signed in | CLI default plus the documented Fable, Opus, and Sonnet aliases |
| Codex | the `codex` CLI installed and signed in | CLI default plus exact models advertised by the local Codex catalog |

No keys needed for Ollama — if it's running, its models just show up. Models without tool support automatically fall back to plain chat.

### Subscription providers (no API key)

If [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or [Codex](https://github.com/openai/codex) is installed and signed in, subscription-backed choices appear in the picker (Claude Pro/Max, ChatGPT Plus/Pro) instead of using a metered API key. `claude-code` and `codex` mean **use that CLI's configured default**. Explicit selectors pass a model override to the CLI: for example, `/model claude-code:sonnet` uses Claude's latest Sonnet alias and `/model codex:gpt-5.6-sol` requests that exact Codex model. The Codex rows come from its local machine-readable model catalog; Claude aliases come from the installed CLI's documented interface. You can also enter a provider-qualified full model ID directly with `/model`, even when it is not listed—the CLI remains authoritative about access.

We never see or store a token — the CLI owns its own auth. shaltaiboltai spawns it headless (`claude --print --output-format stream-json`, `codex exec --json`) and renders its event stream, so it runs as a **sub-agent**: the CLI drives its own tool loop (read/edit/run) and you watch its activity in the transcript. shaltaiboltai's own tools and approval flow don't apply to these providers — the CLI's own permission/sandbox model does. An unavailable model is reported as an error; shaltaiboltai never silently falls back to another model.

Safe defaults, each opt-out via config.toml:

- **Claude Code** runs with `--permission-mode acceptEdits` (reads/edits files autonomously; shell commands auto-denied since there's no interactive prompt). `claude_code_bypass_permissions = true` lets it run shell commands unsupervised.
- **Codex** runs with `--sandbox workspace-write` (OS-sandboxed: edits and commands confined to the working directory, no network). `codex_full_access = true` removes the sandbox (`danger-full-access`).

Each CLI request starts an ephemeral fresh process with an explicit handoff of this app's conversation history. That avoids attaching to an unrelated “last” CLI session in the same directory or duplicating the handoff in CLI session storage. Images are represented in the handoff but their binary contents aren't forwarded to these providers yet (they work with the API providers).

## Keys & commands

Typing `/` opens a command menu above the input (filters as you type, `Up`/`Down` to navigate, `Tab` to complete, `Enter` to run, `Esc` to dismiss). Commands take arguments directly: `/theme nord` switches and persists the theme, `/model qwen` jumps to the unique match or opens the picker pre-filtered, `/model claude-code:opus` and `/model codex:gpt-5.6-sol` select subscription CLI models, `/team 3` arms a coordinated run, and `/refresh` rediscovers providers. The statusline prioritizes the active state on narrow terminals, then adds the model, project, linked-worktree branch, and live context usage as space allows.

The composer stays live while a response streams or a tool runs. Press `Enter` to queue one next message; it is sent automatically only after the current turn finishes cleanly (and after any context compaction). Cancellation, provider errors, truncation, or persistence/compaction failures restore that message to the composer instead. The one-message queue locks after capture, and slash commands wait until the active turn ends.

### Team orchestration

`/team [2-4]` arms the next prompt for one coordinated run (default: 3 workers); `/team off` returns it to a normal solo prompt. Shaltaiboltai is the lead agent—the small mascot in the transcript title dances while it works. Submitting the armed prompt immediately sends one read-only planning request; a CLI planner may inspect workspace files in its read-only mode before the confirmation appears. The overlay shows the exact planner, task summaries, and every exact worker model. Press `Tab` to focus the review, then `y` or `Enter` to start; `n` or `Esc` cancels the plan.

After confirmation, workers run concurrently under a read-only policy and cannot use Shaltaiboltai's mutating tools. API workers whose models support tools get a bounded, app-owned read-only repository tool loop. Claude Code uses safe mode with `Read`, `Glob`, and `Grep`; a local model without tool support instead reasons from the supplied conversation. Codex CLI is deliberately excluded from planning and worker assignments: its `read-only` sandbox blocks writes but allows reads outside the workspace. An explicitly selected Codex model can still be the post-confirmation lead that synthesizes and edits under its normal sandbox. Every selected advisory provider receives the text conversation, so review the provider/model rows before sharing it. Images are omitted from team fan-out; use `/team off` for a vision prompt. Shaltaiboltai waits for every worker request to finish, synthesizes their reports, and becomes the only agent allowed to edit through the normal approval or CLI sandbox rules. This prevents concurrent team edits; it cannot prevent an unrelated process or person from changing the workspace at the same time.

The selected lead provider/model is pinned for synthesis and is also the planner when it has an enforceable workspace-read boundary. With a Codex lead, the overlay identifies the safe alternate planner; team mode requires at least one such advisory model to be available. Because a bare `codex` or `claude-code` selector delegates model choice to that CLI, team mode asks you to choose an explicit row such as `codex:gpt-5.6-sol` or `claude-code:sonnet` first. Worker assignments may deliberately use other providers for diversity; their exact provider/selectors are snapshotted in the plan and never silently replaced. Reaching the confirmation has already used **1 planner call**; accepting it adds at least **N worker calls + 1 synthesis call**, with additional calls possible when workers use read tools or the lead later uses tools. `Esc` cancels planning, workers, or streaming synthesis and terminates their owned requests. Once synthesis reaches a normal tool approval, the usual approval controls apply: `Tab` focuses the review, then `n` or `Esc` denies it. During the worker phase the normal one-message lookahead remains available.

| Key | Action |
|---|---|
| `Enter` | send a message, or queue one next message while the agent is working |
| `Ctrl+V` | attach an image from the clipboard (macOS) — or just drag/type an image path into the message |
| `Ctrl+X` | clear staged attachments without clearing the message or cancelling active work |
| `Alt+Enter` | insert newline (multi-line input; pasting multi-line text also works) |
| `Up` / `Down` | recall previous prompts (when the input is empty), shell-style |
| `Ctrl+U` | clear the input |
| `Ctrl+P` or `/model` | model picker (type to filter, `Enter` to select) |
| `/team [2-4\|off]` | arm one lead-and-workers run, or turn it off |
| `F1` or `/help` | focused keyboard guide |
| `Esc` | cancel an in-flight response or running tool (including CLI sub-agents); in a tool approval, first focus its review controls and then deny the tool |
| `Tab`, then `y` / `a` / `n` | focus a newly arrived tool approval, then approve once / allow only this path, search, or exact command for the current session / deny |
| `Tab`, then `y` / `Enter`; `n` / `Esc` | review and start a team plan; or cancel it |
| `Up` / `Down`, `PgUp` / `PgDn` | scroll a tool approval preview while its actions stay pinned |
| `PgUp` / `PgDn` or mouse wheel | scroll transcript |
| `Ctrl+Home` / `Ctrl+End` | jump to the oldest / latest transcript line |
| `/resume` | pick a saved session to continue |
| `/new` or `/clear` | start a new session (the old one stays saved) |
| `/compact` | summarize the conversation to shrink context |
| `/refresh` | rediscover available models |
| `Ctrl+C` or `/quit` | exit; if a next message is queued, the first Ctrl+C restores it and asks for confirmation |

The trackpad / mouse wheel scrolls the transcript. Because the TUI captures mouse reporting for this, hold `Option` (macOS) or `Shift` (Linux/Windows) while dragging to use the terminal's native text selection.

Messages can include images for vision models: press `Ctrl+V` to stage the clipboard image (screenshots, copied images), or reference a `.png`/`.jpg`/`.gif`/`.webp` path in your message (drag-and-drop onto the terminal works — escaped and quoted paths are handled). Staged attachments show in the input border; `Ctrl+X` clears them at any editable composer, while `Esc` also clears them when no work is active. Images go out as Anthropic image blocks, OpenAI data-URLs, or Ollama's native `images` field, capped at 5MB each.

The transcript uses a quiet activity rail with explicit `YOU`, `ASSISTANT`, and tool-state headers, so long agent runs remain scannable. Assistant responses render markdown (heading hierarchy, bold/italic, accent-bulleted lists, styled blockquotes, and fenced code as full-width surface cards).

## Themes

`/theme` opens a live-preview picker (Up/Down to try, Enter to keep, Esc to revert) — the choice persists across runs. Built-in palettes: `mocha` (default), `tokyo-night`, `rose-pine`, `nord`, `gruvbox`, `latte` (light), and `terminal` (plain ANSI, keeps your terminal's own colors — use this if your emulator lacks truecolor). Each theme defines a base background, an elevated surface tone (input field, status bar, code cards, overlays), and tiered borders, so the UI has depth rather than flat accents. Set an initial theme with `theme = "nord"` in config.toml.

## Sessions & compaction

Conversations auto-save after every completed turn to `~/Library/Application Support/shaltaiboltai/sessions/` (or `$SHALTAIBOLTAI_DATA_DIR/sessions`); resume any of them with `/resume`. Sessions are project-scoped: the picker lists the current directory's sessions first, with sessions from other projects badged by their path. When the context grows past a threshold (`compact_threshold_chars`, default 80,000 chars ≈ 20k tokens) the conversation is summarized in the background by the current model and replaced with the summary, so long sessions keep working on small-context local models too. `/compact` triggers it manually; the status bar shows the live context size.

## Tools & permissions

The agent has seven tools:

- **Read-only** — `read_file`, `list_directory`, `grep` (regex content search, gitignore-aware), `glob` (find files by pattern). Auto-approved **only inside the working directory** after resolving symlinks; reads outside it (dotfiles, other projects, `/etc`…) always prompt before contents are sent to a provider.
- **Mutating** — `write_file`, `edit_file` (exact find/replace, must match uniquely), `run_command`. Always prompt; the approval dialog shows a unified diff of what a file change will do. `a` remembers only the displayed path/search or exact command for the current conversation; grants are cleared by `/new`, `/resume`, and exit.

Commands time out after 60s and tool output is capped at 32 KB. If `AGENTS.md` or `CLAUDE.md` exists in the working directory it is loaded into the system prompt automatically.

## Config (optional)

`~/.config/shaltaiboltai/config.toml` — environment variables take precedence:

```toml
default_model = "qwen3.5:latest"
# default_model = "claude-code:sonnet" # Claude Code subscription alias
# default_model = "codex:gpt-5.6-sol"   # exact Codex subscription model
# compact_threshold_chars = 80000  # auto-compact context beyond this size
# ollama_num_ctx = 16384           # context window requested from Ollama (its default is ~4k)
# anthropic_api_key = "sk-ant-..."
# openai_api_key = "sk-..."
# openai_base_url = "https://api.openai.com/v1"   # any OpenAI-compatible server
# ollama_host = "http://localhost:11434"
# claude_code_bypass_permissions = false          # let the claude-code sub-agent run shell commands unsupervised
# codex_full_access = false                        # remove the codex sub-agent's OS sandbox (danger-full-access)
```

## Development

`cargo run --example smoke [model_id]` exercises the provider layer end-to-end (discovery → streaming → tool call → result → final answer) without the TUI.

Architecture: `src/providers/` speaks each API natively over reqwest (SSE for Anthropic/OpenAI, NDJSON for Ollama) and normalizes everything to one `Message`/`ToolCall`/`ChatEvent` model. Static models appear immediately and dynamic providers publish independently as their probes finish. `src/app.rs` owns the agent loop, approval state machine, and cancellable request/compaction tasks; `src/ui.rs` uses dirty-entry caching plus cumulative line offsets, so each redraw parses only changed content and jumps directly to the visible viewport as conversations grow.

Provider details: transient failures (429/5xx) are retried with backoff honoring `Retry-After`; Anthropic requests use prompt caching (system, tools, and conversation tail breakpoints); truncated responses (`max_tokens`/`length`) are surfaced in the transcript; the status bar shows real token usage reported by the provider.
