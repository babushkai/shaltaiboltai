# shaltaiboltai

> [!NOTE]
> This project is built along with Claude Fable, which is now regulated by US government. 

A multi-provider agentic coding TUI in Rust with a Codex-style authority shell: typed startup policy, OS-enforced command sandboxing, scoped approvals, `/permissions`, `/status`, `/init`, and a polished Ink/Paper interface. Chat with Anthropic, OpenAI, OpenRouter, Ollama, Claude Code, or Codex and switch providers without changing the active safety contract.

## Install

Prebuilt binary (macOS arm64/x86_64, Linux x86_64), no Rust toolchain needed:

```sh
curl -fsSL https://github.com/babushkai/shaltaiboltai/releases/latest/download/install.sh | sh
```

It installs to `~/.local/bin`; override with `SHALTAI_INSTALL_DIR`, or pin a tag with `SHALTAI_VERSION=v0.1.0`. Then run `shaltaiboltai`.

From source (Rust 1.88+; constrained shell execution is supported on macOS and Linux):

```sh
cargo install --git https://github.com/babushkai/shaltaiboltai --locked
```

Or clone and run locally:

```sh
git clone https://github.com/babushkai/shaltaiboltai.git
cd shaltaiboltai
cargo run --release
```

Set at least one API key, start Ollama, or sign in with `codex login` / the Claude CLI first. For example:

```sh
export OPENROUTER_API_KEY="your-key"
cargo run --release -- --model openrouter:anthropic/claude-sonnet-4.6
```

On Linux, constrained `run_command` calls require Bubblewrap at `/usr/bin/bwrap` or `/bin/bwrap`. The app fails closed if the backend is unavailable; install the `bubblewrap` package with your distribution's package manager. Ubuntu 24.04 hosts that restrict unprivileged user namespaces must also load the distribution's capability-dropping `bwrap-userns-restrict` AppArmor profile from `apparmor-profiles`; do not disable AppArmor globally. Shaltaiboltai never retries a constrained command outside the boundary. macOS uses the system Seatbelt backend.

Providers are auto-discovered at startup:

| Provider | Enabled by | Models |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Claude (Fable, Opus, Sonnet, Haiku) |
| OpenAI | `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | fetched from `/v1/models` |
| OpenRouter | `OPENROUTER_API_KEY` (+ optional `OPENROUTER_BASE_URL`) | `openrouter/auto` immediately, then an account-filtered ranked catalog of tool-capable text models |
| Ollama | running locally (`OLLAMA_HOST`, default `http://localhost:11434`) | fetched from `/api/tags` |
| Claude Code | the `claude` CLI installed and signed in (Unix) | CLI default plus the documented Fable, Opus, and Sonnet aliases |
| Codex | the `codex` CLI installed and signed in (Unix) | CLI default; exact model IDs can be entered explicitly |

No keys are needed for Ollama — if it is running, its models just show up. OpenRouter is a separate provider rather than an `OPENAI_BASE_URL` alias, so its provider identity and metered-call boundary stay visible in the model picker and team review. OpenRouter can load-balance a selected model across hosting endpoints and fallbacks; Shaltaiboltai does not claim that the model-author slug identifies the hosting endpoint. Your OpenRouter [routing and privacy settings](https://openrouter.ai/docs/guides/routing/provider-selection) apply.

`/model openrouter:<author>/<model>` selects any exact OpenRouter model ID even when it is not in the curated list. OpenRouter-owned routers such as `openrouter/auto` and `openrouter/free`, plus rolling `~...` aliases, remain available for solo work but are intentionally excluded from team assignments because the concrete model can vary between calls.

### Subscription providers (no API key)

If [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or [Codex](https://github.com/openai/codex) is installed and signed in, subscription-backed choices appear in the picker (Claude Pro/Max, ChatGPT Plus/Pro) instead of using a metered API key. `claude-code` and `codex` mean **use that CLI's configured default**. Explicit selectors pass a model override to the CLI: for example, `/model claude-code:sonnet` uses Claude's latest Sonnet alias and `/model codex:gpt-5.6-sol` requests that exact Codex model. Claude aliases come from the installed CLI's documented interface; Codex shows the current list-visible upstream models from its bundled catalog without executing a CLI during discovery. You can enter any provider-qualified full model ID directly with `/model`, even when it is not listed—the CLI remains authoritative about account access.

CLI discovery is metadata-only and currently Unix-only. At startup the app snapshots the first absolute `claude` and `codex` candidate on `PATH`, rejects candidates inside the workspace or a temporary writable root, and never executes them during discovery. A rejected first hit fails closed instead of falling through to another executable. Restart after installing a CLI or changing `PATH`; `/refresh` does not replace this executable snapshot. The executable identity is revalidated immediately before every launch. Non-Unix CLI bridging stays unavailable until the app can bind a stable executable identity there; API providers remain available.

We never read or store a subscription token — each CLI owns its authentication. Shaltaiboltai launches a fresh headless child (`claude --print --output-format stream-json`, interactive `codex exec --json`) and renders its event stream. Read-only Codex planning/workers use the separately attested `codex app-server --stdio` path described below. The child drives its own tool loop, while Shaltaiboltai maps the active permission snapshot onto every launch instead of inheriting ambient authority:

| Permission | Codex CLI | Claude Code CLI |
|---|---|---|
| Read Only | attested workspace-read profile in an isolated one-request Codex home | `plan` + safe mode, read/search tools only |
| Ask for approval | `workspace-write`, no network, canonical roots only | `plan` + safe mode, read/search tools only |
| Full Access | `danger-full-access` | `bypassPermissions` |

Headless children cannot route an inner approval prompt back into this TUI, so their inner approval policy is fail-closed. Codex Read Only uses a default-deny permission profile with minimal runtime reads, the captured workspace roots as read-only, and network disabled; legacy `--sandbox read-only` is never used for this path. Before launch, Shaltaiboltai creates a private one-request `CODEX_HOME`, links only a safe regular `auth.json` from the signed-in home, and explicitly denies model tools both authentication locations. The isolated home removes ordinary user configuration and trust state; pinned session overrides prevent ambient user or ordinarily untrusted project configuration, global Codex instructions, skills, MCP/browser surfaces, shell snapshots, secret-bearing environment inheritance, and nested Codex agents from widening the advisory child. Managed/system configuration can still be discovered, so the effective boundary and every repository instruction source reported by app-server are independently attested. Repository `AGENTS.md` instructions inside the reviewed workspace may still apply. The app-server must report Codex CLI **0.152.1**, the exact model/provider, canonical workspace roots, no model fallback, `never` approvals, network denial, and the fresh permission-profile ID during `initialize` + `thread/start`; only then does Shaltaiboltai send the first model-bearing `turn/start`. Any mismatch fails with zero planner/worker model calls.

Codex team/read-only execution currently requires exactly CLI 0.152.1 and a safe file-backed login at `$CODEX_HOME/auth.json`; keychain-only login state and unreviewed newer protocol versions fail closed. Interactive Codex use remains available through the normal `codex exec` path. Claude Code cannot express the protected-path carve-outs required by the normal workspace policy, so constrained Claude turns are advisory-only; use the Anthropic API provider for app-owned, policy-mediated edits. Claude Code editing requires the explicit Full Access preset. Full Access is available only through the startup policy or the two-step `/permissions` confirmation; there are no provider-specific bypass switches. An unavailable model is reported as an error and is never silently replaced.

Each CLI request starts an ephemeral fresh process with an explicit handoff of this app's conversation history. Read-only Codex keeps initialization, attestation, and its one turn inside that same owned app-server process. Nothing attaches to an unrelated “last” CLI session or persists the handoff in CLI session storage. Images are represented in the handoff but their binary contents aren't forwarded to these providers yet (they work with the API providers).

## Startup policy

The default is workspace-write with on-request approvals. Startup options expose a supported Codex-style subset:

```text
shaltaiboltai [OPTIONS] [PROMPT]

-C, --cd <DIR>                    Set the working directory
    --add-dir <DIR>               Add a writable workspace root (repeatable)
-s, --sandbox <MODE>              read-only | workspace-write | danger-full-access
-a, --ask-for-approval <POLICY>   on-request | never
-m, --model <MODEL>               Select the initial provider/model
-i, --image <PATH,...>            Attach startup images (repeatable)
    --full-auto                   workspace-write with on-request approval
    --dangerously-bypass-approvals-and-sandbox
    --no-alt-screen               Keep terminal scrollback visible
```

`--dangerously-bypass-approvals-and-sandbox` is intentionally explicit: it enables full disk and network access without approval. Relative `--add-dir` paths are resolved from the selected `--cd` directory; Read Only ignores them with a warning because it has no writable roots. A positional prompt is submitted once after its selected model becomes available. Typed, queued, and dropped relative image paths all resolve from `--cd`.

In workspace mode, protected `.git`, `.agents`, and `.codex` paths are carved back to read-only by pathname; macOS and Windows classification also reserves ASCII case variants. Direct constrained `write_file` and `edit_file` calls reject multiply-linked files before truncation on Unix and Windows.

## Keys & commands

Typing `/` opens a command menu above the input (filters as you type, `Up`/`Down` to navigate, `Tab` to complete, `Enter` to run, `Esc` to dismiss). `/permissions` changes the next-turn authority, `/status` shows the immutable snapshot governing an active turn, and `/init` asks the selected model to create repository-scoped `AGENTS.md` guidance through the normal tool and approval path. Commands also take arguments directly: `/theme paper`, `/model qwen`, `/model codex:gpt-5.6-sol`, `/model openrouter:openai/gpt-5.4`, `/team 3`, and `/refresh`. The statusline prioritizes active state on narrow terminals, then adds model, project, linked-worktree branch, policy, and live context usage as space allows.

The composer stays live while a response streams or a tool runs. Press `Enter` to queue one next message; it is sent automatically only after the current turn finishes cleanly (and after any context compaction). Cancellation, provider errors, truncation, or persistence/compaction failures restore that message to the composer instead. The one-message queue locks after capture, and slash commands wait until the active turn ends.

### Team orchestration

`/team [2-4]` arms the next prompt for one coordinated run (default: 3 workers); `/team off` returns it to a normal solo prompt. Shaltaiboltai is the lead agent. The mascot appears only on genuinely large, quiet canvases; common 120×36 and narrower layouts preserve transcript hierarchy instead of sacrificing half the viewport to decoration. Submitting the armed prompt immediately sends one read-only planning request; a CLI planner may inspect workspace files in its read-only mode before the confirmation appears. The overlay shows the exact planner, task summaries, and every exact worker model. Press `Tab` to focus the review, then `y` or `Enter` to start; `n` or `Esc` cancels the plan.

After confirmation, workers run concurrently under a read-only policy and cannot use Shaltaiboltai's mutating tools. API workers whose models support tools get a bounded, app-owned read-only repository tool loop. Claude Code uses safe mode with `Read`, `Glob`, and `Grep`; explicit Codex models use the default-deny workspace-read permission profile described above. This makes Codex-only teams possible: select an exact row such as `codex:gpt-5.6-sol`, run `/team 3`, and submit the root task. A Codex lead pins that same Codex model for the planner, every worker, and synthesis even when other providers are installed. A local model without tool support instead reasons from the supplied conversation. Every selected advisory provider receives the text conversation; Codex may apply repository instructions found inside the scoped workspace but excludes ambient global instructions, OpenRouter sends text to provider endpoint(s) chosen under the account's routing/privacy settings, and metered APIs may bill every planner, worker, tool-loop, and synthesis request. Review the exact provider/model rows and scope disclosure before sharing it. Images are omitted from team fan-out; use `/team off` for a vision prompt. Shaltaiboltai waits for every worker request to finish, synthesizes their reports, and becomes the only agent allowed to edit through the normal approval or CLI sandbox rules. This prevents concurrent team edits; it cannot prevent an unrelated process or person from changing the workspace at the same time.

The selected lead provider/model is pinned for synthesis and is also the planner when it has an enforceable workspace-read boundary. Because bare `codex` / `claude-code` selectors delegate model choice to those CLIs, while OpenRouter-owned routers and `~...` aliases can vary their concrete model, team mode asks you to choose an exact model first. Non-Codex leads may deliberately use other providers for diversity; their exact provider/selectors are snapshotted in the plan and never silently replaced. Reaching the confirmation has already used **1 planner call**; accepting it adds at least **N worker calls + 1 synthesis call**, with additional calls possible when workers use read tools or the lead later uses tools. `Esc` cancels planning, workers, or streaming synthesis and terminates their owned requests. Once synthesis reaches a normal tool approval, the usual approval controls apply: `Tab` focuses the review, then `n` or `Esc` denies it. During the worker phase the normal one-message lookahead remains available. Authentication failures identify the exact planner; for an expired Codex session, run `codex login` once and retry instead of repeatedly launching the same failed plan.

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
| `/permissions` | choose Read Only, Ask for approval, or Full Access; Full Access requires a second confirmation |
| `/status` | show model, provider, enforcement, workspace, policy, network, instructions, session, and usage |
| `/init` | run a hidden synthetic turn that creates `AGENTS.md` through normal tools and policy |
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

`/theme` opens a live-preview picker (`Up`/`Down` to try, `Enter` to keep, `Esc` to revert), and the choice persists across runs. `ink` is the default; `paper` is its warm light companion. Their base, surface, elevated, hover, typography, border, accent, success, warning, error, code, and indigo tokens are transferred exactly from the original TypeScript Ink & Paper projects. Legacy palettes remain available: `mocha`, `tokyo-night`, `rose-pine`, `nord`, `gruvbox`, `latte`, and `terminal`. Set an initial theme with `theme = "paper"` in config.toml.

## Sessions & compaction

Conversations auto-save after every completed turn to `~/Library/Application Support/shaltaiboltai/sessions/` (or `$SHALTAIBOLTAI_DATA_DIR/sessions`); resume any of them with `/resume`. Sessions are project-scoped: the picker lists the current directory's sessions first, with sessions from other projects badged by their path. When the context grows past a threshold (`compact_threshold_chars`, default 80,000 chars ≈ 20k tokens) the conversation is summarized in the background by the current model and replaced with the summary, so long sessions keep working on small-context local models too. `/compact` triggers it manually; the status bar shows the live context size.

## Tools & permissions

The agent has seven tools:

- **Read/search** — `read_file`, `list_directory`, `grep` (regex, gitignore-aware), and `glob`. Constrained modes allow reads while every target is canonicalized and rechecked immediately before I/O.
- **Edit/run** — `write_file`, `edit_file` (unique exact replacement), and `run_command`. Workspace-path writes run without interruption in the default preset; protected-path and outside writes require explicit authority. File approvals show the exact canonical target and a bounded unified diff.

The presets are deliberately distinct:

- **Read Only** — reads and sandboxed read-only commands are allowed; edits, network, and commands requesting execution outside the sandbox ask first.
- **Ask for approval** — reads, workspace-path edits, and constrained commands are allowed; network, outside writes, and protected `.git`/`.agents`/`.codex` path writes ask first.
- **Full Access** — disk and network access are enabled without prompts. The TUI opens this mode with “Go back” selected and requires a deliberate second action.

An approval is bound to the exact policy generation and canonical target, search, or command displayed. Retargeting a symlink invalidates the review instead of rebinding the click. Session grants are cleared when authority changes, on `/new`, on `/resume`, and at exit.

The broker is a boundary for model-initiated work, not for another hostile process already running as the same OS user. Such a process can race pathname-based file operations or workspace mount setup. OS pathname sandboxes also cannot distinguish a pre-existing workspace hard-link alias from its outside or protected inode, although the app-owned direct writers reject multiply-linked files. A deliberately daemonized Unix child can leave its original process group and outlive group cleanup. Sanitize untrusted local workspaces before allowing shell commands, and use a separate OS account, VM, or container when hard links, detached daemons, or same-user adversaries are in scope.

Constrained shell commands never degrade to a raw shell: macOS uses Seatbelt; Linux uses a read-only Bubblewrap filesystem view, isolated namespaces, and a seccomp network filter. Workspace roots and temporary roots are the only write mounts, with protected paths carved back to read-only by pathname. Command stdin is closed, each child owns a process group whose remaining members are terminated on completion or cancellation, commands time out after 60 seconds, stdout/stderr are drained with bounded memory, and returned tool output is capped at 32 KB.

Repository instructions are loaded from the Git root down to the selected working directory. At each level, `AGENTS.override.md` replaces `AGENTS.md`; the combined context is capped and the exact loaded paths appear in `/status`.

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
# openrouter_api_key = "sk-or-..."
# openrouter_base_url = "https://openrouter.ai/api/v1"
# ollama_host = "http://localhost:11434"
# theme = "paper"                                  # default is ink
# reduced_motion = false                           # freeze the mascot pose while retaining status text
```

Set `SHALTAIBOLTAI_REDUCED_MOTION=1` for the same motion-free behavior without changing the file.

## Development

`cargo run --example smoke [model_id]` exercises the provider layer end-to-end (discovery → streaming → tool call → result → final answer) without the TUI.

Architecture: `src/providers/` speaks each API natively over reqwest (SSE for Anthropic and the shared OpenAI/OpenRouter wire format, NDJSON for Ollama) and normalizes everything to one `Message`/`ToolCall`/`ChatEvent` model. `src/policy.rs` owns typed authority and canonical path classification; `src/sandbox.rs` turns that authority into fail-closed OS process boundaries; `src/tools.rs` reassesses and binds every execution. `src/app.rs` owns immutable turn snapshots, approvals, orchestration, and cancellation. `src/ui.rs` uses dirty-entry caching plus cumulative line offsets, so redraw cost follows changed and visible content rather than transcript length. OpenRouter discovery uses its authenticated user catalog so account privacy/provider restrictions are respected, then defensively validates and caps the ranked tool-capable text rows at 40 before publishing them behind the immediate auto-router row.

The render suite exercises idle, help, status, permissions, Full Access confirmation, and tool approvals down to 40×12, including Ink/Paper semantic styles. CI runs formatting, installer ShellCheck, strict Clippy, all targets, the production Linux sandbox integration test, macOS/Linux release builds, and a Windows compile plus hard-link boundary test before release artifacts are cut.

Provider details: transient failures (429/5xx) are retried with backoff honoring `Retry-After`; Anthropic requests use prompt caching (system, tools, and conversation tail breakpoints); truncated responses (`max_tokens`/`length`) are surfaced in the transcript; the status bar shows real token usage reported by the provider.
