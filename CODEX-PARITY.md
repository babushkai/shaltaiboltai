# Codex parity roadmap

This project tracks the OpenAI Codex CLI interface and interaction model while
retaining Shaltaiboltai's own name, Humpty Dumpty mascot, multi-provider
runtime, and approval boundaries.

## Audited baseline

- Upstream source: [`openai/codex` at `bd0d4a2`](https://github.com/openai/codex/tree/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df)
- Audit date: 2026-08-23
- Product reference: [Codex CLI features](https://learn.chatgpt.com/docs/codex/cli)
- Command reference: [Codex developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)

The upstream repository is Apache-2.0. Shaltaiboltai reimplements interaction
patterns rather than copying source. OpenAI and Codex names, marks, and product
copy are not reused as Shaltaiboltai branding.

## Delivered in `feat/codex-parity`

- A bounded 16-message `VecDeque` follow-up queue with FIFO dispatch.
- Exactly one queued message promoted after each clean turn.
- Frozen per-message model selection and staged/referenced attachment bytes.
- `Tab` to queue while work runs; `Enter` remains a compatibility alias.
- `Option+Up` (`Alt+Up`) to restore the newest queued draft safely.
- A responsive, borderless queued-input preview above the composer.
- Lossless overflow, cancellation, provider-error, and compaction recovery,
  plus confirmation-gated queue discard on quit.
- A redraw scheduler that coalesces updates at no more than 120 FPS.
- A deterministic 80×18 queue/composer render snapshot.
- Pull-request/main CI plus release-time fmt, strict Clippy, and test gates.

Relevant upstream design references:

- [input queue](https://github.com/openai/codex/blob/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df/codex-rs/tui/src/chatwidget/input_queue.rs)
- [input flow](https://github.com/openai/codex/blob/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df/codex-rs/tui/src/chatwidget/input_flow.rs)
- [pending-input preview](https://github.com/openai/codex/blob/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df/codex-rs/tui/src/bottom_pane/pending_input_preview.rs)
- [composer footer](https://github.com/openai/codex/blob/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df/codex-rs/tui/src/bottom_pane/footer.rs)
- [frame requester](https://github.com/openai/codex/blob/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df/codex-rs/tui/src/tui/frame_requester.rs)
- [TUI style guide](https://github.com/openai/codex/blob/bd0d4a23e3d276d6c4addcf12da40042b9c0b4df/codex-rs/tui/styles.md)

## Remaining parity work

These are confirmed gaps, not delivered behavior:

1. **Policy and CLI shell** — typed workdir, image, sandbox, approval, writable-root,
   and alternate-screen options; `/status`, `/permissions`, and `/init`; one
   policy object enforced by every file and command tool.
2. **Runtime protocol** — OpenAI Responses events, durable thread state, active
   turn steering on `Enter`, resumable/forked sessions, reviews, and background
   terminals. The current OpenAI adapter uses Chat Completions and CLI adapters
   launch fresh handoff processes.
3. **Instruction and extension hierarchy** — global/project/nested `AGENTS.md`,
   overrides, skills, MCP, plugins, hooks, rules, and file search.
4. **Command surface** — typed queued actions, safe `!` shell intent, history
   search, copy, plan/goal/agent views, review, forks, and non-interactive
   execution. Busy slash commands currently remain editable drafts.
5. **Exact default shell** — Codex-like terminal-default colors, gutters,
   borderless composer/footer, responsive popups, and snapshot coverage across
   light, dark, narrow, modal, and long-transcript states. Shaltaiboltai's
   current framed themes remain intentionally visible during the transition.
6. **Measured performance** — cold/streaming/scroll/resize benchmarks,
   long-transcript memory budgets, PTY end-to-end tests, and Unicode editor
   differential tests.

## Acceptance rule

A parity item is complete only when its state-machine behavior, terminal render,
safety/failure paths, and performance boundary are covered by deterministic
tests. Local validation is not evidence that GitHub CI, a release build, a
provider session, or a sandbox policy succeeded remotely.
