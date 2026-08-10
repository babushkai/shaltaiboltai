use crate::app::{App, Entry, Mode};
use crate::markdown;
use crate::session;
use crate::theme::{self, Theme};
use crate::tools;
use ratatui::layout::{Alignment, Constraint, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, ListState, Padding, Paragraph, Scrollbar,
    ScrollbarOrientation, ScrollbarState, Wrap,
};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const TOOL_RESULT_PREVIEW_LINES: usize = 6;
const MAX_INPUT_LINES: u16 = 8;
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub fn draw(frame: &mut Frame, app: &mut App) {
    let theme = app.theme;
    if let Some(bg) = theme.bg {
        frame.render_widget(
            Block::default().style(Style::new().bg(bg).fg(theme.fg)),
            frame.area(),
        );
    }

    let input_height = input_height(app, frame.area().height);
    let [transcript_area, status_area, input_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(input_height),
    ])
    .areas(frame.area());

    draw_transcript(frame, app, transcript_area);
    draw_status(frame, app, status_area);
    draw_input(frame, app, input_area);
    if app.mode == Mode::Input && app.slash_menu_active() {
        draw_slash_menu(frame, app, input_area);
    }

    match app.mode {
        Mode::ModelPicker => draw_model_picker(frame, app),
        Mode::SessionPicker => draw_session_picker(frame, app),
        Mode::ThemePicker => draw_theme_picker(frame, app),
        Mode::Approval => draw_approval(frame, app),
        Mode::Help => draw_help(frame, app),
        _ => {}
    }
}

fn input_height(app: &App, total_height: u16) -> u16 {
    let desired = (app.textarea.lines().len() as u16).clamp(1, MAX_INPUT_LINES) + 2;
    // Composer-focused approvals reserve enough room for a visible review and
    // Tab affordance even when the draft itself is tall. At normal terminal
    // sizes the composer still keeps one visible inner row for safe editing.
    if app.mode == Mode::Approval && !app.approval_focused {
        let available = total_height.saturating_sub(5).max(1);
        let minimum = if total_height >= 8 { 3 } else { 1 };
        desired.min(available).max(minimum)
    } else {
        desired.min(total_height.saturating_sub(2).max(1))
    }
}

/// The input renders as an elevated card; its border doubles as the focus
/// indicator — accent while typing is possible, structural otherwise.
fn draw_input(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    let focused = app.composer_accepts_input();
    let queued = app.queued_prompt_count() > 0;
    let lookahead = app.compacting
        || matches!(
            app.mode,
            Mode::Streaming | Mode::RunningTool | Mode::Approval
        );
    let border = if focused {
        theme.accent
    } else if queued {
        theme.accent2
    } else {
        theme.border
    };
    let title = if queued && area.width < 32 {
        " queued "
    } else if queued {
        " next message queued "
    } else if lookahead {
        " next message "
    } else {
        " compose "
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border))
        .title(Line::styled(
            title,
            Style::new().fg(border).add_modifier(Modifier::BOLD),
        ));
    let image_count = if queued {
        app.queued_image_count()
    } else {
        app.pending_image_count()
    };
    if image_count > 0 {
        let mut metadata = Vec::new();
        metadata.push(format!(
            "{image_count} image{}",
            if image_count == 1 { "" } else { "s" }
        ));
        if focused && lookahead {
            metadata.push("Ctrl+X clear".into());
        }
        let budget = area.width.saturating_sub(title.width() as u16 + 4) as usize;
        let metadata = truncate_width(&metadata.join(" · "), budget);
        block = block.title(
            Line::styled(format!(" {metadata} "), Style::new().fg(theme.accent2))
                .alignment(Alignment::Right),
        );
    }
    let footer = if let Some(notice) = app.composer_notice() {
        Some((notice.to_owned(), theme.warning))
    } else if app.mode == Mode::Approval && !app.approval_focused && queued {
        Some(("Tab review tool · next message queued".into(), theme.dim))
    } else if app.mode == Mode::Approval && !app.approval_focused {
        Some((
            "Tab review tool · Enter queue · Alt+Enter newline".into(),
            theme.dim,
        ))
    } else if focused && lookahead {
        Some((
            "Esc cancel · Enter queue · Alt+Enter newline".into(),
            theme.dim,
        ))
    } else if queued && app.mode == Mode::Approval {
        Some(("waiting for tool decision · n / Esc deny".into(), theme.dim))
    } else if queued {
        Some(("Esc cancel · waiting for current turn".into(), theme.dim))
    } else if focused {
        Some((
            "Enter send · Alt+Enter newline · / commands".into(),
            theme.dim,
        ))
    } else {
        None
    };
    if area.width >= 16 {
        if let Some((footer, color)) = footer {
            let footer = truncate_width(&footer, area.width.saturating_sub(4) as usize);
            block = block.title_bottom(
                Line::styled(format!(" {footer} "), Style::new().fg(color))
                    .alignment(Alignment::Right),
            );
        }
    }
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let placeholder = if queued {
        "Waiting for the current turn to finish…"
    } else if lookahead {
        "Type the next request while this one runs…"
    } else {
        "Describe a change or ask a question…"
    };
    app.textarea.set_placeholder_text(placeholder);
    app.textarea.set_cursor_style(if focused {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default()
    });
    app.textarea.set_block(block);
    frame.render_widget(&app.textarea, area);
}

/// Renders the transcript through a per-entry line cache with cumulative line
/// offsets. Only dirty/new entries are parsed, and locating the viewport is a
/// binary search rather than a walk from the beginning of the conversation.
fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    // Borders (2) + horizontal padding (2).
    let width = area.width.saturating_sub(4).max(10) as usize;
    let previous_total = app.render_cache_total_lines;
    let preserve_viewport = app.scroll_from_bottom > 0
        && !app.render_cache.is_empty()
        && app.render_cache_width == width
        && app.render_cache_rev == app.transcript_rev
        && app.render_cache.len() <= app.transcript.len();

    if app.render_cache_width != width || app.render_cache_rev != app.transcript_rev {
        app.render_cache.clear();
        app.render_cache_starts.clear();
        app.render_cache_total_lines = 0;
        app.render_cache_width = width;
        app.render_cache_rev = app.transcript_rev;
        app.transcript_dirty_from = None;
    }
    if app.render_cache.len() > app.transcript.len() {
        app.render_cache.clear();
        app.render_cache_starts.clear();
        app.render_cache_total_lines = 0;
        app.transcript_dirty_from = None;
    }
    if let Some(dirty) = app.transcript_dirty_from.take() {
        let dirty = dirty.min(app.render_cache.len());
        if dirty < app.render_cache.len() {
            app.render_cache.truncate(dirty);
            app.render_cache_starts.truncate(dirty);
            app.render_cache_total_lines = dirty.checked_sub(1).map_or(0, |previous| {
                app.render_cache_starts[previous] + app.render_cache[previous].len()
            });
        }
    }
    let streaming = app.mode == Mode::Streaming;
    while app.render_cache.len() < app.transcript.len() {
        let i = app.render_cache.len();
        let last = i + 1 == app.transcript.len();
        let lines = render_entry(&app.transcript[i], width, last && streaming, &theme);
        let start = if i == 0 {
            0
        } else {
            app.render_cache_total_lines + 1
        };
        app.render_cache_starts.push(start);
        app.render_cache_total_lines = start + lines.len();
        app.render_cache.push(lines);
    }

    let total = app.render_cache_total_lines;
    // `scroll_from_bottom` normally follows the tail. Once the user scrolls,
    // adjust that distance with content growth or shrinkage so the same
    // transcript lines stay under the cursor while a response reflows.
    if preserve_viewport {
        if total >= previous_total {
            app.scroll_from_bottom = app
                .scroll_from_bottom
                .saturating_add(total - previous_total);
        } else {
            app.scroll_from_bottom = app
                .scroll_from_bottom
                .saturating_sub(previous_total - total);
        }
    }
    let visible = area.height.saturating_sub(2) as usize;
    app.scroll_from_bottom = app.scroll_from_bottom.min(total.saturating_sub(visible));
    let start = total.saturating_sub(visible + app.scroll_from_bottom);
    let end = (start + visible).min(total);

    let mut window: Vec<Line> = Vec::with_capacity(end.saturating_sub(start));
    let first = app
        .render_cache_starts
        .partition_point(|entry_start| *entry_start <= start)
        .saturating_sub(1);
    for i in first..app.render_cache.len() {
        let entry_start = app.render_cache_starts[i];
        if i > 0 {
            let separator = entry_start - 1;
            if separator >= start && separator < end {
                window.push(Line::raw(""));
            }
            if separator >= end {
                break;
            }
        }
        let lines = &app.render_cache[i];
        let from = start.saturating_sub(entry_start).min(lines.len());
        let to = end.saturating_sub(entry_start).min(lines.len());
        window.extend(lines[from..to].iter().cloned());
        if entry_start + lines.len() >= end {
            break;
        }
    }

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.border))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            " ◆ shaltaiboltai ",
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        ));
    if app.scroll_from_bottom > 0 {
        let label = if area.width >= 52 {
            format!(
                " ↑ {} lines from latest · Ctrl+End jump ",
                app.scroll_from_bottom
            )
        } else {
            format!(" ↑ {} · Ctrl+End ", app.scroll_from_bottom)
        };
        block = block.title_bottom(
            Line::styled(
                label,
                Style::new().fg(semantic_foreground(theme.warning, theme.bg, theme.fg)),
            )
            .alignment(Alignment::Right),
        );
    }
    frame.render_widget(Paragraph::new(window).block(block), area);

    if total > visible {
        let mut state = ScrollbarState::new(total)
            .position(start)
            .viewport_content_length(visible);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(None)
                .thumb_symbol("▐")
                .style(Style::new().fg(theme.border)),
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut state,
        );
    }
}

fn render_entry(entry: &Entry, width: usize, streaming: bool, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match entry {
        Entry::Banner { title, subtitle } => {
            lines.push(Line::from(vec![
                Span::styled(
                    "◆ ",
                    Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    title.clone(),
                    Style::new().fg(theme.fg).add_modifier(Modifier::BOLD),
                ),
            ]));
            push_wrapped(
                &mut lines,
                "  ",
                Style::new().fg(theme.dim),
                subtitle,
                width,
                Style::new().fg(theme.dim),
            );
        }
        Entry::User(text) => {
            lines.push(Line::from(vec![
                Span::styled(
                    " YOU ",
                    Style::new()
                        .fg(on_color(theme.accent))
                        .bg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  prompt", Style::new().fg(theme.dim)),
            ]));
            push_wrapped(
                &mut lines,
                "│ ",
                Style::new().fg(theme.border),
                text,
                width,
                Style::new().fg(theme.fg),
            );
        }
        Entry::Assistant(text) => {
            if !text.is_empty() || streaming {
                lines.push(Line::from(vec![
                    Span::styled(
                        "◆ ",
                        Style::new().fg(theme.accent2).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "ASSISTANT",
                        Style::new().fg(theme.dim).add_modifier(Modifier::BOLD),
                    ),
                ]));
                if text.is_empty() {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", Style::new().fg(theme.border)),
                        Span::styled("thinking…", Style::new().fg(theme.dim)),
                    ]));
                } else {
                    for line in markdown::render(text, width.saturating_sub(2), theme) {
                        let mut spans = Vec::with_capacity(line.spans.len() + 1);
                        spans.push(Span::styled("│ ", Style::new().fg(theme.border)));
                        spans.extend(line.spans);
                        lines.push(Line::from(spans));
                    }
                }
            }
        }
        Entry::Tool {
            summary,
            result,
            is_error,
        } => {
            let (state, glyph, color) = if *is_error {
                ("FAILED", "✗ ", theme.error)
            } else {
                ("DONE", "✓ ", theme.success)
            };
            let result_lines = result.lines().count();
            lines.push(Line::from(vec![
                Span::styled(
                    glyph,
                    Style::new()
                        .fg(semantic_foreground(color, theme.bg, theme.fg))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {state} "),
                    Style::new()
                        .fg(on_color(color))
                        .bg(color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  TOOL", Style::new().fg(theme.dim)),
                Span::styled(
                    if result_lines > 0 {
                        format!(
                            " · {result_lines} output line{}",
                            if result_lines == 1 { "" } else { "s" }
                        )
                    } else {
                        String::new()
                    },
                    Style::new().fg(theme.dim),
                ),
            ]));
            push_wrapped(
                &mut lines,
                "│ ",
                Style::new().fg(theme.border),
                summary,
                width,
                Style::new().fg(theme.fg),
            );
            let shown = result_lines.min(TOOL_RESULT_PREVIEW_LINES);
            for (i, line) in result.lines().take(shown).enumerate() {
                push_wrapped(
                    &mut lines,
                    "│   ",
                    Style::new().fg(theme.border),
                    line,
                    width,
                    Style::new().fg(theme.dim),
                );
                if i + 1 == shown && result_lines > shown {
                    lines.push(Line::from(vec![
                        Span::styled("│   ", Style::new().fg(theme.border)),
                        Span::styled(
                            format!("… {} more lines", result_lines - shown),
                            Style::new().fg(theme.dim).add_modifier(Modifier::ITALIC),
                        ),
                    ]));
                }
            }
        }
        Entry::Info(text) => {
            push_wrapped(
                &mut lines,
                "· ",
                Style::new().fg(theme.dim),
                text,
                width,
                Style::new().fg(theme.dim).add_modifier(Modifier::ITALIC),
            );
        }
        Entry::Error(text) => {
            let error = semantic_foreground(theme.error, theme.bg, theme.fg);
            push_wrapped(
                &mut lines,
                "! ",
                Style::new().fg(error),
                text,
                width,
                Style::new().fg(error),
            );
        }
    }
    lines
}

/// Wrap `text` to `width` and append, putting `prefix` on the first line with
/// matching indentation on continuations.
fn push_wrapped(
    lines: &mut Vec<Line<'static>>,
    prefix: &str,
    prefix_style: Style,
    text: &str,
    width: usize,
    style: Style,
) {
    let indent = " ".repeat(prefix.chars().count());
    let body_width = width.saturating_sub(prefix.chars().count()).max(10);
    let mut first = true;
    for raw_line in text.lines().chain(text.is_empty().then_some("")) {
        let wrapped = textwrap::wrap(raw_line, body_width);
        let parts: Vec<_> = if wrapped.is_empty() {
            vec!["".into()]
        } else {
            wrapped
        };
        for part in parts {
            let lead = if first {
                prefix.to_owned()
            } else {
                indent.clone()
            };
            first = false;
            lines.push(Line::from(vec![
                Span::styled(lead, prefix_style),
                Span::styled(part.into_owned(), style),
            ]));
        }
    }
}

fn spinner_frame() -> char {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    SPINNER[(ms / 120) as usize % SPINNER.len()]
}

/// One-line status bar on the surface elevation: accent model chip, state
/// (with spinner while busy) on the left, context usage on the right.
fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    if let Some(surface) = theme.surface {
        frame.render_widget(
            Block::default().style(Style::new().bg(surface).fg(theme.fg)),
            area,
        );
    }
    let wide = area.width >= 52;
    let (state, state_color) = if app.compacting {
        ("compacting context…", theme.accent)
    } else if app.discovering && app.mode == Mode::Input {
        ("discovering models…", theme.accent)
    } else {
        match app.mode {
            Mode::Input => ("ready", theme.success),
            Mode::Streaming => (
                if wide {
                    "thinking — Esc to cancel"
                } else {
                    "thinking"
                },
                theme.accent,
            ),
            Mode::RunningTool => (
                if wide {
                    "running tool — Esc to cancel"
                } else {
                    "running tool"
                },
                theme.accent2,
            ),
            Mode::Approval => ("approval needed", theme.warning),
            Mode::ModelPicker => ("selecting model", theme.accent2),
            Mode::SessionPicker => ("selecting session", theme.accent2),
            Mode::ThemePicker => (
                if wide {
                    "previewing theme — Enter keep · Esc revert"
                } else {
                    "previewing theme"
                },
                theme.accent2,
            ),
            Mode::Help => ("keyboard guide", theme.accent2),
        }
    };
    let state = if app.mode == Mode::Approval && !app.approval_focused {
        if app.queued_prompt_count() > 0 {
            format!("{state} · next queued · Tab review")
        } else {
            format!("{state} · Tab to review")
        }
    } else if app.queued_prompt_count() > 0 {
        format!("{state} · next queued")
    } else {
        state.to_owned()
    };
    let spinner_width = if app.is_busy() { 3 } else { 1 };
    let state_width = UnicodeWidthStr::width(state.as_str());
    let chip_budget = (area.width as usize)
        .saturating_sub(state_width + spinner_width + 2)
        .min(36);
    let model = app
        .model
        .as_ref()
        .map(|m| format!("{} · {}", m.id, m.provider.label()))
        .unwrap_or_else(|| {
            if app.discovering {
                "finding models".into()
            } else {
                "no model".into()
            }
        });
    let mut spans = Vec::new();
    if chip_budget >= 8 {
        let model = truncate_width(&model, chip_budget.saturating_sub(4));
        spans.push(Span::styled(
            format!(" ◆ {model} "),
            Style::new().fg(on_color(theme.accent)).bg(theme.accent),
        ));
    }
    if app.is_busy() {
        spans.push(Span::styled(
            format!(" {} ", spinner_frame()),
            Style::new().fg(theme.accent),
        ));
    } else {
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled(
        state,
        Style::new().fg(semantic_foreground(state_color, theme.surface, theme.fg)),
    ));
    let left_width: usize = spans.iter().map(|s| s.width()).sum();
    frame.render_widget(Paragraph::new(Line::from(spans)), area);

    // Right side: cwd · branch · context usage, Claude Code style. On narrow
    // terminals, pieces are dropped (cwd first, then branch) instead of
    // colliding with the left side.
    let approx = (app.last_usage.is_none()).then(|| app.approx_tokens());
    let context = match app.last_usage {
        Some(u) => Some(format!(
            "ctx {} · out {}",
            fmt_count(u.input_tokens as usize),
            fmt_count(u.output_tokens as usize)
        )),
        None => approx
            .filter(|tokens| *tokens > 0)
            .map(|tokens| format!("ctx ~{}", fmt_count(tokens))),
    };
    let context_percent = context.as_ref().and_then(|_| app.context_percent());
    let assemble = |with_cwd: bool, with_branch: bool| -> Vec<Span<'static>> {
        let mut right: Vec<Span> = Vec::new();
        let sep = || Span::styled(" · ", Style::new().fg(theme.border));
        if with_cwd && !app.cwd_display.is_empty() {
            right.push(Span::styled(
                app.cwd_display.clone(),
                Style::new().fg(theme.dim),
            ));
        }
        if with_branch {
            if let Some(branch) = &app.git_branch {
                if !right.is_empty() {
                    right.push(sep());
                }
                right.push(Span::styled(branch.clone(), Style::new().fg(theme.accent2)));
            }
        }
        if let Some(ctx) = &context {
            if !right.is_empty() {
                right.push(sep());
            }
            right.push(Span::styled(ctx.clone(), Style::new().fg(theme.dim)));
            if let Some(pct) = context_percent {
                let color = match pct {
                    0..=69 => theme.dim,
                    70..=89 => theme.warning,
                    _ => theme.error,
                };
                right.push(Span::styled(
                    format!(" {pct}%"),
                    Style::new().fg(semantic_foreground(color, theme.surface, theme.fg)),
                ));
            }
        }
        if !right.is_empty() {
            right.push(Span::raw(" "));
        }
        right
    };
    let fits = |candidate: &[Span]| -> bool {
        let w: usize = candidate.iter().map(|s| s.width()).sum();
        !candidate.is_empty() && left_width + w < area.width as usize
    };
    let right = [(true, true), (false, true), (false, false)]
        .into_iter()
        .find_map(|(cwd, branch)| {
            let candidate = assemble(cwd, branch);
            fits(&candidate).then_some(candidate)
        });
    if let Some(right) = right {
        frame.render_widget(
            Paragraph::new(Line::from(right)).alignment(Alignment::Right),
            area,
        );
    }
}

fn truncate_width(text: &str, max: usize) -> String {
    if UnicodeWidthStr::width(text) <= max {
        return text.to_owned();
    }
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut width = 0;
    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width + 1 > max {
            break;
        }
        out.push(ch);
        width += ch_width;
    }
    out.push('…');
    out
}

/// Pick a legible foreground for filled badges independently of the theme's
/// canvas color. A light theme's base is not necessarily readable on success
/// green, just as a dark theme's base can disappear on a dark error red.
fn on_color(color: Color) -> Color {
    match relative_luminance(color) {
        Some(luminance) if luminance > 0.179 => Color::Black,
        Some(_) => Color::White,
        None => match color {
            Color::Black | Color::Red | Color::Blue | Color::Magenta | Color::DarkGray => {
                Color::White
            }
            _ => Color::Black,
        },
    }
}

/// Preserve a semantic hue only when it remains readable as text on the
/// current surface; otherwise keep the icon/wording as the non-color cue and
/// fall back to the theme's primary foreground.
fn semantic_foreground(preferred: Color, background: Option<Color>, fallback: Color) -> Color {
    let Some(background) = background else {
        return preferred;
    };
    let Some(foreground_luminance) = relative_luminance(preferred) else {
        return preferred;
    };
    let Some(background_luminance) = relative_luminance(background) else {
        return preferred;
    };
    let contrast = (foreground_luminance.max(background_luminance) + 0.05)
        / (foreground_luminance.min(background_luminance) + 0.05);
    if contrast >= 4.5 {
        preferred
    } else {
        fallback
    }
}

fn relative_luminance(color: Color) -> Option<f64> {
    let Color::Rgb(r, g, b) = color else {
        return None;
    };
    let linear = |channel: u8| {
        let channel = f64::from(channel) / 255.0;
        if channel <= 0.04045 {
            channel / 12.92
        } else {
            ((channel + 0.055) / 1.055).powf(2.4)
        }
    };
    Some(0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b))
}

fn fmt_count(n: usize) -> String {
    if n >= 10_000 {
        format!("{}k", n / 1000)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Claude Code-style completion popup anchored above the input: appears as
/// soon as the input is `/…`, filters as you type.
fn draw_slash_menu(frame: &mut Frame, app: &App, input_area: Rect) {
    let theme = app.theme;
    let matches = app.slash_matches();
    let selected = app.slash_index.min(matches.len().saturating_sub(1));

    let label = |c: &crate::app::SlashCommand| -> usize {
        1 + c.name.len() + c.args.map(|a| a.len() + 1).unwrap_or(0)
    };
    let label_width = matches.iter().map(|c| label(c)).max().unwrap_or(0) + 2;
    let available_height = input_area.y.saturating_sub(frame.area().y);
    let available_width = frame.area().width.saturating_sub(2);
    if available_height < 3 || available_width < 4 {
        return;
    }
    let height = ((matches.len() as u16).min(8) + 2).min(available_height);
    let desired_width = matches
        .iter()
        .map(|c| 4 + label_width + c.description.len())
        .max()
        .unwrap_or(20) as u16;
    let width = desired_width.min(available_width);
    let max_x = frame.area().x + frame.area().width - width;
    let area = Rect {
        x: (input_area.x + 1).min(max_x),
        y: input_area.y.saturating_sub(height),
        width,
        height,
    };
    frame.render_widget(Clear, area);

    let items: Vec<ListItem> = matches
        .iter()
        .map(|c| {
            let mut spans = vec![Span::styled(
                format!("/{}", c.name),
                Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
            )];
            if let Some(args) = c.args {
                spans.push(Span::styled(format!(" {args}"), Style::new().fg(theme.dim)));
            }
            spans.push(Span::raw(" ".repeat(label_width.saturating_sub(label(c)))));
            spans.push(Span::styled(c.description, Style::new().fg(theme.dim)));
            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.border));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let list = List::new(items)
        .block(block)
        .highlight_style(Style::new().bg(theme.accent).fg(on_color(theme.accent)))
        .highlight_symbol("❯ ");

    let mut state = ListState::default();
    state.select(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_model_picker(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let models = app.filtered_models();
    let items: Vec<ListItem> = models
        .iter()
        .map(|m| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<10}", m.provider.label()),
                    Style::new().fg(theme.accent2),
                ),
                Span::styled(m.id.clone(), Style::new().fg(theme.fg)),
            ]))
        })
        .collect();
    let title = format!(
        " select model — type to filter: {}▏ ({} shown) ",
        app.picker_filter,
        models.len()
    );
    draw_overlay_list(
        frame,
        &theme,
        title,
        items,
        app.picker_index.min(models.len().saturating_sub(1)),
    );
}

fn draw_session_picker(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let items: Vec<ListItem> = app
        .sessions
        .iter()
        .map(|s| {
            let mut spans = vec![
                Span::styled(s.title.clone(), Style::new().fg(theme.fg)),
                Span::styled(
                    format!("  ·  {}", session::ago(s.updated_at)),
                    Style::new().fg(theme.dim),
                ),
            ];
            // Sessions from other working directories are listed after the
            // current project's, badged with where they came from.
            if crate::app::session_is_foreign(s) {
                if let Some(cwd) = &s.cwd {
                    spans.push(Span::styled(
                        format!("  ·  {}", short_dir(cwd)),
                        Style::new().fg(theme.border),
                    ));
                }
            }
            ListItem::new(Line::from(spans))
        })
        .collect();
    let title = format!(" resume session ({}) ", app.sessions.len());
    draw_overlay_list(frame, &theme, title, items, app.session_index);
}

/// Last two path components, enough to recognize a project in the picker.
fn short_dir(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    match parts.as_slice() {
        [.., a, b] => format!("{a}/{b}"),
        _ => path.to_owned(),
    }
}

fn draw_theme_picker(frame: &mut Frame, app: &App) {
    let current = app.theme;
    let items: Vec<ListItem> = theme::all()
        .iter()
        .map(|t| {
            let mut spans = vec![Span::styled(
                format!("{:<14}", t.name),
                Style::new().fg(current.fg),
            )];
            for color in [t.accent, t.accent2, t.success, t.warning, t.error, t.code] {
                spans.push(Span::styled("██", Style::new().fg(color)));
            }
            if t.name == current.name {
                spans.push(Span::styled("  ✓", Style::new().fg(current.success)));
            }
            ListItem::new(Line::from(spans))
        })
        .collect();
    draw_overlay_list(
        frame,
        &current,
        " select theme — live preview ".into(),
        items,
        app.theme_index,
    );
}

fn draw_overlay_list(
    frame: &mut Frame,
    theme: &Theme,
    title: String,
    items: Vec<ListItem>,
    selected: usize,
) {
    let preferred_height = (items.len() as u16 + 2).clamp(5, 22);
    let area = modal_area(frame.area(), 76, preferred_height);
    frame.render_widget(Clear, area);

    let empty = items.is_empty();
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.accent))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            title,
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        ));
    if area.width >= 34 {
        block = block.title_bottom(
            Line::styled(
                " ↑↓ move · Enter select · Esc close ",
                Style::new().fg(theme.dim),
            )
            .alignment(Alignment::Right),
        );
    }
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let list = List::new(items)
        .block(block)
        .highlight_style(Style::new().bg(theme.accent).fg(on_color(theme.accent)))
        .highlight_symbol("❯ ");

    let mut state = ListState::default();
    state.select((!empty).then_some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_approval(frame: &mut Frame, app: &mut App) {
    let theme = app.theme;
    let warning = semantic_foreground(theme.warning, theme.surface, theme.fg);
    let approval_focused = app.approval_focused;
    let Some(call) = app.pending_approval() else {
        return;
    };
    let description = tools::describe(call);
    let scope_label = tools::approval_scope_label(call);
    let arguments = call.arguments.clone();
    let queue_occupied = app.queued_prompt_count() > 0;
    // When type-ahead owns focus, keep the composer physically visible below
    // the modal. Arming approval focus with Tab expands the review back to the
    // full terminal height.
    let modal_root = if approval_focused {
        frame.area()
    } else {
        let input_height = input_height(app, frame.area().height);
        Rect {
            height: frame.area().height.saturating_sub(input_height),
            ..frame.area()
        }
    };
    let area = modal_area(modal_root, 96, 24);
    frame.render_widget(Clear, area);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(warning))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            if approval_focused {
                " review tool request · approval focus "
            } else if queue_occupied {
                " review tool request · press Tab "
            } else {
                " review tool request · composer focus "
            },
            Style::new().fg(warning).add_modifier(Modifier::BOLD),
        ));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let action_lines = if approval_focused {
        approval_action_lines(scope_label, inner.width as usize, &theme)
    } else if queue_occupied {
        approval_paused_lines(inner.width as usize, &theme)
    } else {
        approval_composer_lines(inner.width as usize, &theme)
    };
    let action_height = (action_lines.len() as u16).min(inner.height);
    let room_before_actions = inner.height.saturating_sub(action_height);
    let summary_height = if room_before_actions >= 7 {
        2
    } else if room_before_actions >= 2 {
        1
    } else {
        0
    };
    let room_for_preview = room_before_actions.saturating_sub(summary_height);
    let meta_height = u16::from(room_for_preview >= 2 && inner.height >= 6);
    let [summary_area, preview_area, meta_area, action_area] = Layout::vertical([
        Constraint::Length(summary_height),
        Constraint::Min(0),
        Constraint::Length(meta_height),
        Constraint::Length(action_height),
    ])
    .areas(inner);
    if summary_area.height > 0 {
        frame.render_widget(
            Paragraph::new(description)
                .style(Style::new().fg(theme.fg).add_modifier(Modifier::BOLD))
                .wrap(Wrap { trim: true }),
            summary_area,
        );
    }

    let inner_width = preview_area.width.saturating_sub(1) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();
    match &app.approval_preview {
        Some(diff) => {
            for (tag, text) in diff {
                let (style, prefix) = match tag {
                    '+' => (
                        Style::new().fg(semantic_foreground(
                            theme.success,
                            theme.surface,
                            theme.fg,
                        )),
                        "+",
                    ),
                    '-' => (
                        Style::new().fg(semantic_foreground(theme.error, theme.surface, theme.fg)),
                        "-",
                    ),
                    '@' => (Style::new().fg(theme.accent2), "@"),
                    '!' => (
                        Style::new()
                            .fg(semantic_foreground(theme.error, theme.surface, theme.fg))
                            .add_modifier(Modifier::BOLD),
                        "!",
                    ),
                    _ => (Style::new().fg(theme.dim), " "),
                };
                push_display_wrapped(
                    &mut lines,
                    text,
                    &format!("{prefix} "),
                    "  ",
                    inner_width,
                    style,
                );
            }
        }
        None => {
            if let Ok(pretty) = serde_json::to_string_pretty(&arguments) {
                for l in pretty.lines().take(200) {
                    push_display_wrapped(
                        &mut lines,
                        l,
                        "",
                        "  ",
                        inner_width,
                        Style::new().fg(theme.dim),
                    );
                }
            }
        }
    }
    let visible = preview_area.height as usize;
    let max_scroll = lines.len().saturating_sub(visible);
    app.approval_scroll = app.approval_scroll.min(max_scroll);
    let start = app.approval_scroll;
    let end = (start + visible).min(lines.len());
    frame.render_widget(Paragraph::new(lines[start..end].to_vec()), preview_area);

    if lines.len() > visible && visible > 0 {
        let mut state = ScrollbarState::new(lines.len())
            .position(start)
            .viewport_content_length(visible);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(None)
                .thumb_symbol("▐")
                .style(Style::new().fg(warning)),
            preview_area,
            &mut state,
        );
    }
    if meta_height > 0 {
        let range = if lines.is_empty() {
            "no preview".to_owned()
        } else {
            format!("lines {}–{} of {}", start + 1, end, lines.len())
        };
        let controls = if approval_focused {
            "  ·  ↑↓ / PgUp PgDn scroll"
        } else if queue_occupied {
            "  ·  PgUp PgDn scroll · Tab review"
        } else {
            "  ·  PgUp PgDn scroll · ↑↓ edit"
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(range, Style::new().fg(theme.dim)),
                Span::styled(controls, Style::new().fg(theme.dim)),
            ])),
            meta_area,
        );
    }

    frame.render_widget(
        Paragraph::new(
            action_lines
                .into_iter()
                .take(action_area.height as usize)
                .collect::<Vec<_>>(),
        ),
        action_area,
    );
}

fn approval_paused_lines(width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let review = Line::from(vec![
        key_span("Tab", theme.accent),
        Span::styled(" review tool request", Style::new().fg(theme.fg)),
    ]);
    let queued = Line::styled(
        "next message already queued",
        Style::new().fg(theme.accent2),
    );
    if review.width() + queued.width() + 3 <= width {
        vec![Line::from(vec![
            key_span("Tab", theme.accent),
            Span::styled(" review   ·   ", Style::new().fg(theme.fg)),
            Span::styled("next queued", Style::new().fg(theme.accent2)),
        ])]
    } else {
        vec![review, queued]
    }
}

fn approval_composer_lines(width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let first = Line::from(vec![
        key_span("Tab", theme.accent),
        Span::styled(" review tool request", Style::new().fg(theme.fg)),
    ]);
    let second = Line::from(vec![
        key_span("Enter", theme.accent2),
        Span::styled(" queue next message", Style::new().fg(theme.fg)),
    ]);
    if first.width() + second.width() + 3 <= width {
        vec![Line::from(vec![
            key_span("Tab", theme.accent),
            Span::styled(" review   ·   ", Style::new().fg(theme.fg)),
            key_span("Enter", theme.accent2),
            Span::styled(" queue next", Style::new().fg(theme.fg)),
        ])]
    } else {
        vec![first, second]
    }
}

/// Wrap approval material into real visual rows before applying vertical
/// scrolling. This keeps the tail of long commands and wide Unicode diff lines
/// reviewable instead of letting `Paragraph` clip them horizontally.
fn push_display_wrapped(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    first_prefix: &str,
    continuation_prefix: &str,
    width: usize,
    style: Style,
) {
    if width == 0 {
        return;
    }

    let mut chars = text.chars().peekable();
    let mut first = true;
    loop {
        let prefix = if first {
            first_prefix
        } else {
            continuation_prefix
        };
        let available = width.saturating_sub(UnicodeWidthStr::width(prefix)).max(1);
        let mut chunk = String::new();
        let mut chunk_width = 0;
        while let Some(&ch) = chars.peek() {
            let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if !chunk.is_empty() && chunk_width + char_width > available {
                break;
            }
            chars.next();
            chunk.push(ch);
            chunk_width += char_width;
            if chunk_width >= available {
                break;
            }
        }

        lines.push(Line::styled(format!("{prefix}{chunk}"), style));
        if chars.peek().is_none() {
            break;
        }
        first = false;
    }
}

/// Keep approval and denial visible before the longer allow-scope wording. At
/// narrow widths the safety choices stay on the first row and the exact scope
/// wraps below them.
fn approval_action_lines(scope: &'static str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let success = semantic_foreground(theme.success, theme.surface, theme.fg);
    let error = semantic_foreground(theme.error, theme.surface, theme.fg);
    let wide = Line::from(vec![
        key_span("y", success),
        Span::styled(" approve   ·   ", Style::new().fg(theme.fg)),
        key_span("n / Esc", error),
        Span::styled(" deny   ·   ", Style::new().fg(theme.fg)),
        key_span("a", success),
        Span::styled(format!(" allow {scope}"), Style::new().fg(theme.fg)),
    ]);
    if wide.width() <= width {
        return vec![wide];
    }

    let compact = Line::from(vec![
        key_span("y", success),
        Span::styled(" yes · ", Style::new().fg(theme.fg)),
        key_span("n", error),
        Span::styled(" no · ", Style::new().fg(theme.fg)),
        key_span("a", success),
        Span::styled(format!(" {scope}"), Style::new().fg(theme.fg)),
    ]);
    if compact.width() <= width {
        return vec![compact];
    }

    let primary = Line::from(vec![
        key_span("y", success),
        Span::styled(" yes · ", Style::new().fg(theme.fg)),
        key_span("n", error),
        Span::styled(" no", Style::new().fg(theme.fg)),
    ]);
    let mut lines = if primary.width() <= width {
        vec![primary]
    } else {
        vec![
            Line::from(vec![
                key_span("n", error),
                Span::styled(" no · Esc", Style::new().fg(theme.fg)),
            ]),
            Line::from(vec![
                key_span("y", success),
                Span::styled(" yes", Style::new().fg(theme.fg)),
            ]),
        ]
    };

    let scope_width = width.saturating_sub(2).max(1);
    for (index, chunk) in textwrap::wrap(scope, scope_width).into_iter().enumerate() {
        if index == 0 {
            lines.push(Line::from(vec![
                key_span("a", success),
                Span::styled(format!(" {chunk}"), Style::new().fg(theme.fg)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(chunk.into_owned(), Style::new().fg(theme.fg)),
            ]));
        }
    }
    lines
}

fn key_span(label: &'static str, color: Color) -> Span<'static> {
    Span::styled(label, Style::new().fg(color).add_modifier(Modifier::BOLD))
}

fn draw_help(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    // The guide is intentionally a focused full-screen layer; clearing the
    // underlying composer/status avoids a noisy double frame on 80×24 shells.
    frame.render_widget(Clear, frame.area());
    if let Some(bg) = theme.bg {
        frame.render_widget(
            Block::default().style(Style::new().bg(bg).fg(theme.fg)),
            frame.area(),
        );
    }
    let area = modal_area(frame.area(), 78, 20);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.accent2))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            " keyboard guide ",
            Style::new().fg(theme.accent2).add_modifier(Modifier::BOLD),
        ))
        .title_bottom(
            Line::styled(" F1 · Enter · Esc close ", Style::new().fg(theme.dim))
                .alignment(Alignment::Right),
        );
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let key_column = if inner.width >= 44 {
        14
    } else if inner.width >= 28 {
        11
    } else {
        9
    };
    let key = |label: &'static str, action: &'static str| {
        Line::from(vec![
            Span::styled(
                format!(" {label:<key_column$}"),
                Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(action, Style::new().fg(theme.fg)),
        ])
    };
    let section = |title: &'static str| {
        Line::styled(
            title.to_owned(),
            Style::new().fg(theme.accent2).add_modifier(Modifier::BOLD),
        )
    };
    let lines = if inner.height >= 18 && inner.width >= 44 {
        vec![
            section("MESSAGE"),
            key("Enter", "send, or queue next while working"),
            key("Alt+Enter", "insert a newline"),
            key("Ctrl+V", "attach clipboard image"),
            key("Ctrl+X", "clear staged attachments"),
            key("Ctrl+U", "clear the composer"),
            key("/", "browse commands"),
            Line::raw(""),
            section("NAVIGATE"),
            key("Ctrl+P", "choose a model"),
            key("↑ / ↓", "recall prompts or move in menus"),
            key("PgUp / PgDn", "scroll conversation or approval"),
            key("Ctrl+Home/End", "oldest / latest message"),
            Line::raw(""),
            section("AGENT"),
            key("Esc", "cancel work; focus / deny approval"),
            key("Tab · y/a/n", "focus approval · decide"),
            key("Ctrl+C", "restore queued, then quit"),
        ]
    } else if inner.height >= 8 {
        vec![
            key("Enter", "send / queue next"),
            key("Alt+Enter", "newline"),
            key("Ctrl+P", "models"),
            key("/", "commands"),
            key("PgUp/PgDn", "scroll"),
            key("Esc", "cancel / deny"),
            key("Tab · y/a/n", "approval choices"),
            key("Ctrl+C", "queue-safe quit"),
        ]
    } else {
        // Safety and exit bindings come first when there is not enough room
        // for even the compact guide; never render more rows than can fit.
        vec![
            key("Esc", "cancel / deny"),
            key("Tab · y/a/n", "approval"),
            key("Ctrl+C", "queue-safe quit"),
            key("Enter", "send"),
            key("/", "commands"),
            key("PgUp/PgDn", "scroll"),
        ]
    };
    frame.render_widget(
        Paragraph::new(
            lines
                .into_iter()
                .take(inner.height as usize)
                .collect::<Vec<_>>(),
        ),
        inner,
    );
}

fn modal_area(area: Rect, preferred_width: u16, preferred_height: u16) -> Rect {
    let horizontal_margin = u16::from(area.width > 4);
    let vertical_margin = u16::from(area.height > 4);
    let width = preferred_width.min(area.width.saturating_sub(horizontal_margin * 2));
    let height = preferred_height.min(area.height.saturating_sub(vertical_margin * 2));
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}
