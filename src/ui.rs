use crate::app::{App, Entry, Mode};
use crate::markdown;
use crate::mascot;
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
use ratatui_image::Image as TerminalImage;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const TOOL_RESULT_PREVIEW_LINES: usize = 6;
const MAX_INPUT_LINES: u16 = 8;
const LEAD_STAGE_WIDTH: u16 = 30;
const LEAD_STAGE_MIN_WIDTH: u16 = 78;
const LEAD_STAGE_MIN_TRANSCRIPT_HEIGHT: u16 = 19;
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub fn draw(frame: &mut Frame, app: &mut App) {
    draw_frame(frame, app, None);
}

/// Production renderer with a high-resolution Kitty image cache. Tests and
/// unsupported terminals keep using [`draw`] and the deterministic cell art.
pub fn draw_with_native_mascot(
    frame: &mut Frame,
    app: &mut App,
    native_mascot: &mascot::NativeMascot,
) {
    draw_frame(frame, app, Some(native_mascot));
}

fn draw_frame(frame: &mut Frame, app: &mut App, native_mascot: Option<&mascot::NativeMascot>) {
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

    let (lead_stage, conversation_area) = lead_stage_layout(transcript_area);
    if let Some(area) = lead_stage {
        // Kitty uses skip-marked placeholder cells. Keep modal frames entirely
        // cell-rendered so Clear can own every overlay cell and accessibility
        // text can never sit behind a terminal image plane.
        let native_mascot = native_mascot.filter(|_| {
            !matches!(
                app.mode,
                Mode::ModelPicker
                    | Mode::SessionPicker
                    | Mode::ThemePicker
                    | Mode::Approval
                    | Mode::OrchestrationConfirm
                    | Mode::Help
            )
        });
        draw_lead_stage(frame, app, area, native_mascot);
    }
    draw_transcript(frame, app, conversation_area, lead_stage.is_some());
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
        Mode::OrchestrationConfirm => draw_orchestration_confirm(frame, app),
        Mode::Help => draw_help(frame, app),
        _ => {}
    }
}

/// Keep the full mascot's footprint geometry-only. Busy/idle transitions can
/// change its pose, but never the transcript height or scroll anchor.
fn lead_stage_layout(area: Rect) -> (Option<Rect>, Rect) {
    if area.width < LEAD_STAGE_MIN_WIDTH || area.height < LEAD_STAGE_MIN_TRANSCRIPT_HEIGHT {
        return (None, area);
    }
    let [stage, conversation] = Layout::horizontal([
        Constraint::Length(LEAD_STAGE_WIDTH),
        Constraint::Min(LEAD_STAGE_MIN_WIDTH - LEAD_STAGE_WIDTH),
    ])
    .areas(area);
    (Some(stage), conversation)
}

fn draw_lead_stage(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    native_mascot: Option<&mascot::NativeMascot>,
) {
    let theme = app.theme;
    let state = app.mascot_state();
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.accent))
        .title(Line::styled(
            " SHALTAIBOLTAI ",
            Style::new()
                .fg(semantic_foreground(
                    theme.accent,
                    theme.surface.or(theme.bg),
                    theme.fg,
                ))
                .add_modifier(Modifier::BOLD),
        ));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if let Some(native_mascot) = native_mascot {
        let image = native_mascot.protocol(state, app.animation_tick());
        let size = image.area();
        if size.width <= inner.width && size.height <= inner.height {
            let image_area = Rect::new(
                inner.x + (inner.width - size.width) / 2,
                inner.y + (inner.height - size.height) / 2,
                size.width,
                size.height,
            );
            frame.render_widget(TerminalImage::new(image), image_area);
            return;
        }
    }

    let pose = mascot::frame(state, app.animation_tick());
    let art = pose
        .cells
        .iter()
        .map(|row| mascot_line(row, &theme))
        .collect::<Vec<_>>();
    let art_width = mascot::FRAME_WIDTH as u16;
    let art_height = mascot::FRAME_HEIGHT as u16;
    let art_area = Rect::new(
        inner.x + inner.width.saturating_sub(art_width) / 2,
        inner.y + inner.height.saturating_sub(art_height) / 2,
        art_width.min(inner.width),
        art_height.min(inner.height),
    );
    frame.render_widget(Paragraph::new(art), art_area);
}

fn mascot_line(row: &[mascot::MascotCell; mascot::FRAME_WIDTH], theme: &Theme) -> Line<'static> {
    let panel_bg = theme.surface.or(theme.bg);
    Line::from(
        row.iter()
            .map(|cell| {
                let top = mascot_color(cell.top(), theme);
                let bottom = mascot_color(cell.bottom(), theme);
                let mut style = Style::new();
                if let Some(background) = panel_bg {
                    style = style.bg(background);
                }
                match (top, bottom) {
                    (None, None) => Span::styled(" ", style),
                    (Some(color), None) => Span::styled("▀", style.fg(color)),
                    (None, Some(color)) => Span::styled("▄", style.fg(color)),
                    (Some(top), Some(bottom)) if top == bottom => Span::styled("█", style.fg(top)),
                    (Some(top), Some(bottom)) => Span::styled("▀", style.fg(top).bg(bottom)),
                }
            })
            .collect::<Vec<_>>(),
    )
}

fn mascot_color(pixel: u32, theme: &Theme) -> Option<Color> {
    mascot::color(pixel).map(|(red, green, blue)| {
        if theme.name == "terminal" {
            terminal_mascot_color(red, green, blue)
        } else {
            Color::Rgb(red, green, blue)
        }
    })
}

fn terminal_mascot_color(red: u8, green: u8, blue: u8) -> Color {
    if red > 175 && green > 135 && blue > 90 {
        Color::White
    } else if red > 105 && red > green.saturating_add(30) {
        Color::Red
    } else if blue > 55 && green > red.saturating_add(15) {
        Color::Cyan
    } else if blue > 45 && blue > red && blue > green {
        Color::Blue
    } else if red > 65 && green > 35 && blue < 75 {
        Color::Yellow
    } else {
        Color::DarkGray
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
            Mode::Streaming | Mode::RunningTool | Mode::Approval | Mode::Orchestrating
        );
    let team_workers = app.team_workers();
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
    } else if team_workers.is_some() && area.width < 40 {
        " team prompt "
    } else if team_workers.is_some() {
        " team · next prompt "
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
    } else if let Some(workers) = team_workers {
        Some((
            format!("Enter starts 1 planning call · {workers} workers after review"),
            theme.accent2,
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
    } else if team_workers.is_some() {
        "Describe what Shaltaiboltai should coordinate…"
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
fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect, full_mascot: bool) {
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

    let brand = if full_mascot {
        " ◆ conversation ".to_owned()
    } else if area.width >= 28 {
        " ◆ shaltaiboltai ".to_owned()
    } else {
        " ◆ chat ".to_owned()
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.border))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            brand,
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        ));
    if app.scroll_from_bottom > 0 {
        let label = if area.width >= 48 {
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
        Entry::Agent {
            name,
            model,
            status,
            summary,
            is_error,
        } => {
            let running = status == "RUNNING";
            let (glyph, color) = if *is_error {
                ("✗ ", theme.error)
            } else if running {
                ("◆ ", theme.accent2)
            } else {
                ("✓ ", theme.success)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    glyph,
                    Style::new()
                        .fg(semantic_foreground(color, theme.bg, theme.fg))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {status} "),
                    Style::new()
                        .fg(on_color(color))
                        .bg(color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  AGENT · ", Style::new().fg(theme.dim)),
                Span::styled(model.clone(), Style::new().fg(theme.accent2)),
            ]));
            push_wrapped(
                &mut lines,
                "│ ",
                Style::new().fg(theme.border),
                name,
                width,
                Style::new().fg(theme.fg).add_modifier(Modifier::BOLD),
            );
            if !summary.is_empty() {
                push_wrapped(
                    &mut lines,
                    "│   ",
                    Style::new().fg(theme.border),
                    summary,
                    width,
                    Style::new().fg(if *is_error { theme.error } else { theme.dim }),
                );
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

fn spinner_frame(tick: u64) -> char {
    SPINNER[tick as usize % SPINNER.len()]
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
    let (state, state_color) = if let Some(status) = app.orchestration_status() {
        (status, theme.accent2)
    } else if app.compacting {
        ("compacting context…".into(), theme.accent)
    } else if app.discovering && app.mode == Mode::Input {
        ("discovering models…".into(), theme.accent)
    } else if let Some(workers) = app.team_workers().filter(|_| app.mode == Mode::Input) {
        (format!("TEAM · {workers} workers armed"), theme.accent2)
    } else {
        match app.mode {
            Mode::Input => ("ready".into(), theme.success),
            Mode::Streaming => (
                if wide {
                    "thinking — Esc to cancel".into()
                } else {
                    "thinking".into()
                },
                theme.accent,
            ),
            Mode::RunningTool => (
                if wide {
                    "running tool — Esc to cancel".into()
                } else {
                    "running tool".into()
                },
                theme.accent2,
            ),
            Mode::Approval => ("approval needed".into(), theme.warning),
            Mode::OrchestrationConfirm => ("TEAM · plan ready".into(), theme.warning),
            Mode::Orchestrating => ("TEAM · working — Esc cancel".into(), theme.accent2),
            Mode::ModelPicker => ("selecting model".into(), theme.accent2),
            Mode::SessionPicker => ("selecting session".into(), theme.accent2),
            Mode::ThemePicker => (
                if wide {
                    "previewing theme — Enter keep · Esc revert".into()
                } else {
                    "previewing theme".into()
                },
                theme.accent2,
            ),
            Mode::Help => ("keyboard guide".into(), theme.accent2),
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
        state
    };
    let spinner_width = if app.is_busy() { 3 } else { 1 };
    let state_width = UnicodeWidthStr::width(state.as_str());
    let chip_budget = (area.width as usize)
        .saturating_sub(state_width + spinner_width + 2)
        .min(36);
    let model = app
        .model
        .as_ref()
        .map(|m| format!("{} · {}", m.display_id(), m.provider.label()))
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
            format!(" {} ", spinner_frame(app.animation_tick())),
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
    let provider_width = models
        .iter()
        .map(|model| UnicodeWidthStr::width(model.provider.label()))
        .max()
        .unwrap_or(0);
    let items: Vec<ListItem> = models
        .iter()
        .map(|m| {
            let mut spans = vec![
                Span::styled(
                    format!("{:<provider_width$}  ", m.provider.label()),
                    Style::new().fg(theme.accent2),
                ),
                Span::styled(m.display_id().to_owned(), Style::new().fg(theme.fg)),
            ];
            if m.provider.is_sub_agent() {
                let detail = if m.is_claude_alias() {
                    " · latest alias · subscription sub-agent"
                } else {
                    " · subscription sub-agent"
                };
                spans.push(Span::styled(detail, Style::new().fg(theme.dim)));
            }
            ListItem::new(Line::from(spans))
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

fn draw_orchestration_confirm(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let warning = semantic_foreground(theme.warning, theme.surface, theme.fg);
    let tasks = app.orchestration_plan();
    let workers = tasks.len();
    let focused = app.orchestration_confirm_focused;
    let detailed_height = (workers as u16).saturating_mul(3).saturating_add(9);
    let show_instructions =
        frame.area().width >= 64 && frame.area().height.saturating_sub(2) >= detailed_height;
    let preferred_height = if show_instructions {
        detailed_height
    } else {
        (workers as u16 + 8).clamp(8, 16)
    };
    let area = modal_area(frame.area(), 92, preferred_height);
    frame.render_widget(Clear, area);

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(warning))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            if focused {
                " team plan · review focus "
            } else {
                " team plan ready · press Tab "
            },
            Style::new().fg(warning).add_modifier(Modifier::BOLD),
        ));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let actions = orchestration_confirm_action_lines(focused, inner.width as usize, &theme);
    let action_height = (actions.len() as u16).min(inner.height);
    let desired_header_height = 3;
    let header_height = inner
        .height
        .saturating_sub(action_height)
        .min(desired_header_height);
    let [header_area, tasks_area, action_area] = Layout::vertical([
        Constraint::Length(header_height),
        Constraint::Min(0),
        Constraint::Length(action_height),
    ])
    .areas(inner);

    let start_line = if show_instructions {
        format!("at least {workers} worker calls → 1 synthesis call")
    } else {
        format!("≥{workers} worker calls → synthesis")
    };
    let planner_line = app.orchestration_planner().map_or_else(
        || "1 planner call already ran".to_owned(),
        |model| {
            format!(
                "1 planner call already ran · {} · {}",
                model.display_id(),
                model.provider.label()
            )
        },
    );
    let header_lines = vec![
        Line::from(vec![
            Span::styled("USED   ", Style::new().fg(theme.dim)),
            Span::styled(
                truncate_width(&planner_line, inner.width.saturating_sub(7) as usize),
                Style::new().fg(theme.fg).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("START  ", Style::new().fg(theme.dim)),
            Span::styled(
                truncate_width(&start_line, inner.width.saturating_sub(7) as usize),
                Style::new().fg(theme.warning),
            ),
        ]),
        Line::from(vec![
            Span::styled("SCOPE  ", Style::new().fg(theme.dim)),
            Span::styled(
                truncate_width(
                    if show_instructions {
                        "text → listed providers · workers read-only · only lead may edit"
                    } else {
                        "text shared · read-only"
                    },
                    inner.width.saturating_sub(7) as usize,
                ),
                Style::new().fg(theme.accent2),
            ),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(
            header_lines
                .into_iter()
                .take(header_area.height as usize)
                .collect::<Vec<_>>(),
        ),
        header_area,
    );

    let row_width = tasks_area.width as usize;
    let mut task_lines = vec![Line::styled(
        if show_instructions {
            "TASK SUMMARIES · EXACT MODELS"
        } else {
            "TASKS · EXACT MODELS"
        },
        Style::new().fg(theme.dim).add_modifier(Modifier::BOLD),
    )];
    for task in tasks {
        let id = format!("{}  ", task.id);
        let mut model = format!(
            "{} · {}",
            task.model.display_id(),
            task.model.provider.label()
        );
        let model_budget = row_width
            .saturating_sub(id.width() + 4)
            .min(if show_instructions { 32 } else { 22 });
        model = truncate_width(&model, model_budget);
        let title_budget = row_width
            .saturating_sub(id.width() + UnicodeWidthStr::width(model.as_str()) + 3)
            .max(1);
        let title = truncate_width(&task.title, title_budget);
        let used = id.width()
            + UnicodeWidthStr::width(title.as_str())
            + UnicodeWidthStr::width(model.as_str());
        let gap = " ".repeat(row_width.saturating_sub(used).max(1));
        task_lines.push(Line::from(vec![
            Span::styled(
                id,
                Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(title, Style::new().fg(theme.fg)),
            Span::raw(gap),
            Span::styled(model, Style::new().fg(theme.dim)),
        ]));

        if show_instructions {
            let prefix = "   ↳ ";
            let preview_width = row_width.saturating_sub(prefix.width()).max(1);
            for (index, preview) in instruction_preview(&task.instructions, preview_width, 2)
                .into_iter()
                .enumerate()
            {
                task_lines.push(Line::from(vec![
                    Span::styled(
                        if index == 0 { prefix } else { "     " },
                        Style::new().fg(theme.accent2),
                    ),
                    Span::styled(preview, Style::new().fg(theme.dim)),
                ]));
            }
        }
    }
    task_lines.truncate(tasks_area.height as usize);
    frame.render_widget(Paragraph::new(task_lines), tasks_area);
    frame.render_widget(Paragraph::new(actions), action_area);
}

fn instruction_preview(text: &str, width: usize, max_lines: usize) -> Vec<String> {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let wrapped = textwrap::wrap(&normalized, width.max(1));
    let was_truncated = wrapped.len() > max_lines;
    let mut preview = wrapped
        .into_iter()
        .take(max_lines)
        .map(|line| line.into_owned())
        .collect::<Vec<_>>();
    if was_truncated {
        if let Some(last) = preview.last_mut() {
            *last = truncate_width(&format!("{last} …"), width);
        }
    }
    preview
}

fn orchestration_confirm_action_lines(
    focused: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let success = semantic_foreground(theme.success, theme.surface, theme.fg);
    let error = semantic_foreground(theme.error, theme.surface, theme.fg);
    if !focused {
        let wide = Line::from(vec![
            key_span("Tab", theme.accent),
            Span::styled(" review plan   ·   ", Style::new().fg(theme.fg)),
            key_span("n / Esc", error),
            Span::styled(" cancel", Style::new().fg(theme.fg)),
        ]);
        if wide.width() <= width {
            return vec![wide];
        }
        return vec![
            Line::from(vec![
                key_span("Tab", theme.accent),
                Span::styled(" review", Style::new().fg(theme.fg)),
            ]),
            Line::from(vec![
                key_span("n / Esc", error),
                Span::styled(" cancel", Style::new().fg(theme.fg)),
            ]),
        ];
    }

    let wide = Line::from(vec![
        key_span("y / Enter", success),
        Span::styled(" start   ·   ", Style::new().fg(theme.fg)),
        key_span("n / Esc", error),
        Span::styled(" cancel   ·   ", Style::new().fg(theme.fg)),
        key_span("Tab", theme.accent),
        Span::styled(" back", Style::new().fg(theme.fg)),
    ]);
    if wide.width() <= width {
        vec![wide]
    } else {
        vec![
            Line::from(vec![
                key_span("y / Enter", success),
                Span::styled(" start", Style::new().fg(theme.fg)),
            ]),
            Line::from(vec![
                key_span("n / Esc", error),
                Span::styled(" cancel · ", Style::new().fg(theme.fg)),
                key_span("Tab", theme.accent),
                Span::styled(" back", Style::new().fg(theme.fg)),
            ]),
        ]
    }
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
            section("AGENT"),
            key("/team [2-4]", "Shaltaiboltai lead + read-only workers"),
            key("Esc", "cancel work; focus / deny approval"),
            key("Tab · y/a/n", "focus approval · decide"),
            key("Ctrl+C", "restore queued, then quit"),
        ]
    } else if inner.height >= 8 {
        vec![
            key("Enter", "send / queue next"),
            key("Alt+Enter", "newline"),
            key("/team [2-4]", "lead + read-only workers"),
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
            key("/team", "lead + workers"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::AppEvent;
    use crate::config::Config;
    use crate::mascot::MascotState;
    use crate::orchestration::PlannedTask;
    use crate::providers::{ModelEntry, ProviderKind};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use ratatui_image::picker::{Picker, ProtocolType};
    use tokio::sync::mpsc::unbounded_channel;

    fn test_app() -> App {
        let data_dir =
            std::env::temp_dir().join(format!("shaltai-orchestration-ui-{}", std::process::id()));
        std::env::set_var("SHALTAIBOLTAI_DATA_DIR", data_dir);
        let config = Config {
            anthropic_api_key: None,
            openai_api_key: None,
            openai_base_url: "http://127.0.0.1:9".into(),
            ollama_host: "http://127.0.0.1:9".into(),
            default_model: None,
            compact_threshold_chars: 80_000,
            ollama_num_ctx: 16_384,
            theme: None,
            claude_code_bypass_permissions: false,
            codex_full_access: false,
            reduced_motion: false,
        };
        let (tx, _rx) = unbounded_channel();
        let mut app = App::new(config, tx);
        app.discovering = false;
        app.model = Some(ModelEntry {
            provider: ProviderKind::Ollama,
            id: "team-test".into(),
        });
        app
    }

    fn screen(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buffer[(x, y)].symbol().to_owned())
                    .collect::<String>()
                    + "\n"
            })
            .collect()
    }

    fn is_mascot_cell(cell: &ratatui::buffer::Cell) -> bool {
        matches!(cell.symbol(), "▀" | "▄" | "█")
    }

    fn has_full_mascot(terminal: &Terminal<TestBackend>) -> bool {
        let buffer = terminal.backend().buffer();
        let stage = lead_stage_layout(Rect::new(0, 0, buffer.area.width, 20)).0;
        stage.is_some_and(|stage| {
            stage.width > 0
                && stage.height > 0
                && (stage.y..stage.bottom()).any(|y| {
                    (stage.x..stage.right()).any(|x| {
                        let cell = &buffer[(x, y)];
                        is_mascot_cell(cell)
                    })
                })
        })
    }

    fn has_native_graphics(terminal: &Terminal<TestBackend>) -> bool {
        terminal.backend().buffer().content().iter().any(|cell| {
            cell.skip || cell.symbol().contains('\u{10eeee}') || cell.symbol().contains('\x1b')
        })
    }

    fn native_mascot() -> mascot::NativeMascot {
        let mut picker = Picker::halfblocks();
        picker.set_protocol_type(ProtocolType::Kitty);
        mascot::NativeMascot::from_picker(picker)
            .unwrap()
            .expect("forced Kitty renderer")
    }

    fn buffer_region(
        terminal: &Terminal<TestBackend>,
        area: Rect,
    ) -> Vec<(String, Color, Color, Modifier)> {
        let buffer = terminal.backend().buffer();
        area.rows()
            .flat_map(|row| {
                row.columns().map(|position| {
                    let cell = &buffer[position];
                    (cell.symbol().to_owned(), cell.fg, cell.bg, cell.modifier)
                })
            })
            .collect()
    }

    fn show_confirmation(app: &mut App) {
        app.textarea.insert_str("/team 2");
        app.submit_input();
        app.textarea.insert_str("coordinate this change");
        app.submit_input();
        let run_id = app.orchestration_run_id().expect("orchestration run");
        let model = app.model.clone().expect("test model");
        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![
                PlannedTask {
                    id: 1,
                    title: "inspect state".into(),
                    instructions:
                        "read the relevant files and summarize concrete evidence without edits"
                            .into(),
                    model: model.clone(),
                },
                PlannedTask {
                    id: 2,
                    title: "review risks".into(),
                    instructions: "identify safety gaps".into(),
                    model,
                },
            ]),
        });
    }

    #[tokio::test]
    async fn lead_mascot_and_worker_card_render_at_standard_size() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        app.transcript = vec![Entry::Agent {
            name: "agent 1 · inspect state".into(),
            model: "team-test · ollama".into(),
            status: "RUNNING".into(),
            summary: "reviewing the task in a read-only sandbox…".into(),
            is_error: false,
        }];
        app.transcript_rev += 1;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("SHALTAIBOLTAI"), "{rendered}");
        assert!(!rendered.contains("REAL AGENT"), "{rendered}");
        assert!(!rendered.contains("DANCING"), "{rendered}");
        assert!(has_full_mascot(&terminal), "{rendered}");
        assert!(rendered.contains("RUNNING"), "{rendered}");
        assert!(
            rendered.contains("AGENT · team-test · ollama"),
            "{rendered}"
        );
        assert!(rendered.contains("read-only sandbox"), "{rendered}");
    }

    #[tokio::test]
    async fn lead_mascot_remains_visible_at_narrow_size() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("◆ shaltaiboltai"), "{rendered}");
        assert!(!rendered.contains("╭⌒▾⌒╮"), "{rendered}");
        assert!(!has_full_mascot(&terminal), "{rendered}");
        assert!(rendered.contains("TEAM"), "{rendered}");
    }

    #[tokio::test]
    async fn native_mascot_yields_to_modals_and_compact_layouts() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Streaming;
        let native = native_mascot();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(has_native_graphics(&terminal));

        app.mode = Mode::Help;
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));

        app.mode = Mode::Streaming;
        terminal.backend_mut().resize(60, 20);
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));
        assert!(screen(&terminal).contains("◆ shaltaiboltai"));
    }

    #[tokio::test]
    async fn medium_terminal_keeps_conversation_space_and_uses_compact_signature() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("◆ shaltaiboltai"), "{rendered}");
        assert!(!rendered.contains("╭⌒▾⌒╮"), "{rendered}");
        assert!(!has_full_mascot(&terminal), "{rendered}");
        assert!(rendered.contains("Ready to build"), "{rendered}");
        assert!(rendered.contains("TEAM"), "{rendered}");
        assert!(rendered.contains("next message"), "{rendered}");
    }

    #[test]
    fn full_stage_layout_is_geometry_only_and_keeps_a_conversation_viewport() {
        let (stage, conversation) = lead_stage_layout(Rect::new(0, 0, 80, 20));
        let stage = stage.expect("standard terminal should show the full mascot");
        assert_eq!(stage, Rect::new(0, 0, LEAD_STAGE_WIDTH, 20));
        assert_eq!(conversation, Rect::new(LEAD_STAGE_WIDTH, 0, 50, 20));

        let (stage, conversation) = lead_stage_layout(Rect::new(0, 0, 80, 19));
        assert_eq!(stage, Some(Rect::new(0, 0, LEAD_STAGE_WIDTH, 19)));
        assert_eq!(conversation, Rect::new(LEAD_STAGE_WIDTH, 0, 50, 19));

        let (stage, conversation) = lead_stage_layout(Rect::new(0, 0, 40, 8));
        assert!(stage.is_none());
        assert_eq!(conversation, Rect::new(0, 0, 40, 8));

        for constrained in [Rect::new(0, 0, 77, 20), Rect::new(0, 0, 80, 18)] {
            let (stage, conversation) = lead_stage_layout(constrained);
            assert!(stage.is_none());
            assert_eq!(conversation, constrained);
        }
    }

    #[tokio::test]
    async fn working_pose_advances_without_touching_transcript_cache_or_scroll() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        app.transcript = (0..30)
            .map(|index| Entry::Info(format!("evidence line {index}")))
            .collect();
        app.transcript_rev += 1;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        app.scroll_from_bottom = 7;
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let (stage, conversation) = lead_stage_layout(Rect::new(0, 0, 80, 20));
        let stage = stage.expect("full mascot stage");
        let before = buffer_region(&terminal, conversation);
        let stage_before = buffer_region(&terminal, stage);
        let cache_len = app.render_cache.len();
        let cache_starts = app.render_cache_starts.clone();
        let cache_total = app.render_cache_total_lines;
        let cache_rev = app.render_cache_rev;
        let scroll = app.scroll_from_bottom;

        app.advance_animation();
        app.advance_animation();
        app.advance_animation();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let after = buffer_region(&terminal, conversation);
        let stage_after = buffer_region(&terminal, stage);

        assert_eq!(before, after);
        assert_ne!(stage_before, stage_after);
        assert_eq!(app.render_cache.len(), cache_len);
        assert_eq!(app.render_cache_starts, cache_starts);
        assert_eq!(app.render_cache_total_lines, cache_total);
        assert_eq!(app.render_cache_rev, cache_rev);
        assert_eq!(app.scroll_from_bottom, scroll);
        assert!(has_full_mascot(&terminal));
    }

    #[tokio::test]
    async fn mascot_state_tracks_agent_lifecycle_without_animating_consent_screens() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        assert_eq!(app.mascot_state(), MascotState::Idle);
        assert!(!app.needs_animation());

        app.mode = Mode::Streaming;
        assert_eq!(app.mascot_state(), MascotState::Working);
        assert!(app.needs_animation());

        app.mode = Mode::Approval;
        assert_eq!(app.mascot_state(), MascotState::Waiting);
        assert!(!app.needs_animation());

        app.mode = Mode::Input;
        app.compacting = true;
        assert_eq!(app.mascot_state(), MascotState::Thinking);
        assert!(app.needs_animation());

        app.compacting = false;
        app.mode = Mode::Streaming;
        app.config.reduced_motion = true;
        app.advance_animation();
        assert_eq!(app.mascot_state(), MascotState::Working);
        assert!(!app.needs_animation());
        assert_eq!(app.animation_tick(), 0);
    }

    #[tokio::test]
    async fn mascot_keeps_source_colors_and_terminal_theme_keeps_reset_background() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        for selected in theme::all() {
            let mut app = test_app();
            app.theme = *selected;
            app.mode = Mode::Orchestrating;
            let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            let buffer = terminal.backend().buffer();
            let luminance = |color| match color {
                Color::White => Some(1.0),
                Color::Black => Some(0.0),
                _ => relative_luminance(color),
            };
            let assert_contrast = |label: &str, foreground: Color, background: Color| {
                let foreground = luminance(foreground).expect("known foreground");
                let background = luminance(background).expect("known background");
                let contrast =
                    (foreground.max(background) + 0.05) / (foreground.min(background) + 0.05);
                assert!(
                    contrast >= 4.5,
                    "{} {label} contrast is {contrast:.2}:1",
                    selected.name
                );
            };
            let stage_bg = selected.surface.or(selected.bg);

            let brand_x = (0..80)
                .find(|x| buffer[(*x, 0)].symbol() == "S")
                .expect("SHALTAIBOLTAI title");
            let brand = &buffer[(brand_x, 0)];
            assert_eq!(
                brand.fg,
                semantic_foreground(selected.accent, stage_bg, selected.fg),
                "{}",
                selected.name
            );

            if let Some(background) = stage_bg {
                assert_contrast("brand", brand.fg, background);
            }

            let stage = lead_stage_layout(Rect::new(0, 0, 80, 20))
                .0
                .expect("mascot stage");
            let mascot_cells = stage
                .rows()
                .flat_map(|row| row.columns())
                .filter(|position| is_mascot_cell(&buffer[*position]))
                .map(|position| buffer[position].clone())
                .collect::<Vec<_>>();
            let colors = mascot_cells
                .iter()
                .flat_map(|cell| [cell.fg, cell.bg])
                .filter_map(|color| match color {
                    Color::Rgb(r, g, b) => Some((r, g, b)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if selected.name == "terminal" {
                assert!(colors.is_empty(), "terminal mascot emitted truecolor");
                let ansi = mascot_cells
                    .iter()
                    .flat_map(|cell| [cell.fg, cell.bg])
                    .collect::<Vec<_>>();
                for expected in [
                    Color::White,
                    Color::Blue,
                    Color::Red,
                    Color::Cyan,
                    Color::Yellow,
                ] {
                    assert!(
                        ansi.contains(&expected),
                        "terminal mascot missing {expected:?}"
                    );
                }
                assert_eq!(buffer[(1, 1)].bg, Color::Reset);
                continue;
            }
            for (label, present) in [
                (
                    "warm egg shell",
                    colors
                        .iter()
                        .any(|&(r, g, b)| r > 190 && g > 155 && b > 105),
                ),
                (
                    "navy coat",
                    colors.iter().any(|&(r, g, b)| b > 45 && b > r && b > g),
                ),
                (
                    "red waistcoat",
                    colors
                        .iter()
                        .any(|&(r, g, _)| r > 120 && r > g.saturating_add(35)),
                ),
                (
                    "teal breeches",
                    colors
                        .iter()
                        .any(|&(r, g, b)| b > 65 && g > r.saturating_add(20)),
                ),
            ] {
                assert!(present, "{} missing source-derived {label}", selected.name);
            }
        }
    }

    #[tokio::test]
    async fn armed_team_composer_discloses_the_planning_call() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.textarea.insert_str("/team 2");
        app.submit_input();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(
            rendered.contains("Enter starts 1 planning call · 2 workers after review"),
            "{rendered}"
        );
    }

    #[tokio::test]
    async fn team_confirmation_is_responsive_and_keeps_safety_controls_visible() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        show_confirmation(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("team plan ready"), "{rendered}");
        assert!(
            rendered.contains("1 planner call already ran"),
            "{rendered}"
        );
        assert!(rendered.contains("2 worker calls"), "{rendered}");
        assert!(rendered.contains("1 synthesis call"), "{rendered}");
        assert!(rendered.contains("listed providers"), "{rendered}");
        assert!(rendered.contains("workers read-only"), "{rendered}");
        assert!(
            rendered.contains("TASK SUMMARIES · EXACT MODELS"),
            "{rendered}"
        );
        assert!(rendered.contains("read the relevant files"), "{rendered}");
        assert!(rendered.contains("team-test · ollama"), "{rendered}");
        assert!(rendered.contains("Tab"), "{rendered}");
        assert!(rendered.contains("n / Esc"), "{rendered}");

        let mut app = test_app();
        show_confirmation(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("team plan ready"), "{rendered}");
        assert!(
            rendered.contains("1 planner call already ran"),
            "{rendered}"
        );
        assert!(rendered.contains("2 worker calls"), "{rendered}");
        assert!(rendered.contains("synthesis"), "{rendered}");
        assert!(rendered.contains("text shared · read-only"), "{rendered}");
        assert!(rendered.contains("TASKS · EXACT MODELS"), "{rendered}");
        assert!(rendered.contains("team-test · ollama"), "{rendered}");
        assert!(rendered.contains("Tab"), "{rendered}");
        assert!(rendered.contains("n / Esc"), "{rendered}");
    }
}
