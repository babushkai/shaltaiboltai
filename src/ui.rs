use crate::app::{App, Entry, Mode, PermissionOverlay, PERMISSION_PRESETS};
use crate::markdown;
use crate::mascot;
use crate::providers::{self, ProviderKind};
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

    let header_height = if frame.area().height >= 6 { 2 } else { 1 };
    let input_height = input_height(app, frame.area().height);
    let [header_area, transcript_area, input_area] = Layout::vertical([
        Constraint::Length(header_height),
        Constraint::Min(1),
        Constraint::Length(input_height),
    ])
    .areas(frame.area());

    draw_header(frame, app, header_area);
    draw_transcript(frame, app, transcript_area);
    let slash_menu_active = app.mode == Mode::Input && app.slash_menu_active();
    if frame.area().width >= 200
        && frame.area().height >= 70
        && app.transcript.len() == 1
        && !app.is_busy()
        && app.permission_overlay.is_none()
        && !slash_menu_active
        && !matches!(
            app.mode,
            Mode::ModelPicker
                | Mode::SessionPicker
                | Mode::ThemePicker
                | Mode::Approval
                | Mode::OrchestrationConfirm
                | Mode::Help
        )
    {
        draw_inline_mascot(frame, app, transcript_area, native_mascot);
    }
    draw_input(frame, app, input_area);
    if slash_menu_active {
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
    match app.permission_overlay {
        Some(PermissionOverlay::Picker) => draw_permissions(frame, app),
        Some(PermissionOverlay::FullAccessConfirm) => draw_full_access_confirmation(frame, app),
        None => {}
    }
}

fn draw_inline_mascot(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    native_mascot: Option<&mascot::NativeMascot>,
) {
    let theme = app.theme;
    let state = app.mascot_state();
    if let Some(native_mascot) = native_mascot {
        let image = native_mascot.protocol(state, app.animation_tick());
        let size = image.area();
        if let Some(image_area) = inline_mascot_area(area, size) {
            if !mascot_region_is_clear(frame, image_area, theme.bg) {
                return;
            }
            frame.render_widget(TerminalImage::new(image), image_area);
            return;
        }
        return;
    }

    let size = Rect::new(
        0,
        0,
        mascot::FRAME_WIDTH as u16,
        mascot::FRAME_HEIGHT as u16,
    );
    let Some(art_area) = inline_mascot_area(area, size) else {
        return;
    };
    if !mascot_region_is_clear(frame, art_area, theme.bg) {
        return;
    }
    let pose = mascot::frame(state, app.animation_tick());
    let art = pose
        .cells
        .iter()
        .map(|row| mascot_line(row, &theme))
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(art), art_area);
}

/// Place the mascot inside the transcript border without changing transcript
/// width. If the full artwork does not fit, the title remains the compact
/// signature instead.
fn inline_mascot_area(area: Rect, size: Rect) -> Option<Rect> {
    let inner = area.inner(Margin {
        vertical: 1,
        horizontal: 1,
    });
    if size.width > inner.width || size.height > inner.height {
        return None;
    }
    Some(Rect::new(
        inner.right() - size.width,
        inner.bottom() - size.height,
        size.width,
        size.height,
    ))
}

/// The mascot is decoration, so transcript content always wins. Inspect the
/// already-rendered main pane and draw only over untouched background cells;
/// this also keeps code-card surfaces and Kitty placeholders intact.
fn mascot_region_is_clear(frame: &mut Frame, area: Rect, background: Option<Color>) -> bool {
    let expected_background = background.unwrap_or(Color::Reset);
    let left_guard = (area.x > frame.area().x).then_some(area.x - 1);
    let buffer = frame.buffer_mut();
    let body_is_clear = area.rows().flat_map(|row| row.columns()).all(|position| {
        buffer.cell(position).is_some_and(|cell| {
            !cell.skip
                && cell.bg == expected_background
                && cell.symbol().chars().all(char::is_whitespace)
        })
    });
    body_is_clear
        && left_guard.is_none_or(|x| {
            (area.y..area.bottom()).all(|y| {
                buffer
                    .cell((x, y))
                    .is_none_or(|cell| cell.symbol().width() <= 1)
            })
        })
}

fn mascot_line(row: &[mascot::MascotCell; mascot::FRAME_WIDTH], theme: &Theme) -> Line<'static> {
    let panel_bg = theme.bg;
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

/// The composer is an elevated band with one structural rule. It deliberately
/// avoids another closed rounded box beneath the conversation surface.
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
    let compact = area.width < 48;
    let title = if queued && !compact {
        " › next message queued "
    } else if queued {
        " › queued "
    } else if team_workers.is_some() {
        " › team prompt "
    } else if lookahead {
        " › next message "
    } else {
        " › "
    };
    let mut block = Block::default()
        .borders(Borders::TOP)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(border))
        .padding(Padding::horizontal(1))
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
    let draft_empty = app
        .textarea
        .lines()
        .iter()
        .all(|line| line.trim().is_empty());
    let footer = if let Some(notice) = app.composer_notice() {
        Some((notice.to_owned(), theme.warning))
    } else if app.mode == Mode::Approval && !app.approval_focused && queued {
        Some(("Tab review tool · next message queued".into(), theme.dim))
    } else if app.mode == Mode::Approval && !app.approval_focused {
        Some((
            if compact {
                "Tab review · Enter queue".into()
            } else {
                "Tab review tool · Enter queue · Alt+Enter newline".into()
            },
            theme.dim,
        ))
    } else if focused && lookahead {
        Some((
            if compact {
                "Esc cancel · Enter queue".into()
            } else {
                "Esc cancel · Enter queue · Alt+Enter newline".into()
            },
            theme.dim,
        ))
    } else if let Some(workers) = team_workers {
        Some((
            if compact {
                format!("Enter plan · {workers} workers")
            } else {
                format!("Enter starts 1 planning call · {workers} workers after review")
            },
            theme.accent2,
        ))
    } else if queued && app.mode == Mode::Approval {
        Some(("waiting for tool decision · n / Esc deny".into(), theme.dim))
    } else if queued {
        Some((
            if compact {
                "Esc cancel · waiting".into()
            } else {
                "Esc cancel · waiting for current turn".into()
            },
            theme.dim,
        ))
    } else if focused && draft_empty {
        Some((
            if compact {
                "Enter send · / commands".into()
            } else {
                "Enter send · Alt+Enter newline · / commands".into()
            },
            theme.dim,
        ))
    } else {
        None
    };
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
    let show_footer = area.height >= 3;
    let [editor_area, footer_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(u16::from(show_footer)),
    ])
    .areas(area);
    frame.render_widget(&app.textarea, editor_area);
    if show_footer {
        if let Some(surface) = theme.surface {
            frame.render_widget(
                Block::default().style(Style::new().bg(surface).fg(theme.fg)),
                footer_area,
            );
        }
        if let Some((footer, color)) = footer {
            let footer = truncate_width(&footer, footer_area.width.saturating_sub(3) as usize);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("  ", Style::new().fg(border)),
                    Span::styled(footer, Style::new().fg(color)),
                ]))
                .alignment(Alignment::Right),
                footer_area,
            );
        }
    }
}

/// Renders the transcript through a per-entry line cache with cumulative line
/// offsets. Only dirty/new entries are parsed, and locating the viewport is a
/// binary search rather than a walk from the beginning of the conversation.
fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    // Two cells of editorial breathing room on each side. There is no outer
    // transcript frame: the HUD and composer rules provide the shell.
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
    let visible = area.height as usize;
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

    let inner = area.inner(Margin {
        vertical: 0,
        horizontal: 2,
    });
    frame.render_widget(Paragraph::new(window), inner);

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
            area,
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
                    "━━╾ ",
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
                    "YOU",
                    Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ·  prompt", Style::new().fg(theme.dim)),
            ]));
            push_wrapped(
                &mut lines,
                "▎ ",
                Style::new().fg(theme.accent),
                text,
                width,
                Style::new().fg(theme.secondary),
            );
        }
        Entry::Assistant(text) => {
            if !text.is_empty() || streaming {
                lines.push(Line::from(vec![
                    Span::styled(
                        "◆  ",
                        Style::new().fg(theme.accent2).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "SHALTAIBOLTAI",
                        Style::new()
                            .fg(theme.secondary)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
                if text.is_empty() {
                    lines.push(Line::from(vec![
                        Span::raw("   "),
                        Span::styled("thinking…", Style::new().fg(theme.dim)),
                    ]));
                } else {
                    for line in markdown::render(text, width.saturating_sub(3), theme) {
                        let mut spans = Vec::with_capacity(line.spans.len() + 1);
                        spans.push(Span::raw("   "));
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
                ("failed", "✗", theme.error)
            } else {
                ("done", "✓", theme.success)
            };
            let result_lines = result.lines().count();
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{glyph}  "),
                    Style::new()
                        .fg(semantic_foreground(color, theme.bg, theme.fg))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "tool",
                    Style::new()
                        .fg(theme.secondary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" · {state}"), Style::new().fg(color)),
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
                "▎ ",
                Style::new().fg(color),
                summary,
                width,
                Style::new().fg(theme.secondary),
            );
            let shown = result_lines.min(TOOL_RESULT_PREVIEW_LINES);
            for (i, line) in result.lines().take(shown).enumerate() {
                push_wrapped(
                    &mut lines,
                    "   ",
                    Style::new().fg(theme.dim),
                    line,
                    width,
                    Style::new().fg(theme.dim),
                );
                if i + 1 == shown && result_lines > shown {
                    lines.push(Line::from(vec![
                        Span::raw("   "),
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
                ("✗", theme.error)
            } else if running {
                ("◆", theme.accent2)
            } else {
                ("✓", theme.success)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{glyph}  "),
                    Style::new()
                        .fg(semantic_foreground(color, theme.bg, theme.fg))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "agent",
                    Style::new()
                        .fg(theme.secondary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" · {} · ", status.to_ascii_lowercase()),
                    Style::new().fg(color),
                ),
                Span::styled(model.clone(), Style::new().fg(theme.accent2)),
            ]));
            push_wrapped(
                &mut lines,
                "▎ ",
                Style::new().fg(color),
                name,
                width,
                Style::new()
                    .fg(theme.secondary)
                    .add_modifier(Modifier::BOLD),
            );
            if !summary.is_empty() {
                push_wrapped(
                    &mut lines,
                    "   ",
                    Style::new().fg(theme.dim),
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
        Entry::Status { title, fields } => {
            let value = |label: &str| {
                fields
                    .iter()
                    .find_map(|(field, value)| (field == label).then_some(value.as_str()))
                    .unwrap_or("—")
            };
            if width < 44 {
                let permissions = value("Permissions");
                let compact_permissions = if permissions.starts_with("Workspace") {
                    "Ask for approval"
                } else if permissions.starts_with("Read Only") {
                    "Read Only"
                } else if permissions.starts_with("Full Access") {
                    "Full Access"
                } else {
                    permissions
                };
                let compact = [
                    ("RUNTIME", value("Model")),
                    ("Provider", value("Model provider")),
                    ("WORKSPACE", value("Directory")),
                    ("Permissions", compact_permissions),
                    ("USAGE", value("Context window")),
                ];
                lines.push(Line::from(vec![
                    Span::styled(
                        "STATUS",
                        Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" · ", Style::new().fg(theme.border)),
                    Span::styled(
                        truncate_width(title, width.saturating_sub(9)),
                        Style::new().fg(theme.secondary),
                    ),
                ]));
                for (label, field_value) in compact {
                    let prefix = format!("{label}  ");
                    lines.push(Line::from(vec![
                        Span::styled(
                            prefix.clone(),
                            Style::new().fg(theme.dim).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            truncate_width(field_value, width.saturating_sub(prefix.width())),
                            Style::new().fg(theme.secondary),
                        ),
                    ]));
                }
                return lines;
            }
            if width < 68 {
                lines.push(Line::from(vec![
                    Span::styled(
                        "STATUS",
                        Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" · ", Style::new().fg(theme.border)),
                    Span::styled(title.clone(), Style::new().fg(theme.secondary)),
                ]));
                for (section, labels) in [
                    (
                        "RUNTIME",
                        ["Model", "Model provider", "Enforcement"].as_slice(),
                    ),
                    (
                        "WORKSPACE",
                        ["Directory", "Permissions", "Network"].as_slice(),
                    ),
                    ("USAGE", ["Token usage", "Context window"].as_slice()),
                ] {
                    lines.push(Line::styled(
                        section,
                        Style::new().fg(theme.dim).add_modifier(Modifier::BOLD),
                    ));
                    for label in labels {
                        let prefix = format!("{label:<14}  ");
                        lines.push(Line::from(vec![
                            Span::styled(prefix.clone(), Style::new().fg(theme.dim)),
                            Span::styled(
                                truncate_width(value(label), width.saturating_sub(prefix.width())),
                                Style::new().fg(theme.secondary),
                            ),
                        ]));
                    }
                }
                return lines;
            }
            let label_width = fields
                .iter()
                .map(|(label, _)| UnicodeWidthStr::width(label.as_str()))
                .max()
                .unwrap_or(0)
                .min(18);
            lines.push(Line::from(vec![
                Span::styled(
                    "STATUS",
                    Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" · ", Style::new().fg(theme.border)),
                Span::styled(title.clone(), Style::new().fg(theme.secondary)),
            ]));
            for (label, value) in fields {
                let section = match label.as_str() {
                    "Model" => Some("RUNTIME"),
                    "Directory" => Some("WORKSPACE"),
                    "Token usage" => Some("USAGE"),
                    _ => None,
                };
                if let Some(section) = section {
                    lines.push(Line::raw(""));
                    lines.push(Line::from(vec![
                        Span::styled(
                            section,
                            Style::new().fg(theme.dim).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ━━━╾", Style::new().fg(theme.accent)),
                    ]));
                }
                let prefix = format!("{label:<label_width$}  ");
                push_wrapped(
                    &mut lines,
                    &prefix,
                    Style::new().fg(theme.dim),
                    value,
                    width,
                    Style::new().fg(theme.secondary),
                );
            }
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

/// Product HUD derived from the original TypeScript shell: one compact seal,
/// one authored identity, quiet runtime metadata, and a single separator.
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let theme = app.theme;
    let surface_style = theme.surface.map_or_else(
        || Style::new().fg(theme.fg),
        |surface| Style::new().bg(surface).fg(theme.fg),
    );
    let mut shell = Block::default().style(surface_style);
    if area.height >= 2 {
        shell = shell
            .borders(Borders::BOTTOM)
            .border_type(BorderType::Plain)
            .border_style(Style::new().fg(theme.border));

        let workspace = workspace_context_label(
            &app.cwd_display,
            app.git_branch.as_deref(),
            (area.width / 2) as usize,
        );
        if !workspace.is_empty() && area.width >= 44 {
            shell = shell.title_bottom(Line::styled(
                format!(" {workspace} "),
                Style::new().fg(theme.dim),
            ));
        }

        let usage = if app.scroll_from_bottom > 0 {
            if area.width >= 48 {
                format!(
                    " ↑ {} lines from latest · Ctrl+End jump ",
                    app.scroll_from_bottom
                )
            } else {
                format!(" ↑ {} · Ctrl+End ", app.scroll_from_bottom)
            }
        } else {
            match app.last_usage {
                Some(usage) => format!(
                    " ctx {} · out {} ",
                    fmt_count(usage.input_tokens as usize),
                    fmt_count(usage.output_tokens as usize)
                ),
                None if app.approx_tokens() > 0 => {
                    format!(" ctx ~{} ", fmt_count(app.approx_tokens()))
                }
                None => String::new(),
            }
        };
        if !usage.is_empty() {
            let color = if app.scroll_from_bottom > 0 {
                semantic_foreground(theme.warning, theme.surface, theme.fg)
            } else {
                theme.dim
            };
            shell = shell.title_bottom(
                Line::styled(usage, Style::new().fg(color)).alignment(Alignment::Right),
            );
        }
    }
    frame.render_widget(shell, area);

    let wide = area.width >= 58;
    let orchestration_status = app.orchestration_status();
    let team_identity_critical = orchestration_status.is_some()
        || app.team_workers().is_some()
        || matches!(app.mode, Mode::OrchestrationConfirm | Mode::Orchestrating);
    let (state, state_color) = if let Some(status) = orchestration_status {
        (status, theme.accent2)
    } else if app.compacting {
        ("compacting context…".into(), theme.accent)
    } else if let Some(workers) = app.team_workers().filter(|_| app.mode == Mode::Input) {
        (format!("TEAM · {workers} workers armed"), theme.accent2)
    } else if app.discovering && app.mode == Mode::Input {
        ("discovering models…".into(), theme.accent)
    } else {
        match app.mode {
            Mode::Input => ("ready".into(), theme.success),
            Mode::Streaming => ("thinking".into(), theme.accent),
            Mode::RunningTool => ("running tool".into(), theme.accent2),
            Mode::Approval => ("approval needed".into(), theme.warning),
            Mode::OrchestrationConfirm => ("team plan ready".into(), theme.warning),
            Mode::Orchestrating => ("team working".into(), theme.accent2),
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
    let mut state = if app.mode == Mode::Approval && !app.approval_focused {
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
    if team_identity_critical && area.width < 58 {
        state = compact_team_state(&state).to_owned();
    }

    let mut left = vec![Span::styled(
        " SB ",
        Style::new()
            .fg(theme.bg.unwrap_or_else(|| on_color(theme.accent)))
            .bg(theme.accent)
            .add_modifier(Modifier::BOLD),
    )];
    if area.width >= 18 && !(team_identity_critical && area.width < 58) {
        left.push(Span::styled(
            if area.width >= 34 {
                "  SHALTAIBOLTAI"
            } else {
                "  SHALTAI"
            },
            Style::new().fg(theme.fg).add_modifier(Modifier::BOLD),
        ));
    }
    let left_width = left.iter().map(Span::width).sum::<usize>();
    frame.render_widget(
        Paragraph::new(Line::from(left)),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let model = app.model.as_ref().map(|model| {
        (
            model.display_id().to_owned(),
            model.provider.label().to_owned(),
        )
    });
    let permission = app
        .effective_execution_policy()
        .matching_preset()
        .map_or_else(
            || app.effective_execution_policy().sandbox_mode().label(),
            |preset| preset.label(),
        );
    let available = area.width as usize;
    let state_width = UnicodeWidthStr::width(state.as_str()) + if app.is_busy() { 2 } else { 1 };
    let mut right = Vec::new();
    let room = available.saturating_sub(left_width + 2);
    let model_width = model
        .as_ref()
        .map(|(id, provider)| UnicodeWidthStr::width(format!("{id} · {provider}").as_str()))
        .unwrap_or_default();
    let permission_width = UnicodeWidthStr::width(permission);
    if team_identity_critical {
        let include_permission =
            area.width >= 100 && room > state_width + permission_width + model_width.min(24) + 6;
        let permission_reserve = if include_permission {
            permission_width + 3
        } else {
            0
        };
        if let Some((id, provider)) = model {
            let model_budget = room.saturating_sub(state_width + permission_reserve + 3);
            if model_budget > 0 {
                right.push(Span::styled(
                    compact_model_identity(&id, &provider, model_budget),
                    Style::new().fg(theme.accent2),
                ));
                right.push(Span::styled(" · ", Style::new().fg(theme.border)));
            }
        }
        if include_permission {
            right.push(Span::styled(permission, Style::new().fg(theme.secondary)));
            right.push(Span::styled(" · ", Style::new().fg(theme.border)));
        }
    } else {
        if wide && room > state_width + permission_width + model_width + 6 {
            if let Some((id, provider)) = model {
                right.push(Span::styled(
                    format!("{id} · {provider}"),
                    Style::new().fg(theme.accent2),
                ));
                right.push(Span::styled(" · ", Style::new().fg(theme.border)));
            }
        }
        if room > state_width + permission_width + 3 {
            right.push(Span::styled(permission, Style::new().fg(theme.secondary)));
            right.push(Span::styled(" · ", Style::new().fg(theme.border)));
        }
    }
    right.push(if app.is_busy() {
        Span::styled(
            format!("{} ", spinner_frame(app.animation_tick())),
            Style::new().fg(theme.accent),
        )
    } else {
        Span::styled("● ", Style::new().fg(state_color))
    });
    right.push(Span::styled(
        state,
        Style::new().fg(semantic_foreground(state_color, theme.surface, theme.fg)),
    ));
    let right_x = area
        .x
        .saturating_add((left_width + 1).min(area.width as usize) as u16);
    let right_area = Rect::new(right_x, area.y, area.right().saturating_sub(right_x), 1);
    frame.render_widget(
        Paragraph::new(Line::from(right)).alignment(Alignment::Right),
        right_area,
    );
}

fn compact_team_state(state: &str) -> &str {
    if state.contains("approval needed") {
        "TEAM · approval"
    } else if state.contains("applying") {
        "TEAM · apply · Esc"
    } else if state.contains("synthesizing") {
        "TEAM · synth · Esc"
    } else if state.contains("armed") {
        "TEAM · armed"
    } else if state.contains("workers") {
        "TEAM · work · Esc"
    } else if state.contains("planning") {
        "TEAM · planning"
    } else if state.contains("plan ready") {
        "TEAM · ready"
    } else if state.contains("working") || state.contains("worker") {
        "TEAM · working"
    } else {
        "TEAM"
    }
}

fn compact_model_identity(id: &str, provider: &str, max: usize) -> String {
    let full = format!("{id} · {provider}");
    if UnicodeWidthStr::width(full.as_str()) <= max {
        return full;
    }
    let provider_width = UnicodeWidthStr::width(provider);
    let separator = " · ";
    let separator_width = UnicodeWidthStr::width(separator);
    if max <= provider_width + separator_width {
        return truncate_width(provider, max);
    }
    let id_budget = max - provider_width - separator_width;
    format!(
        "{}{}{}",
        truncate_left_width(id, id_budget),
        separator,
        provider
    )
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

/// Keep the branch (the volatile, actionable value) intact when possible and
/// compact the directory from the left so its project-name tail remains
/// recognizable. This avoids long worktree paths erasing the branch entirely.
fn workspace_context_label(cwd: &str, branch: Option<&str>, max: usize) -> String {
    let Some(branch) = branch.filter(|branch| !branch.is_empty()) else {
        return truncate_left_width(cwd, max);
    };
    if cwd.is_empty() {
        return truncate_width(branch, max);
    }
    let separator = " · ";
    let branch_width = UnicodeWidthStr::width(branch);
    let separator_width = UnicodeWidthStr::width(separator);
    if branch_width + separator_width >= max {
        return truncate_width(branch, max);
    }
    let cwd_budget = max - branch_width - separator_width;
    format!(
        "{}{}{}",
        truncate_left_width(cwd, cwd_budget),
        separator,
        branch
    )
}

fn truncate_left_width(text: &str, max: usize) -> String {
    if UnicodeWidthStr::width(text) <= max {
        return text.to_owned();
    }
    if max == 0 {
        return String::new();
    }
    let mut reversed = String::new();
    let mut width = 0;
    for ch in text.chars().rev() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width + 1 > max {
            break;
        }
        reversed.push(ch);
        width += ch_width;
    }
    format!("…{}", reversed.chars().rev().collect::<String>())
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

fn selection_style(theme: &Theme) -> Style {
    let mut style = Style::new().fg(theme.fg).add_modifier(Modifier::BOLD);
    if let Some(selection) = theme.hover.or(theme.elevated).or(theme.surface) {
        style = style.bg(selection);
    }
    style
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
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let list = List::new(items)
        .block(block)
        .highlight_style(selection_style(&theme))
        .highlight_symbol("▎ ");

    let mut state = ListState::default();
    state.select(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_model_picker(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let models = app.filtered_models();
    let configured_default = app.config.default_model.as_deref().and_then(|selector| {
        providers::qualified_model_selector(selector).or_else(|| {
            app.models
                .iter()
                .find(|model| model.id == selector)
                .cloned()
        })
    });
    let provider_width = models
        .iter()
        .map(|model| UnicodeWidthStr::width(model.provider.label()))
        .max()
        .unwrap_or(0);
    let picker_width = 110.min(frame.area().width.saturating_sub(2));
    let row_width = picker_width.saturating_sub(4) as usize;
    let items: Vec<ListItem> = models
        .iter()
        .map(|m| {
            let is_default = configured_default
                .as_ref()
                .is_some_and(|default| default.provider == m.provider && default.id == m.id);
            let is_current = app
                .model
                .as_ref()
                .is_some_and(|current| current.provider == m.provider && current.id == m.id);
            let marker = match (is_current, is_default) {
                (true, true) => "● current/default · ",
                (true, false) => "● current · ",
                (false, true) => "◇ default · ",
                (false, false) => "",
            };
            let id_budget =
                row_width.saturating_sub(provider_width + 2 + UnicodeWidthStr::width(marker));
            let mut spans = vec![
                Span::styled(
                    format!("{:<provider_width$}  ", m.provider.label()),
                    Style::new().fg(theme.accent2),
                ),
                Span::styled(
                    marker,
                    Style::new()
                        .fg(if is_current { theme.accent } else { theme.dim })
                        .add_modifier(if is_current {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    truncate_width(m.display_id(), id_budget),
                    Style::new().fg(theme.fg),
                ),
            ];
            let detail = if m.provider.is_sub_agent() {
                Some(if providers::is_cli_default_model(m) {
                    "unpinned · solo-only · subscription sub-agent"
                } else if m.is_claude_alias() {
                    "latest alias · subscription sub-agent"
                } else {
                    "subscription sub-agent"
                })
            } else if m.provider == ProviderKind::OpenRouter {
                Some(if crate::providers::openrouter::is_variable_model(&m.id) {
                    "variable route · solo-only · pricing varies"
                } else {
                    "routed API · pricing varies"
                })
            } else {
                None
            };
            let mut lines = vec![Line::from(std::mem::take(&mut spans))];
            if let Some(detail) = detail {
                lines.push(Line::from(vec![
                    Span::raw(" ".repeat(provider_width + 2)),
                    Span::styled(
                        truncate_width(detail, row_width.saturating_sub(provider_width + 2)),
                        Style::new().fg(theme.dim),
                    ),
                ]));
            }
            ListItem::new(lines)
        })
        .collect();
    let title = format!(
        " select model — type to filter: {}▏ ({} shown) ",
        app.picker_filter,
        models.len()
    );
    let preferred_height = (models
        .iter()
        .map(|model| {
            usize::from(model.provider.is_sub_agent() || model.provider == ProviderKind::OpenRouter)
                + 1
        })
        .sum::<usize>() as u16
        + 2)
    .clamp(5, 22);
    draw_overlay_list_sized(
        frame,
        &theme,
        title,
        items,
        app.picker_index.min(models.len().saturating_sub(1)),
        110,
        preferred_height,
    );
}

fn draw_permissions(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let root = frame.area();
    draw_modal_scrim(frame, &theme, root);
    let items = PERMISSION_PRESETS
        .iter()
        .map(|preset| {
            let current = app.policy.matching_preset() == Some(*preset);
            let marker = if current { "  ✓ current" } else { "" };
            ListItem::new(Line::from(vec![
                Span::styled(preset.label(), Style::new().fg(theme.secondary)),
                Span::styled(marker, Style::new().fg(theme.dim)),
            ]))
        })
        .collect::<Vec<_>>();
    let area = modal_area(frame.area(), 76, 13);
    frame.render_widget(Clear, area);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            " Permissions ",
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        ));
    if area.width >= 38 {
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
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let list_height = (PERMISSION_PRESETS.len() as u16).min(inner.height);
    let [list_area, detail_area] =
        Layout::vertical([Constraint::Length(list_height), Constraint::Min(0)]).areas(inner);
    let list = List::new(items)
        .highlight_style(selection_style(&theme))
        .highlight_symbol("▎ ");
    let mut state = ListState::default();
    state.select(Some(
        app.permission_index
            .min(PERMISSION_PRESETS.len().saturating_sub(1)),
    ));
    frame.render_stateful_widget(list, list_area, &mut state);

    if detail_area.height > 0 {
        let selected = PERMISSION_PRESETS[app
            .permission_index
            .min(PERMISSION_PRESETS.len().saturating_sub(1))];
        frame.render_widget(
            Paragraph::new(vec![
                Line::raw(""),
                Line::styled(
                    "DETAIL",
                    Style::new().fg(theme.dim).add_modifier(Modifier::BOLD),
                ),
                Line::styled(selected.description(), Style::new().fg(theme.secondary)),
            ])
            .wrap(Wrap { trim: true }),
            detail_area,
        );
    }
}

fn draw_full_access_confirmation(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let root = frame.area();
    draw_modal_scrim(frame, &theme, root);
    let warning = semantic_foreground(theme.warning, theme.surface, theme.fg);
    let area = modal_area(frame.area(), 76, 10);
    frame.render_widget(Clear, area);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            " Full Access ",
            Style::new().fg(theme.error).add_modifier(Modifier::BOLD),
        ));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let action_height = inner.height.min(2);
    let [body_area, action_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(action_height)]).areas(inner);
    frame.render_widget(
        Paragraph::new(
            "The agent can edit any file and use the network without asking. This can expose or delete data.",
        )
        .style(Style::new().fg(warning))
        .wrap(Wrap { trim: true }),
        body_area,
    );
    let safe = if app.full_access_enable_selected {
        "  "
    } else {
        "▎ "
    };
    let danger = if app.full_access_enable_selected {
        "▎ "
    } else {
        "  "
    };
    let safe_hint = if app.full_access_enable_selected {
        "  Esc"
    } else {
        "  Enter · Esc"
    };
    let danger_hint = if app.full_access_enable_selected {
        "  Enter"
    } else {
        ""
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(safe, Style::new().fg(theme.accent)),
                Span::styled(
                    "Go back",
                    Style::new()
                        .fg(theme.secondary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(safe_hint, Style::new().fg(theme.dim)),
            ]),
            Line::from(vec![
                Span::styled(danger, Style::new().fg(theme.error)),
                Span::styled(
                    "Enable full access",
                    Style::new().fg(theme.error).add_modifier(Modifier::BOLD),
                ),
                Span::styled(danger_hint, Style::new().fg(theme.dim)),
            ]),
        ]),
        action_area,
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
            if crate::app::session_is_foreign_at(s, app.policy.workspace().cwd()) {
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
    draw_overlay_list_sized(frame, theme, title, items, selected, 76, preferred_height);
}

fn draw_overlay_list_sized(
    frame: &mut Frame,
    theme: &Theme,
    title: String,
    items: Vec<ListItem>,
    selected: usize,
    preferred_width: u16,
    preferred_height: u16,
) {
    let root = frame.area();
    draw_modal_scrim(frame, theme, root);
    let area = modal_area(frame.area(), preferred_width, preferred_height);
    frame.render_widget(Clear, area);

    let empty = items.is_empty();
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border))
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
        .highlight_style(selection_style(theme))
        .highlight_symbol("▎ ");

    let mut state = ListState::default();
    state.select((!empty).then_some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_orchestration_confirm(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let root = frame.area();
    draw_modal_scrim(frame, &theme, root);
    let warning = semantic_foreground(theme.warning, theme.surface, theme.fg);
    let tasks = app.orchestration_plan();
    let workers = tasks.len();
    let focused = app.orchestration_confirm_focused;
    let uses_metered_api = app
        .orchestration_planner()
        .is_some_and(|model| model.provider.is_metered_api())
        || tasks
            .iter()
            .any(|task| task.model.provider.is_metered_api());
    let uses_codex = app
        .orchestration_planner()
        .is_some_and(|model| model.provider == ProviderKind::Codex)
        || tasks
            .iter()
            .any(|task| task.model.provider == ProviderKind::Codex);
    let uses_openrouter = app
        .orchestration_planner()
        .is_some_and(|model| model.provider == ProviderKind::OpenRouter)
        || tasks
            .iter()
            .any(|task| task.model.provider == ProviderKind::OpenRouter);
    // USED, START, RULE, and SHARE are always present. Each conditional risk
    // receives its own row so a long mixed-provider sentence cannot hide the
    // billing or data-boundary disclosure through horizontal truncation.
    let header_line_count = 4_u16
        + u16::from(uses_metered_api)
        + u16::from(uses_codex)
        + 2 * u16::from(uses_openrouter);
    let extra_header_lines = header_line_count.saturating_sub(3);
    let detailed_height = (workers as u16)
        .saturating_mul(3)
        .saturating_add(9)
        .saturating_add(extra_header_lines);
    let show_instructions =
        frame.area().width >= 64 && frame.area().height.saturating_sub(2) >= detailed_height;
    let preferred_height = if show_instructions {
        detailed_height
    } else {
        (workers as u16 + 8 + extra_header_lines).clamp(8, 20)
    };
    let area = modal_area(frame.area(), 92, preferred_height);
    frame.render_widget(Clear, area);

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border))
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
    // Preserve a heading and at least one exact-model row on very short
    // terminals; risk rows are priority ordered below for the remaining room.
    let minimum_task_height = u16::from(!tasks.is_empty()) * 2;
    let desired_header_height = header_line_count;
    let header_height = inner
        .height
        .saturating_sub(action_height.saturating_add(minimum_task_height))
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
    let value_width = inner.width.saturating_sub(7) as usize;
    let header_line = |label: &'static str, value: String, style: Style| {
        Line::from(vec![
            Span::styled(label, Style::new().fg(theme.dim)),
            Span::styled(truncate_width(&value, value_width), style),
        ])
    };
    let mut header_lines = vec![header_line(
        "USED   ",
        planner_line,
        Style::new().fg(theme.fg).add_modifier(Modifier::BOLD),
    )];
    if uses_metered_api {
        header_lines.push(header_line(
            "BILL   ",
            "metered API calls may bill".into(),
            Style::new().fg(theme.warning).add_modifier(Modifier::BOLD),
        ));
    }
    header_lines.push(header_line(
        "RULE   ",
        "workers are read-only".into(),
        Style::new().fg(theme.accent2),
    ));
    header_lines.push(header_line(
        "START  ",
        start_line,
        Style::new().fg(theme.warning),
    ));
    header_lines.push(header_line(
        "SHARE  ",
        "text sent to task-listed services".into(),
        Style::new().fg(theme.accent2),
    ));
    if uses_codex {
        header_lines.push(header_line(
            "CODEX  ",
            "global Codex rules excluded".into(),
            Style::new().fg(theme.accent2),
        ));
    }
    if uses_openrouter {
        header_lines.push(header_line(
            "ROUTE  ",
            "OpenRouter picks endpoints".into(),
            Style::new().fg(theme.accent2),
        ));
        header_lines.push(header_line(
            "PRIV   ",
            "privacy settings apply".into(),
            Style::new().fg(theme.accent2),
        ));
    }
    debug_assert_eq!(header_lines.len(), header_line_count as usize);
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
        let model_budget = row_width
            .saturating_sub(id.width() + 4)
            .min(if show_instructions { 32 } else { 22 });
        let model = compact_model_identity(
            task.model.display_id(),
            task.model.provider.label(),
            model_budget,
        );
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
    draw_modal_scrim(frame, &theme, modal_root);
    let area = modal_area(modal_root, 96, 24);
    frame.render_widget(Clear, area);
    let approval_title = if area.width < 56 {
        if approval_focused {
            " tool approval "
        } else {
            " approval · Tab to review "
        }
    } else if approval_focused {
        " review tool request · approval focus "
    } else if queue_occupied {
        " review tool request · press Tab "
    } else {
        " review tool request · composer focus "
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            approval_title,
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
    draw_modal_scrim(frame, &theme, frame.area());
    let preferred_height = if frame.area().width < 68 { 12 } else { 20 };
    let area = modal_area(frame.area(), 78, preferred_height);
    frame.render_widget(Clear, area);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::new().fg(theme.border))
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
                format!(" {label:<key_column$} "),
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
        let team = if inner.width >= 40 {
            key("/team [2-4]", "lead + read-only workers")
        } else {
            key("/team", "lead + workers")
        };
        let approval = if inner.width >= 40 {
            key("Tab · y/a/n", "approval choices")
        } else {
            key("Tab · y/a/n", "approve / deny")
        };
        vec![
            key("Enter", "send / queue next"),
            key("Alt+Enter", "newline"),
            team,
            key("/", "commands"),
            key("PgUp/PgDn", "scroll"),
            key("Esc", "cancel / deny"),
            approval,
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

fn draw_modal_scrim(frame: &mut Frame, _theme: &Theme, area: Rect) {
    let buffer = frame.buffer_mut();
    for y in area.y..area.bottom() {
        for x in area.x..area.right() {
            let cell = &mut buffer[(x, y)];
            if !cell.symbol().is_empty()
                && cell.symbol().chars().all(|symbol| {
                    matches!(symbol, '─' | '│' | '┌' | '┐' | '└' | '┘' | '━' | '╾' | '▎')
                })
            {
                cell.set_symbol(" ");
            }
        }
    }
    buffer.set_style(area, Style::new().add_modifier(Modifier::DIM));
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
            openrouter_api_key: None,
            openrouter_base_url: "http://127.0.0.1:9".into(),
            ollama_host: "http://127.0.0.1:9".into(),
            default_model: None,
            compact_threshold_chars: 80_000,
            ollama_num_ctx: 16_384,
            theme: None,
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

    fn has_inline_mascot(terminal: &Terminal<TestBackend>) -> bool {
        let buffer = terminal.backend().buffer();
        let transcript = Rect::new(
            0,
            2,
            buffer.area.width,
            buffer.area.height.saturating_sub(5),
        );
        let Some(region) = mascot_region(transcript) else {
            return false;
        };
        (region.y..region.bottom()).any(|y| {
            (region.x..region.right()).any(|x| buffer.cell((x, y)).is_some_and(is_mascot_cell))
        })
    }

    fn mascot_region(area: Rect) -> Option<Rect> {
        inline_mascot_area(
            area,
            Rect::new(
                0,
                0,
                crate::mascot::FRAME_WIDTH as u16,
                crate::mascot::FRAME_HEIGHT as u16,
            ),
        )
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
    async fn standard_working_view_uses_the_seal_instead_of_floating_art() {
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
        assert!(rendered.contains("SB"), "{rendered}");
        assert!(rendered.contains("SHALTAIBOLTAI"), "{rendered}");
        assert!(!rendered.contains("REAL AGENT"), "{rendered}");
        assert!(!rendered.contains("DANCING"), "{rendered}");
        assert!(!has_inline_mascot(&terminal), "{rendered}");
        assert!(
            rendered.contains("agent · running · team-test · ollama"),
            "{rendered}"
        );
        assert!(rendered.contains("read-only sandbox"), "{rendered}");
    }

    #[tokio::test]
    async fn inline_mascot_remains_compact_at_narrow_size() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("SB"), "{rendered}");
        assert!(!rendered.contains("╭⌒▾⌒╮"), "{rendered}");
        assert!(!has_inline_mascot(&terminal), "{rendered}");
        assert!(rendered.contains("TEAM · working"), "{rendered}");
    }

    #[test]
    fn compact_team_states_preserve_phase_and_safety_action() {
        assert_eq!(compact_team_state("TEAM · planning"), "TEAM · planning");
        assert_eq!(compact_team_state("TEAM · 2 workers armed"), "TEAM · armed");
        assert_eq!(
            compact_team_state("TEAM · plan ready · Tab to review"),
            "TEAM · ready"
        );
        assert_eq!(
            compact_team_state("TEAM · workers 1/3 · Esc cancel"),
            "TEAM · work · Esc"
        );
        assert_eq!(
            compact_team_state("TEAM · synthesizing · Esc cancel"),
            "TEAM · synth · Esc"
        );
        assert_eq!(
            compact_team_state("TEAM · applying changes · Esc cancel"),
            "TEAM · apply · Esc"
        );
        assert_eq!(
            compact_team_state("TEAM · applying · approval needed"),
            "TEAM · approval"
        );
    }

    #[tokio::test]
    async fn inline_native_mascot_yields_to_modals_and_compact_layouts() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Streaming;
        app.transcript.clear();
        app.transcript_rev += 1;
        let native = native_mascot();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));

        app.mode = Mode::Input;
        app.transcript = vec![Entry::Banner {
            title: "Ready to build".into(),
            subtitle: "Describe a change.".into(),
        }];
        app.transcript_rev += 1;
        terminal.backend_mut().resize(200, 70);
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(has_native_graphics(&terminal));

        app.mode = Mode::Approval;
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));

        app.mode = Mode::Help;
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));

        app.mode = Mode::Input;
        terminal.backend_mut().resize(60, 20);
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));
        assert!(screen(&terminal).contains("SB"));
    }

    #[tokio::test]
    async fn inline_mascot_never_covers_right_edge_transcript_text() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Streaming;
        app.transcript = vec![Entry::Assistant(format!(
            "{} RIGHT_EDGE_SENTINEL",
            "x".repeat(54)
        ))];
        app.transcript_rev += 1;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("RIGHT_EDGE_SENTINEL"), "{rendered}");
        assert!(!has_inline_mascot(&terminal), "{rendered}");

        let native = native_mascot();
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("RIGHT_EDGE_SENTINEL"), "{rendered}");
        assert!(!has_native_graphics(&terminal), "{rendered}");
    }

    #[tokio::test]
    async fn inline_mascot_yields_to_the_slash_menu_during_discovery() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.discovering = true;
        app.transcript.clear();
        app.transcript_rev += 1;
        app.textarea.insert_str("/");
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(
            rendered.contains("orchestrate the next prompt with read-only workers"),
            "{rendered}"
        );
        assert!(!has_inline_mascot(&terminal), "{rendered}");

        let native = native_mascot();
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        let rendered = screen(&terminal);
        assert!(
            rendered.contains("orchestrate the next prompt with read-only workers"),
            "{rendered}"
        );
        assert!(!has_native_graphics(&terminal), "{rendered}");
    }

    #[tokio::test]
    async fn only_the_large_idle_hero_keeps_static_mascot_art() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        let mut terminal = Terminal::new(TestBackend::new(200, 70)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(has_inline_mascot(&terminal));

        app.config.reduced_motion = true;
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let area = mascot_region(Rect::new(0, 2, 200, 65)).expect("mascot area");
        let before = buffer_region(&terminal, area);
        app.advance_animation();
        app.advance_animation();
        app.advance_animation();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(has_inline_mascot(&terminal));
        assert_eq!(buffer_region(&terminal, area), before);

        let native = native_mascot();
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(has_native_graphics(&terminal));
        assert_eq!(app.animation_tick(), 0);
        app.advance_animation();
        app.advance_animation();
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(has_native_graphics(&terminal));
        assert_eq!(app.animation_tick(), 0);

        app.mode = Mode::Streaming;
        terminal
            .draw(|frame| draw_with_native_mascot(frame, &mut app, &native))
            .unwrap();
        assert!(!has_native_graphics(&terminal));
    }

    #[tokio::test]
    async fn medium_terminal_keeps_conversation_space_and_uses_compact_signature() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered = screen(&terminal);
        assert!(rendered.contains("SB"), "{rendered}");
        assert!(!rendered.contains("╭⌒▾⌒╮"), "{rendered}");
        assert!(!has_inline_mascot(&terminal), "{rendered}");
        assert!(rendered.contains("Ready to build"), "{rendered}");
        assert!(rendered.contains("team working"), "{rendered}");
        assert!(rendered.contains("next message"), "{rendered}");
    }

    #[test]
    fn inline_mascot_region_is_bottom_right_and_conversation_stays_full_width() {
        let full = Rect::new(0, 0, 80, 20);
        let region = mascot_region(full).expect("fallback mascot fits");
        assert_eq!(region.width, crate::mascot::FRAME_WIDTH as u16);
        assert_eq!(region.height, crate::mascot::FRAME_HEIGHT as u16);
        assert_eq!(region.right(), full.right().saturating_sub(1));
        assert_eq!(region.bottom(), full.bottom().saturating_sub(1));
        assert!(
            region.x > 40,
            "mascot must sit in the right half of the pane"
        );

        let native = inline_mascot_area(full, Rect::new(0, 0, 28, 17)).expect("native mascot fits");
        assert_eq!(native, Rect::new(51, 2, 28, 17));
        assert!(inline_mascot_area(Rect::new(0, 0, 29, 19), Rect::new(0, 0, 28, 17)).is_none());
        assert_eq!(
            inline_mascot_area(Rect::new(0, 0, 30, 19), Rect::new(0, 0, 28, 17)),
            Some(Rect::new(1, 1, 28, 17))
        );
    }

    #[test]
    fn inline_mascot_clearance_rejects_content_surfaces_placeholders_and_wide_glyphs() {
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal
            .draw(|frame| {
                let area = Rect::new(10, 2, 5, 3);
                assert!(mascot_region_is_clear(frame, area, None));

                frame
                    .buffer_mut()
                    .cell_mut((10, 2))
                    .expect("content cell")
                    .set_symbol("x");
                assert!(!mascot_region_is_clear(frame, area, None));
                frame
                    .buffer_mut()
                    .cell_mut((10, 2))
                    .expect("content cell")
                    .reset();

                frame
                    .buffer_mut()
                    .cell_mut((10, 2))
                    .expect("surface cell")
                    .set_bg(theme::DEFAULT.surface.expect("surface theme"));
                assert!(!mascot_region_is_clear(frame, area, None));
                frame
                    .buffer_mut()
                    .cell_mut((10, 2))
                    .expect("surface cell")
                    .reset();

                frame
                    .buffer_mut()
                    .cell_mut((10, 2))
                    .expect("placeholder cell")
                    .skip = true;
                assert!(!mascot_region_is_clear(frame, area, None));
                frame
                    .buffer_mut()
                    .cell_mut((10, 2))
                    .expect("placeholder cell")
                    .reset();

                frame
                    .buffer_mut()
                    .cell_mut((9, 2))
                    .expect("left guard cell")
                    .set_symbol("界");
                assert!(!mascot_region_is_clear(frame, area, None));
            })
            .unwrap();
    }

    #[tokio::test]
    async fn working_animation_keeps_large_art_hidden_and_cache_stable() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.mode = Mode::Orchestrating;
        app.transcript = (0..30)
            .map(|index| Entry::Info(format!("evidence line {index}")))
            .collect();
        app.transcript_rev += 1;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(!has_inline_mascot(&terminal));
        app.scroll_from_bottom = 7;
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let cache_len = app.render_cache.len();
        let cache_starts = app.render_cache_starts.clone();
        let cache_total = app.render_cache_total_lines;
        let cache_rev = app.render_cache_rev;
        let scroll = app.scroll_from_bottom;

        app.advance_animation();
        app.advance_animation();
        app.advance_animation();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(!has_inline_mascot(&terminal));
        assert_eq!(app.render_cache.len(), cache_len);
        assert_eq!(app.render_cache_starts, cache_starts);
        assert_eq!(app.render_cache_total_lines, cache_total);
        assert_eq!(app.render_cache_rev, cache_rev);
        assert_eq!(app.scroll_from_bottom, scroll);
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
            let mut terminal = Terminal::new(TestBackend::new(200, 70)).unwrap();
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            let buffer = terminal.backend().buffer();
            let mascot_cells = mascot_region(Rect::new(0, 2, 200, 65))
                .expect("mascot area")
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
                assert_eq!(buffer[(1, 3)].bg, Color::Reset);
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
        assert!(rendered.contains("task-listed services"), "{rendered}");
        assert!(rendered.contains("workers are read-only"), "{rendered}");
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
        assert!(rendered.contains("workers are read-only"), "{rendered}");
        assert!(rendered.contains("text sent to task-listed"), "{rendered}");
        assert!(rendered.contains("TASKS · EXACT MODELS"), "{rendered}");
        assert!(rendered.contains("team-test · ollama"), "{rendered}");
        assert!(rendered.contains("Tab"), "{rendered}");
        assert!(rendered.contains("n / Esc"), "{rendered}");
    }

    #[tokio::test]
    async fn mixed_team_confirmation_keeps_cost_and_routing_boundaries_visible() {
        let _data_dir_guard = session::TEST_DATA_DIR_ENV_LOCK.lock().await;
        let mut app = test_app();
        app.textarea.insert_str("/team 2");
        app.submit_input();
        app.textarea
            .insert_str("coordinate this mixed-provider change");
        app.submit_input();
        let run_id = app.orchestration_run_id().expect("orchestration run");
        app.on_event(AppEvent::OrchestrationPlanned {
            run_id,
            result: Ok(vec![
                PlannedTask {
                    id: 1,
                    title: "inspect with Codex".into(),
                    instructions: "read the relevant code".into(),
                    model: ModelEntry {
                        provider: ProviderKind::Codex,
                        id: "codex:gpt-5.6-sol".into(),
                    },
                },
                PlannedTask {
                    id: 2,
                    title: "review independently".into(),
                    instructions: "check the provider boundary".into(),
                    model: ModelEntry {
                        provider: ProviderKind::OpenRouter,
                        id: "openai/gpt-5.4".into(),
                    },
                },
            ]),
        });

        for width in [40, 80, 120] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            let rendered = screen(&terminal);
            assert!(
                rendered.contains("metered API calls may bill"),
                "{rendered}"
            );
            assert!(rendered.contains("workers are read-only"), "{rendered}");
            assert!(
                rendered.contains("global Codex rules excluded"),
                "{rendered}"
            );
            assert!(
                rendered.contains("OpenRouter picks endpoints"),
                "{rendered}"
            );
            assert!(rendered.contains("privacy settings apply"), "{rendered}");
            assert!(rendered.contains("EXACT MODELS"), "{rendered}");
            assert!(rendered.contains("codex"), "{rendered}");
            assert!(rendered.contains("openrouter"), "{rendered}");
        }
    }
}
