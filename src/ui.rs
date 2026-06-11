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
    ScrollbarOrientation, ScrollbarState,
};
use ratatui::Frame;

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

    let input_height = (app.textarea.lines().len() as u16).clamp(1, MAX_INPUT_LINES) + 2;
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
        _ => {}
    }
}

/// The input renders as an elevated card; its border doubles as the focus
/// indicator — accent while typing is possible, structural otherwise.
fn draw_input(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    let focused = app.mode == Mode::Input && !app.compacting;
    let border = if focused { theme.accent } else { theme.border };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    app.textarea.set_block(block);
    frame.render_widget(&app.textarea, area);
}

/// Renders the transcript through a per-entry line cache: only new entries
/// and the final (possibly streaming) entry are re-rendered each frame, so
/// cost stays constant as the conversation grows.
fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let theme = app.theme;
    // Borders (2) + horizontal padding (2).
    let width = area.width.saturating_sub(4).max(10) as usize;

    if app.render_cache_width != width || app.render_cache_rev != app.transcript_rev {
        app.render_cache.clear();
        app.render_cache_width = width;
        app.render_cache_rev = app.transcript_rev;
    }
    if app.render_cache.len() > app.transcript.len() {
        app.render_cache.clear();
    }
    let streaming = app.mode == Mode::Streaming;
    while app.render_cache.len() < app.transcript.len() {
        let i = app.render_cache.len();
        let last = i + 1 == app.transcript.len();
        app.render_cache.push(render_entry(
            &app.transcript[i],
            width,
            last && streaming,
            &theme,
        ));
    }
    if let Some(last) = app.transcript.last() {
        let i = app.transcript.len() - 1;
        app.render_cache[i] = render_entry(last, width, streaming, &theme);
    }

    let total: usize = app.render_cache.iter().map(Vec::len).sum::<usize>()
        + app.render_cache.len().saturating_sub(1);
    let visible = area.height.saturating_sub(2) as usize;
    app.scroll_from_bottom = app.scroll_from_bottom.min(total.saturating_sub(visible));
    let start = total.saturating_sub(visible + app.scroll_from_bottom);
    let end = (start + visible).min(total);

    let mut window: Vec<Line> = Vec::with_capacity(end.saturating_sub(start));
    let mut pos = 0;
    'outer: for (i, lines) in app.render_cache.iter().enumerate() {
        if i > 0 {
            if pos >= start && pos < end {
                window.push(Line::raw(""));
            }
            pos += 1;
            if pos >= end {
                break;
            }
        }
        for line in lines {
            if pos >= start && pos < end {
                window.push(line.clone());
            }
            pos += 1;
            if pos >= end {
                break 'outer;
            }
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.border))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            " ◆ shaltaiboltai ",
            Style::new().fg(theme.accent).add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(Paragraph::new(window).block(block), area);

    if total > visible {
        let mut state = ScrollbarState::new(total - visible).position(start);
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
            push_wrapped(
                &mut lines,
                "▌ ",
                Style::new().fg(theme.accent),
                text,
                width,
                Style::new().fg(theme.fg).add_modifier(Modifier::BOLD),
            );
        }
        Entry::Assistant(text) => {
            if text.is_empty() && streaming {
                lines.push(Line::styled("…", Style::new().fg(theme.dim)));
            } else {
                lines.extend(markdown::render(text, width, theme));
            }
        }
        Entry::Tool {
            summary,
            result,
            is_error,
        } => {
            let (glyph, color) = if *is_error {
                ("✗ ", theme.error)
            } else {
                ("✓ ", theme.success)
            };
            push_wrapped(
                &mut lines,
                glyph,
                Style::new().fg(color),
                summary,
                width,
                Style::new().fg(theme.fg),
            );
            for (i, line) in result.lines().take(TOOL_RESULT_PREVIEW_LINES).enumerate() {
                let truncated = result.lines().count() > TOOL_RESULT_PREVIEW_LINES
                    && i == TOOL_RESULT_PREVIEW_LINES - 1;
                let text = if truncated {
                    format!("{line} …")
                } else {
                    line.to_owned()
                };
                push_wrapped(
                    &mut lines,
                    "  ▏ ",
                    Style::new().fg(theme.border),
                    &text,
                    width,
                    Style::new().fg(theme.dim),
                );
            }
        }
        Entry::Info(text) => {
            push_wrapped(
                &mut lines,
                "• ",
                Style::new().fg(theme.dim),
                text,
                width,
                Style::new().fg(theme.dim).add_modifier(Modifier::ITALIC),
            );
        }
        Entry::Error(text) => {
            push_wrapped(
                &mut lines,
                "✗ ",
                Style::new().fg(theme.error),
                text,
                width,
                Style::new().fg(theme.error),
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
    let chip_fg = theme.bg.unwrap_or(Color::Black);
    let model = app
        .model
        .as_ref()
        .map(|m| format!("{} · {}", m.id, m.provider.label()))
        .unwrap_or_else(|| "no model".into());

    let mut spans = vec![Span::styled(
        format!(" ◆ {model} "),
        Style::new().fg(chip_fg).bg(theme.accent),
    )];
    if app.is_busy() {
        spans.push(Span::styled(
            format!(" {} ", spinner_frame()),
            Style::new().fg(theme.accent),
        ));
    } else {
        spans.push(Span::raw(" "));
    }
    let state = if app.compacting {
        "compacting context…"
    } else {
        match app.mode {
            Mode::Input => "ready",
            Mode::Streaming => "thinking — Esc to cancel",
            Mode::RunningTool => "running tool — Esc to cancel",
            Mode::Approval => "awaiting approval",
            Mode::ModelPicker => "selecting model",
            Mode::SessionPicker => "selecting session",
            Mode::ThemePicker => "selecting theme — Enter keep · Esc revert",
        }
    };
    spans.push(Span::styled(state, Style::new().fg(theme.dim)));
    let left_width: usize = spans.iter().map(|s| s.width()).sum();
    frame.render_widget(Paragraph::new(Line::from(spans)), area);

    // Right side: cwd · branch · context usage, Claude Code style. On narrow
    // terminals, pieces are dropped (cwd first, then branch) instead of
    // colliding with the left side.
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
        let context = match app.last_usage {
            Some(u) => Some(format!(
                "ctx {} · out {}",
                fmt_count(u.input_tokens as usize),
                fmt_count(u.output_tokens as usize)
            )),
            None if app.approx_tokens() > 0 => {
                Some(format!("ctx ~{}", fmt_count(app.approx_tokens())))
            }
            None => None,
        };
        if let Some(ctx) = context {
            if !right.is_empty() {
                right.push(sep());
            }
            right.push(Span::styled(ctx, Style::new().fg(theme.dim)));
            if let Some(pct) = app.context_percent() {
                let color = match pct {
                    0..=69 => theme.dim,
                    70..=89 => theme.warning,
                    _ => theme.error,
                };
                right.push(Span::styled(format!(" {pct}%"), Style::new().fg(color)));
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
    let right = [
        assemble(true, true),
        assemble(false, true),
        assemble(false, false),
    ]
    .into_iter()
    .find(|candidate| fits(candidate));
    if let Some(right) = right {
        frame.render_widget(
            Paragraph::new(Line::from(right)).alignment(Alignment::Right),
            area,
        );
    }
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
    let height = (matches.len() as u16).min(8) + 2;
    let width = matches
        .iter()
        .map(|c| 4 + label_width + c.description.len())
        .max()
        .unwrap_or(20)
        .min(frame.area().width.saturating_sub(4) as usize) as u16;
    let area = Rect {
        x: input_area.x + 1,
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
        .highlight_style(
            Style::new()
                .bg(theme.accent)
                .fg(theme.bg.unwrap_or(Color::Black)),
        )
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
    let area = centered(frame.area(), 60, 70);
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
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::new()
                .bg(theme.accent)
                .fg(theme.bg.unwrap_or(Color::Black)),
        )
        .highlight_symbol("❯ ");

    let mut state = ListState::default();
    state.select((!empty).then_some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_approval(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let Some(call) = app.pending_approval() else {
        return;
    };
    let area = centered(frame.area(), 80, 70);
    frame.render_widget(Clear, area);
    let inner_width = area.width.saturating_sub(4) as usize;

    let mut lines = vec![Line::styled(
        tools::describe(call),
        Style::new().fg(theme.warning).add_modifier(Modifier::BOLD),
    )];
    lines.push(Line::raw(""));

    match &app.approval_preview {
        Some(diff) => {
            for (tag, text) in diff {
                let (style, prefix) = match tag {
                    '+' => (Style::new().fg(theme.success), "+"),
                    '-' => (Style::new().fg(theme.error), "-"),
                    '@' => (Style::new().fg(theme.accent2), "@"),
                    '!' => (
                        Style::new().fg(theme.error).add_modifier(Modifier::BOLD),
                        "!",
                    ),
                    _ => (Style::new().fg(theme.dim), " "),
                };
                let mut shown = format!("{prefix} {text}");
                shown.truncate(
                    shown
                        .char_indices()
                        .nth(inner_width)
                        .map_or(shown.len(), |(i, _)| i),
                );
                lines.push(Line::styled(shown, style));
            }
        }
        None => {
            if let Ok(pretty) = serde_json::to_string_pretty(&call.arguments) {
                for l in pretty.lines().take(12) {
                    lines.push(Line::styled(l.to_owned(), Style::new().fg(theme.dim)));
                }
            }
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled(
            "[y]",
            Style::new().fg(theme.success).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" approve  ", Style::new().fg(theme.fg)),
        Span::styled(
            "[a]",
            Style::new().fg(theme.success).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" always allow {}  ", call.name),
            Style::new().fg(theme.fg),
        ),
        Span::styled(
            "[n]",
            Style::new().fg(theme.error).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" deny", Style::new().fg(theme.fg)),
    ]));

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.warning))
        .padding(Padding::horizontal(1))
        .title(Line::styled(
            " tool approval ",
            Style::new().fg(theme.warning).add_modifier(Modifier::BOLD),
        ));
    if let Some(surface) = theme.surface {
        block = block.style(Style::new().bg(surface).fg(theme.fg));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn centered(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let [_, mid, _] = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .areas(area);
    let [_, mid, _] = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .areas(mid);
    mid
}
