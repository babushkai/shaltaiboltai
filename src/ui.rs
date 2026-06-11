use crate::app::{App, Entry, Mode};
use crate::markdown;
use crate::session;
use crate::tools;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};
use ratatui::Frame;

const TOOL_RESULT_PREVIEW_LINES: usize = 6;
const MAX_INPUT_LINES: u16 = 8;

pub fn draw(frame: &mut Frame, app: &App) {
    let input_height = (app.textarea.lines().len() as u16).clamp(1, MAX_INPUT_LINES) + 2;
    let [transcript_area, status_area, input_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(input_height),
    ])
    .areas(frame.area());

    draw_transcript(frame, app, transcript_area);
    draw_status(frame, app, status_area);
    frame.render_widget(&app.textarea, input_area);

    match app.mode {
        Mode::ModelPicker => draw_model_picker(frame, app),
        Mode::SessionPicker => draw_session_picker(frame, app),
        Mode::Approval => draw_approval(frame, app),
        _ => {}
    }
}

fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    let width = area.width.saturating_sub(2).max(10) as usize;
    let mut lines: Vec<Line> = Vec::new();

    for entry in &app.transcript {
        if !lines.is_empty() {
            lines.push(Line::raw(""));
        }
        match entry {
            Entry::User(text) => {
                push_wrapped(
                    &mut lines,
                    "you ❯ ",
                    text,
                    width,
                    Style::new().fg(Color::Cyan).bold(),
                );
            }
            Entry::Assistant(text) => {
                if text.is_empty() && app.mode == Mode::Streaming {
                    lines.push(Line::styled("…", Style::new().fg(Color::DarkGray)));
                } else {
                    lines.extend(markdown::render(text, width, Style::new().fg(Color::White)));
                }
            }
            Entry::Tool {
                summary,
                result,
                is_error,
            } => {
                let style = if *is_error {
                    Style::new().fg(Color::Red)
                } else {
                    Style::new().fg(Color::Yellow)
                };
                push_wrapped(&mut lines, "⚒ ", summary, width, style);
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
                        "  │ ",
                        &text,
                        width,
                        Style::new().fg(Color::DarkGray),
                    );
                }
            }
            Entry::Info(text) => {
                push_wrapped(
                    &mut lines,
                    "• ",
                    text,
                    width,
                    Style::new().fg(Color::DarkGray).italic(),
                );
            }
            Entry::Error(text) => {
                push_wrapped(&mut lines, "✗ ", text, width, Style::new().fg(Color::Red));
            }
        }
    }

    let visible = area.height.saturating_sub(2) as usize;
    let start = lines
        .len()
        .saturating_sub(visible + app.scroll_from_bottom)
        .min(lines.len());
    let window: Vec<Line> = lines.into_iter().skip(start).collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" shaltaiboltai ")
        .border_style(Style::new().fg(Color::DarkGray));
    frame.render_widget(Paragraph::new(window).block(block), area);
}

/// Wrap `text` to `width` and append, applying `style` and putting `prefix`
/// on the first line with matching indentation on continuations.
fn push_wrapped(lines: &mut Vec<Line>, prefix: &str, text: &str, width: usize, style: Style) {
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
                Span::styled(lead, style.add_modifier(Modifier::DIM)),
                Span::styled(part.into_owned(), style),
            ]));
        }
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let model = app
        .model
        .as_ref()
        .map(|m| format!("{} ({})", m.id, m.provider.label()))
        .unwrap_or_else(|| "no model".into());
    let state = if app.compacting {
        "compacting context…"
    } else {
        match app.mode {
            Mode::Input => "ready",
            Mode::Streaming => "thinking… (Esc to cancel)",
            Mode::RunningTool => "running tool…",
            Mode::Approval => "awaiting approval",
            Mode::ModelPicker => "selecting model",
            Mode::SessionPicker => "selecting session",
        }
    };
    let mut spans = vec![
        Span::styled(
            format!(" {model} "),
            Style::new().fg(Color::Black).bg(Color::Cyan),
        ),
        Span::styled(format!(" {state}"), Style::new().fg(Color::DarkGray)),
    ];
    let tokens = app.approx_tokens();
    if tokens > 0 {
        spans.push(Span::styled(
            format!(" · ~{tokens} ctx tokens"),
            Style::new().fg(Color::DarkGray),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_model_picker(frame: &mut Frame, app: &App) {
    let models = app.filtered_models();
    let items: Vec<ListItem> = models
        .iter()
        .map(|m| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<10}", m.provider.label()),
                    Style::new().fg(Color::Magenta),
                ),
                Span::raw(m.id.clone()),
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
        title,
        items,
        app.picker_index.min(models.len().saturating_sub(1)),
    );
}

fn draw_session_picker(frame: &mut Frame, app: &App) {
    let items: Vec<ListItem> = app
        .sessions
        .iter()
        .map(|s| {
            ListItem::new(Line::from(vec![
                Span::raw(s.title.clone()),
                Span::styled(
                    format!("  ·  {}", session::ago(s.updated_at)),
                    Style::new().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();
    let title = format!(" resume session ({}) ", app.sessions.len());
    draw_overlay_list(frame, title, items, app.session_index);
}

fn draw_overlay_list(frame: &mut Frame, title: String, items: Vec<ListItem>, selected: usize) {
    let area = centered(frame.area(), 60, 70);
    frame.render_widget(Clear, area);

    let empty = items.is_empty();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(Style::new().bg(Color::Cyan).fg(Color::Black))
        .highlight_symbol("❯ ");

    let mut state = ListState::default();
    state.select((!empty).then_some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_approval(frame: &mut Frame, app: &App) {
    let Some(call) = app.pending_approval() else {
        return;
    };
    let area = centered(frame.area(), 70, 30);
    frame.render_widget(Clear, area);

    let mut lines = vec![
        Line::styled(tools::describe(call), Style::new().fg(Color::Yellow).bold()),
        Line::raw(""),
    ];
    if let Ok(pretty) = serde_json::to_string_pretty(&call.arguments) {
        for l in pretty.lines().take(8) {
            lines.push(Line::styled(l.to_owned(), Style::new().fg(Color::DarkGray)));
        }
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled("[y]", Style::new().fg(Color::Green).bold()),
        Span::raw(" approve  "),
        Span::styled("[a]", Style::new().fg(Color::Green).bold()),
        Span::raw(" approve all  "),
        Span::styled("[n]", Style::new().fg(Color::Red).bold()),
        Span::raw(" deny"),
    ]));

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" tool approval ")
        .border_style(Style::new().fg(Color::Yellow));
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
