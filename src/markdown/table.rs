use std::borrow::Cow;

use crate::theme::Theme;
use pulldown_cmark::Alignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const COLUMN_GAP: usize = 2;
const CELL_PADDING: usize = 1;
const MIN_CELL_CONTENT_WIDTH: usize = 3;
const PREFERRED_EXPANSIVE_WIDTH: usize = 16;
const MIN_ALIGNED_COMPACT_VALUE_WIDTH: usize = 12;
const MIN_ALIGNED_EXPANSIVE_VALUE_WIDTH: usize = 24;
const MIN_SCANNABLE_EXPANSIVE_WIDTH: usize = 12;
const CRAMPED_EXPANSIVE_CELL_LINES: usize = 4;
const CATASTROPHIC_NARRATIVE_CELL_LINES: usize = 7;
const RECORD_VALUE_INDENT: usize = 2;

#[derive(Clone, Debug, Default)]
struct TableCell {
    lines: Vec<Line<'static>>,
}

impl TableCell {
    fn is_blank(&self) -> bool {
        self.lines
            .iter()
            .all(|line| line.spans.iter().all(|span| span.content.trim().is_empty()))
    }

    fn plain_text(&self) -> String {
        self.lines
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn widest_line(&self) -> usize {
        self.lines.iter().map(Line::width).max().unwrap_or(0)
    }

    fn longest_token(&self) -> usize {
        self.lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .flat_map(|span| span.content.split_whitespace())
            .map(UnicodeWidthStr::width)
            .max()
            .unwrap_or(0)
    }
}

/// Parser-side table accumulator. Markdown inline styles stay attached to
/// spans while the layout engine decides whether the table remains a grid or
/// becomes a narrow key/value list.
pub(super) struct TableState {
    alignments: Vec<Alignment>,
    header: Vec<TableCell>,
    rows: Vec<TableBodyRow>,
    current_row: Vec<TableCell>,
    current_row_has_boundary_pipe: bool,
    current_cell_lines: Vec<Line<'static>>,
    current_cell_spans: Vec<Span<'static>>,
    in_header: bool,
    row_open: bool,
    cell_open: bool,
}

impl TableState {
    pub(super) fn new(alignments: Vec<Alignment>) -> Self {
        Self {
            alignments,
            header: Vec::new(),
            rows: Vec::new(),
            current_row: Vec::new(),
            current_row_has_boundary_pipe: false,
            current_cell_lines: Vec::new(),
            current_cell_spans: Vec::new(),
            in_header: false,
            row_open: false,
            cell_open: false,
        }
    }

    pub(super) fn start_head(&mut self) {
        self.finish_row();
        self.in_header = true;
        self.row_open = true;
    }

    pub(super) fn end_head(&mut self) {
        self.finish_row();
        self.in_header = false;
    }

    pub(super) fn start_row(&mut self, has_boundary_pipe: bool) {
        self.finish_row();
        self.row_open = true;
        self.current_row_has_boundary_pipe = has_boundary_pipe;
    }

    pub(super) fn end_row(&mut self) {
        self.finish_row();
    }

    pub(super) fn start_cell(&mut self) {
        self.finish_cell();
        if !self.row_open {
            self.row_open = true;
        }
        self.cell_open = true;
    }

    pub(super) fn end_cell(&mut self) {
        self.finish_cell();
    }

    pub(super) fn push_span(&mut self, span: Span<'static>) {
        if !self.cell_open {
            return;
        }
        push_or_merge_span(&mut self.current_cell_spans, span);
    }

    pub(super) fn hard_break(&mut self) {
        if !self.cell_open {
            return;
        }
        self.flush_cell_line();
    }

    fn flush_cell_line(&mut self) {
        let spans = trim_spans(std::mem::take(&mut self.current_cell_spans));
        self.current_cell_lines.push(Line::from(spans));
    }

    fn finish_cell(&mut self) {
        if !self.cell_open {
            return;
        }
        self.flush_cell_line();
        while self
            .current_cell_lines
            .last()
            .is_some_and(|line| line.spans.is_empty())
            && self.current_cell_lines.len() > 1
        {
            self.current_cell_lines.pop();
        }
        self.current_row.push(TableCell {
            lines: std::mem::take(&mut self.current_cell_lines),
        });
        self.cell_open = false;
    }

    fn finish_row(&mut self) {
        self.finish_cell();
        if !self.row_open {
            return;
        }
        let row = std::mem::take(&mut self.current_row);
        if self.in_header && self.header.is_empty() {
            self.header = row;
        } else {
            self.rows.push(TableBodyRow {
                cells: row,
                has_boundary_pipe: self.current_row_has_boundary_pipe,
            });
        }
        self.current_row_has_boundary_pipe = false;
        self.row_open = false;
    }

    fn finish(mut self) -> Table {
        self.finish_row();
        let column_count = self.alignments.len();
        let mut rows = Vec::with_capacity(self.rows.len());
        let mut spillover = Vec::new();
        for mut row in self.rows {
            let only_first_cell_has_content =
                row.cells.first().is_some_and(|cell| !cell.is_blank())
                    && row.cells.iter().skip(1).all(TableCell::is_blank);
            if column_count > 1 && !row.has_boundary_pipe && only_first_cell_has_content {
                if !row.cells.is_empty() {
                    let cell = row.cells.remove(0);
                    spillover.push((rows.len(), cell));
                }
            } else {
                rows.push(row.cells);
            }
        }
        Table {
            alignments: self.alignments,
            header: self.header,
            rows,
            spillover,
        }
    }
}

struct Table {
    alignments: Vec<Alignment>,
    header: Vec<TableCell>,
    rows: Vec<Vec<TableCell>>,
    /// Prose rows rejected by the table heuristic, paired with the number of
    /// real rows that preceded them in the source.
    spillover: Vec<(usize, TableCell)>,
}

struct TableBodyRow {
    cells: Vec<TableCell>,
    has_boundary_pipe: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnKind {
    Compact,
    Narrative,
    TokenHeavy,
}

#[derive(Clone, Copy, Debug)]
struct ColumnMetrics {
    max_width: usize,
    header_token_width: usize,
    body_token_width: usize,
    kind: ColumnKind,
}

pub(super) fn render(
    state: TableState,
    width: usize,
    first_prefix: &str,
    cont_prefix: &str,
    prefix_style: Style,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut table = state.finish();
    // The delimiter row is the table schema. Lenient parsers can surface a
    // later prose line as an over-wide row; it must not silently add columns.
    let column_count = table.alignments.len();
    if column_count == 0 {
        return Vec::new();
    }

    table.alignments.resize(column_count, Alignment::None);
    normalize_row(&mut table.header, column_count);
    for row in &mut table.rows {
        normalize_row(row, column_count);
    }

    let line_width = width.max(1);
    // Keep one display cell available for content even when deeply nested
    // quote/list gutters would otherwise consume the entire terminal row.
    let prefix_limit = line_width.saturating_sub(1);
    let first_prefix = fit_prefix(first_prefix, prefix_limit);
    let cont_prefix = fit_prefix(cont_prefix, prefix_limit);
    let prefix_width = UnicodeWidthStr::width(first_prefix.as_ref())
        .max(UnicodeWidthStr::width(cont_prefix.as_ref()));
    let available = line_width.saturating_sub(prefix_width).max(1);
    let metrics = collect_metrics(&table, column_count);
    let reserved = column_count
        .saturating_mul(CELL_PADDING * 2)
        .saturating_add(column_count.saturating_sub(1).saturating_mul(COLUMN_GAP));
    let content_budget = available.saturating_sub(reserved);
    let column_widths = allocate_column_widths(&metrics, content_budget);
    let header_style = Style::new().fg(theme.accent2).add_modifier(Modifier::BOLD);
    let separator_style = Style::new().fg(theme.border).add_modifier(Modifier::DIM);

    let mut lines = if let Some(column_widths) = column_widths {
        if should_render_records(&table.rows, &column_widths, &metrics) {
            render_records(
                &table.header,
                &table.rows,
                &table.spillover,
                &metrics,
                available,
                header_style,
                separator_style,
            )
        } else {
            render_grid(
                &table.header,
                &table.rows,
                &table.spillover,
                &table.alignments,
                &column_widths,
                available,
                header_style,
                separator_style,
                theme.fg,
            )
        }
    } else if table.rows.is_empty() {
        let mut fallback =
            render_header_fallback(&table.header, &table.alignments, available, header_style);
        let mut spillover_cursor = 0;
        append_spillovers_at(
            &mut fallback,
            &table.spillover,
            &mut spillover_cursor,
            0,
            available,
        );
        fallback
    } else {
        render_records(
            &table.header,
            &table.rows,
            &table.spillover,
            &metrics,
            available,
            header_style,
            separator_style,
        )
    };
    if first_prefix.is_empty() && cont_prefix.is_empty() {
        return lines;
    }
    for (index, line) in lines.iter_mut().enumerate() {
        let prefix = if index == 0 {
            first_prefix.as_ref()
        } else {
            cont_prefix.as_ref()
        };
        let mut spans = Vec::with_capacity(line.spans.len() + 1);
        spans.push(Span::styled(prefix.to_owned(), prefix_style));
        spans.append(&mut line.spans);
        *line = Line::from(spans);
    }
    lines
}

fn normalize_row(row: &mut Vec<TableCell>, column_count: usize) {
    row.truncate(column_count);
    row.resize(column_count, TableCell::default());
}

fn collect_metrics(table: &Table, column_count: usize) -> Vec<ColumnMetrics> {
    (0..column_count)
        .map(|column| {
            let header = &table.header[column];
            let mut widest = header.widest_line();
            let header_token_width = header.longest_token();
            let mut body_token_width = 0usize;
            let mut body_token_count = 0usize;
            let mut long_body_token_count = 0usize;
            let mut total_words = 0usize;
            let mut total_cell_width = 0usize;
            let mut populated_cells = 0usize;
            for row in &table.rows {
                let cell = &row[column];
                widest = widest.max(cell.widest_line());
                let plain = cell.plain_text();
                let mut words = 0usize;
                for token in plain.split_whitespace() {
                    let token_width = UnicodeWidthStr::width(token);
                    body_token_width = body_token_width.max(token_width);
                    body_token_count += 1;
                    words += 1;
                    long_body_token_count += usize::from(token_width >= 20);
                }
                if words > 0 {
                    total_words += words;
                    total_cell_width =
                        total_cell_width.saturating_add(UnicodeWidthStr::width(plain.as_str()));
                    populated_cells += 1;
                }
            }
            let average_words = if populated_cells == 0 {
                header.plain_text().split_whitespace().count() as f64
            } else {
                total_words as f64 / populated_cells as f64
            };
            let average_width = if populated_cells == 0 {
                header.widest_line() as f64
            } else {
                total_cell_width as f64 / populated_cells as f64
            };
            let kind = if long_body_token_count > 0
                && long_body_token_count >= body_token_count.saturating_sub(long_body_token_count)
            {
                ColumnKind::TokenHeavy
            } else if average_words >= 4.0 || average_width >= 28.0 {
                ColumnKind::Narrative
            } else {
                ColumnKind::Compact
            };
            ColumnMetrics {
                max_width: widest,
                header_token_width,
                body_token_width,
                kind,
            }
        })
        .collect()
}

fn allocate_column_widths(metrics: &[ColumnMetrics], available: usize) -> Option<Vec<usize>> {
    if available < MIN_CELL_CONTENT_WIDTH.saturating_mul(metrics.len()) {
        return None;
    }
    let mut widths = metrics
        .iter()
        .map(|metric| metric.max_width.max(MIN_CELL_CONTENT_WIDTH))
        .collect::<Vec<_>>();

    let mut floors = metrics
        .iter()
        .map(preferred_column_floor)
        .collect::<Vec<_>>();
    let floor_total = floors.iter().sum::<usize>();
    if floor_total > available {
        let hard_floors = vec![MIN_CELL_CONTENT_WIDTH; floors.len()];
        let remaining = shrink_columns(&mut floors, &hard_floors, metrics, floor_total - available);
        if remaining > 0 {
            return None;
        }
    }

    let total = widths.iter().sum::<usize>();
    if total > available {
        let remaining = shrink_columns(&mut widths, &floors, metrics, total - available);
        if remaining > 0 {
            return None;
        }
    }
    Some(widths)
}

fn preferred_column_floor(metric: &ColumnMetrics) -> usize {
    let target = match metric.kind {
        ColumnKind::Narrative | ColumnKind::TokenHeavy => PREFERRED_EXPANSIVE_WIDTH,
        ColumnKind::Compact => metric
            .header_token_width
            .max(metric.body_token_width.min(PREFERRED_EXPANSIVE_WIDTH)),
    };
    target
        .max(MIN_CELL_CONTENT_WIDTH)
        .min(metric.max_width.max(MIN_CELL_CONTENT_WIDTH))
}

/// Bulk water-filling: within one priority class, columns with the widest
/// slack are leveled first. Runtime is O(columns * log(max_slack)), independent
/// of the number of display cells removed.
fn shrink_columns(
    widths: &mut [usize],
    floors: &[usize],
    metrics: &[ColumnMetrics],
    mut amount: usize,
) -> usize {
    for kind in [
        ColumnKind::TokenHeavy,
        ColumnKind::Narrative,
        ColumnKind::Compact,
    ] {
        let capacity = widths
            .iter()
            .enumerate()
            .filter(|(index, _)| metrics[*index].kind == kind)
            .map(|(index, width)| width.saturating_sub(floors[index]))
            .sum::<usize>();
        let target = amount.min(capacity);
        if target == 0 {
            continue;
        }

        let mut low = 0usize;
        let mut high = widths
            .iter()
            .enumerate()
            .filter(|(index, _)| metrics[*index].kind == kind)
            .map(|(index, width)| width.saturating_sub(floors[index]))
            .max()
            .unwrap_or(0);
        while low < high {
            let cap = low + (high - low) / 2;
            let removed = widths
                .iter()
                .enumerate()
                .filter(|(index, _)| metrics[*index].kind == kind)
                .map(|(index, width)| width.saturating_sub(floors[index]).saturating_sub(cap))
                .sum::<usize>();
            if removed > target {
                low = cap + 1;
            } else {
                high = cap;
            }
        }

        let cap = low;
        let mut removed = 0usize;
        for (index, width) in widths.iter_mut().enumerate() {
            if metrics[index].kind != kind {
                continue;
            }
            let reduction = width.saturating_sub(floors[index]).saturating_sub(cap);
            *width -= reduction;
            removed += reduction;
        }
        let mut remainder = target - removed;
        for (index, width) in widths.iter_mut().enumerate() {
            if remainder == 0 {
                break;
            }
            if metrics[index].kind == kind && width.saturating_sub(floors[index]) == cap {
                *width -= 1;
                remainder -= 1;
            }
        }
        debug_assert_eq!(remainder, 0);
        amount -= target;
        if amount == 0 {
            break;
        }
    }
    amount
}

fn should_render_records(
    rows: &[Vec<TableCell>],
    column_widths: &[usize],
    metrics: &[ColumnMetrics],
) -> bool {
    if rows.is_empty() {
        return false;
    }
    let affected = rows
        .iter()
        .filter(|row| {
            let mut cramped_expansive = 0usize;
            let mut unreadable = false;
            for ((cell, column_width), metric) in row.iter().zip(column_widths).zip(metrics) {
                let wrapped_height = wrap_cell(cell, *column_width).len();
                if metric.kind != ColumnKind::Compact
                    && wrapped_height >= CRAMPED_EXPANSIVE_CELL_LINES
                {
                    cramped_expansive += 1;
                }
                let fragmented_token = cell.longest_token() > *column_width
                    && match metric.kind {
                        ColumnKind::Compact => true,
                        ColumnKind::TokenHeavy => *column_width < MIN_SCANNABLE_EXPANSIVE_WIDTH,
                        ColumnKind::Narrative => false,
                    };
                let collapsed_narrative = metric.kind == ColumnKind::Narrative
                    && *column_width < MIN_SCANNABLE_EXPANSIVE_WIDTH
                    && wrapped_height >= CATASTROPHIC_NARRATIVE_CELL_LINES;
                unreadable |= fragmented_token || collapsed_narrative;
            }
            unreadable || cramped_expansive >= 2
        })
        .count();
    let threshold = if rows.len() == 1 {
        1
    } else {
        2.max(rows.len().div_ceil(3))
    };
    affected >= threshold
}

#[allow(clippy::too_many_arguments)]
fn render_grid(
    header: &[TableCell],
    rows: &[Vec<TableCell>],
    spillover: &[(usize, TableCell)],
    alignments: &[Alignment],
    widths: &[usize],
    available: usize,
    header_style: Style,
    separator_style: Style,
    normal_foreground: Color,
) -> Vec<Line<'static>> {
    let mut out = render_grid_row(
        header,
        widths,
        alignments,
        Some((header_style, normal_foreground)),
    );
    out.push(render_grid_separator(widths, '━', separator_style));
    let mut spillover_cursor = 0;
    for (index, row) in rows.iter().enumerate() {
        let interrupted =
            append_spillovers_at(&mut out, spillover, &mut spillover_cursor, index, available);
        if index > 0 && !interrupted {
            out.push(render_grid_separator(widths, '─', separator_style));
        }
        out.extend(render_grid_row(row, widths, alignments, None));
    }
    append_spillovers_at(
        &mut out,
        spillover,
        &mut spillover_cursor,
        rows.len(),
        available,
    );
    out
}

fn render_grid_row(
    row: &[TableCell],
    widths: &[usize],
    alignments: &[Alignment],
    header_style: Option<(Style, Color)>,
) -> Vec<Line<'static>> {
    let wrapped = row
        .iter()
        .zip(widths)
        .map(|(cell, width)| wrap_cell(cell, *width))
        .collect::<Vec<_>>();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
    let mut out = Vec::with_capacity(height);
    for line_index in 0..height {
        let mut spans = Vec::new();
        let Some(last_visible_column) = wrapped
            .iter()
            .rposition(|lines| lines.get(line_index).is_some_and(|line| line.width() > 0))
        else {
            out.push(Line::default());
            continue;
        };
        for column in 0..=last_visible_column {
            let fragment = wrapped[column].get(line_index).cloned().unwrap_or_default();
            let content_width = fragment.width();
            let spare = widths[column].saturating_sub(content_width);
            let (left, right) = aligned_padding(spare, alignments[column]);
            spans.push(Span::raw(" ".repeat(CELL_PADDING + left)));
            for mut span in fragment.spans {
                if let Some((style, normal_foreground)) = header_style {
                    span.style = span.style.add_modifier(Modifier::BOLD);
                    if span.style.fg.is_none() || span.style.fg == Some(normal_foreground) {
                        span.style.fg = style.fg;
                    }
                }
                push_or_merge_span(&mut spans, span);
            }
            if column < last_visible_column {
                spans.push(Span::raw(" ".repeat(right + CELL_PADDING + COLUMN_GAP)));
            }
        }
        out.push(Line::from(spans).style(header_style.map_or(Style::new(), |(style, _)| style)));
    }
    out
}

fn aligned_padding(spare: usize, alignment: Alignment) -> (usize, usize) {
    match alignment {
        Alignment::Right => (spare, 0),
        Alignment::Center => {
            let left = spare / 2;
            (left, spare.saturating_sub(left))
        }
        Alignment::None | Alignment::Left => (0, spare),
    }
}

fn render_grid_separator(widths: &[usize], symbol: char, style: Style) -> Line<'static> {
    let mut spans = Vec::with_capacity(widths.len() * 2);
    for (index, width) in widths.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(" ".repeat(COLUMN_GAP)));
        }
        spans.push(Span::styled(
            symbol
                .to_string()
                .repeat(width.saturating_add(CELL_PADDING * 2)),
            style,
        ));
    }
    Line::from(spans)
}

fn render_records(
    header: &[TableCell],
    rows: &[Vec<TableCell>],
    spillover: &[(usize, TableCell)],
    metrics: &[ColumnMetrics],
    available: usize,
    header_style: Style,
    separator_style: Style,
) -> Vec<Line<'static>> {
    let labels = header.iter().map(TableCell::plain_text).collect::<Vec<_>>();
    let label_width = labels
        .iter()
        .map(|label| UnicodeWidthStr::width(label.as_str()))
        .max()
        .unwrap_or(0);
    let minimum_value_width = if metrics
        .iter()
        .any(|metric| metric.kind != ColumnKind::Compact)
    {
        MIN_ALIGNED_EXPANSIVE_VALUE_WIDTH
    } else {
        MIN_ALIGNED_COMPACT_VALUE_WIDTH
    };
    let aligned = available >= CELL_PADDING + label_width + COLUMN_GAP + minimum_value_width;
    let mut out = Vec::new();
    let mut spillover_cursor = 0;
    for (row_index, row) in rows.iter().enumerate() {
        let interrupted = append_spillovers_at(
            &mut out,
            spillover,
            &mut spillover_cursor,
            row_index,
            available,
        );
        if row_index > 0 && !interrupted {
            out.push(Line::styled("─".repeat(available), separator_style));
        }
        for (column, value) in row.iter().enumerate() {
            if aligned {
                render_aligned_record_field(
                    &mut out,
                    &labels[column],
                    value,
                    label_width,
                    available,
                    header_style,
                );
            } else {
                render_stacked_record_field(
                    &mut out,
                    &labels[column],
                    value,
                    available,
                    header_style,
                );
            }
        }
    }
    append_spillovers_at(
        &mut out,
        spillover,
        &mut spillover_cursor,
        rows.len(),
        available,
    );
    out
}

fn render_aligned_record_field(
    out: &mut Vec<Line<'static>>,
    label: &str,
    value: &TableCell,
    label_width: usize,
    available: usize,
    label_style: Style,
) {
    let value_indent = CELL_PADDING + label_width + COLUMN_GAP;
    let value_width = available.saturating_sub(value_indent).max(1);
    for (index, line) in wrap_cell(value, value_width).into_iter().enumerate() {
        let mut spans = if index == 0 {
            vec![
                Span::raw(" "),
                Span::styled(label.to_owned(), label_style),
                Span::raw(" ".repeat(
                    label_width.saturating_sub(UnicodeWidthStr::width(label)) + COLUMN_GAP,
                )),
            ]
        } else {
            vec![Span::raw(" ".repeat(value_indent))]
        };
        append_line_spans(&mut spans, line);
        out.push(Line::from(spans));
    }
}

fn render_stacked_record_field(
    out: &mut Vec<Line<'static>>,
    label: &str,
    value: &TableCell,
    available: usize,
    label_style: Style,
) {
    let label_indent = CELL_PADDING.min(available.saturating_sub(1));
    let label_width = available.saturating_sub(label_indent).max(1);
    let label_line = Line::from(Span::styled(label.to_owned(), label_style));
    for line in wrap_styled_line(&label_line, label_width) {
        let mut spans = vec![Span::raw(" ".repeat(label_indent))];
        append_line_spans(&mut spans, line);
        out.push(Line::from(spans));
    }
    let value_indent = RECORD_VALUE_INDENT.min(available.saturating_sub(1));
    let value_width = available.saturating_sub(value_indent).max(1);
    for line in wrap_cell(value, value_width) {
        let mut spans = vec![Span::raw(" ".repeat(value_indent))];
        append_line_spans(&mut spans, line);
        out.push(Line::from(spans));
    }
}

fn render_header_fallback(
    header: &[TableCell],
    alignments: &[Alignment],
    available: usize,
    header_style: Style,
) -> Vec<Line<'static>> {
    let header_source = header
        .iter()
        .map(TableCell::plain_text)
        .map(|cell| cell.replace('|', "\\|"))
        .collect::<Vec<_>>()
        .join(" | ");
    let delimiter = alignments
        .iter()
        .map(|alignment| match alignment {
            Alignment::Left => ":---",
            Alignment::Center => ":---:",
            Alignment::Right => "---:",
            Alignment::None => "---",
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let mut out = wrap_styled_line(
        &Line::from(Span::styled(format!("| {header_source} |"), header_style)),
        available,
    );
    out.extend(wrap_styled_line(
        &Line::raw(format!("| {delimiter} |")),
        available,
    ));
    out
}

fn wrap_cell(cell: &TableCell, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    if cell.lines.is_empty() {
        return vec![Line::default()];
    }
    let mut out = Vec::new();
    for line in &cell.lines {
        let wrapped = wrap_styled_line(line, width);
        if wrapped.is_empty() {
            out.push(Line::default());
        } else {
            out.extend(wrapped);
        }
    }
    if out.is_empty() {
        out.push(Line::default());
    }
    out
}

fn wrap_styled_line(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut current_width = 0usize;

    for span in &line.spans {
        let mut token = String::new();
        let mut whitespace = false;
        for character in span.content.chars() {
            let is_whitespace = character.is_whitespace();
            if !token.is_empty() && whitespace != is_whitespace {
                push_wrapped_token(
                    &mut out,
                    &mut current,
                    &mut current_width,
                    std::mem::take(&mut token),
                    whitespace,
                    span.style,
                    width,
                );
            }
            whitespace = is_whitespace;
            token.push(character);
        }
        if !token.is_empty() {
            push_wrapped_token(
                &mut out,
                &mut current,
                &mut current_width,
                token,
                whitespace,
                span.style,
                width,
            );
        }
    }
    flush_wrapped_line(&mut out, &mut current, &mut current_width);
    if out.is_empty() {
        out.push(Line::default());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn push_wrapped_token(
    out: &mut Vec<Line<'static>>,
    current: &mut Vec<Span<'static>>,
    current_width: &mut usize,
    token: String,
    whitespace: bool,
    style: Style,
    width: usize,
) {
    if whitespace {
        if current.is_empty() {
            return;
        }
        let normalized = " ";
        if *current_width < width {
            push_or_merge_span(current, Span::styled(normalized, style));
            *current_width += 1;
        }
        return;
    }

    let token_width = UnicodeWidthStr::width(token.as_str());
    if *current_width > 0 && current_width.saturating_add(token_width) > width {
        trim_trailing_space(current, current_width);
        flush_wrapped_line(out, current, current_width);
    }
    if token_width <= width {
        push_or_merge_span(current, Span::styled(token, style));
        *current_width += token_width;
        return;
    }

    let mut piece = String::new();
    let mut piece_width = 0usize;
    for grapheme in token.graphemes(true) {
        let measured_width = UnicodeWidthStr::width(grapheme);
        let (grapheme, grapheme_width) = if measured_width > width {
            // A terminal cannot display a wide grapheme in a one-cell content
            // budget. Replace the whole cluster rather than splitting it into
            // invalid scalar fragments or overflowing the row.
            ("�", 1)
        } else {
            (grapheme, measured_width)
        };
        if piece_width > 0 && piece_width.saturating_add(grapheme_width) > width {
            push_or_merge_span(current, Span::styled(std::mem::take(&mut piece), style));
            *current_width += piece_width;
            flush_wrapped_line(out, current, current_width);
            piece_width = 0;
        }
        piece.push_str(grapheme);
        piece_width += grapheme_width;
    }
    if !piece.is_empty() {
        push_or_merge_span(current, Span::styled(piece, style));
        *current_width += piece_width;
    }
}

fn append_spillovers_at(
    out: &mut Vec<Line<'static>>,
    spillover: &[(usize, TableCell)],
    cursor: &mut usize,
    row_position: usize,
    available: usize,
) -> bool {
    let start = *cursor;
    while let Some((position, cell)) = spillover.get(*cursor) {
        if *position != row_position {
            break;
        }
        out.extend(wrap_cell(cell, available));
        *cursor += 1;
    }
    *cursor != start
}

fn fit_prefix<'a>(prefix: &'a str, max_width: usize) -> Cow<'a, str> {
    if UnicodeWidthStr::width(prefix) <= max_width {
        return Cow::Borrowed(prefix);
    }

    let mut fitted = String::new();
    let mut fitted_width = 0usize;
    for grapheme in prefix.graphemes(true) {
        let grapheme_width = UnicodeWidthStr::width(grapheme);
        if fitted_width.saturating_add(grapheme_width) > max_width {
            break;
        }
        fitted.push_str(grapheme);
        fitted_width += grapheme_width;
    }
    Cow::Owned(fitted)
}

fn trim_trailing_space(spans: &mut Vec<Span<'static>>, width: &mut usize) {
    if let Some(last) = spans.last_mut() {
        let trimmed = last.content.trim_end().to_owned();
        let removed = UnicodeWidthStr::width(last.content.as_ref())
            .saturating_sub(UnicodeWidthStr::width(trimmed.as_str()));
        last.content = trimmed.into();
        *width = width.saturating_sub(removed);
        if last.content.is_empty() {
            spans.pop();
        }
    }
}

fn flush_wrapped_line(
    out: &mut Vec<Line<'static>>,
    current: &mut Vec<Span<'static>>,
    current_width: &mut usize,
) {
    if current.is_empty() {
        return;
    }
    out.push(Line::from(std::mem::take(current)));
    *current_width = 0;
}

fn append_line_spans(target: &mut Vec<Span<'static>>, line: Line<'static>) {
    for span in line.spans {
        push_or_merge_span(target, span);
    }
}

fn push_or_merge_span(target: &mut Vec<Span<'static>>, span: Span<'static>) {
    if span.content.is_empty() {
        return;
    }
    if let Some(last) = target.last_mut() {
        if last.style == span.style {
            last.content.to_mut().push_str(span.content.as_ref());
            return;
        }
    }
    target.push(span);
}

fn trim_spans(mut spans: Vec<Span<'static>>) -> Vec<Span<'static>> {
    let first_content = spans
        .iter()
        .position(|span| !span.content.trim().is_empty())
        .unwrap_or(spans.len());
    if first_content > 0 {
        spans.drain(..first_content);
    }
    if let Some(first) = spans.first_mut() {
        first.content = first.content.trim_start().to_owned().into();
    }
    while spans
        .last()
        .is_some_and(|span| span.content.trim().is_empty())
    {
        spans.pop();
    }
    if let Some(last) = spans.last_mut() {
        last.content = last.content.trim_end().to_owned().into();
    }
    spans
}

fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::TERMINAL;

    fn push_cell(table: &mut TableState, text: &str) {
        table.start_cell();
        table.push_span(Span::raw(text.to_owned()));
        table.end_cell();
    }

    fn flat(lines: &[Line<'_>]) -> Vec<String> {
        lines.iter().map(line_text).collect()
    }

    #[test]
    fn spillover_prose_stays_between_the_rows_that_surrounded_it() {
        let mut table = TableState::new(vec![Alignment::None, Alignment::None]);
        table.start_head();
        push_cell(&mut table, "Key");
        push_cell(&mut table, "Value");
        table.end_head();

        table.start_row(true);
        push_cell(&mut table, "alpha");
        push_cell(&mut table, "one");
        table.end_row();

        table.start_row(false);
        push_cell(&mut table, "intervening prose");
        table.end_row();

        table.start_row(true);
        push_cell(&mut table, "omega");
        push_cell(&mut table, "two");
        table.end_row();

        let text = flat(&render(table, 40, "", "", Style::new(), &TERMINAL));
        let alpha = text.iter().position(|line| line.contains("alpha")).unwrap();
        let prose = text
            .iter()
            .position(|line| line.contains("intervening prose"))
            .unwrap();
        let omega = text.iter().position(|line| line.contains("omega")).unwrap();
        assert!(alpha < prose && prose < omega, "{text:?}");
    }

    #[test]
    fn deeply_prefixed_wide_grapheme_is_replaced_without_overflow() {
        let mut table = TableState::new(vec![Alignment::None, Alignment::None]);
        table.start_head();
        push_cell(&mut table, "A");
        push_cell(&mut table, "B");
        table.end_head();
        table.start_row(true);
        push_cell(&mut table, "👩‍💻");
        push_cell(&mut table, "x");
        table.end_row();

        let prefix = "▎ ".repeat(5);
        let text = flat(&render(
            table,
            10,
            &prefix,
            &prefix,
            Style::new(),
            &TERMINAL,
        ));
        assert!(
            text.iter()
                .all(|line| UnicodeWidthStr::width(line.as_str()) <= 10),
            "{text:?}"
        );
        let content = text.join("\n");
        assert_eq!(content.matches('�').count(), 1, "{text:?}");
        assert!(!content.contains("👩‍💻"), "{text:?}");
    }
}
