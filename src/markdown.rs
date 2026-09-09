use crate::theme::Theme;
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

mod fences;
mod table;

/// Render markdown to styled, word-wrapped lines for the transcript pane.
/// Tolerates incomplete input (e.g. an unclosed code fence mid-stream).
pub fn render(text: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut r = Renderer::new(width.max(10), *theme);
    let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let normalized = fences::unwrap_table_markdown_fences(text);
    for (event, range) in Parser::new_ext(normalized.as_ref(), options).into_offset_iter() {
        let row_has_boundary_pipe = matches!(&event, Event::Start(Tag::TableRow))
            && normalized
                .get(range)
                .map(str::trim)
                .is_some_and(|row| row.starts_with('|') || row.ends_with('|'));
        r.event(event, row_has_boundary_pipe);
    }
    r.finish()
}

struct Renderer {
    width: usize,
    theme: Theme,
    base: Style,
    lines: Vec<Line<'static>>,
    cur: Vec<Span<'static>>,
    cur_w: usize,
    first_prefix: String,
    cont_prefix: String,
    prefix_style: Style,
    prefix_quote_depth: usize,
    first_line_of_block: bool,
    bold: u32,
    italic: u32,
    strike: u32,
    heading_level: Option<u8>,
    code_block: Option<String>,
    list_stack: Vec<Option<u64>>,
    in_item: u32,
    quote_depth: usize,
    link: Option<(String, bool)>, // (url, text matched url i.e. autolink)
    table: Option<table::TableState>,
    table_first_prefix: String,
    table_cont_prefix: String,
    table_prefix_style: Style,
}

impl Renderer {
    fn new(width: usize, theme: Theme) -> Self {
        Renderer {
            width,
            theme,
            base: Style::new().fg(theme.fg),
            lines: Vec::new(),
            cur: Vec::new(),
            cur_w: 0,
            first_prefix: String::new(),
            cont_prefix: String::new(),
            prefix_style: Style::new().fg(theme.dim),
            prefix_quote_depth: 0,
            first_line_of_block: true,
            bold: 0,
            italic: 0,
            strike: 0,
            heading_level: None,
            code_block: None,
            list_stack: Vec::new(),
            in_item: 0,
            quote_depth: 0,
            link: None,
            table: None,
            table_first_prefix: String::new(),
            table_cont_prefix: String::new(),
            table_prefix_style: Style::new().fg(theme.dim),
        }
    }

    fn event(&mut self, event: Event, row_has_boundary_pipe: bool) {
        match event {
            Event::Start(tag) => self.start(tag, row_has_boundary_pipe),
            Event::End(tag) => self.end(tag),
            Event::Text(t) => {
                if let Some(buf) = &mut self.code_block {
                    buf.push_str(&t);
                } else {
                    if let Some((url, auto)) = &mut self.link {
                        if url == t.as_ref() {
                            *auto = true;
                        }
                    }
                    let style = self.inline_style();
                    self.push_words(&t, style);
                }
            }
            Event::Code(t) => {
                let mut style = Style::new().fg(self.theme.code);
                if let Some(surface) = self.theme.surface {
                    style = style.bg(surface);
                }
                self.push_token(&t, style);
            }
            Event::Html(t) | Event::InlineHtml(t) if self.table.is_some() => {
                // pulldown-cmark can retain inline HTML inside table cells. Keep
                // it visible here even though raw HTML is ignored elsewhere in
                // the terminal transcript.
                let style = self.inline_style();
                self.push_words(&t, style);
            }
            Event::SoftBreak => {
                let style = self.inline_style();
                self.push_words(" ", style);
            }
            Event::HardBreak => {
                if let Some(table) = &mut self.table {
                    table.hard_break();
                } else {
                    self.flush_line();
                }
            }
            Event::Rule => {
                self.gap();
                self.lines.push(Line::styled(
                    "─".repeat(self.width),
                    Style::new().fg(self.theme.border),
                ));
            }
            Event::TaskListMarker(done) => {
                let style = if done {
                    Style::new().fg(self.theme.success)
                } else {
                    Style::new().fg(self.theme.dim)
                };
                self.push_token(if done { "✓ " } else { "○ " }, style);
            }
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag, row_has_boundary_pipe: bool) {
        match tag {
            Tag::Table(alignments) => {
                // A table can begin after inline text without an intervening
                // paragraph boundary (for example, inside a list item). Emit
                // that text before the table and let the table continue from
                // the list indentation instead of stealing its bullet.
                self.flush_line();
                self.gap();
                if self.in_item > 0 && !self.cont_prefix.is_empty() {
                    let nested_quote =
                        "▎ ".repeat(self.quote_depth.saturating_sub(self.prefix_quote_depth));
                    self.table_first_prefix = if self.first_line_of_block {
                        format!("{}{nested_quote}", self.first_prefix)
                    } else {
                        format!("{}{nested_quote}", self.cont_prefix)
                    };
                    self.table_cont_prefix = format!("{}{nested_quote}", self.cont_prefix);
                } else {
                    let quote = self.quote_prefix();
                    self.table_first_prefix = quote.clone();
                    self.table_cont_prefix = quote;
                }
                self.table_prefix_style = if self.quote_depth > 0 {
                    Style::new().fg(self.theme.accent2)
                } else {
                    self.prefix_style
                };
                self.table = Some(table::TableState::new(alignments));
            }
            Tag::TableHead => {
                if let Some(table) = &mut self.table {
                    table.start_head();
                }
            }
            Tag::TableRow => {
                if let Some(table) = &mut self.table {
                    table.start_row(row_has_boundary_pipe);
                }
            }
            Tag::TableCell => {
                if let Some(table) = &mut self.table {
                    table.start_cell();
                }
            }
            Tag::Paragraph => {
                if self.in_item == 0 {
                    self.gap();
                    let q = self.quote_prefix();
                    let style = if self.quote_depth > 0 {
                        Style::new().fg(self.theme.accent2)
                    } else {
                        Style::new().fg(self.theme.dim)
                    };
                    self.set_block(q.clone(), q, style);
                }
            }
            Tag::Heading { level, .. } => {
                self.gap();
                let q = self.quote_prefix();
                self.set_block(q.clone(), q, Style::new().fg(self.theme.dim));
                self.heading_level = Some(match level {
                    HeadingLevel::H1 => 1,
                    HeadingLevel::H2 => 2,
                    HeadingLevel::H3 => 3,
                    _ => 4,
                });
            }
            Tag::BlockQuote(_) => {
                self.gap();
                self.quote_depth += 1;
            }
            Tag::CodeBlock(_) => {
                self.gap();
                self.code_block = Some(String::new());
            }
            Tag::List(start) => {
                if self.list_stack.is_empty() && self.in_item == 0 {
                    self.gap();
                }
                self.list_stack.push(start);
            }
            Tag::Item => {
                self.flush_line();
                self.in_item += 1;
                let bullet = match self.list_stack.last_mut() {
                    Some(Some(n)) => {
                        let s = format!("{n}. ");
                        *n += 1;
                        s
                    }
                    _ => "• ".into(),
                };
                let indent = format!(
                    "{}{}",
                    self.quote_prefix(),
                    "  ".repeat(self.list_stack.len().saturating_sub(1))
                );
                let cont = format!("{indent}{}", " ".repeat(bullet.chars().count()));
                self.set_block(
                    format!("{indent}{bullet}"),
                    cont,
                    Style::new().fg(self.theme.accent2),
                );
            }
            Tag::Strong => self.bold += 1,
            Tag::Emphasis => self.italic += 1,
            Tag::Strikethrough => self.strike += 1,
            Tag::Link { dest_url, .. } => self.link = Some((dest_url.to_string(), false)),
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    let first_prefix = std::mem::take(&mut self.table_first_prefix);
                    let cont_prefix = std::mem::take(&mut self.table_cont_prefix);
                    let rendered = table::render(
                        table,
                        self.width,
                        &first_prefix,
                        &cont_prefix,
                        self.table_prefix_style,
                        &self.theme,
                    );
                    if !rendered.is_empty() {
                        self.first_line_of_block = false;
                    }
                    self.lines.extend(rendered);
                }
            }
            TagEnd::TableHead => {
                if let Some(table) = &mut self.table {
                    table.end_head();
                }
            }
            TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    table.end_row();
                }
            }
            TagEnd::TableCell => {
                if let Some(table) = &mut self.table {
                    table.end_cell();
                }
            }
            TagEnd::Paragraph | TagEnd::Heading(_) => {
                self.flush_line();
                self.heading_level = None;
            }
            TagEnd::BlockQuote(_) => self.quote_depth = self.quote_depth.saturating_sub(1),
            TagEnd::CodeBlock => {
                if let Some(buf) = self.code_block.take() {
                    self.emit_code_block(&buf);
                }
            }
            TagEnd::List(_) => {
                self.list_stack.pop();
            }
            TagEnd::Item => {
                self.flush_line();
                self.in_item = self.in_item.saturating_sub(1);
            }
            TagEnd::Strong => self.bold = self.bold.saturating_sub(1),
            TagEnd::Emphasis => self.italic = self.italic.saturating_sub(1),
            TagEnd::Strikethrough => self.strike = self.strike.saturating_sub(1),
            TagEnd::Link => {
                if let Some((url, autolink)) = self.link.take() {
                    if !autolink && !url.is_empty() {
                        let style = Style::new().fg(self.theme.dim);
                        self.push_words(&format!(" ({url})"), style);
                    }
                }
            }
            _ => {}
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_line();
        if let Some(buf) = self.code_block.take() {
            self.emit_code_block(&buf);
        }
        while self.lines.last().is_some_and(|l| l.spans.is_empty()) {
            self.lines.pop();
        }
        self.lines
    }

    // ---- layout primitives ----

    fn inline_style(&self) -> Style {
        let mut style = match self.heading_level {
            Some(1) => Style::new()
                .fg(self.theme.accent2)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            Some(2) => Style::new()
                .fg(self.theme.accent2)
                .add_modifier(Modifier::BOLD),
            Some(_) => self.base.add_modifier(Modifier::BOLD),
            None if self.quote_depth > 0 => Style::new()
                .fg(self.theme.dim)
                .add_modifier(Modifier::ITALIC),
            None => self.base,
        };
        if self.bold > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.italic > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.strike > 0 {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        style
    }

    fn quote_prefix(&self) -> String {
        "▎ ".repeat(self.quote_depth)
    }

    fn set_block(&mut self, first: String, cont: String, prefix_style: Style) {
        self.first_prefix = first;
        self.cont_prefix = cont;
        self.prefix_style = prefix_style;
        self.prefix_quote_depth = self.quote_depth;
        self.first_line_of_block = true;
    }

    fn prefix(&self) -> &str {
        if self.first_line_of_block {
            &self.first_prefix
        } else {
            &self.cont_prefix
        }
    }

    fn avail(&self) -> usize {
        self.width
            .saturating_sub(UnicodeWidthStr::width(self.prefix()))
            .max(4)
    }

    fn gap(&mut self) {
        if self.lines.last().is_some_and(|l| !l.spans.is_empty()) {
            self.lines.push(Line::raw(""));
        }
    }

    fn flush_line(&mut self) {
        while self.cur.last().is_some_and(|s| s.content.trim().is_empty()) {
            self.cur.pop();
        }
        if self.cur.is_empty() {
            return;
        }
        let mut spans = Vec::new();
        let prefix = self.prefix().to_owned();
        if !prefix.is_empty() {
            spans.push(Span::styled(prefix, self.prefix_style));
        }
        spans.append(&mut self.cur);
        self.lines.push(Line::from(spans));
        self.cur_w = 0;
        self.first_line_of_block = false;
    }

    /// Append text with word wrapping; whitespace runs are kept as separate
    /// tokens so wraps land on word boundaries.
    fn push_words(&mut self, text: &str, style: Style) {
        let mut token = String::new();
        let mut token_is_ws = false;
        for c in text.chars() {
            let is_ws = c.is_whitespace();
            if !token.is_empty() && is_ws != token_is_ws {
                self.push_token(&token, style);
                token.clear();
            }
            token_is_ws = is_ws;
            token.push(c);
        }
        if !token.is_empty() {
            self.push_token(&token, style);
        }
    }

    /// Append one unbreakable token, wrapping (or hard-splitting) as needed.
    /// All measurements use display width, so CJK and emoji wrap correctly.
    fn push_token(&mut self, token: &str, style: Style) {
        if let Some(table) = &mut self.table {
            table.push_span(Span::styled(token.to_owned(), style));
            return;
        }
        let tw = UnicodeWidthStr::width(token);
        if self.cur_w + tw > self.avail() && self.cur_w > 0 {
            self.flush_line();
            if token.trim().is_empty() {
                return; // don't carry the wrapping space to the next line
            }
        }
        if tw > self.avail() {
            let mut piece = String::new();
            let mut piece_w = 0;
            for ch in token.chars() {
                let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                if piece_w + cw > self.avail() && piece_w > 0 {
                    self.cur_w += piece_w;
                    self.cur
                        .push(Span::styled(std::mem::take(&mut piece), style));
                    self.flush_line();
                    piece_w = 0;
                }
                piece.push(ch);
                piece_w += cw;
            }
            if !piece.is_empty() {
                self.cur_w += piece_w;
                self.cur.push(Span::styled(piece, style));
            }
        } else {
            self.cur_w += tw;
            self.cur.push(Span::styled(token.to_owned(), style));
        }
    }

    /// Code blocks render as full-width "cards" on the surface color when the
    /// theme has one, falling back to a gutter bar for plain-ANSI themes.
    /// Code is never word-wrapped; long lines are hard-cut so indentation
    /// stays intact.
    fn emit_code_block(&mut self, buf: &str) {
        match self.theme.surface {
            Some(surface) => {
                let style = Style::new().fg(self.theme.code).bg(surface);
                let inner = self.width.saturating_sub(2).max(4);
                for line in buf.lines() {
                    let (cut, cut_w) = cut_to_width(line, inner);
                    let pad = " ".repeat(self.width.saturating_sub(cut_w + 1));
                    self.lines
                        .push(Line::from(Span::styled(format!(" {cut}{pad}"), style)));
                }
            }
            None => {
                let style = Style::new().fg(self.theme.code);
                let border = Style::new().fg(self.theme.dim);
                let inner = self.width.saturating_sub(2).max(4);
                for line in buf.lines() {
                    let (cut, _) = cut_to_width(line, inner);
                    self.lines.push(Line::from(vec![
                        Span::styled("▏ ", border),
                        Span::styled(cut, style),
                    ]));
                }
            }
        }
    }
}

fn cut_to_width(line: &str, max: usize) -> (String, usize) {
    let mut cut = String::new();
    let mut w = 0;
    for ch in line.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw > max {
            break;
        }
        cut.push(ch);
        w += cw;
    }
    (cut, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{MOCHA, TERMINAL};

    fn flat(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn wraps_paragraphs_at_word_boundaries() {
        let lines = render("alpha beta gamma delta", 12, &TERMINAL);
        let text = flat(&lines);
        assert!(text.len() > 1, "expected wrapping, got {text:?}");
        assert!(text.iter().all(|l| l.chars().count() <= 12), "{text:?}");
        assert!(!text.iter().any(|l| l.ends_with(' ')), "{text:?}");
    }

    #[test]
    fn wraps_cjk_text_by_display_width() {
        let lines = render(
            "日本語のテキストを正しく折り返す必要があります",
            12,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert!(text.len() > 1, "{text:?}");
        assert!(
            text.iter()
                .all(|l| UnicodeWidthStr::width(l.as_str()) <= 12),
            "{text:?}"
        );
    }

    #[test]
    fn renders_code_blocks_verbatim_with_gutter_fallback() {
        let lines = render(
            "text\n\n```rust\nfn main() {}\n    indented\n```",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert!(text.contains(&"▏ fn main() {}".to_string()), "{text:?}");
        assert!(text.contains(&"▏     indented".to_string()), "{text:?}");
    }

    #[test]
    fn themed_code_blocks_render_as_full_width_cards() {
        let lines = render("```rust\nfn main() {}\n```", 40, &MOCHA);
        let text = flat(&lines);
        let card_line = text.iter().find(|l| l.contains("fn main() {}")).unwrap();
        assert_eq!(
            UnicodeWidthStr::width(card_line.as_str()),
            40,
            "{card_line:?}"
        );
        let styled = lines
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.contains("fn main()")))
            .unwrap();
        assert_eq!(styled.spans[0].style.bg, MOCHA.surface);
    }

    #[test]
    fn renders_lists_with_bullets_and_numbers() {
        let lines = render("- one\n- two\n\n1. first\n2. second", 40, &TERMINAL);
        let text = flat(&lines);
        assert!(text.contains(&"• one".to_string()), "{text:?}");
        assert!(text.contains(&"1. first".to_string()), "{text:?}");
        assert!(text.contains(&"2. second".to_string()), "{text:?}");
    }

    #[test]
    fn tolerates_unclosed_fence_mid_stream() {
        let lines = render("start\n\n```py\nprint(1)", 40, &TERMINAL);
        let text = flat(&lines);
        assert!(text.contains(&"▏ print(1)".to_string()), "{text:?}");
    }

    #[test]
    fn inline_code_and_bold_do_not_panic_and_keep_text() {
        let lines = render("use `cargo build` to **compile** it", 80, &TERMINAL);
        let text = flat(&lines).join(" ");
        assert!(text.contains("cargo build"));
        assert!(text.contains("compile"));
    }

    #[test]
    fn table_renders_as_a_borderless_grid() {
        let lines = render("| A | B |\n|---|---|\n| 1 | 2 |\n", 80, &TERMINAL);
        let text = flat(&lines);
        assert_eq!(text, vec![" A      B", "━━━━━  ━━━━━", " 1      2"]);
        assert_eq!(lines[0].style.fg, Some(TERMINAL.accent2));
        assert!(lines[0].style.add_modifier.contains(Modifier::BOLD));
        assert!(lines[0]
            .spans
            .iter()
            .filter(|span| !span.content.trim().is_empty())
            .all(|span| {
                span.style.fg == Some(TERMINAL.accent2)
                    && span.style.add_modifier.contains(Modifier::BOLD)
            }));
        assert!(lines[1].spans[0].style.add_modifier.contains(Modifier::DIM));
    }

    #[test]
    fn table_honors_column_alignment() {
        let lines = render(
            "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |\n",
            80,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert_eq!(text[0], " Left    Center    Right");
        assert_eq!(text[2], " a         b           c");
    }

    #[test]
    fn cramped_table_transposes_to_stacked_records() {
        let markdown = "| Key | Notes |\n| --- | --- |\n\
                        | firstlongid | A readable explanatory sentence for this row. |\n\
                        | secondlongid | Another readable explanatory sentence for this row. |\n\
                        | short | A final readable explanatory sentence for this row. |\n";
        let lines = render(markdown, 17, &TERMINAL);
        let text = flat(&lines);
        assert_eq!(
            text.iter()
                .filter(|line| line.trim_start().starts_with("Key"))
                .count(),
            3
        );
        assert_eq!(text.iter().filter(|line| line.trim() == "Notes").count(), 3);
        assert!(text.iter().any(|line| line == &"─".repeat(17)), "{text:?}");
        assert!(!text.iter().any(|line| line.contains('━')), "{text:?}");
        assert!(text
            .iter()
            .all(|line| UnicodeWidthStr::width(line.as_str()) <= 17));
    }

    #[test]
    fn one_compact_outlier_stays_in_the_grid() {
        let markdown = "| Key | Date | State |\n| --- | --- | --- |\n\
                        | short | 2025-01-01 | Ready |\n\
                        | verylongidentifier | 2025-02-02 | Ready |\n\
                        | final | 2025-03-03 | Done |\n";
        let text = flat(&render(markdown, 40, &TERMINAL));
        assert!(text.iter().any(|line| line.contains('━')), "{text:?}");
        assert_eq!(text.iter().filter(|line| line.contains("Key")).count(), 1);
        assert!(text
            .iter()
            .all(|line| UnicodeWidthStr::width(line.as_str()) <= 40));
    }

    #[test]
    fn systemic_compact_fragmentation_crosses_the_record_threshold() {
        let markdown = "| Key | Date | State |\n| --- | --- | --- |\n\
                        | verylongidentifier | 2025-01-01 | Ready |\n\
                        | secondlongidentifier | 2025-02-02 | Ready |\n\
                        | final | 2025-03-03 | Done |\n";
        let text = flat(&render(markdown, 40, &TERMINAL));
        assert!(!text.iter().any(|line| line.contains('━')), "{text:?}");
        assert_eq!(
            text.iter()
                .filter(|line| line.trim_start().starts_with("Key"))
                .count(),
            3
        );
    }

    #[test]
    fn narrative_fragmentation_only_transposes_when_catastrophic() {
        let moderate = "| Key | Description |\n| --- | --- |\n\
                        | x | one two three four five six seven eight |\n";
        let moderate = flat(&render(moderate, 19, &TERMINAL));
        assert!(
            moderate.iter().any(|line| line.contains('━')),
            "{moderate:?}"
        );

        let catastrophic = format!(
            "| Key | Description |\n| --- | --- |\n| x | {} |\n",
            "one two three four five six seven eight nine ten ".repeat(3)
        );
        let catastrophic = flat(&render(&catastrophic, 19, &TERMINAL));
        assert!(
            !catastrophic.iter().any(|line| line.contains('━')),
            "{catastrophic:?}"
        );
        assert!(catastrophic.iter().any(|line| line.trim() == "Description"));
    }

    #[test]
    fn tables_preserve_unicode_escaped_pipes_and_inline_styles() {
        let markdown = "| Key | Notes |\n| --- | --- |\n\
                        | ｶﾞﾊﾟtail | 日本語 ✅ with an escaped \\| pipe |\n\
                        | style | `cargo test` and **bold** |\n";
        let lines = render(markdown, 34, &TERMINAL);
        let text = flat(&lines);
        let content = text.join("\n");
        assert!(content.contains("ｶﾞﾊﾟtail"), "{text:?}");
        assert!(content.contains("日本語"), "{text:?}");
        assert!(content.contains('✅'), "{text:?}");
        assert!(content.contains("escaped | pipe"), "{text:?}");
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content.contains("cargo test") && span.style.fg == Some(TERMINAL.code)
        }));
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content.contains("bold") && span.style.add_modifier.contains(Modifier::BOLD)
        }));
        assert!(text
            .iter()
            .all(|line| UnicodeWidthStr::width(line.as_str()) <= 34));
    }

    #[test]
    fn table_header_accent_does_not_overwrite_inline_code_color() {
        let lines = render(
            "| Plain | `Command` |\n| --- | --- |\n| value | cargo |\n",
            40,
            &TERMINAL,
        );
        let plain = lines[0]
            .spans
            .iter()
            .find(|span| span.content.contains("Plain"))
            .unwrap();
        let code = lines[0]
            .spans
            .iter()
            .find(|span| span.content.contains("Command"))
            .unwrap();
        assert_eq!(plain.style.fg, Some(TERMINAL.accent2));
        assert_eq!(code.style.fg, Some(TERMINAL.code));
        assert!(plain.style.add_modifier.contains(Modifier::BOLD));
        assert!(code.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn table_in_blockquote_keeps_the_quote_gutter() {
        let lines = render("> | A | B |\n> |---|---|\n> | 1 | 2 |\n", 40, &TERMINAL);
        let text = flat(&lines);
        assert!(text.iter().all(|line| line.starts_with("▎ ")), "{text:?}");
        assert!(lines.iter().all(|line| {
            line.spans
                .first()
                .is_some_and(|span| span.style.fg == Some(TERMINAL.accent2))
        }));
    }

    #[test]
    fn table_nested_in_a_list_uses_the_continuation_indent() {
        let lines = render(
            "- item\n\n  | A | B |\n  |---|---|\n  | 1 | 2 |\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        let table_lines = text
            .iter()
            .filter(|line| line.contains('━') || line.contains(" A") || line.contains(" 1"))
            .collect::<Vec<_>>();
        assert!(!table_lines.is_empty(), "{text:?}");
        assert!(
            table_lines.iter().all(|line| line.starts_with("  ")),
            "{text:?}"
        );
    }

    #[test]
    fn quoted_table_after_list_text_preserves_order_and_nested_prefix() {
        let lines = render(
            "- item\n  > | A | B |\n  > |---|---|\n  > | 1 | 2 |\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert_eq!(text.first().map(String::as_str), Some("• item"), "{text:?}");

        let table = text
            .iter()
            .filter(|line| line.contains('━') || line.contains(" A") || line.contains(" 1"))
            .collect::<Vec<_>>();
        assert_eq!(table.len(), 3, "{text:?}");
        assert!(
            table.iter().all(|line| line.starts_with("  ▎ ")),
            "{text:?}"
        );
        assert!(!table.iter().any(|line| line.starts_with('•')), "{text:?}");
        assert!(
            lines
                .iter()
                .skip(1)
                .filter(|line| !line.spans.is_empty())
                .all(|line| line.spans.first().is_some_and(|span| {
                    span.content == "  ▎ " && span.style.fg == Some(TERMINAL.accent2)
                })),
            "{text:?}"
        );
    }

    #[test]
    fn table_only_list_item_keeps_its_bullet_then_indents() {
        let lines = render("- | A | B |\n  |---|---|\n  | 1 | 2 |\n", 32, &TERMINAL);
        let text = flat(&lines);
        assert_eq!(text[0], "•  A      B", "{text:?}");
        assert!(text[1].starts_with("  ━"), "{text:?}");
        assert!(text[2].starts_with("   1"), "{text:?}");
    }

    #[test]
    fn complete_markdown_fence_around_table_renders_as_table() {
        let lines = render(
            "```markdown\n| Name | State |\n| --- | --- |\n| build | ready |\n```\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert!(text.iter().any(|line| line.contains('━')), "{text:?}");
        assert!(text.iter().any(|line| line.contains("build")), "{text:?}");
        assert!(!text.iter().any(|line| line.starts_with("▏ ")), "{text:?}");
    }

    #[test]
    fn table_does_not_swallow_following_prose_as_a_sparse_row() {
        let lines = render(
            "| A | B |\n| --- | --- |\n| 1 | 2 |\nFollowing prose remains outside the table.\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert_eq!(text.iter().filter(|line| line.contains('━')).count(), 1);
        assert!(!text.iter().any(|line| line.contains('─')), "{text:?}");
        assert!(
            text.iter()
                .any(|line| line.contains("Following prose remains")),
            "{text:?}"
        );
    }

    #[test]
    fn spillover_between_table_rows_keeps_source_order() {
        let lines = render(
            "| A | B |\n| --- | --- |\n| first | one |\nintervening prose\n| last | two |\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        let first = text.iter().position(|line| line.contains("first")).unwrap();
        let prose = text
            .iter()
            .position(|line| line.contains("intervening prose"))
            .unwrap();
        let last = text.iter().position(|line| line.contains("last")).unwrap();
        assert!(first < prose && prose < last, "{text:?}");
    }

    #[test]
    fn explicit_sparse_row_with_boundary_pipes_stays_in_table() {
        let lines = render(
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n| total |\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert!(text.iter().any(|line| line.contains('─')), "{text:?}");
        assert!(text.iter().any(|line| line.contains("total")), "{text:?}");
    }

    #[test]
    fn table_rendering_is_stable_while_markdown_streams() {
        let header_only = flat(&render("| Feature | State |\n", 40, &TERMINAL));
        let partial_schema = flat(&render("| Feature | State |\n| --- | ---", 40, &TERMINAL));
        let complete = flat(&render(
            "| Feature | State |\n| --- | --- |\n| tables | ready |\n",
            40,
            &TERMINAL,
        ));

        assert!(header_only.len() <= 1, "{header_only:?}");
        assert!(partial_schema.len() <= 2, "{partial_schema:?}");
        assert_eq!(complete.iter().filter(|line| line.contains('━')).count(), 1);
        assert_eq!(
            complete
                .iter()
                .filter(|line| line.contains("Feature"))
                .count(),
            1
        );
        assert!(complete.iter().any(|line| line.contains("tables")));
    }

    #[test]
    fn markdown_fences_without_tables_and_unclosed_fences_stay_code() {
        for markdown in [
            "```markdown\n**bold prose**\n```\n",
            "```markdown\n| A | B |\n| --- | --- |\n",
        ] {
            let text = flat(&render(markdown, 40, &TERMINAL));
            assert!(text.iter().any(|line| line.starts_with("▏ ")), "{text:?}");
            assert!(!text.iter().any(|line| line.contains('━')), "{text:?}");
        }
    }

    #[test]
    fn fenced_table_keeps_following_pipe_prose_outside_the_grid() {
        let lines = render(
            "```markdown\n| Name | State |\n| --- | --- |\n| build | ready |\n```\nFollow-up | prose\n",
            40,
            &TERMINAL,
        );
        let text = flat(&lines);
        assert_eq!(text.iter().filter(|line| line.contains('━')).count(), 1);
        assert!(!text.iter().any(|line| line.contains('─')), "{text:?}");
        assert!(
            text.iter().any(|line| line == "Follow-up | prose"),
            "{text:?}"
        );
    }

    #[test]
    fn rerendering_after_width_changes_is_deterministic() {
        let markdown = "| Key | Notes |\n| --- | --- |\n\
                        | alpha | A sentence that wraps when the terminal narrows. |\n\
                        | beta | Another sentence with stable content. |\n";
        let wide = flat(&render(markdown, 48, &TERMINAL));
        let narrow = flat(&render(markdown, 17, &TERMINAL));
        let restored = flat(&render(markdown, 48, &TERMINAL));
        assert_eq!(restored, wide);
        assert_ne!(narrow, wide);
        assert!(narrow
            .iter()
            .all(|line| UnicodeWidthStr::width(line.as_str()) <= 17));
    }

    #[test]
    fn narrow_header_only_table_preserves_schema() {
        let lines = render(
            "| Alpha | Beta | Gamma |\n| :--- | :---: | ---: |\n",
            10,
            &TERMINAL,
        );
        let text = flat(&lines);
        let content = text.join(" ");
        assert!(content.contains("Alpha"), "{text:?}");
        assert!(content.contains("Beta"), "{text:?}");
        assert!(content.contains("Gamma"), "{text:?}");
        assert!(content.contains(":---"), "{text:?}");
        assert!(content.contains(":---:"), "{text:?}");
        assert!(content.contains("---:"), "{text:?}");
        assert!(text
            .iter()
            .all(|line| UnicodeWidthStr::width(line.as_str()) <= 10));
    }

    #[test]
    fn large_table_has_bounded_width_and_complete_output() {
        use std::fmt::Write;

        let mut markdown = String::new();
        for column in 0..10 {
            let _ = write!(markdown, "| C{column} ");
        }
        markdown.push_str("|\n");
        markdown.push_str("| --- ".repeat(10).as_str());
        markdown.push_str("|\n");
        for row in 0..1_000 {
            for column in 0..10 {
                let _ = write!(markdown, "| R{row}C{column} ");
            }
            markdown.push_str("|\n");
        }

        let lines = render(&markdown, 120, &TERMINAL);
        let text = flat(&lines);
        assert!(text.iter().any(|line| line.contains("R999C9")));
        assert_eq!(text.iter().filter(|line| line.contains('━')).count(), 1);
        assert_eq!(text.iter().filter(|line| line.contains('─')).count(), 999);
        assert!(text
            .iter()
            .all(|line| UnicodeWidthStr::width(line.as_str()) <= 120));
    }
}
