use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Render markdown to styled, word-wrapped lines for the transcript pane.
/// Tolerates incomplete input (e.g. an unclosed code fence mid-stream).
pub fn render(text: &str, width: usize, base: Style) -> Vec<Line<'static>> {
    let mut r = Renderer::new(width.max(10), base);
    for event in Parser::new_ext(text, Options::ENABLE_STRIKETHROUGH) {
        r.event(event);
    }
    r.finish()
}

struct Renderer {
    width: usize,
    base: Style,
    lines: Vec<Line<'static>>,
    cur: Vec<Span<'static>>,
    cur_w: usize,
    first_prefix: String,
    cont_prefix: String,
    first_line_of_block: bool,
    bold: u32,
    italic: u32,
    strike: u32,
    heading: bool,
    code_block: Option<String>,
    list_stack: Vec<Option<u64>>,
    in_item: u32,
    quote_depth: usize,
    link: Option<(String, bool)>, // (url, text matched url i.e. autolink)
}

impl Renderer {
    fn new(width: usize, base: Style) -> Self {
        Renderer {
            width,
            base,
            lines: Vec::new(),
            cur: Vec::new(),
            cur_w: 0,
            first_prefix: String::new(),
            cont_prefix: String::new(),
            first_line_of_block: true,
            bold: 0,
            italic: 0,
            strike: 0,
            heading: false,
            code_block: None,
            list_stack: Vec::new(),
            in_item: 0,
            quote_depth: 0,
            link: None,
        }
    }

    fn event(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start(tag),
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
                let style = self.base.fg(Color::Yellow);
                self.push_token(&t, style);
            }
            Event::SoftBreak => {
                let style = self.inline_style();
                self.push_words(" ", style);
            }
            Event::HardBreak => self.flush_line(),
            Event::Rule => {
                self.gap();
                self.lines.push(Line::styled(
                    "─".repeat(self.width),
                    self.base.add_modifier(Modifier::DIM),
                ));
            }
            Event::TaskListMarker(done) => {
                let style = self.inline_style();
                self.push_token(if done { "[x] " } else { "[ ] " }, style);
            }
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag) {
        match tag {
            Tag::Paragraph => {
                if self.in_item == 0 {
                    self.gap();
                    let q = self.quote_prefix();
                    self.set_block(q.clone(), q);
                }
            }
            Tag::Heading { .. } => {
                self.gap();
                let q = self.quote_prefix();
                self.set_block(q.clone(), q);
                self.heading = true;
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
                self.set_block(format!("{indent}{bullet}"), cont);
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
            TagEnd::Paragraph | TagEnd::Heading(_) => {
                self.flush_line();
                self.heading = false;
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
                        let style = self.base.add_modifier(Modifier::DIM);
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
        let mut style = if self.heading {
            self.base.fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            self.base
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

    fn set_block(&mut self, first: String, cont: String) {
        self.first_prefix = first;
        self.cont_prefix = cont;
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
            .saturating_sub(self.prefix().chars().count())
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
            spans.push(Span::styled(prefix, self.base.add_modifier(Modifier::DIM)));
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
    fn push_token(&mut self, token: &str, style: Style) {
        let tw = token.chars().count();
        if self.cur_w + tw > self.avail() && self.cur_w > 0 {
            self.flush_line();
            if token.trim().is_empty() {
                return; // don't carry the wrapping space to the next line
            }
        }
        if tw > self.avail() {
            let mut rest: Vec<char> = token.chars().collect();
            while rest.len() > self.avail() {
                let take = self.avail();
                let piece: String = rest.drain(..take).collect();
                self.cur_w += take;
                self.cur.push(Span::styled(piece, style));
                self.flush_line();
            }
            if !rest.is_empty() {
                self.cur_w += rest.len();
                self.cur
                    .push(Span::styled(rest.into_iter().collect::<String>(), style));
            }
        } else {
            self.cur_w += tw;
            self.cur.push(Span::styled(token.to_owned(), style));
        }
    }

    fn emit_code_block(&mut self, buf: &str) {
        let style = self.base.fg(Color::Yellow);
        let border = self.base.add_modifier(Modifier::DIM);
        let width = self.width.saturating_sub(2).max(4);
        for line in buf.lines() {
            // Code is never word-wrapped; hard-cut to keep indentation intact.
            let cut: String = line.chars().take(width).collect();
            self.lines.push(Line::from(vec![
                Span::styled("▏ ", border),
                Span::styled(cut, style),
            ]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let lines = render("alpha beta gamma delta", 12, Style::default());
        let text = flat(&lines);
        assert!(text.len() > 1, "expected wrapping, got {text:?}");
        assert!(text.iter().all(|l| l.chars().count() <= 12), "{text:?}");
        assert!(!text.iter().any(|l| l.ends_with(' ')), "{text:?}");
    }

    #[test]
    fn renders_code_blocks_verbatim() {
        let lines = render(
            "text\n\n```rust\nfn main() {}\n    indented\n```",
            40,
            Style::default(),
        );
        let text = flat(&lines);
        assert!(text.contains(&"▏ fn main() {}".to_string()), "{text:?}");
        assert!(text.contains(&"▏     indented".to_string()), "{text:?}");
    }

    #[test]
    fn renders_lists_with_bullets_and_numbers() {
        let lines = render("- one\n- two\n\n1. first\n2. second", 40, Style::default());
        let text = flat(&lines);
        assert!(text.contains(&"• one".to_string()), "{text:?}");
        assert!(text.contains(&"1. first".to_string()), "{text:?}");
        assert!(text.contains(&"2. second".to_string()), "{text:?}");
    }

    #[test]
    fn tolerates_unclosed_fence_mid_stream() {
        let lines = render("start\n\n```py\nprint(1)", 40, Style::default());
        let text = flat(&lines);
        assert!(text.contains(&"▏ print(1)".to_string()), "{text:?}");
    }

    #[test]
    fn inline_code_and_bold_do_not_panic_and_keep_text() {
        let lines = render("use `cargo build` to **compile** it", 80, Style::default());
        let text = flat(&lines).join(" ");
        assert!(text.contains("cargo build"));
        assert!(text.contains("compile"));
    }
}
