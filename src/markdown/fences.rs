use std::borrow::Cow;

/// Remove complete Markdown fences when their body contains an actual table.
///
/// Models sometimes wrap Markdown tables in a fenced block. In that form the
/// Markdown parser sees code instead of a table. This pre-pass only removes a
/// fence when all of the following are true:
///
/// - the first info-string token is `md` or `markdown` (case-insensitive);
/// - a matching closing fence is present;
/// - non-blank body lines stay at the fence's blockquote depth; and
/// - two adjacent body lines form a table header and delimiter with the same
///   number of columns.
///
/// Everything else is returned byte-for-byte unchanged. The common no-fence
/// path does not allocate.
pub(super) fn unwrap_table_markdown_fences(source: &str) -> Cow<'_, str> {
    if !source.contains("```") && !source.contains("~~~") {
        return Cow::Borrowed(source);
    }

    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let mut output = String::new();
    let mut copied_through = 0usize;
    let mut line_index = 0usize;
    let mut changed = false;

    while line_index < lines.len() {
        let Some(opening) = parse_opening_fence(line_body(lines[line_index])) else {
            line_index += 1;
            continue;
        };

        let closing_index = (line_index + 1..lines.len())
            .find(|&candidate| is_matching_close(line_body(lines[candidate]), opening));
        let Some(closing_index) = closing_index else {
            // An unclosed fence owns the remainder of the input, so a fence-like
            // line inside its body must not be normalized independently.
            break;
        };

        if opening.is_markdown
            && body_uses_container(&lines[line_index + 1..closing_index], opening.quote_depth)
            && body_contains_table(&lines[line_index + 1..closing_index], opening.quote_depth)
        {
            output.extend(lines[copied_through..line_index].iter().copied());
            output.extend(lines[line_index + 1..closing_index].iter().copied());
            append_block_boundary(
                &mut output,
                &lines[line_index + 1..closing_index],
                lines[closing_index],
                lines.get(closing_index + 1).copied(),
                opening.quote_depth,
            );
            copied_through = closing_index + 1;
            changed = true;
        }

        // Skip every complete fenced block, including non-Markdown blocks. A
        // marker-looking line inside code is content, not another outer fence.
        line_index = closing_index + 1;
    }

    if !changed {
        return Cow::Borrowed(source);
    }

    output.extend(lines[copied_through..].iter().copied());
    Cow::Owned(output)
}

#[derive(Clone, Copy)]
struct Fence {
    marker: u8,
    marker_len: usize,
    quote_depth: usize,
    is_markdown: bool,
}

fn parse_opening_fence(line: &str) -> Option<Fence> {
    let (quote_depth, payload) = split_container_prefix(line);
    let marker = *payload.as_bytes().first()?;
    if !matches!(marker, b'`' | b'~') {
        return None;
    }

    let marker_len = payload.bytes().take_while(|byte| *byte == marker).count();
    if marker_len < 3 {
        return None;
    }

    let info_tail = &payload[marker_len..];
    if marker == b'`' && info_tail.as_bytes().contains(&b'`') {
        return None;
    }
    let info = info_tail.split_whitespace().next().unwrap_or_default();

    Some(Fence {
        marker,
        marker_len,
        quote_depth,
        is_markdown: info.eq_ignore_ascii_case("md") || info.eq_ignore_ascii_case("markdown"),
    })
}

fn is_matching_close(line: &str, opening: Fence) -> bool {
    let (quote_depth, payload) = split_container_prefix(line);
    if quote_depth != opening.quote_depth {
        return false;
    }
    let run = payload
        .bytes()
        .take_while(|byte| *byte == opening.marker)
        .count();
    run >= opening.marker_len && payload[run..].trim().is_empty()
}

/// Separate a Markdown blockquote container from its content. Up to three
/// indentation spaces are accepted at each content boundary. Quote-prefix
/// comparison deliberately uses structural depth rather than whitespace, so
/// `> | A |` and `>| A |` belong to the same container.
fn split_container_prefix(line: &str) -> (usize, &str) {
    let bytes = line.as_bytes();
    let mut cursor = consume_spaces(bytes, 0, 3);
    let mut quote_depth = 0usize;

    while bytes.get(cursor) == Some(&b'>') {
        quote_depth += 1;
        cursor += 1;
        if matches!(bytes.get(cursor), Some(b' ' | b'\t')) {
            cursor += 1;
        }

        let after_indent = consume_spaces(bytes, cursor, 3);
        if bytes.get(after_indent) == Some(&b'>') {
            cursor = after_indent;
        } else {
            cursor = after_indent;
            break;
        }
    }

    (quote_depth, &line[cursor..])
}

fn consume_spaces(bytes: &[u8], mut cursor: usize, limit: usize) -> usize {
    let end = cursor.saturating_add(limit).min(bytes.len());
    while cursor < end && bytes[cursor] == b' ' {
        cursor += 1;
    }
    cursor
}

fn body_uses_container(lines: &[&str], quote_depth: usize) -> bool {
    lines.iter().all(|line| {
        let body = line_body(line);
        if body.trim().is_empty() {
            return true;
        }
        split_container_prefix(body).0 == quote_depth
    })
}

fn body_contains_table(lines: &[&str], quote_depth: usize) -> bool {
    lines.windows(2).any(|pair| {
        let Some(header) = table_payload(pair[0], quote_depth) else {
            return false;
        };
        let Some(delimiter) = table_payload(pair[1], quote_depth) else {
            return false;
        };

        let Some(header_cells) = row_cells(header) else {
            return false;
        };
        let Some(delimiter_count) = delimiter_cells(delimiter) else {
            return false;
        };

        header_cells.len() == delimiter_count
            && header_cells.iter().any(|cell| !cell.trim().is_empty())
            && delimiter_cells(header).is_none()
    })
}

/// A closing fence is a block boundary in the original Markdown. When it is
/// removed between a table and later content, retain an empty line so a
/// pipe-containing prose line cannot become another body row. Reuse the
/// closing line's container prefix and newline convention.
fn append_block_boundary(
    output: &mut String,
    body_lines: &[&str],
    closing_line: &str,
    following_line: Option<&str>,
    quote_depth: usize,
) {
    let Some(following_line) = following_line else {
        return;
    };
    if body_lines
        .last()
        .is_some_and(|line| is_blank_at_depth(line, quote_depth))
        || is_blank_at_depth(following_line, quote_depth)
    {
        return;
    }

    let closing_body = line_body(closing_line);
    let (_, marker) = split_container_prefix(closing_body);
    let prefix_len = closing_body.len().saturating_sub(marker.len());
    output.push_str(&closing_body[..prefix_len]);
    output.push_str(line_ending(closing_line));
}

fn is_blank_at_depth(line: &str, quote_depth: usize) -> bool {
    let body = line_body(line);
    if body.trim().is_empty() {
        return true;
    }
    let (actual_depth, payload) = split_container_prefix(body);
    actual_depth == quote_depth && payload.trim().is_empty()
}

fn table_payload(line: &str, quote_depth: usize) -> Option<&str> {
    let (actual_depth, payload) = split_container_prefix(line_body(line));
    if actual_depth != quote_depth || payload.starts_with([' ', '\t']) {
        return None;
    }
    Some(payload.trim_end())
}

fn delimiter_cells(line: &str) -> Option<usize> {
    let cells = row_cells(line)?;
    if cells.is_empty() || !cells.iter().all(|cell| is_delimiter_cell(cell)) {
        return None;
    }
    Some(cells.len())
}

fn is_delimiter_cell(cell: &str) -> bool {
    let mut value = cell.trim();
    if let Some(rest) = value.strip_prefix(':') {
        value = rest;
    }
    if let Some(rest) = value.strip_suffix(':') {
        value = rest;
    }
    value.len() >= 3 && value.bytes().all(|byte| byte == b'-')
}

/// Split on pipes that are not escaped by an odd run of backslashes. Empty
/// fields introduced by optional outer pipes are removed; interior empty
/// fields remain columns.
fn row_cells(line: &str) -> Option<Vec<&str>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    let mut pipes = Vec::new();
    let mut preceding_backslashes = 0usize;
    for (offset, character) in line.char_indices() {
        if character == '\\' {
            preceding_backslashes += 1;
            continue;
        }
        if character == '|' && preceding_backslashes % 2 == 0 {
            pipes.push(offset);
        }
        preceding_backslashes = 0;
    }
    if pipes.is_empty() {
        return None;
    }

    let leading_pipe = pipes.first() == Some(&0);
    let trailing_pipe = pipes.last().is_some_and(|offset| *offset + 1 == line.len());
    let mut cells = Vec::with_capacity(pipes.len() + 1);
    let mut start = 0usize;
    for pipe in pipes {
        cells.push(&line[start..pipe]);
        start = pipe + 1;
    }
    cells.push(&line[start..]);

    if trailing_pipe {
        cells.pop();
    }
    if leading_pipe && !cells.is_empty() {
        cells.remove(0);
    }
    (!cells.is_empty()).then_some(cells)
}

fn line_body(line: &str) -> &str {
    let without_lf = line.strip_suffix('\n').unwrap_or(line);
    without_lf.strip_suffix('\r').unwrap_or(without_lf)
}

fn line_ending(line: &str) -> &'static str {
    if line.ends_with("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(source: &str) -> String {
        unwrap_table_markdown_fences(source).into_owned()
    }

    #[test]
    fn borrows_input_when_no_fence_changes() {
        let source = "| A | B |\n| --- | --- |\n| 1 | 2 |\n";
        assert!(matches!(
            unwrap_table_markdown_fences(source),
            Cow::Borrowed(value) if value == source
        ));
    }

    #[test]
    fn unwraps_backtick_table_with_outer_pipes() {
        let source = "before\n```markdown\n| A | B |\n| :--- | ---: |\n| 1 | 2 |\n```\nafter\n";
        assert_eq!(
            normalized(source),
            "before\n| A | B |\n| :--- | ---: |\n| 1 | 2 |\n\nafter\n"
        );
    }

    #[test]
    fn keeps_table_boundary_before_pipe_containing_prose() {
        let source =
            "```md\n| Key | Value |\n| --- | --- |\n| one | two |\n```\nFollow-up | prose\n";
        assert_eq!(
            normalized(source),
            "| Key | Value |\n| --- | --- |\n| one | two |\n\nFollow-up | prose\n"
        );
    }

    #[test]
    fn keeps_table_boundary_inside_matching_blockquote() {
        let source = "> ```md\n> | Key | Value |\n> | --- | --- |\n> | one | two |\n> ```\n> Follow-up | prose\n";
        assert_eq!(
            normalized(source),
            "> | Key | Value |\n> | --- | --- |\n> | one | two |\n> \n> Follow-up | prose\n"
        );
    }

    #[test]
    fn does_not_duplicate_an_existing_blank_boundary() {
        let source = "```md\n| A | B |\n| --- | --- |\n```\n\nafter\n";
        assert_eq!(normalized(source), "| A | B |\n| --- | --- |\n\nafter\n");
    }

    #[test]
    fn unwraps_tilde_table_without_outer_pipes() {
        let source = "~~~ md\nName | Value\n--- | :---:\none | two\n~~~~\n";
        assert_eq!(normalized(source), "Name | Value\n--- | :---:\none | two\n");
    }

    #[test]
    fn accepts_case_insensitive_markdown_info_with_metadata() {
        let source = "```Markdown preview\n| A | B |\n| --- | --- |\n```\n";
        assert_eq!(normalized(source), "| A | B |\n| --- | --- |\n");
    }

    #[test]
    fn unwraps_single_column_table() {
        let source = "```md\n| Status |\n| ---: |\n| ready |\n```";
        assert_eq!(normalized(source), "| Status |\n| ---: |\n| ready |\n");
    }

    #[test]
    fn unwraps_matching_nested_blockquote() {
        let source = "> > ```markdown\n>> | A | B |\n> > | --- | --- |\n> > | 1 | 2 |\n> > ```\n";
        assert_eq!(
            normalized(source),
            ">> | A | B |\n> > | --- | --- |\n> > | 1 | 2 |\n"
        );
    }

    #[test]
    fn escaped_pipe_does_not_create_an_extra_column() {
        let source = "```md\nA \\| literal | B\n--- | ---\nx | y\n```\n";
        assert_eq!(normalized(source), "A \\| literal | B\n--- | ---\nx | y\n");
    }

    #[test]
    fn leaves_non_markdown_fence_untouched() {
        let source = "```text\n| A | B |\n| --- | --- |\n```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_invalid_backtick_info_string_untouched() {
        let source = "```md `bad`\n| A | B |\n| --- | --- |\n```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_markdown_fence_without_table_untouched() {
        let source = "```markdown\n# Heading\n\nSome **prose**.\n```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_setext_heading_untouched() {
        let source = "```markdown\nHeading\n---\n```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_blank_separated_header_and_delimiter_untouched() {
        let source = "```md\n| A | B |\n\n| --- | --- |\n```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_mismatched_blockquote_table_untouched() {
        let source = "> ```md\n> | A | B |\n| --- | --- |\n> ```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_mismatched_blockquote_close_untouched() {
        let source = "> ```md\n> | A | B |\n> | --- | --- |\n```\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn leaves_unclosed_fence_and_nested_markers_untouched() {
        let source = "```md\n| A | B |\n| --- | --- |\n~~~ markdown\nX\n---\n~~~\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn does_not_normalize_marker_text_inside_other_code_fence() {
        let source = "````text\n```md\n| A | B |\n| --- | --- |\n```\n````\n";
        assert_eq!(normalized(source), source);
    }

    #[test]
    fn preserves_crlf_and_normalizes_only_qualifying_fences() {
        let source = "```md\r\n| A |\r\n| --- |\r\n| x |\r\n```\r\n~~~rust\r\n---\r\n~~~\r\n";
        assert_eq!(
            normalized(source),
            "| A |\r\n| --- |\r\n| x |\r\n\r\n~~~rust\r\n---\r\n~~~\r\n"
        );
    }

    #[test]
    fn rejects_invalid_or_mismatched_delimiters() {
        for source in [
            "```md\nA | B\n-- | ---\n```\n",
            "```md\nA | B\n---\n```\n",
            "```md\nA | B\n--- | -x-\n```\n",
        ] {
            assert_eq!(normalized(source), source);
        }
    }
}
