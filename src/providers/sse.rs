use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::Value;

const MAX_STREAM_RECORD_BYTES: usize = 1_024 * 1_024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1_024;
const MAX_JSON_BODY_BYTES: usize = 8 * 1_024 * 1_024;

struct RecordBuffer {
    bytes: Vec<u8>,
    label: &'static str,
}

impl RecordBuffer {
    fn new(label: &'static str) -> Self {
        Self {
            bytes: Vec::new(),
            label,
        }
    }

    fn push(
        &mut self,
        mut chunk: &[u8],
        mut on_record: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        while let Some(newline) = chunk.iter().position(|byte| *byte == b'\n') {
            self.extend(&chunk[..newline])?;
            on_record(&self.bytes)?;
            self.bytes.clear();
            chunk = &chunk[newline + 1..];
        }
        self.extend(chunk)
    }

    fn extend(&mut self, fragment: &[u8]) -> Result<()> {
        let record_bytes = self
            .bytes
            .len()
            .checked_add(fragment.len())
            .context("stream record size overflow")?;
        if record_bytes > MAX_STREAM_RECORD_BYTES {
            anyhow::bail!(
                "{} exceeds the {}-byte record limit",
                self.label,
                MAX_STREAM_RECORD_BYTES
            );
        }
        self.bytes.extend_from_slice(fragment);
        Ok(())
    }

    fn finish(self, mut on_record: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        if !self.bytes.is_empty() {
            on_record(&self.bytes)?;
        }
        Ok(())
    }
}

struct BoundedBody {
    bytes: Vec<u8>,
    max_bytes: usize,
    label: &'static str,
}

impl BoundedBody {
    fn new(max_bytes: usize, label: &'static str) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
            label,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Result<()> {
        let body_bytes = self
            .bytes
            .len()
            .checked_add(chunk.len())
            .context("response body size overflow")?;
        if body_bytes > self.max_bytes {
            anyhow::bail!(
                "{} exceeds the {}-byte body limit",
                self.label,
                self.max_bytes
            );
        }
        self.bytes.extend_from_slice(chunk);
        Ok(())
    }
}

/// Drive a Server-Sent-Events response body, invoking `on_data` for each
/// `data:` payload. Handles records and UTF-8 code points split across network
/// chunks while bounding each record before allocation can grow indefinitely.
pub async fn for_each_data(
    response: reqwest::Response,
    mut on_data: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut records = RecordBuffer::new("SSE event");

    while let Some(chunk) = stream.next().await {
        records.push(&chunk?, |record| {
            let line = std::str::from_utf8(record)
                .context("SSE event contains invalid UTF-8")?
                .trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data:") {
                on_data(data.trim_start())?;
            }
            Ok(())
        })?;
    }
    records.finish(|record| {
        let line = std::str::from_utf8(record)
            .context("SSE event contains invalid UTF-8")?
            .trim_end_matches('\r');
        if let Some(data) = line.strip_prefix("data:") {
            on_data(data.trim_start())?;
        }
        Ok(())
    })
}

/// Drive a newline-delimited JSON (NDJSON) response body, as used by Ollama.
pub async fn for_each_ndjson(
    response: reqwest::Response,
    mut on_line: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut records = RecordBuffer::new("NDJSON record");

    while let Some(chunk) = stream.next().await {
        records.push(&chunk?, |record| {
            let line = std::str::from_utf8(record)
                .context("NDJSON record contains invalid UTF-8")?
                .trim();
            if !line.is_empty() {
                on_line(line)?;
            }
            Ok(())
        })?;
    }
    records.finish(|record| {
        let line = std::str::from_utf8(record)
            .context("NDJSON record contains invalid UTF-8")?
            .trim();
        if !line.is_empty() {
            on_line(line)?;
        }
        Ok(())
    })
}

/// Common pre-flight: fail with a bounded response body when the API returns
/// an error.
pub async fn check_status(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = read_error_body(response)
        .await
        .with_context(|| format!("API error {status} body could not be read"))?;
    anyhow::bail!("API error {status}: {body}")
}

/// Read one API error body without allowing an endpoint to grow memory without
/// bound. API errors are expected to be UTF-8 text; malformed encodings fail
/// explicitly instead of being silently replaced.
pub async fn read_error_body(response: reqwest::Response) -> Result<String> {
    let bytes = read_response_bounded(response, MAX_ERROR_BODY_BYTES, "API error response").await?;
    String::from_utf8(bytes).context("API error response contains invalid UTF-8")
}

/// Read and parse a bounded JSON API response. Catalog endpoints are remote
/// input too, even when the caller later publishes only a small subset.
pub async fn read_json_response(response: reqwest::Response) -> Result<Value> {
    let bytes = read_response_bounded(response, MAX_JSON_BODY_BYTES, "API JSON response").await?;
    parse_json_body(&bytes)
}

async fn read_response_bounded(
    response: reqwest::Response,
    max_bytes: usize,
    label: &'static str,
) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|content_length| content_length > max_bytes as u64)
    {
        anyhow::bail!("{label} exceeds the {max_bytes}-byte body limit");
    }
    let mut stream = response.bytes_stream();
    let mut body = BoundedBody::new(max_bytes, label);
    while let Some(chunk) = stream.next().await {
        body.push(&chunk?)?;
    }
    Ok(body.bytes)
}

fn parse_json_body(bytes: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(bytes).context("API JSON response contains invalid UTF-8")?;
    serde_json::from_str(text).context("API returned malformed JSON")
}

/// Send a request, retrying transient failures (429, 5xx, network errors)
/// with backoff. Honors `Retry-After` when present. Only the initial send is
/// retried — an interrupted stream is surfaced to the caller.
pub async fn send_retrying(request: reqwest::RequestBuilder) -> Result<reqwest::Response> {
    const RETRYABLE: [u16; 5] = [429, 500, 502, 503, 529];
    let mut delay = 1u64;

    for _ in 0..2 {
        let Some(cloned) = request.try_clone() else {
            return Ok(request.send().await?);
        };
        match cloned.send().await {
            Ok(response) if !RETRYABLE.contains(&response.status().as_u16()) => {
                return Ok(response);
            }
            Ok(response) => {
                let wait = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(delay)
                    .min(30);
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
            }
            Err(_) => tokio::time::sleep(std::time::Duration::from_secs(delay)).await,
        }
        delay *= 4;
    }
    Ok(request.send().await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_buffer_reassembles_split_utf8_and_resets_per_line() {
        let text = "data: snowman ☃\nnext\n".as_bytes();
        let snowman = text
            .windows(3)
            .position(|window| window == "☃".as_bytes())
            .expect("snowman bytes");
        let mut buffer = RecordBuffer::new("test record");
        let mut records = Vec::new();

        buffer
            .push(&text[..snowman + 1], |_| {
                anyhow::bail!("an incomplete line must not be emitted")
            })
            .unwrap();
        buffer
            .push(&text[snowman + 1..], |record| {
                records.push(std::str::from_utf8(record)?.to_owned());
                Ok(())
            })
            .unwrap();

        assert_eq!(records, ["data: snowman ☃", "next"]);
    }

    #[test]
    fn newline_free_stream_record_is_bounded_before_append() {
        let mut buffer = RecordBuffer::new("SSE event");
        buffer
            .push(&vec![b'x'; MAX_STREAM_RECORD_BYTES], |_| Ok(()))
            .unwrap();
        let error = buffer
            .push(b"x", |_| Ok(()))
            .expect_err("oversized record must fail");
        assert!(error.to_string().contains("SSE event exceeds"));
        assert_eq!(buffer.bytes.len(), MAX_STREAM_RECORD_BYTES);
    }

    #[test]
    fn completed_stream_record_rejects_invalid_utf8_clearly() {
        let mut buffer = RecordBuffer::new("NDJSON record");
        let error = buffer
            .push(&[0xff, b'\n'], |record| {
                std::str::from_utf8(record)
                    .context("NDJSON record contains invalid UTF-8")
                    .map(|_| ())
            })
            .expect_err("invalid UTF-8 must fail");
        assert!(error.to_string().contains("invalid UTF-8"));
    }

    #[test]
    fn bounded_body_accepts_the_limit_and_rejects_the_next_byte() {
        let mut body = BoundedBody::new(4, "test response");
        body.push(b"12").unwrap();
        body.push(b"34").unwrap();
        let error = body.push(b"5").expect_err("oversized body must fail");
        assert!(error.to_string().contains("4-byte body limit"));
        assert_eq!(body.bytes, b"1234");
    }

    #[test]
    fn bounded_json_parser_preserves_utf8_and_rejects_invalid_encoding() {
        assert_eq!(
            parse_json_body(r#"{"message":"snowman ☃"}"#.as_bytes()).unwrap(),
            json!({"message": "snowman ☃"})
        );
        let error = parse_json_body(b"{\"message\":\"\xFF\"}")
            .expect_err("invalid UTF-8 must fail clearly");
        assert!(error.to_string().contains("invalid UTF-8"));
    }
}
