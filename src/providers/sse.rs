use anyhow::Result;
use futures_util::StreamExt;

/// Drive a Server-Sent-Events response body, invoking `on_data` for each
/// `data:` payload. Handles events split across network chunks.
pub async fn for_each_data(
    response: reqwest::Response,
    mut on_data: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim_end_matches('\r').to_owned();
            buf.drain(..=pos);
            if let Some(data) = line.strip_prefix("data:") {
                on_data(data.trim_start())?;
            }
        }
    }
    Ok(())
}

/// Drive a newline-delimited JSON (NDJSON) response body, as used by Ollama.
pub async fn for_each_ndjson(
    response: reqwest::Response,
    mut on_line: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_owned();
            buf.drain(..=pos);
            if !line.is_empty() {
                on_line(&line)?;
            }
        }
    }
    if !buf.trim().is_empty() {
        on_line(buf.trim())?;
    }
    Ok(())
}

/// Common pre-flight: fail with the response body when the API returns an error.
pub async fn check_status(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    anyhow::bail!("API error {status}: {body}")
}
