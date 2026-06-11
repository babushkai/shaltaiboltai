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
