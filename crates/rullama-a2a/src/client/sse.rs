//! SSE (Server-Sent Events) stream parser.

use bytes::Bytes;
use futures::Stream;

use crate::error::A2aError;
use crate::jsonrpc::JsonRpcResponse;
use crate::streaming::StreamResponse;

/// Maximum SSE buffer size (16 MB). If a single SSE frame exceeds this,
/// the parser yields an error and stops.
const MAX_SSE_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Parse an SSE response body into a stream of `StreamResponse`.
///
/// Expects lines of the form `data: {...}\n\n` where each data line
/// contains a JSON-RPC response with a `StreamResponse` as the result.
pub fn parse_sse_stream(body: String) -> impl Stream<Item = Result<StreamResponse, A2aError>> {
    async_stream::stream! {
        for line in body.lines() {
            let line = line.trim();
            if let Some(data) = line.strip_prefix("data: ") {
                match serde_json::from_str::<JsonRpcResponse>(data) {
                    Ok(resp) => {
                        if let Some(err) = resp.error {
                            yield Err(err);
                        } else if let Some(result) = resp.result {
                            match serde_json::from_value::<StreamResponse>(result) {
                                Ok(event) => yield Ok(event),
                                Err(e) => yield Err(A2aError::from(e)),
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(A2aError::parse_error(e.to_string()));
                    }
                }
            }
        }
    }
}

/// Parse SSE from a streaming byte source, yielding events as they arrive.
pub fn parse_sse_bytes(data: Bytes) -> impl Stream<Item = Result<StreamResponse, A2aError>> {
    let text = String::from_utf8_lossy(&data).to_string();
    parse_sse_stream(text)
}

/// Parse an SSE byte stream incrementally (JSON-RPC envelope).
///
/// Reads chunks from a `reqwest::Response::bytes_stream()`, buffers until
/// complete SSE frames (`\n\n` boundaries) are found, then parses each frame.
/// Handles multi-line `data:` fields per the SSE specification.
///
/// The buffer is capped at 16 MiB to prevent unbounded memory
/// growth from malicious or misbehaving servers.
pub fn parse_sse_byte_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<StreamResponse, A2aError>> + Send {
    async_stream::stream! {
        use futures::StreamExt;
        let mut pinned = std::pin::pin!(stream);
        let mut buffer = String::new();

        while let Some(chunk) = pinned.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(A2aError::internal(format!("Stream read error: {e}")));
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            if buffer.len() > MAX_SSE_BUFFER_SIZE {
                yield Err(A2aError::internal(
                    "SSE stream buffer exceeded maximum size"
                ));
                return;
            }

            // Process complete SSE frames (delimited by \n\n)
            while let Some(boundary) = buffer.find("\n\n") {
                let frame = buffer[..boundary].to_string();
                buffer = buffer[boundary + 2..].to_string();

                yield parse_sse_frame_jsonrpc_or_error(&frame);
            }
        }

        // Process any remaining data in the buffer
        if !buffer.trim().is_empty() {
            yield parse_sse_frame_jsonrpc_or_error(&buffer);
        }
    }
}

/// Parse an SSE byte stream incrementally (raw REST — no JSON-RPC envelope).
///
/// Like `parse_sse_byte_stream` but expects `data:` lines containing raw
/// `StreamResponse` JSON rather than a JSON-RPC response wrapper.
///
/// The buffer is capped at 16 MiB.
pub fn parse_sse_rest_byte_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<StreamResponse, A2aError>> + Send {
    async_stream::stream! {
        use futures::StreamExt;
        let mut pinned = std::pin::pin!(stream);
        let mut buffer = String::new();

        while let Some(chunk) = pinned.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(A2aError::internal(format!("Stream read error: {e}")));
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            if buffer.len() > MAX_SSE_BUFFER_SIZE {
                yield Err(A2aError::internal(
                    "SSE stream buffer exceeded maximum size"
                ));
                return;
            }

            while let Some(boundary) = buffer.find("\n\n") {
                let frame = buffer[..boundary].to_string();
                buffer = buffer[boundary + 2..].to_string();

                yield parse_sse_frame_rest_or_error(&frame);
            }
        }

        if !buffer.trim().is_empty() {
            yield parse_sse_frame_rest_or_error(&buffer);
        }
    }
}

/// Extract the concatenated `data:` payload from an SSE frame.
///
/// Per the SSE spec, multiple `data:` lines are concatenated with newlines.
/// Lines starting with `:` are comments and ignored. `event:`, `id:`, and
/// `retry:` fields are ignored.
fn extract_sse_data(frame: &str) -> Option<String> {
    let mut data_parts: Vec<&str> = Vec::new();

    for line in frame.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("data:") {
            data_parts.push(value.strip_prefix(' ').unwrap_or(value));
        }
        // Ignore event:, id:, retry:, and comments (:)
    }

    if data_parts.is_empty() {
        return None;
    }

    let payload = data_parts.join("\n");
    if payload.is_empty() {
        None
    } else {
        Some(payload)
    }
}

/// Parse an SSE frame with JSON-RPC envelope, always returning a result.
///
/// Frames without `data:` lines yield a parse error instead of being silently dropped.
fn parse_sse_frame_jsonrpc_or_error(frame: &str) -> Result<StreamResponse, A2aError> {
    let data = match extract_sse_data(frame) {
        Some(d) => d,
        None => return Err(A2aError::parse_error("SSE frame contains no data field")),
    };

    match serde_json::from_str::<JsonRpcResponse>(&data) {
        Ok(resp) => {
            if let Some(err) = resp.error {
                Err(err)
            } else if let Some(result) = resp.result {
                serde_json::from_value::<StreamResponse>(result).map_err(A2aError::from)
            } else {
                Err(A2aError::parse_error(
                    "JSON-RPC response has neither result nor error",
                ))
            }
        }
        Err(e) => Err(A2aError::parse_error(e.to_string())),
    }
}

/// Parse an SSE frame with raw StreamResponse JSON, always returning a result.
///
/// Frames without `data:` lines yield a parse error instead of being silently dropped.
fn parse_sse_frame_rest_or_error(frame: &str) -> Result<StreamResponse, A2aError> {
    let data = match extract_sse_data(frame) {
        Some(d) => d,
        None => return Err(A2aError::parse_error("SSE frame contains no data field")),
    };

    serde_json::from_str::<StreamResponse>(&data).map_err(|e| A2aError::parse_error(e.to_string()))
}
