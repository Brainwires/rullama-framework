//! SSE (Server-Sent Events) detection and streaming passthrough.

use bytes::Bytes;

/// A single SSE event parsed from a stream.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SseEvent {
    /// Event type (from `event:` field). None if not specified.
    pub event: Option<String>,
    /// Event data (from `data:` field(s), joined with newlines).
    pub data: String,
    /// Event ID (from `id:` field).
    pub id: Option<String>,
    /// Retry interval in milliseconds (from `retry:` field).
    pub retry: Option<u64>,
}

/// Check if HTTP headers indicate an SSE response.
pub fn is_sse_response(headers: &http::HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"))
}

/// Parse SSE events from a byte buffer.
/// Returns parsed events and any remaining unparsed bytes.
pub fn parse_sse_chunk(buf: &[u8]) -> (Vec<SseEvent>, Bytes) {
    let text = String::from_utf8_lossy(buf);
    let mut events = Vec::new();
    let mut current_event: Option<String> = None;
    let mut current_data = Vec::new();
    let mut current_id: Option<String> = None;
    let mut current_retry: Option<u64> = None;
    let mut last_complete = 0;

    for (i, line) in text.split('\n').enumerate() {
        let line = line.trim_end_matches('\r');

        if line.is_empty() {
            // Empty line = event boundary
            if !current_data.is_empty() {
                events.push(SseEvent {
                    event: current_event.take(),
                    data: current_data.join("\n"),
                    id: current_id.take(),
                    retry: current_retry.take(),
                });
                current_data.clear();
            }
            // Track position for remainder calculation
            last_complete = text.split('\n').take(i + 1).map(|l| l.len() + 1).sum();
        } else if let Some(value) = line.strip_prefix("data:") {
            current_data.push(value.trim_start().to_string());
        } else if let Some(value) = line.strip_prefix("event:") {
            current_event = Some(value.trim_start().to_string());
        } else if let Some(value) = line.strip_prefix("id:") {
            current_id = Some(value.trim_start().to_string());
        } else if let Some(value) = line.strip_prefix("retry:") {
            current_retry = value.trim_start().parse().ok();
        }
        // Lines starting with ':' are comments — ignored
    }

    let remainder = if last_complete < buf.len() {
        Bytes::copy_from_slice(&buf[last_complete..])
    } else {
        Bytes::new()
    };

    (events, remainder)
}

/// Check if HTTP headers indicate an SSE request (Accept: text/event-stream).
pub fn is_sse_request(headers: &http::HeaderMap) -> bool {
    headers
        .get(http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"))
}

/// Serialize an SSE event back to wire format.
pub fn serialize_sse_event(event: &SseEvent) -> Bytes {
    let mut out = String::new();
    if let Some(ref ev) = event.event {
        out.push_str(&format!("event: {ev}\n"));
    }
    for line in event.data.split('\n') {
        out.push_str(&format!("data: {line}\n"));
    }
    if let Some(ref id) = event.id {
        out.push_str(&format!("id: {id}\n"));
    }
    if let Some(retry) = event.retry {
        out.push_str(&format!("retry: {retry}\n"));
    }
    out.push('\n');
    Bytes::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::{HeaderMap, HeaderValue, header};

    #[test]
    fn detect_sse_response() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        assert!(is_sse_response(&headers));

        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        assert!(!is_sse_response(&headers));

        let empty = HeaderMap::new();
        assert!(!is_sse_response(&empty));
    }

    #[test]
    fn parse_single_event() {
        let input = b"data: hello world\n\n";
        let (events, remainder) = parse_sse_chunk(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello world");
        assert!(events[0].event.is_none());
        assert!(remainder.is_empty());
    }

    #[test]
    fn parse_event_with_type_and_id() {
        let input = b"event: message\nid: 42\ndata: payload\n\n";
        let (events, _) = parse_sse_chunk(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("message"));
        assert_eq!(events[0].id.as_deref(), Some("42"));
        assert_eq!(events[0].data, "payload");
    }

    #[test]
    fn parse_multiline_data() {
        let input = b"data: line1\ndata: line2\ndata: line3\n\n";
        let (events, _) = parse_sse_chunk(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn parse_multiple_events() {
        let input = b"data: first\n\ndata: second\n\n";
        let (events, _) = parse_sse_chunk(input);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn parse_with_retry() {
        let input = b"retry: 3000\ndata: reconnect\n\n";
        let (events, _) = parse_sse_chunk(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry, Some(3000));
    }

    #[test]
    fn incomplete_event_goes_to_remainder() {
        let input = b"data: complete\n\ndata: partial";
        let (events, remainder) = parse_sse_chunk(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "complete");
        assert!(!remainder.is_empty());
    }

    #[test]
    fn serialize_roundtrip() {
        let event = SseEvent {
            event: Some("update".into()),
            data: "hello".into(),
            id: Some("1".into()),
            retry: Some(5000),
        };

        let bytes = serialize_sse_event(&event);
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("event: update\n"));
        assert!(text.contains("data: hello\n"));
        assert!(text.contains("id: 1\n"));
        assert!(text.contains("retry: 5000\n"));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn serialize_multiline_data() {
        let event = SseEvent {
            event: None,
            data: "line1\nline2".into(),
            id: None,
            retry: None,
        };

        let bytes = serialize_sse_event(&event);
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("data: line1\n"));
        assert!(text.contains("data: line2\n"));
    }
}
