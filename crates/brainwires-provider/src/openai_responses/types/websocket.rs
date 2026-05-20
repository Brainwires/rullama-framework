//! WebSocket-specific message types for the OpenAI Responses API.

use serde::Serialize;

use super::request::CreateResponseRequest;

/// WebSocket request envelope for `response.create`.
///
/// Wraps a `CreateResponseRequest` with the required `"type": "response.create"` field
/// for the WebSocket protocol. The `stream` field is omitted since WebSocket mode is
/// inherently streaming.
#[derive(Debug, Clone, Serialize)]
pub struct WsResponseCreate {
    /// Always `"response.create"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Flattened request fields.
    #[serde(flatten)]
    pub request: CreateResponseRequest,
}

impl WsResponseCreate {
    /// Create a new WebSocket request from an existing `CreateResponseRequest`.
    pub fn new(mut request: CreateResponseRequest) -> Self {
        // WebSocket mode is inherently streaming — clear the stream field
        request.stream = None;
        Self {
            kind: "response.create".to_string(),
            request,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_responses::types::request::ResponseInput;

    #[test]
    fn test_ws_request_type_field() {
        let req = CreateResponseRequest::new("gpt-4o", ResponseInput::Text("Hi".to_string()));
        let ws = WsResponseCreate::new(req);
        let json = serde_json::to_value(&ws).unwrap();
        assert_eq!(json["type"], "response.create");
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["input"], "Hi");
    }

    #[test]
    fn test_ws_request_strips_stream() {
        let mut req = CreateResponseRequest::new("gpt-4o", ResponseInput::Text("Hi".to_string()));
        req.stream = Some(true);
        let ws = WsResponseCreate::new(req);
        let json = serde_json::to_value(&ws).unwrap();
        // stream should be absent (None is skipped)
        assert!(json.get("stream").is_none());
    }
}
