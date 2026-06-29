//! Response object types for the Responses API.

use serde::{Deserialize, Serialize};

use super::output::ResponseOutputItem;

/// Full response object returned by the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObject {
    /// Response ID (e.g. "resp_abc123").
    pub id: String,
    /// Object type: "response".
    #[serde(default)]
    pub object: Option<String>,
    /// Creation timestamp.
    #[serde(default)]
    pub created_at: Option<f64>,
    /// Status: "queued", "in_progress", "completed", "incomplete", "cancelled", "failed".
    #[serde(default)]
    pub status: Option<String>,
    /// Error details (if status is "failed").
    #[serde(default)]
    pub error: Option<ResponseError>,
    /// Incomplete details (if status is "incomplete").
    #[serde(default)]
    pub incomplete_details: Option<serde_json::Value>,
    /// Instructions used.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Model used.
    #[serde(default)]
    pub model: Option<String>,
    /// Output items.
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,
    /// Convenience field: concatenated text output.
    #[serde(default)]
    pub output_text: Option<String>,
    /// Whether parallel tool calls were enabled.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Previous response ID (if chained).
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Reasoning config used.
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    /// Service tier used.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Metadata.
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Temperature used.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Top-p used.
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Max output tokens.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Tool choice used.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Tools used.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Text format config.
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    /// Truncation setting.
    #[serde(default)]
    pub truncation: Option<String>,
    /// Whether stored.
    #[serde(default)]
    pub store: Option<bool>,
    /// Usage statistics.
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    /// End-user identifier.
    #[serde(default)]
    pub user: Option<String>,
}

/// Usage statistics from the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseUsage {
    /// Input tokens.
    pub input_tokens: u32,
    /// Output tokens.
    pub output_tokens: u32,
    /// Total tokens.
    #[serde(default)]
    pub total_tokens: Option<u32>,
    /// Detailed output token breakdown.
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Detailed breakdown of output tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens used.
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

/// Error info in a failed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code.
    #[serde(default)]
    pub code: Option<String>,
    /// Error message.
    #[serde(default)]
    pub message: Option<String>,
}

/// Result of `DELETE /v1/responses/{id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// Response ID.
    pub id: String,
    /// Object type: "response.deleted".
    pub object: String,
    /// Whether deleted.
    pub deleted: bool,
}

/// Paginated list of input items from `GET /v1/responses/{id}/input_items`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputItemsList {
    /// Object type: "list".
    pub object: String,
    /// Input items.
    pub data: Vec<serde_json::Value>,
    /// Whether more items exist.
    #[serde(default)]
    pub has_more: bool,
    /// First item ID.
    #[serde(default)]
    pub first_id: Option<String>,
    /// Last item ID.
    #[serde(default)]
    pub last_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_response_object_minimal() {
        let json = json!({
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "output": [],
        });
        let resp: ResponseObject = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "resp_123");
        assert_eq!(resp.status, Some("completed".to_string()));
        assert!(resp.output.is_empty());
    }

    #[test]
    fn test_response_usage_with_details() {
        let json = json!({
            "input_tokens": 20,
            "output_tokens": 11,
            "total_tokens": 31,
            "output_tokens_details": {
                "reasoning_tokens": 5
            }
        });
        let usage: ResponseUsage = serde_json::from_value(json).unwrap();
        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.output_tokens, 11);
        assert_eq!(usage.total_tokens, Some(31));
        assert_eq!(
            usage.output_tokens_details.unwrap().reasoning_tokens,
            Some(5)
        );
    }

    #[test]
    fn test_response_error() {
        let json = json!({
            "code": "rate_limit_exceeded",
            "message": "Too many requests"
        });
        let err: ResponseError = serde_json::from_value(json).unwrap();
        assert_eq!(err.code, Some("rate_limit_exceeded".to_string()));
    }

    #[test]
    fn test_delete_response() {
        let json = json!({
            "id": "resp_123",
            "object": "response.deleted",
            "deleted": true
        });
        let del: DeleteResponse = serde_json::from_value(json).unwrap();
        assert!(del.deleted);
        assert_eq!(del.object, "response.deleted");
    }

    #[test]
    fn test_input_items_list() {
        let json = json!({
            "object": "list",
            "data": [{"type": "message", "role": "user", "content": "Hi"}],
            "has_more": false,
            "first_id": "msg_1",
            "last_id": "msg_1"
        });
        let list: InputItemsList = serde_json::from_value(json).unwrap();
        assert_eq!(list.data.len(), 1);
        assert!(!list.has_more);
    }

    #[test]
    fn test_full_response_from_spec() {
        let json = json!({
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1741369938.0,
            "status": "completed",
            "error": null,
            "model": "gpt-4o-2024-08-06",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}]
            }],
            "output_text": "Hello!",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 11,
                "total_tokens": 31,
                "output_tokens_details": { "reasoning_tokens": 0 }
            },
            "metadata": {},
            "temperature": 1.0,
            "top_p": 1.0,
            "store": false
        });
        let resp: ResponseObject = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "resp_abc123");
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output_text, Some("Hello!".to_string()));
    }
}
