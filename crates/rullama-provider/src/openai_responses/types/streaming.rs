//! Streaming event types for the Responses API.

use serde::{Deserialize, Serialize};

use super::output::ResponseOutputItem;
use super::response::ResponseObject;

/// A streaming event from the Responses API.
///
/// Events are delivered as SSE with `event:` and `data:` fields.
/// The `data` payload is JSON matching one of these variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseStreamEvent {
    // ── Response lifecycle ───────────────────────────────────────────
    /// Stream initialized.
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// The response object.
        response: ResponseObject,
    },
    /// Processing started.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        /// The response object.
        response: ResponseObject,
    },
    /// Generation finished.
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        /// The full response object (includes usage).
        response: ResponseObject,
    },
    /// Generation failed.
    #[serde(rename = "response.failed")]
    ResponseFailed {
        /// The response object with error details.
        response: ResponseObject,
    },
    /// Response truncated/incomplete.
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        /// The response object.
        response: ResponseObject,
    },

    // ── Output items ────────────────────────────────────────────────
    /// New output item started.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        /// The output item.
        item: ResponseOutputItem,
        /// Index in output array.
        output_index: u32,
    },
    /// Output item completed.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        /// The completed output item.
        item: ResponseOutputItem,
        /// Index in output array.
        output_index: u32,
    },

    // ── Content parts ───────────────────────────────────────────────
    /// Content part started.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        /// The content part.
        part: serde_json::Value,
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
        /// Content index.
        content_index: u32,
    },
    /// Content part completed.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        /// The completed content part.
        part: serde_json::Value,
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
        /// Content index.
        content_index: u32,
    },

    // ── Text deltas ─────────────────────────────────────────────────
    /// Incremental text.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        /// Text delta.
        delta: String,
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
        /// Content index.
        content_index: u32,
    },
    /// Full text completed.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        /// Complete text.
        text: String,
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
        /// Content index.
        content_index: u32,
    },

    // ── Function call arguments ─────────────────────────────────────
    /// Incremental function args.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        /// JSON delta.
        delta: String,
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// Function args completed.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        /// Complete arguments JSON.
        arguments: String,
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },

    // ── Refusal ─────────────────────────────────────────────────────
    /// Incremental refusal text.
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        /// Refusal text delta.
        delta: String,
        /// Item ID.
        item_id: String,
        /// Content index.
        content_index: u32,
    },
    /// Refusal completed.
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        /// Complete refusal text.
        refusal: String,
        /// Item ID.
        item_id: String,
        /// Content index.
        content_index: u32,
    },

    // ── Reasoning ───────────────────────────────────────────────────
    /// Reasoning summary part started.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        /// The summary part.
        part: serde_json::Value,
        /// Item ID.
        item_id: String,
        /// Summary index.
        summary_index: u32,
    },
    /// Incremental reasoning summary text.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        /// Text delta.
        delta: String,
        /// Item ID.
        item_id: String,
        /// Summary index.
        summary_index: u32,
    },
    /// Reasoning summary text completed.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        /// Complete text.
        text: String,
        /// Item ID.
        item_id: String,
        /// Summary index.
        summary_index: u32,
    },
    /// Reasoning summary part completed.
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        /// The completed summary part.
        part: serde_json::Value,
        /// Item ID.
        item_id: String,
        /// Summary index.
        summary_index: u32,
    },

    // ── Built-in tool progress ──────────────────────────────────────
    /// File search in progress.
    #[serde(rename = "response.file_search_call.in_progress")]
    FileSearchCallInProgress {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// File search searching.
    #[serde(rename = "response.file_search_call.searching")]
    FileSearchCallSearching {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// File search completed.
    #[serde(rename = "response.file_search_call.completed")]
    FileSearchCallCompleted {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },

    /// Web search in progress.
    #[serde(rename = "response.web_search_call.in_progress")]
    WebSearchCallInProgress {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// Web search searching.
    #[serde(rename = "response.web_search_call.searching")]
    WebSearchCallSearching {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// Web search completed.
    #[serde(rename = "response.web_search_call.completed")]
    WebSearchCallCompleted {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },

    /// Code interpreter in progress.
    #[serde(rename = "response.code_interpreter_call.in_progress")]
    CodeInterpreterCallInProgress {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// Code interpreter interpreting.
    #[serde(rename = "response.code_interpreter_call.interpreting")]
    CodeInterpreterCallInterpreting {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// Code interpreter completed.
    #[serde(rename = "response.code_interpreter_call.completed")]
    CodeInterpreterCallCompleted {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },

    /// MCP call in progress.
    #[serde(rename = "response.mcp_call.in_progress")]
    McpCallInProgress {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// MCP call completed.
    #[serde(rename = "response.mcp_call.completed")]
    McpCallCompleted {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// MCP call failed.
    #[serde(rename = "response.mcp_call.failed")]
    McpCallFailed {
        /// Item ID.
        item_id: String,
        /// Output index.
        output_index: u32,
    },
    /// MCP list tools in progress.
    #[serde(rename = "response.mcp_list_tools.in_progress")]
    McpListToolsInProgress {
        /// Item ID.
        item_id: String,
    },
    /// MCP list tools completed.
    #[serde(rename = "response.mcp_list_tools.completed")]
    McpListToolsCompleted {
        /// Item ID.
        item_id: String,
    },

    /// Image generation in progress.
    #[serde(rename = "response.image_generation_call.in_progress")]
    ImageGenerationCallInProgress {
        /// Item ID.
        item_id: String,
    },
    /// Image generation generating.
    #[serde(rename = "response.image_generation_call.generating")]
    ImageGenerationCallGenerating {
        /// Item ID.
        item_id: String,
    },
    /// Image generation partial image.
    #[serde(rename = "response.image_generation_call.partial_image")]
    ImageGenerationCallPartialImage {
        /// Item ID.
        item_id: String,
        /// Partial image base64 data.
        partial_image_b64: String,
    },
    /// Image generation completed.
    #[serde(rename = "response.image_generation_call.completed")]
    ImageGenerationCallCompleted {
        /// Item ID.
        item_id: String,
        /// Base64 result.
        #[serde(default)]
        result: Option<String>,
    },

    // ── Error ───────────────────────────────────────────────────────
    /// Error event.
    #[serde(rename = "error")]
    Error {
        /// Error details.
        error: serde_json::Value,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_output_text_delta_roundtrip() {
        let json = json!({
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        assert!(
            matches!(event, ResponseStreamEvent::OutputTextDelta { delta, .. } if delta == "Hello")
        );
    }

    #[test]
    fn test_function_call_arguments_delta() {
        let json = json!({
            "type": "response.function_call_arguments.delta",
            "delta": "{\"loc",
            "item_id": "fc_1",
            "output_index": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(
            event,
            ResponseStreamEvent::FunctionCallArgumentsDelta { .. }
        ));
    }

    #[test]
    fn test_response_completed_event() {
        let json = json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15
                }
            }
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        if let ResponseStreamEvent::ResponseCompleted { response } = event {
            assert_eq!(response.id, "resp_1");
            assert_eq!(response.usage.unwrap().input_tokens, 10);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_output_item_added_function_call() {
        let json = json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "name": "get_weather",
                "arguments": "",
                "call_id": "call_1"
            },
            "output_index": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        if let ResponseStreamEvent::OutputItemAdded { item, output_index } = event {
            assert_eq!(output_index, 0);
            assert!(matches!(item, ResponseOutputItem::FunctionCall { .. }));
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_error_event() {
        let json = json!({
            "type": "error",
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Too many requests"
            }
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, ResponseStreamEvent::Error { .. }));
    }

    #[test]
    fn test_web_search_progress_events() {
        for event_type in &[
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
        ] {
            let json = json!({
                "type": event_type,
                "item_id": "ws_1",
                "output_index": 0
            });
            let _event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        }
    }

    #[test]
    fn test_reasoning_events() {
        let json = json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking...",
            "item_id": "rs_1",
            "summary_index": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(
            event,
            ResponseStreamEvent::ReasoningSummaryTextDelta { .. }
        ));
    }

    #[test]
    fn test_refusal_events() {
        let json = json!({
            "type": "response.refusal.delta",
            "delta": "I cannot",
            "item_id": "msg_1",
            "content_index": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, ResponseStreamEvent::RefusalDelta { .. }));
    }
}
