//! Input item types for the Responses API.

use serde::{Deserialize, Serialize};

/// An input item for the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    /// A message from user/assistant/system/developer.
    Message {
        /// Role: "user", "assistant", "system", "developer".
        role: String,
        /// Content: plain string or array of content parts.
        content: InputContent,
        /// Optional status (e.g. "completed").
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// A function call output (tool result).
    FunctionCallOutput {
        /// The call ID this result is for.
        call_id: String,
        /// The output text.
        output: String,
    },
    /// A computer call output.
    ComputerCallOutput {
        /// The call ID this result is for.
        call_id: String,
        /// The screenshot output.
        output: ComputerScreenshotInput,
    },
    /// An MCP approval response.
    McpApprovalResponse {
        /// Whether to approve.
        approve: bool,
        /// The approval request ID.
        approval_request_id: String,
    },
    /// A reference to a previous output item by ID.
    ItemReference {
        /// The item ID.
        id: String,
    },
}

/// Content of a message input — either a plain string or structured parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    /// Plain text string.
    Text(String),
    /// Array of content parts.
    Parts(Vec<InputContentPart>),
}

impl From<String> for InputContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for InputContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

/// A content part within a structured message input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentPart {
    /// Text content.
    InputText {
        /// The text.
        text: String,
    },
    /// Image content.
    InputImage {
        /// Image URL (can be https:// or data: URI).
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        /// File ID for uploaded images.
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// Detail level: "auto", "low", "high".
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Audio content.
    InputAudio {
        /// Base64-encoded audio data.
        data: String,
        /// Audio format: "wav", "mp3", "flac", "webm", "ogg".
        format: String,
    },
    /// File content.
    InputFile {
        /// File ID for uploaded files.
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// Filename (for inline form).
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        /// Inline file data (data: URI).
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
    },
}

/// Screenshot output for computer call results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerScreenshotInput {
    /// Always "computer_screenshot".
    #[serde(rename = "type")]
    pub kind: String,
    /// The screenshot image URL (data: URI).
    pub image_url: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_input_text_content() {
        let item = ResponseInputItem::Message {
            role: "user".to_string(),
            content: InputContent::Text("Hello".to_string()),
            status: None,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn test_message_input_parts_content() {
        let item = ResponseInputItem::Message {
            role: "user".to_string(),
            content: InputContent::Parts(vec![
                InputContentPart::InputText {
                    text: "Look at this".to_string(),
                },
                InputContentPart::InputImage {
                    image_url: Some("https://example.com/img.png".to_string()),
                    file_id: None,
                    detail: Some("auto".to_string()),
                },
            ]),
            status: None,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["content"][0]["type"], "input_text");
        assert_eq!(json["content"][1]["type"], "input_image");
    }

    #[test]
    fn test_function_call_output_serde() {
        let item = ResponseInputItem::FunctionCallOutput {
            call_id: "call_1".to_string(),
            output: "result".to_string(),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "function_call_output");
        assert_eq!(json["call_id"], "call_1");

        let roundtrip: ResponseInputItem = serde_json::from_value(json).unwrap();
        if let ResponseInputItem::FunctionCallOutput { call_id, output } = roundtrip {
            assert_eq!(call_id, "call_1");
            assert_eq!(output, "result");
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_mcp_approval_response_serde() {
        let item = ResponseInputItem::McpApprovalResponse {
            approve: true,
            approval_request_id: "mcpr_123".to_string(),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "mcp_approval_response");
        assert_eq!(json["approve"], true);
    }

    #[test]
    fn test_item_reference_serde() {
        let item = ResponseInputItem::ItemReference {
            id: "msg_abc".to_string(),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "item_reference");
        assert_eq!(json["id"], "msg_abc");
    }

    #[test]
    fn test_input_content_from_str() {
        let content: InputContent = "hello".into();
        assert!(matches!(content, InputContent::Text(s) if s == "hello"));
    }
}
