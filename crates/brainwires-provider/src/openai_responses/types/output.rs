//! Output item types for the Responses API.

use serde::{Deserialize, Serialize};

/// An output item from the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    /// A text message.
    Message {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Role (always "assistant").
        role: String,
        /// Content blocks.
        content: Vec<OutputContentBlock>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
    },
    /// A function call.
    FunctionCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Function name.
        name: String,
        /// JSON arguments.
        arguments: String,
        /// Call ID.
        call_id: String,
        /// Status.
        #[serde(default)]
        status: Option<String>,
    },
    /// A file search call.
    FileSearchCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
        /// Search queries.
        #[serde(default)]
        queries: Vec<String>,
        /// Search results (only if `include` contains "file_search_call.results").
        #[serde(default)]
        results: Option<Vec<FileSearchResult>>,
    },
    /// A web search call.
    WebSearchCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
    },
    /// A computer use call.
    ComputerCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Call ID.
        call_id: String,
        /// Action to perform.
        action: serde_json::Value,
        /// Pending safety checks.
        #[serde(default)]
        pending_safety_checks: Vec<serde_json::Value>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
    },
    /// A code interpreter call.
    CodeInterpreterCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// The code to execute.
        #[serde(default)]
        code: Option<String>,
        /// Container ID.
        #[serde(default)]
        container_id: Option<String>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
        /// Outputs from code execution.
        #[serde(default)]
        outputs: Vec<CodeInterpreterOutput>,
    },
    /// An MCP tool call.
    McpCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Server label.
        server_label: String,
        /// Tool name.
        name: String,
        /// JSON arguments.
        #[serde(default)]
        arguments: Option<String>,
        /// Output.
        #[serde(default)]
        output: Option<String>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
    },
    /// An MCP approval request.
    McpApprovalRequest {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Tool name.
        name: String,
        /// Arguments object.
        #[serde(default)]
        arguments: serde_json::Value,
        /// Server label.
        server_label: String,
    },
    /// An MCP list tools result.
    McpListTools {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Server label.
        server_label: String,
        /// Tools discovered.
        #[serde(default)]
        tools: Vec<McpToolDef>,
    },
    /// Reasoning output.
    Reasoning {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Summary parts.
        #[serde(default)]
        summary: Vec<ReasoningSummaryPart>,
        /// Encrypted content (for stateless multi-turn).
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    /// An image generation call.
    ImageGenerationCall {
        /// Item ID.
        #[serde(default)]
        id: Option<String>,
        /// Base64 image result.
        #[serde(default)]
        result: Option<String>,
        /// Status.
        #[serde(default)]
        status: Option<String>,
    },
}

/// Content block within a message output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentBlock {
    /// Output text with optional annotations.
    OutputText {
        /// The text content.
        text: String,
        /// Annotations (citations, file paths).
        #[serde(default)]
        annotations: Vec<Annotation>,
    },
    /// Model refused to answer.
    Refusal {
        /// The refusal text.
        refusal: String,
    },
    /// Audio output from the model.
    OutputAudio {
        /// Unique audio ID.
        id: String,
        /// Base64-encoded audio data.
        data: String,
        /// Text transcript of the audio.
        #[serde(default)]
        transcript: Option<String>,
        /// Expiration timestamp (Unix epoch).
        #[serde(default)]
        expires_at: Option<f64>,
    },
}

/// An annotation on output text.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// URL citation.
    UrlCitation {
        /// The cited URL.
        url: String,
        /// Page title.
        #[serde(default)]
        title: Option<String>,
        /// Start index in text.
        start_index: u32,
        /// End index in text.
        end_index: u32,
    },
    /// File citation.
    FileCitation {
        /// File ID.
        file_id: String,
        /// Quoted text.
        #[serde(default)]
        quote: Option<String>,
        /// Start index in text.
        start_index: u32,
        /// End index in text.
        end_index: u32,
    },
    /// File path reference.
    FilePath {
        /// File ID.
        file_id: String,
        /// Start index in text.
        start_index: u32,
        /// End index in text.
        end_index: u32,
    },
}

/// A file search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// File ID.
    pub file_id: String,
    /// Filename.
    #[serde(default)]
    pub filename: Option<String>,
    /// Relevance score.
    #[serde(default)]
    pub score: Option<f64>,
    /// Matching text.
    #[serde(default)]
    pub text: Option<String>,
    /// Metadata attributes.
    #[serde(default)]
    pub attributes: serde_json::Value,
}

/// Output from a code interpreter call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterOutput {
    /// Log output.
    Logs {
        /// Log text.
        logs: String,
    },
    /// Image output.
    Image {
        /// Image URL.
        image_url: String,
    },
    /// File output.
    File {
        /// File ID.
        file_id: String,
    },
}

/// A tool definition from MCP list_tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    /// Tool name.
    pub name: String,
    /// Tool description.
    #[serde(default)]
    pub description: Option<String>,
    /// Input schema.
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

/// A reasoning summary part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummaryPart {
    /// Summary text.
    SummaryText {
        /// The summary text.
        text: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_output_roundtrip() {
        let item = ResponseOutputItem::Message {
            id: Some("msg_123".to_string()),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::OutputText {
                text: "Hello!".to_string(),
                annotations: vec![],
            }],
            status: Some("completed".to_string()),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["content"][0]["type"], "output_text");
        let _roundtrip: ResponseOutputItem = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_function_call_output_roundtrip() {
        let item = ResponseOutputItem::FunctionCall {
            id: Some("fc_1".to_string()),
            name: "get_weather".to_string(),
            arguments: r#"{"location":"SF"}"#.to_string(),
            call_id: "call_1".to_string(),
            status: Some("completed".to_string()),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "function_call");
        assert_eq!(json["name"], "get_weather");
        let _roundtrip: ResponseOutputItem = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_annotation_url_citation() {
        let ann = Annotation::UrlCitation {
            url: "https://example.com".to_string(),
            title: Some("Example".to_string()),
            start_index: 0,
            end_index: 10,
        };
        let json = serde_json::to_value(&ann).unwrap();
        assert_eq!(json["type"], "url_citation");
        let _roundtrip: Annotation = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_output_text_with_annotations() {
        let block = OutputContentBlock::OutputText {
            text: "According to source".to_string(),
            annotations: vec![
                Annotation::UrlCitation {
                    url: "https://example.com".to_string(),
                    title: Some("Source".to_string()),
                    start_index: 0,
                    end_index: 19,
                },
                Annotation::FileCitation {
                    file_id: "file_1".to_string(),
                    quote: Some("relevant text".to_string()),
                    start_index: 5,
                    end_index: 15,
                },
            ],
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["annotations"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_refusal_block() {
        let block = OutputContentBlock::Refusal {
            refusal: "I cannot help with that.".to_string(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "refusal");
        let _roundtrip: OutputContentBlock = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_reasoning_output_item() {
        let item = ResponseOutputItem::Reasoning {
            id: Some("rs_1".to_string()),
            summary: vec![ReasoningSummaryPart::SummaryText {
                text: "I think...".to_string(),
            }],
            encrypted_content: None,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "reasoning");
        assert_eq!(json["summary"][0]["type"], "summary_text");
    }

    #[test]
    fn test_file_search_call_output() {
        let json = json!({
            "type": "file_search_call",
            "id": "fs_1",
            "status": "completed",
            "queries": ["test query"],
            "results": [{
                "file_id": "file_1",
                "filename": "doc.pdf",
                "score": 0.95,
                "text": "relevant text"
            }]
        });
        let item: ResponseOutputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, ResponseOutputItem::FileSearchCall { .. }));
    }

    #[test]
    fn test_code_interpreter_output_variants() {
        let logs = CodeInterpreterOutput::Logs {
            logs: "hello\n".to_string(),
        };
        let json = serde_json::to_value(&logs).unwrap();
        assert_eq!(json["type"], "logs");

        let img = CodeInterpreterOutput::Image {
            image_url: "https://...".to_string(),
        };
        let json = serde_json::to_value(&img).unwrap();
        assert_eq!(json["type"], "image");
    }

    #[test]
    fn test_mcp_call_output() {
        let item = ResponseOutputItem::McpCall {
            id: Some("mcp_1".to_string()),
            server_label: "github".to_string(),
            name: "list_repos".to_string(),
            arguments: Some("{}".to_string()),
            output: Some("result".to_string()),
            status: Some("completed".to_string()),
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "mcp_call");
        assert_eq!(json["server_label"], "github");
    }
}
