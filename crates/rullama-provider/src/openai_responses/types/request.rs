//! Request types for the Responses API.

use serde::{Deserialize, Serialize};

use super::input::ResponseInputItem;
use super::tools::ResponseTool;

/// Audio output configuration for the Responses API.
///
/// Required when `modalities` includes `"audio"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutputConfig {
    /// Voice ID: "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx",
    /// "nova", "sage", "shimmer".
    pub voice: String,
    /// Output format: "wav", "mp3", "flac", "opus", "pcm16".
    pub format: String,
}

/// The `input` field: either a plain text string or an array of input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// Plain text string (convenience form).
    Text(String),
    /// Array of structured input items.
    Items(Vec<ResponseInputItem>),
}

/// Tool choice: string shorthand or structured object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String value: "auto", "required", "none".
    Mode(String),
    /// Force a specific function.
    Function {
        /// Always "function".
        r#type: String,
        /// Function name to force.
        name: String,
    },
}

/// Reasoning configuration for o-series models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Reasoning effort: "low", "medium", "high".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Summary generation: "auto", "concise", "detailed".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<String>,
}

/// Text format configuration (structured outputs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFormatConfig {
    /// The format specification.
    pub format: TextFormat,
}

/// Text format type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    /// Plain text (default).
    Text,
    /// JSON object (unstructured).
    JsonObject,
    /// JSON schema (structured output).
    JsonSchema {
        /// Schema name.
        name: String,
        /// Optional description.
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// The JSON Schema.
        schema: serde_json::Value,
        /// Whether to enforce strict adherence.
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Conversation reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRef {
    /// Conversation ID.
    pub id: String,
}

/// Context management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManagement {
    /// Always "compaction".
    #[serde(rename = "type")]
    pub kind: String,
    /// Token threshold to trigger compaction.
    pub compact_threshold: u32,
}

/// Full request body for `POST /v1/responses`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResponseRequest {
    /// Model ID.
    pub model: String,
    /// Input: text string or array of items.
    pub input: ResponseInput,
    /// System/developer instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,
    /// Tool selection strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Allow parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Max tokens in response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Enable SSE streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Chain to previous response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Persist response server-side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Custom metadata (max 16 key-value pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Truncation strategy: "auto" or "disabled".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    /// Reasoning config (for o-series models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Text format config (structured output).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextFormatConfig>,
    /// Extra fields to include in response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// End-user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Execute asynchronously.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Service tier preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Conversation reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationRef>,
    /// Context management configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagement>>,
    /// Output modalities: `["text"]`, `["text", "audio"]`, or `["audio"]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    /// Audio output configuration (required when modalities includes "audio").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioOutputConfig>,
}

impl CreateResponseRequest {
    /// Create a minimal request.
    pub fn new(model: impl Into<String>, input: ResponseInput) -> Self {
        Self {
            model: model.into(),
            input,
            instructions: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            truncation: None,
            reasoning: None,
            text: None,
            include: None,
            user: None,
            background: None,
            service_tier: None,
            conversation: None,
            context_management: None,
            modalities: None,
            audio: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_minimal_request() {
        let req = CreateResponseRequest::new("gpt-4o", ResponseInput::Text("Hi".to_string()));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["input"], "Hi");
        // Optional fields should be absent
        assert!(json.get("instructions").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn test_tool_choice_auto() {
        let tc = ToolChoice::Mode("auto".to_string());
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json, "auto");
    }

    #[test]
    fn test_tool_choice_function() {
        let tc = ToolChoice::Function {
            r#type: "function".to_string(),
            name: "get_weather".to_string(),
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "get_weather");
    }

    #[test]
    fn test_text_format_json_schema() {
        let fmt = TextFormatConfig {
            format: TextFormat::JsonSchema {
                name: "my_schema".to_string(),
                description: None,
                schema: json!({"type": "object"}),
                strict: Some(true),
            },
        };
        let json = serde_json::to_value(&fmt).unwrap();
        assert_eq!(json["format"]["type"], "json_schema");
        assert_eq!(json["format"]["name"], "my_schema");
        let _roundtrip: TextFormatConfig = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_reasoning_config() {
        let rc = ReasoningConfig {
            effort: Some("high".to_string()),
            generate_summary: Some("concise".to_string()),
        };
        let json = serde_json::to_value(&rc).unwrap();
        assert_eq!(json["effort"], "high");
        assert_eq!(json["generate_summary"], "concise");
    }

    #[test]
    fn test_response_input_text_serde() {
        let input = ResponseInput::Text("Hello".to_string());
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json, "Hello");
    }

    #[test]
    fn test_context_management() {
        let cm = ContextManagement {
            kind: "compaction".to_string(),
            compact_threshold: 50000,
        };
        let json = serde_json::to_value(&cm).unwrap();
        assert_eq!(json["type"], "compaction");
        assert_eq!(json["compact_threshold"], 50000);
    }
}
