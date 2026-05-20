//! Tool type definitions for the Responses API.

use serde::{Deserialize, Serialize};

/// A tool definition for the Responses API (all 7 types).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTool {
    /// A custom function tool.
    Function {
        /// Function name.
        name: String,
        /// Description.
        description: String,
        /// JSON Schema for parameters.
        parameters: serde_json::Value,
        /// Whether to enforce strict schema adherence.
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// Web search preview tool.
    #[serde(rename = "web_search_preview")]
    WebSearchPreview {
        /// Context size: "low", "medium", "high".
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        /// User location for search context.
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    /// File search tool.
    #[serde(rename = "file_search")]
    FileSearch {
        /// Vector store IDs to search.
        vector_store_ids: Vec<String>,
        /// Max results to return.
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
        /// Ranking options.
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<RankingOptions>,
        /// Metadata filters.
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<serde_json::Value>,
    },
    /// Code interpreter tool.
    #[serde(rename = "code_interpreter")]
    CodeInterpreter {
        /// Container configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        container: Option<CodeInterpreterContainer>,
    },
    /// Computer use preview tool.
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview {
        /// Display width in pixels.
        display_width: u32,
        /// Display height in pixels.
        display_height: u32,
        /// Environment type.
        #[serde(skip_serializing_if = "Option::is_none")]
        environment: Option<String>,
    },
    /// MCP server tool.
    Mcp {
        /// Server label.
        server_label: String,
        /// Server URL.
        server_url: String,
        /// Approval requirement: "always" or "never".
        #[serde(skip_serializing_if = "Option::is_none")]
        require_approval: Option<String>,
        /// Custom headers.
        #[serde(skip_serializing_if = "Option::is_none")]
        headers: Option<std::collections::HashMap<String, String>>,
        /// Allowed tools.
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_tools: Option<Vec<String>>,
    },
    /// Image generation tool.
    #[serde(rename = "image_generation")]
    ImageGeneration {
        /// Background mode: "transparent", "opaque", "auto".
        #[serde(skip_serializing_if = "Option::is_none")]
        background: Option<String>,
        /// Output compression (0-100).
        #[serde(skip_serializing_if = "Option::is_none")]
        output_compression: Option<u32>,
        /// Output format: "png", "jpeg", "webp".
        #[serde(skip_serializing_if = "Option::is_none")]
        output_format: Option<String>,
        /// Quality: "low", "medium", "high", "auto".
        #[serde(skip_serializing_if = "Option::is_none")]
        quality: Option<String>,
        /// Size: "1024x1024", "1536x1024", "1024x1536", "auto".
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<String>,
        /// Number of partial images to stream.
        #[serde(skip_serializing_if = "Option::is_none")]
        partial_images: Option<u32>,
    },
}

/// User location for web search context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLocation {
    /// Always "approximate".
    #[serde(rename = "type")]
    pub kind: String,
    /// City name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// Region/state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Country code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Timezone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// Ranking options for file search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingOptions {
    /// Ranker: "auto" or "default_2024_08_21".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    /// Minimum score threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

/// Container configuration for code interpreter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeInterpreterContainer {
    /// Container type: "auto".
    #[serde(rename = "type")]
    pub kind: String,
    /// File IDs to mount.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_function_tool_roundtrip() {
        let tool = ResponseTool::Function {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
            strict: Some(false),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "get_weather");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_web_search_tool() {
        let tool = ResponseTool::WebSearchPreview {
            search_context_size: Some("medium".to_string()),
            user_location: Some(UserLocation {
                kind: "approximate".to_string(),
                city: Some("San Francisco".to_string()),
                region: Some("California".to_string()),
                country: Some("US".to_string()),
                timezone: None,
            }),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "web_search_preview");
        assert_eq!(json["user_location"]["city"], "San Francisco");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_file_search_tool() {
        let tool = ResponseTool::FileSearch {
            vector_store_ids: vec!["vs_123".to_string()],
            max_num_results: Some(20),
            ranking_options: Some(RankingOptions {
                ranker: Some("auto".to_string()),
                score_threshold: Some(0.0),
            }),
            filters: None,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "file_search");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_code_interpreter_tool() {
        let tool = ResponseTool::CodeInterpreter {
            container: Some(CodeInterpreterContainer {
                kind: "auto".to_string(),
                file_ids: vec!["file_1".to_string()],
            }),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "code_interpreter");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_computer_use_tool() {
        let tool = ResponseTool::ComputerUsePreview {
            display_width: 1024,
            display_height: 768,
            environment: Some("browser".to_string()),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "computer_use_preview");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_mcp_tool() {
        let tool = ResponseTool::Mcp {
            server_label: "my_server".to_string(),
            server_url: "https://mcp.example.com".to_string(),
            require_approval: Some("always".to_string()),
            headers: None,
            allowed_tools: Some(vec!["tool1".to_string()]),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "mcp");
        assert_eq!(json["server_label"], "my_server");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn test_image_generation_tool() {
        let tool = ResponseTool::ImageGeneration {
            background: Some("auto".to_string()),
            output_compression: None,
            output_format: Some("png".to_string()),
            quality: Some("high".to_string()),
            size: Some("1024x1024".to_string()),
            partial_images: None,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "image_generation");
        let _roundtrip: ResponseTool = serde_json::from_value(json).unwrap();
    }
}
