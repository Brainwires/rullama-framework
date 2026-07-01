//! MCP Protocol Types
//!
//! This module provides type definitions for the Model Context Protocol.
//! It now uses the official `rmcp` crate with compatibility aliases for
//! backward compatibility during migration.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// Re-export rmcp types with compatibility aliases (native only)
#[cfg(feature = "native")]
pub use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Prompt as RmcpPrompt, ProtocolVersion,
    Resource as RmcpResource, Tool as RmcpTool,
};

// Re-export capabilities (native only)
#[cfg(feature = "native")]
pub use rmcp::model::{
    ClientCapabilities as RmcpClientCapabilities, PromptsCapability, ResourcesCapability,
    ServerCapabilities as RmcpServerCapabilities, ToolsCapability,
};

// ===========================================================================
// BACKWARD COMPATIBILITY ALIASES (native only - require rmcp)
// ===========================================================================

#[cfg(feature = "native")]
/// Compatibility alias for Tool
pub type McpTool = RmcpTool;

#[cfg(feature = "native")]
/// Compatibility alias for Resource
pub type McpResource = RmcpResource;

#[cfg(feature = "native")]
/// Compatibility alias for Prompt
pub type McpPrompt = RmcpPrompt;

#[cfg(feature = "native")]
/// Compatibility alias for CallToolParams
pub type CallToolParams = CallToolRequestParams;

#[cfg(feature = "native")]
/// Compatibility alias for ServerCapabilities
pub type ServerCapabilities = RmcpServerCapabilities;

#[cfg(feature = "native")]
/// Compatibility alias for ClientCapabilities
pub type ClientCapabilities = RmcpClientCapabilities;

// ===========================================================================
// ADDITIONAL TYPES NOT DIRECTLY PROVIDED BY RMCP
// ===========================================================================
// These types are still custom as they handle JSON-RPC layer or are
// specific to our implementation

/// JSON-RPC 2.0 Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (always "2.0").
    pub jsonrpc: String,
    /// Request identifier.
    pub id: serde_json::Value,
    /// Method name.
    pub method: String,
    /// Optional parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    /// Returns an error if params cannot be serialized to JSON.
    pub fn new<T: Serialize>(
        id: impl Into<Value>,
        method: String,
        params: Option<T>,
    ) -> Result<Self, serde_json::Error> {
        let params_value = match params {
            Some(p) => Some(serde_json::to_value(p)?),
            None => None,
        };
        Ok(Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method,
            params: params_value,
        })
    }

    /// Create a new JSON-RPC request, panicking if serialization fails.
    /// Use this only when you're certain serialization cannot fail.
    pub fn new_unchecked<T: Serialize>(
        id: impl Into<Value>,
        method: String,
        params: Option<T>,
    ) -> Self {
        Self::new(id, method, params).expect("Failed to serialize JSON-RPC request params")
    }
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (always "2.0").
    pub jsonrpc: String,
    /// Response identifier matching the request.
    pub id: serde_json::Value,
    /// Result value on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error object on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i32,
    /// Error message.
    pub message: String,
    /// Optional additional data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// JSON-RPC 2.0 Notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// JSON-RPC version (always "2.0").
    pub jsonrpc: String,
    /// Notification method name.
    pub method: String,
    /// Optional parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    /// Create a new JSON-RPC notification (no id field).
    /// Returns an error if params cannot be serialized to JSON.
    pub fn new<T: Serialize>(
        method: impl Into<String>,
        params: Option<T>,
    ) -> Result<Self, serde_json::Error> {
        let params_value = match params {
            Some(p) => Some(serde_json::to_value(p)?),
            None => None,
        };
        Ok(Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params: params_value,
        })
    }

    /// Create a new JSON-RPC notification, panicking if serialization fails.
    /// Use this only when you're certain serialization cannot fail.
    pub fn new_unchecked<T: Serialize>(method: impl Into<String>, params: Option<T>) -> Self {
        Self::new(method, params).expect("Failed to serialize JSON-RPC notification params")
    }
}

/// Generic JSON-RPC message that could be a response or notification
/// Used for bidirectional MCP communication where servers can send notifications
#[derive(Debug, Clone)]
pub enum JsonRpcMessage {
    /// A response to a request (has id field)
    Response(JsonRpcResponse),
    /// A server-initiated notification (no id field)
    Notification(JsonRpcNotification),
}

impl JsonRpcMessage {
    /// Check if this is a response
    pub fn is_response(&self) -> bool {
        matches!(self, JsonRpcMessage::Response(_))
    }

    /// Check if this is a notification
    pub fn is_notification(&self) -> bool {
        matches!(self, JsonRpcMessage::Notification(_))
    }

    /// Try to get the response if this is one
    pub fn as_response(self) -> Option<JsonRpcResponse> {
        match self {
            JsonRpcMessage::Response(r) => Some(r),
            _ => None,
        }
    }

    /// Try to get the notification if this is one
    pub fn as_notification(self) -> Option<JsonRpcNotification> {
        match self {
            JsonRpcMessage::Notification(n) => Some(n),
            _ => None,
        }
    }
}

// ===========================================================================
// MCP PROGRESS NOTIFICATION TYPES
// ===========================================================================

/// Progress notification parameters from MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressParams {
    /// Token identifying which request this progress is for
    #[serde(rename = "progressToken")]
    pub progress_token: String,
    /// Current progress value
    pub progress: f64,
    /// Total expected value (for calculating percentage)
    pub total: Option<f64>,
    /// Human-readable progress message
    pub message: Option<String>,
}

/// Parsed MCP notification types
#[derive(Debug, Clone)]
pub enum McpNotification {
    /// Progress update for a long-running operation
    Progress(ProgressParams),
    /// Unknown/unhandled notification type
    Unknown {
        /// The notification method name.
        method: String,
        /// Optional notification parameters.
        params: Option<Value>,
    },
}

impl McpNotification {
    /// Parse a JsonRpcNotification into a typed McpNotification
    pub fn from_notification(notif: &JsonRpcNotification) -> Self {
        match notif.method.as_str() {
            "notifications/progress" => {
                if let Some(ref params) = notif.params
                    && let Ok(progress) = serde_json::from_value::<ProgressParams>(params.clone())
                {
                    return McpNotification::Progress(progress);
                }
                McpNotification::Unknown {
                    method: notif.method.clone(),
                    params: notif.params.clone(),
                }
            }
            _ => McpNotification::Unknown {
                method: notif.method.clone(),
                params: notif.params.clone(),
            },
        }
    }
}

// ===========================================================================
// MCP INITIALIZATION TYPES
// ===========================================================================

// ===========================================================================
// MCP TYPES (require rmcp - native only)
// ===========================================================================

#[cfg(feature = "native")]
/// MCP Initialize Request Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    /// Protocol version string.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client capabilities.
    pub capabilities: ClientCapabilities,
    /// Client identification info.
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// MCP client identification.
#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name.
    pub name: String,
    /// Client version.
    pub version: String,
}

#[cfg(feature = "native")]
/// MCP Initialize Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    /// Protocol version string.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Server identification info.
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// MCP server identification.
#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name.
    pub name: String,
    /// Server version.
    pub version: String,
}

#[cfg(feature = "native")]
/// Tools List Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    /// List of available tools.
    pub tools: Vec<McpTool>,
}

#[cfg(feature = "native")]
/// Resources List Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    /// List of available resources.
    pub resources: Vec<McpResource>,
}

#[cfg(feature = "native")]
/// Prompts List Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    /// List of available prompts.
    pub prompts: Vec<McpPrompt>,
}

#[cfg(feature = "native")]
/// Resource Read Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    /// Resource URI to read.
    pub uri: String,
}

#[cfg(feature = "native")]
/// Resource Read Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    /// Resource contents.
    pub contents: Vec<ResourceContent>,
}

/// Content of a resource (text or binary blob).
#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResourceContent {
    /// Text content.
    Text {
        /// Resource URI.
        uri: String,
        /// MIME type.
        mime_type: Option<String>,
        /// Text content.
        text: String,
    },
    /// Binary blob content (base64-encoded).
    Blob {
        /// Resource URI.
        uri: String,
        /// MIME type.
        mime_type: Option<String>,
        /// Base64-encoded blob data.
        blob: String,
    },
}

#[cfg(feature = "native")]
/// Prompt Get Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    /// Prompt name.
    pub name: String,
    /// Optional arguments for the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

#[cfg(feature = "native")]
/// Prompt Get Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    /// Prompt description.
    pub description: String,
    /// Prompt messages.
    pub messages: Vec<PromptMessage>,
}

/// A message within a prompt.
#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    /// Message role (e.g., "user", "assistant").
    pub role: String,
    /// Message content.
    pub content: PromptContent,
}

/// Content type within a prompt message.
#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
// reason: `Resource` carries an `McpResource` (~hundreds of bytes); boxing it
// would change this enum's public layout and serialization expectations.
#[allow(clippy::large_enum_variant)]
pub enum PromptContent {
    /// Text content.
    Text {
        /// The text.
        text: String,
    },
    /// Image content.
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type of the image.
        mime_type: String,
    },
    /// Embedded resource content.
    Resource {
        /// The resource.
        resource: McpResource,
    },
}

#[cfg(feature = "native")]
/// Prompt Argument Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    /// Argument name.
    pub name: String,
    /// Argument description.
    pub description: String,
    /// Whether the argument is required.
    pub required: bool,
}

/// Content type within a tool result.
#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
// reason: `Resource` carries an `McpResource` (~hundreds of bytes); boxing it
// would change this enum's public layout and serialization expectations.
#[allow(clippy::large_enum_variant)]
pub enum ToolResultContent {
    /// Text result.
    Text {
        /// The text.
        text: String,
    },
    /// Image result.
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type of the image.
        mime_type: String,
    },
    /// Resource result.
    Resource {
        /// The resource.
        resource: McpResource,
    },
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_rpc_request_new() {
        let request =
            JsonRpcRequest::new(1, "test_method".to_string(), Some(json!({"key": "value"})))
                .unwrap();

        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.id, json!(1));
        assert_eq!(request.method, "test_method");
        assert!(request.params.is_some());
    }

    #[test]
    fn test_json_rpc_request_serialization() {
        let request = JsonRpcRequest::new(1, "test".to_string(), None::<()>).unwrap();
        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("jsonrpc"));
        assert!(json.contains("2.0"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_json_rpc_response_success() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: Some(json!({"status": "ok"})),
            error: None,
        };

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_json_rpc_response_error() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request".to_string(),
                data: None,
            }),
        };

        assert!(response.result.is_none());
        assert!(response.error.is_some());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_type_aliases_work() {
        // Test that our type aliases are properly set up
        let _tool: McpTool;
        let _resource: McpResource;
        let _prompt: McpPrompt;
        // If this compiles, the aliases are working
    }
}
