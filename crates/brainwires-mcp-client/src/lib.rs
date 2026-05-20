#![deny(missing_docs)]
//! Brainwires MCP - Model Context Protocol client and types
//!
//! This crate provides MCP client functionality for the Brainwires Agent Framework:
//!
//! - **McpClient**: Connect to external MCP servers, list/call tools, resources, prompts
//! - **Transport**: Stdio-based transport layer for MCP communication
//! - **Types**: JSON-RPC 2.0 types and MCP protocol types (with rmcp compatibility)
//! - **Config**: MCP server configuration management

// Re-export core types
pub use brainwires_core;

/// MCP client for connecting to external servers.
#[cfg(feature = "native")]
pub mod client;
/// MCP server configuration management.
pub mod config;
/// Stdio-based transport layer for MCP communication.
#[cfg(feature = "native")]
pub mod transport;
/// MCP protocol types and JSON-RPC types.
pub mod types;

// Re-exports - native-only modules
#[cfg(feature = "native")]
pub use client::McpClient;
#[cfg(all(feature = "native", feature = "http"))]
pub use transport::HttpTransport;
#[cfg(feature = "native")]
pub use transport::{StdioTransport, Transport};

// Re-exports - always available
#[cfg(feature = "native")]
pub use config::McpConfigManager;
pub use config::McpServerConfig;

// JSON-RPC types (always available)
pub use types::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    McpNotification, ProgressParams,
};

// MCP types (require rmcp, native only)
#[cfg(feature = "native")]
pub use types::{
    CallToolParams, CallToolResult, ClientCapabilities, ClientInfo, Content, GetPromptParams,
    GetPromptResult, InitializeParams, InitializeResult, ListPromptsResult, ListResourcesResult,
    ListToolsResult, McpPrompt, McpResource, McpTool, PromptArgument, PromptContent, PromptMessage,
    PromptsCapability, ReadResourceParams, ReadResourceResult, ResourceContent,
    ResourcesCapability, ServerCapabilities, ServerInfo, ToolResultContent, ToolsCapability,
};

/// Prelude module for convenient imports
pub mod prelude {
    #[cfg(feature = "native")]
    pub use super::client::McpClient;
    #[cfg(feature = "native")]
    pub use super::config::McpConfigManager;
    pub use super::config::McpServerConfig;
    #[cfg(feature = "native")]
    pub use super::transport::{StdioTransport, Transport};
    #[cfg(feature = "native")]
    pub use super::types::{
        CallToolResult, ClientCapabilities, McpPrompt, McpResource, McpTool, ServerCapabilities,
    };
    pub use super::types::{JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
}
