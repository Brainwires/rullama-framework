#![deny(missing_docs)]
//! # Brainwires MCP Server
//!
//! MCP server framework with middleware pipeline for the Brainwires Agent Framework.
//!
//! Provides everything needed to build an MCP-compliant tool server:
//! - [`McpServer`] — async event loop that reads JSON-RPC, runs middleware, dispatches to handler
//! - [`McpHandler`] — trait defining how your server responds to initialize/list_tools/call_tool
//! - [`McpToolRegistry`] — stores tool definitions + handlers, dispatches tool calls
//! - [`MiddlewareChain`] — ordered middleware pipeline (auth, logging, rate-limiting, tool filtering)
//! - [`ServerTransport`] — pluggable transport (stdio included)

/// WebSocket/HTTP connection types.
pub mod connection;
/// Error types for the MCP server.
pub mod error;
/// MCP request handler trait.
pub mod handler;
/// Stateless HTTP + SSE transport (MCP 2026 spec).
#[cfg(feature = "http")]
pub mod http_transport;
/// MCP server transport (stdio).
pub mod mcp_transport;
/// Middleware pipeline (auth, logging, rate-limiting, tool filtering).
pub mod middleware;
/// MCP tool registry.
pub mod registry;
/// MCP server lifecycle.
pub mod server;
/// MCP Tasks primitive (SEP-1686).
pub mod tasks;

pub use connection::{ClientInfo, RequestContext};
pub use error::AgentNetworkError;
pub use handler::McpHandler;
pub use mcp_transport::{ServerTransport, StdioServerTransport};
pub use middleware::{Middleware, MiddlewareChain, MiddlewareResult};
pub use registry::{McpToolDef, McpToolRegistry, ToolHandler};
pub use server::McpServer;
pub use tasks::{McpTask, McpTaskState, McpTaskStore};

// Re-export HTTP transport types
#[cfg(feature = "http")]
pub use http_transport::{
    HttpServerTransport, McpAuthInfo, McpServerCard, McpToolCardEntry, McpTransportInfo,
    OAuthProtectedResource, build_server_card,
};

// Re-export middleware implementations
pub use middleware::auth::AuthMiddleware;
pub use middleware::logging::LoggingMiddleware;
#[cfg(feature = "oauth")]
pub use middleware::oauth::OAuthMiddleware;
pub use middleware::rate_limit::RateLimitMiddleware;
pub use middleware::tool_filter::ToolFilterMiddleware;
