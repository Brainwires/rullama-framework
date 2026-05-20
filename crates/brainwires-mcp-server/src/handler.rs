use anyhow::Result;
use async_trait::async_trait;
use brainwires_mcp_client::{CallToolResult, InitializeParams, ServerCapabilities, ServerInfo};
use serde_json::Value;

use crate::connection::RequestContext;
use crate::registry::McpToolDef;

/// Trait for handling MCP protocol requests.
#[async_trait]
pub trait McpHandler: Send + Sync + 'static {
    /// Return server identification info.
    fn server_info(&self) -> ServerInfo;
    /// Return server capabilities.
    fn capabilities(&self) -> ServerCapabilities;
    /// List all available tools.
    fn list_tools(&self) -> Vec<McpToolDef>;
    /// Execute a tool call.
    async fn call_tool(
        &self,
        name: &str,
        args: Value,
        ctx: &RequestContext,
    ) -> Result<CallToolResult>;

    /// Called when the client sends an initialize request.
    async fn on_initialize(&self, _params: &InitializeParams) -> Result<()> {
        Ok(())
    }

    /// Called when the server is shutting down.
    async fn on_shutdown(&self) -> Result<()> {
        Ok(())
    }
}
