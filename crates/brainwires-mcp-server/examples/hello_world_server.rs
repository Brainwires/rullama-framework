//! Minimal MCP server example.
//!
//! Registers two tools — `echo` and `greet` — behind a logging middleware,
//! then serves them over stdio.
//!
//! Run with:
//!   cargo run --example hello_world_server -p brainwires-mcp-server
//!
//! Then paste JSON-RPC requests on stdin, e.g.:
//!   {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
//!   {"jsonrpc":"2.0","id":2,"method":"tools/list"}
//!   {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"message":"hello"}}}

use anyhow::Result;
use async_trait::async_trait;
use brainwires_mcp_client::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use brainwires_mcp_server::{
    LoggingMiddleware, McpHandler, McpServer, McpToolDef, McpToolRegistry, RequestContext,
    ToolHandler,
};
use serde_json::{Value, json};

// ── Tool handlers ────────────────────────────────────────────────────────────

struct EchoTool;

#[async_trait]
impl ToolHandler for EchoTool {
    async fn call(&self, args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
        let msg = args
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("(no message)");
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }
}

struct GreetTool;

#[async_trait]
impl ToolHandler for GreetTool {
    async fn call(&self, args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
        let name = args.get("name").and_then(Value::as_str).unwrap_or("World");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Hello, {name}!"
        ))]))
    }
}

// ── Handler ──────────────────────────────────────────────────────────────────

struct HelloWorldHandler {
    registry: McpToolRegistry,
}

impl HelloWorldHandler {
    fn new() -> Self {
        let mut registry = McpToolRegistry::new();

        registry.register(
            "echo",
            "Echo the provided message back to the caller.",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Text to echo" }
                },
                "required": ["message"]
            }),
            EchoTool,
        );

        registry.register(
            "greet",
            "Return a friendly greeting for the given name.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Name to greet" }
                }
            }),
            GreetTool,
        );

        Self { registry }
    }
}

#[async_trait]
impl McpHandler for HelloWorldHandler {
    fn server_info(&self) -> ServerInfo {
        ServerInfo {
            name: "hello-world".to_string(),
            version: "0.1.0".to_string(),
        }
    }

    fn capabilities(&self) -> ServerCapabilities {
        ServerCapabilities::default()
    }

    fn list_tools(&self) -> Vec<McpToolDef> {
        self.registry.list_tool_defs()
    }

    async fn call_tool(
        &self,
        name: &str,
        args: Value,
        ctx: &RequestContext,
    ) -> Result<CallToolResult> {
        self.registry.dispatch(name, args, ctx).await
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    McpServer::new(HelloWorldHandler::new())
        .with_middleware(LoggingMiddleware::new())
        .run()
        .await
}
