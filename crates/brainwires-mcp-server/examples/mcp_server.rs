//! MCP Server with tool registry and middleware pipeline.
//!
//! Demonstrates standing up an `McpServer` with:
//! - A `McpToolRegistry` populated with `McpToolDef` entries
//! - A middleware chain combining auth, logging, rate-limiting, and tool filtering
//! - Server lifecycle (build, configure, summarize)
//!
//! The example does **not** actually start the event loop (that requires a
//! real transport and a connected MCP client), but it exercises the full
//! construction API so you can see how the pieces fit together.
//!
//! ```bash
//! cargo run -p brainwires-network --example mcp_server --features server
//! ```

use anyhow::Result;
use async_trait::async_trait;
use brainwires_mcp_client::{CallToolResult, ServerCapabilities, ServerInfo};
use brainwires_mcp_server::{
    AuthMiddleware, LoggingMiddleware, McpServer, McpToolDef, McpToolRegistry, MiddlewareChain,
    RateLimitMiddleware, RequestContext, ToolFilterMiddleware, ToolHandler,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// 1. Define tool handlers
// ---------------------------------------------------------------------------

/// A simple tool handler that echoes back the input arguments.
struct EchoHandler;

#[async_trait]
impl ToolHandler for EchoHandler {
    async fn call(&self, args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
        println!("    [EchoHandler] called with: {args}");
        Ok(CallToolResult::success(vec![]))
    }
}

/// A tool handler that returns the current (simulated) time.
struct TimeHandler;

#[async_trait]
impl ToolHandler for TimeHandler {
    async fn call(&self, _args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
        let now = chrono::Utc::now().to_rfc3339();
        println!("    [TimeHandler] returning time: {now}");
        Ok(CallToolResult::success(vec![]))
    }
}

// ---------------------------------------------------------------------------
// 2. Define an McpHandler backed by the tool registry
// ---------------------------------------------------------------------------

struct DemoHandler {
    registry: McpToolRegistry,
}

#[async_trait]
impl brainwires_mcp_server::McpHandler for DemoHandler {
    fn server_info(&self) -> ServerInfo {
        ServerInfo {
            name: "demo-mcp-server".to_string(),
            version: "0.7.0".to_string(),
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== MCP Server Example ===\n");

    // Step 1: Build the tool registry
    println!("--- Tool Registry ---");

    let mut registry = McpToolRegistry::new();

    registry.register(
        "echo",
        "Echoes back the provided arguments",
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string", "description": "The message to echo" }
            },
            "required": ["message"]
        }),
        EchoHandler,
    );

    registry.register(
        "get_time",
        "Returns the current UTC time",
        json!({ "type": "object", "properties": {} }),
        TimeHandler,
    );

    for def in registry.list_tools() {
        println!("  Registered tool: {} — {}", def.name, def.description);
    }
    println!();

    // Step 2: Build the middleware chain
    println!("--- Middleware Chain ---");

    let mut chain = MiddlewareChain::new();

    // Auth: require a bearer token for non-initialize requests
    chain.add(AuthMiddleware::new("demo-secret-token"));
    println!("  1. AuthMiddleware       (token = demo-secret-token)");

    // Logging: log every request
    chain.add(LoggingMiddleware::new());
    println!("  2. LoggingMiddleware");

    // Rate limit: 20 requests per second
    chain.add(RateLimitMiddleware::new(20.0));
    println!("  3. RateLimitMiddleware  (20 req/s)");

    // Tool filter: block dangerous tools
    chain.add(ToolFilterMiddleware::deny(["rm_rf", "drop_database"]));
    println!("  4. ToolFilterMiddleware (deny: rm_rf, drop_database)");
    println!();

    // Step 3: Verify the registry can dispatch
    println!("--- Dispatch Test ---");

    let ctx = RequestContext::new(json!(1));

    println!("  Dispatching 'echo':");
    let result = registry
        .dispatch("echo", json!({"message": "hello"}), &ctx)
        .await;
    match &result {
        Ok(_) => println!("    Result: OK"),
        Err(e) => println!("    Result: Error — {e}"),
    }

    println!("  Dispatching 'get_time':");
    let result = registry.dispatch("get_time", json!({}), &ctx).await;
    match &result {
        Ok(_) => println!("    Result: OK"),
        Err(e) => println!("    Result: Error — {e}"),
    }

    println!("  Dispatching 'nonexistent_tool':");
    let result = registry.dispatch("nonexistent_tool", json!({}), &ctx).await;
    match &result {
        Ok(_) => println!("    Result: OK (unexpected)"),
        Err(e) => println!("    Result: Error — {e}"),
    }
    println!();

    // Step 4: Show how McpServer is constructed (without running the event loop)
    println!("--- Server Construction ---");

    let handler = DemoHandler { registry };
    let server = McpServer::new(handler)
        .with_middleware(AuthMiddleware::new("secret"))
        .with_middleware(LoggingMiddleware::new())
        .with_middleware(RateLimitMiddleware::new(10.0))
        .with_middleware(ToolFilterMiddleware::deny(["rm_rf"]));

    println!("  McpServer created with handler + 4 middleware layers");
    println!("  (Skipping server.run() — requires a connected transport)");

    // Prevent unused-variable warning
    drop(server);

    println!("\nDone.");
    Ok(())
}
