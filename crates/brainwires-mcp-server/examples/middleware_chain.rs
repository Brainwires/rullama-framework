//! MCP middleware pipeline — logging, rate-limiting, and tool filtering.
//!
//! ```bash
//! cargo run --example middleware_chain --features server
//! ```

use brainwires_mcp_client::JsonRpcRequest;
use brainwires_mcp_server::{
    LoggingMiddleware, MiddlewareChain, RateLimitMiddleware, RequestContext, ToolFilterMiddleware,
};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Middleware Chain Example ===\n");

    // 1. Build the middleware chain
    println!("--- Build Chain ---");

    let mut chain = MiddlewareChain::new();
    chain.add(LoggingMiddleware::new());
    chain.add(ToolFilterMiddleware::deny(["dangerous_tool", "rm_rf"]));
    chain.add(RateLimitMiddleware::new(10.0));

    println!("  Added: LoggingMiddleware");
    println!("  Added: ToolFilterMiddleware (deny: dangerous_tool, rm_rf)");
    println!("  Added: RateLimitMiddleware  (10 req/s)");
    println!();

    // 2. Process an allowed tool call
    println!("--- Allowed Request ---");

    let allowed_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "agent_spawn",
            "arguments": { "description": "Build a cache" }
        })),
    };

    let mut ctx = RequestContext::new(json!(1));

    match chain.process_request(&allowed_request, &mut ctx).await {
        Ok(()) => println!("  tools/call 'agent_spawn' -> Continue"),
        Err(e) => println!("  tools/call 'agent_spawn' -> Rejected: {}", e.message),
    }
    println!();

    // 3. Process a blocked tool call
    println!("--- Blocked Request ---");

    let blocked_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(2),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "dangerous_tool",
            "arguments": {}
        })),
    };

    let mut ctx2 = RequestContext::new(json!(2));

    match chain.process_request(&blocked_request, &mut ctx2).await {
        Ok(()) => println!("  tools/call 'dangerous_tool' -> Continue (unexpected)"),
        Err(e) => println!("  tools/call 'dangerous_tool' -> Rejected: {}", e.message),
    }
    println!();

    // 4. Non-tool-call requests pass through all middleware
    println!("--- Non-Tool Request ---");

    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(3),
        method: "initialize".to_string(),
        params: None,
    };

    let mut ctx3 = RequestContext::new(json!(3));

    match chain.process_request(&init_request, &mut ctx3).await {
        Ok(()) => {
            println!("  'initialize' -> Continue (tool filter + rate limit skip non-tool methods)")
        }
        Err(e) => println!("  'initialize' -> Rejected: {}", e.message),
    }
    println!();

    // 5. Allow-list filter demo (separate chain)
    println!("--- Allow-List Filter ---");

    let mut strict_chain = MiddlewareChain::new();
    strict_chain.add(ToolFilterMiddleware::allow_only([
        "read_file",
        "list_directory",
    ]));

    let read_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(4),
        method: "tools/call".to_string(),
        params: Some(json!({ "name": "read_file" })),
    };

    let mut ctx4 = RequestContext::new(json!(4));
    match strict_chain.process_request(&read_request, &mut ctx4).await {
        Ok(()) => println!("  tools/call 'read_file'     -> Continue"),
        Err(e) => println!("  tools/call 'read_file'     -> Rejected: {}", e.message),
    }

    let write_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(5),
        method: "tools/call".to_string(),
        params: Some(json!({ "name": "write_file" })),
    };

    let mut ctx5 = RequestContext::new(json!(5));
    match strict_chain
        .process_request(&write_request, &mut ctx5)
        .await
    {
        Ok(()) => println!("  tools/call 'write_file'    -> Continue (unexpected)"),
        Err(e) => println!("  tools/call 'write_file'    -> Rejected: {}", e.message),
    }

    println!("\nDone.");
    Ok(())
}
