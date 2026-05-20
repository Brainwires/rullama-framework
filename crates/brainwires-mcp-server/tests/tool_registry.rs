//! Integration tests for the MCP tool registry — verifying tool registration,
//! dispatch, and interaction with middleware filtering.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_mcp_client::{CallToolResult, JsonRpcRequest};
use brainwires_mcp_server::ToolFilterMiddleware;
use brainwires_mcp_server::connection::RequestContext;
use brainwires_mcp_server::middleware::MiddlewareChain;
use brainwires_mcp_server::registry::{McpToolRegistry, ToolHandler};
use serde_json::{Value, json};

/// A test handler that returns a success result with no content.
struct NoOpHandler;

#[async_trait]
impl ToolHandler for NoOpHandler {
    async fn call(&self, _args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
        Ok(CallToolResult::success(vec![]))
    }
}

/// A test handler that returns an error.
struct FailHandler;

#[async_trait]
impl ToolHandler for FailHandler {
    async fn call(&self, _args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
        anyhow::bail!("intentional failure for testing")
    }
}

/// Test registering multiple tools and dispatching to the correct handler.
#[tokio::test]
async fn register_and_dispatch_multiple_tools() {
    let mut registry = McpToolRegistry::new();

    registry.register(
        "agent_spawn",
        "Spawns an agent",
        json!({"type": "object", "properties": {"description": {"type": "string"}}}),
        NoOpHandler,
    );

    registry.register(
        "agent_list",
        "Lists agents",
        json!({"type": "object"}),
        NoOpHandler,
    );

    let ctx = RequestContext::new(json!(1));

    // Dispatch to agent_spawn
    let result = registry
        .dispatch("agent_spawn", json!({"description": "test task"}), &ctx)
        .await
        .unwrap();
    assert!(!result.is_error.unwrap_or(false));

    // Dispatch to agent_list
    let result = registry
        .dispatch("agent_list", json!({}), &ctx)
        .await
        .unwrap();
    assert!(!result.is_error.unwrap_or(false));
}

/// Test that dispatching to a non-existent tool returns an error.
#[tokio::test]
async fn dispatch_nonexistent_tool_fails() {
    let registry = McpToolRegistry::new();
    let ctx = RequestContext::new(json!(1));

    let result = registry.dispatch("nonexistent", json!({}), &ctx).await;
    assert!(result.is_err());
}

/// Test that a handler returning an error propagates correctly.
#[tokio::test]
async fn dispatch_to_failing_handler_returns_error() {
    let mut registry = McpToolRegistry::new();
    registry.register("failing_tool", "Always fails", json!({}), FailHandler);

    let ctx = RequestContext::new(json!(1));
    let result = registry.dispatch("failing_tool", json!({}), &ctx).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("intentional failure")
    );
}

/// Test that tool registry listing reflects all registered tools.
#[test]
fn list_tools_returns_all_registered() {
    let mut registry = McpToolRegistry::new();

    registry.register("tool_a", "First tool", json!({}), NoOpHandler);
    registry.register("tool_b", "Second tool", json!({}), NoOpHandler);
    registry.register("tool_c", "Third tool", json!({}), NoOpHandler);

    let tools = registry.list_tools();
    assert_eq!(tools.len(), 3);

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"tool_a"));
    assert!(names.contains(&"tool_b"));
    assert!(names.contains(&"tool_c"));
}

/// Test that tool filter middleware and tool registry work together:
/// the middleware controls which tools can be called, the registry dispatches.
#[tokio::test]
async fn tool_filter_gates_registry_dispatch() {
    // Set up registry with tools
    let mut registry = McpToolRegistry::new();
    registry.register("agent_spawn", "Spawn an agent", json!({}), NoOpHandler);
    registry.register("bash", "Run bash command", json!({}), NoOpHandler);

    // Set up middleware that only allows agent_spawn
    let mut chain = MiddlewareChain::new();
    chain.add(ToolFilterMiddleware::allow_only(["agent_spawn"]));

    let ctx = RequestContext::new(json!(1));

    // agent_spawn should pass middleware
    let spawn_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tools/call".to_string(),
        params: Some(json!({"name": "agent_spawn", "arguments": {}})),
    };
    let mut spawn_ctx = ctx.clone();
    assert!(
        chain
            .process_request(&spawn_request, &mut spawn_ctx)
            .await
            .is_ok()
    );

    // bash should be rejected by middleware
    let bash_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(2),
        method: "tools/call".to_string(),
        params: Some(json!({"name": "bash", "arguments": {}})),
    };
    let mut bash_ctx = ctx.clone();
    let result = chain.process_request(&bash_request, &mut bash_ctx).await;
    assert!(result.is_err());

    // Registry can still dispatch bash if called directly (middleware is separate)
    let dispatch_result = registry
        .dispatch("bash", json!({"command": "ls"}), &ctx)
        .await
        .unwrap();
    assert!(!dispatch_result.is_error.unwrap_or(false));
}

/// Test has_tool checks against the registry.
#[test]
fn has_tool_reflects_registration() {
    let mut registry = McpToolRegistry::new();
    assert!(!registry.has_tool("agent_spawn"));

    registry.register("agent_spawn", "Spawn", json!({}), NoOpHandler);
    assert!(registry.has_tool("agent_spawn"));
    assert!(!registry.has_tool("other"));
}

/// Test list_tool_defs returns cloned definitions.
#[test]
fn list_tool_defs_returns_owned_copies() {
    let mut registry = McpToolRegistry::new();
    registry.register(
        "search",
        "Search codebase",
        json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        NoOpHandler,
    );

    let defs = registry.list_tool_defs();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "search");
    assert_eq!(defs[0].description, "Search codebase");

    // Verify we can drop registry and defs are still valid (owned)
    drop(registry);
    assert_eq!(defs[0].name, "search");
}
