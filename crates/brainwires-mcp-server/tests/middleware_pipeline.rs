//! Integration tests for the middleware pipeline — verifying that multiple
//! middleware layers interact correctly when chained together.

use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcRequest, JsonRpcResponse};
use brainwires_mcp_server::connection::RequestContext;
use brainwires_mcp_server::middleware::{Middleware, MiddlewareChain, MiddlewareResult};
use brainwires_mcp_server::{
    AuthMiddleware, LoggingMiddleware, RateLimitMiddleware, ToolFilterMiddleware,
};
use serde_json::json;

/// Helper: build a tools/call request for a given tool name.
fn tool_call_request(tool_name: &str) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_name,
            "arguments": {}
        })),
    }
}

/// Helper: build a generic non-tool request.
fn method_request(method: &str) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: method.to_string(),
        params: None,
    }
}

/// Test a realistic middleware stack: logging + auth + tool filter + rate limit.
/// An authenticated request for an allowed tool should pass all layers.
#[tokio::test]
async fn full_pipeline_allows_valid_authenticated_request() {
    let mut chain = MiddlewareChain::new();
    chain.add(LoggingMiddleware::new());
    chain.add(AuthMiddleware::new("secret-token-42"));
    chain.add(ToolFilterMiddleware::allow_only([
        "agent_spawn",
        "agent_list",
    ]));
    chain.add(RateLimitMiddleware::new(100.0));

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "agent_spawn",
            "arguments": {"description": "test task"},
            "_auth_token": "secret-token-42"
        })),
    };

    let mut ctx = RequestContext::new(json!(1));
    let result = chain.process_request(&request, &mut ctx).await;
    assert!(result.is_ok(), "valid request should pass all middleware");
}

/// Test that auth middleware rejects before tool filter is even reached.
#[tokio::test]
async fn auth_rejects_before_tool_filter() {
    let mut chain = MiddlewareChain::new();
    chain.add(AuthMiddleware::new("correct-token"));
    chain.add(ToolFilterMiddleware::allow_only(["agent_spawn"]));

    // No auth token provided
    let request = tool_call_request("agent_spawn");
    let mut ctx = RequestContext::new(json!(1));

    let result = chain.process_request(&request, &mut ctx).await;
    assert!(result.is_err());
    // Should be auth error (-32003), not tool filter error (-32001)
    assert_eq!(result.unwrap_err().code, -32003);
}

/// Test that tool filter rejects a denied tool even when authenticated.
#[tokio::test]
async fn tool_filter_rejects_denied_tool_after_auth() {
    let mut chain = MiddlewareChain::new();
    chain.add(AuthMiddleware::new("token"));
    chain.add(ToolFilterMiddleware::deny(["bash", "write_file"]));

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "bash",
            "arguments": {"command": "rm -rf /"},
            "_auth_token": "token"
        })),
    };

    let mut ctx = RequestContext::new(json!(1));
    let result = chain.process_request(&request, &mut ctx).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code, -32001);
}

/// Test that non-tool methods bypass tool filter and rate limiter.
#[tokio::test]
async fn non_tool_methods_bypass_tool_filter_and_rate_limit() {
    let mut chain = MiddlewareChain::new();
    chain.add(ToolFilterMiddleware::allow_only(["agent_spawn"]));
    chain.add(RateLimitMiddleware::new(0.001)); // Nearly zero rate

    // "resources/list" is not a tools/call, so filters and rate limit should not apply
    let request = method_request("resources/list");
    let mut ctx = RequestContext::new(json!(1));

    let result = chain.process_request(&request, &mut ctx).await;
    assert!(result.is_ok());
}

/// Test that the initialize method bypasses auth.
#[tokio::test]
async fn initialize_bypasses_auth() {
    let mut chain = MiddlewareChain::new();
    chain.add(AuthMiddleware::new("secret"));
    chain.add(ToolFilterMiddleware::allow_only(["agent_spawn"]));

    let request = method_request("initialize");
    let mut ctx = RequestContext::new(json!(1));

    let result = chain.process_request(&request, &mut ctx).await;
    assert!(result.is_ok(), "initialize should bypass auth");
}

/// Test that auth token set in metadata persists across requests in the same context.
#[tokio::test]
async fn auth_token_persists_in_context_metadata() {
    let auth = AuthMiddleware::new("persistent-token");

    // First request: token in params
    let request_1 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "agent_list",
            "_auth_token": "persistent-token"
        })),
    };

    let mut ctx = RequestContext::new(json!(1));
    let result = auth.process_request(&request_1, &mut ctx).await;
    assert!(matches!(result, MiddlewareResult::Continue));

    // Token should now be in context metadata
    assert!(ctx.get_metadata("auth_token").is_some());

    // Second request: no token in params, but context has it
    let request_2 = tool_call_request("agent_status");
    let result = auth.process_request(&request_2, &mut ctx).await;
    assert!(matches!(result, MiddlewareResult::Continue));
}

/// Test process_response runs through all middleware layers.
#[tokio::test]
async fn process_response_runs_all_layers() {
    /// A middleware that appends a marker to the response result.
    struct MarkerMiddleware {
        marker: String,
    }

    #[async_trait]
    impl Middleware for MarkerMiddleware {
        async fn process_request(
            &self,
            _request: &JsonRpcRequest,
            _ctx: &mut RequestContext,
        ) -> MiddlewareResult {
            MiddlewareResult::Continue
        }

        async fn process_response(&self, response: &mut JsonRpcResponse, _ctx: &RequestContext) {
            // Add marker to the response result
            if let Some(result) = &mut response.result
                && let Some(obj) = result.as_object_mut()
            {
                obj.insert(self.marker.clone(), serde_json::Value::Bool(true));
            }
        }
    }

    let mut chain = MiddlewareChain::new();
    chain.add(MarkerMiddleware {
        marker: "layer_1".into(),
    });
    chain.add(MarkerMiddleware {
        marker: "layer_2".into(),
    });

    let ctx = RequestContext::new(json!(1));
    let mut response = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        result: Some(json!({})),
        error: None,
    };

    chain.process_response(&mut response, &ctx).await;

    let result = response.result.unwrap();
    assert_eq!(result["layer_1"], true);
    assert_eq!(result["layer_2"], true);
}

/// Test rate limiter with per-tool limits lets frequent tools through while blocking others.
#[tokio::test]
async fn rate_limiter_per_tool_limits() {
    let rate_limiter = RateLimitMiddleware::new(1.0) // 1 req/sec global
        .with_tool_limit("fast_tool", 1000.0); // 1000 req/sec for fast_tool

    let mut chain = MiddlewareChain::new();
    chain.add(rate_limiter);

    // fast_tool should handle many requests
    for _ in 0..5 {
        let request = tool_call_request("fast_tool");
        let mut ctx = RequestContext::new(json!(1));
        let result = chain.process_request(&request, &mut ctx).await;
        assert!(result.is_ok(), "fast_tool should not be rate limited");
    }

    // slow_tool (uses global 1 req/sec) should get rate limited after first request
    let first = tool_call_request("slow_tool");
    let mut ctx = RequestContext::new(json!(1));
    assert!(chain.process_request(&first, &mut ctx).await.is_ok());

    let second = tool_call_request("slow_tool");
    let mut ctx = RequestContext::new(json!(2));
    let result = chain.process_request(&second, &mut ctx).await;
    assert!(
        result.is_err(),
        "slow_tool should be rate limited after first request"
    );
    assert_eq!(result.unwrap_err().code, -32002);
}
