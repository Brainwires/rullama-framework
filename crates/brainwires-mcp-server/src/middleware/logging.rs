use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcRequest, JsonRpcResponse};

use super::{Middleware, MiddlewareResult};
use crate::connection::RequestContext;

/// Middleware that logs all requests and responses.
pub struct LoggingMiddleware;

impl LoggingMiddleware {
    /// Create a new logging middleware.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for LoggingMiddleware {
    async fn process_request(
        &self,
        request: &JsonRpcRequest,
        _ctx: &mut RequestContext,
    ) -> MiddlewareResult {
        tracing::debug!(
            method = %request.method,
            id = %request.id,
            "MCP request received"
        );
        MiddlewareResult::Continue
    }

    async fn process_response(&self, response: &mut JsonRpcResponse, _ctx: &RequestContext) {
        if response.error.is_some() {
            tracing::warn!(
                id = %response.id,
                error = ?response.error,
                "MCP response with error"
            );
        } else {
            tracing::debug!(
                id = %response.id,
                "MCP response sent"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::RequestContext;
    use serde_json::json;

    #[tokio::test]
    async fn logging_middleware_always_continues() {
        let middleware = LoggingMiddleware::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tools/list".to_string(),
            params: None,
        };
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn logging_middleware_process_response_does_not_panic() {
        let middleware = LoggingMiddleware;
        let mut response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: Some(json!({})),
            error: None,
        };
        let ctx = RequestContext::new(json!(1));
        middleware.process_response(&mut response, &ctx).await;
    }

    #[tokio::test]
    async fn logging_middleware_handles_error_response() {
        let middleware = LoggingMiddleware::new();
        let mut response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: None,
            error: Some(brainwires_mcp_client::JsonRpcError {
                code: -32603,
                message: "internal error".to_string(),
                data: None,
            }),
        };
        let ctx = RequestContext::new(json!(1));
        middleware.process_response(&mut response, &ctx).await;
    }
}
