/// Authentication middleware.
pub mod auth;
/// Request logging middleware.
pub mod logging;
/// OAuth 2.1 JWT validation middleware.
#[cfg(feature = "oauth")]
pub mod oauth;
/// Rate limiting middleware.
pub mod rate_limit;
/// Tool filtering middleware.
pub mod tool_filter;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

use crate::connection::RequestContext;

/// Result of middleware processing.
pub enum MiddlewareResult {
    /// Allow the request to continue.
    Continue,
    /// Reject the request with an error.
    Reject(JsonRpcError),
}

/// Trait for request/response middleware.
#[async_trait]
pub trait Middleware: Send + Sync + 'static {
    /// Process an incoming request. Return `Continue` or `Reject`.
    async fn process_request(
        &self,
        request: &JsonRpcRequest,
        ctx: &mut RequestContext,
    ) -> MiddlewareResult;

    /// Optionally process the outgoing response (no-op by default).
    async fn process_response(&self, _response: &mut JsonRpcResponse, _ctx: &RequestContext) {}
}

/// Ordered chain of middleware layers.
pub struct MiddlewareChain {
    layers: Vec<Box<dyn Middleware>>,
}

impl MiddlewareChain {
    /// Create a new empty middleware chain.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a middleware layer to the chain.
    pub fn add(&mut self, middleware: impl Middleware) {
        self.layers.push(Box::new(middleware));
    }

    /// Run all middleware on the request, stopping on first reject.
    pub async fn process_request(
        &self,
        request: &JsonRpcRequest,
        ctx: &mut RequestContext,
    ) -> Result<(), JsonRpcError> {
        for layer in &self.layers {
            match layer.process_request(request, ctx).await {
                MiddlewareResult::Continue => continue,
                MiddlewareResult::Reject(err) => return Err(err),
            }
        }
        Ok(())
    }

    /// Run all middleware on the response.
    pub async fn process_response(&self, response: &mut JsonRpcResponse, ctx: &RequestContext) {
        for layer in &self.layers {
            layer.process_response(response, ctx).await;
        }
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct PassMiddleware;

    #[async_trait]
    impl Middleware for PassMiddleware {
        async fn process_request(
            &self,
            _request: &JsonRpcRequest,
            _ctx: &mut RequestContext,
        ) -> MiddlewareResult {
            MiddlewareResult::Continue
        }
    }

    struct RejectMiddleware;

    #[async_trait]
    impl Middleware for RejectMiddleware {
        async fn process_request(
            &self,
            _request: &JsonRpcRequest,
            _ctx: &mut RequestContext,
        ) -> MiddlewareResult {
            MiddlewareResult::Reject(JsonRpcError {
                code: -32003,
                message: "Rejected".to_string(),
                data: None,
            })
        }
    }

    #[tokio::test]
    async fn test_chain_all_pass() {
        let mut chain = MiddlewareChain::new();
        chain.add(PassMiddleware);
        chain.add(PassMiddleware);

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "test".to_string(),
            params: None,
        };
        let mut ctx = RequestContext::new(json!(1));
        assert!(chain.process_request(&request, &mut ctx).await.is_ok());
    }

    #[tokio::test]
    async fn test_chain_reject_stops() {
        let mut chain = MiddlewareChain::new();
        chain.add(PassMiddleware);
        chain.add(RejectMiddleware);
        chain.add(PassMiddleware);

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "test".to_string(),
            params: None,
        };
        let mut ctx = RequestContext::new(json!(1));
        let result = chain.process_request(&request, &mut ctx).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, -32003);
    }
}
