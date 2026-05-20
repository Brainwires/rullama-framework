use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcError, JsonRpcRequest};

use super::{Middleware, MiddlewareResult};
use crate::connection::RequestContext;

/// Token-based authentication middleware.
pub struct AuthMiddleware {
    token: String,
}

impl AuthMiddleware {
    /// Create a new auth middleware with the given token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[async_trait]
impl Middleware for AuthMiddleware {
    async fn process_request(
        &self,
        request: &JsonRpcRequest,
        ctx: &mut RequestContext,
    ) -> MiddlewareResult {
        // Skip auth for initialize - clients haven't authenticated yet
        if request.method == "initialize" {
            return MiddlewareResult::Continue;
        }

        // Check for token in metadata (set during initialize)
        if let Some(serde_json::Value::String(token)) = ctx.get_metadata("auth_token")
            && token == &self.token
        {
            return MiddlewareResult::Continue;
        }

        // Check params for auth token
        if let Some(params) = &request.params
            && let Some(token) = params.get("_auth_token").and_then(|v| v.as_str())
            && token == self.token
        {
            ctx.set_metadata(
                "auth_token".to_string(),
                serde_json::Value::String(token.to_string()),
            );
            return MiddlewareResult::Continue;
        }

        MiddlewareResult::Reject(JsonRpcError {
            code: -32003,
            message: "Unauthorized: invalid or missing auth token".to_string(),
            data: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::RequestContext;
    use serde_json::json;

    fn make_request(method: &str, params: Option<serde_json::Value>) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: method.to_string(),
            params,
        }
    }

    #[tokio::test]
    async fn initialize_method_skips_auth() {
        let middleware = AuthMiddleware::new("secret");
        let req = make_request("initialize", None);
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn valid_token_in_params_passes() {
        let middleware = AuthMiddleware::new("my-token");
        let req = make_request("tools/list", Some(json!({"_auth_token": "my-token"})));
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn invalid_token_in_params_rejects() {
        let middleware = AuthMiddleware::new("correct-token");
        let req = make_request("tools/list", Some(json!({"_auth_token": "wrong-token"})));
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Reject(_)));
    }

    #[tokio::test]
    async fn no_token_rejects() {
        let middleware = AuthMiddleware::new("secret");
        let req = make_request("tools/list", None);
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Reject(ref e) if e.code == -32003));
    }

    #[tokio::test]
    async fn token_cached_in_metadata_allows_subsequent_requests() {
        let middleware = AuthMiddleware::new("secret");
        let mut ctx = RequestContext::new(json!(1));
        ctx.set_metadata("auth_token".to_string(), json!("secret"));
        let req = make_request("tools/list", None);
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }
}
