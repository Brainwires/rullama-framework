use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcError, JsonRpcRequest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Instant;

use super::{Middleware, MiddlewareResult};
use crate::connection::RequestContext;

struct RateLimitBucket {
    tokens: f64,
    last_refill: Instant,
}

/// Token-bucket rate limiting middleware.
pub struct RateLimitMiddleware {
    max_requests_per_second: f64,
    per_tool_limits: HashMap<String, f64>,
    buckets: Arc<Mutex<HashMap<String, RateLimitBucket>>>,
}

impl RateLimitMiddleware {
    /// Create a new rate limiter with a global requests-per-second limit.
    pub fn new(max_requests_per_second: f64) -> Self {
        Self {
            max_requests_per_second,
            per_tool_limits: HashMap::new(),
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set a per-tool rate limit override.
    pub fn with_tool_limit(mut self, tool_name: &str, limit: f64) -> Self {
        self.per_tool_limits.insert(tool_name.to_string(), limit);
        self
    }

    fn get_limit(&self, key: &str) -> f64 {
        self.per_tool_limits
            .get(key)
            .copied()
            .unwrap_or(self.max_requests_per_second)
    }
}

#[async_trait]
impl Middleware for RateLimitMiddleware {
    async fn process_request(
        &self,
        request: &JsonRpcRequest,
        _ctx: &mut RequestContext,
    ) -> MiddlewareResult {
        // Only rate-limit tools/call
        if request.method != "tools/call" {
            return MiddlewareResult::Continue;
        }

        let tool_name = request
            .params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown");

        let limit = self.get_limit(tool_name);
        let key = format!("tool:{tool_name}");

        let mut buckets = self.buckets.lock().await;
        let bucket = buckets.entry(key).or_insert(RateLimitBucket {
            tokens: limit,
            last_refill: Instant::now(),
        });

        // Token bucket refill
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * limit).min(limit);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            MiddlewareResult::Continue
        } else {
            MiddlewareResult::Reject(JsonRpcError {
                code: -32002,
                message: format!("Rate limited: too many requests for tool '{tool_name}'"),
                data: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::RequestContext;
    use serde_json::json;

    fn tools_call_request(tool_name: &str) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tools/call".to_string(),
            params: Some(json!({"name": tool_name})),
        }
    }

    fn non_tool_request() -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "tools/list".to_string(),
            params: None,
        }
    }

    #[tokio::test]
    async fn non_tool_call_passes_rate_limiter() {
        let middleware = RateLimitMiddleware::new(1.0);
        let req = non_tool_request();
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn first_tool_call_passes() {
        let middleware = RateLimitMiddleware::new(10.0);
        let req = tools_call_request("my_tool");
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn zero_rate_limit_rejects_immediately() {
        // A limit of 0 means no tokens are ever available after the first call
        // Initial bucket starts with `limit` tokens, so first call with limit=0 has 0 tokens
        let middleware = RateLimitMiddleware::new(0.0);
        let req = tools_call_request("slow_tool");
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Reject(ref e) if e.code == -32002));
    }

    #[tokio::test]
    async fn per_tool_limit_override_applied() {
        let middleware = RateLimitMiddleware::new(100.0).with_tool_limit("special_tool", 0.0);
        let req = tools_call_request("special_tool");
        let mut ctx = RequestContext::new(json!(1));
        let result = middleware.process_request(&req, &mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Reject(_)));
    }

    #[test]
    fn get_limit_uses_default_when_no_override() {
        let middleware = RateLimitMiddleware::new(5.0);
        assert_eq!(middleware.get_limit("any_tool"), 5.0);
    }

    #[test]
    fn get_limit_uses_override_when_set() {
        let middleware = RateLimitMiddleware::new(5.0).with_tool_limit("special", 2.0);
        assert_eq!(middleware.get_limit("special"), 2.0);
        assert_eq!(middleware.get_limit("other"), 5.0);
    }
}
