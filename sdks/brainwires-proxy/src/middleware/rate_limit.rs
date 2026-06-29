//! Token-bucket rate limiter middleware.

use crate::error::ProxyResult;
use crate::middleware::{LayerAction, ProxyLayer};
use crate::types::{ProxyRequest, ProxyResponse};
use http::StatusCode;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

/// Token-bucket rate limiter.
pub struct RateLimitLayer {
    bucket: Arc<Mutex<TokenBucket>>,
}

struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

impl RateLimitLayer {
    /// Create a rate limiter allowing `capacity` burst requests
    /// with a sustained rate of `per_second` requests/sec.
    pub fn new(capacity: f64, per_second: f64) -> Self {
        Self {
            bucket: Arc::new(Mutex::new(TokenBucket::new(capacity, per_second))),
        }
    }
}

#[async_trait::async_trait]
impl ProxyLayer for RateLimitLayer {
    async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction> {
        let mut bucket = self.bucket.lock().await;
        if bucket.try_acquire() {
            Ok(LayerAction::Forward(request))
        } else {
            tracing::warn!(request_id = %request.id, "rate limited");
            Ok(LayerAction::Respond(
                ProxyResponse::for_request(request.id, StatusCode::TOO_MANY_REQUESTS)
                    .with_body("Rate limit exceeded"),
            ))
        }
    }

    fn name(&self) -> &str {
        "rate_limit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::Method;

    fn make_request() -> ProxyRequest {
        ProxyRequest::new(Method::GET, "/test".parse().unwrap())
    }

    #[tokio::test]
    async fn allows_within_capacity() {
        let limiter = RateLimitLayer::new(3.0, 1.0);
        // First 3 should pass (burst capacity)
        for _ in 0..3 {
            let result = limiter.on_request(make_request()).await.unwrap();
            assert!(matches!(result, LayerAction::Forward(_)));
        }
    }

    #[tokio::test]
    async fn rejects_over_capacity() {
        let limiter = RateLimitLayer::new(2.0, 0.0); // 2 burst, no refill
        // Consume both tokens
        limiter.on_request(make_request()).await.unwrap();
        limiter.on_request(make_request()).await.unwrap();

        // Third should be rejected
        let result = limiter.on_request(make_request()).await.unwrap();
        match result {
            LayerAction::Respond(resp) => {
                assert_eq!(resp.status, StatusCode::TOO_MANY_REQUESTS);
            }
            LayerAction::Forward(_) => panic!("should have been rate limited"),
        }
    }

    #[tokio::test]
    async fn refills_over_time() {
        let limiter = RateLimitLayer::new(1.0, 100.0); // 1 burst, 100/sec refill
        // Consume token
        limiter.on_request(make_request()).await.unwrap();

        // Wait for refill
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Should have refilled
        let result = limiter.on_request(make_request()).await.unwrap();
        assert!(matches!(result, LayerAction::Forward(_)));
    }
}
