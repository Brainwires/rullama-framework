//! Rate-limited HTTP client wrapper.
//!
//! Wraps `reqwest::Client` with an optional [`RateLimiter`] so that all
//! outgoing requests are throttled according to the configured limit.

use reqwest::{Client, RequestBuilder};
use std::sync::Arc;

use super::rate_limiter::RateLimiter;

/// An HTTP client that optionally enforces rate limiting on every request.
#[derive(Clone, Debug)]
pub struct RateLimitedClient {
    client: Client,
    limiter: Option<Arc<RateLimiter>>,
}

impl RateLimitedClient {
    /// Create a new client with no rate limiting.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            limiter: None,
        }
    }

    /// Create a new client with the given requests-per-minute limit.
    pub fn with_rate_limit(requests_per_minute: u32) -> Self {
        Self {
            client: Client::new(),
            limiter: Some(Arc::new(RateLimiter::new(requests_per_minute))),
        }
    }

    /// Create from an existing `reqwest::Client`, adding a rate limiter.
    pub fn from_client(client: Client, requests_per_minute: Option<u32>) -> Self {
        Self {
            client,
            limiter: requests_per_minute.map(|rpm| Arc::new(RateLimiter::new(rpm))),
        }
    }

    /// Get a reference to the inner `reqwest::Client`.
    pub fn inner(&self) -> &Client {
        &self.client
    }

    /// Build a GET request, waiting for rate-limit clearance first.
    pub async fn get(&self, url: &str) -> RequestBuilder {
        self.wait_for_token().await;
        self.client.get(url)
    }

    /// Build a POST request, waiting for rate-limit clearance first.
    pub async fn post(&self, url: &str) -> RequestBuilder {
        self.wait_for_token().await;
        self.client.post(url)
    }

    /// Wait for a rate-limit token (no-op if no limiter is configured).
    async fn wait_for_token(&self) {
        if let Some(ref limiter) = self.limiter {
            limiter.acquire().await;
        }
    }

    /// Get the current number of available tokens (for diagnostics).
    /// Returns `None` if no rate limiter is configured.
    pub fn available_tokens(&self) -> Option<u32> {
        self.limiter.as_ref().map(|l| l.available_tokens())
    }
}

impl Default for RateLimitedClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_client() {
        let client = RateLimitedClient::new();
        assert!(client.available_tokens().is_none());
    }

    #[test]
    fn test_rate_limited_client() {
        let client = RateLimitedClient::with_rate_limit(60);
        assert_eq!(client.available_tokens(), Some(60));
    }

    #[tokio::test]
    async fn test_post_consumes_token() {
        let client = RateLimitedClient::with_rate_limit(10);
        let _req = client.post("https://example.com").await;
        assert_eq!(client.available_tokens(), Some(9));
    }
}
