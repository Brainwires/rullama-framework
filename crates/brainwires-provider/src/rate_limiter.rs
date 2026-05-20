//! Token-bucket rate limiter for API request throttling.
//!
//! Provides a simple, lock-free rate limiter that enforces a maximum number of
//! requests per minute using the token-bucket algorithm.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// A token-bucket rate limiter.
///
/// Tokens are refilled at a fixed rate. Each request consumes one token.
/// When no tokens are available, `acquire()` waits until one is refilled.
pub struct RateLimiter {
    tokens: AtomicU32,
    max_tokens: u32,
    refill_interval: Duration,
    last_refill: Mutex<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter with the given requests-per-minute limit.
    ///
    /// A limit of 0 means no requests are allowed.
    pub fn new(requests_per_minute: u32) -> Self {
        let refill_interval = if requests_per_minute > 0 {
            Duration::from_secs(60) / requests_per_minute
        } else {
            Duration::from_secs(u64::MAX / 2) // effectively infinite wait
        };

        Self {
            tokens: AtomicU32::new(requests_per_minute),
            max_tokens: requests_per_minute,
            refill_interval,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Wait until a token is available, then consume it.
    #[tracing::instrument(name = "provider.rate_limit", skip(self), fields(max_rpm = self.max_tokens))]
    pub async fn acquire(&self) {
        loop {
            // Try to consume a token
            let current = self.tokens.load(Ordering::Relaxed);
            if current > 0 {
                if self
                    .tokens
                    .compare_exchange(current, current - 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
                // CAS failed, retry
                continue;
            }

            // No tokens available — refill and wait
            self.refill();

            // If still no tokens, sleep for one refill interval
            if self.tokens.load(Ordering::Relaxed) == 0 {
                tokio::time::sleep(self.refill_interval).await;
                self.refill();
            }
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&self) {
        let mut last = self
            .last_refill
            .lock()
            .expect("rate limiter state lock poisoned");
        let elapsed = last.elapsed();
        let new_tokens = (elapsed.as_millis() / self.refill_interval.as_millis().max(1)) as u32;

        if new_tokens > 0 {
            let current = self.tokens.load(Ordering::Relaxed);
            let refilled = (current + new_tokens).min(self.max_tokens);
            self.tokens.store(refilled, Ordering::Release);
            *last = Instant::now();
        }
    }

    /// Get the current number of available tokens (for diagnostics).
    pub fn available_tokens(&self) -> u32 {
        self.tokens.load(Ordering::Relaxed)
    }

    /// Get the configured requests-per-minute limit.
    pub fn max_requests_per_minute(&self) -> u32 {
        self.max_tokens
    }
}

impl std::fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimiter")
            .field("max_tokens", &self.max_tokens)
            .field("available", &self.tokens.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new(60);
        assert_eq!(limiter.max_requests_per_minute(), 60);
        assert_eq!(limiter.available_tokens(), 60);
    }

    #[test]
    fn test_rate_limiter_zero() {
        let limiter = RateLimiter::new(0);
        assert_eq!(limiter.max_requests_per_minute(), 0);
        assert_eq!(limiter.available_tokens(), 0);
    }

    #[tokio::test]
    async fn test_acquire_consumes_token() {
        let limiter = RateLimiter::new(10);
        assert_eq!(limiter.available_tokens(), 10);

        limiter.acquire().await;
        assert_eq!(limiter.available_tokens(), 9);

        limiter.acquire().await;
        assert_eq!(limiter.available_tokens(), 8);
    }

    #[tokio::test]
    async fn test_multiple_acquires() {
        let limiter = RateLimiter::new(5);

        for _ in 0..5 {
            limiter.acquire().await;
        }

        assert_eq!(limiter.available_tokens(), 0);
    }
}
