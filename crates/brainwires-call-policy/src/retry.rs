//! Retry decorator with exponential backoff + jitter.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use rand::Rng;

use brainwires_core::message::{ChatResponse, Message, StreamChunk};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;

use crate::classify::{classify_error, parse_retry_after};
use crate::error::ResilienceError;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts including the first. `1` disables retry.
    pub max_attempts: u32,
    /// Base backoff. Effective delay is `base * 2^(attempt-1)` clamped to `max`.
    pub base: Duration,
    /// Upper bound on a single sleep.
    pub max: Duration,
    /// Proportional jitter applied to each delay (0.0..=1.0).
    pub jitter: f64,
    /// If true, honor `retry-after` hints embedded in error messages.
    pub honor_retry_after: bool,
    /// Hard wall-clock ceiling for the entire retry sequence. When set, the
    /// loop exits with [`ResilienceError::DeadlineExceeded`] once the elapsed
    /// time exceeds this value, regardless of `max_attempts`. `None` disables
    /// the ceiling — retries run up to `max_attempts` only.
    pub overall_deadline: Option<Duration>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base: Duration::from_millis(500),
            max: Duration::from_secs(30),
            jitter: 0.2,
            honor_retry_after: true,
            overall_deadline: Some(Duration::from_secs(60)),
        }
    }
}

impl RetryPolicy {
    /// Disable retries entirely.
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            ..Self::default()
        }
    }

    fn backoff_for(&self, attempt: u32) -> Duration {
        let shift = attempt.saturating_sub(1).min(16);
        let nominal = self.base.saturating_mul(1u32 << shift);
        let capped = nominal.min(self.max);
        apply_jitter(capped, self.jitter)
    }
}

fn apply_jitter(base: Duration, factor: f64) -> Duration {
    if factor <= 0.0 {
        return base;
    }
    let ms = base.as_millis() as f64;
    let spread = ms * factor.clamp(0.0, 1.0);
    let delta: f64 = rand::thread_rng().gen_range(-spread..=spread);
    let jittered = (ms + delta).max(0.0);
    Duration::from_millis(jittered as u64)
}

/// A `Provider` decorator that retries transient failures with exponential
/// backoff and optional jitter.
///
/// Wraps another `Provider`. Only non-streaming [`chat`](Provider::chat)
/// requests are retried — streaming passes through unchanged because a
/// partially-consumed stream cannot be safely replayed.
pub struct RetryProvider<P: Provider + ?Sized> {
    inner: Arc<P>,
    policy: RetryPolicy,
}

impl<P: Provider + ?Sized> RetryProvider<P> {
    /// Create a new retry wrapper with the given policy.
    pub fn new(inner: Arc<P>, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    /// Access the wrapped provider.
    pub fn inner(&self) -> &Arc<P> {
        &self.inner
    }
}

#[async_trait]
impl<P: Provider + ?Sized + 'static> Provider for RetryProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn max_output_tokens(&self) -> Option<u32> {
        self.inner.max_output_tokens()
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let mut last_err: Option<anyhow::Error> = None;
        let started = Instant::now();

        for attempt in 1..=self.policy.max_attempts {
            match self.inner.chat(messages, tools, options).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    let class = classify_error(&e);
                    if !class.is_retryable() || attempt == self.policy.max_attempts {
                        if attempt > 1 {
                            return Err(ResilienceError::RetriesExhausted {
                                attempts: attempt,
                                source: e,
                            }
                            .into());
                        }
                        return Err(e);
                    }

                    let mut delay = if self.policy.honor_retry_after {
                        parse_retry_after(&e).unwrap_or_else(|| self.policy.backoff_for(attempt))
                    } else {
                        self.policy.backoff_for(attempt)
                    };

                    if let Some(deadline) = self.policy.overall_deadline {
                        let elapsed = started.elapsed();
                        if elapsed >= deadline || elapsed.saturating_add(delay) >= deadline {
                            tracing::warn!(
                                provider = self.inner.name(),
                                attempt,
                                elapsed_ms = elapsed.as_millis() as u64,
                                deadline_ms = deadline.as_millis() as u64,
                                "retry deadline reached, giving up"
                            );
                            return Err(ResilienceError::DeadlineExceeded {
                                attempts: attempt,
                                elapsed_ms: elapsed.as_millis() as u64,
                                source: e,
                            }
                            .into());
                        }
                        // Cap the next sleep so it doesn't overshoot the deadline.
                        let remaining = deadline.saturating_sub(elapsed);
                        if delay > remaining {
                            delay = remaining;
                        }
                    }

                    tracing::warn!(
                        provider = self.inner.name(),
                        attempt,
                        delay_ms = delay.as_millis() as u64,
                        class = ?class,
                        "retrying provider call after transient error"
                    );
                    last_err = Some(e);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        Err(ResilienceError::RetriesExhausted {
            attempts: self.policy.max_attempts,
            source: last_err.unwrap_or_else(|| anyhow::anyhow!("unknown retry failure")),
        }
        .into())
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        // Streaming responses are not retried: a partially-delivered stream
        // would be user-visible, and we cannot replay one safely. Upstream
        // retry logic belongs at the `chat` call site.
        self.inner.stream_chat(messages, tools, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_sensible() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_attempts, 4);
        assert!(p.honor_retry_after);
    }

    #[test]
    fn backoff_doubles_then_caps() {
        let p = RetryPolicy {
            max_attempts: 10,
            base: Duration::from_millis(100),
            max: Duration::from_millis(800),
            jitter: 0.0,
            honor_retry_after: false,
            overall_deadline: None,
        };
        assert_eq!(p.backoff_for(1), Duration::from_millis(100));
        assert_eq!(p.backoff_for(2), Duration::from_millis(200));
        assert_eq!(p.backoff_for(3), Duration::from_millis(400));
        assert_eq!(p.backoff_for(4), Duration::from_millis(800));
        assert_eq!(p.backoff_for(5), Duration::from_millis(800));
    }

    #[test]
    fn none_policy_disables_retry() {
        assert_eq!(RetryPolicy::none().max_attempts, 1);
    }

    #[test]
    fn jitter_stays_within_bounds() {
        let base = Duration::from_millis(1000);
        for _ in 0..50 {
            let j = apply_jitter(base, 0.2);
            assert!(j >= Duration::from_millis(800) && j <= Duration::from_millis(1200));
        }
    }
}
