//! End-to-end resilience decorator integration tests.
//!
//! Exercises the public API using simple in-process `Provider` mocks defined
//! inline (the in-crate `tests_util` module is `#[cfg(test)]` and not reachable
//! from integration-test targets).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_core::message::{ChatResponse, Message, StreamChunk, Usage};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;

use brainwires_call_policy::{
    BudgetConfig, BudgetGuard, BudgetProvider, CircuitBreakerConfig, CircuitBreakerProvider,
    ResilienceError, RetryPolicy, RetryProvider,
};

struct CountingProvider {
    name: &'static str,
    failures_remaining: AtomicU32,
    err_msg: &'static str,
    calls: AtomicU32,
}

impl CountingProvider {
    fn new(name: &'static str, failures: u32, err_msg: &'static str) -> Self {
        Self {
            name,
            failures_remaining: AtomicU32::new(failures),
            err_msg,
            calls: AtomicU32::new(0),
        }
    }
    fn calls(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Provider for CountingProvider {
    fn name(&self) -> &str {
        self.name
    }
    async fn chat(
        &self,
        _: &[Message],
        _: Option<&[Tool]>,
        _: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let left = self.failures_remaining.fetch_sub(1, Ordering::Relaxed);
        if left > 0 {
            return Err(anyhow::anyhow!("{}", self.err_msg));
        }
        self.failures_remaining.store(0, Ordering::Relaxed);
        Ok(ChatResponse {
            message: Message::assistant("ok"),
            usage: Usage::new(50, 50),
            finish_reason: Some("stop".into()),
        })
    }
    fn stream_chat<'a>(
        &'a self,
        _: &'a [Message],
        _: Option<&'a [Tool]>,
        _: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        Box::pin(futures::stream::empty())
    }
}

#[tokio::test]
async fn retry_succeeds_after_transient_429s() {
    let inner = Arc::new(CountingProvider::new("t", 2, "429 rate limit"));
    let wrapped = RetryProvider::new(
        inner.clone(),
        RetryPolicy {
            max_attempts: 5,
            base: Duration::from_millis(1),
            max: Duration::from_millis(4),
            jitter: 0.0,
            honor_retry_after: false,
            overall_deadline: None,
        },
    );
    let resp = wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .expect("retry should recover");
    assert_eq!(resp.message.text(), Some("ok"));
    assert_eq!(inner.calls(), 3, "two failures + one success");
}

#[tokio::test]
async fn retry_gives_up_with_typed_error() {
    let inner = Arc::new(CountingProvider::new("t", 10, "429 rate limit"));
    let wrapped = RetryProvider::new(
        inner.clone(),
        RetryPolicy {
            max_attempts: 3,
            base: Duration::from_millis(1),
            max: Duration::from_millis(2),
            jitter: 0.0,
            honor_retry_after: false,
            overall_deadline: None,
        },
    );
    let err = wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        err.downcast_ref::<ResilienceError>(),
        Some(ResilienceError::RetriesExhausted { attempts: 3, .. }),
    ));
    assert_eq!(inner.calls(), 3);
}

#[tokio::test]
async fn retry_skips_non_retryable() {
    let inner = Arc::new(CountingProvider::new("t", 5, "401 Unauthorized"));
    let wrapped = RetryProvider::new(
        inner.clone(),
        RetryPolicy {
            max_attempts: 4,
            base: Duration::from_millis(1),
            max: Duration::from_millis(2),
            jitter: 0.0,
            honor_retry_after: false,
            overall_deadline: None,
        },
    );
    assert!(
        wrapped
            .chat(&[], None, &ChatOptions::default())
            .await
            .is_err()
    );
    assert_eq!(inner.calls(), 1, "auth errors must not be retried");
}

#[tokio::test]
async fn budget_caps_tokens_post_flight() {
    let inner = Arc::new(CountingProvider::new("t", 0, ""));
    let guard = BudgetGuard::new(BudgetConfig {
        max_tokens: Some(120),
        ..Default::default()
    });
    let wrapped = BudgetProvider::new(inner.clone(), guard.clone());
    wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .unwrap(); // 100 tokens consumed
    // Second call pre-check passes (100 < 120), post-accumulates to 200.
    wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .unwrap();
    assert_eq!(guard.tokens_consumed(), 200);

    // Third call fails pre-flight.
    let err = wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        err.downcast_ref::<ResilienceError>(),
        Some(ResilienceError::BudgetExceeded { kind: "tokens", .. }),
    ));
}

#[tokio::test]
async fn stacked_retry_over_budget_composes_cleanly() {
    let inner = Arc::new(CountingProvider::new("t", 1, "503 service unavailable"));
    let guard = BudgetGuard::new(BudgetConfig {
        max_rounds: Some(10),
        ..Default::default()
    });
    let budget = Arc::new(BudgetProvider::new(inner.clone(), guard.clone()));
    let wrapped = RetryProvider::new(
        budget,
        RetryPolicy {
            max_attempts: 3,
            base: Duration::from_millis(1),
            max: Duration::from_millis(2),
            jitter: 0.0,
            honor_retry_after: false,
            overall_deadline: None,
        },
    );
    wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .expect("should succeed after one retry");
    assert_eq!(inner.calls(), 2, "one 503 + one success");
    assert_eq!(
        guard.rounds_consumed(),
        2,
        "budget ticks once per inner call"
    );
}

#[tokio::test]
async fn circuit_breaker_opens_then_recovers() {
    let inner = Arc::new(CountingProvider::new("t", 100, "500 internal server error"));
    let cb = CircuitBreakerProvider::new(
        inner.clone(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            cooldown: Duration::from_millis(20),
        },
    );

    for _ in 0..2 {
        let _ = cb.chat(&[], None, &ChatOptions::default()).await;
    }
    // Third call: circuit open → fast-fail, inner not called.
    let before = inner.calls();
    let err = cb
        .chat(&[], None, &ChatOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        err.downcast_ref::<ResilienceError>(),
        Some(ResilienceError::CircuitOpen { .. }),
    ));
    assert_eq!(inner.calls(), before, "inner not re-invoked while open");
}
