//! Verifies that `RetryPolicy.overall_deadline` bounds the total retry window
//! and surfaces a typed `ResilienceError::DeadlineExceeded`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_core::message::{ChatResponse, Message, StreamChunk};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;

use brainwires_call_policy::{ResilienceError, RetryPolicy, RetryProvider};

struct AlwaysTransientProvider {
    calls: AtomicU32,
}

impl AlwaysTransientProvider {
    fn new() -> Self {
        Self {
            calls: AtomicU32::new(0),
        }
    }
    fn calls(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Provider for AlwaysTransientProvider {
    fn name(&self) -> &str {
        "always-transient"
    }
    async fn chat(
        &self,
        _: &[Message],
        _: Option<&[Tool]>,
        _: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(anyhow::anyhow!("503 service unavailable"))
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
async fn deadline_short_circuits_retry_loop() {
    let inner = Arc::new(AlwaysTransientProvider::new());
    let wrapped = RetryProvider::new(
        inner.clone(),
        RetryPolicy {
            max_attempts: 100,
            base: Duration::from_millis(50),
            max: Duration::from_millis(200),
            jitter: 0.0,
            honor_retry_after: false,
            overall_deadline: Some(Duration::from_millis(150)),
        },
    );

    let started = Instant::now();
    let err = wrapped
        .chat(&[], None, &ChatOptions::default())
        .await
        .expect_err("provider always errors, retry must give up");
    let elapsed = started.elapsed();

    let typed = err
        .downcast_ref::<ResilienceError>()
        .expect("typed ResilienceError");
    match typed {
        ResilienceError::DeadlineExceeded { attempts, .. } => {
            assert!(
                *attempts >= 1 && *attempts < 100,
                "expected deadline to short-circuit before max_attempts, got {attempts}"
            );
        }
        other => panic!("expected DeadlineExceeded, got {other:?}"),
    }

    assert!(
        elapsed < Duration::from_millis(500),
        "deadline=150ms but call took {}ms — loop did not honor deadline",
        elapsed.as_millis()
    );
    assert!(
        inner.calls() < 100,
        "deadline did not short-circuit; got {} calls",
        inner.calls()
    );
}

#[tokio::test]
async fn no_deadline_runs_full_retry_budget() {
    let inner = Arc::new(AlwaysTransientProvider::new());
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
        .expect_err("transient errors exhaust retries");
    assert!(matches!(
        err.downcast_ref::<ResilienceError>(),
        Some(ResilienceError::RetriesExhausted { attempts: 3, .. })
    ));
    assert_eq!(inner.calls(), 3);
}
