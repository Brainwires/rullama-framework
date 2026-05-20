//! Circuit-breaker decorator.
//!
//! Tracks consecutive failures per `(provider, model)` and opens the circuit
//! when a threshold is crossed. While open, calls fail fast with
//! [`ResilienceError::CircuitOpen`] instead of hitting the provider. After a
//! cooldown the breaker enters a half-open state: the next call is allowed
//! through; success closes the circuit, failure reopens it.
//!
//! An optional fallback provider can be supplied. If set, `chat` calls failing
//! the breaker route to the fallback instead of returning an error.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::sync::Mutex;

use brainwires_core::message::{ChatResponse, Message, StreamChunk};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;

use crate::error::ResilienceError;

/// Circuit-breaker state for a single provider/model key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — calls pass through.
    Closed,
    /// Too many failures. All calls are rejected until `open_until` elapses.
    Open,
    /// Cooldown elapsed. One probe call is permitted; its result determines
    /// whether we re-close or fall back to Open.
    HalfOpen,
}

/// Circuit-breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures required to open the circuit.
    pub failure_threshold: u32,
    /// How long a tripped circuit stays Open before entering HalfOpen.
    pub cooldown: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            cooldown: Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
struct Entry {
    state: CircuitState,
    failures: u32,
    open_until: Option<Instant>,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failures: 0,
            open_until: None,
        }
    }
}

/// Circuit-breaker decorator around any [`Provider`].
pub struct CircuitBreakerProvider<P: Provider + ?Sized> {
    inner: Arc<P>,
    fallback: Option<Arc<dyn Provider>>,
    cfg: CircuitBreakerConfig,
    state: Arc<Mutex<HashMap<String, Entry>>>,
}

impl<P: Provider + ?Sized> CircuitBreakerProvider<P> {
    /// Create a new breaker without a fallback.
    pub fn new(inner: Arc<P>, cfg: CircuitBreakerConfig) -> Self {
        Self {
            inner,
            fallback: None,
            cfg,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Attach a fallback provider used when the circuit is open.
    pub fn with_fallback(mut self, fallback: Arc<dyn Provider>) -> Self {
        self.fallback = Some(fallback);
        self
    }

    /// Inspect the current state for a given model key.
    pub async fn state_for(&self, model: &str) -> CircuitState {
        let guard = self.state.lock().await;
        let key = self.key(model);
        guard.get(&key).map_or(CircuitState::Closed, |e| e.state)
    }

    fn key(&self, model: &str) -> String {
        format!("{}::{}", self.inner.name(), model)
    }

    async fn transition_in(&self, key: &str) -> Result<(), ResilienceError> {
        let mut guard = self.state.lock().await;
        let entry = guard.entry(key.to_string()).or_default();
        if entry.state == CircuitState::Open {
            if let Some(deadline) = entry.open_until
                && Instant::now() >= deadline
            {
                entry.state = CircuitState::HalfOpen;
            } else {
                return Err(ResilienceError::CircuitOpen {
                    provider: self.inner.name().to_string(),
                    model: key.to_string(),
                    failures: entry.failures,
                });
            }
        }
        Ok(())
    }

    async fn record_success(&self, key: &str) {
        let mut guard = self.state.lock().await;
        let entry = guard.entry(key.to_string()).or_default();
        entry.state = CircuitState::Closed;
        entry.failures = 0;
        entry.open_until = None;
    }

    async fn record_failure(&self, key: &str) {
        let mut guard = self.state.lock().await;
        let entry = guard.entry(key.to_string()).or_default();
        entry.failures = entry.failures.saturating_add(1);
        if entry.failures >= self.cfg.failure_threshold || entry.state == CircuitState::HalfOpen {
            entry.state = CircuitState::Open;
            entry.open_until = Some(Instant::now() + self.cfg.cooldown);
        }
    }
}

#[async_trait]
impl<P: Provider + ?Sized + 'static> Provider for CircuitBreakerProvider<P> {
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
        let model_label = options.model.as_deref().unwrap_or("default");
        let key = self.key(model_label);

        if let Err(e) = self.transition_in(&key).await {
            if let Some(fallback) = &self.fallback {
                tracing::warn!(
                    provider = self.inner.name(),
                    model = model_label,
                    "circuit open; routing to fallback"
                );
                return fallback.chat(messages, tools, options).await;
            }
            return Err(e.into());
        }

        match self.inner.chat(messages, tools, options).await {
            Ok(resp) => {
                self.record_success(&key).await;
                Ok(resp)
            }
            Err(e) => {
                self.record_failure(&key).await;
                Err(e)
            }
        }
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        // Streaming bypasses the breaker. Classifying a partial stream as
        // "failure" is ambiguous (early chunks may succeed then the tail
        // errors); agents that care can wrap the stream themselves.
        self.inner.stream_chat(messages, tools, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn state_starts_closed() {
        use crate::tests_util::EchoProvider;
        let cb = CircuitBreakerProvider::new(
            Arc::new(EchoProvider::ok("p1")),
            CircuitBreakerConfig::default(),
        );
        assert_eq!(cb.state_for("any").await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn opens_after_threshold() {
        use crate::tests_util::EchoProvider;
        let cb = CircuitBreakerProvider::new(
            Arc::new(EchoProvider::always_err("p1", "500 internal server error")),
            CircuitBreakerConfig {
                failure_threshold: 3,
                cooldown: Duration::from_millis(50),
            },
        );
        let opts = ChatOptions::default();
        for _ in 0..3 {
            let _ = cb.chat(&[], None, &opts).await;
        }
        let key = cb.key("default");
        assert_eq!(
            cb.state.lock().await.get(&key).map(|e| e.state),
            Some(CircuitState::Open),
        );

        // Fast-fail while open.
        let err = cb.chat(&[], None, &opts).await.unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ResilienceError>(),
            Some(ResilienceError::CircuitOpen { .. }),
        ));
    }

    #[tokio::test]
    async fn half_open_then_closes_on_success() {
        use crate::tests_util::ToggleProvider;
        let prov = Arc::new(ToggleProvider::new("p1"));
        let cb = CircuitBreakerProvider::new(
            prov.clone(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                cooldown: Duration::from_millis(20),
            },
        );
        let opts = ChatOptions::default();

        // Fail twice to open.
        prov.set_fail(true);
        let _ = cb.chat(&[], None, &opts).await;
        let _ = cb.chat(&[], None, &opts).await;

        // Wait cooldown → half-open → success closes circuit.
        tokio::time::sleep(Duration::from_millis(30)).await;
        prov.set_fail(false);
        cb.chat(&[], None, &opts).await.expect("half-open success");

        assert_eq!(cb.state_for("default").await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn half_open_reopens_on_failure() {
        use crate::tests_util::ToggleProvider;
        let prov = Arc::new(ToggleProvider::new("p1"));
        let cb = CircuitBreakerProvider::new(
            prov.clone(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                cooldown: Duration::from_millis(20),
            },
        );
        let opts = ChatOptions::default();

        // Fail twice to open.
        prov.set_fail(true);
        let _ = cb.chat(&[], None, &opts).await;
        let _ = cb.chat(&[], None, &opts).await;
        assert_eq!(cb.state_for("default").await, CircuitState::Open);

        // Wait cooldown so the next call enters half-open. Provider still failing →
        // breaker must trip back to Open immediately, not require another threshold.
        tokio::time::sleep(Duration::from_millis(30)).await;
        let err = cb.chat(&[], None, &opts).await.unwrap_err();
        // The half-open trial call surfaces the provider's transient error
        // (not CircuitOpen) — that's the diagnostic signal the breaker re-tripped.
        assert!(
            err.to_string().contains("500"),
            "expected provider error from half-open trial, got: {err}"
        );
        assert_eq!(
            cb.state_for("default").await,
            CircuitState::Open,
            "half-open + failure must re-open the circuit",
        );

        // And the next call should be fast-failed by the now-open circuit.
        let err2 = cb.chat(&[], None, &opts).await.unwrap_err();
        assert!(matches!(
            err2.downcast_ref::<ResilienceError>(),
            Some(ResilienceError::CircuitOpen { .. })
        ));
    }
}
