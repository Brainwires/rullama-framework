//! `FailoverProvider` — a provider chain that walks through a list of
//! candidates on transient failure.
//!
//! Combine with `RetryProvider` for per-provider retry, then wrap the chain
//! in `FailoverProvider` to escape to the next provider once the inner
//! provider's retry budget is exhausted (or the failure is non-transient
//! but `treat_all_errors_as_transient` is set).
//!
//! The first provider to return `Ok` wins. Each provider is only attempted
//! once per call — re-trying the same provider belongs in `RetryProvider`.
//!
//! Errors from individual providers are collected into a
//! `ResilienceError::FailoverExhausted` so the caller can see why each
//! candidate was rejected.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use rullama_core::message::{ChatResponse, Message, StreamChunk};
use rullama_core::provider::{ChatOptions, Provider};
use rullama_core::tool::Tool;

use crate::classify::classify_error;
use crate::error::ResilienceError;

/// Decorator that walks through a list of providers, returning the first
/// success.
pub struct FailoverProvider {
    providers: Vec<Arc<dyn Provider>>,
    /// When `true`, every error advances to the next provider, regardless
    /// of classification. When `false` (default), only errors classified
    /// as transient advance — permanent errors (auth, schema, etc.) abort
    /// immediately so downstream callers can fix the root cause.
    treat_all_errors_as_transient: bool,
}

impl FailoverProvider {
    /// Build a chain that walks through `providers` in order. Errors must
    /// be transient (per `classify_error`) for the next provider to be
    /// tried; otherwise the chain aborts and returns the error from the
    /// provider that produced it.
    pub fn new(providers: Vec<Arc<dyn Provider>>) -> Self {
        assert!(
            !providers.is_empty(),
            "FailoverProvider requires at least one provider"
        );
        Self {
            providers,
            treat_all_errors_as_transient: false,
        }
    }

    /// Set whether every error should advance to the next provider (not
    /// just transient ones). Use when the chain is intentionally
    /// heterogeneous — e.g. primary is a paid endpoint and secondary is a
    /// local fallback you want to hit even if the primary returns
    /// auth/quota errors.
    pub fn with_treat_all_errors_as_transient(mut self, on: bool) -> Self {
        self.treat_all_errors_as_transient = on;
        self
    }

    /// Number of providers in the chain.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether the chain is empty (always `false` since `new` panics on
    /// empty input; kept for clippy ergonomics).
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

#[async_trait]
impl Provider for FailoverProvider {
    fn name(&self) -> &str {
        "failover"
    }

    fn max_output_tokens(&self) -> Option<u32> {
        // Use the most conservative cap across the chain so callers
        // requesting outputs that the worst-case backend can't deliver
        // still know up-front.
        self.providers
            .iter()
            .filter_map(|p| p.max_output_tokens())
            .min()
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let mut errors: Vec<(String, anyhow::Error)> = Vec::new();
        for (idx, provider) in self.providers.iter().enumerate() {
            match provider.chat(messages, tools, options).await {
                Ok(resp) => {
                    if idx > 0 {
                        tracing::info!(
                            failover_idx = idx,
                            provider = provider.name(),
                            "FailoverProvider succeeded on fallback {}",
                            idx
                        );
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    let class = classify_error(&e);
                    let advance = self.treat_all_errors_as_transient || class.is_retryable();
                    tracing::warn!(
                        failover_idx = idx,
                        provider = provider.name(),
                        class = ?class,
                        advance,
                        "FailoverProvider got error from {}: {e}",
                        provider.name()
                    );
                    let provider_name = provider.name().to_string();
                    errors.push((provider_name, e));
                    if !advance {
                        // Permanent error — abort the chain.
                        break;
                    }
                }
            }
        }
        Err(ResilienceError::FailoverExhausted {
            attempts: errors.len(),
            errors_summary: errors
                .iter()
                .map(|(name, e)| format!("{name}: {e}"))
                .collect::<Vec<_>>()
                .join("; "),
        }
        .into())
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        // Streaming failover is not supported: a partially-delivered stream
        // can't be retried on a different provider without breaking
        // chunk-ordering invariants on the consumer side. Defer to the
        // first provider; callers who want streaming failover should
        // implement application-level cutover.
        self.providers[0].stream_chat(messages, tools, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_util::EchoProvider;

    fn ok(name: &'static str) -> Arc<dyn Provider> {
        Arc::new(EchoProvider::ok(name))
    }
    fn transient_err(name: &'static str) -> Arc<dyn Provider> {
        Arc::new(EchoProvider::always_err(name, "connection reset by peer"))
    }
    fn permanent_err(name: &'static str) -> Arc<dyn Provider> {
        Arc::new(EchoProvider::always_err(name, "401 unauthorized"))
    }

    #[tokio::test]
    async fn first_success_wins() {
        let f = FailoverProvider::new(vec![ok("primary"), ok("secondary")]);
        let resp = f
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn falls_through_to_secondary_on_transient() {
        let f = FailoverProvider::new(vec![transient_err("primary"), ok("secondary")]);
        let resp = f
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn aborts_on_permanent_by_default() {
        let f = FailoverProvider::new(vec![permanent_err("primary"), ok("secondary")]);
        let err = f
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await
            .unwrap_err();
        let msg = err.to_string();
        // Crucial: the secondary's name must not appear (it wasn't tried).
        assert!(!msg.contains("secondary"), "got: {msg}");
        assert!(msg.contains("primary") || msg.contains("unauthorized"));
    }

    #[tokio::test]
    async fn treat_all_as_transient_bypasses_classification() {
        let f = FailoverProvider::new(vec![permanent_err("primary"), ok("secondary")])
            .with_treat_all_errors_as_transient(true);
        let resp = f
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn exhausts_with_collected_errors() {
        let f = FailoverProvider::new(vec![transient_err("primary"), transient_err("secondary")]);
        let err = f
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("primary"));
        assert!(msg.contains("secondary"));
    }
}
