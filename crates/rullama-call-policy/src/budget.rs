//! Budget decorator — atomic caps on tokens, USD, and rounds.
//!
//! A [`BudgetGuard`] holds atomic counters shared across tasks. Wrap a provider
//! with [`BudgetProvider`] so every `chat` / `stream_chat` call goes through
//! the guard: post-flight we accumulate `Usage` from the response (or from
//! `StreamChunk::Usage` during streaming), and pre-flight we reject if a
//! configured cap has already been reached.
//!
//! Pre-flight token projection is deliberately coarse (lower bound only),
//! because token counting differs per provider. A cheap heuristic — counting
//! characters in the pending messages — is used to estimate whether the
//! request itself would push us over `max_tokens`. For exact counting, use a
//! `ModelTokenizer` in a future change.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::Mutex;

use rullama_core::message::{ChatResponse, Message, StreamChunk, Usage};
use rullama_core::provider::{ChatOptions, Provider};
use rullama_core::tool::Tool;

use crate::error::ResilienceError;
use crate::tokenizer::{HeuristicTokenizer, Tokenizer};

/// Caps to enforce on a single [`BudgetGuard`].
///
/// `None` means unbounded for that dimension.
#[derive(Debug, Clone, Default)]
pub struct BudgetConfig {
    /// Maximum total tokens (prompt + completion) across the guard's lifetime.
    pub max_tokens: Option<u64>,
    /// Maximum spend in USD cents.
    pub max_usd_cents: Option<u64>,
    /// Maximum number of provider calls (agent rounds).
    pub max_rounds: Option<u64>,
}

/// Shared atomic budget counters.
///
/// Cheap to `clone` (just an `Arc` bump) so the same guard can be shared across
/// multiple providers, agent runs, or concurrent tasks.
#[derive(Clone, Debug)]
pub struct BudgetGuard {
    cfg: BudgetConfig,
    state: Arc<BudgetState>,
}

#[derive(Debug, Default)]
struct BudgetState {
    tokens: AtomicU64,
    usd_cents: AtomicU64,
    rounds: AtomicU64,
}

impl BudgetGuard {
    /// Create a fresh guard with the given caps.
    pub fn new(cfg: BudgetConfig) -> Self {
        Self {
            cfg,
            state: Arc::new(BudgetState::default()),
        }
    }

    /// Currently observed configuration.
    pub fn config(&self) -> &BudgetConfig {
        &self.cfg
    }

    /// Total tokens accumulated so far.
    pub fn tokens_consumed(&self) -> u64 {
        self.state.tokens.load(Ordering::Relaxed)
    }

    /// Total spend (in USD cents) accumulated so far.
    pub fn usd_cents_consumed(&self) -> u64 {
        self.state.usd_cents.load(Ordering::Relaxed)
    }

    /// Total provider calls attempted so far.
    pub fn rounds_consumed(&self) -> u64 {
        self.state.rounds.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.state.tokens.store(0, Ordering::Relaxed);
        self.state.usd_cents.store(0, Ordering::Relaxed);
        self.state.rounds.store(0, Ordering::Relaxed);
    }

    /// Pre-flight check. Rejects with [`ResilienceError::BudgetExceeded`] if
    /// any cap has already been reached. Use this inside agent loops to stop
    /// before spending more.
    pub fn check(&self) -> Result<(), ResilienceError> {
        if let Some(limit) = self.cfg.max_tokens {
            let consumed = self.tokens_consumed();
            if consumed >= limit {
                return Err(ResilienceError::BudgetExceeded {
                    kind: "tokens",
                    consumed,
                    limit,
                });
            }
        }
        if let Some(limit) = self.cfg.max_usd_cents {
            let consumed = self.usd_cents_consumed();
            if consumed >= limit {
                return Err(ResilienceError::BudgetExceeded {
                    kind: "usd_cents",
                    consumed,
                    limit,
                });
            }
        }
        if let Some(limit) = self.cfg.max_rounds {
            let consumed = self.rounds_consumed();
            if consumed >= limit {
                return Err(ResilienceError::BudgetExceeded {
                    kind: "rounds",
                    consumed,
                    limit,
                });
            }
        }
        Ok(())
    }

    /// Atomically check caps and tick the `rounds` counter. Intended to be
    /// called once per agent iteration.
    pub fn check_and_tick(&self) -> Result<(), ResilienceError> {
        self.check()?;
        self.state.rounds.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Accumulate observed usage into the counters.
    pub fn record_usage(&self, usage: &Usage) {
        self.state
            .tokens
            .fetch_add(usage.total_tokens as u64, Ordering::Relaxed);
    }

    /// Accumulate observed spend (in USD cents).
    pub fn record_cost_cents(&self, cents: u64) {
        self.state.usd_cents.fetch_add(cents, Ordering::Relaxed);
    }
}

/// A `Provider` decorator that enforces a [`BudgetGuard`] around every call.
///
/// The pre-flight token check uses a pluggable [`Tokenizer`] so callers can
/// trade off accuracy vs. dependency weight. The default `new` ctor wires
/// up a [`HeuristicTokenizer`] (chars / 4) for backwards-compatible
/// behaviour; for exact per-provider counting use `with_tokenizer`.
pub struct BudgetProvider<P: Provider + ?Sized> {
    inner: Arc<P>,
    guard: BudgetGuard,
    tokenizer: Arc<dyn Tokenizer>,
}

impl<P: Provider + ?Sized> BudgetProvider<P> {
    /// Wrap a provider with a budget guard. Token pre-flight uses the
    /// cheap chars/4 heuristic; for provider-accurate counts use
    /// [`Self::with_tokenizer`].
    pub fn new(inner: Arc<P>, guard: BudgetGuard) -> Self {
        Self {
            inner,
            guard,
            tokenizer: Arc::new(HeuristicTokenizer),
        }
    }

    /// Wrap a provider with a budget guard and a specific tokenizer.
    pub fn with_tokenizer(
        inner: Arc<P>,
        guard: BudgetGuard,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Self {
        Self {
            inner,
            guard,
            tokenizer,
        }
    }

    /// Access the shared guard — useful for inspecting usage or resetting.
    pub fn guard(&self) -> &BudgetGuard {
        &self.guard
    }

    /// Access the wrapped provider.
    pub fn inner(&self) -> &Arc<P> {
        &self.inner
    }

    /// Access the configured tokenizer.
    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }
}

#[async_trait]
impl<P: Provider + ?Sized + 'static> Provider for BudgetProvider<P> {
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
        self.guard.check()?;

        // Cheap pre-flight: reject if the raw payload alone would blow the
        // token cap. Completion tokens aren't known yet, so we only compare
        // inputs-consumed against the limit.
        if let Some(limit) = self.guard.cfg.max_tokens {
            let projected =
                self.guard.tokens_consumed() + self.tokenizer.count(messages) as u64;
            if projected > limit {
                return Err(ResilienceError::BudgetExceeded {
                    kind: "tokens",
                    consumed: projected,
                    limit,
                }
                .into());
            }
        }

        self.guard.state.rounds.fetch_add(1, Ordering::Relaxed);

        let resp = self.inner.chat(messages, tools, options).await?;
        self.guard.record_usage(&resp.usage);
        Ok(resp)
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        // Clone the guard into the stream so the counters get accumulated as
        // `StreamChunk::Usage` events arrive.
        let guard = self.guard.clone();

        // Fail-fast if we're already over budget.
        if let Err(e) = guard.check() {
            let err_stream = futures::stream::once(async move { Err(anyhow::Error::from(e)) });
            return Box::pin(err_stream);
        }

        if let Some(limit) = guard.cfg.max_tokens {
            let projected = guard.tokens_consumed() + self.tokenizer.count(messages) as u64;
            if projected > limit {
                let err = ResilienceError::BudgetExceeded {
                    kind: "tokens",
                    consumed: projected,
                    limit,
                };
                let err_stream =
                    futures::stream::once(async move { Err(anyhow::Error::from(err)) });
                return Box::pin(err_stream);
            }
        }

        guard.state.rounds.fetch_add(1, Ordering::Relaxed);

        let upstream = self.inner.stream_chat(messages, tools, options);
        let mapped = upstream.map(move |chunk| {
            if let Ok(StreamChunk::Usage(ref u)) = chunk {
                guard.record_usage(u);
            }
            chunk
        });
        Box::pin(mapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_tracks_tokens() {
        let g = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(100),
            ..Default::default()
        });
        g.record_usage(&Usage::new(40, 40));
        assert_eq!(g.tokens_consumed(), 80);
        g.check().expect("under budget");
        g.record_usage(&Usage::new(30, 0));
        assert_eq!(g.tokens_consumed(), 110);
        let err = g.check().unwrap_err();
        assert!(matches!(
            err,
            ResilienceError::BudgetExceeded { kind: "tokens", .. }
        ));
    }

    #[test]
    fn guard_tracks_rounds() {
        let g = BudgetGuard::new(BudgetConfig {
            max_rounds: Some(2),
            ..Default::default()
        });
        g.check_and_tick().unwrap();
        g.check_and_tick().unwrap();
        let err = g.check_and_tick().unwrap_err();
        assert!(matches!(
            err,
            ResilienceError::BudgetExceeded { kind: "rounds", .. }
        ));
    }

    #[test]
    fn guard_reset_zeroes_everything() {
        let g = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(100),
            max_rounds: Some(5),
            ..Default::default()
        });
        g.record_usage(&Usage::new(5, 5));
        g.check_and_tick().unwrap();
        g.reset();
        assert_eq!(g.tokens_consumed(), 0);
        assert_eq!(g.rounds_consumed(), 0);
    }

    #[test]
    fn heuristic_tokens_text_and_blocks() {
        let msgs = vec![
            Message::user("abcd".repeat(40)), // 160 chars → ~40 tokens
        ];
        let n = crate::tokenizer::HeuristicTokenizer.count(&msgs);
        assert_eq!(n, 40);
    }
}

// ── KeyedBudgetGuard ─────────────────────────────────────────────────────────

/// Per-caller budget tracking. Each unique `K` (e.g. `user_id`, `tenant_id`)
/// gets its own private [`BudgetGuard`] with the same per-key caps.
///
/// Concurrent callers with different keys are fully isolated — exhausting
/// user A's quota does not affect user B. Concurrent callers sharing a key
/// see the same shared atomic counters (since [`BudgetGuard`] is `Clone`
/// via `Arc`-bump).
///
/// Example:
///
/// ```rust,no_run
/// use rullama_call_policy::{BudgetConfig, BudgetProvider, KeyedBudgetGuard};
/// # use rullama_core::Provider; use std::sync::Arc;
/// # async fn example(real_provider: Arc<dyn Provider>) -> anyhow::Result<()> {
/// let kbg = KeyedBudgetGuard::<String>::new(BudgetConfig {
///     max_rounds: Some(3),
///     ..Default::default()
/// });
/// let user_a_guard = kbg.for_key(&"user_a".to_string()).await;
/// let provider_for_a = Arc::new(BudgetProvider::new(real_provider, user_a_guard));
/// # let _ = provider_for_a;
/// # Ok(()) }
/// ```
pub struct KeyedBudgetGuard<K: Hash + Eq + Clone + Send + Sync + 'static> {
    cfg: BudgetConfig,
    guards: Arc<Mutex<HashMap<K, BudgetGuard>>>,
}

impl<K: Hash + Eq + Clone + Send + Sync + 'static> KeyedBudgetGuard<K> {
    /// Create a fresh keyed guard. Every key starts with a fresh
    /// [`BudgetGuard`] configured with `cfg`; per-key counters are
    /// allocated lazily on first `for_key` lookup.
    pub fn new(cfg: BudgetConfig) -> Self {
        Self {
            cfg,
            guards: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the [`BudgetGuard`] for `key`, creating one on first access.
    /// Subsequent calls for the same key return a guard sharing the same
    /// atomic counters.
    pub async fn for_key(&self, key: &K) -> BudgetGuard {
        let mut guards = self.guards.lock().await;
        guards
            .entry(key.clone())
            .or_insert_with(|| BudgetGuard::new(self.cfg.clone()))
            .clone()
    }

    /// Lookup without creating — returns `None` if the key hasn't been
    /// observed yet.
    pub async fn try_get(&self, key: &K) -> Option<BudgetGuard> {
        let guards = self.guards.lock().await;
        guards.get(key).cloned()
    }

    /// Number of distinct keys currently tracked.
    pub async fn len(&self) -> usize {
        self.guards.lock().await.len()
    }

    /// Whether no keys are tracked yet.
    pub async fn is_empty(&self) -> bool {
        self.guards.lock().await.is_empty()
    }

    /// The shared budget configuration applied to every per-key guard.
    pub fn config(&self) -> &BudgetConfig {
        &self.cfg
    }
}

impl<K: Hash + Eq + Clone + Send + Sync + 'static> Clone for KeyedBudgetGuard<K> {
    fn clone(&self) -> Self {
        Self {
            cfg: self.cfg.clone(),
            guards: Arc::clone(&self.guards),
        }
    }
}

#[cfg(test)]
mod keyed_tests {
    use super::*;

    #[tokio::test]
    async fn distinct_keys_are_isolated() {
        let kbg: KeyedBudgetGuard<&'static str> = KeyedBudgetGuard::new(BudgetConfig {
            max_rounds: Some(2),
            ..Default::default()
        });
        let a = kbg.for_key(&"alice").await;
        let b = kbg.for_key(&"bob").await;
        a.check_and_tick().unwrap();
        a.check_and_tick().unwrap();
        // Alice is exhausted; Bob must still have full quota.
        assert!(a.check().is_err());
        assert!(b.check().is_ok());
        b.check_and_tick().unwrap();
        assert!(b.check().is_ok());
    }

    #[tokio::test]
    async fn same_key_shares_counters() {
        let kbg: KeyedBudgetGuard<String> = KeyedBudgetGuard::new(BudgetConfig {
            max_rounds: Some(2),
            ..Default::default()
        });
        let h1 = kbg.for_key(&"x".to_string()).await;
        let h2 = kbg.for_key(&"x".to_string()).await;
        h1.check_and_tick().unwrap();
        h2.check_and_tick().unwrap();
        // Two distinct handles, same counter — third tick must reject.
        assert!(h1.check().is_err());
        assert!(h2.check().is_err());
    }

    #[tokio::test]
    async fn try_get_is_lookup_only() {
        let kbg: KeyedBudgetGuard<&'static str> = KeyedBudgetGuard::new(BudgetConfig::default());
        assert!(kbg.try_get(&"unseen").await.is_none());
        let _ = kbg.for_key(&"seen").await;
        assert!(kbg.try_get(&"seen").await.is_some());
    }
}
