//! Resilience-specific error types.

use thiserror::Error;

/// Errors surfaced by resilience decorators.
///
/// Wrapped in `anyhow::Error` when returned through the `Provider` trait. Use
/// `err.downcast_ref::<ResilienceError>()` to recover the typed variant.
#[derive(Debug, Error)]
pub enum ResilienceError {
    /// Budget cap reached before the request could be sent.
    #[error("budget exceeded: {kind} ({consumed}/{limit})")]
    BudgetExceeded {
        /// Which cap tripped: "tokens", "usd_cents", or "rounds".
        kind: &'static str,
        /// Amount already consumed.
        consumed: u64,
        /// Configured limit.
        limit: u64,
    },

    /// Circuit breaker is open and rejecting calls for the cooldown window.
    #[error("circuit open for {provider}/{model}: {failures} consecutive failures")]
    CircuitOpen {
        /// Provider name at the time the circuit tripped.
        provider: String,
        /// Model the circuit is keyed against.
        model: String,
        /// Consecutive failure count that tripped the breaker.
        failures: u32,
    },

    /// Retries exhausted — the final attempt's error is attached.
    #[error("retries exhausted after {attempts} attempts: {source}")]
    RetriesExhausted {
        /// How many attempts were made in total.
        attempts: u32,
        /// The final error that caused retry abandonment.
        #[source]
        source: anyhow::Error,
    },

    /// `RetryPolicy.overall_deadline` elapsed before the call could succeed.
    #[error("retry deadline exceeded after {elapsed_ms}ms ({attempts} attempts): {source}")]
    DeadlineExceeded {
        /// How many attempts were made before the deadline tripped.
        attempts: u32,
        /// Wall-clock elapsed since the first attempt, in milliseconds.
        elapsed_ms: u64,
        /// The most recent attempt's error.
        #[source]
        source: anyhow::Error,
    },
}
