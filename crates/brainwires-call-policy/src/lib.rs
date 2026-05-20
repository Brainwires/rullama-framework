#![deny(missing_docs)]
//! Provider-layer resilience middleware for the Brainwires Agent Framework.
//!
//! Wraps any `brainwires_core::Provider` with composable decorators:
//!
//! - [`RetryProvider`] — exponential backoff with jitter on transient failures.
//! - [`BudgetProvider`] — atomic token/USD/round caps with pre-flight rejection.
//! - [`CircuitBreakerProvider`] — half-open state machine, optional fallback.
//!
//! Decorators wrap `Arc<dyn Provider>` and return `Arc<dyn Provider>`, so they
//! compose freely. Typical stacking (outermost first):
//!
//! ```text
//! CircuitBreaker → Retry → Budget → base Provider
//! ```

mod budget;
mod cache;
mod circuit;
mod classify;
mod error;
mod retry;

#[cfg(test)]
mod tests_util;

pub use budget::{BudgetConfig, BudgetGuard, BudgetProvider};
#[cfg(feature = "cache")]
pub use cache::SqliteCache;
pub use cache::{
    CacheBackend, CacheKey, CachedProvider, CachedResponse, MemoryCache, cache_key_for,
};
pub use circuit::{CircuitBreakerConfig, CircuitBreakerProvider, CircuitState};
pub use classify::{ErrorClass, classify_error};
pub use error::ResilienceError;
pub use retry::{RetryPolicy, RetryProvider};
