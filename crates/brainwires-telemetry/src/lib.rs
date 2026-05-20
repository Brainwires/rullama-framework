#![deny(missing_docs)]

//! Unified telemetry for the Brainwires Agent Framework.
//!
//! Covers both observability (analytics events, tracing layer, SQLite
//! persistence) and billing (usage events, billing hook trait).
//!
//! # Analytics
//!
//! 1. **Explicit emission** — call [`AnalyticsCollector::record`] directly.
//! 2. **`tracing` layer** — register [`AnalyticsLayer`] to intercept known
//!    span names (`provider.chat`, etc.) automatically.
//! 3. **[`AnalyticsQuery`]** (feature `sqlite`) — query aggregated data: cost
//!    by model, tool frequency, daily summaries.
//!
//! # Billing hooks
//!
//! Implement [`BillingHook`] and pass it into `TaskAgentConfig::billing_hook`
//! to receive a [`UsageEvent`] at every provider call and tool call.
//! Full implementations (ledger, wallet, Stripe) live in
//! `extras/brainwires-billing`.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use brainwires_telemetry::{AnalyticsCollector, SqliteAnalyticsSink, AnalyticsLayer};
//! use tracing_subscriber::prelude::*;
//!
//! let sink      = SqliteAnalyticsSink::new_default()?;
//! let collector = AnalyticsCollector::new(vec![Box::new(sink)]);
//!
//! tracing_subscriber::registry()
//!     .with(tracing_subscriber::fmt::layer())
//!     .with(AnalyticsLayer::new(collector.clone()))
//!     .init();
//! ```

/// Fan-out `AnalyticsCollector` that dispatches events to every attached sink.
pub mod collector;
/// Typed error enum for analytics recording / query failures.
pub mod error;
/// `AnalyticsEvent` variants — provider calls, tool calls, agent runs, custom.
pub mod events;
/// Export helpers: flush buffered events to disk / CSV / JSONL.
pub mod export;
/// `tracing_subscriber::Layer` that intercepts known spans and emits analytics events.
pub mod layer;
/// PII redaction helpers (session-id hashing, payload scrubbing).
pub mod pii;
/// SQLite schema migrations for the `sqlite` sink.
pub mod schema;
/// `AnalyticsSink` trait + `BoxedSink` alias.
pub mod sink;
/// Concrete sink implementations: in-memory ring buffer, SQLite.
pub mod sinks;

// Billing hook surface
/// `BillingHook` trait and error type — invoked at every provider / tool call.
pub mod billing_hook;
/// `UsageEvent` — the payload handed to a `BillingHook`.
pub mod usage;

// Outcome metrics + Prometheus text export
/// Outcome metric counters + Prometheus text exposition.
pub mod metrics;

/// Anomaly detection over observed audit-style events (consumer-agnostic via `ObservedEvent` trait).
pub mod anomaly;

/// SQL-backed analytics queries (cost by model, tool frequency, daily summaries).
#[cfg(feature = "sqlite")]
pub mod query;

pub use collector::AnalyticsCollector;
pub use error::{AnalyticsError, AnalyticsResult};
pub use events::AnalyticsEvent;
pub use layer::AnalyticsLayer;
pub use sink::{AnalyticsSink, BoxedSink};
pub use sinks::memory::{DEFAULT_CAPACITY, MemoryAnalyticsSink};

pub use anomaly::{
    AnomalyConfig, AnomalyDetector, AnomalyEvent, AnomalyKind, EventCategory, ObservedEvent,
};
pub use billing_hook::{BillingError, BillingHook};
pub use metrics::{MetricsRegistry, OutcomeMetrics};
pub use usage::UsageEvent;

#[cfg(feature = "sqlite")]
pub use sinks::sqlite::SqliteAnalyticsSink;

#[cfg(feature = "sqlite")]
pub use query::{AnalyticsQuery, CostByModelRow, DailySummaryRow, ToolFrequencyRow};
