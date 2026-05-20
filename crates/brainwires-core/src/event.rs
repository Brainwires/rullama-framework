//! Unified event schema with trace IDs and sequence numbers.
//!
//! Defines a common [`Event`](crate::event::Event) trait and
//! [`EventEnvelope<E>`](crate::event::EventEnvelope) wrapper that can carry
//! any domain event with correlation metadata. This enables cross-system
//! tracing across the provider, A2A, and agent-network layers without forcing
//! a breaking change on existing event types.
//!
//! # Adoption strategy
//!
//! Existing event types (`ResponseStreamEvent`, `TaskStatusUpdateEvent`,
//! `NetworkEvent`) do not need to implement `Event` directly. Instead, wrap
//! them in `EventEnvelope` at consumption boundaries (OTel export, audit
//! logger) to attach trace context without touching internal structs.
//!
//! # Example
//!
//! ```ignore
//! use brainwires_core::event::{EventEnvelope, new_trace_id};
//!
//! let trace = new_trace_id();
//! let env = EventEnvelope::new(trace, 0, my_event);
//! assert_eq!(env.trace_id, trace);
//! ```

use chrono::{DateTime, Utc};
use uuid::Uuid;

// ─── Trait ──────────────────────────────────────────────────────────────────

/// Common interface for structured events that carry correlation metadata.
///
/// Implementors gain first-class trace correlation for free, enabling
/// cross-system log joins, distributed tracing, and event replay.
///
/// Implementing this trait is **optional** — prefer [`EventEnvelope`] at
/// boundaries rather than retrofitting existing event structs.
pub trait Event: Send + Sync + std::fmt::Debug {
    /// Unique ID for this specific event instance.
    fn event_id(&self) -> Uuid;

    /// Trace ID shared by all events in a single logical operation
    /// (e.g. one `TaskAgent::execute()` invocation).
    fn trace_id(&self) -> Uuid;

    /// Monotonically increasing sequence number within the trace.
    /// Used to reorder out-of-order events and detect gaps.
    fn sequence(&self) -> u64;

    /// Wall-clock timestamp when this event was emitted.
    fn occurred_at(&self) -> DateTime<Utc>;

    /// Short lowercase label identifying the event kind (e.g. `"tool_executed"`).
    fn event_type(&self) -> &'static str;
}

// ─── Envelope ───────────────────────────────────────────────────────────────

/// Generic envelope that wraps an arbitrary payload with correlation metadata.
///
/// Use this instead of implementing [`Event`] on existing types — wrap at the
/// point where events are produced or consumed (audit logger, OTel exporter).
#[derive(Debug, Clone)]
pub struct EventEnvelope<E> {
    /// Unique ID for this envelope instance.
    pub event_id: Uuid,
    /// Trace ID shared across all envelopes for one logical operation.
    pub trace_id: Uuid,
    /// Monotonically increasing counter within the trace.
    pub sequence: u64,
    /// Wall-clock time when this envelope was created.
    pub occurred_at: DateTime<Utc>,
    /// The wrapped event payload.
    pub payload: E,
}

impl<E: std::fmt::Debug + Send + Sync> EventEnvelope<E> {
    /// Wrap `payload` in a new envelope with the given trace context.
    pub fn new(trace_id: Uuid, sequence: u64, payload: E) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            trace_id,
            sequence,
            occurred_at: Utc::now(),
            payload,
        }
    }

    /// Map the payload to a different type, preserving all correlation fields.
    pub fn map<F, U: std::fmt::Debug + Send + Sync>(self, f: F) -> EventEnvelope<U>
    where
        F: FnOnce(E) -> U,
    {
        EventEnvelope {
            event_id: self.event_id,
            trace_id: self.trace_id,
            sequence: self.sequence,
            occurred_at: self.occurred_at,
            payload: f(self.payload),
        }
    }
}

impl<E: std::fmt::Debug + Send + Sync> Event for EventEnvelope<E> {
    fn event_id(&self) -> Uuid {
        self.event_id
    }

    fn trace_id(&self) -> Uuid {
        self.trace_id
    }

    fn sequence(&self) -> u64 {
        self.sequence
    }

    fn occurred_at(&self) -> DateTime<Utc> {
        self.occurred_at
    }

    fn event_type(&self) -> &'static str {
        "envelope"
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Generate a fresh trace ID for a new logical operation.
///
/// Convenience wrapper around `Uuid::new_v4()` that makes call-sites
/// self-documenting.
pub fn new_trace_id() -> Uuid {
    Uuid::new_v4()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip() {
        let trace = new_trace_id();
        let env = EventEnvelope::new(trace, 1, "hello");
        assert_eq!(env.trace_id, trace);
        assert_eq!(env.sequence, 1);
        assert_eq!(env.payload, "hello");
    }

    #[test]
    fn envelope_map_preserves_correlation() {
        let trace = new_trace_id();
        let env = EventEnvelope::new(trace, 42, 10u32);
        let mapped = env.map(|v| v.to_string());
        assert_eq!(mapped.trace_id, trace);
        assert_eq!(mapped.sequence, 42);
        assert_eq!(mapped.payload, "10");
    }
}
