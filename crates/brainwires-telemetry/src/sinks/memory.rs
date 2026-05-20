use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;

use crate::{AnalyticsError, AnalyticsEvent, AnalyticsSink};

/// Default ring-buffer capacity for [`MemoryAnalyticsSink`].
pub const DEFAULT_CAPACITY: usize = 1_000;

/// Emit a `tracing::warn!` once every N drops to surface saturation without
/// flooding the log on a steady-state overflow.
const DROP_WARN_EVERY: u64 = 1_000;

/// In-memory ring-buffer analytics sink.
///
/// Stores up to `capacity` events, evicting the oldest when full. Eviction is
/// counted via [`dropped_count`](Self::dropped_count) and a `tracing::warn!`
/// is emitted on the first drop and once per [`DROP_WARN_EVERY`] thereafter.
/// Useful for testing and embedded scenarios where persistence is not needed.
pub struct MemoryAnalyticsSink {
    capacity: usize,
    events: Mutex<VecDeque<AnalyticsEvent>>,
    dropped: AtomicU64,
}

impl MemoryAnalyticsSink {
    /// Create a new sink with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            events: Mutex::new(VecDeque::with_capacity(capacity)),
            dropped: AtomicU64::new(0),
        }
    }

    /// Total number of events evicted because the ring buffer was full since
    /// this sink was constructed. Monotonic, never reset by `drain`/`retain`.
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn note_drop(&self) {
        let n = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_multiple_of(DROP_WARN_EVERY) {
            tracing::warn!(
                target: "telemetry.dropped",
                dropped = n,
                capacity = self.capacity,
                "MemoryAnalyticsSink evicted oldest event; consider raising capacity or draining more often"
            );
        }
    }

    /// Deposit an event synchronously (bypasses the async record path).
    /// Evicts the oldest event if over capacity.
    pub fn deposit(&self, event: AnalyticsEvent) {
        let mut events = self.events.lock().expect("lock poisoned");
        if events.len() >= self.capacity {
            events.pop_front();
            self.note_drop();
        }
        events.push_back(event);
    }

    /// Drain and return all buffered events, clearing the buffer.
    pub fn drain(&self) -> Vec<AnalyticsEvent> {
        let mut events = self
            .events
            .lock()
            .expect("MemoryAnalyticsSink lock poisoned");
        events.drain(..).collect()
    }

    /// Drain events matching `pred`, leaving non-matching events in place.
    pub fn drain_matching(&self, pred: impl Fn(&AnalyticsEvent) -> bool) -> Vec<AnalyticsEvent> {
        let mut events = self.events.lock().expect("lock poisoned");
        let mut matched = Vec::new();
        let mut remaining = VecDeque::new();
        for event in events.drain(..) {
            if pred(&event) {
                matched.push(event);
            } else {
                remaining.push_back(event);
            }
        }
        *events = remaining;
        matched
    }

    /// Retain events matching `pred`; remove and count non-matching events.
    pub fn retain(&self, pred: impl Fn(&AnalyticsEvent) -> bool) -> usize {
        let mut events = self.events.lock().expect("lock poisoned");
        let before = events.len();
        events.retain(|e| pred(e));
        before - events.len()
    }

    /// Number of events currently in the buffer.
    pub fn len(&self) -> usize {
        self.events
            .lock()
            .expect("MemoryAnalyticsSink lock poisoned")
            .len()
    }

    /// True when the ring buffer holds no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Peek at a snapshot of all buffered events (cloned).
    pub fn snapshot(&self) -> Vec<AnalyticsEvent> {
        self.events
            .lock()
            .expect("MemoryAnalyticsSink lock poisoned")
            .iter()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl AnalyticsSink for MemoryAnalyticsSink {
    async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError> {
        let mut events = self
            .events
            .lock()
            .expect("MemoryAnalyticsSink lock poisoned");
        if events.len() >= self.capacity {
            events.pop_front();
            self.note_drop();
        }
        events.push_back(event);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_event(name: &str) -> AnalyticsEvent {
        AnalyticsEvent::Custom {
            session_id: None,
            name: name.to_string(),
            payload: serde_json::Value::Null,
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_ring_buffer_capacity() {
        let sink = MemoryAnalyticsSink::new(3);
        for i in 0..5 {
            sink.record(make_event(&format!("e{i}"))).await.unwrap();
        }
        assert_eq!(sink.len(), 3);
        let events = sink.drain();
        // Oldest 2 should be evicted; only e2, e3, e4 remain
        assert!(matches!(&events[0], AnalyticsEvent::Custom { name, .. } if name == "e2"));
        assert!(matches!(&events[2], AnalyticsEvent::Custom { name, .. } if name == "e4"));
    }

    #[tokio::test]
    async fn test_drain_clears_buffer() {
        let sink = MemoryAnalyticsSink::new(10);
        sink.record(make_event("a")).await.unwrap();
        sink.record(make_event("b")).await.unwrap();
        assert_eq!(sink.len(), 2);
        let drained = sink.drain();
        assert_eq!(drained.len(), 2);
        assert!(sink.is_empty());
    }

    #[tokio::test]
    async fn test_dropped_count_tracks_overflow() {
        let sink = MemoryAnalyticsSink::new(2);
        assert_eq!(sink.dropped_count(), 0);
        for i in 0..5 {
            sink.record(make_event(&format!("e{i}"))).await.unwrap();
        }
        assert_eq!(sink.len(), 2);
        assert_eq!(
            sink.dropped_count(),
            3,
            "5 records into a 2-capacity ring = 3 evictions"
        );
    }

    #[test]
    fn test_dropped_count_includes_deposit_path() {
        let sink = MemoryAnalyticsSink::new(1);
        sink.deposit(make_event("a"));
        sink.deposit(make_event("b"));
        sink.deposit(make_event("c"));
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.dropped_count(), 2);
    }
}
