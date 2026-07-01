//! Ring-buffer event store with query support.

use crate::inspector::{EventDirection, TrafficEvent, TrafficEventKind};
use std::collections::VecDeque;
use std::sync::Mutex;

/// Bounded ring-buffer that stores the most recent traffic events.
pub struct EventStore {
    inner: Mutex<StoreInner>,
}

struct StoreInner {
    events: VecDeque<TrafficEvent>,
    capacity: usize,
    total_pushed: u64,
}

impl EventStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(StoreInner {
                events: VecDeque::with_capacity(capacity),
                capacity,
                total_pushed: 0,
            }),
        }
    }

    /// Push an event into the store, evicting the oldest if at capacity.
    pub fn push(&self, event: TrafficEvent) {
        let mut inner = self.inner.lock().expect("event store lock poisoned");
        if inner.events.len() >= inner.capacity {
            inner.events.pop_front();
        }
        inner.events.push_back(event);
        inner.total_pushed += 1;
    }

    /// Query events with optional filters.
    pub fn query(&self, filter: &EventFilter) -> Vec<TrafficEvent> {
        let inner = self.inner.lock().expect("event store lock poisoned");
        inner
            .events
            .iter()
            .filter(|e| filter.matches(e))
            .take(filter.limit.unwrap_or(usize::MAX))
            .cloned()
            .collect()
    }

    /// Get all events (up to the buffer capacity).
    pub fn all(&self) -> Vec<TrafficEvent> {
        let inner = self.inner.lock().expect("event store lock poisoned");
        inner.events.iter().cloned().collect()
    }

    /// Current number of stored events.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("event store lock poisoned")
            .events
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total events ever pushed (including evicted).
    pub fn total_pushed(&self) -> u64 {
        self.inner
            .lock()
            .expect("event store lock poisoned")
            .total_pushed
    }

    /// Clear all events.
    pub fn clear(&self) {
        self.inner
            .lock()
            .expect("event store lock poisoned")
            .events
            .clear();
    }

    /// Get store statistics.
    pub fn stats(&self) -> StoreStats {
        let inner = self.inner.lock().expect("event store lock poisoned");
        StoreStats {
            stored: inner.events.len(),
            capacity: inner.capacity,
            total_pushed: inner.total_pushed,
            evicted: inner.total_pushed.saturating_sub(inner.events.len() as u64),
        }
    }
}

/// Filter criteria for querying events.
#[derive(Debug, Default)]
pub struct EventFilter {
    pub direction: Option<EventDirection>,
    pub request_id: Option<String>,
    pub kind: Option<String>,
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    pub limit: Option<usize>,
}

impl EventFilter {
    pub fn matches(&self, event: &TrafficEvent) -> bool {
        if let Some(dir) = self.direction
            && event.direction != dir
        {
            return false;
        }
        if let Some(ref rid) = self.request_id
            && event.request_id.to_string() != *rid
        {
            return false;
        }
        if let Some(ref kind) = self.kind {
            let event_kind = match &event.kind {
                TrafficEventKind::Request { .. } => "request",
                TrafficEventKind::Response { .. } => "response",
                TrafficEventKind::SseEvent { .. } => "sse",
                TrafficEventKind::WebSocketMessage { .. } => "websocket",
                TrafficEventKind::Error { .. } => "error",
                TrafficEventKind::Connection { .. } => "connection",
                TrafficEventKind::Conversion { .. } => "conversion",
            };
            if event_kind != kind.as_str() {
                return false;
            }
        }
        if let Some(since) = self.since
            && event.timestamp < since
        {
            return false;
        }
        true
    }
}

/// Store statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StoreStats {
    pub stored: usize,
    pub capacity: usize,
    pub total_pushed: u64,
    pub evicted: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_id::RequestId;
    use std::collections::HashMap;

    fn make_event(direction: EventDirection, kind: TrafficEventKind) -> TrafficEvent {
        TrafficEvent {
            id: uuid::Uuid::new_v4(),
            request_id: RequestId::new(),
            timestamp: chrono::Utc::now(),
            direction,
            kind,
        }
    }

    fn make_request_event() -> TrafficEvent {
        make_event(
            EventDirection::Inbound,
            TrafficEventKind::Request {
                method: "GET".into(),
                uri: "/test".into(),
                headers: HashMap::new(),
                body_size: 0,
            },
        )
    }

    fn make_response_event() -> TrafficEvent {
        make_event(
            EventDirection::Outbound,
            TrafficEventKind::Response {
                status: 200,
                headers: HashMap::new(),
                body_size: 42,
            },
        )
    }

    #[test]
    fn push_and_retrieve() {
        let store = EventStore::new(100);
        assert!(store.is_empty());

        store.push(make_request_event());
        store.push(make_response_event());

        assert_eq!(store.len(), 2);
        assert_eq!(store.total_pushed(), 2);
        assert_eq!(store.all().len(), 2);
    }

    #[test]
    fn eviction_at_capacity() {
        let store = EventStore::new(3);

        for _ in 0..5 {
            store.push(make_request_event());
        }

        assert_eq!(store.len(), 3); // capacity is 3
        assert_eq!(store.total_pushed(), 5);

        let stats = store.stats();
        assert_eq!(stats.stored, 3);
        assert_eq!(stats.evicted, 2);
    }

    #[test]
    fn filter_by_direction() {
        let store = EventStore::new(100);
        store.push(make_request_event());
        store.push(make_response_event());
        store.push(make_request_event());

        let filter = EventFilter {
            direction: Some(EventDirection::Inbound),
            ..Default::default()
        };
        let results = store.query(&filter);
        assert_eq!(results.len(), 2);

        let filter = EventFilter {
            direction: Some(EventDirection::Outbound),
            ..Default::default()
        };
        let results = store.query(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn filter_by_kind() {
        let store = EventStore::new(100);
        store.push(make_request_event());
        store.push(make_response_event());
        store.push(make_event(
            EventDirection::Inbound,
            TrafficEventKind::Error {
                message: "oops".into(),
            },
        ));

        let filter = EventFilter {
            kind: Some("request".into()),
            ..Default::default()
        };
        assert_eq!(store.query(&filter).len(), 1);

        let filter = EventFilter {
            kind: Some("error".into()),
            ..Default::default()
        };
        assert_eq!(store.query(&filter).len(), 1);
    }

    #[test]
    fn filter_with_limit() {
        let store = EventStore::new(100);
        for _ in 0..10 {
            store.push(make_request_event());
        }

        let filter = EventFilter {
            limit: Some(3),
            ..Default::default()
        };
        assert_eq!(store.query(&filter).len(), 3);
    }

    #[test]
    fn clear_removes_all() {
        let store = EventStore::new(100);
        store.push(make_request_event());
        store.push(make_request_event());
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
        // total_pushed persists
        assert_eq!(store.total_pushed(), 2);
    }
}
