//! Broadcast channel wrapper for live traffic event subscriptions.

use crate::inspector::TrafficEvent;
use tokio::sync::broadcast;

/// Wraps a `tokio::sync::broadcast` channel for publishing traffic events
/// to multiple live subscribers.
pub struct EventBroadcaster {
    tx: broadcast::Sender<TrafficEvent>,
}

impl EventBroadcaster {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish an event to all subscribers. Returns the number of receivers
    /// that received the event (0 if none are subscribed).
    pub fn send(&self, event: TrafficEvent) -> usize {
        self.tx.send(event).unwrap_or(0)
    }

    /// Subscribe to live events. Returns a receiver that yields events.
    pub fn subscribe(&self) -> broadcast::Receiver<TrafficEvent> {
        self.tx.subscribe()
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inspector::{EventDirection, TrafficEventKind};
    use crate::request_id::RequestId;
    use std::collections::HashMap;

    fn make_event() -> TrafficEvent {
        TrafficEvent {
            id: uuid::Uuid::new_v4(),
            request_id: RequestId::new(),
            timestamp: chrono::Utc::now(),
            direction: EventDirection::Inbound,
            kind: TrafficEventKind::Request {
                method: "GET".into(),
                uri: "/test".into(),
                headers: HashMap::new(),
                body_size: 0,
            },
        }
    }

    #[test]
    fn send_without_subscribers_returns_zero() {
        let broadcaster = EventBroadcaster::new(16);
        assert_eq!(broadcaster.send(make_event()), 0);
    }

    #[tokio::test]
    async fn subscribers_receive_events() {
        let broadcaster = EventBroadcaster::new(16);
        let mut rx = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 1);

        broadcaster.send(make_event());

        let event = rx.recv().await.unwrap();
        assert!(matches!(event.kind, TrafficEventKind::Request { .. }));
    }

    #[tokio::test]
    async fn multiple_subscribers() {
        let broadcaster = EventBroadcaster::new(16);
        let mut rx1 = broadcaster.subscribe();
        let mut rx2 = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 2);

        let count = broadcaster.send(make_event());
        assert_eq!(count, 2);

        rx1.recv().await.unwrap();
        rx2.recv().await.unwrap();
    }
}
