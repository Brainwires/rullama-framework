//! Traffic inspection — event capture, storage, broadcast, and query API.

#[cfg(feature = "inspector-api")]
pub mod api;
pub mod broadcast;
pub mod store;

use crate::request_id::RequestId;
use std::collections::HashMap;

pub use broadcast::EventBroadcaster;
pub use store::EventStore;

/// Direction of a traffic event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EventDirection {
    Inbound,
    Outbound,
}

/// Kind-specific payload for a traffic event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrafficEventKind {
    Request {
        method: String,
        uri: String,
        headers: HashMap<String, String>,
        body_size: usize,
    },
    Response {
        status: u16,
        headers: HashMap<String, String>,
        body_size: usize,
    },
    SseEvent {
        event_type: Option<String>,
        data_preview: String,
    },
    WebSocketMessage {
        is_binary: bool,
        size: usize,
    },
    Error {
        message: String,
    },
    Connection {
        peer: String,
        connected: bool,
    },
    Conversion {
        source_format: String,
        target_format: String,
        body_size: usize,
    },
}

/// A captured traffic event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrafficEvent {
    pub id: uuid::Uuid,
    pub request_id: RequestId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub direction: EventDirection,
    pub kind: TrafficEventKind,
}

/// Convert `HeaderMap` to a simple `HashMap<String, String>` for serialization.
pub fn headers_to_map(headers: &http::HeaderMap) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            map.insert(name.to_string(), v.to_string());
        }
    }
    map
}
