//! Traffic capture middleware — feeds events to the inspector subsystem.

use crate::error::ProxyResult;
use crate::inspector::{
    EventBroadcaster, EventDirection, EventStore, TrafficEvent, TrafficEventKind,
};
use crate::middleware::{LayerAction, ProxyLayer};
use crate::types::{ProxyRequest, ProxyResponse};
use std::sync::Arc;

/// Middleware that captures traffic events and publishes them to the
/// inspector's store and broadcast channel.
pub struct InspectorLayer {
    store: Arc<EventStore>,
    broadcaster: Arc<EventBroadcaster>,
}

impl InspectorLayer {
    pub fn new(store: Arc<EventStore>, broadcaster: Arc<EventBroadcaster>) -> Self {
        Self { store, broadcaster }
    }
}

#[async_trait::async_trait]
impl ProxyLayer for InspectorLayer {
    async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction> {
        let event = TrafficEvent {
            id: uuid::Uuid::new_v4(),
            request_id: request.id.clone(),
            timestamp: chrono::Utc::now(),
            direction: EventDirection::Inbound,
            kind: TrafficEventKind::Request {
                method: request.method.to_string(),
                uri: request.uri.to_string(),
                headers: crate::inspector::headers_to_map(&request.headers),
                body_size: request.body.len(),
            },
        };

        self.store.push(event.clone());
        let _ = self.broadcaster.send(event);

        Ok(LayerAction::Forward(request))
    }

    async fn on_response(&self, response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        let event = TrafficEvent {
            id: uuid::Uuid::new_v4(),
            request_id: response.id.clone(),
            timestamp: chrono::Utc::now(),
            direction: EventDirection::Outbound,
            kind: TrafficEventKind::Response {
                status: response.status.as_u16(),
                headers: crate::inspector::headers_to_map(&response.headers),
                body_size: response.body.len(),
            },
        };

        self.store.push(event.clone());
        let _ = self.broadcaster.send(event);

        Ok(response)
    }

    fn name(&self) -> &str {
        "inspector"
    }
}
