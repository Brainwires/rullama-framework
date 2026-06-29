//! Integration tests for middleware composition + inspector plumbing.
//!
//! The unit tests inside `src/middleware/*.rs` already exercise each layer
//! in isolation. This file covers the *composed* behaviour: real concrete
//! middleware types chained together in a `MiddlewareStack`, plus the pair
//! of `EventStore` + `EventBroadcaster` used by the inspector subsystem.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use brainwires_proxy::inspector::broadcast::EventBroadcaster;
use brainwires_proxy::inspector::store::EventStore;
use brainwires_proxy::inspector::{EventDirection, TrafficEvent, TrafficEventKind};
use brainwires_proxy::middleware::header_inject::HeaderInjectLayer;
use brainwires_proxy::middleware::rate_limit::RateLimitLayer;
use brainwires_proxy::middleware::{LayerAction, MiddlewareStack, ProxyLayer};
use brainwires_proxy::request_id::RequestId;
use brainwires_proxy::types::ProxyRequest;

use async_trait::async_trait;
use brainwires_proxy::error::ProxyResult;
use http::{HeaderName, HeaderValue, Method, StatusCode};

/// Recording layer that pushes its id to a shared vector on request.
struct RecordingLayer {
    id: &'static str,
    log: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl ProxyLayer for RecordingLayer {
    async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction> {
        self.log.lock().unwrap().push(self.id);
        Ok(LayerAction::Forward(request))
    }

    fn name(&self) -> &str {
        self.id
    }
}

fn make_request() -> ProxyRequest {
    ProxyRequest::new(Method::GET, "/probe".parse().unwrap())
}

#[tokio::test]
async fn middleware_chain_runs_in_order() {
    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let mut stack = MiddlewareStack::new();
    stack.push(RecordingLayer {
        id: "first",
        log: log.clone(),
    });
    stack.push(RecordingLayer {
        id: "second",
        log: log.clone(),
    });

    let result = stack.process_request(make_request()).await.unwrap();
    assert!(result.is_ok(), "expected all layers to forward");
    let (_, depth) = result.unwrap();
    assert_eq!(depth, 2);

    let recorded = log.lock().unwrap().clone();
    assert_eq!(recorded, vec!["first", "second"]);
}

#[tokio::test]
async fn rate_limit_blocks_after_threshold() {
    // 3 burst tokens, 0 refill: first 3 succeed, 4th is short-circuited.
    let limiter = RateLimitLayer::new(3.0, 0.0);

    for _ in 0..3 {
        let action = limiter.on_request(make_request()).await.unwrap();
        assert!(
            matches!(action, LayerAction::Forward(_)),
            "expected forward within threshold",
        );
    }

    let action = limiter.on_request(make_request()).await.unwrap();
    match action {
        LayerAction::Respond(resp) => {
            assert_eq!(resp.status, StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(resp.body.as_bytes(), b"Rate limit exceeded");
        }
        LayerAction::Forward(_) => panic!("4th request should have been rate limited"),
    }
}

#[tokio::test]
async fn header_inject_adds_header() {
    let layer = HeaderInjectLayer::new().set_request_header(
        HeaderName::from_static("x-test"),
        HeaderValue::from_static("brainwires"),
    );

    let action = layer.on_request(make_request()).await.unwrap();
    match action {
        LayerAction::Forward(req) => {
            assert_eq!(req.headers.get("x-test").unwrap(), "brainwires");
        }
        LayerAction::Respond(_) => panic!("header inject should never short-circuit"),
    }
}

#[tokio::test]
async fn composed_stack_injects_then_rate_limits() {
    // Verify ordering: header injection runs before rate limiting.
    // The first burst is allowed through and receives the injected header.
    // Subsequent calls past the limit are rejected regardless of headers.
    let mut stack = MiddlewareStack::new();
    stack.push(HeaderInjectLayer::new().set_request_header(
        HeaderName::from_static("x-via"),
        HeaderValue::from_static("proxy"),
    ));
    stack.push(RateLimitLayer::new(1.0, 0.0));

    // First request: both layers run, header present.
    let result = stack.process_request(make_request()).await.unwrap();
    let (req, depth) = result.expect("first request should be forwarded");
    assert_eq!(depth, 2);
    assert_eq!(req.headers.get("x-via").unwrap(), "proxy");

    // Second request: header injection runs, rate limiter short-circuits.
    let result = stack.process_request(make_request()).await.unwrap();
    let resp = result.expect_err("second request should be rate limited");
    assert_eq!(resp.status, StatusCode::TOO_MANY_REQUESTS);
}

fn make_event(kind: TrafficEventKind) -> TrafficEvent {
    TrafficEvent {
        id: uuid::Uuid::new_v4(),
        request_id: RequestId::new(),
        timestamp: chrono::Utc::now(),
        direction: EventDirection::Inbound,
        kind,
    }
}

#[tokio::test]
async fn inspector_store_fires_broadcast_events() {
    // Simulate the real wiring: events pushed into the store are also
    // fanned out through a broadcaster to live subscribers.
    let store = EventStore::new(16);
    let broadcaster = EventBroadcaster::new(16);
    let mut rx = broadcaster.subscribe();
    assert_eq!(broadcaster.subscriber_count(), 1);

    let payloads = [
        TrafficEventKind::Request {
            method: "GET".into(),
            uri: "/a".into(),
            headers: HashMap::new(),
            body_size: 0,
        },
        TrafficEventKind::Response {
            status: 200,
            headers: HashMap::new(),
            body_size: 5,
        },
        TrafficEventKind::Error {
            message: "boom".into(),
        },
    ];

    for p in payloads.iter().cloned() {
        let evt = make_event(p);
        store.push(evt.clone());
        let delivered = broadcaster.send(evt);
        assert_eq!(delivered, 1, "broadcaster should deliver to our subscriber");
    }

    assert_eq!(store.len(), 3);
    assert_eq!(store.total_pushed(), 3);

    // Subscriber receives events in push order.
    let mut kinds = Vec::new();
    for _ in 0..3 {
        let evt = rx.recv().await.expect("subscriber receives event");
        kinds.push(match evt.kind {
            TrafficEventKind::Request { .. } => "request",
            TrafficEventKind::Response { .. } => "response",
            TrafficEventKind::Error { .. } => "error",
            _ => "other",
        });
    }
    assert_eq!(kinds, vec!["request", "response", "error"]);
}
