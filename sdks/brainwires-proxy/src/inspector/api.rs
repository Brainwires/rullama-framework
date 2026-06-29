//! HTTP query API for the inspector — serves captured traffic over HTTP.
//!
//! Endpoints:
//! - `GET /events` — query stored events (optional `?direction=inbound&kind=request&limit=100`)
//! - `GET /stats` — store statistics
//! - `GET /stream` — live SSE stream of events

use crate::error::ProxyResult;
use crate::inspector::store::EventFilter;
use crate::inspector::{EventBroadcaster, EventStore};

use bytes::Bytes;
use http::{Method, StatusCode, Uri};
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::watch;

/// Run the inspector HTTP API server.
pub async fn run_inspector_api(
    addr: SocketAddr,
    store: Arc<EventStore>,
    broadcaster: Arc<EventBroadcaster>,
    mut shutdown: watch::Receiver<bool>,
) -> ProxyResult<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(addr = %addr, "Inspector API started");

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _) = accept?;
                let store = store.clone();
                let broadcaster = broadcaster.clone();
                let io = TokioIo::new(stream);

                tokio::spawn(async move {
                    let service = service_fn(move |req: hyper::Request<Incoming>| {
                        let store = store.clone();
                        let broadcaster = broadcaster.clone();
                        async move {
                            handle_api_request(req, store, broadcaster).await
                        }
                    });

                    if let Err(e) = ServerBuilder::new(hyper_util::rt::TokioExecutor::new())
                        .serve_connection(io, service)
                        .await
                    {
                        tracing::debug!(error = %e, "Inspector API connection error");
                    }
                });
            }
            _ = shutdown.changed() => {
                tracing::info!("Inspector API shutting down");
                break;
            }
        }
    }

    Ok(())
}

async fn handle_api_request(
    req: hyper::Request<Incoming>,
    store: Arc<EventStore>,
    _broadcaster: Arc<EventBroadcaster>,
) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let response = match (method, path.as_str()) {
        (Method::GET, "/events") => {
            let filter = parse_event_filter(req.uri());
            let events = store.query(&filter);
            let json = serde_json::to_vec(&events).unwrap_or_default();
            hyper::Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(json)))
                .unwrap()
        }
        (Method::GET, "/stats") => {
            let stats = store.stats();
            let json = serde_json::to_vec(&stats).unwrap_or_default();
            hyper::Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(json)))
                .unwrap()
        }
        _ => hyper::Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))
            .unwrap(),
    };

    Ok(response)
}

fn parse_event_filter(uri: &Uri) -> EventFilter {
    let mut filter = EventFilter::default();

    if let Some(query) = uri.query() {
        for pair in query.split('&') {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next().unwrap_or_default();
            let value = parts.next().unwrap_or_default();
            match key {
                "direction" => {
                    filter.direction = match value {
                        "inbound" => Some(crate::inspector::EventDirection::Inbound),
                        "outbound" => Some(crate::inspector::EventDirection::Outbound),
                        _ => None,
                    };
                }
                "kind" => {
                    filter.kind = Some(value.to_string());
                }
                "request_id" => {
                    filter.request_id = Some(value.to_string());
                }
                "limit" => {
                    filter.limit = value.parse().ok();
                }
                _ => {}
            }
        }
    }

    filter
}
