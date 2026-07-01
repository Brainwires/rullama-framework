//! Stateless HTTP + SSE server transport for MCP (MCP 2026 spec, June target).
//!
//! Implements the MCP Streamable HTTP transport:
//! - `POST /mcp`  — JSON-RPC request/response
//! - `GET  /mcp/events` — server-sent events for server-initiated messages
//! - `GET  /.well-known/mcp/server-card.json` — MCP Server Card (SEP-1649)
//! - `GET  /.well-known/oauth-protected-resource` — RFC9728 metadata (when `oauth` feature enabled)
//!
//! The transport bridges an axum HTTP server into the [`ServerTransport`] trait
//! so it slots into the existing [`McpServer`](crate::McpServer) event loop.
//!
//! # Feature flag
//! This module is only compiled when the `http` feature is enabled.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use axum::{
    Router,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::sse::{Event, KeepAlive},
    response::{IntoResponse, Sse},
    routing::{get, post},
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::mcp_transport::ServerTransport;

/// Capacity of the request queue channel between axum handlers and `read_request()`.
const REQUEST_CHANNEL_CAPACITY: usize = 128;

/// Timeout for a single JSON-RPC request/response round-trip (seconds).
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// SSE keep-alive ping interval (seconds).
const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// A pending request waiting for a response.
type PendingRequest = (String, oneshot::Sender<String>);

/// Shared state for the axum handlers.
#[derive(Clone)]
struct HttpState {
    /// Channel to forward incoming requests into the `read_request()` path.
    request_tx: mpsc::Sender<PendingRequest>,
    /// Server card JSON, pre-serialised at startup.
    server_card_json: Arc<String>,
    /// RFC9728 protected resource metadata JSON (empty if no auth configured).
    oauth_resource_json: Arc<String>,
}

/// Stateless HTTP + SSE transport.
///
/// Spawn with [`HttpServerTransport::bind`], then pass to
/// [`McpServer::with_transport`](crate::server::McpServer::with_transport).
pub struct HttpServerTransport {
    request_rx: mpsc::Receiver<PendingRequest>,
    /// The socket address the server is bound to.
    pub addr: SocketAddr,
}

impl HttpServerTransport {
    /// Bind to the given address and start the axum server.
    ///
    /// # Arguments
    /// * `addr` — socket address to listen on (e.g. `"127.0.0.1:3001".parse()?`)
    /// * `server_card` — pre-built [`McpServerCard`] served at `/.well-known/mcp/server-card.json`
    /// * `oauth_resource` — optional RFC9728 JSON; pass `None` for unauthenticated servers
    pub async fn bind(
        addr: SocketAddr,
        server_card: Option<McpServerCard>,
        oauth_resource: Option<OAuthProtectedResource>,
    ) -> Result<Self> {
        let (request_tx, request_rx) = mpsc::channel::<PendingRequest>(REQUEST_CHANNEL_CAPACITY);

        let server_card_json = Arc::new(
            server_card
                .map(|c| serde_json::to_string(&c).unwrap_or_default())
                .unwrap_or_else(|| "{}".to_string()),
        );

        let oauth_resource_json = Arc::new(
            oauth_resource
                .map(|r| serde_json::to_string(&r).unwrap_or_default())
                .unwrap_or_default(),
        );

        let state = HttpState {
            request_tx,
            server_card_json,
            oauth_resource_json,
        };

        let app = Router::new()
            .route("/mcp", post(handle_mcp_post))
            .route("/mcp/events", get(handle_mcp_events))
            .route("/.well-known/mcp/server-card.json", get(handle_server_card))
            .route(
                "/.well-known/oauth-protected-resource",
                get(handle_oauth_resource),
            )
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        let bound_addr = listener.local_addr()?;

        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!(error = %e, "MCP HTTP server error");
            }
        });

        Ok(Self {
            request_rx,
            addr: bound_addr,
        })
    }
}

#[async_trait]
impl ServerTransport for HttpServerTransport {
    /// Receive the next JSON-RPC request from an HTTP POST.
    ///
    /// Returns `None` when the channel is closed (server shutting down).
    async fn read_request(&mut self) -> Result<Option<String>> {
        match self.request_rx.recv().await {
            Some((body, _response_tx)) => {
                // NOTE: We stash the response sender in a thread-local so that
                // write_response can retrieve it. Since McpServer calls
                // read_request → write_response sequentially this is safe.
                PENDING_RESPONSE_TX.with(|cell| {
                    *cell.borrow_mut() = Some(_response_tx);
                });
                Ok(Some(body))
            }
            None => Ok(None),
        }
    }

    /// Send the JSON-RPC response back to the waiting HTTP handler.
    async fn write_response(&mut self, response: &str) -> Result<()> {
        let maybe_tx = PENDING_RESPONSE_TX.with(|cell| cell.borrow_mut().take());
        if let Some(tx) = maybe_tx {
            let _ = tx.send(response.to_string());
        }
        Ok(())
    }
}

// Thread-local holding the oneshot sender for the current in-flight request.
thread_local! {
    static PENDING_RESPONSE_TX: std::cell::RefCell<Option<oneshot::Sender<String>>> =
        const { std::cell::RefCell::new(None) };
}

// ── axum handlers ─────────────────────────────────────────────────────────────

async fn handle_mcp_post(State(state): State<HttpState>, body: String) -> impl IntoResponse {
    let (response_tx, response_rx) = oneshot::channel::<String>();

    if state.request_tx.send((body, response_tx)).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "MCP server is shutting down",
        )
            .into_response();
    }

    match tokio::time::timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS), response_rx).await {
        Ok(Ok(response)) => {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            (StatusCode::OK, headers, response).into_response()
        }
        Ok(Err(_)) => (StatusCode::INTERNAL_SERVER_ERROR, "Response dropped").into_response(),
        Err(_) => (StatusCode::GATEWAY_TIMEOUT, "Request timed out").into_response(),
    }
}

async fn handle_mcp_events(
    State(_state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    // Currently emits only a keep-alive stream; server-initiated notifications
    // will be wired here in a future iteration.
    let stream = futures::stream::pending();
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS))
            .text("keep-alive"),
    )
}

async fn handle_server_card(State(state): State<HttpState>) -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
    (
        StatusCode::OK,
        headers,
        state.server_card_json.as_str().to_string(),
    )
        .into_response()
}

async fn handle_oauth_resource(State(state): State<HttpState>) -> impl IntoResponse {
    if state.oauth_resource_json.is_empty() {
        return StatusCode::NOT_FOUND.into_response();
    }
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    (
        StatusCode::OK,
        headers,
        state.oauth_resource_json.as_str().to_string(),
    )
        .into_response()
}

// ── Server Card types (SEP-1649) ───────────────────────────────────────────

/// MCP Server Card served at `/.well-known/mcp/server-card.json`.
///
/// Provides machine-readable discovery metadata for MCP registries and crawlers.
#[derive(Debug, Clone, Serialize)]
pub struct McpServerCard {
    /// Human-readable server name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Tools exposed by this server.
    pub tools: Vec<McpToolCardEntry>,
    /// Authentication requirements.
    pub auth: McpAuthInfo,
    /// Supported transport bindings.
    pub transport: Vec<McpTransportInfo>,
}

/// One tool entry in the server card.
#[derive(Debug, Clone, Serialize)]
pub struct McpToolCardEntry {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for tool inputs.
    pub input_schema: serde_json::Value,
}

/// Authentication info embedded in the server card.
#[derive(Debug, Clone, Serialize)]
pub struct McpAuthInfo {
    /// `"none"` | `"bearer"` | `"oauth2"`
    pub scheme: String,
    /// Authorization server URL (present for `"oauth2"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_server: Option<String>,
}

/// Transport binding description.
#[derive(Debug, Clone, Serialize)]
pub struct McpTransportInfo {
    /// `"stdio"` | `"http+sse"`
    pub kind: String,
    /// Base URL for HTTP transports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Build a [`McpServerCard`] from a tool registry and config.
pub fn build_server_card(
    name: impl Into<String>,
    version: impl Into<String>,
    description: Option<String>,
    tools: Vec<McpToolCardEntry>,
    auth: McpAuthInfo,
    transport: Vec<McpTransportInfo>,
) -> McpServerCard {
    McpServerCard {
        name: name.into(),
        version: version.into(),
        description,
        tools,
        auth,
        transport,
    }
}

// ── RFC9728 OAuth Protected Resource Metadata ──────────────────────────────

/// RFC9728 `/.well-known/oauth-protected-resource` response body.
#[derive(Debug, Clone, Serialize)]
pub struct OAuthProtectedResource {
    /// The resource identifier URI.
    pub resource: String,
    /// Authorization server URLs.
    pub authorization_servers: Vec<String>,
    /// Supported OAuth scopes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scopes_supported: Vec<String>,
    /// Supported bearer token methods.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bearer_methods_supported: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_http_transport_bind_and_post() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let card = build_server_card(
            "test-server",
            "0.1.0",
            None,
            vec![McpToolCardEntry {
                name: "ping".to_string(),
                description: "Ping tool".to_string(),
                input_schema: serde_json::json!({}),
            }],
            McpAuthInfo {
                scheme: "none".to_string(),
                authorization_server: None,
            },
            vec![McpTransportInfo {
                kind: "http+sse".to_string(),
                url: None,
            }],
        );

        let mut transport = HttpServerTransport::bind(addr, Some(card), None)
            .await
            .expect("bind failed");

        let bound_addr = transport.addr;

        // Spawn a task that handles one request
        let handler = tokio::spawn(async move {
            let req = transport.read_request().await.unwrap();
            assert!(req.is_some());
            transport
                .write_response(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#)
                .await
                .unwrap();
        });

        // POST a request via reqwest
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{}/mcp", bound_addr))
            .body(r#"{"jsonrpc":"2.0","method":"ping","id":1}"#)
            .header("content-type", "application/json")
            .send()
            .await
            .expect("HTTP POST failed");

        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("result"));

        handler.await.unwrap();
    }

    #[tokio::test]
    async fn test_server_card_endpoint() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let card = build_server_card(
            "my-server",
            "1.0.0",
            Some("Test server".to_string()),
            vec![],
            McpAuthInfo {
                scheme: "none".to_string(),
                authorization_server: None,
            },
            vec![],
        );

        let transport = HttpServerTransport::bind(addr, Some(card), None)
            .await
            .expect("bind failed");

        let client = reqwest::Client::new();
        let resp = client
            .get(format!(
                "http://{}/.well-known/mcp/server-card.json",
                transport.addr
            ))
            .send()
            .await
            .expect("GET failed");

        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["name"], "my-server");
        assert_eq!(json["version"], "1.0.0");
    }

    #[tokio::test]
    async fn test_oauth_resource_endpoint() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let oauth = OAuthProtectedResource {
            resource: "https://example.com/mcp".to_string(),
            authorization_servers: vec!["https://auth.example.com".to_string()],
            scopes_supported: vec!["mcp:tools".to_string()],
            bearer_methods_supported: vec!["header".to_string()],
        };

        let transport = HttpServerTransport::bind(addr, None, Some(oauth))
            .await
            .expect("bind failed");

        let client = reqwest::Client::new();
        let resp = client
            .get(format!(
                "http://{}/.well-known/oauth-protected-resource",
                transport.addr
            ))
            .send()
            .await
            .expect("GET failed");

        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["resource"], "https://example.com/mcp");
    }

    #[test]
    fn test_server_card_serialization() {
        let card = build_server_card(
            "test",
            "0.0.1",
            None,
            vec![McpToolCardEntry {
                name: "tool1".to_string(),
                description: "A tool".to_string(),
                input_schema: serde_json::json!({"type":"object"}),
            }],
            McpAuthInfo {
                scheme: "oauth2".to_string(),
                authorization_server: Some("https://auth.example.com".to_string()),
            },
            vec![McpTransportInfo {
                kind: "http+sse".to_string(),
                url: Some("https://mcp.example.com".to_string()),
            }],
        );

        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(json["tools"][0]["name"], "tool1");
        assert_eq!(json["auth"]["scheme"], "oauth2");
        assert_eq!(json["transport"][0]["kind"], "http+sse");
    }
}
