//! WebSocket client for the OpenAI Responses API.
//!
//! Maintains a persistent WebSocket connection to `wss://api.openai.com/v1/responses`,
//! enabling lower-latency multi-turn interactions via connection-local caching.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::SinkExt;
use futures::stream::{BoxStream, StreamExt};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use super::types::websocket::WsResponseCreate;
use super::types::{CreateResponseRequest, ResponseStreamEvent};

const DEFAULT_WS_URL: &str = "wss://api.openai.com/v1/responses";

/// Maximum connection lifetime before reconnect is required.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(59 * 60); // 59 min (buffer before 60-min limit)

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Persistent WebSocket connection state.
struct WsConnection {
    stream: WsStream,
    connected_at: Instant,
}

/// WebSocket client for the OpenAI Responses API.
///
/// Unlike the HTTP client, this maintains a persistent connection for lower-latency
/// multi-turn tool-call sequences. The server caches the most recent response
/// in-memory per connection, enabling incremental input.
///
/// # Limitations
/// - 60-minute connection lifetime (auto-reconnect handled)
/// - Sequential processing only — one in-flight response per connection
/// - Only the most recent response is cached per connection
pub struct ResponsesWebSocket {
    api_key: String,
    ws_url: String,
    organization: Option<String>,
    connection: Arc<Mutex<Option<WsConnection>>>,
}

impl ResponsesWebSocket {
    /// Create a new WebSocket client (does not connect yet).
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ws_url: DEFAULT_WS_URL.to_string(),
            organization: None,
            connection: Arc::new(Mutex::new(None)),
        }
    }

    /// Set a custom WebSocket URL.
    pub fn with_ws_url(mut self, url: String) -> Self {
        self.ws_url = url;
        self
    }

    /// Set an organization header.
    pub fn with_organization(mut self, org: String) -> Self {
        self.organization = Some(org);
        self
    }

    /// Establish or re-establish the WebSocket connection.
    async fn ensure_connected(&self) -> Result<()> {
        let mut conn = self.connection.lock().await;

        // Check if existing connection is still valid
        if let Some(ref c) = *conn {
            if c.connected_at.elapsed() < CONNECTION_TIMEOUT {
                return Ok(());
            }
            tracing::info!("WebSocket connection approaching 60-min limit, reconnecting");
        }

        // Build the WebSocket request with auth headers
        let mut builder = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&self.ws_url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            builder = builder.header("OpenAI-Organization", org.as_str());
        }

        let request = builder
            .body(())
            .context("Failed to build WebSocket request")?;

        let (ws_stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .context("Failed to connect to OpenAI Responses WebSocket")?;

        tracing::info!(url = %self.ws_url, "WebSocket connection established");

        *conn = Some(WsConnection {
            stream: ws_stream,
            connected_at: Instant::now(),
        });

        Ok(())
    }

    /// Send a `response.create` request and return a stream of events.
    ///
    /// The connection is established lazily on first call and reused for subsequent
    /// calls. If the connection has expired (60-min limit), it is automatically
    /// re-established.
    pub fn create_stream<'a>(
        &'a self,
        req: &'a CreateResponseRequest,
    ) -> BoxStream<'a, Result<ResponseStreamEvent>> {
        Box::pin(async_stream::stream! {
            // Ensure we have a live connection
            if let Err(e) = self.ensure_connected().await {
                yield Err(e);
                return;
            }

            let ws_msg = WsResponseCreate::new(req.clone());
            let json = match serde_json::to_string(&ws_msg) {
                Ok(j) => j,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to serialize WebSocket request: {}", e));
                    return;
                }
            };

            let mut conn = self.connection.lock().await;
            let ws = match conn.as_mut() {
                Some(c) => &mut c.stream,
                None => {
                    yield Err(anyhow::anyhow!("WebSocket connection lost"));
                    return;
                }
            };

            // Send the request
            if let Err(e) = ws.send(WsMessage::Text(json.into())).await {
                // Connection failed — drop it so next call reconnects
                *conn = None;
                yield Err(anyhow::anyhow!("Failed to send WebSocket message: {}", e));
                return;
            }

            // Read response events until we get a terminal event
            loop {
                match ws.next().await {
                    Some(Ok(WsMessage::Text(text))) => {
                        match serde_json::from_str::<ResponseStreamEvent>(&text) {
                            Ok(event) => {
                                let is_terminal = matches!(
                                    &event,
                                    ResponseStreamEvent::ResponseCompleted { .. }
                                    | ResponseStreamEvent::ResponseFailed { .. }
                                    | ResponseStreamEvent::ResponseIncomplete { .. }
                                );
                                yield Ok(event);
                                if is_terminal {
                                    return;
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse WebSocket event: {} — data: {}",
                                    e,
                                    &text[..text.len().min(200)]
                                );
                            }
                        }
                    }
                    Some(Ok(WsMessage::Close(frame))) => {
                        let reason = frame
                            .map(|f| f.reason.to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        tracing::info!(reason = %reason, "WebSocket closed by server");
                        *conn = None;
                        yield Err(anyhow::anyhow!("WebSocket closed by server: {}", reason));
                        return;
                    }
                    Some(Ok(WsMessage::Ping(data))) => {
                        // Respond to ping with pong
                        if let Err(e) = ws.send(WsMessage::Pong(data)).await {
                            tracing::warn!("Failed to send pong: {}", e);
                        }
                    }
                    Some(Ok(_)) => {
                        // Binary, Pong, Frame — ignore
                    }
                    Some(Err(e)) => {
                        *conn = None;
                        yield Err(anyhow::anyhow!("WebSocket error: {}", e));
                        return;
                    }
                    None => {
                        *conn = None;
                        yield Err(anyhow::anyhow!("WebSocket stream ended unexpectedly"));
                        return;
                    }
                }
            }
        })
    }

    /// Disconnect the WebSocket connection.
    pub async fn disconnect(&self) {
        let mut conn = self.connection.lock().await;
        if let Some(mut c) = conn.take() {
            let _ = c.stream.close(None).await;
        }
    }

    /// Check if the connection is currently established and not expired.
    pub async fn is_connected(&self) -> bool {
        let conn = self.connection.lock().await;
        conn.as_ref()
            .is_some_and(|c| c.connected_at.elapsed() < CONNECTION_TIMEOUT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_url() {
        let ws = ResponsesWebSocket::new("test-key".to_string());
        assert_eq!(ws.ws_url, "wss://api.openai.com/v1/responses");
    }

    #[test]
    fn test_custom_url() {
        let ws = ResponsesWebSocket::new("test-key".to_string())
            .with_ws_url("wss://custom.api.com/v1/responses".to_string());
        assert_eq!(ws.ws_url, "wss://custom.api.com/v1/responses");
    }

    #[test]
    fn test_organization() {
        let ws = ResponsesWebSocket::new("test-key".to_string())
            .with_organization("org-123".to_string());
        assert_eq!(ws.organization, Some("org-123".to_string()));
    }

    #[tokio::test]
    async fn test_not_connected_initially() {
        let ws = ResponsesWebSocket::new("test-key".to_string());
        assert!(!ws.is_connected().await);
    }
}
