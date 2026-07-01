//! WebSocket transport — listener and connector.

use crate::error::{ProxyError, ProxyResult};
use crate::request_id::RequestId;
use crate::transport::{InboundConnection, TransportConnector, TransportListener};
use crate::types::{Extensions, ProxyBody, ProxyRequest, ProxyResponse, TransportKind};

use futures::{SinkExt, StreamExt};
use http::{Method, StatusCode, Uri};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_tungstenite::tungstenite::Message;

/// WebSocket listener — accepts WebSocket upgrades, each message becomes a ProxyRequest.
pub struct WebSocketListener {
    addr: SocketAddr,
}

impl WebSocketListener {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

#[async_trait::async_trait]
impl TransportListener for WebSocketListener {
    async fn listen(
        &self,
        tx: mpsc::Sender<InboundConnection>,
        mut shutdown: watch::Receiver<bool>,
    ) -> ProxyResult<()> {
        let listener = TcpListener::bind(self.addr).await?;
        tracing::info!(addr = %self.addr, "WebSocket listener started");

        loop {
            tokio::select! {
                accept = listener.accept() => {
                    let (stream, peer) = accept?;
                    let tx = tx.clone();

                    tokio::spawn(async move {
                        let ws_stream = match tokio_tungstenite::accept_async(stream).await {
                            Ok(ws) => ws,
                            Err(e) => {
                                tracing::debug!(peer = %peer, error = %e, "WebSocket handshake failed");
                                return;
                            }
                        };

                        let (mut sink, mut stream) = ws_stream.split();

                        while let Some(msg) = stream.next().await {
                            let msg = match msg {
                                Ok(m) => m,
                                Err(e) => {
                                    tracing::debug!(peer = %peer, error = %e, "WebSocket read error");
                                    break;
                                }
                            };

                            let body = match &msg {
                                Message::Text(t) => ProxyBody::from(t.as_bytes().to_vec()),
                                Message::Binary(b) => ProxyBody::from(b.to_vec()),
                                Message::Ping(_) | Message::Pong(_) => continue,
                                Message::Close(_) => break,
                                _ => continue,
                            };

                            let uri: Uri = format!("ws://{peer}").parse().unwrap_or_default();
                            let request = ProxyRequest {
                                id: RequestId::new(),
                                method: Method::POST,
                                uri,
                                headers: http::HeaderMap::new(),
                                body,
                                transport: TransportKind::WebSocket,
                                timestamp: chrono::Utc::now(),
                                extensions: Extensions::new(),
                            };

                            let (resp_tx, resp_rx) = oneshot::channel();
                            if tx.send((request, resp_tx)).await.is_err() {
                                break;
                            }

                            if let Ok(resp) = resp_rx.await {
                                let out_msg = Message::Binary(resp.body.into_bytes().to_vec().into());
                                if sink.send(out_msg).await.is_err() {
                                    break;
                                }
                            }
                        }
                    });
                }
                _ = shutdown.changed() => {
                    tracing::info!("WebSocket listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    fn transport_name(&self) -> &str {
        "websocket"
    }
}

/// WebSocket connector — connects to an upstream WebSocket server.
pub struct WebSocketConnector {
    upstream_url: url::Url,
}

impl WebSocketConnector {
    pub fn new(upstream_url: url::Url) -> Self {
        Self { upstream_url }
    }
}

#[async_trait::async_trait]
impl TransportConnector for WebSocketConnector {
    async fn forward(&self, request: ProxyRequest) -> ProxyResult<ProxyResponse> {
        let (mut ws_stream, _) = tokio_tungstenite::connect_async(self.upstream_url.as_str())
            .await
            .map_err(|e| ProxyError::UpstreamUnreachable(e.to_string()))?;

        let msg = Message::Binary(request.body.into_bytes().to_vec().into());
        ws_stream
            .send(msg)
            .await
            .map_err(|e| ProxyError::Transport(e.to_string()))?;

        match ws_stream.next().await {
            Some(Ok(Message::Text(t))) => Ok(ProxyResponse {
                id: request.id,
                status: StatusCode::OK,
                headers: http::HeaderMap::new(),
                body: ProxyBody::from(t.as_bytes().to_vec()),
                timestamp: chrono::Utc::now(),
                extensions: Extensions::new(),
            }),
            Some(Ok(Message::Binary(b))) => Ok(ProxyResponse {
                id: request.id,
                status: StatusCode::OK,
                headers: http::HeaderMap::new(),
                body: ProxyBody::from(b.to_vec()),
                timestamp: chrono::Utc::now(),
                extensions: Extensions::new(),
            }),
            Some(Err(e)) => Err(ProxyError::Transport(e.to_string())),
            _ => Err(ProxyError::Transport("unexpected WebSocket message".into())),
        }
    }

    fn connector_name(&self) -> &str {
        "websocket"
    }
}
