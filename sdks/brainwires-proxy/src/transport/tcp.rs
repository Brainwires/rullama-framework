//! Raw TCP bidirectional proxy transport.

use crate::error::ProxyResult;
use crate::request_id::RequestId;
use crate::transport::{InboundConnection, TransportListener};
use crate::types::{Extensions, ProxyBody, ProxyRequest, TransportKind};

use http::{Method, Uri};
use std::net::SocketAddr;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, watch};

const DEFAULT_MAX_READ_BYTES: usize = 10 * 1024 * 1024;
const BUFFER_INITIAL_CAPACITY: usize = 4096;
const READ_CHUNK_SIZE: usize = 8192;

/// Raw TCP listener — reads the full payload as a single ProxyRequest body.
pub struct TcpRawListener {
    addr: SocketAddr,
    max_read: usize,
}

impl TcpRawListener {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            max_read: DEFAULT_MAX_READ_BYTES, // 10 MiB default
        }
    }

    pub fn with_max_read(mut self, max: usize) -> Self {
        self.max_read = max;
        self
    }
}

#[async_trait::async_trait]
impl TransportListener for TcpRawListener {
    async fn listen(
        &self,
        tx: mpsc::Sender<InboundConnection>,
        mut shutdown: watch::Receiver<bool>,
    ) -> ProxyResult<()> {
        let listener = TcpListener::bind(self.addr).await?;
        tracing::info!(addr = %self.addr, "TCP raw listener started");

        loop {
            tokio::select! {
                accept = listener.accept() => {
                    let (mut stream, peer) = accept?;
                    let tx = tx.clone();
                    let max_read = self.max_read;

                    tokio::spawn(async move {
                        let mut buf = Vec::with_capacity(BUFFER_INITIAL_CAPACITY);
                        let mut tmp = vec![0u8; READ_CHUNK_SIZE];

                        loop {
                            match stream.read(&mut tmp).await {
                                Ok(0) => break,
                                Ok(n) => {
                                    buf.extend_from_slice(&tmp[..n]);
                                    if buf.len() >= max_read {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    tracing::debug!(peer = %peer, error = %e, "TCP read error");
                                    return;
                                }
                            }
                        }

                        // Represent as a synthetic HTTP-like request
                        let uri: Uri = format!("tcp://{peer}").parse().unwrap_or_default();
                        let request = ProxyRequest {
                            id: RequestId::new(),
                            method: Method::POST,
                            uri,
                            headers: http::HeaderMap::new(),
                            body: ProxyBody::from(buf),
                            transport: TransportKind::Tcp,
                            timestamp: chrono::Utc::now(),
                            extensions: Extensions::new(),
                        };

                        let (resp_tx, resp_rx) = oneshot::channel();
                        if tx.send((request, resp_tx)).await.is_ok()
                            && let Ok(resp) = resp_rx.await {
                                use tokio::io::AsyncWriteExt;
                                let _ = stream.write_all(resp.body.as_bytes()).await;
                            }
                    });
                }
                _ = shutdown.changed() => {
                    tracing::info!("TCP raw listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    fn transport_name(&self) -> &str {
        "tcp"
    }
}
