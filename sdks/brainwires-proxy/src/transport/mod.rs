//! Transport abstractions — listeners accept connections, connectors forward to upstream.

#[cfg(feature = "http")]
pub mod http;
#[cfg(feature = "http")]
pub mod sse;

pub mod tcp;
pub mod unix;

#[cfg(feature = "websocket")]
pub mod websocket;

use crate::error::ProxyResult;
use crate::types::{ProxyRequest, ProxyResponse};
use tokio::sync::{mpsc, oneshot};

/// A connection received by a listener: the request plus a channel to send the response back.
pub type InboundConnection = (ProxyRequest, oneshot::Sender<ProxyResponse>);

/// Where a transport listener binds.
#[derive(Debug, Clone)]
pub enum ListenAddr {
    Tcp(std::net::SocketAddr),
    Unix(std::path::PathBuf),
}

/// Where a transport connector sends traffic.
#[derive(Debug, Clone)]
pub enum UpstreamTarget {
    Url(url::Url),
    Tcp { host: String, port: u16 },
    Unix(std::path::PathBuf),
}

/// Accepts inbound connections and sends `(request, response_channel)` pairs
/// through an mpsc channel for the proxy core to process.
#[async_trait::async_trait]
pub trait TransportListener: Send + Sync {
    /// Start accepting connections. Sends each connection through `tx`.
    /// Returns when the listener is shut down.
    async fn listen(
        &self,
        tx: mpsc::Sender<InboundConnection>,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> ProxyResult<()>;

    /// Human-readable transport name.
    fn transport_name(&self) -> &str;
}

/// Forwards a `ProxyRequest` to an upstream target and returns the `ProxyResponse`.
#[async_trait::async_trait]
pub trait TransportConnector: Send + Sync {
    /// Forward a request to the upstream and return the response.
    async fn forward(&self, request: ProxyRequest) -> ProxyResult<ProxyResponse>;

    /// Human-readable connector name.
    fn connector_name(&self) -> &str;
}
