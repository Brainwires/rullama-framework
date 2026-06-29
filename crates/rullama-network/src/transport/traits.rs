use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{MessageEnvelope, TransportType};

/// A network address that a transport can connect to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportAddress {
    /// A Unix domain socket path (IPC).
    Unix(PathBuf),
    /// A TCP socket address (mesh peer-to-peer).
    Tcp(SocketAddr),
    /// A URL endpoint (HTTP, WebSocket, A2A).
    Url(String),
    /// A pub/sub channel or topic name.
    Channel(String),
}

impl std::fmt::Display for TransportAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportAddress::Unix(path) => write!(f, "unix://{}", path.display()),
            TransportAddress::Tcp(addr) => write!(f, "tcp://{addr}"),
            TransportAddress::Url(url) => write!(f, "{url}"),
            TransportAddress::Channel(ch) => write!(f, "channel://{ch}"),
        }
    }
}

/// The core transport trait.
///
/// Every networking paradigm (IPC, Remote, TCP, A2A, Pub/Sub) implements
/// this trait so that the routing and application layers can work with
/// any transport uniformly.
///
/// # Lifecycle
///
/// 1. Create the transport (constructor is transport-specific)
/// 2. Call [`connect`](Transport::connect) with a target address
/// 3. Use [`send`](Transport::send) and [`receive`](Transport::receive)
/// 4. Call [`disconnect`](Transport::disconnect) when done
#[async_trait]
pub trait Transport: Send + Sync {
    /// Connect to a remote peer or service.
    async fn connect(&mut self, target: &TransportAddress) -> Result<()>;

    /// Disconnect from the current peer or service.
    async fn disconnect(&mut self) -> Result<()>;

    /// Send a message envelope to the connected peer(s).
    async fn send(&self, envelope: &MessageEnvelope) -> Result<()>;

    /// Receive the next message envelope from the connected peer(s).
    ///
    /// This blocks (async) until a message is available or the connection
    /// is closed. Returns `None` on clean shutdown.
    async fn receive(&self) -> Result<Option<MessageEnvelope>>;

    /// The transport type identifier.
    fn transport_type(&self) -> TransportType;

    /// Whether this transport is currently connected.
    fn is_connected(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn transport_address_display() {
        let unix = TransportAddress::Unix(PathBuf::from("/tmp/agent.sock"));
        assert_eq!(unix.to_string(), "unix:///tmp/agent.sock");

        let tcp = TransportAddress::Tcp(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9090));
        assert_eq!(tcp.to_string(), "tcp://127.0.0.1:9090");

        let url = TransportAddress::Url("https://example.com/a2a".into());
        assert_eq!(url.to_string(), "https://example.com/a2a");

        let channel = TransportAddress::Channel("status-updates".into());
        assert_eq!(channel.to_string(), "channel://status-updates");
    }

    #[test]
    fn transport_address_serde_roundtrip() {
        let addrs = vec![
            TransportAddress::Unix(PathBuf::from("/tmp/test.sock")),
            TransportAddress::Tcp("127.0.0.1:8080".parse().unwrap()),
            TransportAddress::Url("wss://api.example.com".into()),
            TransportAddress::Channel("events".into()),
        ];
        for addr in addrs {
            let json = serde_json::to_string(&addr).unwrap();
            let deserialized: TransportAddress = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, addr);
        }
    }

    #[test]
    fn transport_address_equality() {
        let a = TransportAddress::Channel("test".into());
        let b = TransportAddress::Channel("test".into());
        let c = TransportAddress::Channel("other".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
