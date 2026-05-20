use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufWriter};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use super::traits::{Transport, TransportAddress};
use crate::{MessageEnvelope, TransportType};

/// Maximum message size (16 MB).
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Direct TCP transport for peer-to-peer agent communication.
///
/// Uses length-prefixed JSON serialization of [`MessageEnvelope`]s
/// over a TCP stream. Designed for mesh networking scenarios where
/// agents connect directly to each other.
///
/// # Wire format
///
/// ```text
/// [4-byte big-endian length][JSON-encoded MessageEnvelope]
/// ```
pub struct TcpTransport {
    /// Remote address (set on connect).
    remote_addr: Option<SocketAddr>,
    /// Active connection state.
    conn: Option<Arc<TcpConn>>,
}

struct TcpConn {
    reader: Mutex<tokio::io::BufReader<tokio::net::tcp::OwnedReadHalf>>,
    writer: Mutex<BufWriter<tokio::net::tcp::OwnedWriteHalf>>,
}

impl TcpTransport {
    /// Create a new TCP transport.
    pub fn new() -> Self {
        Self {
            remote_addr: None,
            conn: None,
        }
    }

    /// Create a TCP transport from an already-connected stream.
    ///
    /// Useful on the server side after accepting a connection.
    pub fn from_stream(stream: TcpStream) -> Result<Self> {
        let remote_addr = stream.peer_addr().ok();
        let (read_half, write_half) = stream.into_split();

        Ok(Self {
            remote_addr,
            conn: Some(Arc::new(TcpConn {
                reader: Mutex::new(tokio::io::BufReader::new(read_half)),
                writer: Mutex::new(BufWriter::new(write_half)),
            })),
        })
    }
}

impl Default for TcpTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transport for TcpTransport {
    async fn connect(&mut self, target: &TransportAddress) -> Result<()> {
        let addr = match target {
            TransportAddress::Tcp(a) => *a,
            _ => bail!("TcpTransport only supports TCP addresses"),
        };

        let stream = TcpStream::connect(addr)
            .await
            .with_context(|| format!("Failed to connect to {addr}"))?;

        // Disable Nagle's algorithm for lower latency
        stream.set_nodelay(true)?;

        let (read_half, write_half) = stream.into_split();

        self.remote_addr = Some(addr);
        self.conn = Some(Arc::new(TcpConn {
            reader: Mutex::new(tokio::io::BufReader::new(read_half)),
            writer: Mutex::new(BufWriter::new(write_half)),
        }));

        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.conn = None;
        self.remote_addr = None;
        Ok(())
    }

    async fn send(&self, envelope: &MessageEnvelope) -> Result<()> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TcpTransport not connected"))?;

        let json = serde_json::to_vec(envelope)?;
        if json.len() > MAX_MESSAGE_SIZE {
            bail!("Message exceeds maximum size of {MAX_MESSAGE_SIZE} bytes");
        }

        let mut writer = conn.writer.lock().await;
        let len_buf = (json.len() as u32).to_be_bytes();
        writer.write_all(&len_buf).await?;
        writer.write_all(&json).await?;
        writer.flush().await?;

        Ok(())
    }

    async fn receive(&self) -> Result<Option<MessageEnvelope>> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TcpTransport not connected"))?;

        let mut reader = conn.reader.lock().await;

        // Read length prefix
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let msg_len = u32::from_be_bytes(len_buf) as usize;
        if msg_len > MAX_MESSAGE_SIZE {
            bail!("Message exceeds maximum size of {MAX_MESSAGE_SIZE} bytes");
        }

        let mut buf = vec![0u8; msg_len];
        reader.read_exact(&mut buf).await?;

        let envelope: MessageEnvelope =
            serde_json::from_slice(&buf).context("Failed to parse MessageEnvelope")?;

        Ok(Some(envelope))
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Tcp
    }

    fn is_connected(&self) -> bool {
        self.conn.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Payload;
    use tokio::net::TcpListener;
    use uuid::Uuid;

    #[tokio::test]
    async fn tcp_transport_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut transport = TcpTransport::from_stream(stream).unwrap();
            assert!(transport.is_connected());

            // Receive message
            let env = transport.receive().await.unwrap().unwrap();
            match &env.payload {
                Payload::Text(s) => assert_eq!(s, "ping"),
                _ => panic!("expected Text"),
            }

            // Send reply
            let reply = env.reply(Uuid::new_v4(), Payload::Text("pong".into()));
            transport.send(&reply).await.unwrap();

            transport.disconnect().await.unwrap();
        });

        let mut client = TcpTransport::new();
        client.connect(&TransportAddress::Tcp(addr)).await.unwrap();
        assert!(client.is_connected());

        let env =
            MessageEnvelope::direct(Uuid::new_v4(), Uuid::new_v4(), Payload::Text("ping".into()));
        client.send(&env).await.unwrap();

        let reply = client.receive().await.unwrap().unwrap();
        match reply.payload {
            Payload::Text(s) => assert_eq!(s, "pong"),
            _ => panic!("expected Text"),
        }

        client.disconnect().await.unwrap();
        server.await.unwrap();
    }

    #[test]
    fn tcp_transport_type() {
        let t = TcpTransport::new();
        assert_eq!(t.transport_type(), TransportType::Tcp);
        assert!(!t.is_connected());
    }
}
