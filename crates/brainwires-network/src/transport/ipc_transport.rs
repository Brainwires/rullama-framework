use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufWriter};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

use super::traits::{Transport, TransportAddress};
use crate::ipc::IpcCipher;
use crate::{MessageEnvelope, TransportType};

/// Maximum message size (16 MB).
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Unix-socket IPC transport for agent-to-agent messaging.
///
/// Uses the same ChaCha20-Poly1305 encryption as the existing IPC layer
/// but speaks the [`MessageEnvelope`] wire format instead of the
/// viewer/agent protocol.
///
/// # Wire format
///
/// Each message on the wire is:
/// ```text
/// [4-byte big-endian length][encrypted JSON blob]
/// ```
///
/// The encrypted blob is a ChaCha20-Poly1305 authenticated ciphertext
/// of the JSON-serialized [`MessageEnvelope`].
pub struct IpcTransport {
    /// Socket path (set via connect or constructor).
    socket_path: Option<PathBuf>,
    /// Optional shared secret for encryption (if None, messages are plaintext).
    secret: Option<String>,
    /// Active connection state.
    conn: Option<Arc<IpcConn>>,
}

struct IpcConn {
    reader: Mutex<tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>>,
    writer: Mutex<BufWriter<tokio::net::unix::OwnedWriteHalf>>,
    cipher: Option<IpcCipher>,
}

impl IpcTransport {
    /// Create a new IPC transport.
    ///
    /// If `secret` is provided, messages are encrypted with ChaCha20-Poly1305
    /// using a key derived from the secret.
    pub fn new(secret: Option<String>) -> Self {
        Self {
            socket_path: None,
            secret,
            conn: None,
        }
    }

    /// Create a new IPC transport pre-configured with a socket path.
    pub fn with_path(path: impl Into<PathBuf>, secret: Option<String>) -> Self {
        Self {
            socket_path: Some(path.into()),
            secret,
            conn: None,
        }
    }
}

#[async_trait]
impl Transport for IpcTransport {
    async fn connect(&mut self, target: &TransportAddress) -> Result<()> {
        let path = match target {
            TransportAddress::Unix(p) => p.clone(),
            _ => bail!("IpcTransport only supports Unix addresses"),
        };

        if !path.exists() {
            bail!("Socket not found: {}", path.display());
        }

        let stream = UnixStream::connect(&path)
            .await
            .with_context(|| format!("Failed to connect to {}", path.display()))?;

        let (read_half, write_half) = stream.into_split();
        let cipher = self.secret.as_deref().map(IpcCipher::from_session_token);

        self.socket_path = Some(path);
        self.conn = Some(Arc::new(IpcConn {
            reader: Mutex::new(tokio::io::BufReader::new(read_half)),
            writer: Mutex::new(BufWriter::new(write_half)),
            cipher,
        }));

        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.conn = None;
        Ok(())
    }

    async fn send(&self, envelope: &MessageEnvelope) -> Result<()> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("IpcTransport not connected"))?;

        let json = serde_json::to_vec(envelope)?;
        if json.len() > MAX_MESSAGE_SIZE {
            bail!("Message exceeds maximum size of {MAX_MESSAGE_SIZE} bytes");
        }

        let wire_bytes = if let Some(cipher) = &conn.cipher {
            cipher.encrypt(&json).context("Failed to encrypt message")?
        } else {
            json
        };

        let mut writer = conn.writer.lock().await;
        let len_buf = (wire_bytes.len() as u32).to_be_bytes();
        writer.write_all(&len_buf).await?;
        writer.write_all(&wire_bytes).await?;
        writer.flush().await?;

        Ok(())
    }

    async fn receive(&self) -> Result<Option<MessageEnvelope>> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("IpcTransport not connected"))?;

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

        // Read message bytes
        let mut wire_bytes = vec![0u8; msg_len];
        reader.read_exact(&mut wire_bytes).await?;

        // Decrypt if needed
        let json_bytes = if let Some(cipher) = &conn.cipher {
            cipher
                .decrypt(&wire_bytes)
                .context("Failed to decrypt message")?
        } else {
            wire_bytes
        };

        let envelope: MessageEnvelope =
            serde_json::from_slice(&json_bytes).context("Failed to parse MessageEnvelope")?;

        Ok(Some(envelope))
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Ipc
    }

    fn is_connected(&self) -> bool {
        self.conn.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Payload;
    use tokio::net::UnixListener;
    use uuid::Uuid;

    #[tokio::test]
    async fn ipc_transport_plaintext_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("test.sock");

        let listener = UnixListener::bind(&socket_path).unwrap();
        let path_clone = socket_path.clone();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, write_half) = stream.into_split();
            let conn = Arc::new(IpcConn {
                reader: Mutex::new(tokio::io::BufReader::new(read_half)),
                writer: Mutex::new(BufWriter::new(write_half)),
                cipher: None,
            });

            // Receive
            let mut reader = conn.reader.lock().await;
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf).await.unwrap();
            let msg_len = u32::from_be_bytes(len_buf) as usize;
            let mut buf = vec![0u8; msg_len];
            reader.read_exact(&mut buf).await.unwrap();
            let envelope: MessageEnvelope = serde_json::from_slice(&buf).unwrap();
            drop(reader);

            // Send reply
            let reply = envelope.reply(Uuid::new_v4(), Payload::Text("pong".into()));
            let json = serde_json::to_vec(&reply).unwrap();
            let mut writer = conn.writer.lock().await;
            writer
                .write_all(&(json.len() as u32).to_be_bytes())
                .await
                .unwrap();
            writer.write_all(&json).await.unwrap();
            writer.flush().await.unwrap();
        });

        let mut transport = IpcTransport::new(None);
        transport
            .connect(&TransportAddress::Unix(path_clone))
            .await
            .unwrap();

        assert!(transport.is_connected());

        let env =
            MessageEnvelope::direct(Uuid::new_v4(), Uuid::new_v4(), Payload::Text("ping".into()));
        transport.send(&env).await.unwrap();

        let reply = transport.receive().await.unwrap().unwrap();
        assert_eq!(reply.correlation_id, Some(env.id));
        match reply.payload {
            Payload::Text(s) => assert_eq!(s, "pong"),
            _ => panic!("expected Text payload"),
        }

        transport.disconnect().await.unwrap();
        assert!(!transport.is_connected());

        server.await.unwrap();
    }

    #[test]
    fn ipc_transport_type() {
        let t = IpcTransport::new(None);
        assert_eq!(t.transport_type(), TransportType::Ipc);
    }
}
