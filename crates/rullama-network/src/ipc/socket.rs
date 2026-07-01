//! Unix Socket IPC Utilities
//!
//! Provides async read/write helpers for Unix domain socket communication
//! between the TUI viewer and Agent process.
//!
//! # Encryption
//!
//! This module provides both plaintext and encrypted IPC:
//! - `IpcReader`/`IpcWriter` - Plaintext JSON over newlines (legacy, for handshake)
//! - `EncryptedIpcReader`/`EncryptedIpcWriter` - ChaCha20-Poly1305 encrypted messages
//!
//! The encrypted variants use the session token to derive the encryption key,
//! providing confidentiality and integrity for all messages after handshake.

use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

use super::crypto::IpcCipher;

/// Maximum message size (16 MB)
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Reader for receiving IPC messages
pub struct IpcReader {
    reader: BufReader<OwnedReadHalf>,
}

impl IpcReader {
    /// Create a new IPC reader from a Unix stream read half
    pub fn new(read_half: OwnedReadHalf) -> Self {
        Self {
            reader: BufReader::new(read_half),
        }
    }

    /// Read a message from the socket
    ///
    /// Messages are newline-delimited JSON.
    /// Returns None on EOF.
    pub async fn read<T: DeserializeOwned>(&mut self) -> Result<Option<T>> {
        let mut line = String::new();
        let bytes_read = self.reader.read_line(&mut line).await?;

        if bytes_read == 0 {
            return Ok(None); // EOF
        }

        if line.len() > MAX_MESSAGE_SIZE {
            bail!("Message exceeds maximum size of {} bytes", MAX_MESSAGE_SIZE);
        }

        let message: T = serde_json::from_str(line.trim())
            .with_context(|| format!("Failed to parse IPC message: {}", line.trim()))?;

        Ok(Some(message))
    }
}

/// Writer for sending IPC messages
pub struct IpcWriter {
    writer: BufWriter<OwnedWriteHalf>,
}

impl IpcWriter {
    /// Create a new IPC writer from a Unix stream write half
    pub fn new(write_half: OwnedWriteHalf) -> Self {
        Self {
            writer: BufWriter::new(write_half),
        }
    }

    /// Write a message to the socket
    ///
    /// Messages are serialized as newline-delimited JSON.
    pub async fn write<T: Serialize>(&mut self, message: &T) -> Result<()> {
        let json = serde_json::to_string(message).context("Failed to serialize IPC message")?;

        if json.len() > MAX_MESSAGE_SIZE {
            bail!("Message exceeds maximum size of {} bytes", MAX_MESSAGE_SIZE);
        }

        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        Ok(())
    }
}

/// IPC connection handle (combines reader and writer)
pub struct IpcConnection {
    /// The reader half of the connection.
    pub reader: IpcReader,
    /// The writer half of the connection.
    pub writer: IpcWriter,
}

impl IpcConnection {
    /// Create an IPC connection from a Unix stream
    pub fn from_stream(stream: UnixStream) -> Self {
        let (read_half, write_half) = stream.into_split();
        Self {
            reader: IpcReader::new(read_half),
            writer: IpcWriter::new(write_half),
        }
    }

    /// Connect to an agent socket by path
    pub async fn connect(socket_path: &Path) -> Result<Self> {
        if !socket_path.exists() {
            bail!("Agent socket not found: {}", socket_path.display());
        }

        let stream = UnixStream::connect(socket_path).await.with_context(|| {
            format!(
                "Failed to connect to agent socket: {}",
                socket_path.display()
            )
        })?;

        Ok(Self::from_stream(stream))
    }

    /// Connect to an agent by session ID, looking up the socket in sessions_dir
    pub async fn connect_to_agent(sessions_dir: &Path, session_id: &str) -> Result<Self> {
        let socket_path = get_agent_socket_path(sessions_dir, session_id);
        Self::connect(&socket_path).await
    }

    /// Split into reader and writer
    pub fn split(self) -> (IpcReader, IpcWriter) {
        (self.reader, self.writer)
    }

    /// Upgrade to encrypted connection using the session token
    ///
    /// This should be called after the handshake is complete and both
    /// sides have agreed on the session token.
    pub fn upgrade_to_encrypted(self, session_token: &str) -> EncryptedIpcConnection {
        let cipher = Arc::new(IpcCipher::from_session_token(session_token));
        let (read_half, write_half) = (self.reader, self.writer);
        EncryptedIpcConnection {
            reader: EncryptedIpcReader::new(read_half, Arc::clone(&cipher)),
            writer: EncryptedIpcWriter::new(write_half, cipher),
        }
    }
}

// ============================================================================
// Encrypted IPC (ChaCha20-Poly1305)
// ============================================================================

/// Encrypted reader for receiving IPC messages
///
/// Uses ChaCha20-Poly1305 authenticated encryption.
/// Message format: [4-byte length][encrypted data]
pub struct EncryptedIpcReader {
    inner: IpcReader,
    cipher: Arc<IpcCipher>,
}

impl EncryptedIpcReader {
    /// Create a new encrypted IPC reader
    pub fn new(reader: IpcReader, cipher: Arc<IpcCipher>) -> Self {
        Self {
            inner: reader,
            cipher,
        }
    }

    /// Read an encrypted message from the socket
    ///
    /// Messages are length-prefixed encrypted blobs.
    /// Returns None on EOF.
    pub async fn read<T: DeserializeOwned>(&mut self) -> Result<Option<T>> {
        // Read length prefix (4 bytes, big-endian)
        let mut len_buf = [0u8; 4];
        match self.inner.reader.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let msg_len = u32::from_be_bytes(len_buf) as usize;

        if msg_len > MAX_MESSAGE_SIZE {
            bail!(
                "Encrypted message exceeds maximum size of {} bytes",
                MAX_MESSAGE_SIZE
            );
        }

        // Read encrypted message
        let mut encrypted = vec![0u8; msg_len];
        self.inner.reader.read_exact(&mut encrypted).await?;

        // Decrypt
        let plaintext = self
            .cipher
            .decrypt(&encrypted)
            .context("Failed to decrypt IPC message")?;

        // Deserialize
        let message: T =
            serde_json::from_slice(&plaintext).context("Failed to parse decrypted IPC message")?;

        Ok(Some(message))
    }
}

/// Encrypted writer for sending IPC messages
///
/// Uses ChaCha20-Poly1305 authenticated encryption.
/// Message format: [4-byte length][encrypted data]
pub struct EncryptedIpcWriter {
    inner: IpcWriter,
    cipher: Arc<IpcCipher>,
}

impl EncryptedIpcWriter {
    /// Create a new encrypted IPC writer
    pub fn new(writer: IpcWriter, cipher: Arc<IpcCipher>) -> Self {
        Self {
            inner: writer,
            cipher,
        }
    }

    /// Write an encrypted message to the socket
    ///
    /// Messages are serialized, encrypted, then length-prefixed.
    pub async fn write<T: Serialize>(&mut self, message: &T) -> Result<()> {
        // Serialize
        let json = serde_json::to_vec(message).context("Failed to serialize IPC message")?;

        if json.len() > MAX_MESSAGE_SIZE {
            bail!("Message exceeds maximum size of {} bytes", MAX_MESSAGE_SIZE);
        }

        // Encrypt
        let encrypted = self
            .cipher
            .encrypt(&json)
            .context("Failed to encrypt IPC message")?;

        // Write length prefix (4 bytes, big-endian)
        let len_buf = (encrypted.len() as u32).to_be_bytes();
        self.inner.writer.write_all(&len_buf).await?;

        // Write encrypted message
        self.inner.writer.write_all(&encrypted).await?;
        self.inner.writer.flush().await?;

        Ok(())
    }
}

/// Encrypted IPC connection handle
pub struct EncryptedIpcConnection {
    /// The encrypted reader half of the connection.
    pub reader: EncryptedIpcReader,
    /// The encrypted writer half of the connection.
    pub writer: EncryptedIpcWriter,
}

impl EncryptedIpcConnection {
    /// Create an encrypted IPC connection from a Unix stream and session token
    pub fn from_stream(stream: UnixStream, session_token: &str) -> Self {
        let cipher = Arc::new(IpcCipher::from_session_token(session_token));
        let (read_half, write_half) = stream.into_split();
        Self {
            reader: EncryptedIpcReader::new(IpcReader::new(read_half), Arc::clone(&cipher)),
            writer: EncryptedIpcWriter::new(IpcWriter::new(write_half), cipher),
        }
    }

    /// Split into encrypted reader and writer
    pub fn split(self) -> (EncryptedIpcReader, EncryptedIpcWriter) {
        (self.reader, self.writer)
    }
}

// ============================================================================
// Path Helpers
// ============================================================================

/// Get the socket path for an agent session
pub fn get_agent_socket_path(sessions_dir: &Path, session_id: &str) -> PathBuf {
    sessions_dir.join(format!("{}.sock", session_id))
}

/// Get the token file path for an agent session
pub fn get_session_token_path(sessions_dir: &Path, session_id: &str) -> PathBuf {
    sessions_dir.join(format!("{}.token", session_id))
}

// ============================================================================
// Session Token Management (for secure IPC authentication)
// ============================================================================

/// Generate a cryptographically secure session token (64 hex characters = 256 bits)
pub fn generate_session_token() -> String {
    use rand::Rng;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// Write session token to disk with secure permissions (0600)
/// This should only be called by the agent process that owns the session
pub fn write_session_token(sessions_dir: &Path, session_id: &str, token: &str) -> Result<()> {
    let token_path = get_session_token_path(sessions_dir, session_id);

    // Ensure parent directory exists
    if let Some(parent) = token_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write token
    std::fs::write(&token_path, token)?;

    // Set secure permissions (0600 = owner read/write only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&token_path, std::fs::Permissions::from_mode(0o600))?;
    }

    tracing::debug!(
        "Wrote session token: {} (0600 permissions)",
        token_path.display()
    );
    Ok(())
}

/// Read session token from disk
/// This is used by clients that need to reattach to a session
pub fn read_session_token(sessions_dir: &Path, session_id: &str) -> Result<Option<String>> {
    let token_path = get_session_token_path(sessions_dir, session_id);

    if !token_path.exists() {
        return Ok(None);
    }

    let token = std::fs::read_to_string(&token_path)
        .with_context(|| format!("Failed to read session token from {}", token_path.display()))?;

    Ok(Some(token.trim().to_string()))
}

/// Delete session token file
pub fn delete_session_token(sessions_dir: &Path, session_id: &str) -> Result<()> {
    let token_path = get_session_token_path(sessions_dir, session_id);

    if token_path.exists() {
        std::fs::remove_file(&token_path)
            .with_context(|| format!("Failed to delete session token: {}", token_path.display()))?;
        tracing::debug!("Deleted session token: {}", token_path.display());
    }

    Ok(())
}

/// Validate that a provided token matches the stored token for a session
/// Returns true if tokens match, false if they don't match or no token exists
pub fn validate_session_token(sessions_dir: &Path, session_id: &str, provided_token: &str) -> bool {
    match read_session_token(sessions_dir, session_id) {
        Ok(Some(stored_token)) => {
            // Use constant-time comparison to prevent timing attacks
            use subtle::ConstantTimeEq;
            provided_token
                .as_bytes()
                .ct_eq(stored_token.as_bytes())
                .into()
        }
        Ok(None) => {
            tracing::warn!("No session token found for session {}", session_id);
            false
        }
        Err(e) => {
            tracing::error!("Failed to read session token for {}: {}", session_id, e);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{AgentMessage, ViewerMessage};
    use super::*;
    use tokio::net::UnixListener;

    #[tokio::test]
    async fn test_ipc_roundtrip() {
        // Create a temporary socket
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("test.sock");

        // Start listener
        let listener = UnixListener::bind(&socket_path).unwrap();

        // Spawn server task
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut conn = IpcConnection::from_stream(stream);

            // Read message from client
            let msg: ViewerMessage = conn.reader.read().await.unwrap().unwrap();
            match msg {
                ViewerMessage::UserInput { content, .. } => {
                    assert_eq!(content, "Hello");
                }
                _ => panic!("Unexpected message type"),
            }

            // Send response
            let response = AgentMessage::Ack {
                command: "user_input".to_string(),
            };
            conn.writer.write(&response).await.unwrap();
        });

        // Client connects and sends message
        let stream = UnixStream::connect(&socket_path).await.unwrap();
        let mut conn = IpcConnection::from_stream(stream);

        let msg = ViewerMessage::UserInput {
            content: "Hello".to_string(),
            context_files: vec![],
        };
        conn.writer.write(&msg).await.unwrap();

        // Read response
        let response: AgentMessage = conn.reader.read().await.unwrap().unwrap();
        match response {
            AgentMessage::Ack { command } => {
                assert_eq!(command, "user_input");
            }
            _ => panic!("Unexpected response type"),
        }

        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_encrypted_ipc_roundtrip() {
        // Create a temporary socket
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("encrypted_test.sock");

        // Start listener
        let listener = UnixListener::bind(&socket_path).unwrap();

        // Shared session token for encryption
        let session_token = "test-session-token-for-encrypted-ipc";

        let server_token = session_token.to_string();
        // Spawn server task
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();

            // Upgrade to encrypted connection
            let conn = IpcConnection::from_stream(stream);
            let mut encrypted_conn = conn.upgrade_to_encrypted(&server_token);

            // Read encrypted message from client
            let msg: ViewerMessage = encrypted_conn.reader.read().await.unwrap().unwrap();
            match msg {
                ViewerMessage::UserInput { content, .. } => {
                    assert_eq!(content, "Encrypted Hello!");
                }
                _ => panic!("Unexpected message type"),
            }

            // Send encrypted response
            let response = AgentMessage::Ack {
                command: "encrypted_user_input".to_string(),
            };
            encrypted_conn.writer.write(&response).await.unwrap();
        });

        // Client connects and sends encrypted message
        let stream = UnixStream::connect(&socket_path).await.unwrap();
        let conn = IpcConnection::from_stream(stream);
        let mut encrypted_conn = conn.upgrade_to_encrypted(session_token);

        let msg = ViewerMessage::UserInput {
            content: "Encrypted Hello!".to_string(),
            context_files: vec![],
        };
        encrypted_conn.writer.write(&msg).await.unwrap();

        // Read encrypted response
        let response: AgentMessage = encrypted_conn.reader.read().await.unwrap().unwrap();
        match response {
            AgentMessage::Ack { command } => {
                assert_eq!(command, "encrypted_user_input");
            }
            _ => panic!("Unexpected response type"),
        }

        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_encrypted_ipc_wrong_key_fails() {
        // Create a temporary socket
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("wrong_key_test.sock");

        // Start listener
        let listener = UnixListener::bind(&socket_path).unwrap();

        // Spawn server task with DIFFERENT token
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();

            // Server uses different token
            let conn = IpcConnection::from_stream(stream);
            let mut encrypted_conn = conn.upgrade_to_encrypted("server-token-different");

            // Read should fail due to wrong key
            let result: Result<Option<ViewerMessage>> = encrypted_conn.reader.read().await;
            assert!(result.is_err(), "Should fail to decrypt with wrong key");
        });

        // Client connects with different token
        let stream = UnixStream::connect(&socket_path).await.unwrap();
        let conn = IpcConnection::from_stream(stream);
        let mut encrypted_conn = conn.upgrade_to_encrypted("client-token-different");

        let msg = ViewerMessage::UserInput {
            content: "This will fail".to_string(),
            context_files: vec![],
        };
        encrypted_conn.writer.write(&msg).await.unwrap();

        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_encrypted_multiple_messages() {
        // Create a temporary socket
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("multi_msg_test.sock");

        // Start listener
        let listener = UnixListener::bind(&socket_path).unwrap();
        let session_token = "multi-message-token";

        let server_token = session_token.to_string();
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let conn = IpcConnection::from_stream(stream);
            let mut encrypted_conn = conn.upgrade_to_encrypted(&server_token);

            // Read and respond to multiple messages
            for i in 0..5 {
                let msg: ViewerMessage = encrypted_conn.reader.read().await.unwrap().unwrap();
                match msg {
                    ViewerMessage::UserInput { content, .. } => {
                        assert_eq!(content, format!("Message {}", i));
                    }
                    _ => panic!("Unexpected message type"),
                }

                let response = AgentMessage::Ack {
                    command: format!("ack_{}", i),
                };
                encrypted_conn.writer.write(&response).await.unwrap();
            }
        });

        // Client sends multiple encrypted messages
        let stream = UnixStream::connect(&socket_path).await.unwrap();
        let conn = IpcConnection::from_stream(stream);
        let mut encrypted_conn = conn.upgrade_to_encrypted(session_token);

        for i in 0..5 {
            let msg = ViewerMessage::UserInput {
                content: format!("Message {}", i),
                context_files: vec![],
            };
            encrypted_conn.writer.write(&msg).await.unwrap();

            let response: AgentMessage = encrypted_conn.reader.read().await.unwrap().unwrap();
            match response {
                AgentMessage::Ack { command } => {
                    assert_eq!(command, format!("ack_{}", i));
                }
                _ => panic!("Unexpected response type"),
            }
        }

        server_task.await.unwrap();
    }

    #[test]
    fn test_session_token_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let sessions_dir = temp_dir.path();

        let token = generate_session_token();
        assert_eq!(token.len(), 64); // 32 bytes = 64 hex chars

        write_session_token(sessions_dir, "test-session", &token).unwrap();
        let read_token = read_session_token(sessions_dir, "test-session").unwrap();
        assert_eq!(read_token, Some(token.clone()));

        assert!(validate_session_token(sessions_dir, "test-session", &token));
        assert!(!validate_session_token(
            sessions_dir,
            "test-session",
            "wrong-token"
        ));

        delete_session_token(sessions_dir, "test-session").unwrap();
        let read_after_delete = read_session_token(sessions_dir, "test-session").unwrap();
        assert_eq!(read_after_delete, None);
    }
}
