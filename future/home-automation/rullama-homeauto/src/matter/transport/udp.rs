/// Matter UDP transport layer.
///
/// Provides authenticated encryption using AES-128-CCM over UDP sockets.
///
/// ## Encryption scheme
///
/// - Cipher   : AES-128-CCM (16-byte key, 13-byte nonce, 16-byte tag)
/// - Nonce    : session_id (2 LE) ‖ message_counter (4 LE) ‖ 0x00 × 7 padding
/// - AAD      : the encoded message header bytes (everything before the payload)
/// - Encrypt  : payload → ciphertext ‖ tag  (tag appended)
/// - Decrypt  : strip trailing 16-byte tag, verify MAC, return plaintext
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::UdpSocket;
use tokio::sync::Mutex;
use zeroize::Zeroize;

// ccm re-exports aead at ccm::aead, so we avoid a direct aead dep.
use ccm::{
    Ccm,
    aead::{Aead, KeyInit, generic_array::GenericArray},
    consts::{U13, U16},
};

use aes::Aes128;

use super::message::MatterMessage;
use crate::matter::error::{MatterError, MatterResult};

// ── AES-128-CCM type alias ────────────────────────────────────────────────────

/// AES-128-CCM with a 13-byte nonce and 16-byte authentication tag.
type Aes128Ccm = Ccm<Aes128, U16, U13>;

// ── Session keys ──────────────────────────────────────────────────────────────

/// Symmetric keys for one Matter session (one direction each).
#[derive(Clone, Zeroize)]
pub struct SessionKeys {
    /// Key used to encrypt outbound payloads.
    pub encrypt_key: [u8; 16],
    /// Key used to decrypt inbound payloads.
    pub decrypt_key: [u8; 16],
}

/// Thread-safe session-key store keyed by 16-bit session ID.
pub type SessionMap = Arc<Mutex<HashMap<u16, SessionKeys>>>;

// ── Nonce construction ────────────────────────────────────────────────────────

/// Build the 13-byte Matter AES-CCM nonce.
///
/// Layout: session_id (2 LE) ‖ message_counter (4 LE) ‖ 0x00 × 7
fn build_nonce(session_id: u16, message_counter: u32) -> [u8; 13] {
    let mut nonce = [0u8; 13];
    nonce[0..2].copy_from_slice(&session_id.to_le_bytes());
    nonce[2..6].copy_from_slice(&message_counter.to_le_bytes());
    // bytes 6..13 remain zero
    nonce
}

// ── Header bytes helper ───────────────────────────────────────────────────────

/// Return the header portion of an encoded message (everything before the
/// payload), which is used as AAD for AES-CCM.
fn header_bytes(msg: &MatterMessage) -> Vec<u8> {
    // Encode the full message and slice off the payload tail.
    let full = msg.encode();
    let payload_len = msg.payload.len();
    if full.len() > payload_len {
        full[..full.len() - payload_len].to_vec()
    } else {
        full
    }
}

// ── UdpTransport ─────────────────────────────────────────────────────────────

/// Matter UDP transport: encrypted send/receive using per-session AES-128-CCM.
///
/// The default Matter UDP port is 5540.
pub struct UdpTransport {
    socket: Arc<UdpSocket>,
    /// Per-session symmetric keys.  Sessions are added by the commissioning
    /// layer and removed when a session expires.
    pub sessions: SessionMap,
}

impl UdpTransport {
    /// Bind to `0.0.0.0:<port>`.  Pass `0` for an OS-assigned ephemeral port.
    pub async fn new(port: u16) -> MatterResult<Self> {
        Self::bind_addr(&format!("0.0.0.0:{port}")).await
    }

    /// Bind to the supplied address string (e.g. `"[::]:5540"`).
    pub async fn bind_addr(addr: &str) -> MatterResult<Self> {
        let socket = UdpSocket::bind(addr).await?;
        Ok(Self {
            socket: Arc::new(socket),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Send a [`MatterMessage`] to `peer`.
    ///
    /// If the message's `session_id` is non-zero and session keys exist for
    /// that ID, the payload is encrypted with AES-128-CCM (ciphertext + 16-byte
    /// tag) before transmission.  Session 0 is sent in the clear (commissioning
    /// exchange).
    pub async fn send(&self, msg: &MatterMessage, peer: SocketAddr) -> MatterResult<()> {
        let session_id = msg.header.session_id;

        let wire = if session_id != 0 {
            let keys_guard = self.sessions.lock().await;
            if let Some(keys) = keys_guard.get(&session_id) {
                let enc_key = keys.encrypt_key;
                drop(keys_guard); // release lock before heavy crypto

                let aad = header_bytes(msg);
                let nonce_bytes = build_nonce(session_id, msg.header.message_counter);
                let nonce = GenericArray::from_slice(&nonce_bytes);

                let cipher = Aes128Ccm::new(GenericArray::from_slice(&enc_key));
                let ciphertext = cipher
                    .encrypt(
                        nonce,
                        ccm::aead::Payload {
                            msg: &msg.payload,
                            aad: &aad,
                        },
                    )
                    .map_err(|_| MatterError::Transport("AES-CCM encrypt failed".into()))?;

                // Replace payload with ciphertext (includes 16-byte appended tag).
                let mut out_msg = msg.clone();
                out_msg.payload = ciphertext;
                out_msg.encode()
            } else {
                drop(keys_guard);
                msg.encode()
            }
        } else {
            msg.encode()
        };

        self.socket
            .send_to(&wire, peer)
            .await
            .map_err(|e| MatterError::Transport(format!("send_to failed: {e}")))?;
        Ok(())
    }

    /// Receive the next UDP datagram and decode/decrypt it.
    ///
    /// Returns the decoded [`MatterMessage`] and the sender's address.
    /// If session keys are present for the decoded session ID, the payload is
    /// decrypted; otherwise the payload is returned as-is (commissioning).
    pub async fn recv(&self) -> MatterResult<(MatterMessage, SocketAddr)> {
        let mut buf = vec![0u8; 1280]; // MTU for Matter UDP
        let (n, peer) = self
            .socket
            .recv_from(&mut buf)
            .await
            .map_err(|e| MatterError::Transport(format!("recv_from failed: {e}")))?;
        buf.truncate(n);

        // Decode the raw frame first (payload may be ciphertext at this point).
        let raw_msg = MatterMessage::decode(&buf)?;
        let session_id = raw_msg.header.session_id;

        if session_id == 0 {
            return Ok((raw_msg, peer));
        }

        let keys_guard = self.sessions.lock().await;
        if let Some(keys) = keys_guard.get(&session_id) {
            let dec_key = keys.decrypt_key;
            drop(keys_guard);

            let aad = header_bytes(&raw_msg);
            let nonce_bytes = build_nonce(session_id, raw_msg.header.message_counter);
            let nonce = GenericArray::from_slice(&nonce_bytes);

            let cipher = Aes128Ccm::new(GenericArray::from_slice(&dec_key));
            let plaintext = cipher
                .decrypt(
                    nonce,
                    ccm::aead::Payload {
                        msg: &raw_msg.payload,
                        aad: &aad,
                    },
                )
                .map_err(|_| MatterError::Transport("AES-CCM decrypt/verify failed".into()))?;

            let mut out_msg = raw_msg;
            out_msg.payload = plaintext;
            Ok((out_msg, peer))
        } else {
            drop(keys_guard);
            Ok((raw_msg, peer))
        }
    }

    /// Local address the transport is bound to.
    ///
    /// Useful in tests that bind to an ephemeral port (`127.0.0.1:0`) and
    /// need to know which port the OS assigned.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Register session keys for a given session ID.
    ///
    /// This is called by the commissioning layer once SPAKE2+ is complete and
    /// the session encryption keys have been derived.
    pub fn add_session(&self, id: u16, keys: SessionKeys) {
        let sessions = Arc::clone(&self.sessions);
        tokio::spawn(async move {
            sessions.lock().await.insert(id, keys);
        });
    }

    /// Remove and zeroize session keys for the given session ID.
    pub fn remove_session(&self, id: u16) {
        let sessions = Arc::clone(&self.sessions);
        tokio::spawn(async move {
            if let Some(mut keys) = sessions.lock().await.remove(&id) {
                keys.zeroize();
            }
        });
    }
}
