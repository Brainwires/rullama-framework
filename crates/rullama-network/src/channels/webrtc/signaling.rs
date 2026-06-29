//! Signaling abstraction for SDP and ICE candidate exchange.
//!
//! The [`WebRtcSignaling`] trait decouples how WebRTC negotiation messages travel
//! from the `WebRtcSession` itself. Adapters can implement it over any transport:
//! WebSocket, A2A JSON-RPC, or even encoded inside regular [`ChannelMessage`](crate::channels::message::ChannelMessage)s.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

use super::super::identity::ConversationId;

use super::session::WebRtcSessionId;

// ── SignalingMessage ──────────────────────────────────────────────────────────

/// A WebRTC negotiation message exchanged during session establishment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalingMessage {
    /// An SDP offer from the initiating peer.
    Offer {
        session_id: WebRtcSessionId,
        /// Full SDP offer body.
        sdp: String,
    },
    /// An SDP answer from the responding peer.
    Answer {
        session_id: WebRtcSessionId,
        /// Full SDP answer body.
        sdp: String,
    },
    /// A locally gathered ICE candidate to send to the remote peer.
    IceCandidate {
        session_id: WebRtcSessionId,
        /// Serialized ICE candidate string.
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u16>,
    },
    /// ICE gathering is complete; no further candidates will follow.
    IceGatheringComplete { session_id: WebRtcSessionId },
}

impl SignalingMessage {
    pub fn session_id(&self) -> &WebRtcSessionId {
        match self {
            SignalingMessage::Offer { session_id, .. } => session_id,
            SignalingMessage::Answer { session_id, .. } => session_id,
            SignalingMessage::IceCandidate { session_id, .. } => session_id,
            SignalingMessage::IceGatheringComplete { session_id } => session_id,
        }
    }
}

// ── WebRtcSignaling trait ─────────────────────────────────────────────────────

/// Abstraction over the channel used to exchange SDP offers/answers and ICE candidates.
///
/// # Contract
///
/// - `send_signaling` must deliver the message reliably (it is not fire-and-forget).
/// - `receive_signaling` is called in a per-conversation Tokio task loop; it blocks
///   until a message arrives and returns `None` when the channel is closed.
///
/// # Implementations shipped in this crate
///
/// - [`BroadcastSignaling`] — in-process broadcast channel; ideal for testing and
///   for gateways that act as the signaling intermediary between two sessions.
/// - [`ChannelMessageSignaling`] — encodes signaling as JSON inside a regular
///   [`ChannelMessage`](crate::channels::message::ChannelMessage)(crate::message::ChannelMessage) with a well-known metadata
///   key, allowing signaling to flow through any existing adapter without changes.
#[async_trait]
pub trait WebRtcSignaling: Send + Sync {
    /// Send a signaling message to the remote peer for the given conversation.
    async fn send_signaling(
        &self,
        target: &ConversationId,
        message: SignalingMessage,
    ) -> Result<()>;

    /// Receive the next signaling message for the given conversation.
    ///
    /// Returns `None` when the signaling channel is closed.
    async fn receive_signaling(&self, target: &ConversationId) -> Result<Option<SignalingMessage>>;
}

// ── BroadcastSignaling ────────────────────────────────────────────────────────

/// In-process signaling using `tokio::sync::broadcast`.
///
/// Suitable for:
/// - Unit and integration tests (two sessions in the same process)
/// - Gateway-mediated signaling where the gateway relays messages between sessions
///
/// # Usage
///
/// Create one shared instance and give it to both peers:
/// ```rust,ignore
/// let signaling = Arc::new(BroadcastSignaling::new(32));
/// let peer_a_session = WebRtcSession::new(config.clone(), conv.clone()).await?;
/// let peer_b_session = WebRtcSession::new(config, conv).await?;
/// // Both peers use `signaling` for offer/answer/ICE exchange.
/// ```
#[derive(Clone)]
pub struct BroadcastSignaling {
    tx: broadcast::Sender<(ConversationId, SignalingMessage)>,
}

impl BroadcastSignaling {
    /// Create a new `BroadcastSignaling` with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to the raw broadcast stream (useful for gateway relay logic).
    pub fn subscribe(&self) -> broadcast::Receiver<(ConversationId, SignalingMessage)> {
        self.tx.subscribe()
    }
}

#[async_trait]
impl WebRtcSignaling for BroadcastSignaling {
    async fn send_signaling(
        &self,
        target: &ConversationId,
        message: SignalingMessage,
    ) -> Result<()> {
        // Ignore the error if there are no active subscribers yet.
        let _ = self.tx.send((target.clone(), message));
        Ok(())
    }

    async fn receive_signaling(&self, target: &ConversationId) -> Result<Option<SignalingMessage>> {
        let mut rx = self.tx.subscribe();
        loop {
            match rx.recv().await {
                Ok((conv, msg)) if &conv == target => return Ok(Some(msg)),
                Ok(_) => continue, // message for a different conversation
                Err(broadcast::error::RecvError::Closed) => return Ok(None),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    }
}

// ── ChannelMessageSignaling ───────────────────────────────────────────────────

/// Signaling that encodes WebRTC negotiation messages as JSON inside regular
/// [`ChannelMessage`](crate::channels::message::ChannelMessage)(crate::message::ChannelMessage)s.
///
/// Messages are identified by the metadata key `"_bw_webrtc_signaling"` set to `"1"`.
/// This allows signaling to flow through any existing channel adapter (Discord, Telegram,
/// Slack, …) without any modification to the adapter.
///
/// The inbound side is driven by calling [`inject`](ChannelMessageSignaling::inject)
/// whenever a `ChannelEvent::MessageReceived` arrives that carries the marker key.
pub struct ChannelMessageSignaling {
    /// Per-conversation inbound queues.
    queues: Arc<RwLock<HashMap<String, broadcast::Sender<SignalingMessage>>>>,
}

/// Metadata key used to tag signaling messages.
pub const SIGNALING_METADATA_KEY: &str = "_bw_webrtc_signaling";

impl ChannelMessageSignaling {
    pub fn new() -> Self {
        Self {
            queues: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn conv_key(conv: &ConversationId) -> String {
        format!("{}::{}", conv.platform, conv.channel_id)
    }

    /// Inject a serialized signaling payload (from a received `ChannelMessage`).
    ///
    /// Call this from your channel adapter's event loop when you see a message
    /// whose metadata contains `"_bw_webrtc_signaling": "1"`.
    pub async fn inject(&self, conv: &ConversationId, json_payload: &str) -> Result<()> {
        let msg: SignalingMessage = serde_json::from_str(json_payload)?;
        let key = Self::conv_key(conv);
        let queues: tokio::sync::RwLockReadGuard<
            '_,
            HashMap<String, broadcast::Sender<SignalingMessage>>,
        > = self.queues.read().await;
        if let Some(tx) = queues.get(&key) {
            let _ = tx.send(msg);
        }
        Ok(())
    }
}

impl Default for ChannelMessageSignaling {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WebRtcSignaling for ChannelMessageSignaling {
    async fn send_signaling(
        &self,
        _target: &ConversationId,
        _message: SignalingMessage,
    ) -> Result<()> {
        // Outbound path: the adapter should serialize the message as JSON and include
        // it as the text body of a ChannelMessage with metadata key SIGNALING_METADATA_KEY.
        // This method intentionally returns an error — the adapter drives sending.
        Err(anyhow::anyhow!(
            "ChannelMessageSignaling outbound sending must be handled by the channel adapter; \
             serialize the SignalingMessage as JSON and send it via Channel::send_message"
        ))
    }

    async fn receive_signaling(&self, target: &ConversationId) -> Result<Option<SignalingMessage>> {
        let key = Self::conv_key(target);
        // Ensure a queue exists for this conversation.
        let tx = {
            let mut queues = self.queues.write().await;
            queues
                .entry(key)
                .or_insert_with(|| broadcast::channel(64).0)
                .clone()
        };
        let mut rx = tx.subscribe();
        match rx.recv().await {
            Ok(msg) => Ok(Some(msg)),
            Err(broadcast::error::RecvError::Closed) => Ok(None),
            Err(broadcast::error::RecvError::Lagged(_)) => Ok(None),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn test_conv() -> ConversationId {
        ConversationId {
            platform: "test".to_string(),
            channel_id: "chan-1".to_string(),
            server_id: None,
        }
    }

    fn test_session_id() -> WebRtcSessionId {
        WebRtcSessionId(Uuid::new_v4())
    }

    #[test]
    fn signaling_message_serde_roundtrip() {
        let msg = SignalingMessage::IceCandidate {
            session_id: test_session_id(),
            candidate: "candidate:1 1 UDP 2130706431 192.168.1.1 5000 typ host".to_string(),
            sdp_mid: Some("audio".to_string()),
            sdp_mline_index: Some(0),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let rt: SignalingMessage = serde_json::from_str(&json).unwrap();
        match rt {
            SignalingMessage::IceCandidate { sdp_mid, .. } => {
                assert_eq!(sdp_mid, Some("audio".to_string()));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn broadcast_signaling_loopback() {
        let sig = BroadcastSignaling::new(16);
        let conv = test_conv();
        let sid = test_session_id();

        let sig2 = sig.clone();
        let conv2 = conv.clone();
        let sid2 = sid.clone();

        // Spawn a task that listens, then send from the main task.
        let handle =
            tokio::spawn(async move { sig2.receive_signaling(&conv2).await.unwrap().unwrap() });

        // Small yield to let the subscriber task register.
        tokio::task::yield_now().await;

        sig.send_signaling(
            &conv,
            SignalingMessage::Offer {
                session_id: sid,
                sdp: "v=0\r\n...".to_string(),
            },
        )
        .await
        .unwrap();

        let received = handle.await.unwrap();
        assert_eq!(received.session_id(), &sid2);
    }
}
