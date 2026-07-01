//! Media track and DataChannel handle types.
//!
//! These are thin wrappers around `webrtc`-crate internals that provide a stable
//! public API without leaking `webrtc` types across module boundaries.

use std::sync::Arc;

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use webrtc::media_stream::track_local::static_sample::TrackLocalStaticSample;
use webrtc::media_stream::track_remote::TrackRemote;
pub use webrtc::media_stream::track_remote::TrackRemoteEvent;

// ── TrackId ───────────────────────────────────────────────────────────────────

/// A unique string identifier for a media track.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct TrackId(pub String);

impl TrackId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn new_random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TrackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── TrackDirection ────────────────────────────────────────────────────────────

/// The direction of a media track relative to the local peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum TrackDirection {
    /// Local peer sends, remote peer receives.
    SendOnly,
    /// Local peer receives, remote peer sends.
    RecvOnly,
    /// Both peers send and receive.
    SendRecv,
    /// Track is negotiated but inactive.
    Inactive,
}

// ── DataChannel ───────────────────────────────────────────────────────────────

/// A message received or sent on a WebRTC DataChannel.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub enum DataChannelMessage {
    /// A UTF-8 text message.
    Text(String),
    /// A raw binary message.
    Binary(Vec<u8>),
}

/// Configuration for creating a new DataChannel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChannelConfig {
    /// Human-readable label for the channel.
    pub label: String,
    /// Whether messages are delivered in order (default: `true`).
    pub ordered: bool,
    /// Maximum number of retransmissions for unreliable mode (`None` = reliable).
    pub max_retransmits: Option<u16>,
    /// Sub-protocol string (e.g. `"json"`, `"binary-framing"`).
    pub protocol: Option<String>,
}

impl Default for DataChannelConfig {
    fn default() -> Self {
        Self {
            label: "data".to_string(),
            ordered: true,
            max_retransmits: None,
            protocol: None,
        }
    }
}

/// A handle to an open WebRTC DataChannel.
///
/// Receive inbound messages with [`receive`](DataChannel::receive).
/// Send outbound messages with [`send`](DataChannel::send) / [`send_text`](DataChannel::send_text).
pub struct DataChannel {
    /// The numeric DataChannel ID assigned during negotiation.
    pub id: u16,
    /// The channel label.
    pub label: String,
    /// The underlying `webrtc` DataChannel.
    inner: Arc<dyn webrtc::data_channel::DataChannel>,
}

impl DataChannel {
    pub(crate) async fn new(inner: Arc<dyn webrtc::data_channel::DataChannel>) -> Result<Self> {
        let label = inner.label().await.map_err(|e| anyhow::anyhow!("{e}"))?;
        let id = inner.id();
        Ok(Self { id, label, inner })
    }

    /// Receive the next inbound message.  Returns `None` when the channel closes.
    pub async fn receive(&self) -> Option<DataChannelMessage> {
        use webrtc::data_channel::DataChannelEvent;
        loop {
            match self.inner.poll().await {
                Some(DataChannelEvent::OnMessage(msg)) => {
                    return Some(if msg.is_string {
                        DataChannelMessage::Text(String::from_utf8_lossy(&msg.data).into_owned())
                    } else {
                        DataChannelMessage::Binary(msg.data.to_vec())
                    });
                }
                Some(DataChannelEvent::OnClose) | None => return None,
                _ => continue,
            }
        }
    }

    /// Send a text message on this DataChannel.
    pub async fn send_text(&self, text: &str) -> Result<()> {
        self.inner
            .send_text(text)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Send a binary message on this DataChannel.
    pub async fn send_bytes(&self, data: &[u8]) -> Result<()> {
        use bytes::BytesMut;
        self.inner
            .send(BytesMut::from(data))
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Send a [`DataChannelMessage`] (routes to the appropriate underlying call).
    pub async fn send(&self, msg: &DataChannelMessage) -> Result<()> {
        match msg {
            DataChannelMessage::Text(s) => self.send_text(s).await,
            DataChannelMessage::Binary(b) => self.send_bytes(b).await,
        }
    }

    /// Close the DataChannel.
    pub async fn close(&self) -> Result<()> {
        self.inner.close().await.map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// ── Audio/Video tracks ────────────────────────────────────────────────────────

/// A local audio track that can be added to a [`WebRtcSession`](super::session::WebRtcSession).
///
/// Write audio frames with [`write_sample`](AudioTrack::write_sample).
pub struct AudioTrack {
    pub id: TrackId,
    pub direction: TrackDirection,
    /// The SSRC assigned at track-construction time; required by `write_sample`.
    pub(crate) ssrc: u32,
    pub(crate) inner: Arc<TrackLocalStaticSample>,
}

impl AudioTrack {
    /// Write an encoded audio sample (e.g. an Opus frame).
    pub async fn write_sample(&self, sample: &rtc::media::Sample) -> Result<()> {
        self.inner
            .write_sample(self.ssrc, sample, &[])
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// A local video track that can be added to a [`WebRtcSession`](super::session::WebRtcSession).
///
/// Write video frames with [`write_sample`](VideoTrack::write_sample).
pub struct VideoTrack {
    pub id: TrackId,
    pub direction: TrackDirection,
    /// The SSRC assigned at track-construction time; required by `write_sample`.
    pub(crate) ssrc: u32,
    pub(crate) inner: Arc<TrackLocalStaticSample>,
}

impl VideoTrack {
    /// Write an encoded video frame (e.g. a VP8 or H.264 NAL unit).
    pub async fn write_sample(&self, sample: &rtc::media::Sample) -> Result<()> {
        self.inner
            .write_sample(self.ssrc, sample, &[])
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// ── RemoteTrack ───────────────────────────────────────────────────────────────

/// A handle to an incoming remote media track from a WebRTC PeerConnection.
///
/// Use [`poll`](RemoteTrack::poll) to receive RTP packets and lifecycle events.
/// The track is available via `WebRtcSession::get_remote_track` after a
/// `ChannelEvent::TrackAdded` is received (see the `channels::events`
/// module for the event type and `channels::webrtc::session` for the
/// session type).
pub struct RemoteTrack {
    /// Unique identifier for this track.
    pub id: TrackId,
    /// Media kind: `"audio"` or `"video"`.
    pub kind: String,
    /// Negotiated codec MIME type (e.g. `"audio/opus"`, `"video/VP8"`), if known.
    pub codec: Option<String>,
    pub(crate) inner: Arc<dyn TrackRemote>,
}

impl RemoteTrack {
    pub(crate) fn new(
        id: TrackId,
        kind: String,
        codec: Option<String>,
        inner: Arc<dyn TrackRemote>,
    ) -> Self {
        Self {
            id,
            kind,
            codec,
            inner,
        }
    }

    /// Poll for the next event from this remote track.
    ///
    /// Returns `None` when the track has ended or an unrecoverable error occurred.
    /// Yields [`TrackRemoteEvent::OnRtpPacket`] for each incoming RTP packet,
    /// and [`TrackRemoteEvent::OnEnded`] when the remote side stops the track.
    pub async fn poll(&self) -> Option<TrackRemoteEvent> {
        self.inner.poll().await
    }
}

/// A type-erased handle to either an [`AudioTrack`] or a [`VideoTrack`].
pub enum MediaTrack {
    Audio(AudioTrack),
    Video(VideoTrack),
}

impl MediaTrack {
    pub fn id(&self) -> &TrackId {
        match self {
            MediaTrack::Audio(t) => &t.id,
            MediaTrack::Video(t) => &t.id,
        }
    }

    pub fn direction(&self) -> TrackDirection {
        match self {
            MediaTrack::Audio(t) => t.direction,
            MediaTrack::Video(t) => t.direction,
        }
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, MediaTrack::Audio(_))
    }

    pub fn is_video(&self) -> bool {
        matches!(self, MediaTrack::Video(_))
    }
}
