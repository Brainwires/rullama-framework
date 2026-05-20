//! WebRTC real-time media extension for channel adapters.
//!
//! Provides:
//! - [`WebRtcChannel`] — extension trait for adapters that support real-time media
//! - [`WebRtcSession`] — manages a single `RTCPeerConnection` (offer/answer, ICE, tracks)
//! - [`WebRtcConfig`] — serde-serializable peer connection configuration
//! - [`WebRtcSignaling`] — abstraction over the SDP/ICE exchange channel
//! - [`AudioTrack`] / [`VideoTrack`] / [`DataChannel`] — media handles
//!
//! # Feature flag
//!
//! This entire module requires the `webrtc` Cargo feature:
//! ```toml
//! brainwires-channels = { ..., features = ["webrtc"] }
//! ```
//!
//! # Quick start
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use crate::channels::webrtc::{
//!     WebRtcChannel, WebRtcConfig, WebRtcSession, SdpType,
//!     AudioCodec, BroadcastSignaling, WebRtcSignaling, SignalingMessage,
//! };
//!
//! // 1. Create session
//! let session = Arc::new(WebRtcSession::new(WebRtcConfig::default(), conv.clone()).await?);
//! session.open().await?;
//!
//! // 2. Add tracks before creating the offer
//! let audio = session.add_audio_track(AudioCodec::Opus).await?;
//!
//! // 3. Create offer and send via signaling
//! let sdp = session.create_offer().await?;
//! signaling.send_signaling(&conv, SignalingMessage::Offer { session_id: session.id.clone(), sdp }).await?;
//!
//! // 4. Apply remote answer
//! session.set_remote_description(answer_sdp, SdpType::Answer).await?;
//!
//! // 5. Write audio frames
//! audio.write_sample(&sample).await?;
//! ```

pub mod config;
pub mod session;
pub mod signaling;
pub mod track;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use config::{
    AudioCodec, BandwidthConstraints, CodecPreferences, DtlsRole, IceServer, IceTransportPolicy,
    VideoCodec, WebRtcConfig,
};
pub use rtc::statistics::StatsSelector;
pub use rtc::statistics::report::RTCStatsReport;
pub use session::{
    IceConnectionState, PeerConnectionState, SdpType, SignalingState, WebRtcSession,
    WebRtcSessionId,
};
pub use signaling::{
    BroadcastSignaling, ChannelMessageSignaling, SIGNALING_METADATA_KEY, SignalingMessage,
    WebRtcSignaling,
};
pub use track::{
    AudioTrack, DataChannel, DataChannelConfig, DataChannelMessage, MediaTrack, RemoteTrack,
    TrackDirection, TrackId, TrackRemoteEvent, VideoTrack,
};

// ── WebRtcChannel trait ───────────────────────────────────────────────────────

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::channels::capabilities::ChannelCapabilities;
use crate::channels::identity::ConversationId;

/// Extension trait for channel adapters that support real-time WebRTC media.
///
/// Implementors also implement [`Channel`](crate::traits::Channel).
/// The framework identifies WebRTC-capable channels by checking
/// `capabilities().contains(ChannelCapabilities::VOICE | ChannelCapabilities::VIDEO)`.
///
/// # Session management
///
/// Adapters are responsible for storing sessions (e.g. in an
/// `Arc<RwLock<HashMap<WebRtcSessionId, Arc<WebRtcSession>>>>`).
///
/// # Signaling
///
/// Return a [`WebRtcSignaling`] impl from [`signaling`](Self::signaling), or `None`
/// if the adapter handles signaling internally (e.g. encodes SDP into regular messages
/// and drives the state machine itself).
#[async_trait]
pub trait WebRtcChannel: crate::channels::Channel {
    /// Create a new [`WebRtcSession`] for the given conversation.
    ///
    /// The returned session is fully initialized but not yet connected.
    /// Call `session.open()` and then `session.create_offer()` for the
    /// initiating side, or wait for a `ChannelEvent::SdpOffer` and call
    /// `session.set_remote_description()` for the answering side.
    async fn initiate_session(
        &self,
        target: &ConversationId,
        config: WebRtcConfig,
    ) -> Result<Arc<WebRtcSession>>;

    /// Look up an existing session by ID.
    async fn get_session(&self, id: &WebRtcSessionId) -> Result<Option<Arc<WebRtcSession>>>;

    /// Close and remove a session.
    async fn close_session(&self, id: &WebRtcSessionId) -> Result<()>;

    /// The signaling mechanism this adapter uses to exchange SDP and ICE candidates.
    ///
    /// Return `None` if the adapter drives signaling internally.
    fn signaling(&self) -> Option<&dyn WebRtcSignaling>;

    // ── Default capability queries ─────────────────────────────────────────

    /// Whether this adapter supports voice calls (VOICE capability is declared).
    fn supports_voice(&self) -> bool {
        self.capabilities().contains(ChannelCapabilities::VOICE)
    }

    /// Whether this adapter supports video calls (VIDEO capability is declared).
    fn supports_video(&self) -> bool {
        self.capabilities().contains(ChannelCapabilities::VIDEO)
    }

    /// Whether this adapter supports WebRTC DataChannels.
    ///
    /// Defaults to `true` for all `WebRtcChannel` implementors because DataChannels
    /// are part of the core WebRTC spec and require no additional negotiation.
    fn supports_data_channels(&self) -> bool {
        true
    }
}
