#![deny(missing_docs)]
//! # Brainwires Channels
//!
//! Universal messaging channel contract for the Brainwires Agent Framework.
//!
//! This crate defines the traits and types that every messaging channel adapter
//! (Discord, Telegram, Slack, etc.) must implement. It is used by the gateway
//! daemon and all channel adapters to ensure a consistent interface.

/// Channel capability flags.
pub mod capabilities;
/// Conversion between `ChannelMessage` and agent-network `MessageEnvelope`.
#[cfg(feature = "channels")]
pub mod conversion;
/// Channel events (message received, edited, deleted, reactions, etc.).
pub mod events;
/// Gateway handshake protocol types.
pub mod handshake;
/// User and session identity types.
pub mod identity;
/// Core message types for channel communication.
pub mod message;
/// The `Channel` trait that all adapters must implement.
pub mod traits;
/// WebRTC real-time media extension (voice, video, DataChannels).
#[cfg(feature = "channels-webrtc")]
#[allow(missing_docs)]
pub mod webrtc;

// Re-export core types at crate root
pub use capabilities::ChannelCapabilities;
pub use events::{ChannelEvent, PresenceStatus};
pub use handshake::{ChannelHandshake, ChannelHandshakeResponse};
pub use identity::{ChannelSession, ChannelUser, ConversationId};
pub use message::{
    Attachment, ChannelMessage, EmbedField, EmbedPayload, MediaPayload, MediaType, MessageContent,
    MessageId, ThreadId,
};
pub use traits::Channel;

#[cfg(feature = "channels-webrtc")]
pub use webrtc::{
    // Config
    AudioCodec,
    // Tracks & DataChannels
    AudioTrack,
    BandwidthConstraints,
    // Signaling
    BroadcastSignaling,
    ChannelMessageSignaling,
    CodecPreferences,
    DataChannel,
    DataChannelConfig,
    DataChannelMessage,
    DtlsRole,
    // Session & Stats
    IceConnectionState,
    IceServer,
    IceTransportPolicy,
    MediaTrack,
    PeerConnectionState,
    RTCStatsReport,
    RemoteTrack,
    SIGNALING_METADATA_KEY,
    SdpType,
    SignalingMessage,
    SignalingState,
    StatsSelector,
    TrackDirection,
    TrackId,
    TrackRemoteEvent,
    VideoCodec,
    VideoTrack,
    // Trait
    WebRtcChannel,
    WebRtcConfig,
    WebRtcSession,
    WebRtcSessionId,
    WebRtcSignaling,
};
