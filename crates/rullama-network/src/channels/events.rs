//! Channel events representing things that happen on a messaging platform.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::identity::{ChannelUser, ConversationId};
use super::message::{ChannelMessage, MessageId, ThreadId};

#[cfg(feature = "channels-webrtc")]
use super::webrtc::{
    session::{IceConnectionState, PeerConnectionState, SignalingState, WebRtcSessionId},
    track::{DataChannelMessage, TrackDirection, TrackId},
};

/// An event from a messaging channel.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub enum ChannelEvent {
    /// A new message was received.
    MessageReceived(ChannelMessage),
    /// An existing message was edited.
    MessageEdited(ChannelMessage),
    /// A message was deleted.
    MessageDeleted {
        /// The ID of the deleted message.
        message_id: MessageId,
        /// The conversation the message was in.
        conversation: ConversationId,
    },
    /// A reaction was added to a message.
    ReactionAdded {
        /// The message that was reacted to.
        message_id: MessageId,
        /// The user who added the reaction.
        user: ChannelUser,
        /// The emoji used for the reaction.
        emoji: String,
    },
    /// A reaction was removed from a message.
    ReactionRemoved {
        /// The message the reaction was removed from.
        message_id: MessageId,
        /// The user who removed the reaction.
        user: ChannelUser,
        /// The emoji that was removed.
        emoji: String,
    },
    /// A user started typing in a conversation.
    TypingStarted {
        /// The conversation where typing is occurring.
        conversation: ConversationId,
        /// The user who started typing.
        user: ChannelUser,
    },
    /// A user's presence status changed.
    PresenceChanged {
        /// The user whose presence changed.
        user: ChannelUser,
        /// The new presence status.
        status: PresenceStatus,
    },
    /// A new thread was created from a message.
    ThreadCreated {
        /// The parent message that spawned the thread.
        parent_message_id: MessageId,
        /// The ID of the newly created thread.
        thread_id: ThreadId,
    },

    // ── WebRTC signaling & media events (requires the `webrtc` feature) ──────
    /// A local ICE candidate was gathered or a remote candidate was received.
    #[cfg(feature = "channels-webrtc")]
    IceCandidate {
        /// The WebRTC session this candidate belongs to.
        session_id: WebRtcSessionId,
        /// Serialized ICE candidate (RFC 5245 candidate-attribute string).
        candidate: String,
        /// SDP media line identifier (e.g. "audio", "video", "data").
        sdp_mid: Option<String>,
        /// SDP media line index.
        sdp_mline_index: Option<u16>,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// A remote peer sent an SDP offer for a new WebRTC session.
    #[cfg(feature = "channels-webrtc")]
    SdpOffer {
        /// The WebRTC session this offer initiates.
        session_id: WebRtcSessionId,
        /// Full SDP offer body.
        sdp: String,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// A remote peer replied with an SDP answer.
    #[cfg(feature = "channels-webrtc")]
    SdpAnswer {
        /// The WebRTC session this answer completes.
        session_id: WebRtcSessionId,
        /// Full SDP answer body.
        sdp: String,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// A remote media track was added to the PeerConnection.
    #[cfg(feature = "channels-webrtc")]
    TrackAdded {
        /// The WebRTC session this track belongs to.
        session_id: WebRtcSessionId,
        /// Unique identifier for the track.
        track_id: TrackId,
        /// "audio" or "video".
        kind: String,
        /// Negotiated codec MIME type (e.g. "audio/opus", "video/VP8").
        codec: Option<String>,
        /// The send/receive direction of the track.
        direction: TrackDirection,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// A remote media track was removed or ended.
    #[cfg(feature = "channels-webrtc")]
    TrackRemoved {
        /// The WebRTC session this track belongs to.
        session_id: WebRtcSessionId,
        /// Unique identifier for the removed track.
        track_id: TrackId,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// A message arrived on a WebRTC DataChannel.
    #[cfg(feature = "channels-webrtc")]
    WebRtcDataChannel {
        /// The WebRTC session this channel belongs to.
        session_id: WebRtcSessionId,
        /// The DataChannel label.
        channel_label: String,
        /// The received message payload.
        message: DataChannelMessage,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// The PeerConnection state changed (e.g. Connected, Disconnected, Failed).
    #[cfg(feature = "channels-webrtc")]
    PeerConnectionStateChanged {
        /// The WebRTC session whose state changed.
        session_id: WebRtcSessionId,
        /// The new PeerConnection state.
        state: PeerConnectionState,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// The ICE connection state changed (e.g. Checking, Connected, Failed).
    #[cfg(feature = "channels-webrtc")]
    IceConnectionStateChanged {
        /// The WebRTC session whose ICE state changed.
        session_id: WebRtcSessionId,
        /// The new ICE connection state.
        state: IceConnectionState,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// ICE candidate gathering has completed; no further candidates will be gathered.
    #[cfg(feature = "channels-webrtc")]
    IceGatheringComplete {
        /// The WebRTC session that finished ICE gathering.
        session_id: WebRtcSessionId,
        /// The conversation context.
        conversation: ConversationId,
    },

    /// The SDP signaling state changed.
    #[cfg(feature = "channels-webrtc")]
    SignalingStateChanged {
        /// The WebRTC session whose signaling state changed.
        session_id: WebRtcSessionId,
        /// The new signaling state.
        state: SignalingState,
        /// The conversation context.
        conversation: ConversationId,
    },
}

/// A user's presence status on the platform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PresenceStatus {
    /// User is online and active.
    Online,
    /// User is away or idle.
    Away,
    /// User has enabled do-not-disturb mode.
    DoNotDisturb,
    /// User is offline.
    Offline,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::identity::ConversationId;
    use crate::channels::message::MessageId;

    #[test]
    fn channel_event_serde_roundtrip() {
        let event = ChannelEvent::MessageDeleted {
            message_id: MessageId::new("msg-123"),
            conversation: ConversationId {
                platform: "discord".to_string(),
                channel_id: "general".to_string(),
                server_id: None,
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: ChannelEvent = serde_json::from_str(&json).unwrap();
        match deserialized {
            ChannelEvent::MessageDeleted {
                message_id,
                conversation,
            } => {
                assert_eq!(message_id, MessageId::new("msg-123"));
                assert_eq!(conversation.channel_id, "general");
            }
            _ => panic!("expected MessageDeleted variant"),
        }
    }

    #[test]
    fn presence_status_serde_roundtrip() {
        let status = PresenceStatus::DoNotDisturb;
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: PresenceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }
}
