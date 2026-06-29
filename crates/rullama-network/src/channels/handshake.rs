//! Gateway handshake protocol types.
//!
//! When a channel adapter connects to the gateway daemon, it performs a
//! handshake to advertise its type, version, capabilities, and credentials.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::capabilities::ChannelCapabilities;

/// A handshake request sent by a channel adapter to the gateway.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelHandshake {
    /// The type of channel (e.g., "discord", "telegram", "slack").
    pub channel_type: String,
    /// The version of the channel adapter.
    pub channel_version: String,
    /// The capabilities this channel supports.
    pub capabilities: ChannelCapabilities,
    /// An authentication token for the gateway.
    pub auth_token: String,
}

/// The gateway's response to a channel handshake.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelHandshakeResponse {
    /// Whether the handshake was accepted.
    pub accepted: bool,
    /// The assigned channel ID, if accepted.
    pub channel_id: Option<Uuid>,
    /// An error message, if the handshake was rejected.
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::capabilities::ChannelCapabilities;

    #[test]
    fn handshake_serde_roundtrip() {
        let hs = ChannelHandshake {
            channel_type: "discord".to_string(),
            channel_version: "1.0.0".to_string(),
            capabilities: ChannelCapabilities::RICH_TEXT | ChannelCapabilities::EMBEDS,
            auth_token: "secret-token".to_string(),
        };
        let json = serde_json::to_string(&hs).unwrap();
        let deserialized: ChannelHandshake = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.channel_type, "discord");
        assert!(
            deserialized
                .capabilities
                .contains(ChannelCapabilities::RICH_TEXT)
        );
    }

    #[test]
    fn handshake_response_serde_roundtrip() {
        let resp = ChannelHandshakeResponse {
            accepted: true,
            channel_id: Some(Uuid::new_v4()),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ChannelHandshakeResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.accepted);
        assert!(deserialized.channel_id.is_some());
    }
}
