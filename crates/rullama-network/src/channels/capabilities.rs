//! Channel capability flags.
//!
//! Uses `bitflags` to define what features a particular messaging channel supports.

use bitflags::bitflags;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

bitflags! {
    /// Flags describing the capabilities of a messaging channel.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ChannelCapabilities: u32 {
        /// Channel supports rich text (markdown, formatting).
        const RICH_TEXT         = 1 << 0;
        /// Channel supports uploading media/files.
        const MEDIA_UPLOAD      = 1 << 1;
        /// Channel supports threaded conversations.
        const THREADS           = 1 << 2;
        /// Channel supports message reactions (emoji).
        const REACTIONS         = 1 << 3;
        /// Channel supports typing indicators.
        const TYPING_INDICATOR  = 1 << 4;
        /// Channel supports editing sent messages.
        const EDIT_MESSAGES     = 1 << 5;
        /// Channel supports deleting messages.
        const DELETE_MESSAGES   = 1 << 6;
        /// Channel supports voice communication.
        const VOICE             = 1 << 7;
        /// Channel supports video communication.
        const VIDEO             = 1 << 8;
        /// Channel supports read receipts.
        const READ_RECEIPTS     = 1 << 9;
        /// Channel supports @mentions.
        const MENTIONS          = 1 << 10;
        /// Channel supports rich embeds/cards.
        const EMBEDS            = 1 << 11;
        /// Channel supports WebRTC DataChannels (arbitrary binary/text streams).
        const DATA_CHANNELS     = 1 << 12;
        /// Channel uses DTLS-SRTP media encryption (always true for WebRTC sessions).
        const ENCRYPTED_MEDIA   = 1 << 13;
    }
}

impl Serialize for ChannelCapabilities {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ChannelCapabilities {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits = u32::deserialize(deserializer)?;
        ChannelCapabilities::from_bits(bits)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid capability bits: {bits}")))
    }
}

impl JsonSchema for ChannelCapabilities {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ChannelCapabilities")
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        u32::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_combine() {
        let caps = ChannelCapabilities::RICH_TEXT | ChannelCapabilities::THREADS;
        assert!(caps.contains(ChannelCapabilities::RICH_TEXT));
        assert!(caps.contains(ChannelCapabilities::THREADS));
        assert!(!caps.contains(ChannelCapabilities::VOICE));
    }

    #[test]
    fn capabilities_serde_roundtrip() {
        let caps = ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::REACTIONS;
        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: ChannelCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, caps);
    }
}
