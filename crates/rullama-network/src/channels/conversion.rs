//! Conversion between `ChannelMessage` and agent-network `MessageEnvelope`.
//!
//! These conversions allow channel messages to be routed through the
//! agent network as standard message envelopes and vice versa.

use crate::{MessageEnvelope, Payload};
use uuid::Uuid;

use super::message::{ChannelMessage, MessageContent};

/// Convert a `ChannelMessage` into a `MessageEnvelope` for the agent network.
///
/// The channel message content is serialized as JSON into the envelope payload.
/// A new sender UUID is generated; callers should set the correct sender on the
/// returned envelope if a specific agent identity is required.
impl From<&ChannelMessage> for MessageEnvelope {
    fn from(msg: &ChannelMessage) -> Self {
        let text_content = match &msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { fallback_plain, .. } => fallback_plain.clone(),
            _ => serde_json::to_string(&msg.content).unwrap_or_default(),
        };

        // Serialize the full channel message as JSON for rich payloads
        let payload = match serde_json::to_value(msg) {
            Ok(v) => Payload::Json(v),
            Err(_) => Payload::Text(text_content),
        };

        MessageEnvelope::broadcast(Uuid::nil(), payload)
    }
}

/// Try to extract a `ChannelMessage` from a `MessageEnvelope`.
///
/// This succeeds when the envelope's payload is a JSON object that can be
/// deserialized as a `ChannelMessage`. For text payloads, this will fail.
impl TryFrom<&MessageEnvelope> for ChannelMessage {
    type Error = anyhow::Error;

    fn try_from(envelope: &MessageEnvelope) -> Result<Self, Self::Error> {
        match &envelope.payload {
            Payload::Json(v) => {
                let msg: ChannelMessage = serde_json::from_value(v.clone())?;
                Ok(msg)
            }
            Payload::Text(t) => Err(anyhow::anyhow!(
                "cannot convert text payload to ChannelMessage: {}",
                t
            )),
            Payload::Binary(_) => Err(anyhow::anyhow!(
                "cannot convert binary payload to ChannelMessage"
            )),
        }
    }
}

/// Convert an owned `ChannelMessage` into a `MessageEnvelope`.
impl From<ChannelMessage> for MessageEnvelope {
    fn from(msg: ChannelMessage) -> Self {
        MessageEnvelope::from(&msg)
    }
}

/// Try to extract a `ChannelMessage` from an owned `MessageEnvelope`.
impl TryFrom<MessageEnvelope> for ChannelMessage {
    type Error = anyhow::Error;

    fn try_from(envelope: MessageEnvelope) -> Result<Self, Self::Error> {
        ChannelMessage::try_from(&envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::super::identity::ConversationId;
    use super::super::message::{ChannelMessage, MessageContent, MessageId};
    use super::*;
    use crate::MessageTarget;
    use chrono::Utc;
    use std::collections::HashMap;

    fn sample_message() -> ChannelMessage {
        ChannelMessage {
            id: MessageId::new("msg-conv-001"),
            conversation: ConversationId {
                platform: "discord".to_string(),
                channel_id: "general".to_string(),
                server_id: Some("srv-1".to_string()),
            },
            author: "bot".to_string(),
            content: MessageContent::Text("Hello from channel".to_string()),
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn channel_message_to_envelope() {
        let msg = sample_message();
        let envelope = MessageEnvelope::from(&msg);
        assert_eq!(envelope.recipient, MessageTarget::Broadcast);
        match &envelope.payload {
            Payload::Json(_) => {} // expected
            _ => panic!("expected JSON payload"),
        }
    }

    #[test]
    fn envelope_roundtrip() {
        let msg = sample_message();
        let envelope = MessageEnvelope::from(&msg);
        let recovered = ChannelMessage::try_from(&envelope).unwrap();
        assert_eq!(recovered.id, msg.id);
        assert_eq!(recovered.author, msg.author);
    }

    #[test]
    fn text_payload_fails_conversion() {
        let envelope = MessageEnvelope::broadcast(Uuid::new_v4(), "plain text");
        let result = ChannelMessage::try_from(&envelope);
        assert!(result.is_err());
    }
}
