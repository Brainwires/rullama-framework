use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A message envelope that wraps any payload with routing metadata.
///
/// This is the universal message format used across all transports.
/// Transports serialize/deserialize envelopes to their wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Unique message identifier.
    pub id: Uuid,
    /// The sender's agent identity UUID.
    pub sender: Uuid,
    /// Who this message is addressed to.
    pub recipient: MessageTarget,
    /// The message payload.
    pub payload: Payload,
    /// When the message was created.
    pub timestamp: DateTime<Utc>,
    /// Optional time-to-live (number of hops before the message is dropped).
    pub ttl: Option<u32>,
    /// Optional correlation ID for request-response patterns.
    pub correlation_id: Option<Uuid>,
    /// Optional trace ID for cross-system event correlation.
    ///
    /// Set this to the same UUID used by the originating `TaskAgent` so that
    /// network hops can be joined with audit log entries and A2A stream events.
    pub trace_id: Option<Uuid>,
}

impl MessageEnvelope {
    /// Create a new envelope addressed to a specific agent.
    pub fn direct(sender: Uuid, recipient: Uuid, payload: impl Into<Payload>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender,
            recipient: MessageTarget::Direct(recipient),
            payload: payload.into(),
            timestamp: Utc::now(),
            ttl: None,
            correlation_id: None,
            trace_id: None,
        }
    }

    /// Create a new broadcast envelope.
    pub fn broadcast(sender: Uuid, payload: impl Into<Payload>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender,
            recipient: MessageTarget::Broadcast,
            payload: payload.into(),
            timestamp: Utc::now(),
            ttl: None,
            correlation_id: None,
            trace_id: None,
        }
    }

    /// Create a new topic-addressed envelope.
    pub fn topic(sender: Uuid, topic: impl Into<String>, payload: impl Into<Payload>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender,
            recipient: MessageTarget::Topic(topic.into()),
            payload: payload.into(),
            timestamp: Utc::now(),
            ttl: None,
            correlation_id: None,
            trace_id: None,
        }
    }

    /// Set the TTL on this envelope.
    pub fn with_ttl(mut self, ttl: u32) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set a correlation ID for request-response tracking.
    pub fn with_correlation(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Create a reply envelope to this message.
    ///
    /// The reply inherits the sender's `trace_id` so the full
    /// request-response exchange shares one trace.
    pub fn reply(&self, sender: Uuid, payload: impl Into<Payload>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender,
            recipient: MessageTarget::Direct(self.sender),
            payload: payload.into(),
            timestamp: Utc::now(),
            ttl: None,
            correlation_id: Some(self.id),
            trace_id: self.trace_id,
        }
    }

    /// Attach a trace ID to this envelope (builder pattern).
    pub fn with_trace(mut self, trace_id: Uuid) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
}

/// The target of a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageTarget {
    /// Send to a specific agent by UUID.
    Direct(Uuid),
    /// Send to all known peers.
    Broadcast,
    /// Send to all agents subscribed to a topic.
    Topic(String),
}

/// The payload of a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Payload {
    /// Structured JSON data.
    Json(serde_json::Value),
    /// Raw binary data (base64-encoded in JSON serialization).
    #[serde(with = "base64_bytes")]
    Binary(Vec<u8>),
    /// Plain text.
    Text(String),
}

impl From<serde_json::Value> for Payload {
    fn from(v: serde_json::Value) -> Self {
        Payload::Json(v)
    }
}

impl From<String> for Payload {
    fn from(s: String) -> Self {
        Payload::Text(s)
    }
}

impl From<&str> for Payload {
    fn from(s: &str) -> Self {
        Payload::Text(s.to_string())
    }
}

impl From<Vec<u8>> for Payload {
    fn from(b: Vec<u8>) -> Self {
        Payload::Binary(b)
    }
}

/// Serde helper for base64-encoding binary payloads in JSON.
mod base64_bytes {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        STANDARD.encode(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let encoded = String::deserialize(d)?;
        STANDARD.decode(&encoded).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_envelope_fields() {
        let sender = Uuid::new_v4();
        let recipient = Uuid::new_v4();
        let env = MessageEnvelope::direct(sender, recipient, "hello");

        assert_eq!(env.sender, sender);
        assert_eq!(env.recipient, MessageTarget::Direct(recipient));
        assert!(env.ttl.is_none());
        assert!(env.correlation_id.is_none());
    }

    #[test]
    fn broadcast_envelope() {
        let sender = Uuid::new_v4();
        let env = MessageEnvelope::broadcast(sender, "ping");
        assert_eq!(env.recipient, MessageTarget::Broadcast);
    }

    #[test]
    fn topic_envelope() {
        let sender = Uuid::new_v4();
        let env = MessageEnvelope::topic(sender, "status-updates", "agent online");
        assert_eq!(env.recipient, MessageTarget::Topic("status-updates".into()));
    }

    #[test]
    fn reply_sets_correlation() {
        let sender_a = Uuid::new_v4();
        let sender_b = Uuid::new_v4();
        let original = MessageEnvelope::direct(sender_a, sender_b, "request");
        let reply = original.reply(sender_b, "response");

        assert_eq!(reply.sender, sender_b);
        assert_eq!(reply.recipient, MessageTarget::Direct(sender_a));
        assert_eq!(reply.correlation_id, Some(original.id));
    }

    #[test]
    fn with_ttl() {
        let env = MessageEnvelope::broadcast(Uuid::new_v4(), "test").with_ttl(5);
        assert_eq!(env.ttl, Some(5));
    }

    #[test]
    fn envelope_serde_roundtrip() {
        let env = MessageEnvelope::direct(Uuid::new_v4(), Uuid::new_v4(), "hello");
        let json = serde_json::to_string(&env).unwrap();
        let deserialized: MessageEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, env.id);
        assert_eq!(deserialized.sender, env.sender);
    }

    #[test]
    fn binary_payload_serde_roundtrip() {
        let env =
            MessageEnvelope::direct(Uuid::new_v4(), Uuid::new_v4(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let json = serde_json::to_string(&env).unwrap();
        let deserialized: MessageEnvelope = serde_json::from_str(&json).unwrap();
        match deserialized.payload {
            Payload::Binary(bytes) => assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]),
            _ => panic!("expected Binary payload"),
        }
    }

    #[test]
    fn json_payload_from_value() {
        let payload: Payload = serde_json::json!({"key": "value"}).into();
        match payload {
            Payload::Json(v) => assert_eq!(v["key"], "value"),
            _ => panic!("expected Json payload"),
        }
    }
}
