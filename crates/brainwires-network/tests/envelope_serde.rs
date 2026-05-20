//! Integration tests for message envelope serialization across payload types.
//!
//! These tests verify that envelopes with different payload types serialize
//! and deserialize correctly through the full JSON round-trip, and that
//! cross-payload envelope interactions (reply chains, correlation IDs)
//! work as expected.

use brainwires_network::{MessageEnvelope, MessageTarget, Payload};
use uuid::Uuid;

/// Verify that all three payload variants survive a JSON round-trip
/// when used in different envelope types (direct, broadcast, topic).
#[test]
fn all_payload_variants_roundtrip_across_envelope_types() {
    let sender = Uuid::new_v4();
    let recipient = Uuid::new_v4();

    let envelopes = vec![
        // Direct + JSON payload
        MessageEnvelope::direct(
            sender,
            recipient,
            serde_json::json!({"action": "code-review", "file": "main.rs"}),
        ),
        // Broadcast + Text payload
        MessageEnvelope::broadcast(sender, "heartbeat-ping"),
        // Topic + Binary payload
        MessageEnvelope::topic(sender, "file-updates", vec![0xCA, 0xFE, 0xBA, 0xBE]),
    ];

    for original in &envelopes {
        let json = serde_json::to_string(original).expect("serialization should succeed");
        let restored: MessageEnvelope =
            serde_json::from_str(&json).expect("deserialization should succeed");

        assert_eq!(restored.id, original.id);
        assert_eq!(restored.sender, original.sender);
        assert_eq!(restored.recipient, original.recipient);
        assert_eq!(restored.ttl, original.ttl);
        assert_eq!(restored.correlation_id, original.correlation_id);
    }
}

/// Verify that a multi-hop reply chain preserves correlation IDs correctly.
#[test]
fn reply_chain_preserves_correlation() {
    let agent_a = Uuid::new_v4();
    let agent_b = Uuid::new_v4();
    let agent_c = Uuid::new_v4();

    // A sends to B
    let msg_1 = MessageEnvelope::direct(agent_a, agent_b, "task: review PR #42");

    // B replies to A
    let msg_2 = msg_1.reply(agent_b, "review complete, 2 issues found");
    assert_eq!(msg_2.correlation_id, Some(msg_1.id));
    assert_eq!(msg_2.recipient, MessageTarget::Direct(agent_a));

    // A forwards result to C (new message, not a reply)
    let msg_3 = MessageEnvelope::direct(agent_a, agent_c, "PR #42 has 2 issues");
    assert!(msg_3.correlation_id.is_none());

    // C replies to A
    let msg_4 = msg_3.reply(agent_c, "acknowledged");
    assert_eq!(msg_4.correlation_id, Some(msg_3.id));

    // All messages should have unique IDs
    let ids = [msg_1.id, msg_2.id, msg_3.id, msg_4.id];
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 4, "all message IDs should be unique");
}

/// Verify that envelope builder methods (with_ttl, with_correlation) compose.
#[test]
fn envelope_builder_methods_compose() {
    let correlation = Uuid::new_v4();
    let envelope = MessageEnvelope::broadcast(Uuid::new_v4(), "status-update")
        .with_ttl(3)
        .with_correlation(correlation);

    assert_eq!(envelope.ttl, Some(3));
    assert_eq!(envelope.correlation_id, Some(correlation));

    // Should survive serialization
    let json = serde_json::to_string(&envelope).unwrap();
    let restored: MessageEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.ttl, Some(3));
    assert_eq!(restored.correlation_id, Some(correlation));
}

/// Verify binary payload with non-trivial data (all byte values) round-trips.
#[test]
fn binary_payload_all_byte_values() {
    let all_bytes: Vec<u8> = (0..=255).collect();
    let envelope = MessageEnvelope::direct(Uuid::new_v4(), Uuid::new_v4(), all_bytes.clone());

    let json = serde_json::to_string(&envelope).unwrap();
    let restored: MessageEnvelope = serde_json::from_str(&json).unwrap();

    match restored.payload {
        Payload::Binary(bytes) => assert_eq!(bytes, all_bytes),
        other => panic!("expected Binary payload, got {other:?}"),
    }
}

/// Verify that MessageTarget variants serialize/deserialize with correct discrimination.
#[test]
fn message_target_serde_discrimination() {
    let targets = vec![
        MessageTarget::Direct(Uuid::new_v4()),
        MessageTarget::Broadcast,
        MessageTarget::Topic("build-events".to_string()),
    ];

    for target in &targets {
        let json = serde_json::to_string(target).unwrap();
        let restored: MessageTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, target);
    }
}
