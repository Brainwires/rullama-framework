/// Matter Message Layer implementation per Matter spec §4.4.
///
/// Wire format (all fields little-endian):
/// ```text
/// Flags(1) | Session ID(2) | Security Flags(1) | Message Counter(4) |
/// [Source Node ID(8) if S flag set] |
/// [Dest Node ID(8) if DSIZ=01, or Dest Group ID(2) if DSIZ=10] |
/// Payload...
/// ```
///
/// Message Flags byte:
///   - bits 2:0 = version (always 0 for Matter 1.x)
///   - bit 2    = S flag (source node ID present)
///   - bits 4:3 = DSIZ (00=none, 01=64-bit node, 10=16-bit group)
///   - bits 7:5 = reserved
///
/// Security Flags byte:
///   - bit 0   = P flag (privacy)
///   - bit 1   = C flag (control message)
///   - bit 2   = MX flag (message extensions present)
///   - bits 7:3 = session type (0=unicast, 1=group)
///
/// Session ID 0 = unencrypted commissioning session.
use crate::matter::error::{MatterError, MatterResult};

// ── Message Flags bit positions ───────────────────────────────────────────────

const FLAG_VERSION_MASK: u8 = 0x07;
const FLAG_S: u8 = 0x04; // source node ID present
const FLAG_DSIZ_MASK: u8 = 0x18; // bits 4:3
const FLAG_DSIZ_NONE: u8 = 0x00;
const FLAG_DSIZ_NODE64: u8 = 0x08; // 64-bit unicast node ID
const FLAG_DSIZ_GROUP16: u8 = 0x10; // 16-bit group ID

// ── Security Flags bit positions / shifts ─────────────────────────────────────

const SECFLAG_SESSION_TYPE_SHIFT: u8 = 3;
const SECFLAG_SESSION_TYPE_MASK: u8 = 0b1111_1000; // bits 7:3

/// Session type carried in the Security Flags field (bits 7:3).
#[derive(Debug, Clone, PartialEq)]
pub enum SessionType {
    /// Unicast session — addressed to a single peer node.
    Unicast,
    /// Group session — addressed to a multicast group.
    Group,
}

/// Destination address of a Matter message.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeAddress {
    /// 64-bit unicast node ID (DSIZ = 01).
    Node(u64),
    /// 16-bit group ID (DSIZ = 10).
    Group(u16),
}

/// Decoded Matter message header.
#[derive(Debug, Clone)]
pub struct MessageHeader {
    /// Matter protocol version (always 0 for Matter 1.x).
    pub version: u8,
    /// Session identifier (0 = unencrypted commissioning).
    pub session_id: u16,
    /// Session type (unicast or group) from Security Flags.
    pub session_type: SessionType,
    /// Source node ID, present when the S flag is set.
    pub source_node_id: Option<u64>,
    /// Destination node or group ID.
    pub dest_node_id: Option<NodeAddress>,
    /// Per-session monotonically-increasing message counter.
    pub message_counter: u32,
    /// Raw Security Flags byte.
    pub security_flags: u8,
}

/// A fully parsed Matter message.
#[derive(Debug, Clone)]
pub struct MatterMessage {
    /// Header (version, session, counter, source/dest, flags).
    pub header: MessageHeader,
    /// Payload bytes — TLV-encoded after decryption.
    pub payload: Vec<u8>,
}

impl MatterMessage {
    /// Encode the message to its on-wire representation.
    pub fn encode(&self) -> Vec<u8> {
        let h = &self.header;

        // Build Message Flags byte.
        let mut msg_flags: u8 = h.version & FLAG_VERSION_MASK;
        if h.source_node_id.is_some() {
            msg_flags |= FLAG_S;
        }
        match &h.dest_node_id {
            None => {} // DSIZ = 00
            Some(NodeAddress::Node(_)) => msg_flags |= FLAG_DSIZ_NODE64,
            Some(NodeAddress::Group(_)) => msg_flags |= FLAG_DSIZ_GROUP16,
        }

        let mut buf = Vec::with_capacity(10 + self.payload.len());
        buf.push(msg_flags);
        buf.extend_from_slice(&h.session_id.to_le_bytes());
        buf.push(h.security_flags);
        buf.extend_from_slice(&h.message_counter.to_le_bytes());

        if let Some(src) = h.source_node_id {
            buf.extend_from_slice(&src.to_le_bytes());
        }
        match &h.dest_node_id {
            None => {}
            Some(NodeAddress::Node(n)) => buf.extend_from_slice(&n.to_le_bytes()),
            Some(NodeAddress::Group(g)) => buf.extend_from_slice(&g.to_le_bytes()),
        }

        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a Matter message from its on-wire bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.len() < 8 {
            return Err(MatterError::Transport(
                "message too short: need at least 8 bytes for fixed header".into(),
            ));
        }

        let mut cursor = 0usize;

        let msg_flags = bytes[cursor];
        cursor += 1;

        let session_id = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        cursor += 2;

        let security_flags = bytes[cursor];
        cursor += 1;

        let message_counter = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        cursor += 4;

        // Source node ID (S flag).
        let source_node_id = if msg_flags & FLAG_S != 0 {
            if cursor + 8 > bytes.len() {
                return Err(MatterError::Transport(
                    "truncated message: not enough bytes for source node ID".into(),
                ));
            }
            let id = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            Some(id)
        } else {
            None
        };

        // Destination node/group ID (DSIZ).
        let dsiz = msg_flags & FLAG_DSIZ_MASK;
        let dest_node_id = match dsiz {
            FLAG_DSIZ_NONE => None,
            FLAG_DSIZ_NODE64 => {
                if cursor + 8 > bytes.len() {
                    return Err(MatterError::Transport(
                        "truncated message: not enough bytes for dest node ID".into(),
                    ));
                }
                let id = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
                cursor += 8;
                Some(NodeAddress::Node(id))
            }
            FLAG_DSIZ_GROUP16 => {
                if cursor + 2 > bytes.len() {
                    return Err(MatterError::Transport(
                        "truncated message: not enough bytes for dest group ID".into(),
                    ));
                }
                let id = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
                cursor += 2;
                Some(NodeAddress::Group(id))
            }
            _ => {
                return Err(MatterError::Transport(format!(
                    "unknown DSIZ value in message flags: {msg_flags:#04x}"
                )));
            }
        };

        // Session type from Security Flags bits 7:3.
        let session_type_bits =
            (security_flags & SECFLAG_SESSION_TYPE_MASK) >> SECFLAG_SESSION_TYPE_SHIFT;
        let session_type = if session_type_bits == 0 {
            SessionType::Unicast
        } else {
            SessionType::Group
        };

        let version = msg_flags & FLAG_VERSION_MASK;

        let header = MessageHeader {
            version,
            session_id,
            session_type,
            source_node_id,
            dest_node_id,
            message_counter,
            security_flags,
        };

        let payload = bytes[cursor..].to_vec();
        Ok(MatterMessage { header, payload })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(
        session_id: u16,
        source: Option<u64>,
        dest: Option<NodeAddress>,
        counter: u32,
        payload: Vec<u8>,
    ) -> MatterMessage {
        let session_type = SessionType::Unicast;
        let security_flags: u8 = 0x00;
        MatterMessage {
            header: MessageHeader {
                version: 0,
                session_id,
                session_type,
                source_node_id: source,
                dest_node_id: dest,
                message_counter: counter,
                security_flags,
            },
            payload,
        }
    }

    #[test]
    fn message_encode_decode_roundtrip_with_node_id() {
        let msg = make_message(
            0x0042,
            Some(0x0102_0304_0506_0708),
            Some(NodeAddress::Node(0xDEAD_BEEF_CAFE_1234)),
            0x0000_0001,
            b"hello matter".to_vec(),
        );

        let encoded = msg.encode();
        let decoded = MatterMessage::decode(&encoded).expect("decode failed");

        assert_eq!(decoded.header.session_id, 0x0042);
        assert_eq!(decoded.header.source_node_id, Some(0x0102_0304_0506_0708));
        assert_eq!(
            decoded.header.dest_node_id,
            Some(NodeAddress::Node(0xDEAD_BEEF_CAFE_1234))
        );
        assert_eq!(decoded.header.message_counter, 0x0000_0001);
        assert_eq!(decoded.payload, b"hello matter");
    }

    #[test]
    fn message_encode_decode_roundtrip_minimal() {
        // No source, no dest — bare minimum header.
        let msg = make_message(0x0000, None, None, 0x0000_FFFF, b"ping".to_vec());
        let encoded = msg.encode();
        let decoded = MatterMessage::decode(&encoded).expect("decode failed");

        assert_eq!(decoded.header.source_node_id, None);
        assert_eq!(decoded.header.dest_node_id, None);
        assert_eq!(decoded.header.message_counter, 0x0000_FFFF);
        assert_eq!(decoded.payload, b"ping");
    }

    #[test]
    fn session_id_zero_is_commissioning() {
        let msg = make_message(0, None, None, 1, vec![]);
        let encoded = msg.encode();
        let decoded = MatterMessage::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.header.session_id, 0);
    }

    #[test]
    fn message_counter_preserved() {
        let counter = 0xABCD_1234_u32;
        let msg = make_message(1, None, None, counter, vec![0xDE, 0xAD]);
        let encoded = msg.encode();
        let decoded = MatterMessage::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.header.message_counter, counter);
    }

    #[test]
    fn group_dest_encode_decode() {
        let msg = make_message(5, None, Some(NodeAddress::Group(0xFF01)), 99, vec![1, 2, 3]);
        let encoded = msg.encode();
        let decoded = MatterMessage::decode(&encoded).expect("decode failed");
        assert_eq!(
            decoded.header.dest_node_id,
            Some(NodeAddress::Group(0xFF01))
        );
        assert_eq!(decoded.payload, vec![1, 2, 3]);
    }
}
