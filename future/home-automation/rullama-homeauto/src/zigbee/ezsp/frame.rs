//! EZSP frame encoding/decoding (EZSP v8, EmberZNet 7.x).
//!
//! Frame layout (before ASH wrapping):
//! ```text
//! SEQ (1B) | FC_LOW (1B) | FC_HIGH (1B) | CMD_ID_LOW (1B) | CMD_ID_HIGH (1B) | PARAMS...
//! ```
//!
//! FC (Frame Control) bits (v8):
//! - bit 0    : frame direction (0 = host→NCP, 1 = NCP→host)
//! - bit 1    : sleep mode
//! - bit 4    : network index (4 bits)
//! - bit 5-7  : padding
//! - high byte: padding / extended frame control

/// An EZSP v8 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EzspFrame {
    /// Sequence number (0–255, wraps).
    pub seq: u8,
    /// 16-bit frame control word.
    pub frame_control: u16,
    /// 16-bit command / callback ID.
    pub cmd_id: u16,
    /// Command parameters or response data.
    pub params: Vec<u8>,
}

impl EzspFrame {
    /// Create a new host→NCP command frame.
    pub fn command(seq: u8, cmd_id: u16, params: Vec<u8>) -> Self {
        Self {
            seq,
            // FC_LOW: direction=0 (host), sleep_mode=0, network_index=0
            // FC_HIGH: no overflow, no truncated
            frame_control: 0x0000,
            cmd_id,
            params,
        }
    }

    /// Encode to wire bytes (without ASH framing).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(5 + self.params.len());
        buf.push(self.seq);
        buf.push((self.frame_control & 0xFF) as u8);
        buf.push((self.frame_control >> 8) as u8);
        buf.push((self.cmd_id & 0xFF) as u8);
        buf.push((self.cmd_id >> 8) as u8);
        buf.extend_from_slice(&self.params);
        buf
    }

    /// Decode from wire bytes (without ASH framing). Returns the frame or an error.
    pub fn decode(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 5 {
            return Err("EZSP frame too short (need ≥5 bytes)");
        }
        let seq = data[0];
        let frame_control = data[1] as u16 | ((data[2] as u16) << 8);
        let cmd_id = data[3] as u16 | ((data[4] as u16) << 8);
        let params = data[5..].to_vec();
        Ok(Self {
            seq,
            frame_control,
            cmd_id,
            params,
        })
    }

    /// Return true if this is a NCP→host callback/response (direction bit set).
    pub fn is_response(&self) -> bool {
        self.frame_control & 0x0001 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ezsp_frame_encode_get_node_id() {
        // EZSP_GET_NODE_ID = 0x0027
        let frame = EzspFrame::command(1, 0x0027, vec![]);
        let encoded = frame.encode();
        assert_eq!(encoded, [0x01, 0x00, 0x00, 0x27, 0x00]);
    }

    #[test]
    fn ezsp_frame_decode_response() {
        // Response from NCP: seq=1, FC=0x0001 (response direction), cmd_id=0x0027, params=[0x00, 0x00]
        let raw = [0x01u8, 0x01, 0x00, 0x27, 0x00, 0x00, 0x00];
        let frame = EzspFrame::decode(&raw).unwrap();
        assert_eq!(frame.seq, 1);
        assert_eq!(frame.cmd_id, 0x0027);
        assert!(frame.is_response());
        assert_eq!(frame.params, vec![0x00, 0x00]);
    }

    #[test]
    fn ezsp_frame_sequence_wraps_at_255() {
        let mut seq = 255u8;
        seq = seq.wrapping_add(1);
        assert_eq!(seq, 0);
    }

    #[test]
    fn ezsp_frame_roundtrip() {
        let params = vec![0xAA, 0xBB, 0xCC];
        let frame = EzspFrame::command(42, 0x0013, params.clone());
        let encoded = frame.encode();
        let decoded = EzspFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.seq, 42);
        assert_eq!(decoded.cmd_id, 0x0013);
        assert_eq!(decoded.params, params);
    }

    #[test]
    fn ezsp_frame_too_short() {
        assert!(EzspFrame::decode(&[0x01, 0x00]).is_err());
    }
}
