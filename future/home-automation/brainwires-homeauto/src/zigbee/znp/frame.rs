//! ZNP (Zigbee Network Processor) frame encoding/decoding for TI Z-Stack 3.x.
//!
//! Frame layout:
//! ```text
//! SOF (0xFE) | LEN (1B) | TYPE_SUBSYSTEM (1B) | CMD (1B) | PAYLOAD (LEN bytes) | FCS (1B)
//! ```
//!
//! TYPE byte encodes both the message type (upper nibble) and subsystem (lower nibble):
//! - SREQ = 0x20  (synchronous request, host→NCP)
//! - SRSP = 0x60  (synchronous response, NCP→host)
//! - AREQ = 0x41  (asynchronous event, NCP→host)
//!
//! FCS = XOR of all bytes from LEN through the last payload byte.

/// Start-of-Frame byte — every ZNP UART frame begins with `0xFE`.
pub const SOF: u8 = 0xFE;

/// ZNP message type: synchronous request, host→NCP (`0x20`).
pub const TYPE_SREQ: u8 = 0x20;
/// ZNP message type: synchronous response, NCP→host (`0x60`).
pub const TYPE_SRSP: u8 = 0x60;
/// ZNP message type: asynchronous callback, NCP→host (`0x40`).
pub const TYPE_AREQ: u8 = 0x40;

/// A decoded ZNP frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZnpFrame {
    /// Message type: SREQ, SRSP, or AREQ.
    pub msg_type: u8,
    /// Subsystem ID (lower 5 bits of the TYPE_SUBSYSTEM byte).
    pub subsystem: u8,
    /// Command byte.
    pub cmd: u8,
    /// Payload bytes.
    pub payload: Vec<u8>,
}

impl ZnpFrame {
    /// Create a new SREQ frame.
    pub fn sreq(subsystem: u8, cmd: u8, payload: Vec<u8>) -> Self {
        Self {
            msg_type: TYPE_SREQ,
            subsystem,
            cmd,
            payload,
        }
    }

    /// Encode to wire bytes (SOF | LEN | type_sub | cmd | payload | FCS).
    pub fn encode(&self) -> Vec<u8> {
        let len = self.payload.len() as u8;
        let type_sub = self.msg_type | (self.subsystem & 0x1F);
        let mut buf = Vec::with_capacity(5 + self.payload.len());
        buf.push(SOF);
        buf.push(len);
        buf.push(type_sub);
        buf.push(self.cmd);
        buf.extend_from_slice(&self.payload);
        buf.push(fcs(&buf[1..])); // FCS over LEN..last payload byte
        buf
    }

    /// Decode from wire bytes (with SOF byte included or not).
    /// Returns the frame and the number of bytes consumed, or an error.
    pub fn decode(data: &[u8]) -> Result<(Self, usize), &'static str> {
        let data = if data.first() == Some(&SOF) {
            &data[1..]
        } else {
            data
        };
        if data.len() < 4 {
            return Err("ZNP frame too short");
        }
        let len = data[0] as usize;
        if data.len() < 4 + len {
            return Err("ZNP frame incomplete");
        }
        let type_sub = data[1];
        let msg_type = type_sub & 0xE0;
        let subsystem = type_sub & 0x1F;
        let cmd = data[2];
        let payload = data[3..3 + len].to_vec();
        let received_fcs = data[3 + len];
        let computed_fcs = fcs(&data[0..3 + len]);
        if received_fcs != computed_fcs {
            return Err("ZNP FCS mismatch");
        }
        let consumed = 1 + 4 + len; // SOF + header + payload + FCS
        Ok((
            Self {
                msg_type,
                subsystem,
                cmd,
                payload,
            },
            consumed,
        ))
    }
}

/// Compute ZNP FCS: XOR of all bytes in `data`.
pub fn fcs(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |acc, &b| acc ^ b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn znp_fcs_xor_correctness() {
        // XOR of [0x02, 0x21, 0x08, 0xAA, 0xBB] = 0x02^0x21^0x08^0xAA^0xBB
        let data = [0x02u8, 0x21, 0x08, 0xAA, 0xBB];
        let expected = 0x02u8 ^ 0x21 ^ 0x08 ^ 0xAA ^ 0xBB;
        assert_eq!(fcs(&data), expected);
    }

    #[test]
    fn znp_fcs_single_byte() {
        assert_eq!(fcs(&[0x42]), 0x42);
    }

    #[test]
    fn znp_fcs_empty() {
        assert_eq!(fcs(&[]), 0x00);
    }

    #[test]
    fn znp_sreq_frame_encode() {
        // SYS_PING (subsystem=0x21, cmd=0x01, no payload)
        let frame = ZnpFrame::sreq(0x21, 0x01, vec![]);
        let encoded = frame.encode();
        // SOF | LEN=0 | TYPE_SUB=0x20|0x21=0x21 (but wait, 0x20 | 0x01 = 0x21) ...
        // Actually subsystem 0x21 = SYS: type_sub = 0x20 | (0x21 & 0x1F) = 0x20 | 0x01 = 0x21
        assert_eq!(encoded[0], SOF);
        assert_eq!(encoded[1], 0); // LEN
        assert_eq!(encoded[2], TYPE_SREQ | (0x21 & 0x1F)); // type_sub
        assert_eq!(encoded[3], 0x01); // cmd
        // FCS = XOR(0, type_sub, 0x01) = type_sub ^ 0x01
        assert_eq!(encoded[4], fcs(&encoded[1..4]));
    }

    #[test]
    fn znp_areq_frame_decode() {
        // Construct a valid AREQ: SOF | LEN=2 | 0x45 (AREQ | ZDO=0x05) | cmd=0xFF | payload=[0x01,0x02] | FCS
        let subsystem_zdo: u8 = 0x05;
        let mut raw = vec![SOF, 2, TYPE_AREQ | subsystem_zdo, 0xFF, 0x01, 0x02];
        raw.push(fcs(&raw[1..]));
        let (frame, consumed) = ZnpFrame::decode(&raw).unwrap();
        assert_eq!(frame.msg_type, TYPE_AREQ);
        assert_eq!(frame.subsystem, subsystem_zdo);
        assert_eq!(frame.cmd, 0xFF);
        assert_eq!(frame.payload, vec![0x01, 0x02]);
        assert_eq!(consumed, raw.len());
    }

    #[test]
    fn znp_frame_roundtrip() {
        let frame = ZnpFrame::sreq(0x25, 0x10, vec![0xAA, 0xBB, 0xCC]);
        let encoded = frame.encode();
        let (decoded, _) = ZnpFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, TYPE_SREQ);
        assert_eq!(decoded.subsystem, 0x05); // 0x25 & 0x1F
        assert_eq!(decoded.cmd, 0x10);
        assert_eq!(decoded.payload, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn znp_frame_fcs_mismatch() {
        let mut frame = ZnpFrame::sreq(0x21, 0x01, vec![]).encode();
        // Corrupt the FCS byte
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        assert!(ZnpFrame::decode(&frame).is_err());
    }
}
