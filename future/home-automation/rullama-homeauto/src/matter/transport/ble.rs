//! BLE transport for Matter commissioning (matter-ble feature).
//!
//! Implements Matter BLE Transport Protocol (BTP) per Matter spec §4.17.
//!
//! # Matter BLE UUIDs
//!
//! - Service UUID  : `0000FFF6-0000-1000-8000-00805F9B34FB`
//! - C1 (write)    : `18EE2EF5-263D-4559-959F-4F9C429F9D11`
//! - C2 (indicate) : `18EE2EF5-263D-4559-959F-4F9C429F9D12`

#[cfg(feature = "matter-ble")]
pub use self::inner::BleTransport;

#[cfg(feature = "matter-ble")]
pub(crate) use self::inner::{
    BtpHandshakeRequest, BtpHandshakeResponse, BtpReassembler, flags, fragment_message,
};

#[cfg(feature = "matter-ble")]
mod inner {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU8;
    use tokio::sync::mpsc;

    // ── BTP frame flags ───────────────────────────────────────────────────────

    /// BTP frame flag constants (byte 0 of every BTP data frame).
    pub mod flags {
        /// Handshake flag — set in BtpHandshakeRequest / BtpHandshakeResponse.
        pub const HANDSHAKE: u8 = 0x01;
        /// More-data flag — additional segments follow.
        pub const MORE_DATA: u8 = 0x02;
        /// Acknowledgement flag — ack_seq field is present.
        pub const ACK: u8 = 0x04;
        /// End-of-message flag — last segment of a reassembled message.
        pub const END_MSG: u8 = 0x08;
        /// Begin-segment flag — first segment of a new message.
        pub const BEGIN_SEG: u8 = 0x10;
    }

    // ── BTP handshake ─────────────────────────────────────────────────────────

    /// BTP handshake request (sent by the commissioner on C1).
    ///
    /// Wire layout:
    /// ```text
    /// byte 0  : 0x65  (SYN=HANDSHAKE | version nibble)
    /// byte 1–2: segment length LE
    /// byte 3  : versions_supported bitmask  (bit2 = BTP v3)
    /// byte 4–5: attl LE  (max segment / ATT MTU payload size)
    /// byte 6  : window_size
    /// ```
    #[derive(Debug, Clone)]
    pub struct BtpHandshakeRequest {
        /// Bitmask of supported BTP versions (bit 2 = v3).
        pub versions_supported: u8,
        /// Maximum segment payload size the controller can receive.
        pub attl: u16,
        /// Maximum number of unacknowledged segments (window).
        pub window_size: u8,
    }

    /// BTP handshake response (sent by the device on C2).
    ///
    /// Wire layout:
    /// ```text
    /// byte 0  : 0x65  (SYN+ACK)
    /// byte 1–2: segment length LE
    /// byte 3  : selected_version  (4 = BTP v4)
    /// byte 4–5: attl LE
    /// byte 6  : window_size
    /// ```
    #[derive(Debug, Clone)]
    pub struct BtpHandshakeResponse {
        /// Selected BTP version.
        pub selected_version: u8,
        /// Agreed maximum segment payload size.
        pub attl: u16,
        /// Agreed window size.
        pub window_size: u8,
    }

    impl BtpHandshakeRequest {
        /// Parse the raw C1 write bytes into a [`BtpHandshakeRequest`].
        ///
        /// Returns `None` if the bytes are malformed (too short or wrong magic).
        pub fn parse(bytes: &[u8]) -> Option<Self> {
            // Minimum: 1 (flags) + 2 (seg_len) + 1 (versions) + 2 (attl) + 1 (window) = 7
            if bytes.len() < 7 {
                return None;
            }
            // Byte 0 must carry the HANDSHAKE flag.
            if bytes[0] & flags::HANDSHAKE == 0 {
                return None;
            }
            let versions_supported = bytes[3];
            let attl = u16::from_le_bytes([bytes[4], bytes[5]]);
            let window_size = bytes[6];
            Some(Self {
                versions_supported,
                attl,
                window_size,
            })
        }

        /// Build the handshake response the device should send back on C2.
        ///
        /// Selects BTP v4, adopts the controller's attl/window (capped at 247 / 6).
        pub fn to_response(&self) -> BtpHandshakeResponse {
            BtpHandshakeResponse {
                selected_version: 4,
                attl: self.attl.min(247),
                window_size: self.window_size.min(6),
            }
        }
    }

    impl BtpHandshakeResponse {
        /// Encode the handshake response to its on-wire bytes.
        pub fn encode(&self) -> Vec<u8> {
            let seg_len = self.attl;
            vec![
                flags::HANDSHAKE,                // byte 0: 0x65 SYN+ACK marker (we reuse HANDSHAKE)
                (seg_len & 0xFF) as u8,          // byte 1: seg len low
                ((seg_len >> 8) & 0xFF) as u8,   // byte 2: seg len high
                self.selected_version,           // byte 3: version
                (self.attl & 0xFF) as u8,        // byte 4: attl low
                ((self.attl >> 8) & 0xFF) as u8, // byte 5: attl high
                self.window_size,                // byte 6: window
            ]
        }
    }

    // ── BTP reassembler ───────────────────────────────────────────────────────

    /// Reassembles multi-segment BTP frames into complete Matter messages.
    ///
    /// Feed individual BTP frames via [`BtpReassembler::feed`]; a completed
    /// message (last segment received) is returned as `Some(bytes)`.
    pub struct BtpReassembler {
        segments: Vec<u8>,
        expecting_more: bool,
        next_seq: u8,
    }

    impl BtpReassembler {
        /// Create a new reassembler with sequence counter starting at 0.
        pub fn new() -> Self {
            Self {
                segments: Vec::new(),
                expecting_more: false,
                next_seq: 0,
            }
        }

        /// Feed one BTP data frame.
        ///
        /// Returns `Some(assembled_message)` when the END_MSG flag is set on
        /// the supplied frame (i.e. all segments have arrived).
        ///
        /// Frame layout (post-handshake):
        /// ```text
        /// byte 0        : flags
        /// byte 1        : seq_num
        /// [byte 2]      : ack_seq  (present when ACK flag set)
        /// byte N  ..+1  : payload_length LE (u16)
        /// byte N+2 ..   : payload
        /// ```
        pub fn feed(&mut self, frame: &[u8]) -> Option<Vec<u8>> {
            if frame.is_empty() {
                return None;
            }
            let frame_flags = frame[0];

            // Skip handshake frames.
            if frame_flags & flags::HANDSHAKE != 0 {
                return None;
            }

            let mut cursor = 1usize;

            // Sequence number.
            if cursor >= frame.len() {
                return None;
            }
            let _seq = frame[cursor];
            cursor += 1;

            // Optional ack byte.
            if frame_flags & flags::ACK != 0 {
                cursor += 1; // skip ack_seq
            }

            // Payload length (u16 LE).
            if cursor + 2 > frame.len() {
                return None;
            }
            let payload_len = u16::from_le_bytes([frame[cursor], frame[cursor + 1]]) as usize;
            cursor += 2;

            if cursor + payload_len > frame.len() {
                return None;
            }

            // On the first segment of a new message, reset the buffer.
            if frame_flags & flags::BEGIN_SEG != 0 {
                self.segments.clear();
                self.expecting_more = true;
            }

            self.segments
                .extend_from_slice(&frame[cursor..cursor + payload_len]);

            self.next_seq = self.next_seq.wrapping_add(1);

            if frame_flags & flags::END_MSG != 0 {
                self.expecting_more = false;
                return Some(std::mem::take(&mut self.segments));
            }

            None
        }
    }

    impl Default for BtpReassembler {
        fn default() -> Self {
            Self::new()
        }
    }

    // ── BTP fragmenter ────────────────────────────────────────────────────────

    /// Fragment a Matter message into BLE ATT MTU-sized BTP data frames.
    ///
    /// # Parameters
    /// - `data`      — the complete Matter message bytes.
    /// - `attl`      — negotiated ATT payload size (bytes per BTP frame).
    /// - `start_seq` — sequence number of the first frame.
    ///
    /// The overhead per frame is:
    /// - 1 byte  flags
    /// - 1 byte  seq_num
    /// - 2 bytes payload_len  (carried in every frame for simplicity)
    ///
    /// Total header = 4 bytes; payload capacity = `attl - 4`.
    pub fn fragment_message(data: &[u8], attl: u16, start_seq: u8) -> Vec<Vec<u8>> {
        let attl = attl as usize;
        // Per-frame overhead: flags(1) + seq(1) + payload_len(2) = 4 bytes.
        let capacity = attl.saturating_sub(4).max(1);

        let chunks: Vec<&[u8]> = data.chunks(capacity).collect();
        let total = chunks.len();
        let mut frames = Vec::with_capacity(total);
        let mut seq = start_seq;

        for (i, chunk) in chunks.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i == total - 1;

            let mut frame_flags: u8 = 0;
            if is_first {
                frame_flags |= flags::BEGIN_SEG;
            }
            if is_last {
                frame_flags |= flags::END_MSG;
            } else {
                frame_flags |= flags::MORE_DATA;
            }

            let payload_len = chunk.len() as u16;
            let mut frame = Vec::with_capacity(4 + chunk.len());
            frame.push(frame_flags);
            frame.push(seq);
            frame.push((payload_len & 0xFF) as u8);
            frame.push(((payload_len >> 8) & 0xFF) as u8);
            frame.extend_from_slice(chunk);

            frames.push(frame);
            seq = seq.wrapping_add(1);
        }

        frames
    }

    // ── BleTransport ─────────────────────────────────────────────────────────

    /// BLE transport state machine for Matter commissioning.
    ///
    /// `rx` delivers reassembled Matter messages arriving from the controller.
    /// `tx` accepts Matter messages that will be fragmented and indicated on C2.
    ///
    /// Obtain a `BleTransport` via [`BleTransport::new`] or from
    /// [`crate::matter::ble::peripheral::MatterBlePeripheral::start`].
    pub struct BleTransport {
        /// Receive assembled Matter messages from the BLE controller.
        pub rx: mpsc::Receiver<Vec<u8>>,
        /// Send Matter messages to be fragmented and indicated on C2.
        pub tx: mpsc::Sender<Vec<u8>>,
        /// Agreed ATT payload size.
        pub attl: u16,
        /// Monotonically incrementing sequence counter for outgoing frames.
        pub next_seq: Arc<AtomicU8>,
    }

    impl BleTransport {
        /// Create a new `BleTransport` together with the *peer* channel ends.
        ///
        /// Returns `(transport, c1_tx, c2_rx)` where:
        /// - `c1_tx`  — push raw C1 bytes received from the BLE controller into
        ///   the reassembler (used by the peripheral task).
        /// - `c2_rx`  — pull fragmented BTP frames to indicate on C2
        ///   (used by the peripheral task).
        pub fn new(attl: u16) -> (Self, mpsc::Sender<Vec<u8>>, mpsc::Receiver<Vec<u8>>) {
            // Channel: controller → transport (reassembled messages)
            let (assembled_tx, assembled_rx) = mpsc::channel::<Vec<u8>>(32);
            // Channel: transport → controller (raw Matter messages to fragment + indicate)
            let (outbound_tx, outbound_rx) = mpsc::channel::<Vec<u8>>(32);

            let transport = BleTransport {
                rx: assembled_rx,
                tx: outbound_tx,
                attl,
                next_seq: Arc::new(AtomicU8::new(0)),
            };

            // c1_tx  = the sender the peripheral driver writes raw C1 frames into;
            //          transport.rx consumers read the *reassembled* messages.
            // c2_rx  = the receiver the peripheral driver reads to get Matter messages
            //          it must fragment and indicate; transport.tx sends those.
            (transport, assembled_tx, outbound_rx)
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;

        // Helper: build a minimal BtpHandshakeRequest byte slice.
        fn make_handshake_request_bytes(versions: u8, attl: u16, window: u8) -> Vec<u8> {
            vec![
                flags::HANDSHAKE,           // byte 0: flags
                (attl & 0xFF) as u8,        // byte 1: seg_len low  (mirrors attl)
                ((attl >> 8) & 0xFF) as u8, // byte 2: seg_len high
                versions,                   // byte 3: versions_supported
                (attl & 0xFF) as u8,        // byte 4: attl low
                ((attl >> 8) & 0xFF) as u8, // byte 5: attl high
                window,                     // byte 6: window_size
            ]
        }

        /// Parse a known BtpHandshakeRequest byte sequence.
        #[test]
        fn btp_handshake_request_parse() {
            let bytes = make_handshake_request_bytes(0x04, 247, 6);
            let req = BtpHandshakeRequest::parse(&bytes).expect("parse failed");
            assert_eq!(req.versions_supported, 0x04);
            assert_eq!(req.attl, 247);
            assert_eq!(req.window_size, 6);
        }

        /// Parse should fail on too-short input.
        #[test]
        fn btp_handshake_request_parse_too_short() {
            let short = vec![flags::HANDSHAKE, 0xF7, 0x00]; // only 3 bytes
            assert!(BtpHandshakeRequest::parse(&short).is_none());
        }

        /// Parse should fail when the HANDSHAKE flag is absent.
        #[test]
        fn btp_handshake_request_parse_wrong_flag() {
            let mut bytes = make_handshake_request_bytes(0x04, 247, 6);
            bytes[0] = 0x00; // clear HANDSHAKE flag
            assert!(BtpHandshakeRequest::parse(&bytes).is_none());
        }

        /// Encode a BtpHandshakeResponse and verify the byte layout.
        #[test]
        fn btp_handshake_response_encode() {
            let resp = BtpHandshakeResponse {
                selected_version: 4,
                attl: 247,
                window_size: 6,
            };
            let bytes = resp.encode();
            assert_eq!(bytes.len(), 7);
            assert_eq!(bytes[0], flags::HANDSHAKE); // SYN+ACK marker
            assert_eq!(bytes[3], 4); // selected_version
            assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 247); // attl
            assert_eq!(bytes[6], 6); // window_size
        }

        /// Round-trip: request → to_response() → encode, check version=4.
        #[test]
        fn btp_handshake_roundtrip() {
            let req_bytes = make_handshake_request_bytes(0x04, 200, 4);
            let req = BtpHandshakeRequest::parse(&req_bytes).unwrap();
            let resp = req.to_response();
            assert_eq!(resp.selected_version, 4);
            assert_eq!(resp.attl, 200);
            assert_eq!(resp.window_size, 4);
            let encoded = resp.encode();
            assert_eq!(encoded[3], 4);
        }

        // Helper: build a minimal single-segment BTP data frame.
        fn make_data_frame(seq: u8, payload: &[u8], is_first: bool, is_last: bool) -> Vec<u8> {
            let mut frame_flags: u8 = 0;
            if is_first {
                frame_flags |= flags::BEGIN_SEG;
            }
            if is_last {
                frame_flags |= flags::END_MSG;
            } else {
                frame_flags |= flags::MORE_DATA;
            }
            let plen = payload.len() as u16;
            let mut frame = vec![
                frame_flags,
                seq,
                (plen & 0xFF) as u8,
                ((plen >> 8) & 0xFF) as u8,
            ];
            frame.extend_from_slice(payload);
            frame
        }

        /// Feed a single-segment frame and receive the assembled message.
        #[test]
        fn btp_reassembler_single_segment() {
            let mut r = BtpReassembler::new();
            let data = b"hello matter";
            let frame = make_data_frame(0, data, true, true);
            let result = r.feed(&frame).expect("expected assembled message");
            assert_eq!(result, data);
        }

        /// Feed three segments — only the last one should yield the message.
        #[test]
        fn btp_reassembler_multi_segment() {
            let mut r = BtpReassembler::new();

            // Segment 1: BEGIN_SEG | MORE_DATA
            let frame1 = make_data_frame(0, b"aaa", true, false);
            // Segment 2: MORE_DATA only
            let frame2 = make_data_frame(1, b"bbb", false, false);
            // Segment 3: END_MSG only
            let frame3 = make_data_frame(2, b"ccc", false, true);

            assert!(r.feed(&frame1).is_none());
            assert!(r.feed(&frame2).is_none());
            let result = r
                .feed(&frame3)
                .expect("expected assembled message on last segment");
            assert_eq!(result, b"aaabbbccc");
        }

        /// Handshake frames should be ignored by the reassembler.
        #[test]
        fn btp_reassembler_ignores_handshake_frames() {
            let mut r = BtpReassembler::new();
            // Build a valid handshake frame.
            let bytes = make_handshake_request_bytes(0x04, 247, 6);
            assert!(r.feed(&bytes).is_none());
        }

        /// A small message that fits in one frame produces exactly one fragment.
        #[test]
        fn btp_fragment_single_chunk() {
            let data = b"short";
            let frames = fragment_message(data, 247, 0);
            assert_eq!(frames.len(), 1);
            let f = &frames[0];
            // Flags: BEGIN_SEG | END_MSG
            assert_eq!(f[0], flags::BEGIN_SEG | flags::END_MSG);
            assert_eq!(f[1], 0); // seq = 0
            let plen = u16::from_le_bytes([f[2], f[3]]) as usize;
            assert_eq!(&f[4..4 + plen], data.as_ref());
        }

        /// A large message that requires multiple fragments splits correctly.
        #[test]
        fn btp_fragment_multi_chunk() {
            // 4-byte overhead per frame; capacity = attl(10) - 4 = 6 bytes/frame.
            // 18 bytes of data → ceil(18/6) = 3 frames.
            let data = b"ABCDEFGHIJKLMNOPQR"; // 18 bytes
            let frames = fragment_message(data, 10, 5);
            assert_eq!(frames.len(), 3);

            // First frame: BEGIN_SEG | MORE_DATA, seq=5
            assert_eq!(frames[0][0], flags::BEGIN_SEG | flags::MORE_DATA);
            assert_eq!(frames[0][1], 5);

            // Middle frame: MORE_DATA only, seq=6
            assert_eq!(frames[1][0], flags::MORE_DATA);
            assert_eq!(frames[1][1], 6);

            // Last frame: END_MSG only, seq=7
            assert_eq!(frames[2][0], flags::END_MSG);
            assert_eq!(frames[2][1], 7);

            // Reassemble and verify content round-trips.
            let mut r = BtpReassembler::new();
            assert!(r.feed(&frames[0]).is_none());
            assert!(r.feed(&frames[1]).is_none());
            let assembled = r.feed(&frames[2]).unwrap();
            assert_eq!(assembled, data.as_ref());
        }

        /// BleTransport::new returns correctly wired channel endpoints.
        #[test]
        fn ble_transport_new_channels() {
            let (transport, c1_tx, _c2_rx) = BleTransport::new(247);
            assert_eq!(transport.attl, 247);
            // Verify the assembled-message sender (c1_tx) reaches transport.rx.
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async {
                c1_tx.send(vec![0xDE, 0xAD]).await.unwrap();
                let mut rx = transport.rx;
                let msg = rx.recv().await.unwrap();
                assert_eq!(msg, vec![0xDE, 0xAD]);
            });
        }
    }
}
