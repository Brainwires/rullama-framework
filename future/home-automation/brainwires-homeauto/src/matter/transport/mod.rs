/// Matter transport layer.
///
/// Provides the wire-level message encoding/decoding, reliable delivery
/// (MRP), and UDP socket I/O for the Matter 1.3 protocol stack.
///
/// # Modules
///
/// | Module | Description |
/// |--------|-------------|
/// | [`message`] | Matter Message Layer header and framing (§4.4) |
/// | [`mrp`]     | Message Reliability Protocol state machine (§4.12) |
/// | [`udp`]     | UDP send/receive with AES-128-CCM encryption |
/// | [`ble`]     | BLE transport stub (Phase 8, `matter-ble` feature) |
///
/// Matter Message Layer: header encoding/decoding and wire format.
pub mod message;

/// Message Reliability Protocol: retransmit tracking and ACK payloads.
pub mod mrp;

/// UDP transport: AES-128-CCM encrypted send/receive over UDP sockets.
pub mod udp;

/// BLE transport stub (`matter-ble` feature, fully implemented in Phase 8).
pub mod ble;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use message::{MatterMessage, MessageHeader, NodeAddress, SessionType};
pub use mrp::{MrpConfig, MrpExchange};
pub use udp::{SessionKeys, SessionMap, UdpTransport};
