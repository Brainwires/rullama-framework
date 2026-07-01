//! Matter 1.3 Secure Channel — PASE and CASE session establishment.
//!
//! Implements Matter Core Specification §4.13 (CASE/SIGMA) and §4.14 (PASE).
//!
//! # Session establishment flow
//!
//! ```text
//! PASE (commissioning, password-based):
//!   Commissioner ──PBKDFParamRequest──>  Commissionee
//!   Commissioner <─PBKDFParamResponse──  Commissionee
//!   Commissioner ──Pake1─────────────>  Commissionee
//!   Commissioner <─Pake2──────────────  Commissionee
//!   Commissioner ──Pake3─────────────>  Commissionee
//!   (both derive session keys from SPAKE2+ Ke)
//!
//! CASE (operational, certificate-based):
//!   Initiator ──Sigma1──>  Responder
//!   Initiator <─Sigma2───  Responder
//!   Initiator ──Sigma3──>  Responder
//!   (both derive session keys via ECDH + HKDF)
//! ```

/// CASE (Certificate Authenticated Session Establishment) — operational sessions.
pub mod case;
/// PASE (Password Authenticated Session Establishment) — commissioning sessions.
pub mod pase;

// ── Protocol constants ────────────────────────────────────────────────────────

/// Secure Channel protocol identifier (used in Exchange header).
pub const SECURE_CHANNEL_PROTOCOL_ID: u16 = 0x0000;

// ── Protocol opcodes ──────────────────────────────────────────────────────────

/// Opcodes for Secure Channel protocol messages.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecureChannelOpcode {
    /// `0x00` — MsgCounterSyncReq — request peer message-counter sync.
    MsgCounterSyncReq = 0x00,
    /// `0x01` — MsgCounterSyncRsp — response carrying the sync'd counter.
    MsgCounterSyncRsp = 0x01,
    /// `0x10` — MrpStandaloneAck — MRP acknowledgement without a payload.
    MrpStandaloneAck = 0x10,
    /// `0x20` — PBKDFParamRequest — first PASE message.
    PbkdfParamRequest = 0x20,
    /// `0x21` — PBKDFParamResponse — PASE parameters from the commissionee.
    PbkdfParamResponse = 0x21,
    /// `0x22` — Pake1 — SPAKE2+ PA message (commissioner → commissionee).
    Pake1 = 0x22,
    /// `0x23` — Pake2 — SPAKE2+ PB + cB (commissionee → commissioner).
    Pake2 = 0x23,
    /// `0x24` — Pake3 — SPAKE2+ cA (commissioner → commissionee).
    Pake3 = 0x24,
    /// `0x40` — StatusReport — generic secure-channel status.
    StatusReport = 0x40,
    /// `0x30` — Sigma1 — first CASE message.
    Sigma1 = 0x30,
    /// `0x31` — Sigma2 — CASE response with responder-authenticated fields.
    Sigma2 = 0x31,
    /// `0x32` — Sigma3 — initiator's authenticated reply completing CASE.
    Sigma3 = 0x32,
    /// `0x33` — Sigma2Resume — fast-path CASE session resumption.
    Sigma2Resume = 0x33,
}

// ── Established session ───────────────────────────────────────────────────────

/// A fully established Matter session with symmetric keys ready for use.
///
/// After a successful PASE or CASE handshake, both sides hold an
/// `EstablishedSession` with:
/// - A session ID pair (local ↔ peer).
/// - Symmetric AES-128 keys: `encrypt_key` (outbound) and `decrypt_key` (inbound).
/// - A 32-byte attestation challenge.
/// - (CASE only) The authenticated peer Node ID.
#[derive(Debug, Clone)]
pub struct EstablishedSession {
    /// This node's local session ID.
    pub session_id: u16,
    /// The peer's session ID.
    pub peer_session_id: u16,
    /// Key for encrypting outbound messages.
    pub encrypt_key: [u8; 16],
    /// Key for decrypting inbound messages.
    pub decrypt_key: [u8; 16],
    /// 32-byte attestation challenge derived alongside the session keys.
    pub attestation_challenge: [u8; 32],
    /// Peer Node ID (set by CASE, `None` for PASE).
    pub peer_node_id: Option<u64>,
}

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use case::{CaseInitiator, CaseResponder};
pub use pase::{PaseCommissionee, PaseCommissioner};
