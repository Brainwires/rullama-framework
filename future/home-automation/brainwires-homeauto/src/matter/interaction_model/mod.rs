//! Matter 1.3 Interaction Model (IM) — protocol ID 0x0001.
//!
//! All IM messages are TLV-encoded in the decrypted payload of a
//! `MatterMessage`. The opcode byte (embedded as the first byte of the
//! serialized payload) selects the message type. The sub-modules here
//! implement TLV encode/decode for each message body.
//!
//! Reference: Matter spec §8 (Interaction Model).

/// InvokeRequest / InvokeResponse — cluster-command invocation.
pub mod invoke;
/// ReadRequest / ReportData — attribute reads and reports.
pub mod read;
/// SubscribeRequest / SubscribeResponse — long-lived attribute subscriptions.
pub mod subscribe;
/// WriteRequest / WriteResponse — attribute writes.
pub mod write;

// ── Protocol constant ─────────────────────────────────────────────────────────

/// IM protocol identifier (used in the Matter exchange header).
pub const PROTOCOL_ID: u16 = 0x0001;

// ── Opcode enum ───────────────────────────────────────────────────────────────

/// IM protocol opcodes (Matter spec §8.10, Table 44).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImOpcode {
    /// `0x01` — StatusResponse: generic status-code reply.
    StatusResponse = 0x01,
    /// `0x02` — ReadRequest: attribute read.
    ReadRequest = 0x02,
    /// `0x03` — SubscribeRequest: open a subscription.
    SubscribeRequest = 0x03,
    /// `0x04` — SubscribeResponse: confirm a subscription.
    SubscribeResponse = 0x04,
    /// `0x05` — ReportData: subscription data push.
    ReportData = 0x05,
    /// `0x06` — WriteRequest: attribute write.
    WriteRequest = 0x06,
    /// `0x07` — WriteResponse: reply to WriteRequest.
    WriteResponse = 0x07,
    /// `0x08` — InvokeRequest: cluster-command invocation.
    InvokeRequest = 0x08,
    /// `0x09` — InvokeResponse: reply to InvokeRequest.
    InvokeResponse = 0x09,
    /// `0x0A` — TimedRequest: FailSafe-protected timed action.
    TimedRequest = 0x0A,
}

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use invoke::{InvokeRequest, InvokeResponse, InvokeResponseItem};
pub use read::{AttributeData, ReadRequest, ReportData};
pub use subscribe::{SubscribeRequest, SubscribeResponse};
pub use write::{AttributeStatus, InteractionStatus, WriteRequest, WriteResponse};
