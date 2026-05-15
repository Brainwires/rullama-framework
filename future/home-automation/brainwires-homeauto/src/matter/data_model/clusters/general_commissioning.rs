//! GeneralCommissioning cluster server (cluster ID 0x0030).
//!
//! Handles FailSafe, regulatory config, and CommissioningComplete.
//! Matter spec §11.9.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::matter::clusters::tlv;
use crate::matter::data_model::ClusterServer;
use crate::matter::error::{MatterError, MatterResult};

// ── Attribute IDs ─────────────────────────────────────────────────────────────

/// `0x0000` — Breadcrumb attribute (commissioner-supplied tracking value).
pub const ATTR_BREADCRUMB: u32 = 0x0000;
/// `0x0001` — BasicCommissioningInfo attribute (FailSafe defaults).
pub const ATTR_BASIC_COMMISSIONING_INFO: u32 = 0x0001;
/// `0x0002` — RegulatoryConfig attribute (indoor / outdoor / both).
pub const ATTR_REGULATORY_CONFIG: u32 = 0x0002;
/// `0x0003` — LocationCapability attribute (supported regulatory locations).
pub const ATTR_LOCATION_CAPABILITY: u32 = 0x0003;
/// `0x0004` — SupportsConcurrentConnection attribute.
pub const ATTR_SUPPORTS_CONCURRENT_CONNECTION: u32 = 0x0004;

// ── Command IDs ───────────────────────────────────────────────────────────────

/// `0x00` — ArmFailSafe command (opens the FailSafe window).
pub const CMD_ARM_FAIL_SAFE: u32 = 0x00;
/// `0x02` — SetRegulatoryConfig command.
pub const CMD_SET_REGULATORY_CONFIG: u32 = 0x02;
/// `0x04` — CommissioningComplete command (closes FailSafe on success).
pub const CMD_COMMISSIONING_COMPLETE: u32 = 0x04;

const CLUSTER_ID: u32 = 0x0030;

// ── TLV encoding helpers (local) ──────────────────────────────────────────────

fn tlv_uint8(tag: u8, val: u8) -> Vec<u8> {
    vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1, tag, val]
}

fn tlv_uint16(tag: u8, val: u16) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_2, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

fn tlv_uint64(tag: u8, val: u64) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | 0x07, tag]; // TYPE_UNSIGNED_INT_8 = 0x07
    v.extend_from_slice(&val.to_le_bytes());
    v
}

fn tlv_bool(tag: u8, val: bool) -> Vec<u8> {
    let ty = if val {
        tlv::TYPE_BOOL_TRUE
    } else {
        tlv::TYPE_BOOL_FALSE
    };
    vec![tlv::TAG_CONTEXT_1 | ty, tag]
}

fn tlv_utf8_string(tag: u8, s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut v = vec![tlv::TAG_CONTEXT_1 | 0x0C, tag, bytes.len() as u8];
    v.extend_from_slice(bytes);
    v
}

fn wrap_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_STRUCTURE];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

/// Build the CommissioningComplete response:
/// `struct { tag 0: ErrorCode(uint8), tag 1: DebugText(string) }`
fn commissioning_complete_response(error_code: u8, debug_text: &str) -> Vec<u8> {
    let mut inner = tlv_uint8(0, error_code);
    inner.extend_from_slice(&tlv_utf8_string(1, debug_text));
    wrap_struct(&inner)
}

// ── State ─────────────────────────────────────────────────────────────────────

/// Mutable state for the GeneralCommissioning cluster.
#[derive(Debug, Default)]
pub struct GeneralCommissioningState {
    /// The Breadcrumb attribute (mutable, default 0).
    pub breadcrumb: u64,
    /// Whether ArmFailSafe has been called without a CommissioningComplete.
    pub failsafe_armed: bool,
    /// Requested FailSafe expiry length in seconds.
    pub failsafe_expiry_seconds: u16,
}

// ── GeneralCommissioningCluster ───────────────────────────────────────────────

/// Server for the GeneralCommissioning cluster (0x0030).
pub struct GeneralCommissioningCluster {
    state: Arc<Mutex<GeneralCommissioningState>>,
}

impl GeneralCommissioningCluster {
    /// Create a new cluster server with default state.
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(GeneralCommissioningState::default())),
        }
    }
}

impl Default for GeneralCommissioningCluster {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ClusterServer for GeneralCommissioningCluster {
    fn cluster_id(&self) -> u32 {
        CLUSTER_ID
    }

    async fn read_attribute(&self, attr_id: u32) -> MatterResult<Vec<u8>> {
        match attr_id {
            ATTR_BREADCRUMB => {
                let breadcrumb = self.state.lock().unwrap().breadcrumb;
                Ok(tlv_uint64(0, breadcrumb))
            }
            ATTR_BASIC_COMMISSIONING_INFO => {
                // struct { MaxCumulativeFailsafeSeconds(0): uint16=900, FailSafeExpiryLengthSeconds(1): uint16=60 }
                let mut inner = tlv_uint16(0, 900);
                inner.extend_from_slice(&tlv_uint16(1, 60));
                Ok(wrap_struct(&inner))
            }
            ATTR_REGULATORY_CONFIG => {
                // 0 = Indoor
                Ok(tlv_uint8(0, 0))
            }
            ATTR_LOCATION_CAPABILITY => {
                // 2 = IndoorOutdoor
                Ok(tlv_uint8(0, 2))
            }
            ATTR_SUPPORTS_CONCURRENT_CONNECTION => Ok(tlv_bool(0, true)),
            _ => Err(MatterError::Transport("unsupported attribute".into())),
        }
    }

    async fn write_attribute(&self, attr_id: u32, value: &[u8]) -> MatterResult<()> {
        match attr_id {
            ATTR_BREADCRUMB => {
                // Expect an 8-byte LE uint64 payload (after the TLV prefix).
                // Value may arrive as raw bytes or TLV-encoded; we accept both forms.
                let raw = if value.len() >= 2 && value[0] == (tlv::TAG_CONTEXT_1 | 0x07) {
                    // TLV-encoded: [ctrl, tag, 8 bytes LE]
                    if value.len() < 10 {
                        return Err(MatterError::Transport("bad Breadcrumb TLV".into()));
                    }
                    &value[2..10]
                } else if value.len() >= 8 {
                    &value[..8]
                } else {
                    return Err(MatterError::Transport("Breadcrumb value too short".into()));
                };
                let bc = u64::from_le_bytes(raw.try_into().unwrap());
                self.state.lock().unwrap().breadcrumb = bc;
                Ok(())
            }
            _ => Err(MatterError::Transport("attribute not writable".into())),
        }
    }

    async fn invoke_command(&self, cmd_id: u32, args: &[u8]) -> MatterResult<Vec<u8>> {
        match cmd_id {
            CMD_ARM_FAIL_SAFE => {
                // Parse ExpiryLengthSeconds from args TLV (tag 0, uint16).
                let expiry = parse_u16_tag0(args).unwrap_or(60);
                {
                    let mut st = self.state.lock().unwrap();
                    st.failsafe_armed = expiry > 0;
                    st.failsafe_expiry_seconds = expiry;
                }
                Ok(commissioning_complete_response(0, ""))
            }
            CMD_SET_REGULATORY_CONFIG => {
                // Parse Breadcrumb (tag 2, uint64) if present and update state.
                if let Some(bc) = parse_u64_tag2(args) {
                    self.state.lock().unwrap().breadcrumb = bc;
                }
                Ok(commissioning_complete_response(0, ""))
            }
            CMD_COMMISSIONING_COMPLETE => {
                {
                    let mut st = self.state.lock().unwrap();
                    st.failsafe_armed = false;
                }
                Ok(commissioning_complete_response(0, ""))
            }
            _ => Err(MatterError::Transport(format!(
                "unknown command {cmd_id:#06x}"
            ))),
        }
    }

    fn attribute_ids(&self) -> Vec<u32> {
        vec![
            ATTR_BREADCRUMB,
            ATTR_BASIC_COMMISSIONING_INFO,
            ATTR_REGULATORY_CONFIG,
            ATTR_LOCATION_CAPABILITY,
            ATTR_SUPPORTS_CONCURRENT_CONNECTION,
        ]
    }

    fn command_ids(&self) -> Vec<u32> {
        vec![
            CMD_ARM_FAIL_SAFE,
            CMD_SET_REGULATORY_CONFIG,
            CMD_COMMISSIONING_COMPLETE,
        ]
    }
}

// ── Argument parsers ──────────────────────────────────────────────────────────

/// Parse a context-tagged uint16 at tag 0 from TLV bytes.
fn parse_u16_tag0(args: &[u8]) -> Option<u16> {
    // Args arrive as struct body bytes (without wrapping TYPE_STRUCTURE).
    // Look for: [TAG_CONTEXT_1 | TYPE_UNSIGNED_INT_2, 0, lo, hi]
    let ctrl = tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_2;
    let mut i = 0;
    // Skip struct opener if present.
    if args.first() == Some(&tlv::TYPE_STRUCTURE) {
        i += 1;
    }
    while i + 3 < args.len() {
        if args[i] == ctrl && args[i + 1] == 0 {
            return Some(u16::from_le_bytes([args[i + 2], args[i + 3]]));
        }
        i += 1;
    }
    None
}

/// Parse a context-tagged uint64 at tag 2 from TLV bytes.
fn parse_u64_tag2(args: &[u8]) -> Option<u64> {
    let ctrl = tlv::TAG_CONTEXT_1 | 0x07; // TYPE_UNSIGNED_INT_8
    let mut i = 0;
    if args.first() == Some(&tlv::TYPE_STRUCTURE) {
        i += 1;
    }
    while i + 9 < args.len() {
        if args[i] == ctrl && args[i + 1] == 2 {
            let raw: [u8; 8] = args[i + 2..i + 10].try_into().ok()?;
            return Some(u64::from_le_bytes(raw));
        }
        i += 1;
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster() -> GeneralCommissioningCluster {
        GeneralCommissioningCluster::new()
    }

    #[tokio::test]
    async fn arm_failsafe_returns_success_response() {
        let cluster = make_cluster();
        // Build ArmFailSafe args: struct { ExpiryLengthSeconds(0): uint16=60 }
        let mut inner = tlv_uint16(0, 60);
        let args = {
            let mut v = vec![tlv::TYPE_STRUCTURE];
            v.append(&mut inner);
            v.push(tlv::TYPE_END_OF_CONTAINER);
            v
        };
        let resp = cluster
            .invoke_command(CMD_ARM_FAIL_SAFE, &args)
            .await
            .expect("ArmFailSafe failed");

        // Response must be a struct starting with TYPE_STRUCTURE.
        assert_eq!(resp[0], tlv::TYPE_STRUCTURE, "response should be a struct");
        // Extract ErrorCode (tag 0, uint8) — should be 0.
        let error_code = extract_uint8_tag0(&resp).expect("ErrorCode not found");
        assert_eq!(error_code, 0);
        // FailSafe should now be armed.
        assert!(cluster.state.lock().unwrap().failsafe_armed);
    }

    #[tokio::test]
    async fn commissioning_complete_returns_error_code_zero() {
        let cluster = make_cluster();
        // Arm first so we can disarm.
        cluster.state.lock().unwrap().failsafe_armed = true;

        let resp = cluster
            .invoke_command(CMD_COMMISSIONING_COMPLETE, &[])
            .await
            .expect("CommissioningComplete failed");

        assert_eq!(resp[0], tlv::TYPE_STRUCTURE);
        let error_code = extract_uint8_tag0(&resp).expect("ErrorCode not found");
        assert_eq!(error_code, 0);
        assert!(!cluster.state.lock().unwrap().failsafe_armed);
    }

    /// Extract a context-tagged uint8 at tag 0 from a TLV struct.
    fn extract_uint8_tag0(data: &[u8]) -> Option<u8> {
        let ctrl = tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1;
        let mut i = 1; // skip TYPE_STRUCTURE
        while i + 2 < data.len() {
            if data[i] == ctrl && data[i + 1] == 0 {
                return Some(data[i + 2]);
            }
            i += 1;
        }
        None
    }
}
