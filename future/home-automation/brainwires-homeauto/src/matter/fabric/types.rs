//! Matter fabric identity types.
//!
//! A Matter fabric is an administrative domain identified by a root CA certificate,
//! a fabric ID, and a node ID.  The `FabricIndex` is a per-device handle (1..=254)
//! used in the Fabric Descriptor cluster.

// ── Primitive identity wrappers ───────────────────────────────────────────────

/// Local per-node index for a commissioned fabric (1-based, max 254 per Matter spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FabricIndex(pub u8);

/// Operational Node ID within a fabric (Matter spec §2.5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OperationalNodeId(pub u64);

/// 64-bit compressed fabric identifier (derived from root CA public key + fabric ID
/// via HKDF; used in operational node advertisements per Matter §4.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompressedFabricId(pub u64);

// ── Fabric Descriptor ─────────────────────────────────────────────────────────

/// Information about a commissioned fabric stored on a node
/// (mirrors the FabricDescriptorStruct in Matter spec §9.6.5.4).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FabricDescriptor {
    /// Local index assigned to this fabric by the device.
    pub fabric_index: FabricIndex,
    /// Uncompressed P-256 root public key (65 bytes, 0x04 prefix).
    pub root_public_key: Vec<u8>,
    /// Vendor ID of the commissioner that provisioned this fabric.
    pub vendor_id: u16,
    /// Global fabric identifier.
    pub fabric_id: u64,
    /// This node's Node ID within the fabric.
    pub node_id: u64,
    /// Human-readable label assigned by the commissioner (max 32 bytes per spec).
    pub label: String,
}
