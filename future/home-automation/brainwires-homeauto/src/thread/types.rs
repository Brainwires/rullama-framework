use serde::{Deserialize, Serialize};

/// Operational state of this Thread node (as reported by OTBR REST API).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ThreadRole {
    /// Thread stack is disabled — node is not participating in any network.
    Disabled,
    /// Stack enabled but not currently attached to a partition.
    Detached,
    /// End device attached to a parent router.
    Child,
    /// Router — forwards packets for attached children.
    Router,
    /// Leader — the single partition leader managing network state.
    Leader,
    /// Any role string reported by OTBR that isn't in this enum.
    #[serde(other)]
    Unknown,
}

/// Info about the local Thread node, from `GET /node`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadNodeInfo {
    /// 16-bit RLOC address in hex (e.g. `"0x0400"`).
    pub rloc16: Option<String>,
    /// Extended 64-bit MAC address (EUI-64) in hex.
    pub ext_address: Option<String>,
    /// Extended PAN ID (8 bytes hex).
    pub ext_panid: Option<String>,
    /// Human-readable network name.
    pub network_name: Option<String>,
    /// Router/child/leader/etc.
    pub role: Option<ThreadRole>,
    /// Whether border routing is active.
    pub border_routing_state: Option<String>,
    /// Thread dataset version.
    pub version_threshold: Option<u32>,
}

/// A Thread neighbor entry from `GET /node/neighbors`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadNeighbor {
    /// EUI-64 of the neighbor, hex-encoded.
    pub ext_address: Option<String>,
    /// 16-bit RLOC address of the neighbor, hex-encoded.
    pub rloc16: Option<String>,
    /// Received signal strength in dBm.
    pub rssi: Option<i32>,
    /// Incoming link quality metric (0–3 per Thread spec).
    pub link_quality_in: Option<u8>,
    /// Seconds since this neighbor was last heard.
    pub age: Option<u32>,
    /// True if the neighbor is a Full Thread Device (FTD), false for MTD.
    pub full_thread_device: Option<bool>,
    /// True if the neighbor keeps its receiver on when idle.
    pub rx_on_when_idle: Option<bool>,
}

/// Active operational dataset from `GET /node/dataset/active`.
/// The OTBR REST API returns this as a hex TLV string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadNetworkDataset {
    /// Raw TLV-encoded dataset as hex string.
    pub active_dataset: String,
}
