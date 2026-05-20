//! NetworkCommissioning cluster server (cluster ID 0x0031).
//!
//! Simplified on-network commissioning implementation. For IP-connected
//! devices that are already on-network, all WiFi/Thread scan/connect commands
//! return immediate success so the commissioning flow can continue.
//!
//! Matter spec §11.8.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::matter::clusters::tlv;
use crate::matter::data_model::ClusterServer;
use crate::matter::error::{MatterError, MatterResult};

// ── Attribute IDs ─────────────────────────────────────────────────────────────

/// `0x0000` — MaxNetworks attribute (max provisioned networks).
pub const ATTR_MAX_NETWORKS: u32 = 0x0000;
/// `0x0001` — Networks attribute (list of provisioned networks).
pub const ATTR_NETWORKS: u32 = 0x0001;
/// `0x0002` — ScanMaxTimeSeconds attribute.
pub const ATTR_SCAN_MAX_TIME_SECONDS: u32 = 0x0002;
/// `0x0003` — ConnectMaxTimeSeconds attribute.
pub const ATTR_CONNECT_MAX_TIME_SECONDS: u32 = 0x0003;
/// `0x0004` — InterfaceEnabled attribute.
pub const ATTR_INTERFACE_ENABLED: u32 = 0x0004;
/// `0x0005` — LastNetworkingStatus attribute (last network-op result code).
pub const ATTR_LAST_NETWORKING_STATUS: u32 = 0x0005;
/// `0x0006` — LastNetworkID attribute (last network we attempted to connect to).
pub const ATTR_LAST_NETWORK_ID: u32 = 0x0006;
/// `0x0007` — LastConnectErrorValue attribute.
pub const ATTR_LAST_CONNECT_ERROR_VALUE: u32 = 0x0007;

// ── Command IDs ───────────────────────────────────────────────────────────────

/// `0x00` — ScanNetworks command.
pub const CMD_SCAN_NETWORKS: u32 = 0x00;
/// `0x02` — AddOrUpdateWiFiNetwork command.
pub const CMD_ADD_OR_UPDATE_WIFI_NETWORK: u32 = 0x02;
/// `0x06` — ConnectNetwork command.
pub const CMD_CONNECT_NETWORK: u32 = 0x06;
/// `0x07` — ReorderNetwork command.
pub const CMD_REORDER_NETWORK: u32 = 0x07;

const CLUSTER_ID: u32 = 0x0031;

// ── TLV encoding helpers (local) ──────────────────────────────────────────────

fn tlv_uint8(tag: u8, val: u8) -> Vec<u8> {
    vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1, tag, val]
}

fn tlv_bool(tag: u8, val: bool) -> Vec<u8> {
    let ty = if val {
        tlv::TYPE_BOOL_TRUE
    } else {
        tlv::TYPE_BOOL_FALSE
    };
    vec![tlv::TAG_CONTEXT_1 | ty, tag]
}

fn tlv_null(tag: u8) -> Vec<u8> {
    vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_NULL, tag]
}

fn wrap_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_STRUCTURE];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

fn wrap_list(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_LIST];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

/// Build a NetworkConfigResponse: `struct { NetworkingStatus(0): uint8, Networks(1): list }`
fn network_config_response(status: u8) -> Vec<u8> {
    let mut inner = tlv_uint8(0, status);
    // Empty network list.
    let list_ctrl: u8 = tlv::TAG_CONTEXT_1 | tlv::TYPE_LIST;
    inner.extend_from_slice(&[list_ctrl, 1u8, tlv::TYPE_END_OF_CONTAINER]);
    wrap_struct(&inner)
}

/// Build a ConnectNetworkResponse:
/// `struct { NetworkingStatus(0): uint8, ErrorValue(1): null }`
fn connect_network_response(status: u8) -> Vec<u8> {
    let mut inner = tlv_uint8(0, status);
    inner.extend_from_slice(&tlv_null(1));
    wrap_struct(&inner)
}

// ── Network entry ─────────────────────────────────────────────────────────────

/// A stored network entry (added via AddOrUpdateWiFiNetwork).
#[derive(Debug, Clone)]
pub struct NetworkEntry {
    /// Network identifier — SSID for WiFi, Extended PAN ID for Thread.
    pub network_id: Vec<u8>,
    /// Whether the device currently has connectivity on this network.
    pub connected: bool,
}

// ── State ─────────────────────────────────────────────────────────────────────

/// Mutable state for the NetworkCommissioning cluster.
#[derive(Debug, Default)]
pub struct NetworkCommissioningState {
    /// Provisioned networks (SSIDs / Thread datasets).
    pub networks: Vec<NetworkEntry>,
    /// Result code of the most recent network operation.
    pub last_networking_status: Option<u8>,
    /// Last network identifier the device attempted to connect to.
    pub last_network_id: Option<Vec<u8>>,
}

// ── NetworkCommissioningCluster ───────────────────────────────────────────────

/// Server for the NetworkCommissioning cluster (0x0031).
pub struct NetworkCommissioningCluster {
    state: Arc<Mutex<NetworkCommissioningState>>,
}

impl NetworkCommissioningCluster {
    /// Create a new cluster server with empty state.
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(NetworkCommissioningState::default())),
        }
    }
}

impl Default for NetworkCommissioningCluster {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ClusterServer for NetworkCommissioningCluster {
    fn cluster_id(&self) -> u32 {
        CLUSTER_ID
    }

    async fn read_attribute(&self, attr_id: u32) -> MatterResult<Vec<u8>> {
        match attr_id {
            ATTR_MAX_NETWORKS => Ok(tlv_uint8(0, 1)),
            ATTR_NETWORKS => {
                let st = self.state.lock().unwrap();
                let mut items = Vec::new();
                for entry in &st.networks {
                    let mut inner =
                        vec![tlv::TAG_CONTEXT_1 | 0x10, 0u8, entry.network_id.len() as u8];
                    inner.extend_from_slice(&entry.network_id);
                    inner.extend_from_slice(&tlv_bool(1, entry.connected));
                    items.extend_from_slice(&wrap_struct(&inner));
                }
                Ok(wrap_list(&items))
            }
            ATTR_SCAN_MAX_TIME_SECONDS => Ok(tlv_uint8(0, 10)),
            ATTR_CONNECT_MAX_TIME_SECONDS => Ok(tlv_uint8(0, 20)),
            ATTR_INTERFACE_ENABLED => Ok(tlv_bool(0, true)),
            ATTR_LAST_NETWORKING_STATUS => {
                let st = self.state.lock().unwrap();
                match st.last_networking_status {
                    Some(s) => Ok(tlv_uint8(0, s)),
                    None => Ok(tlv_null(0)),
                }
            }
            ATTR_LAST_NETWORK_ID => {
                let st = self.state.lock().unwrap();
                match &st.last_network_id {
                    Some(id) => {
                        let mut v = vec![tlv::TAG_CONTEXT_1 | 0x10, 0u8, id.len() as u8];
                        v.extend_from_slice(id);
                        Ok(v)
                    }
                    None => Ok(tlv_null(0)),
                }
            }
            ATTR_LAST_CONNECT_ERROR_VALUE => {
                // null initially (nullable int32)
                Ok(tlv_null(0))
            }
            _ => Err(MatterError::Transport("unsupported attribute".into())),
        }
    }

    async fn write_attribute(&self, attr_id: u32, value: &[u8]) -> MatterResult<()> {
        match attr_id {
            ATTR_INTERFACE_ENABLED => {
                // Accept write silently (stub).
                let _ = value;
                Ok(())
            }
            _ => Err(MatterError::Transport("attribute not writable".into())),
        }
    }

    async fn invoke_command(&self, cmd_id: u32, _args: &[u8]) -> MatterResult<Vec<u8>> {
        match cmd_id {
            CMD_SCAN_NETWORKS => {
                // On-network devices don't scan; return empty list with success.
                Ok(network_config_response(0))
            }
            CMD_ADD_OR_UPDATE_WIFI_NETWORK => {
                // Accept the network configuration and return success.
                Ok(network_config_response(0))
            }
            CMD_CONNECT_NETWORK => {
                // Already connected (on-network device).
                Ok(connect_network_response(0))
            }
            CMD_REORDER_NETWORK => Ok(network_config_response(0)),
            _ => Err(MatterError::Transport(format!(
                "unknown command {cmd_id:#06x}"
            ))),
        }
    }

    fn attribute_ids(&self) -> Vec<u32> {
        vec![
            ATTR_MAX_NETWORKS,
            ATTR_NETWORKS,
            ATTR_SCAN_MAX_TIME_SECONDS,
            ATTR_CONNECT_MAX_TIME_SECONDS,
            ATTR_INTERFACE_ENABLED,
            ATTR_LAST_NETWORKING_STATUS,
            ATTR_LAST_NETWORK_ID,
            ATTR_LAST_CONNECT_ERROR_VALUE,
        ]
    }

    fn command_ids(&self) -> Vec<u32> {
        vec![
            CMD_SCAN_NETWORKS,
            CMD_ADD_OR_UPDATE_WIFI_NETWORK,
            CMD_CONNECT_NETWORK,
            CMD_REORDER_NETWORK,
        ]
    }
}
