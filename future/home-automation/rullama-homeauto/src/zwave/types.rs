use super::super::types::Capability;
use serde::{Deserialize, Serialize};

/// Z-Wave node ID (1–232, 0 = invalid).
pub type NodeId = u8;

/// Z-Wave device type classification.
///
/// Maps the generic/specific device-type codes reported during
/// `NodeProtocolInfo` into a coarse user-facing category. Finer distinctions
/// live in `command_classes` on the [`ZWaveNode`] record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZWaveNodeKind {
    /// Binary on/off switch or relay.
    Switch,
    /// Dimmable on/off switch (binary class with dim ramp).
    DimmableSwitch,
    /// Multi-level switch (0–99 percent position).
    MultiLevelSwitch,
    /// Binary sensor (door/window contact, occupancy).
    BinarySensor,
    /// Multi-level sensor (temperature, luminance, etc.).
    MultiLevelSensor,
    /// Thermostat with setpoint control.
    Thermostat,
    /// Door-lock actuator.
    DoorLock,
    /// Siren / alarm.
    Siren,
    /// Multi-outlet power strip.
    PowerStrip,
    /// Energy-monitoring meter (kWh, W, V, A).
    EnergyMeter,
    /// Device type reported by the node that isn't in this enum.
    Unknown,
}

/// A Z-Wave node on the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZWaveNode {
    /// 8-bit Z-Wave node identifier (1..=232).
    pub node_id: NodeId,
    /// User-assigned friendly name, when known.
    pub name: Option<String>,
    /// 16-bit manufacturer identifier assigned by Z-Wave Alliance.
    pub manufacturer_id: u16,
    /// Manufacturer-scoped product type code.
    pub product_type: u16,
    /// Manufacturer-scoped product ID code.
    pub product_id: u16,
    /// Coarse device kind classification.
    pub kind: ZWaveNodeKind,
    /// High-level capabilities exposed by this node.
    pub capabilities: Vec<Capability>,
    /// Supported command-class IDs discovered on this node.
    pub command_classes: Vec<u8>,
    /// True when the node keeps its receiver on (mains-powered, always-listening).
    pub is_listening: bool,
    /// True when the node is currently reachable.
    pub online: bool,
}

impl ZWaveNode {
    /// Create a new node record with only the `node_id` set; all other fields
    /// default to empty / zero / `Unknown` until populated by discovery.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            name: None,
            manufacturer_id: 0,
            product_type: 0,
            product_id: 0,
            kind: ZWaveNodeKind::Unknown,
            capabilities: Vec::new(),
            command_classes: Vec::new(),
            is_listening: false,
            online: false,
        }
    }
}

/// Z-Wave Z/IP or Serial API frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Host → controller command (`0x00`).
    Request = 0x00,
    /// Controller → host reply (`0x01`).
    Response = 0x01,
}
