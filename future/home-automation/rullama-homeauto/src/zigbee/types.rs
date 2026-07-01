use super::super::types::Capability;
use serde::{Deserialize, Serialize};

/// 64-bit IEEE (EUI-64) extended address.
pub type IeeeAddr = u64;
/// 16-bit network (short) address.
pub type NwkAddr = u16;

/// A Zigbee device address — may be addressed by either form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZigbeeAddr {
    /// 64-bit IEEE extended address (EUI-64) — permanent per-device identifier.
    pub ieee: IeeeAddr,
    /// 16-bit short network address — reassigned each time the device rejoins.
    pub nwk: NwkAddr,
}

impl ZigbeeAddr {
    /// Build an address from an IEEE/network pair.
    pub fn new(ieee: IeeeAddr, nwk: NwkAddr) -> Self {
        Self { ieee, nwk }
    }
}

impl std::fmt::Display for ZigbeeAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x} ({:#06x})", self.ieee, self.nwk)
    }
}

/// Standard Zigbee cluster IDs (ZCL Foundation, Zigbee 3.0).
///
/// Every constant here is a cluster ID defined by the ZCL Library
/// Specification — variant names correspond 1:1 to the spec.
#[allow(non_upper_case_globals)]
pub mod cluster_id {
    /// `0x0000` — Basic cluster (manufacturer, model, power source).
    pub const BASIC: u16 = 0x0000;
    /// `0x0001` — Power Configuration (battery level, voltage).
    pub const POWER_CONFIG: u16 = 0x0001;
    /// `0x0003` — Identify cluster (blink / beep for location).
    pub const IDENTIFY: u16 = 0x0003;
    /// `0x0004` — Groups cluster (multicast addressing).
    pub const GROUPS: u16 = 0x0004;
    /// `0x0005` — Scenes cluster (pre-programmed states).
    pub const SCENES: u16 = 0x0005;
    /// `0x0006` — On/Off cluster (binary actuators).
    pub const ON_OFF: u16 = 0x0006;
    /// `0x0007` — On/Off Switch Configuration.
    pub const ON_OFF_SWITCH_CONFIG: u16 = 0x0007;
    /// `0x0008` — Level Control (dimmers, position).
    pub const LEVEL_CONTROL: u16 = 0x0008;
    /// `0x0009` — Alarms cluster.
    pub const ALARMS: u16 = 0x0009;
    /// `0x000A` — Time cluster.
    pub const TIME: u16 = 0x000A;
    /// `0x0019` — Over-the-Air Upgrade cluster.
    pub const OTA_UPGRADE: u16 = 0x0019;
    /// `0x0101` — Door Lock cluster.
    pub const DOOR_LOCK: u16 = 0x0101;
    /// `0x0102` — Window Covering (blinds, shades).
    pub const WINDOW_COVERING: u16 = 0x0102;
    /// `0x0300` — Color Control (HSV, CIE, color temperature).
    pub const COLOR_CONTROL: u16 = 0x0300;
    /// `0x0400` — Illuminance Measurement.
    pub const ILLUMINANCE: u16 = 0x0400;
    /// `0x0402` — Temperature Measurement.
    pub const TEMPERATURE: u16 = 0x0402;
    /// `0x0405` — Relative Humidity Measurement.
    pub const HUMIDITY: u16 = 0x0405;
    /// `0x0406` — Occupancy Sensing.
    pub const OCCUPANCY: u16 = 0x0406;
    /// `0x0500` — IAS Zone (security sensors).
    pub const IAS_ZONE: u16 = 0x0500;
    /// `0x0702` — Metering (smart meter readings).
    pub const METERING: u16 = 0x0702;
    /// `0x0B04` — Electrical Measurement (V / A / W / Hz).
    pub const ELECTRICAL_MEASUREMENT: u16 = 0x0B04;
}

/// Newtype for Zigbee cluster IDs.
pub type ZigbeeClusterId = u16;
/// Newtype for Zigbee attribute IDs.
pub type ZigbeeAttrId = u16;

/// Device kind inferred from ZDO Basic cluster `deviceType` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZigbeeDeviceKind {
    /// Binary on/off lighting fixture.
    Light,
    /// Dimmable white-only light.
    DimmableLight,
    /// Color-capable light (HSV / CIE / color-temp).
    ColorLight,
    /// Switch / remote / scene controller.
    Switch,
    /// Temperature sensor (reporting via cluster 0x0402).
    TemperatureSensor,
    /// Relative humidity sensor (cluster 0x0405).
    HumiditySensor,
    /// PIR / occupancy sensor (cluster 0x0406).
    OccupancySensor,
    /// Door lock actuator (cluster 0x0101).
    DoorLock,
    /// Thermostat (cluster 0x0201).
    Thermostat,
    /// Smart plug / outlet.
    PowerOutlet,
    /// Manufacturer-specific or unrecognised device type — holds the raw
    /// `deviceType` field for inspection.
    Other(u16),
}

/// A Zigbee end-device or router on the coordinator's network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZigbeeDevice {
    /// Addressing pair (IEEE + NWK) for this device.
    pub addr: ZigbeeAddr,
    /// User-assigned friendly name, when known.
    pub name: Option<String>,
    /// Manufacturer string from the Basic cluster.
    pub manufacturer: Option<String>,
    /// Model identifier from the Basic cluster.
    pub model: Option<String>,
    /// Inferred device kind.
    pub kind: ZigbeeDeviceKind,
    /// List of cluster IDs the device supports (server side).
    pub clusters: Vec<ZigbeeClusterId>,
    /// Abstract capabilities surfaced to the framework consumer.
    pub capabilities: Vec<Capability>,
    /// Whether the device is currently online/reachable.
    pub online: bool,
}

impl ZigbeeDevice {
    /// Build a new device record with only addressing + kind populated.
    pub fn new(addr: ZigbeeAddr, kind: ZigbeeDeviceKind) -> Self {
        Self {
            addr,
            name: None,
            manufacturer: None,
            model: None,
            kind,
            clusters: Vec::new(),
            capabilities: Vec::new(),
            online: true,
        }
    }
}
