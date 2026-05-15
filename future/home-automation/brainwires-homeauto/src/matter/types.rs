use super::super::types::Capability;
use serde::{Deserialize, Serialize};

/// A commissioned Matter device on the fabric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatterDevice {
    /// 64-bit node ID assigned during commissioning.
    pub node_id: u64,
    /// Fabric index (0-based).
    pub fabric_index: u8,
    /// Human-readable name (optional, user-assigned).
    pub name: Option<String>,
    /// Vendor ID.
    pub vendor_id: u16,
    /// Product ID.
    pub product_id: u16,
    /// List of endpoints exposed by this device.
    pub endpoints: Vec<MatterEndpoint>,
    /// Whether the device is currently reachable.
    pub online: bool,
}

impl MatterDevice {
    /// New `MatterDevice` record with only `node_id` set; every other field
    /// defaults to 0 / empty / `false` until populated by discovery.
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            fabric_index: 0,
            name: None,
            vendor_id: 0,
            product_id: 0,
            endpoints: Vec::new(),
            online: false,
        }
    }
}

/// A Matter endpoint (logical device within a node).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatterEndpoint {
    /// Endpoint identifier (0..=65534; 0 is the root endpoint).
    pub endpoint_id: u16,
    /// Device type ID from the Matter spec (e.g. 0x0100 = On/Off Light).
    pub device_type: u32,
    /// Cluster IDs supported by this endpoint (server-side).
    pub clusters: Vec<u32>,
    /// High-level capabilities surfaced to framework consumers.
    pub capabilities: Vec<Capability>,
}

/// Configuration for a [`MatterDeviceServer`](super::MatterDeviceServer) instance.
///
/// Use [`MatterDeviceConfig::builder`] for ergonomic construction.
#[derive(Debug, Clone)]
pub struct MatterDeviceConfig {
    /// Device name as advertised over mDNS and in the Basic Information cluster.
    pub device_name: String,
    /// Vendor ID (0xFFF1 = test/development).
    pub vendor_id: u16,
    /// Product ID.
    pub product_id: u16,
    /// 12-bit discriminator (0–4095) used to identify the device during commissioning.
    pub discriminator: u16,
    /// SPAKE2+ commissioning passcode (PIN). Must not be a forbidden value.
    pub passcode: u32,
    /// Path to store persistent fabric data (certificates, node IDs, etc.).
    pub storage_path: std::path::PathBuf,
    /// UDP port to listen on (default: 5540, the standard Matter port).
    pub port: u16,
}

impl MatterDeviceConfig {
    /// Start an empty builder — apply the `device_name()` / `vendor_id()` etc.
    /// setters, then `build()` to materialize the config.
    pub fn builder() -> MatterDeviceConfigBuilder {
        MatterDeviceConfigBuilder::default()
    }
}

/// Builder for [`MatterDeviceConfig`] — every field has a sensible default so
/// only the ones you care about need to be set.
#[derive(Default)]
pub struct MatterDeviceConfigBuilder {
    device_name: Option<String>,
    vendor_id: Option<u16>,
    product_id: Option<u16>,
    discriminator: Option<u16>,
    passcode: Option<u32>,
    storage_path: Option<std::path::PathBuf>,
    port: Option<u16>,
}

impl MatterDeviceConfigBuilder {
    /// Device name advertised over mDNS and in Basic Information.
    pub fn device_name(mut self, name: impl Into<String>) -> Self {
        self.device_name = Some(name.into());
        self
    }
    /// Set Vendor ID (VID) — use `0xFFF1..=0xFFF4` for development.
    pub fn vendor_id(mut self, vid: u16) -> Self {
        self.vendor_id = Some(vid);
        self
    }
    /// Set Product ID (PID).
    pub fn product_id(mut self, pid: u16) -> Self {
        self.product_id = Some(pid);
        self
    }
    /// 12-bit discriminator (0..=4095) — extra bits are masked off.
    pub fn discriminator(mut self, d: u16) -> Self {
        self.discriminator = Some(d & 0x0FFF);
        self
    }
    /// SPAKE2+ passcode / setup PIN.
    pub fn passcode(mut self, p: u32) -> Self {
        self.passcode = Some(p);
        self
    }
    /// Directory where the fabric's operational credentials are persisted.
    pub fn storage_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.storage_path = Some(path.into());
        self
    }
    /// UDP port to bind (standard Matter port is 5540).
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }
    /// Materialize into a [`MatterDeviceConfig`], filling any unset field
    /// with its default (test VID/PID, discriminator 3840, passcode 20202021,
    /// storage under `/tmp/brainwires-matter`, port 5540).
    pub fn build(self) -> MatterDeviceConfig {
        MatterDeviceConfig {
            device_name: self
                .device_name
                .unwrap_or_else(|| "Brainwires Device".into()),
            vendor_id: self.vendor_id.unwrap_or(0xFFF1), // test VID
            product_id: self.product_id.unwrap_or(0x8001),
            discriminator: self.discriminator.unwrap_or(3840),
            passcode: self.passcode.unwrap_or(20202021),
            storage_path: self
                .storage_path
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp/brainwires-matter")),
            port: self.port.unwrap_or(5540),
        }
    }
}

// ── Well-known Matter device type IDs (Matter 1.3) ────────────────────────────

/// Matter device-type identifiers, per the Device Library spec (Matter 1.3).
pub mod device_type {
    /// `0x0100` — On/Off Light.
    pub const ON_OFF_LIGHT: u32 = 0x0100;
    /// `0x0101` — Dimmable Light.
    pub const DIMMABLE_LIGHT: u32 = 0x0101;
    /// `0x010C` — Color Temperature Light.
    pub const COLOR_TEMPERATURE_LIGHT: u32 = 0x010C;
    /// `0x010D` — Extended Color Light (HSV / CIE).
    pub const EXTENDED_COLOR_LIGHT: u32 = 0x010D;
    /// `0x010A` — On/Off Plug-in Unit.
    pub const ON_OFF_PLUG: u32 = 0x010A;
    /// `0x010B` — Dimmable Plug-in Unit.
    pub const DIMMABLE_PLUG: u32 = 0x010B;
    /// `0x0303` — Pump.
    pub const PUMP: u32 = 0x0303;
    /// `0x0301` — Thermostat.
    pub const THERMOSTAT: u32 = 0x0301;
    /// `0x002B` — Fan.
    pub const FAN: u32 = 0x002B;
    /// `0x0202` — Window Covering.
    pub const WINDOW_COVERING: u32 = 0x0202;
    /// `0x000A` — Door Lock.
    pub const DOOR_LOCK: u32 = 0x000A;
    /// `0x0107` — Occupancy Sensor.
    pub const OCCUPANCY_SENSOR: u32 = 0x0107;
    /// `0x0302` — Temperature Sensor.
    pub const TEMPERATURE_SENSOR: u32 = 0x0302;
    /// `0x0307` — Humidity Sensor.
    pub const HUMIDITY_SENSOR: u32 = 0x0307;
    /// `0x0106` — Light Sensor.
    pub const LIGHT_SENSOR: u32 = 0x0106;
    /// `0x0015` — Contact Sensor.
    pub const CONTACT_SENSOR: u32 = 0x0015;
    /// `0x0306` — Flow Sensor.
    pub const FLOW_SENSOR: u32 = 0x0306;
    /// `0x0305` — Pressure Sensor.
    pub const PRESSURE_SENSOR: u32 = 0x0305;
    /// `0x050C` — EV Supply Equipment (charger, Matter 1.3).
    pub const EV_CHARGER: u32 = 0x050C;
}

// ── Well-known Matter cluster IDs (Matter 1.3) ────────────────────────────────

/// Matter cluster identifiers, per the Application Clusters spec (Matter 1.3).
pub mod cluster_id {
    // Foundation
    /// `0x0028` — Basic Information cluster.
    pub const BASIC_INFORMATION: u32 = 0x0028;
    /// `0x0029` — OTA Software Update Provider.
    pub const OTA_SOFTWARE_UPDATE: u32 = 0x0029;
    /// `0x0030` — General Commissioning cluster.
    pub const GENERAL_COMMISSIONING: u32 = 0x0030;
    /// `0x0031` — Network Commissioning cluster (WiFi / Thread / Ethernet).
    pub const NETWORK_COMMISSIONING: u32 = 0x0031;
    /// `0x0032` — Diagnostic Logs cluster.
    pub const DIAGNOSTIC_LOGS: u32 = 0x0032;
    /// `0x0033` — General Diagnostics cluster.
    pub const GENERAL_DIAGNOSTICS: u32 = 0x0033;
    /// `0x003E` — Operational Credentials (NOC, root CA, fabric table).
    pub const OPERATIONAL_CREDENTIALS: u32 = 0x003E;
    /// `0x003E` — Alias for `OPERATIONAL_CREDENTIALS` (legacy spec name).
    pub const NODE_OPERATIONAL_CREDENTIALS: u32 = 0x003E;
    /// `0x0040` — Fixed Label cluster.
    pub const FIXED_LABEL: u32 = 0x0040;

    // Device capabilities
    /// `0x0003` — Identify cluster.
    pub const IDENTIFY: u32 = 0x0003;
    /// `0x0004` — Groups cluster.
    pub const GROUPS: u32 = 0x0004;
    /// `0x0005` — Scenes cluster.
    pub const SCENES: u32 = 0x0005;
    /// `0x0006` — On/Off cluster.
    pub const ON_OFF: u32 = 0x0006;
    /// `0x0008` — Level Control cluster.
    pub const LEVEL_CONTROL: u32 = 0x0008;
    /// `0x001D` — Descriptor cluster (endpoint composition).
    pub const DESCRIPTOR: u32 = 0x001D;
    /// `0x001E` — Binding cluster.
    pub const BINDING: u32 = 0x001E;

    // Color
    /// `0x0300` — Color Control cluster.
    pub const COLOR_CONTROL: u32 = 0x0300;

    // Window covering
    /// `0x0102` — Window Covering cluster.
    pub const WINDOW_COVERING: u32 = 0x0102;

    // HVAC
    /// `0x0201` — Thermostat cluster.
    pub const THERMOSTAT: u32 = 0x0201;
    /// `0x0204` — Thermostat User Interface Configuration.
    pub const THERMOSTAT_UI_CONFIG: u32 = 0x0204;
    /// `0x0202` — Fan Control cluster.
    pub const FAN_CONTROL: u32 = 0x0202;

    // Security
    /// `0x0101` — Door Lock cluster.
    pub const DOOR_LOCK: u32 = 0x0101;

    // Sensors
    /// `0x0402` — Temperature Measurement cluster.
    pub const TEMPERATURE_MEASUREMENT: u32 = 0x0402;
    /// `0x0405` — Relative Humidity Measurement cluster.
    pub const RELATIVE_HUMIDITY: u32 = 0x0405;
    /// `0x0406` — Occupancy Sensing cluster.
    pub const OCCUPANCY_SENSING: u32 = 0x0406;
    /// `0x0400` — Illuminance Measurement cluster.
    pub const ILLUMINANCE_MEASUREMENT: u32 = 0x0400;
    /// `0x0403` — Pressure Measurement cluster.
    pub const PRESSURE_MEASUREMENT: u32 = 0x0403;
    /// `0x0404` — Flow Measurement cluster.
    pub const FLOW_MEASUREMENT: u32 = 0x0404;

    // Energy (Matter 1.3)
    /// `0x0B04` — Electrical Measurement cluster.
    pub const ELECTRICAL_MEASUREMENT: u32 = 0x0B04;
    /// `0x002F` — Power Source cluster (battery / mains).
    pub const POWER_SOURCE: u32 = 0x002F;
    /// `0x0099` — EV Charging cluster (Matter 1.3).
    pub const EV_CHARGING: u32 = 0x0099;
}
