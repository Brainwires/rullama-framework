use serde::{Deserialize, Serialize};

/// Which protocol a device speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    /// IEEE 802.15.4-based Zigbee (Cluster Library / HA profile).
    Zigbee,
    /// Sub-GHz proprietary Z-Wave.
    ZWave,
    /// IEEE 802.15.4-based Thread (Matter transport).
    Thread,
    /// Matter-over-Thread or Matter-over-WiFi.
    Matter,
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Protocol::Zigbee => write!(f, "Zigbee"),
            Protocol::ZWave => write!(f, "Z-Wave"),
            Protocol::Thread => write!(f, "Thread"),
            Protocol::Matter => write!(f, "Matter"),
        }
    }
}

/// High-level capability that a home device exposes.
///
/// Protocol-agnostic — individual backends map cluster/command IDs onto this
/// set so consumers don't need to know which protocol produced an event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Binary on/off control (switches, relays, plugs).
    OnOff,
    /// Continuous brightness 0..=100%.
    Dimming,
    /// Mireds-based white-temperature control.
    ColorTemperature,
    /// HSV or RGB color control.
    ColorRgb,
    /// Temperature sensor reading.
    Temperature,
    /// Humidity sensor reading.
    Humidity,
    /// Barometric pressure reading.
    Pressure,
    /// PIR / motion sensor presence bit.
    Motion,
    /// Door/window contact sensor.
    Contact,
    /// Electronic lock actuator.
    Lock,
    /// Heating/cooling setpoint control.
    Thermostat,
    /// Per-outlet energy monitoring (kWh, W, V, A).
    EnergyMonitoring,
    /// Blinds, shades, garage doors.
    WindowCovering,
    /// Manufacturer-specific capability — free-form label.
    Custom(String),
}

/// A unified home automation device record (protocol-agnostic view).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeDevice {
    /// Protocol-specific unique identifier (IEEE address, node ID, etc.).
    pub id: String,
    /// User-assigned friendly name, when known.
    pub name: Option<String>,
    /// Which protocol this device speaks.
    pub protocol: Protocol,
    /// Manufacturer string reported by the device, when available.
    pub manufacturer: Option<String>,
    /// Model identifier reported by the device.
    pub model: Option<String>,
    /// Current firmware revision string.
    pub firmware_version: Option<String>,
    /// Capabilities the device exposes (populated during discovery).
    pub capabilities: Vec<Capability>,
}

impl HomeDevice {
    /// Construct a new device record with only the required `id` + `protocol`
    /// set. All metadata fields default to `None` / empty.
    pub fn new(id: impl Into<String>, protocol: Protocol) -> Self {
        Self {
            id: id.into(),
            name: None,
            protocol,
            manufacturer: None,
            model: None,
            firmware_version: None,
            capabilities: Vec::new(),
        }
    }
}

/// Typed value returned from an attribute read or carried in an event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    /// Boolean attribute (e.g. on/off, occupancy).
    Bool(bool),
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 8-bit integer.
    I8(i8),
    /// Signed 16-bit integer (e.g. temperature in 0.01°C).
    I16(i16),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// UTF-8 string attribute.
    String(String),
    /// Opaque byte buffer (rendered as lowercase hex by `Display`).
    Bytes(Vec<u8>),
    /// Attribute is reported by the device but carries no value.
    Null,
}

impl std::fmt::Display for AttributeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttributeValue::Bool(v) => write!(f, "{v}"),
            AttributeValue::U8(v) => write!(f, "{v}"),
            AttributeValue::U16(v) => write!(f, "{v}"),
            AttributeValue::U32(v) => write!(f, "{v}"),
            AttributeValue::U64(v) => write!(f, "{v}"),
            AttributeValue::I8(v) => write!(f, "{v}"),
            AttributeValue::I16(v) => write!(f, "{v}"),
            AttributeValue::I32(v) => write!(f, "{v}"),
            AttributeValue::F32(v) => write!(f, "{v}"),
            AttributeValue::F64(v) => write!(f, "{v}"),
            AttributeValue::String(v) => write!(f, "{v}"),
            AttributeValue::Bytes(v) => write!(f, "0x{}", hex_encode(v)),
            AttributeValue::Null => write!(f, "null"),
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Events emitted by any home automation hub/coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HomeAutoEvent {
    /// A new device joined the network.
    DeviceJoined(HomeDevice),

    /// A device left or was removed from the network.
    DeviceLeft {
        /// Protocol-specific identifier of the departing device.
        id: String,
        /// Which network the device left.
        protocol: Protocol,
    },

    /// An attribute value changed (e.g. temperature sensor update, switch toggled).
    AttributeChanged {
        /// Protocol-specific identifier of the reporting device.
        device_id: String,
        /// Which protocol produced the report.
        protocol: Protocol,
        /// Human-readable cluster name or hex ID.
        cluster: String,
        /// Human-readable attribute name or hex ID.
        attribute: String,
        /// New attribute value.
        value: AttributeValue,
    },

    /// A command was successfully sent to a device.
    CommandSent {
        /// Protocol-specific identifier of the command target.
        device_id: String,
        /// Which protocol carried the command.
        protocol: Protocol,
        /// Human-readable cluster name or hex ID.
        cluster: String,
        /// Human-readable command name or hex ID.
        command: String,
    },
}
