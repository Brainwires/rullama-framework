//! Z-Wave Command Class encoding.
//!
//! Implements Z-Wave Plus v2 (Specification 7.x) command class encoding for
//! the most common device interactions. Variant values are the on-wire
//! command-class ID bytes defined by the Z-Wave spec.

/// Z-Wave command-class identifier.
///
/// Each variant maps to the spec-defined command-class ID byte. `Unknown(id)`
/// is the pass-through for command classes not yet modelled by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CommandClass {
    /// `0x20` — BASIC. Generic on/off/level fallback supported by most nodes.
    Basic = 0x20,
    /// `0x25` — SWITCH_BINARY. Binary on/off switches.
    SwitchBinary = 0x25,
    /// `0x26` — SWITCH_MULTILEVEL. Dimmer / percentage-position devices.
    SwitchMultilevel = 0x26,
    /// `0x27` — SWITCH_ALL. Broadcast all-on / all-off.
    SwitchAll = 0x27,
    /// `0x33` — SWITCH_COLOR. Colored light control.
    SwitchColor = 0x33,
    /// `0x30` — SENSOR_BINARY. Binary trip sensors (door, motion).
    SensorBinary = 0x30,
    /// `0x31` — SENSOR_MULTILEVEL. Multi-sensor readings (temp, humidity…).
    SensorMultilevel = 0x31,
    /// `0x32` — METER. Energy / water / gas meters.
    Meter = 0x32,
    /// `0x71` — NOTIFICATION (formerly ALARM).
    Notification = 0x71,
    /// `0x40` — THERMOSTAT_MODE.
    ThermostatMode = 0x40,
    /// `0x43` — THERMOSTAT_SETPOINT.
    ThermostatSetpoint = 0x43,
    /// `0x44` — THERMOSTAT_FAN_MODE.
    ThermostatFanMode = 0x44,
    /// `0x42` — THERMOSTAT_OPERATING_STATE.
    ThermostatOperatingState = 0x42,
    /// `0x62` — DOOR_LOCK.
    DoorLock = 0x62,
    /// `0x63` — USER_CODE (door-lock PINs).
    UserCode = 0x63,
    /// `0x80` — BATTERY level reporting.
    Battery = 0x80,
    /// `0x85` — ASSOCIATION. Direct node→node grouping.
    Association = 0x85,
    /// `0x8E` — MULTI_CHANNEL_ASSOCIATION.
    MultiChannelAssociation = 0x8E,
    /// `0x70` — CONFIGURATION. Manufacturer-defined parameters.
    Configuration = 0x70,
    /// `0x86` — VERSION. Firmware / command-class version reporting.
    Version = 0x86,
    /// `0x72` — MANUFACTURER_SPECIFIC.
    ManufacturerSpecific = 0x72,
    /// `0x84` — WAKE_UP. Battery-powered polling schedule.
    WakeUp = 0x84,
    /// `0x98` — SECURITY (S0).
    Security = 0x98,
    /// `0x9F` — SECURITY_2 (S2).
    Security2 = 0x9F,
    /// `0x55` — TRANSPORT_SERVICE. Segmentation for large payloads.
    TransportService = 0x55,
    /// `0x5E` — ZWAVEPLUS_INFO. Z-Wave Plus identifying metadata.
    ZWavePlusInfo = 0x5E,
    /// `0x60` — MULTI_CHANNEL. Endpoint-addressed sub-devices.
    MultiChannel = 0x60,
    /// Any command-class ID not modelled above.
    Unknown(u8),
}

impl CommandClass {
    /// Return the on-wire command-class ID byte for this variant.
    pub fn id(&self) -> u8 {
        match self {
            Self::Basic => 0x20,
            Self::SwitchBinary => 0x25,
            Self::SwitchMultilevel => 0x26,
            Self::SwitchAll => 0x27,
            Self::SwitchColor => 0x33,
            Self::SensorBinary => 0x30,
            Self::SensorMultilevel => 0x31,
            Self::Meter => 0x32,
            Self::Notification => 0x71,
            Self::ThermostatMode => 0x40,
            Self::ThermostatSetpoint => 0x43,
            Self::ThermostatFanMode => 0x44,
            Self::ThermostatOperatingState => 0x42,
            Self::DoorLock => 0x62,
            Self::UserCode => 0x63,
            Self::Battery => 0x80,
            Self::Association => 0x85,
            Self::MultiChannelAssociation => 0x8E,
            Self::Configuration => 0x70,
            Self::Version => 0x86,
            Self::ManufacturerSpecific => 0x72,
            Self::WakeUp => 0x84,
            Self::Security => 0x98,
            Self::Security2 => 0x9F,
            Self::TransportService => 0x55,
            Self::ZWavePlusInfo => 0x5E,
            Self::MultiChannel => 0x60,
            Self::Unknown(id) => *id,
        }
    }

    /// Parse a wire command-class ID byte into a typed variant; unknown codes
    /// map to `Unknown(id)`.
    pub fn from_id(id: u8) -> Self {
        match id {
            0x20 => Self::Basic,
            0x25 => Self::SwitchBinary,
            0x26 => Self::SwitchMultilevel,
            0x27 => Self::SwitchAll,
            0x33 => Self::SwitchColor,
            0x30 => Self::SensorBinary,
            0x31 => Self::SensorMultilevel,
            0x32 => Self::Meter,
            0x71 => Self::Notification,
            0x40 => Self::ThermostatMode,
            0x43 => Self::ThermostatSetpoint,
            0x44 => Self::ThermostatFanMode,
            0x42 => Self::ThermostatOperatingState,
            0x62 => Self::DoorLock,
            0x63 => Self::UserCode,
            0x80 => Self::Battery,
            0x85 => Self::Association,
            0x8E => Self::MultiChannelAssociation,
            0x70 => Self::Configuration,
            0x86 => Self::Version,
            0x72 => Self::ManufacturerSpecific,
            0x84 => Self::WakeUp,
            0x98 => Self::Security,
            0x9F => Self::Security2,
            0x55 => Self::TransportService,
            0x5E => Self::ZWavePlusInfo,
            0x60 => Self::MultiChannel,
            other => Self::Unknown(other),
        }
    }
}

// ── Command encoders ─────────────────────────────────────────────────────────

/// Encode a SWITCH_BINARY_SET command. `value`: 0x00 = off, 0xFF = on.
pub fn switch_binary_set(on: bool) -> Vec<u8> {
    vec![
        CommandClass::SwitchBinary.id(),
        0x01,
        if on { 0xFF } else { 0x00 },
    ]
}

/// Encode a SWITCH_BINARY_GET command.
pub fn switch_binary_get() -> Vec<u8> {
    vec![CommandClass::SwitchBinary.id(), 0x02]
}

/// Encode a SWITCH_MULTILEVEL_SET command. `level`: 0–99, 0xFF = last non-zero.
/// `duration`: 0 = instant, 1–127 = seconds, 128–254 = minutes (value - 127).
pub fn switch_multilevel_set(level: u8, duration: u8) -> Vec<u8> {
    vec![
        CommandClass::SwitchMultilevel.id(),
        0x01,
        level.min(99),
        duration,
    ]
}

/// Encode a SWITCH_MULTILEVEL_GET command.
pub fn switch_multilevel_get() -> Vec<u8> {
    vec![CommandClass::SwitchMultilevel.id(), 0x02]
}

/// Encode a SENSOR_MULTILEVEL_GET command for a specific sensor type.
/// Common sensor types: 0x01=temperature, 0x03=luminance, 0x05=humidity.
pub fn sensor_multilevel_get(sensor_type: u8) -> Vec<u8> {
    vec![CommandClass::SensorMultilevel.id(), 0x04, sensor_type, 0x00]
}

/// Encode a THERMOSTAT_SETPOINT_SET command.
/// `setpoint_type`: 0x01=heating, 0x02=cooling.
/// `value_celsius`: temperature as signed 16-bit integer in 0.1 °C units (e.g. 215 = 21.5°C).
pub fn thermostat_setpoint_set(setpoint_type: u8, value_tenths_celsius: i16) -> Vec<u8> {
    // precision=1 (1 decimal), scale=0 (°C), size=2 (2 bytes)
    // reason: the `(0 << 3)` term is kept for symmetry with the precision/size
    // fields; it documents the scale slot in the bit-packed level byte.
    #[allow(clippy::identity_op)]
    let level = (1 << 5) | (0 << 3) | 2; // precision=1, scale=0, size=2
    let bytes = value_tenths_celsius.to_be_bytes();
    vec![
        CommandClass::ThermostatSetpoint.id(),
        0x01, // SET
        setpoint_type,
        level,
        bytes[0],
        bytes[1],
    ]
}

/// Encode a THERMOSTAT_SETPOINT_GET command.
pub fn thermostat_setpoint_get(setpoint_type: u8) -> Vec<u8> {
    vec![CommandClass::ThermostatSetpoint.id(), 0x02, setpoint_type]
}

/// Encode a DOOR_LOCK_OPERATION_SET. `mode`: 0x00=unsecured, 0xFF=secured.
pub fn door_lock_set(locked: bool) -> Vec<u8> {
    vec![
        CommandClass::DoorLock.id(),
        0x01, // DOOR_LOCK_OPERATION_SET
        if locked { 0xFF } else { 0x00 },
    ]
}

/// Encode a DOOR_LOCK_OPERATION_GET.
pub fn door_lock_get() -> Vec<u8> {
    vec![CommandClass::DoorLock.id(), 0x02]
}

/// Encode a BATTERY_GET command.
pub fn battery_get() -> Vec<u8> {
    vec![CommandClass::Battery.id(), 0x02]
}

/// Encode a CONFIGURATION_SET command (Z-Wave parameter).
/// `param_no`: parameter number, `value`: parameter value (1–4 bytes, big-endian).
pub fn configuration_set(param_no: u8, value: i32, size: u8) -> Vec<u8> {
    let mut buf = vec![CommandClass::Configuration.id(), 0x04, param_no, size];
    match size {
        1 => buf.push(value as u8),
        2 => buf.extend_from_slice(&(value as i16).to_be_bytes()),
        4 => buf.extend_from_slice(&value.to_be_bytes()),
        _ => buf.push(value as u8),
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_class_binary_switch_set_on() {
        let cmd = switch_binary_set(true);
        assert_eq!(cmd, vec![0x25, 0x01, 0xFF]);
    }

    #[test]
    fn command_class_binary_switch_set_off() {
        let cmd = switch_binary_set(false);
        assert_eq!(cmd, vec![0x25, 0x01, 0x00]);
    }

    #[test]
    fn command_class_binary_switch_get() {
        assert_eq!(switch_binary_get(), vec![0x25, 0x02]);
    }

    #[test]
    fn command_class_multilevel_set_50_percent() {
        let cmd = switch_multilevel_set(50, 0);
        assert_eq!(cmd, vec![0x26, 0x01, 50, 0]);
    }

    #[test]
    fn command_class_multilevel_set_clamps_at_99() {
        let cmd = switch_multilevel_set(100, 0);
        assert_eq!(cmd[2], 99);
    }

    #[test]
    fn command_class_sensor_multilevel_get() {
        let cmd = sensor_multilevel_get(0x01); // temperature
        assert_eq!(cmd[0], CommandClass::SensorMultilevel.id());
        assert_eq!(cmd[2], 0x01);
    }

    #[test]
    fn command_class_thermostat_setpoint_set_heating() {
        // 21.5°C → 215 in 0.1°C units
        let cmd = thermostat_setpoint_set(0x01, 215);
        assert_eq!(cmd[0], CommandClass::ThermostatSetpoint.id());
        assert_eq!(cmd[1], 0x01); // SET
        assert_eq!(cmd[2], 0x01); // heating setpoint
        // value = 215 as big-endian i16
        let val = i16::from_be_bytes([cmd[4], cmd[5]]);
        assert_eq!(val, 215);
    }

    #[test]
    fn command_class_door_lock_set_locked() {
        let cmd = door_lock_set(true);
        assert_eq!(cmd, vec![0x62, 0x01, 0xFF]);
    }

    #[test]
    fn command_class_door_lock_set_unlocked() {
        let cmd = door_lock_set(false);
        assert_eq!(cmd, vec![0x62, 0x01, 0x00]);
    }

    #[test]
    fn command_class_unknown_passthrough() {
        let cc = CommandClass::from_id(0xEE);
        assert!(matches!(cc, CommandClass::Unknown(0xEE)));
        assert_eq!(cc.id(), 0xEE);
    }

    #[test]
    fn command_class_roundtrip_known() {
        for id in [0x25u8, 0x26, 0x31, 0x62, 0x80] {
            let cc = CommandClass::from_id(id);
            assert_eq!(cc.id(), id, "roundtrip failed for {id:#04x}");
        }
    }
}
