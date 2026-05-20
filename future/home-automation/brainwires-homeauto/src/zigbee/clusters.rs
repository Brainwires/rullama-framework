//! High-level helpers for the most common Zigbee clusters.
//!
//! Each function returns the `(cluster_id, command_id, payload)` tuple ready
//! to pass to `ZigbeeCoordinator::invoke_command`, or `(cluster_id, attr_id)`
//! for attribute reads/writes. Constants match the ZCL Library Specification.

use super::super::types::AttributeValue;
use super::types::cluster_id;

// ── On/Off cluster (0x0006) ──────────────────────────────────────────────────

/// `0x00` — OnOff cluster OFF command.
pub const CMD_ON_OFF_OFF: u8 = 0x00;
/// `0x01` — OnOff cluster ON command.
pub const CMD_ON_OFF_ON: u8 = 0x01;
/// `0x02` — OnOff cluster TOGGLE command.
pub const CMD_ON_OFF_TOGGLE: u8 = 0x02;

/// `0x0000` — OnOff attribute, current binary state.
pub const ATTR_ON_OFF_ON_OFF: u16 = 0x0000;

/// Returns (cluster, cmd, payload) to turn a device on or off.
pub fn on_off_command(on: bool) -> (u16, u8, Vec<u8>) {
    let cmd = if on { CMD_ON_OFF_ON } else { CMD_ON_OFF_OFF };
    (cluster_id::ON_OFF, cmd, vec![])
}

/// Returns (cluster, cmd, payload) to toggle a device.
pub fn toggle_command() -> (u16, u8, Vec<u8>) {
    (cluster_id::ON_OFF, CMD_ON_OFF_TOGGLE, vec![])
}

// ── Level Control cluster (0x0008) ───────────────────────────────────────────

/// `0x00` — Level Control Move-to-Level command.
pub const CMD_LEVEL_MOVE_TO_LEVEL: u8 = 0x00;
/// `0x04` — Level Control Move-to-Level (with OnOff) — turns fixture on if off.
pub const CMD_LEVEL_MOVE_TO_LEVEL_ON_OFF: u8 = 0x04;

/// `0x0000` — Current Level attribute (0..=254).
pub const ATTR_LEVEL_CURRENT_LEVEL: u16 = 0x0000;

/// Returns (cluster, cmd, payload) to move to a level (0–254) with a transition time
/// in tenths of a second. Pass `with_on_off = true` to also turn on if off.
pub fn move_to_level(level: u8, transition_time_ds: u16, with_on_off: bool) -> (u16, u8, Vec<u8>) {
    let cmd = if with_on_off {
        CMD_LEVEL_MOVE_TO_LEVEL_ON_OFF
    } else {
        CMD_LEVEL_MOVE_TO_LEVEL
    };
    let mut payload = vec![level];
    payload.extend_from_slice(&transition_time_ds.to_le_bytes());
    (cluster_id::LEVEL_CONTROL, cmd, payload)
}

// ── Color Control cluster (0x0300) ───────────────────────────────────────────

/// `0x06` — Color Control Move-to-Hue-and-Saturation command.
pub const CMD_COLOR_MOVE_TO_HUE_SAT: u8 = 0x06;
/// `0x0A` — Color Control Move-to-Color-Temperature command.
pub const CMD_COLOR_MOVE_TO_COLOR_TEMP: u8 = 0x0A;

/// `0x0000` — Current Hue attribute.
pub const ATTR_COLOR_CURRENT_HUE: u16 = 0x0000;
/// `0x0001` — Current Saturation attribute.
pub const ATTR_COLOR_CURRENT_SAT: u16 = 0x0001;
/// `0x0007` — Color Temperature (mireds) attribute.
pub const ATTR_COLOR_COLOR_TEMP: u16 = 0x0007;
/// `0x0008` — Color Mode attribute (HSV vs CIE vs color-temp).
pub const ATTR_COLOR_COLOR_MODE: u16 = 0x0008;

/// Returns (cluster, cmd, payload) to move to hue + saturation.
pub fn move_to_hue_sat(hue: u8, sat: u8, transition_time_ds: u16) -> (u16, u8, Vec<u8>) {
    let mut payload = vec![hue, sat];
    payload.extend_from_slice(&transition_time_ds.to_le_bytes());
    (
        cluster_id::COLOR_CONTROL,
        CMD_COLOR_MOVE_TO_HUE_SAT,
        payload,
    )
}

/// Returns (cluster, cmd, payload) to move to a color temperature in mireds (153–500 typical).
pub fn move_to_color_temp(mireds: u16, transition_time_ds: u16) -> (u16, u8, Vec<u8>) {
    let mut payload = Vec::new();
    payload.extend_from_slice(&mireds.to_le_bytes());
    payload.extend_from_slice(&transition_time_ds.to_le_bytes());
    (
        cluster_id::COLOR_CONTROL,
        CMD_COLOR_MOVE_TO_COLOR_TEMP,
        payload,
    )
}

// ── Temperature Measurement cluster (0x0402) ─────────────────────────────────

/// `0x0000` — Measured Temperature attribute (signed, 0.01°C units).
pub const ATTR_TEMP_MEASURED_VALUE: u16 = 0x0000;
/// `0x0001` — Minimum measurable temperature (0.01°C units).
pub const ATTR_TEMP_MIN_MEASURED: u16 = 0x0001;
/// `0x0002` — Maximum measurable temperature (0.01°C units).
pub const ATTR_TEMP_MAX_MEASURED: u16 = 0x0002;
/// `0x0003` — Sensor tolerance (0.01°C units).
pub const ATTR_TEMP_TOLERANCE: u16 = 0x0003;

/// Decode a raw ZCL temperature attribute value (0.01 °C units, signed) to f32 °C.
pub fn decode_temperature(raw: &AttributeValue) -> Option<f32> {
    match raw {
        AttributeValue::I16(v) => {
            if *v == i16::MIN {
                None // 0x8000 = invalid
            } else {
                Some(*v as f32 / 100.0)
            }
        }
        AttributeValue::U16(v) => Some(*v as i16 as f32 / 100.0),
        _ => None,
    }
}

// ── Humidity Measurement cluster (0x0405) ─────────────────────────────────────

/// `0x0000` — Measured Relative Humidity attribute (unsigned, 0.01% units).
pub const ATTR_HUMIDITY_MEASURED_VALUE: u16 = 0x0000;

/// Decode a raw ZCL humidity value (0.01 % units, unsigned) to f32 %.
pub fn decode_humidity(raw: &AttributeValue) -> Option<f32> {
    match raw {
        AttributeValue::U16(v) => {
            if *v > 10_000 {
                None // invalid
            } else {
                Some(*v as f32 / 100.0)
            }
        }
        _ => None,
    }
}

// ── Occupancy Sensing cluster (0x0406) ───────────────────────────────────────

/// `0x0000` — Occupancy bitmap (bit 0 = occupied).
pub const ATTR_OCCUPANCY_OCCUPANCY: u16 = 0x0000;

// ── Door Lock cluster (0x0101) ───────────────────────────────────────────────

/// `0x00` — Door Lock LOCK command.
pub const CMD_DOOR_LOCK_LOCK: u8 = 0x00;
/// `0x01` — Door Lock UNLOCK command.
pub const CMD_DOOR_LOCK_UNLOCK: u8 = 0x01;
/// `0x0000` — LockState attribute (0=not-fully-locked, 1=locked, 2=unlocked).
pub const ATTR_DOOR_LOCK_STATE: u16 = 0x0000;

/// Returns (cluster, cmd, payload) to lock or unlock a door lock.
/// Pass an optional PIN code (as ASCII bytes). May be empty if PIN not required.
pub fn door_lock_command(lock: bool, pin: &[u8]) -> (u16, u8, Vec<u8>) {
    let cmd = if lock {
        CMD_DOOR_LOCK_LOCK
    } else {
        CMD_DOOR_LOCK_UNLOCK
    };
    // ZCL door lock command payload: PIN code string (length-prefixed octet string)
    let mut payload = vec![pin.len() as u8];
    payload.extend_from_slice(pin);
    (cluster_id::DOOR_LOCK, cmd, payload)
}

// ── IAS Zone cluster (0x0500) ────────────────────────────────────────────────

/// `0x0002` — ZoneStatus bitmap (alarm / tamper / battery / etc.).
pub const ATTR_IAS_ZONE_STATUS: u16 = 0x0002;
/// `0x0001` — ZoneType attribute (see `IAS_ZONE_TYPE_*`).
pub const ATTR_IAS_ZONE_TYPE: u16 = 0x0001;

/// IAS zone type: door / window contact (`0x0015`).
pub const IAS_ZONE_TYPE_CONTACT: u16 = 0x0015;
/// IAS zone type: motion / PIR (`0x000D`).
pub const IAS_ZONE_TYPE_MOTION: u16 = 0x000D;
/// IAS zone type: smoke detector (`0x0028`).
pub const IAS_ZONE_TYPE_SMOKE: u16 = 0x0028;
/// IAS zone type: carbon-monoxide sensor (`0x002B`).
pub const IAS_ZONE_TYPE_CO: u16 = 0x002B;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_off_command_on() {
        let (cl, cmd, payload) = on_off_command(true);
        assert_eq!(cl, cluster_id::ON_OFF);
        assert_eq!(cmd, CMD_ON_OFF_ON);
        assert!(payload.is_empty());
    }

    #[test]
    fn on_off_command_off() {
        let (cl, cmd, payload) = on_off_command(false);
        assert_eq!(cl, cluster_id::ON_OFF);
        assert_eq!(cmd, CMD_ON_OFF_OFF);
        assert!(payload.is_empty());
    }

    #[test]
    fn toggle_command_correct() {
        let (cl, cmd, _) = toggle_command();
        assert_eq!(cl, cluster_id::ON_OFF);
        assert_eq!(cmd, CMD_ON_OFF_TOGGLE);
    }

    #[test]
    fn move_to_level_payload() {
        let (cl, cmd, payload) = move_to_level(127, 10, false);
        assert_eq!(cl, cluster_id::LEVEL_CONTROL);
        assert_eq!(cmd, CMD_LEVEL_MOVE_TO_LEVEL);
        assert_eq!(payload, vec![127, 10, 0]);
    }

    #[test]
    fn move_to_level_with_on_off() {
        let (_, cmd, _) = move_to_level(200, 5, true);
        assert_eq!(cmd, CMD_LEVEL_MOVE_TO_LEVEL_ON_OFF);
    }

    #[test]
    fn decode_temperature_valid() {
        // 2350 = 23.50°C
        assert_eq!(decode_temperature(&AttributeValue::I16(2350)), Some(23.5));
    }

    #[test]
    fn decode_temperature_invalid() {
        assert_eq!(decode_temperature(&AttributeValue::I16(i16::MIN)), None);
    }

    #[test]
    fn decode_humidity_valid() {
        // 4500 = 45.00%
        assert_eq!(decode_humidity(&AttributeValue::U16(4500)), Some(45.0));
    }

    #[test]
    fn decode_humidity_out_of_range() {
        assert_eq!(decode_humidity(&AttributeValue::U16(10_001)), None);
    }

    #[test]
    fn door_lock_lock_no_pin() {
        let (cl, cmd, payload) = door_lock_command(true, &[]);
        assert_eq!(cl, cluster_id::DOOR_LOCK);
        assert_eq!(cmd, CMD_DOOR_LOCK_LOCK);
        assert_eq!(payload, vec![0x00]); // length = 0
    }

    #[test]
    fn move_to_color_temp_payload() {
        let (cl, cmd, payload) = move_to_color_temp(300, 20);
        assert_eq!(cl, cluster_id::COLOR_CONTROL);
        assert_eq!(cmd, CMD_COLOR_MOVE_TO_COLOR_TEMP);
        // mireds 300 LE + transition 20 LE
        assert_eq!(payload, vec![0x2C, 0x01, 0x14, 0x00]);
    }
}
