/// Typed cluster helpers for Matter 1.3 (TLV-encoded command/attribute payloads).
///
/// The Matter interaction model uses TLV (Tag-Length-Value) encoding for all payloads.
/// These helpers produce the TLV bytes for the most common cluster interactions.
///
/// TLV encoding reference: Matter spec §A.7 (TLV Format).
use super::types::cluster_id;

// ── TLV primitives ────────────────────────────────────────────────────────────

/// Matter TLV element types (control byte upper nibble).
#[allow(dead_code)]
pub(super) mod tlv {
    pub const TYPE_SIGNED_INT_1: u8 = 0x00;
    pub const TYPE_SIGNED_INT_2: u8 = 0x01;
    pub const TYPE_SIGNED_INT_4: u8 = 0x02;
    pub const TYPE_UNSIGNED_INT_1: u8 = 0x04;
    pub const TYPE_UNSIGNED_INT_2: u8 = 0x05;
    pub const TYPE_UNSIGNED_INT_4: u8 = 0x06;
    pub const TYPE_BOOL_FALSE: u8 = 0x08;
    pub const TYPE_BOOL_TRUE: u8 = 0x09;
    pub const TYPE_NULL: u8 = 0x14;
    pub const TYPE_STRUCTURE: u8 = 0x15;
    /// TLV array (ordered, anonymous-tagged elements).
    pub const TYPE_ARRAY: u8 = 0x16;
    /// TLV list (elements may carry context tags).
    pub const TYPE_LIST: u8 = 0x17;
    pub const TYPE_END_OF_CONTAINER: u8 = 0x18;

    pub const TAG_ANONYMOUS: u8 = 0x00; // anonymous (no tag)
    pub const TAG_CONTEXT_1: u8 = 0x20; // context-specific 1-byte tag
}

pub(super) fn tlv_uint8(tag: u8, val: u8) -> Vec<u8> {
    vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1, tag, val]
}

pub(super) fn tlv_uint16(tag: u8, val: u16) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_2, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

pub(super) fn tlv_uint32(tag: u8, val: u32) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_4, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

pub(super) fn tlv_bool(tag: u8, val: bool) -> Vec<u8> {
    let ty = if val {
        tlv::TYPE_BOOL_TRUE
    } else {
        tlv::TYPE_BOOL_FALSE
    };
    vec![tlv::TAG_CONTEXT_1 | ty, tag]
}

pub(super) fn tlv_null(tag: u8) -> Vec<u8> {
    vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_NULL, tag]
}

pub(super) fn wrap_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_STRUCTURE];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

/// Wrap `inner` bytes in a context-tagged TLV structure: `{ tag: struct { inner } }`.
pub(super) fn wrap_struct_tagged(tag: u8, inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE, tag];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

/// Wrap `inner` bytes in a context-tagged TLV list: `{ tag: list { inner } }`.
pub(super) fn wrap_list_tagged(tag: u8, inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_LIST, tag];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

/// Read a little-endian u16 from `bytes` at `offset`. Returns `(value, offset+2)`.
pub(super) fn read_u16_le(bytes: &[u8], offset: usize) -> Option<(u16, usize)> {
    if offset + 2 > bytes.len() {
        return None;
    }
    let v = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    Some((v, offset + 2))
}

/// Read a little-endian u32 from `bytes` at `offset`. Returns `(value, offset+4)`.
pub(super) fn read_u32_le(bytes: &[u8], offset: usize) -> Option<(u32, usize)> {
    if offset + 4 > bytes.len() {
        return None;
    }
    let v = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    Some((v, offset + 4))
}

// ── Path types for Interaction Model ─────────────────────────────────────────

/// An attribute path used in Read, Write, Subscribe, and Report interactions.
///
/// Any field may be `None` to indicate a wildcard.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributePath {
    /// Endpoint identifier (`None` = wildcard).
    pub endpoint_id: Option<u16>,
    /// Cluster identifier (`None` = wildcard).
    pub cluster_id: Option<u32>,
    /// Attribute identifier (`None` = wildcard).
    pub attribute_id: Option<u32>,
}

impl AttributePath {
    /// Construct a fully-specified (non-wildcard) attribute path.
    pub fn specific(endpoint_id: u16, cluster_id: u32, attribute_id: u32) -> Self {
        Self {
            endpoint_id: Some(endpoint_id),
            cluster_id: Some(cluster_id),
            attribute_id: Some(attribute_id),
        }
    }

    /// Construct a fully-wildcard attribute path.
    pub fn wildcard() -> Self {
        Self {
            endpoint_id: None,
            cluster_id: None,
            attribute_id: None,
        }
    }

    /// TLV-encode as a struct with context tags:
    /// tag 2 = endpoint_id (uint16), tag 3 = cluster_id (uint32), tag 4 = attribute_id (uint32).
    /// Missing fields are omitted (wildcard).
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        if let Some(ep) = self.endpoint_id {
            inner.extend_from_slice(&tlv_uint16(2, ep));
        }
        if let Some(cl) = self.cluster_id {
            inner.extend_from_slice(&tlv_uint32(3, cl));
        }
        if let Some(attr) = self.attribute_id {
            inner.extend_from_slice(&tlv_uint32(4, attr));
        }
        wrap_struct(&inner)
    }

    /// Decode an `AttributePath` from TLV bytes produced by [`AttributePath::encode`].
    ///
    /// Returns `None` if the bytes are malformed.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        // Expect: TYPE_STRUCTURE (0x15) ... END_OF_CONTAINER (0x18)
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return None;
        }
        let mut endpoint_id = None;
        let mut cluster_id = None;
        let mut attribute_id = None;
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return None;
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F; // lower 5 bits = element type
            match (tag, type_bits) {
                (2, t) if t == tlv::TYPE_UNSIGNED_INT_2 => {
                    let (v, next) = read_u16_le(bytes, i)?;
                    endpoint_id = Some(v);
                    i = next;
                }
                (3, t) if t == tlv::TYPE_UNSIGNED_INT_4 => {
                    let (v, next) = read_u32_le(bytes, i)?;
                    cluster_id = Some(v);
                    i = next;
                }
                (4, t) if t == tlv::TYPE_UNSIGNED_INT_4 => {
                    let (v, next) = read_u32_le(bytes, i)?;
                    attribute_id = Some(v);
                    i = next;
                }
                _ => return None, // unknown field
            }
        }
        Some(Self {
            endpoint_id,
            cluster_id,
            attribute_id,
        })
    }
}

/// A command path used in Invoke interactions.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandPath {
    /// Endpoint identifier.
    pub endpoint_id: u16,
    /// Cluster identifier.
    pub cluster_id: u32,
    /// Command identifier.
    pub command_id: u32,
}

impl CommandPath {
    /// Construct a new `CommandPath`.
    pub fn new(endpoint_id: u16, cluster_id: u32, command_id: u32) -> Self {
        Self {
            endpoint_id,
            cluster_id,
            command_id,
        }
    }

    /// TLV-encode as a struct:
    /// tag 0 = endpoint_id (uint16), tag 1 = cluster_id (uint32), tag 2 = command_id (uint32).
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_uint16(0, self.endpoint_id));
        inner.extend_from_slice(&tlv_uint32(1, self.cluster_id));
        inner.extend_from_slice(&tlv_uint32(2, self.command_id));
        wrap_struct(&inner)
    }

    /// Decode a `CommandPath` from TLV bytes produced by [`CommandPath::encode`].
    ///
    /// Returns `None` if the bytes are malformed or required fields are missing.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return None;
        }
        let mut endpoint_id = None::<u16>;
        let mut cluster_id = None::<u32>;
        let mut command_id = None::<u32>;
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return None;
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;
            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_UNSIGNED_INT_2 => {
                    let (v, next) = read_u16_le(bytes, i)?;
                    endpoint_id = Some(v);
                    i = next;
                }
                (1, t) if t == tlv::TYPE_UNSIGNED_INT_4 => {
                    let (v, next) = read_u32_le(bytes, i)?;
                    cluster_id = Some(v);
                    i = next;
                }
                (2, t) if t == tlv::TYPE_UNSIGNED_INT_4 => {
                    let (v, next) = read_u32_le(bytes, i)?;
                    command_id = Some(v);
                    i = next;
                }
                _ => return None,
            }
        }
        Some(Self {
            endpoint_id: endpoint_id?,
            cluster_id: cluster_id?,
            command_id: command_id?,
        })
    }
}

// ── On/Off cluster (0x0006) ───────────────────────────────────────────────────

/// On/Off cluster (`0x0006`) — binary actuator control.
pub mod on_off {
    use super::*;
    /// Cluster ID for this module.
    pub const CLUSTER_ID: u32 = cluster_id::ON_OFF;

    /// `0x0000` — OnOff attribute (boolean current state).
    pub const ATTR_ON_OFF: u32 = 0x0000;

    /// `0x00` — Off command.
    pub const CMD_OFF: u32 = 0x00;
    /// `0x01` — On command.
    pub const CMD_ON: u32 = 0x01;
    /// `0x02` — Toggle command.
    pub const CMD_TOGGLE: u32 = 0x02;

    /// TLV payload for the On command (empty struct).
    pub fn on_tlv() -> Vec<u8> {
        wrap_struct(&[])
    }
    /// TLV payload for the Off command (empty struct).
    pub fn off_tlv() -> Vec<u8> {
        wrap_struct(&[])
    }
    /// TLV payload for the Toggle command (empty struct).
    pub fn toggle_tlv() -> Vec<u8> {
        wrap_struct(&[])
    }
}

// ── Level Control cluster (0x0008) ───────────────────────────────────────────

/// Level Control cluster (`0x0008`) — dimmer / percentage-position control.
pub mod level_control {
    use super::*;
    /// Cluster ID for this module.
    pub const CLUSTER_ID: u32 = cluster_id::LEVEL_CONTROL;

    /// `0x0000` — CurrentLevel attribute (0..=254).
    pub const ATTR_CURRENT_LEVEL: u32 = 0x0000;
    /// `0x0001` — RemainingTime attribute (0.1 s units).
    pub const ATTR_REMAINING_TIME: u32 = 0x0001;
    /// `0x0011` — OnLevel attribute (level applied when OnOff goes true).
    pub const ATTR_ON_LEVEL: u32 = 0x0011;

    /// `0x00` — MoveToLevel command.
    pub const CMD_MOVE_TO_LEVEL: u32 = 0x00;
    /// `0x01` — Move command (at a given rate).
    pub const CMD_MOVE: u32 = 0x01;
    /// `0x02` — Step command.
    pub const CMD_STEP: u32 = 0x02;
    /// `0x03` — Stop command.
    pub const CMD_STOP: u32 = 0x03;
    /// `0x04` — MoveToLevelWithOnOff — turns the fixture on if off.
    pub const CMD_MOVE_TO_LEVEL_WITH_ON_OFF: u32 = 0x04;

    /// TLV for MoveToLevel: `{ level(0): u8, transitionTime(1): u16 | null, optionsMask(2): u8, optionsOverride(3): u8 }`
    pub fn move_to_level_tlv(level: u8, transition_time_tenths: Option<u16>) -> Vec<u8> {
        let mut inner = tlv_uint8(0, level);
        inner.extend_from_slice(&match transition_time_tenths {
            Some(t) => tlv_uint16(1, t),
            None => tlv_null(1),
        });
        inner.extend_from_slice(&tlv_uint8(2, 0)); // optionsMask
        inner.extend_from_slice(&tlv_uint8(3, 0)); // optionsOverride
        wrap_struct(&inner)
    }
}

// ── Color Control cluster (0x0300) ───────────────────────────────────────────

/// Color Control cluster (`0x0300`) — HSV / CIE / color-temperature.
pub mod color_control {
    use super::*;
    /// Cluster ID for this module.
    pub const CLUSTER_ID: u32 = cluster_id::COLOR_CONTROL;

    /// `0x0000` — CurrentHue attribute.
    pub const ATTR_CURRENT_HUE: u32 = 0x0000;
    /// `0x0001` — CurrentSaturation attribute.
    pub const ATTR_CURRENT_SAT: u32 = 0x0001;
    /// `0x0007` — ColorTemperatureMireds attribute.
    pub const ATTR_COLOR_TEMP_MIREDS: u32 = 0x0007;
    /// `0x0008` — ColorMode attribute (HSV vs CIE vs color-temp).
    pub const ATTR_COLOR_MODE: u32 = 0x0008;

    /// `0x00` — MoveToHue command.
    pub const CMD_MOVE_TO_HUE: u32 = 0x00;
    /// `0x03` — MoveToSaturation command.
    pub const CMD_MOVE_TO_SAT: u32 = 0x03;
    /// `0x06` — MoveToHueAndSaturation command.
    pub const CMD_MOVE_TO_HUE_AND_SAT: u32 = 0x06;
    /// `0x0A` — MoveToColorTemperature command.
    pub const CMD_MOVE_TO_COLOR_TEMP: u32 = 0x0A;

    /// TLV for MoveToHueAndSaturation.
    pub fn move_to_hue_and_sat_tlv(hue: u8, sat: u8, transition_time_tenths: u16) -> Vec<u8> {
        let mut inner = tlv_uint8(0, hue);
        inner.extend_from_slice(&tlv_uint8(1, sat));
        inner.extend_from_slice(&tlv_uint16(2, transition_time_tenths));
        inner.extend_from_slice(&tlv_uint8(3, 0)); // optionsMask
        inner.extend_from_slice(&tlv_uint8(4, 0)); // optionsOverride
        wrap_struct(&inner)
    }

    /// TLV for MoveToColorTemperature.
    pub fn move_to_color_temp_tlv(mireds: u16, transition_time_tenths: u16) -> Vec<u8> {
        let mut inner = tlv_uint16(0, mireds);
        inner.extend_from_slice(&tlv_uint16(1, transition_time_tenths));
        inner.extend_from_slice(&tlv_uint8(2, 0));
        inner.extend_from_slice(&tlv_uint8(3, 0));
        wrap_struct(&inner)
    }
}

// ── Thermostat cluster (0x0201) ───────────────────────────────────────────────

/// Thermostat cluster (`0x0201`) — HVAC setpoint and mode control.
pub mod thermostat {
    use super::*;
    /// Cluster ID for this module.
    pub const CLUSTER_ID: u32 = cluster_id::THERMOSTAT;

    /// `0x0000` — LocalTemperature attribute (signed 0.01°C).
    pub const ATTR_LOCAL_TEMP: u32 = 0x0000;
    /// `0x0011` — OccupiedCoolingSetpoint attribute.
    pub const ATTR_OCCUPIED_COOLING_SETPOINT: u32 = 0x0011;
    /// `0x0012` — OccupiedHeatingSetpoint attribute.
    pub const ATTR_OCCUPIED_HEATING_SETPOINT: u32 = 0x0012;
    /// `0x001C` — SystemMode attribute (Off/Auto/Cool/Heat/…).
    pub const ATTR_SYSTEM_MODE: u32 = 0x001C;

    /// `0x01` — SetWeeklySchedule command.
    pub const CMD_SET_WEEKLY_SCHEDULE: u32 = 0x01;
    /// `0x00` — SetpointRaiseLower command.
    pub const CMD_SET_SETPOINT_RAISE_LOWER: u32 = 0x00;

    /// TLV for SetpointRaiseLower: `{ mode(0): u8, amount(1): i8 }`
    /// `mode`: 0=Heat, 1=Cool, 2=Both. `amount`: signed 0.1°C steps.
    pub fn setpoint_raise_lower_tlv(mode: u8, amount: i8) -> Vec<u8> {
        let mut inner = tlv_uint8(0, mode);
        inner.push(tlv::TAG_CONTEXT_1 | tlv::TYPE_SIGNED_INT_1);
        inner.push(1);
        inner.push(amount as u8);
        wrap_struct(&inner)
    }
}

// ── Door Lock cluster (0x0101) ────────────────────────────────────────────────

/// Door Lock cluster (`0x0101`) — lock actuator and PIN management.
pub mod door_lock {
    use super::*;
    /// Cluster ID for this module.
    pub const CLUSTER_ID: u32 = cluster_id::DOOR_LOCK;

    /// `0x0000` — LockState attribute.
    pub const ATTR_LOCK_STATE: u32 = 0x0000;
    /// `0x0001` — LockType attribute.
    pub const ATTR_LOCK_TYPE: u32 = 0x0001;

    /// `0x00` — LockDoor command.
    pub const CMD_LOCK_DOOR: u32 = 0x00;
    /// `0x01` — UnlockDoor command.
    pub const CMD_UNLOCK_DOOR: u32 = 0x01;

    /// TLV for LockDoor / UnlockDoor: `{ PINCode(0)?: octet_string }` (PIN optional)
    pub fn lock_tlv(pin: Option<&[u8]>) -> Vec<u8> {
        let inner = if let Some(p) = pin {
            // TLV octet string: context_tag(0) + type_octet_string + length(1B) + data
            let mut v = vec![0x30u8, 0, p.len() as u8];
            v.extend_from_slice(p);
            v
        } else {
            vec![]
        };
        wrap_struct(&inner)
    }
}

// ── Window Covering cluster (0x0102) ─────────────────────────────────────────

/// Window Covering cluster (`0x0102`) — blinds / shades / awnings.
pub mod window_covering {
    use super::*;
    /// Cluster ID for this module.
    pub const CLUSTER_ID: u32 = cluster_id::WINDOW_COVERING;

    /// `0x0008` — CurrentPositionLiftPercentage.
    pub const ATTR_CURRENT_POSITION_LIFT_PCT: u32 = 0x0008;
    /// `0x0009` — CurrentPositionTiltPercentage.
    pub const ATTR_CURRENT_POSITION_TILT_PCT: u32 = 0x0009;

    /// `0x00` — UpOrOpen command.
    pub const CMD_UP_OR_OPEN: u32 = 0x00;
    /// `0x01` — DownOrClose command.
    pub const CMD_DOWN_OR_CLOSE: u32 = 0x01;
    /// `0x02` — StopMotion command.
    pub const CMD_STOP_MOTION: u32 = 0x02;
    /// `0x05` — GoToLiftPercentage command.
    pub const CMD_GO_TO_LIFT_PERCENTAGE: u32 = 0x05;
    /// `0x08` — GoToTiltPercentage command.
    pub const CMD_GO_TO_TILT_PERCENTAGE: u32 = 0x08;

    /// TLV for `GoToLiftPercentage { liftPercent100thsValue }`.
    pub fn go_to_lift_percentage_tlv(percent: u8) -> Vec<u8> {
        let inner = tlv_uint8(0, percent);
        wrap_struct(&inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_off_command_encodes_correctly() {
        let on = on_off::on_tlv();
        assert_eq!(on, vec![tlv::TYPE_STRUCTURE, tlv::TYPE_END_OF_CONTAINER]);
        let off = on_off::off_tlv();
        assert_eq!(on, off); // both empty structs
    }

    #[test]
    fn level_control_move_to_level_tlv() {
        let tlv = level_control::move_to_level_tlv(128, Some(10));
        // Should start with structure type and end with end-of-container
        assert_eq!(tlv[0], 0x15); // TYPE_STRUCTURE
        assert_eq!(*tlv.last().unwrap(), 0x18); // END_OF_CONTAINER
        // level byte (128)
        assert!(tlv.contains(&128));
    }

    #[test]
    fn thermostat_setpoint_tlv_roundtrip() {
        let tlv = thermostat::setpoint_raise_lower_tlv(0, 10); // Heat, +1.0°C
        assert_eq!(tlv[0], 0x15); // TYPE_STRUCTURE
        assert_eq!(*tlv.last().unwrap(), 0x18);
    }

    #[test]
    fn door_lock_lock_no_pin() {
        let tlv = door_lock::lock_tlv(None);
        assert_eq!(tlv, vec![0x15, 0x18]); // empty struct
    }

    #[test]
    fn door_lock_lock_with_pin() {
        let tlv = door_lock::lock_tlv(Some(b"1234"));
        assert!(tlv.len() > 2);
        assert_eq!(tlv[0], 0x15); // structure
    }

    #[test]
    fn color_temp_tlv_encodes_mireds() {
        let tlv = color_control::move_to_color_temp_tlv(300, 10);
        assert_eq!(tlv[0], 0x15);
        // mireds 300 = 0x012C, should appear as LE bytes somewhere in TLV
        let has_mireds = tlv.windows(2).any(|w| w == [0x2C, 0x01]);
        assert!(has_mireds);
    }

    // ── Path type tests ───────────────────────────────────────────────────────

    #[test]
    fn attribute_path_specific_encode_decode_roundtrip() {
        let path = AttributePath::specific(1, 0x0006, 0x0000);
        let encoded = path.encode();
        // starts with structure, ends with end-of-container
        assert_eq!(encoded[0], tlv::TYPE_STRUCTURE);
        assert_eq!(*encoded.last().unwrap(), tlv::TYPE_END_OF_CONTAINER);
        let decoded = AttributePath::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, path);
    }

    #[test]
    fn attribute_path_wildcard_encode_has_no_fields() {
        let path = AttributePath::wildcard();
        let encoded = path.encode();
        // wildcard = empty struct: [0x15, 0x18]
        assert_eq!(
            encoded,
            vec![tlv::TYPE_STRUCTURE, tlv::TYPE_END_OF_CONTAINER]
        );
        let decoded = AttributePath::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, path);
    }

    #[test]
    fn command_path_encode_decode_roundtrip() {
        let path = CommandPath::new(0, 0x0006, 0x01);
        let encoded = path.encode();
        assert_eq!(encoded[0], tlv::TYPE_STRUCTURE);
        assert_eq!(*encoded.last().unwrap(), tlv::TYPE_END_OF_CONTAINER);
        let decoded = CommandPath::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, path);
    }
}
