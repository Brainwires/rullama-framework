use serde::{Deserialize, Serialize};

/// Represents an audio device (input or output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDevice {
    /// Unique device identifier (platform-specific).
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Whether this is the system default device.
    pub is_default: bool,
    /// Device direction.
    pub direction: DeviceDirection,
}

/// Whether a device is for input (capture) or output (playback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceDirection {
    /// Capture / microphone input.
    Input,
    /// Playback / speaker output.
    Output,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_device_creation_and_field_access() {
        let dev = AudioDevice {
            id: "hw:0".to_string(),
            name: "Built-in Microphone".to_string(),
            is_default: true,
            direction: DeviceDirection::Input,
        };
        assert_eq!(dev.id, "hw:0");
        assert_eq!(dev.name, "Built-in Microphone");
        assert!(dev.is_default);
        assert_eq!(dev.direction, DeviceDirection::Input);
    }

    #[test]
    fn audio_device_non_default_output() {
        let dev = AudioDevice {
            id: "hw:1".to_string(),
            name: "HDMI Output".to_string(),
            is_default: false,
            direction: DeviceDirection::Output,
        };
        assert!(!dev.is_default);
        assert_eq!(dev.direction, DeviceDirection::Output);
    }

    #[test]
    fn device_direction_variants_are_distinct() {
        assert_ne!(DeviceDirection::Input, DeviceDirection::Output);
    }
}
