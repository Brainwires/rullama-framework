//! GPIO chip and line discovery.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Information about a GPIO chip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioChipInfo {
    /// Chip number (e.g., 0 for /dev/gpiochip0).
    pub chip: u32,
    /// Device path.
    pub path: PathBuf,
    /// Chip label/name.
    pub label: String,
    /// Number of lines on this chip.
    pub num_lines: u32,
}

/// Information about a GPIO line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioLineInfo {
    /// Chip number.
    pub chip: u32,
    /// Line number within the chip.
    pub line: u32,
    /// Line name, if set.
    pub name: Option<String>,
    /// Whether the line is currently in use.
    pub in_use: bool,
    /// Current direction.
    pub direction: Option<GpioDirection>,
}

/// GPIO pin direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpioDirection {
    /// Input (reading values).
    Input,
    /// Output (writing values).
    Output,
}

impl std::fmt::Display for GpioDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => write!(f, "input"),
            Self::Output => write!(f, "output"),
        }
    }
}

/// Discover available GPIO chips on the system.
pub fn discover_chips() -> Vec<GpioChipInfo> {
    let mut chips = Vec::new();

    for i in 0..16 {
        let path = PathBuf::from(format!("/dev/gpiochip{i}"));
        if path.exists() {
            chips.push(GpioChipInfo {
                chip: i,
                path,
                label: format!("gpiochip{i}"),
                num_lines: 0, // Would need ioctl to get actual count
            });
        }
    }

    chips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpio_direction_display() {
        assert_eq!(GpioDirection::Input.to_string(), "input");
        assert_eq!(GpioDirection::Output.to_string(), "output");
    }

    #[test]
    fn gpio_chip_info_serialization() {
        let info = GpioChipInfo {
            chip: 0,
            path: PathBuf::from("/dev/gpiochip0"),
            label: "gpiochip0".to_string(),
            num_lines: 32,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: GpioChipInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.chip, 0);
    }
}
