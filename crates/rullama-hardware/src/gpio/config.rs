use serde::{Deserialize, Serialize};

/// Configuration for the GPIO pin manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioConfig {
    /// Allowed (chip, line) pairs — empty means no access.
    #[serde(default)]
    pub allowed_pins: Vec<(u32, u32)>,
    /// Maximum concurrent pins an agent may hold.
    pub max_concurrent_pins: usize,
    /// Timeout in seconds before auto-releasing a pin from an unhealthy agent.
    pub auto_release_timeout_secs: u64,
}

impl Default for GpioConfig {
    fn default() -> Self {
        Self {
            allowed_pins: Vec::new(),
            max_concurrent_pins: 4,
            auto_release_timeout_secs: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_no_allowed_pins() {
        let cfg = GpioConfig::default();
        assert!(cfg.allowed_pins.is_empty());
        assert_eq!(cfg.max_concurrent_pins, 4);
        assert_eq!(cfg.auto_release_timeout_secs, 300);
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = GpioConfig {
            allowed_pins: vec![(0, 17), (0, 18), (1, 4)],
            max_concurrent_pins: 8,
            auto_release_timeout_secs: 60,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GpioConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.allowed_pins.len(), 3);
        assert_eq!(back.max_concurrent_pins, 8);
        assert_eq!(back.auto_release_timeout_secs, 60);
    }

    #[test]
    fn config_default_deserialization_uses_empty_pins() {
        let json = r#"{"max_concurrent_pins": 2, "auto_release_timeout_secs": 120}"#;
        let cfg: GpioConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.allowed_pins.is_empty());
        assert_eq!(cfg.max_concurrent_pins, 2);
    }
}
