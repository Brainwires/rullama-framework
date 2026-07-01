//! GPIO-specific safety policies.

use std::collections::HashSet;

use super::config::GpioConfig;

/// Safety policy for GPIO access that enforces an explicit allow-list of (chip, line) pairs.
pub struct GpioSafetyPolicy {
    allowed_pins: HashSet<(u32, u32)>,
}

impl GpioSafetyPolicy {
    /// Create from configuration.
    pub fn from_config(config: &GpioConfig) -> Self {
        Self {
            allowed_pins: config.allowed_pins.iter().cloned().collect(),
        }
    }

    /// Check if access to a pin is allowed.
    pub fn check(
        &self,
        chip: u32,
        line: u32,
        _direction: &str,
        _agent_id: &str,
    ) -> Result<(), String> {
        if self.allowed_pins.is_empty() {
            return Err("No GPIO pins are configured in the allow-list. \
                 Add pins to GpioConfig.allowed_pins to enable access."
                .to_string());
        }

        if !self.allowed_pins.contains(&(chip, line)) {
            return Err(format!(
                "GPIO pin chip{chip}/line{line} is not in the allow-list"
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_allow_list_rejects_all() {
        let policy = GpioSafetyPolicy::from_config(&GpioConfig::default());
        assert!(policy.check(0, 17, "output", "agent-1").is_err());
    }

    #[test]
    fn allowed_pin_passes() {
        let config = GpioConfig {
            allowed_pins: vec![(0, 17)],
            ..Default::default()
        };
        let policy = GpioSafetyPolicy::from_config(&config);
        assert!(policy.check(0, 17, "output", "agent-1").is_ok());
        assert!(policy.check(0, 18, "output", "agent-1").is_err());
    }
}
