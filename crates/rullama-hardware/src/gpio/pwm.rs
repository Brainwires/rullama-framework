//! Software PWM output support.
//!
//! Provides software-timed PWM for GPIO output pins. Hardware PWM
//! requires kernel support and is not covered here.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// PWM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PwmConfig {
    /// Frequency in Hz.
    pub frequency_hz: f64,
    /// Duty cycle (0.0 to 1.0).
    pub duty_cycle: f64,
}

impl PwmConfig {
    /// Create a new PWM configuration.
    pub fn new(frequency_hz: f64, duty_cycle: f64) -> Result<Self, String> {
        if frequency_hz <= 0.0 || frequency_hz > 100_000.0 {
            return Err(format!(
                "Frequency must be between 0 and 100kHz, got {frequency_hz}Hz"
            ));
        }
        if !(0.0..=1.0).contains(&duty_cycle) {
            return Err(format!(
                "Duty cycle must be between 0.0 and 1.0, got {duty_cycle}"
            ));
        }
        Ok(Self {
            frequency_hz,
            duty_cycle,
        })
    }

    /// Calculate the period duration.
    pub fn period(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.frequency_hz)
    }

    /// Calculate the high time duration.
    pub fn high_time(&self) -> Duration {
        Duration::from_secs_f64(self.duty_cycle / self.frequency_hz)
    }

    /// Calculate the low time duration.
    pub fn low_time(&self) -> Duration {
        Duration::from_secs_f64((1.0 - self.duty_cycle) / self.frequency_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pwm_config_valid() {
        let config = PwmConfig::new(1000.0, 0.5).unwrap();
        assert_eq!(config.frequency_hz, 1000.0);
        assert_eq!(config.duty_cycle, 0.5);
    }

    #[test]
    fn pwm_config_rejects_bad_frequency() {
        assert!(PwmConfig::new(0.0, 0.5).is_err());
        assert!(PwmConfig::new(-1.0, 0.5).is_err());
        assert!(PwmConfig::new(200_000.0, 0.5).is_err());
    }

    #[test]
    fn pwm_config_rejects_bad_duty_cycle() {
        assert!(PwmConfig::new(1000.0, -0.1).is_err());
        assert!(PwmConfig::new(1000.0, 1.1).is_err());
    }

    #[test]
    fn pwm_period_calculation() {
        let config = PwmConfig::new(1000.0, 0.5).unwrap();
        let period = config.period();
        assert!((period.as_secs_f64() - 0.001).abs() < 0.0001);
    }
}
