//! GPIO pin allocation and state tracking.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::config::GpioConfig;
use super::device::GpioDirection;
use super::safety::GpioSafetyPolicy;

/// A handle to a GPIO pin with auto-release on drop.
#[derive(Debug)]
pub struct GpioPin {
    /// Chip number.
    pub chip: u32,
    /// Line number.
    pub line: u32,
    /// Current direction.
    pub direction: GpioDirection,
    /// Agent that owns this pin.
    pub agent_id: String,
    /// When the pin was acquired.
    pub acquired_at: Instant,
}

/// Internal tracking of an active pin.
struct ActivePin {
    agent_id: String,
    direction: GpioDirection,
    acquired_at: Instant,
}

/// Manages GPIO pin allocation with safety enforcement.
///
/// Enforces the allow-list, concurrent pin limits, and automatic release
/// of pins held past the configured timeout.
pub struct GpioPinManager {
    safety: GpioSafetyPolicy,
    active_pins: HashMap<(u32, u32), ActivePin>,
    max_concurrent: usize,
    auto_release_timeout: Duration,
}

impl GpioPinManager {
    /// Create a new pin manager from configuration.
    pub fn from_config(config: &GpioConfig) -> Self {
        Self {
            safety: GpioSafetyPolicy::from_config(config),
            active_pins: HashMap::new(),
            max_concurrent: config.max_concurrent_pins,
            auto_release_timeout: Duration::from_secs(config.auto_release_timeout_secs),
        }
    }

    /// Acquire a GPIO pin for the given agent.
    pub fn acquire(
        &mut self,
        chip: u32,
        line: u32,
        direction: GpioDirection,
        agent_id: &str,
    ) -> Result<GpioPin, String> {
        // Check safety policy
        self.safety
            .check(chip, line, &direction.to_string(), agent_id)?;

        // Check concurrent limit
        if self.active_pins.len() >= self.max_concurrent {
            return Err(format!(
                "Maximum concurrent pins ({}) reached",
                self.max_concurrent
            ));
        }

        // Check if already acquired
        let key = (chip, line);
        if let Some(existing) = self.active_pins.get(&key) {
            return Err(format!(
                "Pin chip{chip}/line{line} already held by agent '{}'",
                existing.agent_id
            ));
        }

        let now = Instant::now();
        self.active_pins.insert(
            key,
            ActivePin {
                agent_id: agent_id.to_string(),
                direction,
                acquired_at: now,
            },
        );

        Ok(GpioPin {
            chip,
            line,
            direction,
            agent_id: agent_id.to_string(),
            acquired_at: now,
        })
    }

    /// Release a GPIO pin.
    pub fn release(&mut self, chip: u32, line: u32) {
        self.active_pins.remove(&(chip, line));
    }

    /// Release all pins held by a specific agent.
    pub fn release_agent(&mut self, agent_id: &str) {
        self.active_pins.retain(|_, pin| pin.agent_id != agent_id);
    }

    /// Release pins that have exceeded the auto-release timeout.
    pub fn release_timed_out(&mut self) -> Vec<(u32, u32, String)> {
        let now = Instant::now();
        let mut released = Vec::new();

        self.active_pins.retain(|&(chip, line), pin| {
            if now.duration_since(pin.acquired_at) >= self.auto_release_timeout {
                released.push((chip, line, pin.agent_id.clone()));
                tracing::warn!(
                    "Auto-releasing GPIO pin chip{chip}/line{line} from agent '{}' (timeout)",
                    pin.agent_id
                );
                false
            } else {
                true
            }
        });

        released
    }

    /// Get the number of active pins.
    pub fn active_count(&self) -> usize {
        self.active_pins.len()
    }

    /// List all active pins.
    pub fn active_pins(&self) -> Vec<GpioPin> {
        self.active_pins
            .iter()
            .map(|(&(chip, line), pin)| GpioPin {
                chip,
                line,
                direction: pin.direction,
                agent_id: pin.agent_id.clone(),
                acquired_at: pin.acquired_at,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GpioConfig {
        GpioConfig {
            allowed_pins: vec![(0, 17), (0, 27), (0, 22)],
            max_concurrent_pins: 2,
            auto_release_timeout_secs: 60,
        }
    }

    #[test]
    fn acquire_allowed_pin() {
        let mut mgr = GpioPinManager::from_config(&test_config());
        let pin = mgr.acquire(0, 17, GpioDirection::Output, "agent-1");
        assert!(pin.is_ok());
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn reject_disallowed_pin() {
        let mut mgr = GpioPinManager::from_config(&test_config());
        let pin = mgr.acquire(0, 99, GpioDirection::Output, "agent-1");
        assert!(pin.is_err());
    }

    #[test]
    fn reject_concurrent_limit() {
        let mut mgr = GpioPinManager::from_config(&test_config());
        mgr.acquire(0, 17, GpioDirection::Output, "agent-1")
            .unwrap();
        mgr.acquire(0, 27, GpioDirection::Output, "agent-1")
            .unwrap();
        let third = mgr.acquire(0, 22, GpioDirection::Output, "agent-1");
        assert!(third.is_err());
    }

    #[test]
    fn reject_double_acquire() {
        let mut mgr = GpioPinManager::from_config(&test_config());
        mgr.acquire(0, 17, GpioDirection::Output, "agent-1")
            .unwrap();
        let second = mgr.acquire(0, 17, GpioDirection::Input, "agent-2");
        assert!(second.is_err());
    }

    #[test]
    fn release_frees_pin() {
        let mut mgr = GpioPinManager::from_config(&test_config());
        mgr.acquire(0, 17, GpioDirection::Output, "agent-1")
            .unwrap();
        mgr.release(0, 17);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn release_agent_frees_all() {
        let mut mgr = GpioPinManager::from_config(&test_config());
        mgr.acquire(0, 17, GpioDirection::Output, "agent-1")
            .unwrap();
        mgr.acquire(0, 27, GpioDirection::Output, "agent-1")
            .unwrap();
        mgr.release_agent("agent-1");
        assert_eq!(mgr.active_count(), 0);
    }
}
