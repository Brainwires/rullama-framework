//! GPIO hardware access for autonomous agents.
//!
//! Provides safe, controlled access to GPIO pins with strict allow-lists,
//! auto-release on agent unhealthy, and direction change approval.
//! Uses `gpio-cdev` (modern character device API) as primary,
//! with `sysfs_gpio` as fallback for older kernels.

/// GPIO runtime configuration (chip paths, safety policy, pin allowlists).
pub mod config;
/// Chip and line enumeration backed by `gpio-cdev`.
pub mod device;
/// `GpioPinManager` — claim/release pins with agent-lifecycle safety hooks.
pub mod pin_manager;
/// Software PWM over claimed pins.
pub mod pwm;
/// `GpioSafetyPolicy` — allow-list + direction change rules enforced by the pin manager.
pub mod safety;

pub use device::{GpioChipInfo, GpioLineInfo};
pub use pin_manager::{GpioPin, GpioPinManager};
pub use safety::GpioSafetyPolicy;
