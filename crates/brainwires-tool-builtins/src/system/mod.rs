#![deny(missing_docs)]
//! # brainwires-system
//!
//! Generic OS-level primitives for the Brainwires Agent Framework:
//! filesystem event watching and system service management (systemd, Docker, processes).
//!
//! ## Feature flags
//!
//! | Feature    | Description                                      |
//! |------------|--------------------------------------------------|
//! | `reactor`  | Filesystem event watcher (requires `notify` 7)   |
//! | `services` | systemd / Docker / process management            |
//! | `full`     | All features enabled                             |

pub mod config;

/// Filesystem event reactor — watch directories and trigger actions on changes.
#[cfg(feature = "reactor")]
pub mod reactor;

/// System service management — systemd, Docker containers, and processes.
#[cfg(feature = "services")]
pub mod services;

pub use config::{ReactorConfig, ServiceConfig};
