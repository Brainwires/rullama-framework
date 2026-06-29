//! Remote Bridge - Backend communication client
//!
//! Maintains communication with the rullama-studio backend using either:
//! 1. **Supabase Realtime** (preferred) - Bidirectional WebSocket for instant commands
//! 2. **HTTP Polling** (fallback) - For environments where Realtime isn't available
//!
//! All CLI-specific dependencies have been removed:
//! - `PlatformPaths` → `BridgeConfig.sessions_dir` / `BridgeConfig.attachment_dir`
//! - `crate::build_info::VERSION` → `BridgeConfig.version`
//! - `spawn_agent_process_with_options` → `AgentSpawner` trait object
//! - `crate::ipc::*` → bridge-internal `crate::ipc::*`

mod agent_relay;
mod commands;
mod core;
mod loops;
mod registration;
mod types;

#[cfg(test)]
mod tests;

// Public re-exports — callers continue to import from `crate::remote::bridge::*`
// without any changes.
pub use core::RemoteBridge;
pub use types::{BridgeConfig, BridgeState, ConnectionMode, RealtimeCredentials};
