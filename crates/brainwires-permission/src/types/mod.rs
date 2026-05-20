//! Core permission system types
//!
//! This module defines the capability-based permission model for agents,
//! including filesystem, tool, network, spawning, git, and quota capabilities.

/// Agent capabilities and profile types.
pub mod agent;
/// Granular capability sub-types (filesystem, tools, network, spawning, git, quotas).
pub mod capabilities;
/// Glob-based path pattern type used by filesystem capabilities.
pub mod path_pattern;

#[cfg(test)]
mod tests;

// Re-export all public types so callers continue to work without path changes.
pub use agent::{AgentCapabilities, CapabilityProfile};
pub use capabilities::{
    FilesystemCapabilities, GitCapabilities, GitOperation, NetworkCapabilities, ResourceQuotas,
    SpawningCapabilities, ToolCapabilities, ToolCategory,
};
pub use path_pattern::PathPattern;
