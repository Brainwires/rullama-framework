//! Permission mode types
//!
//! This module defines the `PermissionMode` enum for controlling tool execution
//! permission levels. Capability types (AgentCapabilities, profiles, etc.) are
//! defined in the `brainwires-permission` crate.

use serde::{Deserialize, Serialize};

// ── Permission Mode ──────────────────────────────────────────────────

/// Permission mode for tool execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum PermissionMode {
    /// Read-only mode - deny all write operations
    ReadOnly,
    /// Auto mode - approve safe operations, ask for dangerous ones
    #[default]
    Auto,
    /// Full mode - auto-approve all operations
    Full,
}

impl PermissionMode {
    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "read-only" | "readonly" => Some(Self::ReadOnly),
            "auto" => Some(Self::Auto),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Auto => "auto",
            Self::Full => "full",
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_mode_from_str() {
        assert_eq!(
            PermissionMode::parse("read-only"),
            Some(PermissionMode::ReadOnly)
        );
        assert_eq!(PermissionMode::parse("auto"), Some(PermissionMode::Auto));
        assert_eq!(PermissionMode::parse("full"), Some(PermissionMode::Full));
        assert_eq!(PermissionMode::parse("invalid"), None);
    }

    #[test]
    fn test_permission_mode_default() {
        assert_eq!(PermissionMode::default(), PermissionMode::Auto);
    }
}
