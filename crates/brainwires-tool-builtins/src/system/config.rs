//! Configuration types for `brainwires-system`.

use serde::{Deserialize, Serialize};

/// File system reactor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorConfig {
    /// Maximum events per minute before rate limiting kicks in.
    pub max_events_per_minute: u32,
    /// Global debounce window in milliseconds.
    pub global_debounce_ms: u64,
    /// Maximum recursive watch depth.
    pub max_watch_depth: u32,
    /// Reactor rules.
    #[serde(default)]
    pub rules: Vec<ReactorRuleDef>,
}

impl Default for ReactorConfig {
    fn default() -> Self {
        Self {
            max_events_per_minute: 60,
            global_debounce_ms: 500,
            max_watch_depth: 10,
            rules: Vec::new(),
        }
    }
}

/// Definition of a reactor rule in configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorRuleDef {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Paths to watch.
    pub watch_paths: Vec<String>,
    /// Glob patterns for file matching.
    #[serde(default)]
    pub patterns: Vec<String>,
    /// Patterns to exclude.
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
    /// File system event types to react to.
    #[serde(default)]
    pub event_types: Vec<String>,
    /// Per-rule debounce in milliseconds.
    #[serde(default = "default_debounce")]
    pub debounce_ms: u64,
    /// Whether this rule is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_debounce() -> u64 {
    1000
}

fn default_true() -> bool {
    true
}

/// System service management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Explicit allow-list of service names (no wildcards).
    #[serde(default)]
    pub allowed_services: Vec<String>,
    /// Additional forbidden services (supplements hardcoded deny-list).
    #[serde(default)]
    pub forbidden_services: Vec<String>,
    /// If true, only status/list/logs are permitted (no start/stop/restart).
    #[serde(default = "default_true")]
    pub read_only: bool,
    /// Docker socket path override.
    #[serde(default)]
    pub docker_socket_path: Option<String>,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            allowed_services: Vec::new(),
            forbidden_services: Vec::new(),
            read_only: true,
            docker_socket_path: None,
        }
    }
}
