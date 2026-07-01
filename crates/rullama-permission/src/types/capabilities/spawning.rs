use serde::{Deserialize, Serialize};

/// Agent spawning capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawningCapabilities {
    /// Can spawn child agents
    #[serde(default)]
    pub can_spawn: bool,

    /// Maximum concurrent child agents
    #[serde(default = "default_max_children")]
    pub max_children: u32,

    /// Maximum depth of agent hierarchy
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,

    /// Can spawn agents with elevated privileges (requires approval)
    #[serde(default)]
    pub can_elevate: bool,
}

fn default_max_children() -> u32 {
    3
}

fn default_max_depth() -> u32 {
    2
}

impl Default for SpawningCapabilities {
    fn default() -> Self {
        Self {
            can_spawn: false,
            max_children: 3,
            max_depth: 2,
            can_elevate: false,
        }
    }
}

impl SpawningCapabilities {
    /// Create disabled spawning capabilities
    pub fn disabled() -> Self {
        Self {
            can_spawn: false,
            max_children: 0,
            max_depth: 0,
            can_elevate: false,
        }
    }

    /// Create full spawning capabilities
    pub fn full() -> Self {
        Self {
            can_spawn: true,
            max_children: 10,
            max_depth: 5,
            can_elevate: true,
        }
    }
}
