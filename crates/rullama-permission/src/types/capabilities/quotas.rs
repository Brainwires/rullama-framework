use serde::{Deserialize, Serialize};

/// Resource quota limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    /// Maximum execution time (seconds)
    #[serde(default)]
    pub max_execution_time: Option<u64>,

    /// Maximum memory usage (bytes)
    #[serde(default)]
    pub max_memory: Option<u64>,

    /// Maximum API tokens consumed
    #[serde(default)]
    pub max_tokens: Option<u64>,

    /// Maximum tool calls per session
    #[serde(default)]
    pub max_tool_calls: Option<u32>,

    /// Maximum files modified per session
    #[serde(default)]
    pub max_files_modified: Option<u32>,
}

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            max_execution_time: Some(30 * 60), // 30 minutes
            max_memory: None,
            max_tokens: Some(100_000),
            max_tool_calls: Some(500),
            max_files_modified: Some(50),
        }
    }
}

impl ResourceQuotas {
    /// Create conservative quotas
    pub fn conservative() -> Self {
        Self {
            max_execution_time: Some(5 * 60),    // 5 minutes
            max_memory: Some(512 * 1024 * 1024), // 512MB
            max_tokens: Some(10_000),
            max_tool_calls: Some(50),
            max_files_modified: Some(10),
        }
    }

    /// Create standard quotas
    pub fn standard() -> Self {
        Self::default()
    }

    /// Create generous quotas
    pub fn generous() -> Self {
        Self {
            max_execution_time: Some(2 * 60 * 60), // 2 hours
            max_memory: None,
            max_tokens: Some(500_000),
            max_tool_calls: Some(2000),
            max_files_modified: Some(200),
        }
    }
}
