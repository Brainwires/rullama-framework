//! AgentManager trait and supporting types for agent lifecycle abstraction
//!
//! Provides a reusable trait that any MCP server implementation can implement
//! to expose agent spawning, monitoring, and control operations without
//! depending on CLI-specific types.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for spawning a new agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnConfig {
    /// Description of the task for the agent to execute
    pub description: String,
    /// Optional working directory for file operations
    pub working_directory: Option<String>,
    /// Optional maximum number of iterations (default: 100)
    pub max_iterations: Option<u32>,
    /// Enable automatic validation checks before completion
    pub enable_validation: Option<bool>,
    /// Build type for validation (e.g. "npm", "cargo", "typescript")
    pub build_type: Option<String>,
    /// Opaque blob for implementation-specific config (e.g. MDAP settings)
    pub extra: Option<Value>,
}

/// Information about a running or completed agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Unique agent identifier
    pub agent_id: String,
    /// Current agent status (e.g. "running", "completed", "failed")
    pub status: String,
    /// Description of the task the agent is working on
    pub task_description: String,
    /// Number of iterations completed so far
    pub iterations: u32,
}

/// Result from a completed agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Unique agent identifier
    pub agent_id: String,
    /// Whether the agent completed successfully
    pub success: bool,
    /// Human-readable summary of what was accomplished
    pub summary: String,
    /// Total number of iterations used
    pub iterations: u32,
}

/// Trait for agent lifecycle management
///
/// Implement this trait to expose agent spawning and control capabilities
/// via an MCP server without coupling to CLI-specific internals.
#[async_trait]
pub trait AgentManager: Send + Sync {
    /// Spawn a new agent and return its ID
    async fn spawn_agent(&self, config: SpawnConfig) -> Result<String>;

    /// List all currently active agents
    async fn list_agents(&self) -> Result<Vec<AgentInfo>>;

    /// Get the current status of a specific agent
    async fn agent_status(&self, agent_id: &str) -> Result<AgentInfo>;

    /// Stop a running agent
    async fn stop_agent(&self, agent_id: &str) -> Result<()>;

    /// Wait for an agent to complete and return its result
    ///
    /// If `timeout_secs` is `Some`, returns an error if the agent has not
    /// completed within the given number of seconds.
    async fn await_agent(&self, agent_id: &str, timeout_secs: Option<u64>) -> Result<AgentResult>;

    /// Return pool-level statistics as a JSON value
    async fn pool_stats(&self) -> Result<Value>;

    /// Return all currently held file locks as a JSON value
    async fn file_locks(&self) -> Result<Value>;
}
