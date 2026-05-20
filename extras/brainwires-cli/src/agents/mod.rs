//! Agent System
//!
//! Re-exports core agent infrastructure from brainwires-agent framework crate,
//! plus CLI-specific orchestration, management, and execution modules.
#![allow(hidden_glob_reexports)]

// Re-export framework agent types (communication, confidence, file_locks,
// operation_tracker, resource_checker, resource_locks, state_model,
// task_manager, task_queue, validation_loop, validation_agent, worktree,
// contract_net, market_allocation, saga, optimistic, wait_queue)
pub use brainwires::agents::*;

// CLI-specific modules
mod manager;
mod orchestrator;
mod pool;
mod task_agent;
mod worker;

pub use manager::AgentManager;
pub use orchestrator::OrchestratorAgent;
pub use pool::AgentPool;

pub use task_agent::{TaskAgent, TaskAgentConfig, TaskAgentResult, TaskAgentStatus};
pub use worker::*;

/// Prelude module for convenient imports
pub mod prelude {
    // Framework types
    pub use brainwires::agents::prelude::*;

    // CLI-specific types
    pub use super::manager::AgentManager;
    pub use super::orchestrator::OrchestratorAgent;
    pub use super::pool::AgentPool;
    pub use super::task_agent::{TaskAgent, TaskAgentConfig, TaskAgentResult, TaskAgentStatus};
}
