//! TaskAgent - Autonomous agent that executes a task in a loop using AI + tools
//!
//! Each `TaskAgent` owns its conversation history and calls the AI provider
//! repeatedly, executing tool requests and running validation before it
//! signals completion.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use brainwires_agent::{AgentContext, TaskAgent, TaskAgentConfig, TaskAgentResult};
//! use brainwires_core::Task;
//!
//! let context = Arc::new(AgentContext::new(
//!     "/my/project",
//!     Arc::new(my_executor),
//!     Arc::clone(&hub),
//!     Arc::clone(&lock_manager),
//! ));
//!
//! let agent = Arc::new(TaskAgent::new(
//!     "agent-1".to_string(),
//!     Task::new("task-1", "Refactor src/lib.rs"),
//!     Arc::clone(&provider),
//!     Arc::clone(&context),
//!     TaskAgentConfig::default(),
//! ));
//!
//! let result: TaskAgentResult = agent.execute().await?;
//! ```

mod agent;
mod spawn;
mod types;

#[cfg(test)]
mod tests;

// ── Billing hook newtype (Debug-safe Arc<dyn BillingHook>) ───────────────────

/// A `Debug`-safe wrapper around `Arc<dyn BillingHook>`.
///
/// `dyn BillingHook` doesn't implement `Debug`, so we can't derive it on
/// `TaskAgentConfig` directly. This newtype provides a no-op `Debug` impl
/// so the rest of the config can keep `#[derive(Debug)]`.
#[cfg(feature = "telemetry")]
#[derive(Clone)]
pub struct BillingHookRef(pub std::sync::Arc<dyn brainwires_telemetry::BillingHook>);

#[cfg(feature = "telemetry")]
impl std::fmt::Debug for BillingHookRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BillingHook")
    }
}

#[cfg(feature = "telemetry")]
impl BillingHookRef {
    /// Wrap a [`BillingHook`](brainwires_telemetry::BillingHook) implementation.
    pub fn new(hook: impl brainwires_telemetry::BillingHook) -> Self {
        Self(std::sync::Arc::new(hook))
    }
}

// ── Public re-exports ────────────────────────────────────────────────────────

pub use agent::TaskAgent;
pub use spawn::spawn_task_agent;
pub use types::{
    FailureCategory, LoopDetectionConfig, TaskAgentConfig, TaskAgentResult, TaskAgentStatus,
};
