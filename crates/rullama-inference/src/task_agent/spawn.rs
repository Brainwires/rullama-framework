//! Free function for spawning a [`TaskAgent`] on a Tokio background task.

use std::sync::Arc;

use anyhow::Result;

use super::agent::TaskAgent;
use super::types::TaskAgentResult;

/// Spawn a task agent on a Tokio background task.
///
/// Returns a [`JoinHandle`][tokio::task::JoinHandle] that resolves to the
/// agent's [`TaskAgentResult`] when execution finishes.
pub fn spawn_task_agent(agent: Arc<TaskAgent>) -> tokio::task::JoinHandle<Result<TaskAgentResult>> {
    tokio::spawn(async move { agent.execute().await })
}
