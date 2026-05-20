//! Agent Pool - Manages a pool of background task agents
//!
//! [`AgentPool`] handles the lifecycle of [`TaskAgent`]s: spawning, monitoring,
//! stopping, and awaiting results. All agents in the pool share the same
//! [`Provider`], tool executor, communication hub, and file lock manager.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use brainwires_agent::{AgentPool, TaskAgentConfig};
//! use brainwires_core::Task;
//!
//! let pool = AgentPool::new(
//!     10,
//!     Arc::clone(&provider),
//!     Arc::clone(&tool_executor),
//!     Arc::clone(&hub),
//!     Arc::clone(&lock_manager),
//!     "/my/project".to_string(),
//! );
//!
//! let agent_id = pool.spawn_agent(
//!     Task::new("t-1", "Implement feature X"),
//!     None,
//! ).await?;
//!
//! let result = pool.await_completion(&agent_id).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use brainwires_core::{Provider, Task};
use brainwires_tool_runtime::ToolExecutor;

use crate::context::AgentContext;
use crate::task_agent::{
    TaskAgent, TaskAgentConfig, TaskAgentResult, TaskAgentStatus, spawn_task_agent,
};
use brainwires_agent::communication::CommunicationHub;
use brainwires_agent::file_locks::FileLockManager;

// ── Internal handle ────────────────────────────────────────────────────────

struct AgentHandle {
    agent: Arc<TaskAgent>,
    join_handle: JoinHandle<Result<TaskAgentResult>>,
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Manages a pool of background [`TaskAgent`]s.
///
/// All agents share the same provider, tool executor, communication hub,
/// file lock manager, and working directory. Each agent gets its own
/// conversation history and working set.
pub struct AgentPool {
    max_agents: usize,
    agents: Arc<RwLock<HashMap<String, AgentHandle>>>,
    communication_hub: Arc<CommunicationHub>,
    file_lock_manager: Arc<FileLockManager>,
    provider: Arc<dyn Provider>,
    tool_executor: Arc<dyn ToolExecutor>,
    working_directory: String,
}

impl AgentPool {
    /// Create a new agent pool.
    ///
    /// # Parameters
    /// - `max_agents`: maximum number of concurrently running agents.
    /// - `provider`: AI provider shared by all agents.
    /// - `tool_executor`: tool executor shared by all agents.
    /// - `communication_hub`: inter-agent message bus.
    /// - `file_lock_manager`: file coordination across agents.
    /// - `working_directory`: default working directory for spawned agents.
    pub fn new(
        max_agents: usize,
        provider: Arc<dyn Provider>,
        tool_executor: Arc<dyn ToolExecutor>,
        communication_hub: Arc<CommunicationHub>,
        file_lock_manager: Arc<FileLockManager>,
        working_directory: impl Into<String>,
    ) -> Self {
        Self {
            max_agents,
            agents: Arc::new(RwLock::new(HashMap::new())),
            communication_hub,
            file_lock_manager,
            provider,
            tool_executor,
            working_directory: working_directory.into(),
        }
    }

    /// Spawn a new task agent and start it on a Tokio background task.
    ///
    /// Returns the agent ID. Use [`await_completion`][Self::await_completion]
    /// to wait for the result.
    ///
    /// Returns an error if the pool is already at capacity.
    pub async fn spawn_agent(&self, task: Task, config: Option<TaskAgentConfig>) -> Result<String> {
        {
            let agents = self.agents.read().await;
            if agents.len() >= self.max_agents {
                return Err(anyhow!(
                    "Agent pool is full ({}/{})",
                    agents.len(),
                    self.max_agents
                ));
            }
        }

        let agent_id = format!("agent-{}", uuid::Uuid::new_v4());
        let config = config.unwrap_or_default();

        let context = Arc::new(AgentContext::new(
            self.working_directory.clone(),
            Arc::clone(&self.tool_executor),
            Arc::clone(&self.communication_hub),
            Arc::clone(&self.file_lock_manager),
        ));

        let agent = Arc::new(TaskAgent::new(
            agent_id.clone(),
            task,
            Arc::clone(&self.provider),
            context,
            config,
        ));

        let handle = spawn_task_agent(Arc::clone(&agent));

        self.agents.write().await.insert(
            agent_id.clone(),
            AgentHandle {
                agent,
                join_handle: handle,
            },
        );

        tracing::info!(agent_id = %agent_id, "spawned agent");
        Ok(agent_id)
    }

    /// Spawn a new task agent with a custom [`AgentContext`].
    ///
    /// Unlike [`spawn_agent`][Self::spawn_agent] which uses the pool's default
    /// working directory, this method accepts a pre-built context. This is
    /// useful for workers that run in isolated worktrees with per-agent
    /// working directories.
    ///
    /// Returns the agent ID.
    pub async fn spawn_agent_with_context(
        &self,
        task: Task,
        context: Arc<AgentContext>,
        config: Option<TaskAgentConfig>,
    ) -> Result<String> {
        {
            let agents = self.agents.read().await;
            if agents.len() >= self.max_agents {
                return Err(anyhow!(
                    "Agent pool is full ({}/{})",
                    agents.len(),
                    self.max_agents
                ));
            }
        }

        let agent_id = format!("agent-{}", uuid::Uuid::new_v4());
        let config = config.unwrap_or_default();

        let agent = Arc::new(TaskAgent::new(
            agent_id.clone(),
            task,
            Arc::clone(&self.provider),
            context,
            config,
        ));

        let handle = spawn_task_agent(Arc::clone(&agent));

        self.agents.write().await.insert(
            agent_id.clone(),
            AgentHandle {
                agent,
                join_handle: handle,
            },
        );

        tracing::info!(agent_id = %agent_id, "spawned agent with custom context");
        Ok(agent_id)
    }

    /// Get the current status of an agent.
    ///
    /// Returns `None` if the agent is not in the pool.
    pub async fn get_status(&self, agent_id: &str) -> Option<TaskAgentStatus> {
        let agents = self.agents.read().await;
        let handle = agents.get(agent_id)?;
        Some(handle.agent.status().await)
    }

    /// Get a snapshot of the task assigned to an agent.
    pub async fn get_task(&self, agent_id: &str) -> Option<Task> {
        let agents = self.agents.read().await;
        let handle = agents.get(agent_id)?;
        Some(handle.agent.task().await)
    }

    /// Abort an agent and remove it from the pool.
    ///
    /// File locks held by the agent are released immediately.
    pub async fn stop_agent(&self, agent_id: &str) -> Result<()> {
        let handle = self
            .agents
            .write()
            .await
            .remove(agent_id)
            .ok_or_else(|| anyhow!("Agent {} not found", agent_id))?;

        handle.join_handle.abort();
        self.file_lock_manager.release_all_locks(agent_id).await;
        tracing::info!(agent_id = %agent_id, "stopped agent");
        Ok(())
    }

    /// Wait for an agent to finish and return its result.
    ///
    /// The agent is removed from the pool once it completes.
    pub async fn await_completion(&self, agent_id: &str) -> Result<TaskAgentResult> {
        let handle = self.agents.write().await.remove(agent_id);

        match handle {
            Some(h) => match h.join_handle.await {
                Ok(result) => result,
                Err(e) => Err(anyhow!("Agent task panicked: {}", e)),
            },
            None => Err(anyhow!("Agent {} not found", agent_id)),
        }
    }

    /// List all agents currently in the pool with their status.
    pub async fn list_active(&self) -> Vec<(String, TaskAgentStatus)> {
        let agents = self.agents.read().await;
        let mut out = Vec::with_capacity(agents.len());
        for (id, handle) in agents.iter() {
            out.push((id.clone(), handle.agent.status().await));
        }
        out
    }

    /// Number of agents currently in the pool (running or pending cleanup).
    pub async fn active_count(&self) -> usize {
        self.agents.read().await.len()
    }

    /// Returns `true` if the agent is still running (join handle not finished).
    pub async fn is_running(&self, agent_id: &str) -> bool {
        let agents = self.agents.read().await;
        agents
            .get(agent_id)
            .map(|h| !h.join_handle.is_finished())
            .unwrap_or(false)
    }

    /// Remove all finished agents from the pool and return their results.
    pub async fn cleanup_completed(&self) -> Vec<(String, Result<TaskAgentResult>)> {
        let finished: Vec<String> = {
            let agents = self.agents.read().await;
            agents
                .iter()
                .filter(|(_, h)| h.join_handle.is_finished())
                .map(|(id, _)| id.clone())
                .collect()
        };

        let mut results = Vec::new();
        let mut agents = self.agents.write().await;
        for id in finished {
            if let Some(handle) = agents.remove(&id) {
                let result = match handle.join_handle.await {
                    Ok(r) => r,
                    Err(e) => Err(anyhow!("Agent task panicked: {}", e)),
                };
                results.push((id, result));
            }
        }
        results
    }

    /// Wait for every agent in the pool to finish.
    pub async fn await_all(&self) -> Vec<(String, Result<TaskAgentResult>)> {
        let ids: Vec<String> = self.agents.read().await.keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            results.push((id.clone(), self.await_completion(&id).await));
        }
        results
    }

    /// Abort all agents and clear the pool.
    pub async fn shutdown(&self) {
        let mut agents = self.agents.write().await;
        for (agent_id, handle) in agents.drain() {
            handle.join_handle.abort();
            self.file_lock_manager.release_all_locks(&agent_id).await;
        }
        tracing::info!("agent pool shut down");
    }

    /// Get a statistical snapshot of the pool.
    pub async fn stats(&self) -> AgentPoolStats {
        let agents = self.agents.read().await;
        let mut running = 0usize;
        let mut completed = 0usize;

        for (_, handle) in agents.iter() {
            if handle.join_handle.is_finished() {
                completed += 1;
            } else {
                running += 1;
            }
        }

        AgentPoolStats {
            max_agents: self.max_agents,
            total_agents: agents.len(),
            running,
            completed,
            failed: 0, // Not distinguishable without awaiting the handle.
        }
    }

    /// Get the shared file lock manager.
    pub fn file_lock_manager(&self) -> Arc<FileLockManager> {
        Arc::clone(&self.file_lock_manager)
    }

    /// Get the shared communication hub.
    pub fn communication_hub(&self) -> Arc<CommunicationHub> {
        Arc::clone(&self.communication_hub)
    }
}

/// Statistics about the agent pool.
#[derive(Debug, Clone)]
pub struct AgentPoolStats {
    /// Maximum concurrent agents allowed.
    pub max_agents: usize,
    /// Total agents currently tracked (running + awaiting cleanup).
    pub total_agents: usize,
    /// Agents that are currently running.
    pub running: usize,
    /// Agents that have finished but not yet cleaned up.
    pub completed: usize,
    /// Agents that are known to have failed (requires awaiting the handle).
    pub failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use brainwires_agent::communication::CommunicationHub;
    use brainwires_agent::file_locks::FileLockManager;
    use brainwires_core::{
        ChatOptions, ChatResponse, Message, StreamChunk, Tool, ToolContext, ToolResult, ToolUse,
        Usage,
    };
    use brainwires_tool_runtime::ToolExecutor;
    use futures::stream::BoxStream;

    struct MockProvider(ChatResponse);

    impl MockProvider {
        fn done(text: &str) -> Self {
            Self(ChatResponse {
                message: Message::assistant(text),
                finish_reason: Some("stop".to_string()),
                usage: Usage::default(),
            })
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn chat(
            &self,
            _: &[Message],
            _: Option<&[Tool]>,
            _: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(self.0.clone())
        }

        fn stream_chat<'a>(
            &'a self,
            _: &'a [Message],
            _: Option<&'a [Tool]>,
            _: &'a ChatOptions,
        ) -> BoxStream<'a, Result<StreamChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    struct NoOpExecutor;

    #[async_trait]
    impl ToolExecutor for NoOpExecutor {
        async fn execute(&self, tu: &ToolUse, _: &ToolContext) -> Result<ToolResult> {
            Ok(ToolResult::success(tu.id.clone(), "ok".to_string()))
        }

        fn available_tools(&self) -> Vec<Tool> {
            vec![]
        }
    }

    fn make_pool(max: usize) -> AgentPool {
        AgentPool::new(
            max,
            Arc::new(MockProvider::done("Done")),
            Arc::new(NoOpExecutor),
            Arc::new(CommunicationHub::new()),
            Arc::new(FileLockManager::new()),
            "/tmp",
        )
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let pool = make_pool(5);
        assert_eq!(pool.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_spawn_and_count() {
        let pool = make_pool(5);
        let _ = pool
            .spawn_agent(
                Task::new("t-1", "Test"),
                Some(TaskAgentConfig {
                    validation_config: None,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(pool.active_count().await, 1);
    }

    #[tokio::test]
    async fn test_max_agents_limit() {
        let pool = make_pool(2);
        let cfg = || {
            Some(TaskAgentConfig {
                validation_config: None,
                ..Default::default()
            })
        };

        pool.spawn_agent(Task::new("t-1", "T1"), cfg())
            .await
            .unwrap();
        pool.spawn_agent(Task::new("t-2", "T2"), cfg())
            .await
            .unwrap();

        let err = pool.spawn_agent(Task::new("t-3", "T3"), cfg()).await;
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("full"));
    }

    #[tokio::test]
    async fn test_await_completion() {
        let pool = make_pool(5);
        let id = pool
            .spawn_agent(
                Task::new("t-1", "Finish me"),
                Some(TaskAgentConfig {
                    validation_config: None,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        let result = pool.await_completion(&id).await.unwrap();
        assert!(result.success);
        assert_eq!(result.task_id, "t-1");
    }

    #[tokio::test]
    async fn test_stop_agent() {
        let pool = make_pool(5);
        let id = pool.spawn_agent(Task::new("t-1", "T"), None).await.unwrap();

        pool.stop_agent(&id).await.unwrap();
        assert_eq!(pool.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let pool = make_pool(5);
        pool.spawn_agent(Task::new("t-1", "T1"), None)
            .await
            .unwrap();
        pool.spawn_agent(Task::new("t-2", "T2"), None)
            .await
            .unwrap();

        pool.shutdown().await;
        assert_eq!(pool.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let pool = make_pool(10);
        let stats = pool.stats().await;
        assert_eq!(stats.max_agents, 10);
        assert_eq!(stats.total_agents, 0);
    }

    #[tokio::test]
    async fn test_list_active() {
        let pool = make_pool(5);
        pool.spawn_agent(Task::new("t-1", "T1"), None)
            .await
            .unwrap();
        pool.spawn_agent(Task::new("t-2", "T2"), None)
            .await
            .unwrap();

        let active = pool.list_active().await;
        assert_eq!(active.len(), 2);
    }
}
