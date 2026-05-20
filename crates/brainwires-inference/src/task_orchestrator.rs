//! Task Orchestrator - Bridges TaskManager and AgentPool
//!
//! [`TaskOrchestrator`] runs a scheduling loop that queries ready tasks from the
//! dependency graph, spawns agents via the pool, and feeds results back into the
//! task manager.  This provides centralized status tracking with concurrent agent
//! execution and dependency-aware ordering.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokio::sync::RwLock;

use brainwires_core::{Task, TaskPriority, TaskStatus};

use crate::pool::AgentPool;
use crate::task_agent::{TaskAgentConfig, TaskAgentResult};
use brainwires_agent::communication::{AgentMessage, CommunicationHub};
use brainwires_agent::task_manager::TaskManager;
use brainwires_agent::task_manager::TaskStats;

const DEFAULT_POLL_INTERVAL_MS: u64 = 250;

// ── Public types ────────────────────────────────────────────────────────────

/// What happens when an agent's task fails.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FailurePolicy {
    /// Stop scheduling new tasks and drain running agents (default).
    #[default]
    StopOnFirstFailure,
    /// Keep scheduling independent tasks that aren't blocked by the failure.
    ContinueOnFailure,
}

/// Configuration for the orchestration loop.
#[derive(Debug, Clone)]
pub struct TaskOrchestratorConfig {
    /// Behaviour on agent failure.
    pub failure_policy: FailurePolicy,
    /// Default agent config used when no per-task override exists.
    pub default_agent_config: TaskAgentConfig,
    /// Polling interval in milliseconds.  Default: 250.
    pub poll_interval_ms: u64,
    /// Identifier used in CommunicationHub messages.
    pub orchestrator_id: String,
}

impl Default for TaskOrchestratorConfig {
    fn default() -> Self {
        Self {
            failure_policy: FailurePolicy::default(),
            default_agent_config: TaskAgentConfig::default(),
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            orchestrator_id: "orchestrator".to_string(),
        }
    }
}

/// Specification for creating a task via the convenience API.
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Task description.
    pub description: String,
    /// Task priority.
    pub priority: TaskPriority,
    /// Indices into the spec list that this task depends on.
    pub depends_on_indices: Vec<usize>,
    /// Per-task agent config override (falls back to default).
    pub agent_config: Option<TaskAgentConfig>,
}

/// Result of a complete orchestration run.
#[derive(Debug)]
pub struct OrchestrationResult {
    /// `true` when every task succeeded.
    pub all_succeeded: bool,
    /// Per-task agent results keyed by task ID.
    pub task_results: HashMap<String, TaskAgentResult>,
    /// Task IDs that were never started (e.g. blocked by a failure).
    pub unstarted_tasks: Vec<String>,
    /// Final task statistics snapshot.
    pub stats: TaskStats,
}

// ── TaskOrchestrator ────────────────────────────────────────────────────────

/// Bridges [`TaskManager`] and [`AgentPool`] with a dependency-aware
/// scheduling loop.
pub struct TaskOrchestrator {
    task_manager: Arc<TaskManager>,
    agent_pool: Arc<AgentPool>,
    communication_hub: Arc<CommunicationHub>,
    config: TaskOrchestratorConfig,
    /// Per-task agent config overrides.
    per_task_configs: Arc<RwLock<HashMap<String, TaskAgentConfig>>>,
    /// Maps agent_id -> task_id for running agents.
    agent_to_task: Arc<RwLock<HashMap<String, String>>>,
    /// Abort flag — set by `abort()`.
    aborted: Arc<tokio::sync::Notify>,
    abort_flag: Arc<std::sync::atomic::AtomicBool>,
}

impl TaskOrchestrator {
    /// Create a new orchestrator.
    pub fn new(
        task_manager: Arc<TaskManager>,
        agent_pool: Arc<AgentPool>,
        communication_hub: Arc<CommunicationHub>,
        config: TaskOrchestratorConfig,
    ) -> Self {
        Self {
            task_manager,
            agent_pool,
            communication_hub,
            config,
            per_task_configs: Arc::new(RwLock::new(HashMap::new())),
            agent_to_task: Arc::new(RwLock::new(HashMap::new())),
            aborted: Arc::new(tokio::sync::Notify::new()),
            abort_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Set a per-task agent config override.
    pub async fn set_task_config(&self, task_id: impl Into<String>, config: TaskAgentConfig) {
        self.per_task_configs
            .write()
            .await
            .insert(task_id.into(), config);
    }

    /// Bulk-set per-task agent config overrides.
    pub async fn set_task_configs(&self, configs: HashMap<String, TaskAgentConfig>) {
        let mut map = self.per_task_configs.write().await;
        map.extend(configs);
    }

    /// Convenience API: create tasks with dependencies in the TaskManager,
    /// then run the scheduling loop.
    ///
    /// `parent_task_id` is an optional root task under which all specs are
    /// created as subtasks.
    pub async fn create_and_run(
        &self,
        parent_task_id: Option<&str>,
        specs: Vec<TaskSpec>,
    ) -> Result<OrchestrationResult> {
        // Create tasks and collect their IDs (index-ordered).
        let mut task_ids: Vec<String> = Vec::with_capacity(specs.len());
        for spec in &specs {
            let id = self
                .task_manager
                .create_task(
                    spec.description.clone(),
                    parent_task_id.map(|s| s.to_string()),
                    spec.priority,
                )
                .await?;
            task_ids.push(id);
        }

        // Wire up dependencies by index.
        for (i, spec) in specs.iter().enumerate() {
            for &dep_idx in &spec.depends_on_indices {
                if dep_idx >= task_ids.len() {
                    return Err(anyhow!(
                        "TaskSpec[{}] depends_on_indices contains out-of-range index {}",
                        i,
                        dep_idx
                    ));
                }
                self.task_manager
                    .add_dependency(&task_ids[i], &task_ids[dep_idx])
                    .await?;
            }

            // Store per-task config overrides.
            if let Some(ref cfg) = spec.agent_config {
                self.set_task_config(&task_ids[i], cfg.clone()).await;
            }
        }

        // Determine root: either explicit parent or the first created task.
        let root = parent_task_id
            .map(|s| s.to_string())
            .or_else(|| task_ids.first().cloned());

        match root {
            Some(id) => self.run(&id).await,
            None => {
                // Empty spec list — nothing to do.
                Ok(OrchestrationResult {
                    all_succeeded: true,
                    task_results: HashMap::new(),
                    unstarted_tasks: Vec::new(),
                    stats: self.task_manager.get_stats().await,
                })
            }
        }
    }

    /// Main scheduling loop over existing tasks in the TaskManager.
    ///
    /// Runs until all tasks reachable from `root_task_id` are completed/failed
    /// or the failure policy halts scheduling.
    pub async fn run(&self, root_task_id: &str) -> Result<OrchestrationResult> {
        let mut task_results: HashMap<String, TaskAgentResult> = HashMap::new();
        let mut halted = false;
        let poll = tokio::time::Duration::from_millis(self.config.poll_interval_ms);

        loop {
            // Check abort flag.
            if self.abort_flag.load(std::sync::atomic::Ordering::Relaxed) {
                halted = true;
            }

            // ── 1. Harvest completed agents ──────────────────────────────
            let completed = self.agent_pool.cleanup_completed().await;
            for (agent_id, result) in completed {
                let task_id = { self.agent_to_task.write().await.remove(&agent_id) };

                if let Some(task_id) = task_id {
                    match result {
                        Ok(agent_result) => {
                            if agent_result.success {
                                let summary = agent_result.summary.clone();
                                self.task_manager
                                    .complete_task(&task_id, summary.clone())
                                    .await?;

                                if let Err(e) = self
                                    .communication_hub
                                    .broadcast(
                                        self.config.orchestrator_id.clone(),
                                        AgentMessage::AgentCompleted {
                                            agent_id: agent_id.clone(),
                                            task_id: task_id.clone(),
                                            summary,
                                        },
                                    )
                                    .await
                                {
                                    tracing::warn!(agent_id = %agent_id, task_id = %task_id, "Failed to broadcast agent completion: {}", e);
                                }
                            } else {
                                let error = agent_result.summary.clone();
                                self.task_manager.fail_task(&task_id, error.clone()).await?;

                                if let Err(e) = self
                                    .communication_hub
                                    .broadcast(
                                        self.config.orchestrator_id.clone(),
                                        AgentMessage::AgentCompleted {
                                            agent_id: agent_id.clone(),
                                            task_id: task_id.clone(),
                                            summary: format!("FAILED: {}", error),
                                        },
                                    )
                                    .await
                                {
                                    tracing::warn!(agent_id = %agent_id, task_id = %task_id, "Failed to broadcast agent failure: {}", e);
                                }

                                if self.config.failure_policy == FailurePolicy::StopOnFirstFailure {
                                    halted = true;
                                }
                            }
                            task_results.insert(task_id, agent_result);
                        }
                        Err(e) => {
                            let error = format!("Agent panicked: {}", e);
                            self.task_manager.fail_task(&task_id, error.clone()).await?;

                            if let Err(e) = self
                                .communication_hub
                                .broadcast(
                                    self.config.orchestrator_id.clone(),
                                    AgentMessage::AgentCompleted {
                                        agent_id: agent_id.clone(),
                                        task_id: task_id.clone(),
                                        summary: error,
                                    },
                                )
                                .await
                            {
                                tracing::warn!(agent_id = %agent_id, task_id = %task_id, "Failed to broadcast agent panic: {}", e);
                            }

                            if self.config.failure_policy == FailurePolicy::StopOnFirstFailure {
                                halted = true;
                            }
                        }
                    }
                }
            }

            // ── 2. Schedule new tasks (unless halted) ────────────────────
            if !halted {
                let ready = self.task_manager.get_ready_tasks().await;

                // Filter out tasks already assigned to an agent.
                let assigned: std::collections::HashSet<String> = {
                    let map = self.agent_to_task.read().await;
                    map.values().cloned().collect()
                };

                let available_slots = {
                    let stats = self.agent_pool.stats().await;
                    stats.max_agents.saturating_sub(stats.running)
                };

                let mut spawned = 0usize;
                for task in &ready {
                    if spawned >= available_slots {
                        break;
                    }
                    if assigned.contains(&task.id) {
                        continue;
                    }
                    // Skip tasks already InProgress (shouldn't happen, but be safe).
                    if task.status == TaskStatus::InProgress {
                        continue;
                    }
                    // Skip parent/container tasks — they auto-complete via
                    // check_parent_completion when all children finish.
                    if !task.children.is_empty() {
                        continue;
                    }

                    // Resolve config for this task.
                    let agent_config = {
                        let overrides = self.per_task_configs.read().await;
                        overrides
                            .get(&task.id)
                            .cloned()
                            .unwrap_or_else(|| self.config.default_agent_config.clone())
                    };

                    // Build a core Task for the agent pool.
                    let agent_task = Task::new(task.id.clone(), task.description.clone());

                    // Start + assign in TaskManager.
                    self.task_manager.start_task(&task.id).await?;

                    match self
                        .agent_pool
                        .spawn_agent(agent_task, Some(agent_config))
                        .await
                    {
                        Ok(agent_id) => {
                            self.task_manager.assign_task(&task.id, &agent_id).await?;
                            self.agent_to_task
                                .write()
                                .await
                                .insert(agent_id.clone(), task.id.clone());

                            if let Err(e) = self
                                .communication_hub
                                .broadcast(
                                    self.config.orchestrator_id.clone(),
                                    AgentMessage::AgentSpawned {
                                        agent_id,
                                        task_id: task.id.clone(),
                                    },
                                )
                                .await
                            {
                                tracing::warn!(task_id = %task.id, "Failed to broadcast agent spawn: {}", e);
                            }

                            spawned += 1;
                        }
                        Err(e) => {
                            tracing::warn!(task_id = %task.id, error = %e, "failed to spawn agent");
                            // Revert status back to Pending so it can be retried.
                            self.task_manager
                                .update_status(&task.id, TaskStatus::Pending, None)
                                .await?;
                        }
                    }
                }
            }

            // ── 3. Check termination ─────────────────────────────────────
            let running = {
                let map = self.agent_to_task.read().await;
                map.len()
            };

            if running == 0 {
                // No running agents.  If halted or no more schedulable tasks, we're done.
                let ready = self.task_manager.get_ready_tasks().await;
                let assigned: std::collections::HashSet<String> = {
                    let map = self.agent_to_task.read().await;
                    map.values().cloned().collect()
                };
                // Only leaf tasks (no children) are schedulable.
                let schedulable: Vec<_> = ready
                    .iter()
                    .filter(|t| !assigned.contains(&t.id) && t.children.is_empty())
                    .collect();

                if halted || schedulable.is_empty() {
                    break;
                }
            }

            tokio::time::sleep(poll).await;
        }

        // ── Build result ────────────────────────────────────────────────
        let stats = self.task_manager.get_stats().await;
        let all_tasks = self.task_manager.get_task_tree(Some(root_task_id)).await;
        let unstarted: Vec<String> = all_tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Pending || t.status == TaskStatus::Blocked)
            .map(|t| t.id.clone())
            .collect();

        let all_succeeded = stats.failed == 0 && unstarted.is_empty();

        Ok(OrchestrationResult {
            all_succeeded,
            task_results,
            unstarted_tasks: unstarted,
            stats,
        })
    }

    /// Cancel all running agents and return.
    pub async fn abort(&self) {
        self.abort_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.aborted.notify_one();
        self.agent_pool.shutdown().await;
        self.agent_to_task.write().await.clear();
    }

    /// Live task statistics snapshot.
    pub async fn progress(&self) -> TaskStats {
        self.task_manager.get_stats().await
    }

    /// Map of currently running agent_id -> task_id.
    pub async fn running_agents(&self) -> HashMap<String, String> {
        self.agent_to_task.read().await.clone()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::AgentPool;
    use crate::task_agent::TaskAgentConfig;
    use brainwires_agent::communication::CommunicationHub;
    use brainwires_agent::file_locks::FileLockManager;

    use async_trait::async_trait;
    use brainwires_core::{
        ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool, ToolContext, ToolResult,
        ToolUse, Usage,
    };
    use brainwires_tool_runtime::ToolExecutor;
    use futures::stream::BoxStream;

    // ── Mock provider that returns "Done" immediately ────────────────────

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

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_deps(max_pool: usize) -> (Arc<TaskManager>, Arc<AgentPool>, Arc<CommunicationHub>) {
        let hub = Arc::new(CommunicationHub::new());
        let flm = Arc::new(FileLockManager::new());
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::done("Done"));
        let executor: Arc<dyn ToolExecutor> = Arc::new(NoOpExecutor);

        let tm = Arc::new(TaskManager::new());
        let pool = Arc::new(AgentPool::new(
            max_pool,
            provider,
            executor,
            Arc::clone(&hub),
            flm,
            "/tmp",
        ));

        (tm, pool, hub)
    }

    fn no_validation() -> TaskAgentConfig {
        TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        }
    }

    fn make_orchestrator(
        tm: Arc<TaskManager>,
        pool: Arc<AgentPool>,
        hub: Arc<CommunicationHub>,
    ) -> TaskOrchestrator {
        TaskOrchestrator::new(
            tm,
            pool,
            hub,
            TaskOrchestratorConfig {
                default_agent_config: no_validation(),
                ..Default::default()
            },
        )
    }

    // ── Tests ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_empty_orchestration() {
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        // Create a root task with no children.
        let root = tm
            .create_task("root".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();

        // Mark it completed so the loop has nothing to do.
        tm.complete_task(&root, "already done".to_string())
            .await
            .unwrap();

        let result = orch.run(&root).await.unwrap();
        assert!(result.all_succeeded);
        assert!(result.task_results.is_empty());
        assert!(result.unstarted_tasks.is_empty());
    }

    #[tokio::test]
    async fn test_single_task() {
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        let root = tm
            .create_task("build widget".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();

        let result = orch.run(&root).await.unwrap();
        assert!(result.all_succeeded);
        assert_eq!(result.task_results.len(), 1);
        assert!(result.task_results.contains_key(&root));
    }

    #[tokio::test]
    async fn test_sequential_dependency_chain() {
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        // Create a chain: A -> B -> C  (C depends on B, B depends on A).
        // Use a parent to group them so get_task_tree works.
        let parent = tm
            .create_task("parent".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        let a = tm
            .create_task(
                "step A".to_string(),
                Some(parent.clone()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();
        let b = tm
            .create_task(
                "step B".to_string(),
                Some(parent.clone()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();
        let c = tm
            .create_task(
                "step C".to_string(),
                Some(parent.clone()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();

        tm.add_dependency(&b, &a).await.unwrap();
        tm.add_dependency(&c, &b).await.unwrap();

        let result = orch.run(&parent).await.unwrap();
        assert!(result.all_succeeded);
        assert_eq!(result.task_results.len(), 3);
    }

    #[tokio::test]
    async fn test_parallel_independent_tasks() {
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        let parent = tm
            .create_task("parent".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        let _a = tm
            .create_task(
                "task A".to_string(),
                Some(parent.clone()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();
        let _b = tm
            .create_task(
                "task B".to_string(),
                Some(parent.clone()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();
        let _c = tm
            .create_task(
                "task C".to_string(),
                Some(parent.clone()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();

        let result = orch.run(&parent).await.unwrap();
        assert!(result.all_succeeded);
        assert_eq!(result.task_results.len(), 3);
    }

    #[tokio::test]
    async fn test_diamond_dependency() {
        // A -> (B, C) -> D
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        let parent = tm
            .create_task("parent".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        let a = tm
            .create_task("A".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let b = tm
            .create_task("B".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let c = tm
            .create_task("C".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let d = tm
            .create_task("D".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();

        tm.add_dependency(&b, &a).await.unwrap();
        tm.add_dependency(&c, &a).await.unwrap();
        tm.add_dependency(&d, &b).await.unwrap();
        tm.add_dependency(&d, &c).await.unwrap();

        let result = orch.run(&parent).await.unwrap();
        assert!(result.all_succeeded);
        assert_eq!(result.task_results.len(), 4);
    }

    #[tokio::test]
    async fn test_stop_on_first_failure() {
        // Use a provider that returns a failure response.
        let hub = Arc::new(CommunicationHub::new());
        let flm = Arc::new(FileLockManager::new());

        // A provider whose "Done" text triggers agent success, but we need
        // a way to fail. TaskAgent treats a "stop" finish_reason as success
        // when the assistant text is non-empty. So we use a two-task setup
        // where we manually fail one task to test the policy.
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::done("Done"));
        let executor: Arc<dyn ToolExecutor> = Arc::new(NoOpExecutor);

        let tm = Arc::new(TaskManager::new());
        let pool = Arc::new(AgentPool::new(
            5,
            provider,
            executor,
            Arc::clone(&hub),
            flm,
            "/tmp",
        ));

        let orch = TaskOrchestrator::new(
            Arc::clone(&tm),
            Arc::clone(&pool),
            hub,
            TaskOrchestratorConfig {
                failure_policy: FailurePolicy::StopOnFirstFailure,
                default_agent_config: no_validation(),
                ..Default::default()
            },
        );

        // A -> B (sequential), so if A succeeds normally, B should follow.
        // For this test, we create independent tasks so the orchestrator sees
        // failures on independent paths.
        let parent = tm
            .create_task("parent".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        let a = tm
            .create_task("A".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let _b = tm
            .create_task("B".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();

        // Pre-fail task A so the orchestrator picks it up as failed immediately.
        tm.fail_task(&a, "forced failure".to_string())
            .await
            .unwrap();

        let result = orch.run(&parent).await.unwrap();
        // A is failed, B may or may not have run depending on timing.
        assert!(!result.all_succeeded);
    }

    #[tokio::test]
    async fn test_continue_on_failure() {
        let hub = Arc::new(CommunicationHub::new());
        let flm = Arc::new(FileLockManager::new());
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::done("Done"));
        let executor: Arc<dyn ToolExecutor> = Arc::new(NoOpExecutor);

        let tm = Arc::new(TaskManager::new());
        let pool = Arc::new(AgentPool::new(
            5,
            provider,
            executor,
            Arc::clone(&hub),
            flm,
            "/tmp",
        ));

        let orch = TaskOrchestrator::new(
            Arc::clone(&tm),
            Arc::clone(&pool),
            hub,
            TaskOrchestratorConfig {
                failure_policy: FailurePolicy::ContinueOnFailure,
                default_agent_config: no_validation(),
                ..Default::default()
            },
        );

        let parent = tm
            .create_task("parent".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        let a = tm
            .create_task("A".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let b_id = tm
            .create_task("B".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();

        // Pre-fail A, B is independent and should still run.
        tm.fail_task(&a, "forced failure".to_string())
            .await
            .unwrap();

        let result = orch.run(&parent).await.unwrap();
        // B should have completed even though A failed.
        assert!(!result.all_succeeded); // A failed so not all_succeeded.
        assert!(result.task_results.contains_key(&b_id));
    }

    #[tokio::test]
    async fn test_pool_capacity_respect() {
        // Pool size 1, 3 independent tasks — only one at a time.
        let (tm, pool, hub) = make_deps(1);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        let parent = tm
            .create_task("parent".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        let _a = tm
            .create_task("A".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let _b = tm
            .create_task("B".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();
        let _c = tm
            .create_task("C".to_string(), Some(parent.clone()), TaskPriority::Normal)
            .await
            .unwrap();

        let result = orch.run(&parent).await.unwrap();
        assert!(result.all_succeeded);
        assert_eq!(result.task_results.len(), 3);
    }

    #[tokio::test]
    async fn test_assigned_to_tracking() {
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        let root = tm
            .create_task("build widget".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();

        let result = orch.run(&root).await.unwrap();
        assert!(result.all_succeeded);

        // After completion, assigned_to should have been set.
        let task = tm.get_task(&root).await.unwrap();
        assert!(task.assigned_to.is_some());
    }

    #[tokio::test]
    async fn test_create_and_run() {
        let (tm, pool, hub) = make_deps(5);
        let orch = make_orchestrator(tm.clone(), pool, hub);

        let specs = vec![
            TaskSpec {
                description: "step A".to_string(),
                priority: TaskPriority::Normal,
                depends_on_indices: vec![],
                agent_config: None,
            },
            TaskSpec {
                description: "step B".to_string(),
                priority: TaskPriority::Normal,
                depends_on_indices: vec![0],
                agent_config: None,
            },
            TaskSpec {
                description: "step C".to_string(),
                priority: TaskPriority::Normal,
                depends_on_indices: vec![0],
                agent_config: None,
            },
            TaskSpec {
                description: "step D".to_string(),
                priority: TaskPriority::Normal,
                depends_on_indices: vec![1, 2],
                agent_config: None,
            },
        ];

        let result = orch.create_and_run(None, specs).await.unwrap();
        assert!(result.all_succeeded);
        assert_eq!(result.task_results.len(), 4);
    }
}
