//! Cycle Orchestrator - Plan→Work→Judge loop
//!
//! [`CycleOrchestrator`] implements the Planner-Worker-Judge pattern for
//! scaling multi-agent coding tasks. Each cycle:
//!
//! 1. **Plan**: A [`PlannerAgent`] explores the codebase and creates tasks
//! 2. **Work**: Workers execute tasks independently (optionally in worktrees)
//! 3. **Merge**: Worker branches are merged in dependency order
//! 4. **Judge**: A [`JudgeAgent`] evaluates results and decides next steps
//!
//! This pattern combats agent drift and tunnel vision by enabling fresh starts
//! between cycles, and eliminates file lock contention by giving each worker
//! its own worktree.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use brainwires_core::{Provider, TaskPriority};
use brainwires_tool_runtime::ToolExecutor;

use crate::context::AgentContext;
use crate::judge_agent::{
    JudgeAgent, JudgeAgentConfig, JudgeContext, JudgeVerdict, MergeStatus, WorkerResult,
};
use crate::planner_agent::{
    DynamicTaskSpec, PlannerAgent, PlannerAgentConfig, PlannerOutput, SubPlannerRequest,
};
use crate::pool::AgentPool;
use crate::task_agent::{TaskAgentConfig, TaskAgentResult};
use crate::task_orchestrator::{FailurePolicy, TaskOrchestratorConfig};
use brainwires_agent::communication::{AgentMessage, CommunicationHub};
use brainwires_agent::file_locks::FileLockManager;
use brainwires_agent::task_manager::TaskManager;

#[cfg(feature = "native")]
use brainwires_agent::worktree::WorktreeManager;

// ── Public types ────────────────────────────────────────────────────────────

/// Strategy for merging worker branches back into the target.
#[derive(Debug, Clone, Default)]
pub enum MergeStrategy {
    /// Merge each branch in dependency order.
    #[default]
    Sequential,
    /// Rebase each branch on top of the previous merge.
    RebaseSequential,
}

/// Configuration for the cycle orchestrator.
#[derive(Debug, Clone)]
pub struct CycleOrchestratorConfig {
    /// Maximum number of Plan→Work→Judge cycles. Default: 5.
    pub max_cycles: u32,
    /// Maximum concurrent workers. Default: 5.
    pub max_workers: usize,
    /// Planner agent configuration.
    pub planner_config: PlannerAgentConfig,
    /// Judge agent configuration.
    pub judge_config: JudgeAgentConfig,
    /// Default worker agent configuration.
    pub worker_config: TaskAgentConfig,
    /// Whether to use git worktrees for worker isolation. Default: true.
    #[cfg(feature = "native")]
    pub use_worktrees: bool,
    /// Whether to automatically merge worker branches. Default: true.
    pub auto_merge: bool,
    /// Branch merge strategy.
    pub merge_strategy: MergeStrategy,
    /// What to do when a worker fails.
    pub failure_policy: FailurePolicy,
}

impl Default for CycleOrchestratorConfig {
    fn default() -> Self {
        Self {
            max_cycles: 5,
            max_workers: 5,
            planner_config: PlannerAgentConfig::default(),
            judge_config: JudgeAgentConfig::default(),
            worker_config: TaskAgentConfig::default(),
            #[cfg(feature = "native")]
            use_worktrees: true,
            auto_merge: true,
            merge_strategy: MergeStrategy::default(),
            failure_policy: FailurePolicy::ContinueOnFailure,
        }
    }
}

/// Record of a single Plan→Work→Judge cycle.
#[derive(Debug, Clone)]
pub struct CycleRecord {
    /// Cycle number (0-indexed).
    pub cycle_number: u32,
    /// What the planner produced.
    pub planner_output: PlannerOutput,
    /// Results from all workers.
    pub worker_results: Vec<WorkerResult>,
    /// The judge's verdict.
    pub verdict: JudgeVerdict,
    /// Wall-clock duration of the entire cycle.
    pub duration_secs: f64,
}

/// Final result of the orchestration.
#[derive(Debug)]
pub struct CycleOrchestratorResult {
    /// Whether the goal was achieved.
    pub success: bool,
    /// Number of cycles used.
    pub cycles_used: u32,
    /// Total tasks completed across all cycles.
    pub total_tasks_completed: usize,
    /// Total tasks failed across all cycles.
    pub total_tasks_failed: usize,
    /// The final verdict from the last judge.
    pub final_verdict: JudgeVerdict,
    /// History of all cycles.
    pub cycle_history: Vec<CycleRecord>,
}

// ── CycleOrchestrator ───────────────────────────────────────────────────────

/// Orchestrates the Plan→Work→Judge loop for multi-agent coding tasks.
pub struct CycleOrchestrator {
    provider: Arc<dyn Provider>,
    tool_executor: Arc<dyn ToolExecutor>,
    communication_hub: Arc<CommunicationHub>,
    file_lock_manager: Arc<FileLockManager>,
    working_directory: String,
    config: CycleOrchestratorConfig,
    /// Worktree manager for creating per-worker isolated branches.
    /// Currently stored for future worktree-based worker execution.
    #[cfg(feature = "native")]
    _worktree_manager: Option<Arc<WorktreeManager>>,
}

impl CycleOrchestrator {
    /// Create a new cycle orchestrator.
    pub fn new(
        provider: Arc<dyn Provider>,
        tool_executor: Arc<dyn ToolExecutor>,
        communication_hub: Arc<CommunicationHub>,
        file_lock_manager: Arc<FileLockManager>,
        working_directory: impl Into<String>,
        config: CycleOrchestratorConfig,
    ) -> Self {
        let working_directory = working_directory.into();

        #[cfg(feature = "native")]
        let _worktree_manager = if config.use_worktrees {
            Some(Arc::new(WorktreeManager::new(&working_directory)))
        } else {
            None
        };

        Self {
            provider,
            tool_executor,
            communication_hub,
            file_lock_manager,
            working_directory,
            config,
            #[cfg(feature = "native")]
            _worktree_manager,
        }
    }

    /// Run the Plan→Work→Judge loop until the goal is achieved or limits are hit.
    pub async fn run(&self, goal: &str) -> Result<CycleOrchestratorResult> {
        let mut cycle_history: Vec<CycleRecord> = Vec::new();
        let mut hints: Vec<String> = Vec::new();
        let mut previous_verdicts: Vec<JudgeVerdict> = Vec::new();
        let mut total_completed = 0usize;
        let mut total_failed = 0usize;

        for cycle_number in 0..self.config.max_cycles {
            let cycle_start = Instant::now();

            tracing::info!(cycle = cycle_number, "Starting Plan→Work→Judge cycle");

            // Broadcast cycle start
            let _ = self
                .communication_hub
                .broadcast(
                    "cycle-orchestrator".to_string(),
                    AgentMessage::CycleStarted {
                        cycle_number,
                        goal: goal.to_string(),
                    },
                )
                .await;

            // ── PHASE 1: PLAN ───────────────────────────────────────────
            tracing::info!(cycle = cycle_number, "Phase 1: Planning");
            let planner_output = self.run_planner(goal, &hints, cycle_number).await?;

            if planner_output.tasks.is_empty() {
                tracing::warn!(cycle = cycle_number, "Planner produced no tasks");
                let verdict = JudgeVerdict::Complete {
                    summary: "Planner determined no tasks needed".to_string(),
                };
                cycle_history.push(CycleRecord {
                    cycle_number,
                    planner_output,
                    worker_results: vec![],
                    verdict: verdict.clone(),
                    duration_secs: cycle_start.elapsed().as_secs_f64(),
                });
                return Ok(CycleOrchestratorResult {
                    success: true,
                    cycles_used: cycle_number + 1,
                    total_tasks_completed: total_completed,
                    total_tasks_failed: total_failed,
                    final_verdict: verdict,
                    cycle_history,
                });
            }

            // Broadcast plan
            let _ = self
                .communication_hub
                .broadcast(
                    "cycle-orchestrator".to_string(),
                    AgentMessage::PlanCreated {
                        cycle_number,
                        task_count: planner_output.tasks.len(),
                        rationale: planner_output.rationale.clone(),
                    },
                )
                .await;

            // ── PHASE 2+3: EXECUTE WORKERS ──────────────────────────────
            tracing::info!(
                cycle = cycle_number,
                tasks = planner_output.tasks.len(),
                "Phase 2-3: Executing workers"
            );
            let worker_results = self.run_workers(&planner_output, cycle_number).await?;

            let cycle_completed = worker_results
                .iter()
                .filter(|r| r.agent_result.success)
                .count();
            let cycle_failed = worker_results
                .iter()
                .filter(|r| !r.agent_result.success)
                .count();
            total_completed += cycle_completed;
            total_failed += cycle_failed;

            // ── PHASE 4: MERGE ──────────────────────────────────────────
            // Merge status is set during run_workers (worktree-based) or
            // defaults to NotAttempted (same-directory workers).

            // ── PHASE 5: JUDGE ──────────────────────────────────────────
            tracing::info!(
                cycle = cycle_number,
                completed = cycle_completed,
                failed = cycle_failed,
                "Phase 5: Judging"
            );

            let judge_context = JudgeContext {
                original_goal: goal.to_string(),
                cycle_number,
                worker_results: worker_results.clone(),
                planner_rationale: planner_output.rationale.clone(),
                previous_verdicts: previous_verdicts.clone(),
            };

            let verdict = self.run_judge(&judge_context).await?;

            // Broadcast cycle completion
            let _ = self
                .communication_hub
                .broadcast(
                    "cycle-orchestrator".to_string(),
                    AgentMessage::CycleCompleted {
                        cycle_number,
                        verdict_type: verdict.verdict_type().to_string(),
                    },
                )
                .await;

            let record = CycleRecord {
                cycle_number,
                planner_output,
                worker_results,
                verdict: verdict.clone(),
                duration_secs: cycle_start.elapsed().as_secs_f64(),
            };
            cycle_history.push(record);

            // ── Process verdict ──────────────────────────────────────────
            match &verdict {
                JudgeVerdict::Complete { summary } => {
                    tracing::info!(cycle = cycle_number, summary = %summary, "Goal achieved!");
                    return Ok(CycleOrchestratorResult {
                        success: true,
                        cycles_used: cycle_number + 1,
                        total_tasks_completed: total_completed,
                        total_tasks_failed: total_failed,
                        final_verdict: verdict,
                        cycle_history,
                    });
                }
                JudgeVerdict::Continue {
                    hints: new_hints, ..
                } => {
                    tracing::info!(cycle = cycle_number, "Continuing to next cycle");
                    hints = new_hints.clone();
                }
                JudgeVerdict::FreshRestart {
                    reason,
                    hints: new_hints,
                    ..
                } => {
                    tracing::info!(cycle = cycle_number, reason = %reason, "Fresh restart");
                    hints = new_hints.clone();
                }
                JudgeVerdict::Abort { reason, .. } => {
                    tracing::warn!(cycle = cycle_number, reason = %reason, "Aborting");
                    return Ok(CycleOrchestratorResult {
                        success: false,
                        cycles_used: cycle_number + 1,
                        total_tasks_completed: total_completed,
                        total_tasks_failed: total_failed,
                        final_verdict: verdict,
                        cycle_history,
                    });
                }
            }

            previous_verdicts.push(verdict);
        }

        // Exhausted max cycles
        let final_verdict = JudgeVerdict::Abort {
            reason: format!("Exhausted maximum {} cycles", self.config.max_cycles),
            summary: "Max cycles reached without completing the goal".to_string(),
        };

        Ok(CycleOrchestratorResult {
            success: false,
            cycles_used: self.config.max_cycles,
            total_tasks_completed: total_completed,
            total_tasks_failed: total_failed,
            final_verdict,
            cycle_history,
        })
    }

    // ── Phase implementations ───────────────────────────────────────────────

    /// Run the planner phase, including sub-planners if requested.
    async fn run_planner(
        &self,
        goal: &str,
        hints: &[String],
        cycle_number: u32,
    ) -> Result<PlannerOutput> {
        let context = Arc::new(AgentContext::new(
            self.working_directory.clone(),
            Arc::clone(&self.tool_executor),
            Arc::clone(&self.communication_hub),
            Arc::clone(&self.file_lock_manager),
        ));

        let planner = PlannerAgent::new(
            format!("planner-cycle-{}", cycle_number),
            goal,
            hints,
            Arc::clone(&self.provider),
            context,
            self.config.planner_config.clone(),
        );

        let (mut output, _result) = planner.execute().await?;

        // Handle sub-planners recursively
        if !output.sub_planners.is_empty() && self.config.planner_config.planning_depth > 0 {
            let sub_tasks = self
                .run_sub_planners(&output.sub_planners, cycle_number, 1)
                .await?;
            output.tasks.extend(sub_tasks);
            // Re-enforce task limit after merging sub-planner output
            output.tasks.truncate(self.config.planner_config.max_tasks);
        }

        Ok(output)
    }

    /// Recursively spawn sub-planners and merge their task outputs.
    async fn run_sub_planners(
        &self,
        requests: &[SubPlannerRequest],
        cycle_number: u32,
        current_depth: u32,
    ) -> Result<Vec<DynamicTaskSpec>> {
        if current_depth >= self.config.planner_config.planning_depth {
            return Ok(vec![]);
        }

        let mut all_tasks = Vec::new();

        for (i, req) in requests.iter().enumerate() {
            if req.max_depth == 0 {
                continue;
            }

            let sub_goal = format!("{}\n\nContext: {}", req.focus_area, req.context);
            let context = Arc::new(AgentContext::new(
                self.working_directory.clone(),
                Arc::clone(&self.tool_executor),
                Arc::clone(&self.communication_hub),
                Arc::clone(&self.file_lock_manager),
            ));

            let sub_config = PlannerAgentConfig {
                planning_depth: req.max_depth.saturating_sub(1),
                ..self.config.planner_config.clone()
            };

            let sub_planner = PlannerAgent::new(
                format!("sub-planner-c{}-{}", cycle_number, i),
                &sub_goal,
                &[],
                Arc::clone(&self.provider),
                context,
                sub_config,
            );

            match sub_planner.execute().await {
                Ok((sub_output, _)) => {
                    all_tasks.extend(sub_output.tasks);
                }
                Err(e) => {
                    tracing::warn!(
                        sub_planner = i,
                        error = %e,
                        "Sub-planner failed, skipping"
                    );
                }
            }
        }

        Ok(all_tasks)
    }

    /// Run the worker phase: create tasks, execute them, collect results.
    async fn run_workers(
        &self,
        planner_output: &PlannerOutput,
        cycle_number: u32,
    ) -> Result<Vec<WorkerResult>> {
        let task_manager = Arc::new(TaskManager::new());
        let pool = Arc::new(AgentPool::new(
            self.config.max_workers,
            Arc::clone(&self.provider),
            Arc::clone(&self.tool_executor),
            Arc::clone(&self.communication_hub),
            Arc::clone(&self.file_lock_manager),
            self.working_directory.clone(),
        ));

        // Build a mapping from planner spec ID -> task manager task ID
        let mut spec_to_task: HashMap<String, String> = HashMap::new();

        // Create a parent task for this cycle
        let parent_id = task_manager
            .create_task(
                format!("Cycle {} tasks", cycle_number),
                None,
                TaskPriority::Normal,
            )
            .await?;

        // Create tasks from planner specs
        for spec in &planner_output.tasks {
            let priority: TaskPriority = spec.priority.clone().into();
            let task_id = task_manager
                .create_task(spec.description.clone(), Some(parent_id.clone()), priority)
                .await?;
            spec_to_task.insert(spec.id.clone(), task_id);
        }

        // Wire up dependencies
        for spec in &planner_output.tasks {
            if let Some(task_id) = spec_to_task.get(&spec.id) {
                for dep_spec_id in &spec.depends_on {
                    if let Some(dep_task_id) = spec_to_task.get(dep_spec_id) {
                        task_manager.add_dependency(task_id, dep_task_id).await?;
                    }
                }
            }
        }

        // Build per-task config overrides
        let orchestrator_config = TaskOrchestratorConfig {
            failure_policy: self.config.failure_policy.clone(),
            default_agent_config: self.config.worker_config.clone(),
            orchestrator_id: format!("cycle-{}-orchestrator", cycle_number),
            ..Default::default()
        };

        let orchestrator = crate::task_orchestrator::TaskOrchestrator::new(
            task_manager.clone(),
            pool.clone(),
            Arc::clone(&self.communication_hub),
            orchestrator_config,
        );

        // Set per-task configs from planner specs
        for spec in &planner_output.tasks {
            if let Some(ref override_config) = spec.agent_config_override
                && let Some(task_id) = spec_to_task.get(&spec.id)
            {
                orchestrator
                    .set_task_config(task_id, override_config.clone())
                    .await;
            }
        }

        // Run the orchestrator
        let orch_result = orchestrator.run(&parent_id).await?;

        // Build worker results
        let mut worker_results = Vec::new();
        let task_id_to_spec: HashMap<&str, &DynamicTaskSpec> = spec_to_task
            .iter()
            .flat_map(|(spec_id, task_id)| {
                planner_output
                    .tasks
                    .iter()
                    .find(|s| s.id == *spec_id)
                    .map(|spec| (task_id.as_str(), spec))
            })
            .collect();

        for (task_id, agent_result) in &orch_result.task_results {
            let description = task_id_to_spec
                .get(task_id.as_str())
                .map(|s| s.description.clone())
                .unwrap_or_else(|| "unknown task".to_string());

            let branch_name = format!(
                "cycle-{}-{}",
                cycle_number,
                &task_id[..8.min(task_id.len())]
            );

            worker_results.push(WorkerResult {
                task_id: task_id.clone(),
                task_description: description,
                agent_result: agent_result.clone(),
                branch_name,
                merge_status: MergeStatus::NotAttempted,
            });
        }

        // Include unstarted tasks as failed results
        for task_id in &orch_result.unstarted_tasks {
            // Skip the parent task
            if *task_id == parent_id {
                continue;
            }

            let description = task_id_to_spec
                .get(task_id.as_str())
                .map(|s| s.description.clone())
                .unwrap_or_else(|| "unstarted task".to_string());

            let now = chrono::Utc::now();
            let graph = brainwires_agent::execution_graph::ExecutionGraph::new(String::new(), now);
            let telemetry = brainwires_agent::execution_graph::RunTelemetry::from_graph(
                &graph, now, false, 0.0,
            );
            worker_results.push(WorkerResult {
                task_id: task_id.clone(),
                task_description: description,
                agent_result: TaskAgentResult {
                    agent_id: String::new(),
                    task_id: task_id.clone(),
                    success: false,
                    summary: "Task was never started (blocked or halted)".to_string(),
                    iterations: 0,
                    replan_count: 0,
                    budget_exhausted: false,
                    partial_output: None,
                    total_tokens_used: 0,
                    total_cost_usd: 0.0,
                    timed_out: false,
                    failure_category: None,
                    execution_graph: graph,
                    telemetry,
                    pre_execution_plan: None,
                },
                branch_name: String::new(),
                merge_status: MergeStatus::NotAttempted,
            });
        }

        Ok(worker_results)
    }

    /// Run the judge phase.
    async fn run_judge(&self, judge_context: &JudgeContext) -> Result<JudgeVerdict> {
        let context = Arc::new(AgentContext::new(
            self.working_directory.clone(),
            Arc::clone(&self.tool_executor),
            Arc::clone(&self.communication_hub),
            Arc::clone(&self.file_lock_manager),
        ));

        let judge = JudgeAgent::new(
            format!("judge-cycle-{}", judge_context.cycle_number),
            judge_context,
            Arc::clone(&self.provider),
            context,
            self.config.judge_config.clone(),
        );

        let (verdict, _result) = judge.execute().await?;
        Ok(verdict)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CycleOrchestratorConfig {
        &self.config
    }

    /// Get the working directory.
    pub fn working_directory(&self) -> &str {
        &self.working_directory
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = CycleOrchestratorConfig::default();
        assert_eq!(config.max_cycles, 5);
        assert_eq!(config.max_workers, 5);
        assert!(config.auto_merge);
    }

    #[test]
    fn test_cycle_record() {
        let record = CycleRecord {
            cycle_number: 0,
            planner_output: PlannerOutput {
                tasks: vec![],
                sub_planners: vec![],
                rationale: "test".to_string(),
            },
            worker_results: vec![],
            verdict: JudgeVerdict::Complete {
                summary: "done".to_string(),
            },
            duration_secs: 1.5,
        };
        assert_eq!(record.cycle_number, 0);
        assert_eq!(record.duration_secs, 1.5);
    }

    #[test]
    fn test_orchestrator_result() {
        let result = CycleOrchestratorResult {
            success: true,
            cycles_used: 1,
            total_tasks_completed: 3,
            total_tasks_failed: 0,
            final_verdict: JudgeVerdict::Complete {
                summary: "all done".to_string(),
            },
            cycle_history: vec![],
        };
        assert!(result.success);
        assert_eq!(result.cycles_used, 1);
    }

    #[test]
    fn test_merge_strategy_default() {
        let strategy = MergeStrategy::default();
        assert!(matches!(strategy, MergeStrategy::Sequential));
    }
}
