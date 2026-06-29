//! Parallel coordinator — fan-out/fan-in for multi-agent task execution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A unit of work in a parallel execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelTask {
    /// Unique task identifier.
    pub id: String,
    /// Task description for the agent.
    pub description: String,
    /// Working directory.
    pub working_directory: String,
    /// Task dependencies (must complete before this task starts).
    pub depends_on: Vec<String>,
    /// Maximum iterations for this task.
    pub max_iterations: u32,
}

/// Result from a parallel task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelTaskResult {
    /// Identifier of the completed task.
    pub task_id: String,
    /// Whether the task succeeded.
    pub success: bool,
    /// Summary of the task result.
    pub summary: String,
    /// Number of iterations used.
    pub iterations: u32,
    /// Estimated cost in USD.
    pub cost: f64,
}

/// Status of a parallel execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelPlanStatus {
    /// Plan has not started executing.
    Pending,
    /// Plan is currently running.
    Running {
        /// Number of tasks completed so far.
        completed: usize,
        /// Total number of tasks in the plan.
        total: usize,
    },
    /// All tasks completed successfully.
    Completed {
        /// Results from all tasks.
        results: Vec<ParallelTaskResult>,
    },
    /// One or more tasks failed.
    Failed {
        /// Reason for the failure.
        reason: String,
        /// Results collected before the failure.
        partial_results: Vec<ParallelTaskResult>,
    },
}

/// Configuration for the parallel coordinator.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum concurrent agents.
    pub max_concurrent: usize,
    /// Whether to use MDAP voting for task results.
    pub use_mdap: bool,
    /// Fail fast: stop all tasks if any task fails.
    pub fail_fast: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 5,
            use_mdap: false,
            fail_fast: false,
        }
    }
}

/// Coordinates parallel execution of multiple agent tasks.
///
/// Handles task dependency resolution, concurrent execution limits,
/// and result aggregation.
pub struct ParallelCoordinator {
    config: ParallelConfig,
    tasks: Vec<ParallelTask>,
    results: HashMap<String, ParallelTaskResult>,
}

impl ParallelCoordinator {
    /// Create a new parallel coordinator with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            tasks: Vec::new(),
            results: HashMap::new(),
        }
    }

    /// Add a task to the execution plan.
    pub fn add_task(&mut self, task: ParallelTask) {
        self.tasks.push(task);
    }

    /// Get tasks that are ready to execute (all dependencies satisfied).
    pub fn ready_tasks(&self) -> Vec<&ParallelTask> {
        self.tasks
            .iter()
            .filter(|t| {
                !self.results.contains_key(&t.id)
                    && t.depends_on
                        .iter()
                        .all(|dep| self.results.get(dep).is_some_and(|r| r.success))
            })
            .collect()
    }

    /// Record the result of a completed task.
    pub fn record_result(&mut self, result: ParallelTaskResult) {
        self.results.insert(result.task_id.clone(), result);
    }

    /// Check if all tasks are completed.
    pub fn is_complete(&self) -> bool {
        self.tasks.iter().all(|t| self.results.contains_key(&t.id))
    }

    /// Check if any task has failed (relevant for fail-fast mode).
    pub fn has_failure(&self) -> bool {
        self.results.values().any(|r| !r.success)
    }

    /// Get the current plan status.
    pub fn status(&self) -> ParallelPlanStatus {
        if self.results.is_empty() && !self.tasks.is_empty() {
            return ParallelPlanStatus::Pending;
        }

        let completed = self.results.len();
        let total = self.tasks.len();

        if completed < total {
            if self.config.fail_fast && self.has_failure() {
                return ParallelPlanStatus::Failed {
                    reason: "fail-fast: a task failed".to_string(),
                    partial_results: self.results.values().cloned().collect(),
                };
            }
            return ParallelPlanStatus::Running { completed, total };
        }

        let results: Vec<ParallelTaskResult> = self.results.values().cloned().collect();
        if results.iter().all(|r| r.success) {
            ParallelPlanStatus::Completed { results }
        } else {
            ParallelPlanStatus::Failed {
                reason: "one or more tasks failed".to_string(),
                partial_results: results,
            }
        }
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> ParallelStats {
        let results: Vec<&ParallelTaskResult> = self.results.values().collect();
        ParallelStats {
            total_tasks: self.tasks.len(),
            completed: results.len(),
            succeeded: results.iter().filter(|r| r.success).count(),
            failed: results.iter().filter(|r| !r.success).count(),
            total_iterations: results.iter().map(|r| r.iterations as u64).sum(),
            total_cost: results.iter().map(|r| r.cost).sum(),
        }
    }
}

/// Aggregate statistics for a parallel execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelStats {
    /// Total number of tasks in the plan.
    pub total_tasks: usize,
    /// Number of tasks completed.
    pub completed: usize,
    /// Number of tasks that succeeded.
    pub succeeded: usize,
    /// Number of tasks that failed.
    pub failed: usize,
    /// Total iterations across all tasks.
    pub total_iterations: u64,
    /// Total cost across all tasks in USD.
    pub total_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task(id: &str, deps: Vec<&str>) -> ParallelTask {
        ParallelTask {
            id: id.to_string(),
            description: format!("Task {id}"),
            working_directory: "/tmp".to_string(),
            depends_on: deps.into_iter().map(|s| s.to_string()).collect(),
            max_iterations: 10,
        }
    }

    fn make_result(task_id: &str, success: bool) -> ParallelTaskResult {
        ParallelTaskResult {
            task_id: task_id.to_string(),
            success,
            summary: "done".to_string(),
            iterations: 5,
            cost: 0.01,
        }
    }

    #[test]
    fn new_coordinator_is_empty() {
        let coord = ParallelCoordinator::new(ParallelConfig::default());
        assert!(coord.is_complete()); // no tasks = vacuously complete
        assert!(!coord.has_failure());
    }

    #[test]
    fn add_task_and_ready_tasks() {
        let mut coord = ParallelCoordinator::new(ParallelConfig::default());
        coord.add_task(make_task("a", vec![]));
        coord.add_task(make_task("b", vec!["a"]));

        let ready = coord.ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "a");
    }

    #[test]
    fn record_result_unlocks_dependents() {
        let mut coord = ParallelCoordinator::new(ParallelConfig::default());
        coord.add_task(make_task("a", vec![]));
        coord.add_task(make_task("b", vec!["a"]));

        coord.record_result(make_result("a", true));
        let ready = coord.ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "b");
    }

    #[test]
    fn failed_dependency_blocks_dependents() {
        let mut coord = ParallelCoordinator::new(ParallelConfig::default());
        coord.add_task(make_task("a", vec![]));
        coord.add_task(make_task("b", vec!["a"]));

        coord.record_result(make_result("a", false));
        // "b" depends on "a" succeeding, so it should NOT be ready
        let ready = coord.ready_tasks();
        assert!(ready.is_empty());
    }

    #[test]
    fn is_complete_when_all_done() {
        let mut coord = ParallelCoordinator::new(ParallelConfig::default());
        coord.add_task(make_task("a", vec![]));
        coord.add_task(make_task("b", vec![]));

        assert!(!coord.is_complete());
        coord.record_result(make_result("a", true));
        assert!(!coord.is_complete());
        coord.record_result(make_result("b", true));
        assert!(coord.is_complete());
    }

    #[test]
    fn stats_aggregates_correctly() {
        let mut coord = ParallelCoordinator::new(ParallelConfig::default());
        coord.add_task(make_task("a", vec![]));
        coord.add_task(make_task("b", vec![]));
        coord.record_result(make_result("a", true));
        coord.record_result(make_result("b", false));

        let stats = coord.stats();
        assert_eq!(stats.total_tasks, 2);
        assert_eq!(stats.completed, 2);
        assert_eq!(stats.succeeded, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.total_iterations, 10);
        assert!((stats.total_cost - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn status_transitions() {
        let mut coord = ParallelCoordinator::new(ParallelConfig::default());
        coord.add_task(make_task("a", vec![]));

        assert!(matches!(coord.status(), ParallelPlanStatus::Pending));

        coord.record_result(make_result("a", true));
        assert!(matches!(
            coord.status(),
            ParallelPlanStatus::Completed { .. }
        ));
    }
}
