//! Self-improvement controller — orchestrates improvement cycles.
//!
//! Refactored from the CLI's controller to accept an `Arc<dyn Provider>`
//! instead of creating providers internally. Bridge path execution is
//! left to CLI-specific code.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

use brainwires_core::Provider;

use super::comparator::PathResult;
use super::strategies::ImprovementTask;
use super::task_generator::TaskGenerator;
use crate::config::SelfImprovementConfig;
use crate::metrics::{SessionMetrics, SessionReport};
use crate::safety::{SafetyGuard, SafetyStop};

/// Result of a single improvement cycle.
pub struct CycleResult {
    /// The task that was executed.
    pub task: ImprovementTask,
    /// Path result from execution, if available.
    pub result: Option<PathResult>,
    /// Whether the changes were committed.
    pub committed: bool,
    /// Commit hash, if committed.
    pub commit_hash: Option<String>,
}

/// Orchestrates self-improvement sessions: generates tasks, executes them
/// via an AI provider, validates results, and commits changes.
pub struct SelfImprovementController {
    config: SelfImprovementConfig,
    task_generator: TaskGenerator,
    metrics: SessionMetrics,
    safety: SafetyGuard,
    provider: Arc<dyn Provider>,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl SelfImprovementController {
    /// Create a controller with the default strategy registry.
    pub fn new(config: SelfImprovementConfig, provider: Arc<dyn Provider>) -> Self {
        let task_generator = TaskGenerator::from_config(&config);
        let safety = SafetyGuard::new(&config);
        Self {
            config,
            task_generator,
            metrics: SessionMetrics::new(),
            safety,
            provider,
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Create a controller with custom strategies (e.g. for eval-driven mode).
    pub fn new_with_strategies(
        config: SelfImprovementConfig,
        provider: Arc<dyn Provider>,
        strategies: Vec<Box<dyn super::strategies::ImprovementStrategy>>,
    ) -> Self {
        let task_generator = TaskGenerator::new(strategies);
        let safety = SafetyGuard::new(&config);
        Self {
            config,
            task_generator,
            metrics: SessionMetrics::new(),
            safety,
            provider,
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Attach an analytics collector to record AutonomySession events.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<brainwires_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }

    /// Run the complete self-improvement session.
    pub async fn run(&mut self) -> Result<SessionReport> {
        let start = Instant::now();
        tracing::info!("Starting self-improvement loop");
        tracing::info!("Strategies: {:?}", self.task_generator.strategy_names());

        let tasks = self.task_generator.generate_all(&self.config).await?;
        tracing::info!("Generated {} improvement tasks", tasks.len());

        for task in &tasks {
            self.metrics.record_generated(&task.strategy, 1);
        }

        if self.config.dry_run {
            self.print_dry_run(&tasks);
            return Ok(SessionReport::new(
                self.metrics.clone(),
                start.elapsed(),
                None,
            ));
        }

        let mut stop_reason: Option<SafetyStop> = None;

        for task in tasks {
            if let Err(reason) = self.safety.check_can_continue() {
                tracing::warn!("Safety stop: {reason}");
                stop_reason = Some(reason);
                break;
            }

            tracing::info!(
                "Cycle {}/{}: {} (strategy: {})",
                self.safety.cycles_completed() + 1,
                self.config.max_cycles,
                task.description.chars().take(80).collect::<String>(),
                task.strategy,
            );

            self.safety.heartbeat();

            match self.run_cycle(&task).await {
                Ok(result) => {
                    self.metrics.record_attempt(&task.strategy);

                    let success = result.result.as_ref().is_some_and(|r| r.success);

                    if success {
                        let iterations = result.result.as_ref().map(|r| r.iterations).unwrap_or(0);
                        let diff_lines = result.result.as_ref().map(|r| r.diff_lines).unwrap_or(0);

                        self.safety.record_success(diff_lines);
                        self.metrics.record_success(&task.strategy, iterations);

                        if let Some(hash) = result.commit_hash {
                            self.metrics.record_commit(hash);
                        }
                    } else {
                        self.safety.record_failure();
                        self.metrics.record_failure(&task.strategy);
                    }
                }
                Err(e) => {
                    tracing::error!("Cycle failed: {e}");
                    self.safety.record_failure();
                    self.metrics.record_failure(&task.strategy);
                }
            }
        }

        let report = SessionReport::new(self.metrics.clone(), start.elapsed(), stop_reason);

        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use brainwires_telemetry::AnalyticsEvent;
            let m = &report.metrics;
            collector.record(AnalyticsEvent::AutonomySession {
                session_id: None,
                tasks_attempted: m.tasks_attempted,
                tasks_succeeded: m.tasks_succeeded,
                tasks_failed: m.tasks_failed,
                total_cost_usd: m.total_cost,
                duration_ms: report.duration.as_millis() as u64,
                timestamp: chrono::Utc::now(),
            });
        }

        Ok(report)
    }

    async fn run_cycle(&self, task: &ImprovementTask) -> Result<CycleResult> {
        let repo_path = std::env::current_dir()?.to_string_lossy().to_string();

        let start = Instant::now();

        // Execute the improvement task
        let result = self.execute_task(task, &repo_path).await;
        let elapsed = start.elapsed();

        let path_result = match result {
            Ok(pr) => Some(pr),
            Err(e) => Some(PathResult::failure(e.to_string(), elapsed)),
        };

        // Commit if successful and within diff limits
        let mut committed = false;
        let mut commit_hash = None;

        if let Some(ref pr) = path_result
            && pr.success
            && pr.diff_lines <= self.config.max_diff_per_task
        {
            match self.commit_changes(&repo_path, task).await {
                Ok(hash) => {
                    committed = true;
                    commit_hash = Some(hash);
                }
                Err(e) => {
                    tracing::warn!("Failed to commit: {e}");
                }
            }
        }

        Ok(CycleResult {
            task: task.clone(),
            result: path_result,
            committed,
            commit_hash,
        })
    }

    async fn execute_task(&self, task: &ImprovementTask, _working_dir: &str) -> Result<PathResult> {
        let start = Instant::now();

        // Build the prompt for the AI provider
        let prompt = format!(
            "{}\n\nContext:\n{}\n\nTarget files: {}",
            task.description,
            task.context,
            task.target_files.join(", ")
        );

        // Use the injected provider to execute the task
        let messages = vec![brainwires_core::Message::user(prompt)];
        let options = brainwires_core::ChatOptions::default();

        match self.provider.chat(&messages, None, &options).await {
            Ok(_response) => {
                let diff = get_git_diff_stat().await.unwrap_or_default();
                let diff_lines = diff.lines().count() as u32;

                Ok(PathResult {
                    success: true,
                    iterations: 1,
                    diff,
                    diff_lines,
                    duration: start.elapsed(),
                    error: None,
                })
            }
            Err(e) => Ok(PathResult::failure(e.to_string(), start.elapsed())),
        }
    }

    async fn commit_changes(&self, worktree_path: &str, task: &ImprovementTask) -> Result<String> {
        let add = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(worktree_path)
            .output()
            .await?;

        if !add.status.success() {
            anyhow::bail!("git add failed");
        }

        let status = tokio::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(worktree_path)
            .output()
            .await?;

        let status_output = String::from_utf8_lossy(&status.stdout);
        if status_output.trim().is_empty() {
            anyhow::bail!("No changes to commit");
        }

        let message = format!(
            "self-improve({}): {}\n\nStrategy: {}\nCategory: {}\nTarget files: {}",
            task.strategy,
            task.description.chars().take(72).collect::<String>(),
            task.strategy,
            task.category,
            task.target_files.join(", ")
        );

        let commit = tokio::process::Command::new("git")
            .args(["commit", "-m", &message])
            .current_dir(worktree_path)
            .output()
            .await?;

        if !commit.status.success() {
            let stderr = String::from_utf8_lossy(&commit.stderr);
            anyhow::bail!("git commit failed: {stderr}");
        }

        let hash = tokio::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(worktree_path)
            .output()
            .await?;

        let hash = String::from_utf8_lossy(&hash.stdout).trim().to_string();
        Ok(hash)
    }

    fn print_dry_run(&self, tasks: &[ImprovementTask]) {
        println!("\n=== Self-Improvement Dry Run ===\n");
        println!("Found {} tasks:\n", tasks.len());
        for (i, task) in tasks.iter().enumerate() {
            println!(
                "  {}. [{}] [P{}] {}",
                i + 1,
                task.strategy,
                task.priority,
                task.description.chars().take(100).collect::<String>()
            );
            if !task.target_files.is_empty() {
                println!("     Files: {}", task.target_files.join(", "));
            }
            println!("     Est. diff: ~{} lines", task.estimated_diff_lines);
            println!();
        }
        println!("Config:");
        println!("  Max cycles: {}", self.config.max_cycles);
        println!("  Max budget: ${:.2}", self.config.max_budget);
        println!("  Agent iterations: {}", self.config.agent_iterations);
        println!(
            "  Max diff per task: {} lines",
            self.config.max_diff_per_task
        );
    }

    /// Access to the provider (for testing or extension).
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// Access to the session metrics.
    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }
}

async fn get_git_diff_stat() -> Result<String> {
    let output = tokio::process::Command::new("git")
        .args(["diff", "--stat"])
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
