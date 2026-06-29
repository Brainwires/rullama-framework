//! Core crash detection and recovery orchestration.
//!
//! Integrates with the supervisor to detect crashes, diagnose root causes
//! via AI, apply fixes, rebuild, and relaunch from checkpoints.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use brainwires_core::Provider;

use super::crash_diagnostics::CrashDiagnostics;
use super::recovery_state::{CrashContext, RecoveryPlanState, RecoveryState};
use crate::config::CrashRecoveryConfig;

/// Strategy to apply when recovering from a crash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixStrategy {
    /// Revert the last commit that caused the crash.
    RevertLastCommit,
    /// Apply an AI-generated patch.
    ApplyPatch(String),
    /// Skip the problematic task and continue from the next cycle.
    SkipTask,
    /// Roll back to the last known good checkpoint.
    RollbackToCheckpoint,
}

impl FixStrategy {
    /// Parse a fix strategy from a string label.
    pub fn from_label(label: &str) -> Self {
        match label.trim().to_lowercase().as_str() {
            "revert_last_commit" => Self::RevertLastCommit,
            "rollback_to_checkpoint" => Self::RollbackToCheckpoint,
            "skip_task" => Self::SkipTask,
            other if other.starts_with("apply_patch") => Self::ApplyPatch(other.to_string()),
            _ => Self::SkipTask,
        }
    }

    /// Return a string label for serialization.
    pub fn label(&self) -> &str {
        match self {
            Self::RevertLastCommit => "revert_last_commit",
            Self::ApplyPatch(_) => "apply_patch",
            Self::SkipTask => "skip_task",
            Self::RollbackToCheckpoint => "rollback_to_checkpoint",
        }
    }
}

/// Recovery plan produced by crash diagnostics.
pub struct RecoveryPlan {
    /// Root cause description.
    pub root_cause: String,
    /// Fix strategy to apply.
    pub fix_strategy: FixStrategy,
    /// Files that need fixing.
    pub files_to_fix: Vec<String>,
    /// Whether a git rollback is needed.
    pub rollback_needed: bool,
    /// Cycle index to resume from.
    pub resume_from_cycle: u32,
}

impl From<RecoveryPlanState> for RecoveryPlan {
    fn from(state: RecoveryPlanState) -> Self {
        Self {
            root_cause: state.root_cause,
            fix_strategy: FixStrategy::from_label(&state.fix_strategy),
            files_to_fix: state.files_to_fix,
            rollback_needed: state.rollback_needed,
            resume_from_cycle: state.resume_from_cycle,
        }
    }
}

/// Crash handler that integrates with the supervisor and self-improvement controller.
///
/// Detects meta-crashes (the handler itself failing repeatedly), persists recovery
/// state across process restarts, and orchestrates the diagnose-fix-rebuild cycle.
pub struct CrashHandler {
    provider: Arc<dyn Provider>,
    config: CrashRecoveryConfig,
    state_file: PathBuf,
}

impl CrashHandler {
    /// Create a new crash handler.
    pub fn new(provider: Arc<dyn Provider>, config: CrashRecoveryConfig) -> Self {
        let state_file = PathBuf::from(&config.state_file);
        Self {
            provider,
            config,
            state_file,
        }
    }

    /// Check if there is a pending recovery from a previous crash.
    pub fn has_pending_recovery(&self) -> Result<bool> {
        Ok(self.state_file.exists())
    }

    /// Load a pending recovery state.
    pub fn load_recovery(&self) -> Result<Option<RecoveryState>> {
        RecoveryState::load(&self.state_file)
    }

    /// Handle a crash: capture context, diagnose, and produce a recovery plan.
    pub async fn handle_crash(&self, context: CrashContext) -> Result<RecoveryPlan> {
        // Check for meta-crash (crash handler itself keeps crashing)
        let mut state = if let Some(existing) = self.load_recovery()? {
            if existing.is_meta_crash() {
                anyhow::bail!(
                    "Meta-crash detected: crash handler has failed {} times, aborting recovery",
                    existing.fix_attempts
                );
            }
            // Increment fix attempts
            RecoveryState {
                fix_attempts: existing.fix_attempts + 1,
                crash_context: context.clone(),
                ..existing
            }
        } else {
            RecoveryState {
                version: 1,
                crash_id: uuid::Uuid::new_v4().to_string(),
                crash_context: context.clone(),
                fix_attempts: 0,
                max_fix_attempts: self.config.max_fix_attempts,
                recovery_plan: None,
            }
        };

        // Save state before diagnosis (in case diagnosis itself crashes)
        state.save(&self.state_file)?;

        // Diagnose
        let diagnostics = CrashDiagnostics::new(self.provider.clone());
        let plan_state = diagnostics.diagnose(&context).await?;

        // Save the recovery plan
        state.recovery_plan = Some(plan_state.clone());
        state.save(&self.state_file)?;

        Ok(RecoveryPlan::from(plan_state))
    }

    /// Apply the fix strategy from a recovery plan.
    pub async fn apply_fix(&self, plan: &RecoveryPlan, working_dir: &str) -> Result<()> {
        // Rollback if needed
        if plan.rollback_needed {
            self.rollback(working_dir).await?;
        }

        match &plan.fix_strategy {
            FixStrategy::RevertLastCommit => {
                self.revert_last_commit(working_dir).await?;
            }
            FixStrategy::ApplyPatch(patch) => {
                self.apply_patch(working_dir, patch).await?;
            }
            FixStrategy::SkipTask => {
                tracing::info!("Skipping problematic task, will resume from next cycle");
            }
            FixStrategy::RollbackToCheckpoint => {
                // Rollback already handled above, just stash dirty changes
                self.stash_changes(working_dir).await?;
            }
        }

        Ok(())
    }

    /// Rebuild the project and verify the fix compiles.
    pub async fn rebuild(&self, working_dir: &str) -> Result<bool> {
        let output = tokio::process::Command::new("cargo")
            .args(["build"])
            .current_dir(working_dir)
            .output()
            .await?;

        if output.status.success() {
            tracing::info!("Rebuild successful after crash recovery");
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!(
                "Rebuild failed: {}",
                stderr.chars().take(500).collect::<String>()
            );
            Ok(false)
        }
    }

    /// Clean up recovery state after successful recovery.
    pub fn cleanup(&self) -> Result<()> {
        RecoveryState::cleanup(&self.state_file)?;
        // Also clean up checkpoint file
        let checkpoint = super::recovery_state::checkpoint_path(&self.state_file);
        RecoveryState::cleanup(&checkpoint)?;
        Ok(())
    }

    /// Get the resume cycle index from a pending recovery.
    pub fn resume_cycle(&self) -> Result<Option<u32>> {
        if let Some(state) = self.load_recovery()? {
            if let Some(plan) = &state.recovery_plan {
                return Ok(Some(plan.resume_from_cycle));
            }
            // No plan yet — resume from the cycle after the crash
            return Ok(Some(state.crash_context.last_cycle_index + 1));
        }
        Ok(None)
    }

    async fn rollback(&self, working_dir: &str) -> Result<()> {
        tracing::info!("Rolling back: stashing uncommitted changes");
        self.stash_changes(working_dir).await
    }

    async fn stash_changes(&self, working_dir: &str) -> Result<()> {
        let output = tokio::process::Command::new("git")
            .args(["stash", "--include-untracked"])
            .current_dir(working_dir)
            .output()
            .await?;
        if !output.status.success() {
            tracing::warn!("git stash failed (may have no changes to stash)");
        }
        Ok(())
    }

    async fn revert_last_commit(&self, working_dir: &str) -> Result<()> {
        tracing::info!("Reverting last commit");
        let output = tokio::process::Command::new("git")
            .args(["revert", "--no-edit", "HEAD"])
            .current_dir(working_dir)
            .output()
            .await?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git revert failed: {stderr}");
        }
        Ok(())
    }

    async fn apply_patch(&self, working_dir: &str, patch: &str) -> Result<()> {
        tracing::info!("Applying AI-generated patch");
        let mut child = tokio::process::Command::new("git")
            .args(["apply", "--"])
            .current_dir(working_dir)
            .stdin(std::process::Stdio::piped())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(patch.as_bytes()).await?;
        }

        let status = child.wait().await?;
        if !status.success() {
            anyhow::bail!("Failed to apply patch");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_strategy_from_label() {
        assert_eq!(
            FixStrategy::from_label("revert_last_commit"),
            FixStrategy::RevertLastCommit
        );
        assert_eq!(FixStrategy::from_label("skip_task"), FixStrategy::SkipTask);
        assert_eq!(
            FixStrategy::from_label("rollback_to_checkpoint"),
            FixStrategy::RollbackToCheckpoint
        );
        assert_eq!(
            FixStrategy::from_label("unknown_strategy"),
            FixStrategy::SkipTask
        );
    }

    #[test]
    fn fix_strategy_labels_roundtrip() {
        let strategies = vec![
            FixStrategy::RevertLastCommit,
            FixStrategy::SkipTask,
            FixStrategy::RollbackToCheckpoint,
        ];
        for s in strategies {
            let label = s.label();
            let parsed = FixStrategy::from_label(label);
            assert_eq!(parsed, s);
        }
    }
}
