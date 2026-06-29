//! Agent-based code change maker.
//!
//! Uses an AI provider to interpret investigation results and apply code fixes,
//! then stages and commits the changes.

use std::sync::Arc;

use brainwires_core::Provider;
use serde::{Deserialize, Serialize};

use super::investigator::InvestigationResult;

/// Result of making changes to fix an issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeResult {
    /// Whether the changes were applied successfully.
    pub success: bool,
    /// List of modified file paths.
    pub files_modified: Vec<String>,
    /// Summary of the changes made.
    pub summary: String,
    /// Number of diff lines produced.
    pub diff_lines: u32,
    /// Number of iterations used.
    pub iterations: u32,
}

/// Makes code changes based on investigation results using an AI provider.
///
/// Sends the investigation summary to the AI model, then inspects the git diff
/// to determine what was changed.
pub struct ChangeMaker {
    provider: Arc<dyn Provider>,
    _max_iterations: u32,
}

impl ChangeMaker {
    /// Create a new change maker with the given provider and iteration limit.
    pub fn new(provider: Arc<dyn Provider>, max_iterations: u32) -> Self {
        Self {
            provider,
            _max_iterations: max_iterations,
        }
    }

    /// Apply fixes based on the investigation result.
    pub async fn make_changes(
        &self,
        investigation: &InvestigationResult,
        repo_path: &str,
    ) -> anyhow::Result<ChangeResult> {
        let prompt = format!(
            "Fix the following issue in the codebase at {repo_path}:\n\n\
             Summary: {}\n\
             Approach: {}\n\
             Affected files: {}\n\n\
             Make the necessary code changes to fix this issue.",
            investigation.summary,
            investigation.approach,
            investigation.affected_files.join(", ")
        );

        let messages = vec![brainwires_core::Message::user(prompt)];
        let options = brainwires_core::ChatOptions::default();

        match self.provider.chat(&messages, None, &options).await {
            Ok(_response) => {
                // Get the diff to see what changed
                let diff_output = tokio::process::Command::new("git")
                    .args(["diff", "--stat"])
                    .current_dir(repo_path)
                    .output()
                    .await?;

                let diff = String::from_utf8_lossy(&diff_output.stdout);
                let diff_lines = diff.lines().count() as u32;

                let files_output = tokio::process::Command::new("git")
                    .args(["diff", "--name-only"])
                    .current_dir(repo_path)
                    .output()
                    .await?;

                let files: Vec<String> = String::from_utf8_lossy(&files_output.stdout)
                    .lines()
                    .map(|s| s.to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                Ok(ChangeResult {
                    success: !files.is_empty(),
                    files_modified: files,
                    summary: "Changes applied based on investigation".to_string(),
                    diff_lines,
                    iterations: 1,
                })
            }
            Err(e) => Ok(ChangeResult {
                success: false,
                files_modified: Vec::new(),
                summary: format!("Failed to make changes: {e}"),
                diff_lines: 0,
                iterations: 1,
            }),
        }
    }

    /// Commit the changes with a structured message.
    pub async fn commit_changes(
        &self,
        repo_path: &str,
        issue_number: u64,
        summary: &str,
    ) -> anyhow::Result<String> {
        let _ = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(repo_path)
            .output()
            .await?;

        let message = format!("fix(#{issue_number}): {summary}");

        let commit = tokio::process::Command::new("git")
            .args(["commit", "-m", &message])
            .current_dir(repo_path)
            .output()
            .await?;

        if !commit.status.success() {
            let stderr = String::from_utf8_lossy(&commit.stderr);
            anyhow::bail!("git commit failed: {stderr}");
        }

        let hash = tokio::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repo_path)
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&hash.stdout).trim().to_string())
    }
}
