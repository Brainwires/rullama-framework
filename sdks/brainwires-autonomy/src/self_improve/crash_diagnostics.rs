//! AI-powered crash root-cause analysis.

use std::sync::Arc;

use anyhow::Result;
use brainwires_core::Provider;

use super::recovery_state::{CrashContext, RecoveryPlanState};

/// Diagnoses crashes using an AI provider to analyze stderr output, git diffs,
/// and crash context, producing a structured [`RecoveryPlanState`].
pub struct CrashDiagnostics {
    provider: Arc<dyn Provider>,
}

impl CrashDiagnostics {
    /// Create a new crash diagnostics instance.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self { provider }
    }

    /// Analyze a crash context and produce a recovery plan.
    pub async fn diagnose(&self, context: &CrashContext) -> Result<RecoveryPlanState> {
        let git_diff = self.get_recent_diff(&context.working_directory).await;

        let prompt = format!(
            "You are debugging a crash in a self-improvement automation system.\n\
             \n\
             ## Crash Details\n\
             - Exit code: {:?}\n\
             - Signal: {:?}\n\
             - Last cycle index: {}\n\
             - Strategy: {}\n\
             - Working directory: {}\n\
             - Branch: {}\n\
             - Last commit: {}\n\
             - Dirty files: {}\n\
             \n\
             ## Stderr Output (last lines)\n\
             ```\n{}\n```\n\
             \n\
             ## Recent Git Diff\n\
             ```\n{}\n```\n\
             \n\
             ## Instructions\n\
             Analyze this crash and respond with EXACTLY this format:\n\
             ROOT_CAUSE: <one-line description of the root cause>\n\
             FIX_STRATEGY: <one of: revert_last_commit, apply_patch, skip_task, rollback_to_checkpoint>\n\
             ROLLBACK_NEEDED: <true or false>\n\
             FILES_TO_FIX: <comma-separated file paths, or 'none'>\n\
             RESUME_FROM: <cycle index to resume from>\n",
            context.exit_code,
            context.signal,
            context.last_cycle_index,
            context.last_strategy.as_deref().unwrap_or("unknown"),
            context.working_directory,
            context.git_state.branch,
            context.git_state.last_commit,
            context.git_state.dirty_files.join(", "),
            context.stderr_tail,
            git_diff.unwrap_or_default(),
        );

        let messages = vec![brainwires_core::Message::user(prompt)];
        let options = brainwires_core::ChatOptions::default();
        let response = self.provider.chat(&messages, None, &options).await?;
        let text = response.message.text().unwrap_or_default().to_string();

        Self::parse_diagnosis(&text, context.last_cycle_index)
    }

    fn parse_diagnosis(text: &str, fallback_cycle: u32) -> Result<RecoveryPlanState> {
        let root_cause =
            extract_field(text, "ROOT_CAUSE").unwrap_or_else(|| "Unknown crash cause".to_string());
        let fix_strategy =
            extract_field(text, "FIX_STRATEGY").unwrap_or_else(|| "skip_task".to_string());
        let rollback_needed = extract_field(text, "ROLLBACK_NEEDED")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
        let files_to_fix = extract_field(text, "FILES_TO_FIX")
            .map(|v| {
                if v.to_lowercase() == "none" {
                    Vec::new()
                } else {
                    v.split(',').map(|s| s.trim().to_string()).collect()
                }
            })
            .unwrap_or_default();
        let resume_from = extract_field(text, "RESUME_FROM")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(fallback_cycle + 1);

        Ok(RecoveryPlanState {
            root_cause,
            fix_strategy,
            files_to_fix,
            rollback_needed,
            resume_from_cycle: resume_from,
        })
    }

    async fn get_recent_diff(&self, working_dir: &str) -> Option<String> {
        let output = tokio::process::Command::new("git")
            .args(["diff", "HEAD~1..HEAD", "--stat"])
            .current_dir(working_dir)
            .output()
            .await
            .ok()?;
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

fn extract_field(text: &str, field: &str) -> Option<String> {
    let prefix = format!("{field}:");
    text.lines()
        .find(|l| l.trim().starts_with(&prefix))
        .map(|l| {
            l.trim()
                .strip_prefix(&prefix)
                .unwrap_or("")
                .trim()
                .to_string()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_diagnosis_extracts_fields() {
        let text = "\
ROOT_CAUSE: Stack overflow in recursive function
FIX_STRATEGY: revert_last_commit
ROLLBACK_NEEDED: true
FILES_TO_FIX: src/lib.rs, src/main.rs
RESUME_FROM: 5";

        let plan = CrashDiagnostics::parse_diagnosis(text, 4).unwrap();
        assert_eq!(plan.root_cause, "Stack overflow in recursive function");
        assert_eq!(plan.fix_strategy, "revert_last_commit");
        assert!(plan.rollback_needed);
        assert_eq!(plan.files_to_fix, vec!["src/lib.rs", "src/main.rs"]);
        assert_eq!(plan.resume_from_cycle, 5);
    }

    #[test]
    fn parse_diagnosis_uses_defaults_for_missing_fields() {
        let text = "Some random AI response without proper formatting";
        let plan = CrashDiagnostics::parse_diagnosis(text, 3).unwrap();
        assert_eq!(plan.root_cause, "Unknown crash cause");
        assert_eq!(plan.fix_strategy, "skip_task");
        assert!(!plan.rollback_needed);
        assert!(plan.files_to_fix.is_empty());
        assert_eq!(plan.resume_from_cycle, 4); // fallback + 1
    }
}
