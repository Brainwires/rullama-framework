//! Judge Agent - LLM-powered cycle evaluator
//!
//! [`JudgeAgent`] wraps a [`TaskAgent`] with a judge-specific system prompt.
//! It evaluates the results of a Plan→Work cycle and produces a [`JudgeVerdict`]
//! that determines what happens next: complete, continue, fresh restart, or abort.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use brainwires_core::{Provider, Task};

use crate::context::AgentContext;
use crate::planner_agent::DynamicTaskSpec;
use crate::system_prompts::judge_agent_prompt;
use crate::task_agent::{TaskAgent, TaskAgentConfig, TaskAgentResult};

// ── Public types ────────────────────────────────────────────────────────────

/// The judge's decision after evaluating a cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum JudgeVerdict {
    /// The goal is fully achieved.
    Complete {
        /// Summary of what was accomplished.
        summary: String,
    },
    /// Partial progress — additional work needed.
    Continue {
        /// Feedback on what's missing or needs improvement.
        #[serde(default)]
        summary: String,
        /// New tasks to add for the next cycle.
        #[serde(default)]
        additional_tasks: Vec<DynamicTaskSpec>,
        /// IDs of tasks that should be retried.
        #[serde(default)]
        retry_tasks: Vec<String>,
        /// Hints for the next planner.
        #[serde(default)]
        hints: Vec<String>,
    },
    /// Significant drift detected — re-plan from scratch.
    FreshRestart {
        /// Why a fresh start is needed.
        reason: String,
        /// Guidance for the next planner cycle.
        #[serde(default)]
        hints: Vec<String>,
        /// Summary of the assessment.
        #[serde(default)]
        summary: String,
    },
    /// Fatal error or impossible goal — stop entirely.
    Abort {
        /// Why the goal cannot be achieved.
        reason: String,
        /// Summary of the assessment.
        #[serde(default)]
        summary: String,
    },
}

impl JudgeVerdict {
    /// Returns the verdict type as a string for logging/messaging.
    pub fn verdict_type(&self) -> &'static str {
        match self {
            JudgeVerdict::Complete { .. } => "complete",
            JudgeVerdict::Continue { .. } => "continue",
            JudgeVerdict::FreshRestart { .. } => "fresh_restart",
            JudgeVerdict::Abort { .. } => "abort",
        }
    }

    /// Returns hints from the verdict, if any.
    pub fn hints(&self) -> &[String] {
        match self {
            JudgeVerdict::Continue { hints, .. } | JudgeVerdict::FreshRestart { hints, .. } => {
                hints
            }
            _ => &[],
        }
    }
}

/// Merge status for a worker's branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStatus {
    /// Successfully merged into target branch.
    Merged,
    /// Merge conflict was resolved automatically.
    ConflictResolved,
    /// Merge conflict could not be resolved.
    ConflictFailed(String),
    /// Merge was not attempted.
    NotAttempted,
}

impl std::fmt::Display for MergeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeStatus::Merged => write!(f, "merged"),
            MergeStatus::ConflictResolved => write!(f, "conflict_resolved"),
            MergeStatus::ConflictFailed(msg) => write!(f, "conflict_failed: {}", msg),
            MergeStatus::NotAttempted => write!(f, "not_attempted"),
        }
    }
}

/// Result from a single worker in the cycle.
#[derive(Debug, Clone)]
pub struct WorkerResult {
    /// The task ID the worker was assigned.
    pub task_id: String,
    /// Description of the task.
    pub task_description: String,
    /// The agent's execution result.
    pub agent_result: TaskAgentResult,
    /// Git branch the worker operated on.
    pub branch_name: String,
    /// Status of merging the worker's branch.
    pub merge_status: MergeStatus,
}

/// Context provided to the judge for evaluation.
#[derive(Debug, Clone)]
pub struct JudgeContext {
    /// The original high-level goal.
    pub original_goal: String,
    /// Which cycle number this is.
    pub cycle_number: u32,
    /// Results from all workers in this cycle.
    pub worker_results: Vec<WorkerResult>,
    /// The planner's rationale for this cycle.
    pub planner_rationale: String,
    /// Verdicts from previous cycles (for context).
    pub previous_verdicts: Vec<JudgeVerdict>,
}

/// Configuration for the judge agent.
#[derive(Debug, Clone)]
pub struct JudgeAgentConfig {
    /// LLM call budget for judging.
    pub max_iterations: u32,
    /// Whether the judge can read files to verify work.
    pub inspect_files: bool,
    /// Whether the judge can inspect git diffs.
    pub inspect_diffs: bool,
    /// Temperature for the judge LLM call.
    pub temperature: f32,
    /// Max tokens per LLM response.
    pub max_tokens: u32,
}

impl Default for JudgeAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 15,
            inspect_files: true,
            inspect_diffs: true,
            temperature: 0.3, // Lower temperature for more consistent judging
            max_tokens: 4096,
        }
    }
}

// ── JudgeAgent ──────────────────────────────────────────────────────────────

/// An LLM-powered judge that evaluates cycle results and decides next steps.
pub struct JudgeAgent {
    agent: Arc<TaskAgent>,
}

impl JudgeAgent {
    /// Create a new judge agent.
    ///
    /// # Parameters
    /// - `id`: Unique agent identifier.
    /// - `judge_context`: Full context for the evaluation.
    /// - `provider`: AI provider for LLM calls.
    /// - `context`: Agent context (working directory, tools, etc.).
    /// - `config`: Judge-specific configuration.
    pub fn new(
        id: String,
        judge_context: &JudgeContext,
        provider: Arc<dyn Provider>,
        context: Arc<AgentContext>,
        config: JudgeAgentConfig,
    ) -> Self {
        let system_prompt = judge_agent_prompt(&id, &context.working_directory);

        let agent_config = TaskAgentConfig {
            max_iterations: config.max_iterations,
            system_prompt: Some(system_prompt),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            validation_config: None,
            ..Default::default()
        };

        // Build the task description with full context for the judge
        let task_description = Self::build_task_description(judge_context);

        let task = Task::new(
            format!("judge-cycle-{}", judge_context.cycle_number),
            task_description,
        );

        let agent = Arc::new(TaskAgent::new(id, task, provider, context, agent_config));

        Self { agent }
    }

    /// Execute the judge and return the parsed verdict.
    pub async fn execute(&self) -> Result<(JudgeVerdict, TaskAgentResult)> {
        let result = self.agent.execute().await?;

        if !result.success {
            return Err(anyhow!("Judge agent failed: {}", result.summary));
        }

        let verdict = Self::parse_verdict(&result.summary)?;
        Ok((verdict, result))
    }

    /// Parse a judge verdict from the agent's summary text.
    pub fn parse_verdict(text: &str) -> Result<JudgeVerdict> {
        let json_str = extract_json_block(text)
            .ok_or_else(|| anyhow!("No JSON block found in judge output"))?;

        serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse judge verdict JSON: {}", e))
    }

    /// Build the task description that gives the judge full context.
    fn build_task_description(ctx: &JudgeContext) -> String {
        let mut desc = format!(
            "# Evaluate Cycle {} Results\n\n## Original Goal\n{}\n\n## Planner Rationale\n{}\n\n",
            ctx.cycle_number, ctx.original_goal, ctx.planner_rationale
        );

        desc.push_str("## Worker Results\n\n");
        for (i, wr) in ctx.worker_results.iter().enumerate() {
            desc.push_str(&format!(
                "### Worker {} (task: {})\n- **Task**: {}\n- **Success**: {}\n- **Summary**: {}\n- **Branch**: {}\n- **Merge**: {}\n- **Iterations**: {}\n\n",
                i + 1,
                wr.task_id,
                wr.task_description,
                wr.agent_result.success,
                wr.agent_result.summary,
                wr.branch_name,
                wr.merge_status,
                wr.agent_result.iterations,
            ));
        }

        if !ctx.previous_verdicts.is_empty() {
            desc.push_str("## Previous Verdicts\n\n");
            for (i, v) in ctx.previous_verdicts.iter().enumerate() {
                desc.push_str(&format!("- Cycle {}: {}\n", i, v.verdict_type()));
            }
            desc.push('\n');
        }

        desc.push_str(
            "## Your Task\n\n\
             Evaluate the above results against the original goal. \
             Output your verdict as a JSON block. \
             If you need to inspect files or diffs for verification, use the available tools first.",
        );

        desc
    }

    /// Get a reference to the underlying agent.
    pub fn agent(&self) -> &Arc<TaskAgent> {
        &self.agent
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Extract a JSON block from text (shared with planner_agent).
fn extract_json_block(text: &str) -> Option<String> {
    // Try ```json ... ``` fences
    if let Some(start) = text.find("```json") {
        let content_start = start + "```json".len();
        if let Some(end) = text[content_start..].find("```") {
            return Some(text[content_start..content_start + end].trim().to_string());
        }
    }

    // Try ``` ... ``` fences
    if let Some(start) = text.find("```") {
        let content_start = start + "```".len();
        let line_end = text[content_start..]
            .find('\n')
            .unwrap_or(text[content_start..].len());
        let actual_start = content_start + line_end + 1;
        if actual_start < text.len()
            && let Some(end) = text[actual_start..].find("```")
        {
            let candidate = text[actual_start..actual_start + end].trim();
            if candidate.starts_with('{') {
                return Some(candidate.to_string());
            }
        }
    }

    // Try raw JSON
    if let Some(start) = text.find('{') {
        let mut depth = 0;
        let mut end = start;
        for (i, ch) in text[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 && end > start {
            return Some(text[start..end].to_string());
        }
    }

    None
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_complete_verdict() {
        let text = r#"```json
{
  "verdict": "complete",
  "summary": "All tasks completed successfully"
}
```"#;
        let verdict = JudgeAgent::parse_verdict(text).unwrap();
        assert!(matches!(verdict, JudgeVerdict::Complete { .. }));
        assert_eq!(verdict.verdict_type(), "complete");
    }

    #[test]
    fn test_parse_continue_verdict() {
        let text = r#"```json
{
  "verdict": "continue",
  "summary": "Two tasks still need work",
  "additional_tasks": [
    {
      "id": "fix-1",
      "description": "Fix the remaining bug",
      "files_involved": ["src/bug.rs"],
      "depends_on": [],
      "priority": "high"
    }
  ],
  "retry_tasks": ["task-3"],
  "hints": ["Focus on error handling"]
}
```"#;
        let verdict = JudgeAgent::parse_verdict(text).unwrap();
        match &verdict {
            JudgeVerdict::Continue {
                additional_tasks,
                retry_tasks,
                hints,
                ..
            } => {
                assert_eq!(additional_tasks.len(), 1);
                assert_eq!(retry_tasks, &["task-3"]);
                assert_eq!(hints, &["Focus on error handling"]);
            }
            _ => panic!("Expected Continue verdict"),
        }
    }

    #[test]
    fn test_parse_fresh_restart_verdict() {
        let text = r#"```json
{
  "verdict": "fresh_restart",
  "reason": "Agents went down the wrong path",
  "hints": ["Try a different approach", "Focus on the API first"],
  "summary": "Need to restart"
}
```"#;
        let verdict = JudgeAgent::parse_verdict(text).unwrap();
        match &verdict {
            JudgeVerdict::FreshRestart { reason, hints, .. } => {
                assert!(reason.contains("wrong path"));
                assert_eq!(hints.len(), 2);
            }
            _ => panic!("Expected FreshRestart verdict"),
        }
    }

    #[test]
    fn test_parse_abort_verdict() {
        let text = r#"```json
{
  "verdict": "abort",
  "reason": "The goal requires external API access we don't have",
  "summary": "Cannot proceed"
}
```"#;
        let verdict = JudgeAgent::parse_verdict(text).unwrap();
        assert!(matches!(verdict, JudgeVerdict::Abort { .. }));
        assert_eq!(verdict.verdict_type(), "abort");
    }

    #[test]
    fn test_verdict_hints() {
        let complete = JudgeVerdict::Complete {
            summary: "done".into(),
        };
        assert!(complete.hints().is_empty());

        let cont = JudgeVerdict::Continue {
            summary: "partial".into(),
            additional_tasks: vec![],
            retry_tasks: vec![],
            hints: vec!["hint1".into()],
        };
        assert_eq!(cont.hints().len(), 1);
    }

    #[test]
    fn test_merge_status_display() {
        assert_eq!(MergeStatus::Merged.to_string(), "merged");
        assert_eq!(MergeStatus::NotAttempted.to_string(), "not_attempted");
        assert!(
            MergeStatus::ConflictFailed("oops".into())
                .to_string()
                .contains("oops")
        );
    }
}
