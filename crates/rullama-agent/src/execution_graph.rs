//! Execution DAG and telemetry for TaskAgent runs
//!
//! Provides [`ExecutionGraph`] (one node per provider-call iteration with tool
//! call records) and [`RunTelemetry`] (aggregate summary derived from the
//! graph at run completion).

use chrono::{DateTime, Utc};

/// One tool call within a single iteration step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRecord {
    /// Unique identifier for this tool use invocation.
    pub tool_use_id: String,
    /// Name of the tool that was called.
    pub tool_name: String,
    /// Whether the tool call resulted in an error.
    pub is_error: bool,
    /// When the tool call was executed.
    pub executed_at: DateTime<Utc>,
}

/// One provider-call iteration in the `execute()` loop.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepNode {
    /// Iteration number within the execution loop.
    pub iteration: u32,
    /// When this step started.
    pub started_at: DateTime<Utc>,
    /// When this step ended.
    pub ended_at: DateTime<Utc>,
    /// Prompt tokens for this call (from `Usage::prompt_tokens`).
    pub prompt_tokens: u32,
    /// Completion tokens for this call (from `Usage::completion_tokens`).
    pub completion_tokens: u32,
    /// Tool calls made during this step.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Reason the provider stopped generating.
    pub finish_reason: Option<String>,
}

/// Full execution trace for one `TaskAgent` run.
///
/// Contains one [`StepNode`] per provider call and a flat ordered
/// [`tool_sequence`][ExecutionGraph::tool_sequence] for easy comparison
/// against expected sequences in behavioral tests (Phase 2 recorder).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionGraph {
    /// SHA-256 of (system prompt bytes + sorted tool name bytes), hex-encoded.
    /// Changes whenever the prompt or tool registry changes.
    pub prompt_hash: String,
    /// When the run started.
    pub run_started_at: DateTime<Utc>,
    /// One [`StepNode`] per provider call iteration.
    pub steps: Vec<StepNode>,
    /// Flat ordered list of tool names across all steps (Phase 2 recorder).
    pub tool_sequence: Vec<String>,
}

impl ExecutionGraph {
    /// Create a new, empty graph with the given prompt hash and start time.
    pub fn new(prompt_hash: String, run_started_at: DateTime<Utc>) -> Self {
        Self {
            prompt_hash,
            run_started_at,
            steps: Vec::new(),
            tool_sequence: Vec::new(),
        }
    }

    /// Start a new step; returns its index for later finalization.
    pub fn push_step(&mut self, iteration: u32, started_at: DateTime<Utc>) -> usize {
        let idx = self.steps.len();
        self.steps.push(StepNode {
            iteration,
            started_at,
            ended_at: started_at,
            prompt_tokens: 0,
            completion_tokens: 0,
            tool_calls: Vec::new(),
            finish_reason: None,
        });
        idx
    }

    /// Fill in token counts and finish_reason after the provider call returns.
    pub fn finalize_step(
        &mut self,
        step_idx: usize,
        ended_at: DateTime<Utc>,
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: Option<String>,
    ) {
        if let Some(s) = self.steps.get_mut(step_idx) {
            s.ended_at = ended_at;
            s.prompt_tokens = prompt_tokens;
            s.completion_tokens = completion_tokens;
            s.finish_reason = finish_reason;
        }
    }

    /// Record a tool call and append its name to the flat sequence.
    pub fn record_tool_call(&mut self, step_idx: usize, record: ToolCallRecord) {
        self.tool_sequence.push(record.tool_name.clone());
        if let Some(s) = self.steps.get_mut(step_idx) {
            s.tool_calls.push(record);
        }
    }
}

/// Structured telemetry summary for a completed run.
///
/// Derived from an [`ExecutionGraph`] via [`RunTelemetry::from_graph`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunTelemetry {
    /// Hash of the system prompt and tool registry.
    pub prompt_hash: String,
    /// When the run started.
    pub run_started_at: DateTime<Utc>,
    /// When the run ended.
    pub run_ended_at: DateTime<Utc>,
    /// Total run duration in milliseconds.
    pub duration_ms: u64,
    /// Number of provider call iterations.
    pub total_iterations: u32,
    /// Total number of tool calls across all iterations.
    pub total_tool_calls: u32,
    /// Number of tool calls that returned errors.
    pub tool_error_count: u32,
    /// Unique tool names, deduped in first-use order.
    pub tools_used: Vec<String>,
    /// Total prompt tokens consumed.
    pub total_prompt_tokens: u32,
    /// Total completion tokens consumed.
    pub total_completion_tokens: u32,
    /// Total estimated cost in USD.
    pub total_cost_usd: f64,
    /// Whether the run completed successfully.
    pub success: bool,
}

impl RunTelemetry {
    /// Build a telemetry record from a completed [`ExecutionGraph`].
    pub fn from_graph(
        graph: &ExecutionGraph,
        run_ended_at: DateTime<Utc>,
        success: bool,
        total_cost_usd: f64,
    ) -> Self {
        let duration_ms = (run_ended_at - graph.run_started_at)
            .num_milliseconds()
            .max(0) as u64;
        let total_tool_calls: u32 = graph.steps.iter().map(|s| s.tool_calls.len() as u32).sum();
        let tool_error_count: u32 = graph
            .steps
            .iter()
            .flat_map(|s| s.tool_calls.iter())
            .filter(|tc| tc.is_error)
            .count() as u32;
        let total_prompt_tokens: u32 = graph.steps.iter().map(|s| s.prompt_tokens).sum();
        let total_completion_tokens: u32 = graph.steps.iter().map(|s| s.completion_tokens).sum();
        let mut seen = std::collections::HashSet::new();
        let tools_used: Vec<String> = graph
            .tool_sequence
            .iter()
            .filter(|n| seen.insert((*n).clone()))
            .cloned()
            .collect();
        Self {
            prompt_hash: graph.prompt_hash.clone(),
            run_started_at: graph.run_started_at,
            run_ended_at,
            duration_ms,
            total_iterations: graph.steps.len() as u32,
            total_tool_calls,
            tool_error_count,
            tools_used,
            total_prompt_tokens,
            total_completion_tokens,
            total_cost_usd,
            success,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_graph() -> ExecutionGraph {
        ExecutionGraph::new("abc123".to_string(), Utc::now())
    }

    #[test]
    fn test_push_step_returns_index() {
        let mut g = make_graph();
        let idx0 = g.push_step(1, Utc::now());
        let idx1 = g.push_step(2, Utc::now());
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(g.steps.len(), 2);
    }

    #[test]
    fn test_finalize_step_sets_tokens() {
        let mut g = make_graph();
        let idx = g.push_step(1, Utc::now());
        let end = Utc::now();
        g.finalize_step(idx, end, 100, 50, Some("stop".to_string()));
        assert_eq!(g.steps[idx].prompt_tokens, 100);
        assert_eq!(g.steps[idx].completion_tokens, 50);
        assert_eq!(g.steps[idx].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_record_tool_call_appends_sequence() {
        let mut g = make_graph();
        let idx = g.push_step(1, Utc::now());
        g.record_tool_call(
            idx,
            ToolCallRecord {
                tool_use_id: "u1".to_string(),
                tool_name: "read_file".to_string(),
                is_error: false,
                executed_at: Utc::now(),
            },
        );
        g.record_tool_call(
            idx,
            ToolCallRecord {
                tool_use_id: "u2".to_string(),
                tool_name: "write_file".to_string(),
                is_error: false,
                executed_at: Utc::now(),
            },
        );
        assert_eq!(g.tool_sequence, vec!["read_file", "write_file"]);
        assert_eq!(g.steps[idx].tool_calls.len(), 2);
    }

    #[test]
    fn test_telemetry_from_graph() {
        let start = Utc::now();
        let mut g = ExecutionGraph::new("hash".to_string(), start);
        let idx = g.push_step(1, start);
        g.finalize_step(idx, Utc::now(), 100, 50, None);
        g.record_tool_call(
            idx,
            ToolCallRecord {
                tool_use_id: "u1".to_string(),
                tool_name: "bash".to_string(),
                is_error: false,
                executed_at: Utc::now(),
            },
        );
        g.record_tool_call(
            idx,
            ToolCallRecord {
                tool_use_id: "u2".to_string(),
                tool_name: "bash".to_string(),
                is_error: true,
                executed_at: Utc::now(),
            },
        );

        let telem = RunTelemetry::from_graph(&g, Utc::now(), true, 0.01);
        assert_eq!(telem.total_iterations, 1);
        assert_eq!(telem.total_tool_calls, 2);
        assert_eq!(telem.tool_error_count, 1);
        // "bash" appears twice but tools_used should deduplicate
        assert_eq!(telem.tools_used, vec!["bash"]);
        assert_eq!(telem.total_prompt_tokens, 100);
        assert_eq!(telem.total_completion_tokens, 50);
        assert!(telem.success);
    }

    #[test]
    fn test_tool_sequence_preserves_order() {
        let mut g = make_graph();
        let idx = g.push_step(1, Utc::now());
        for name in &["a", "b", "c", "b", "a"] {
            g.record_tool_call(
                idx,
                ToolCallRecord {
                    tool_use_id: "x".to_string(),
                    tool_name: name.to_string(),
                    is_error: false,
                    executed_at: Utc::now(),
                },
            );
        }
        assert_eq!(g.tool_sequence, vec!["a", "b", "c", "b", "a"]);
    }
}
