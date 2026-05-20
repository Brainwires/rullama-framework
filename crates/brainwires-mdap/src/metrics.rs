//! MDAP Execution Metrics and Analytics
//!
//! Comprehensive metrics collection for MDAP execution, enabling:
//! - Performance analysis and optimization
//! - SEAL learning integration
//! - Cost tracking and budgeting
//! - Red-flag pattern analysis

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Comprehensive MDAP execution metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MdapMetrics {
    /// Unique identifier for this execution
    pub execution_id: String,
    /// Start time of execution
    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub start_time: Option<DateTime<Utc>>,
    /// End time of execution
    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub end_time: Option<DateTime<Utc>>,
    /// Configuration used for this execution
    pub config_summary: Option<ConfigSummary>,

    // Step-level metrics
    /// Metrics for each subtask
    pub subtask_metrics: Vec<SubtaskMetric>,
    /// Total number of steps in the task
    pub total_steps: u64,
    /// Number of completed steps
    pub completed_steps: u64,
    /// Number of failed steps
    pub failed_steps: u64,

    // Sampling metrics
    /// Total number of samples taken
    pub total_samples: u64,
    /// Number of valid (non-red-flagged) samples
    pub valid_samples: u64,
    /// Number of red-flagged samples
    pub red_flagged_samples: u64,
    /// Breakdown of red-flags by reason
    pub red_flag_breakdown: HashMap<String, u64>,

    // Voting metrics
    /// Metrics for each voting round
    pub voting_rounds: Vec<VotingRoundMetric>,
    /// Average number of votes needed per step
    pub average_votes_per_step: f64,
    /// Maximum votes needed for any single step
    pub max_votes_for_single_step: u32,
    /// Minimum votes needed for any single step
    pub min_votes_for_single_step: u32,

    // Cost metrics
    /// Actual cost in USD
    pub actual_cost_usd: f64,
    /// Estimated cost in USD (from scaling laws)
    pub estimated_cost_usd: f64,
    /// Average cost per step
    pub cost_per_step: f64,
    /// Total input tokens used
    pub total_input_tokens: u64,
    /// Total output tokens used
    pub total_output_tokens: u64,

    // Time metrics
    /// Total execution time in seconds
    pub total_time_seconds: f64,
    /// Average time per step in milliseconds
    pub average_time_per_step_ms: f64,
    /// Time spent on voting
    pub voting_time_seconds: f64,
    /// Time spent on decomposition
    pub decomposition_time_seconds: f64,

    // Success metrics
    /// Whether the execution succeeded
    pub final_success: bool,
    /// Estimated success probability (from scaling laws)
    pub estimated_success_probability: f64,
    /// Actual success rate observed
    pub actual_success_rate: f64,

    // Model/Provider info
    /// Model used for execution
    pub model: Option<String>,
    /// Provider used
    pub provider: Option<String>,
}

/// Summary of MDAP configuration for metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigSummary {
    /// First-to-ahead-by-k value.
    pub k: u32,
    /// Target success rate.
    pub target_success_rate: f64,
    /// Number of parallel samples.
    pub parallel_samples: u32,
    /// Maximum samples per subtask.
    pub max_samples_per_subtask: u32,
    /// Decomposition strategy name.
    pub decomposition_strategy: String,
}

/// Metrics for a single subtask execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubtaskMetric {
    /// Subtask identifier
    pub subtask_id: String,
    /// Description of the subtask
    pub description: String,
    /// Number of samples needed to reach consensus
    pub samples_needed: u32,
    /// Number of red-flags encountered
    pub red_flags_hit: u32,
    /// Reasons for red-flags
    pub red_flag_reasons: Vec<String>,
    /// Confidence of the final answer
    pub final_confidence: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Votes for the winning answer
    pub winner_votes: u32,
    /// Total valid votes
    pub total_votes: u32,
    /// Whether this subtask succeeded
    pub succeeded: bool,
    /// Input tokens for this subtask
    pub input_tokens: u64,
    /// Output tokens for this subtask
    pub output_tokens: u64,
    /// Complexity estimate (0.0-1.0)
    pub complexity_estimate: f32,
}

/// Metrics for a single voting round
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VotingRoundMetric {
    /// Which step this voting was for
    pub step: u64,
    /// Which round of voting (may need multiple rounds)
    pub round: u32,
    /// Vote distribution among candidates
    pub candidates: HashMap<String, u32>,
    /// The winner (if any)
    pub winner: Option<String>,
    /// Number of red-flagged samples this round
    pub red_flagged_this_round: u32,
    /// Time for this round in milliseconds
    pub round_time_ms: u64,
}

impl MdapMetrics {
    /// Create a new metrics instance
    pub fn new(execution_id: impl Into<String>) -> Self {
        Self {
            execution_id: execution_id.into(),
            start_time: Some(Utc::now()),
            min_votes_for_single_step: u32::MAX,
            ..Default::default()
        }
    }

    /// Create with configuration summary
    pub fn with_config(execution_id: impl Into<String>, config: ConfigSummary) -> Self {
        let mut metrics = Self::new(execution_id);
        metrics.config_summary = Some(config);
        metrics
    }

    /// Record the start of execution
    pub fn start(&mut self) {
        self.start_time = Some(Utc::now());
    }

    /// Record a subtask completion
    pub fn record_subtask(&mut self, metric: SubtaskMetric) {
        self.total_samples += metric.samples_needed as u64;
        self.red_flagged_samples += metric.red_flags_hit as u64;
        self.valid_samples += metric.total_votes as u64;
        self.total_input_tokens += metric.input_tokens;
        self.total_output_tokens += metric.output_tokens;

        if metric.succeeded {
            self.completed_steps += 1;
        } else {
            self.failed_steps += 1;
        }

        // Track vote statistics
        if metric.total_votes > 0 {
            self.max_votes_for_single_step = self.max_votes_for_single_step.max(metric.total_votes);
            self.min_votes_for_single_step = self.min_votes_for_single_step.min(metric.total_votes);
        }

        // Track red-flag breakdown
        for reason in &metric.red_flag_reasons {
            *self.red_flag_breakdown.entry(reason.clone()).or_insert(0) += 1;
        }

        self.subtask_metrics.push(metric);
    }

    /// Record a voting round
    pub fn record_voting_round(&mut self, round: VotingRoundMetric) {
        self.voting_rounds.push(round);
    }

    /// Add cost for a sample
    pub fn add_sample_cost(&mut self, cost_usd: f64) {
        self.actual_cost_usd += cost_usd;
    }

    /// Finalize metrics after execution
    pub fn finalize(&mut self, success: bool) {
        self.end_time = Some(Utc::now());
        self.final_success = success;

        // Calculate total time
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            self.total_time_seconds = (end - start).num_milliseconds() as f64 / 1000.0;
        }

        // Calculate averages
        if self.completed_steps > 0 {
            self.average_votes_per_step = self
                .subtask_metrics
                .iter()
                .map(|m| m.total_votes as f64)
                .sum::<f64>()
                / self.completed_steps as f64;

            self.average_time_per_step_ms = self
                .subtask_metrics
                .iter()
                .map(|m| m.execution_time_ms as f64)
                .sum::<f64>()
                / self.completed_steps as f64;

            self.cost_per_step = self.actual_cost_usd / self.completed_steps as f64;
        }

        // Calculate actual success rate
        if self.total_steps > 0 {
            self.actual_success_rate = self.completed_steps as f64 / self.total_steps as f64;
        }

        // Fix min votes if no subtasks recorded
        if self.min_votes_for_single_step == u32::MAX {
            self.min_votes_for_single_step = 0;
        }
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let red_flag_rate = if self.total_samples > 0 {
            (self.red_flagged_samples as f64 / self.total_samples as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "MDAP Execution Summary:\n\
             - Steps: {}/{} completed ({} failed)\n\
             - Samples: {} total, {} valid, {} red-flagged ({:.1}%)\n\
             - Avg votes/step: {:.1} (min: {}, max: {})\n\
             - Cost: ${:.4} (${:.6}/step)\n\
             - Tokens: {} in, {} out\n\
             - Time: {:.1}s ({:.0}ms/step)\n\
             - Success: {}",
            self.completed_steps,
            self.total_steps,
            self.failed_steps,
            self.total_samples,
            self.valid_samples,
            self.red_flagged_samples,
            red_flag_rate,
            self.average_votes_per_step,
            self.min_votes_for_single_step,
            self.max_votes_for_single_step,
            self.actual_cost_usd,
            self.cost_per_step,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_time_seconds,
            self.average_time_per_step_ms,
            if self.final_success { "YES" } else { "NO" }
        )
    }

    /// Generate detailed red-flag analysis
    pub fn red_flag_analysis(&self) -> String {
        if self.red_flag_breakdown.is_empty() {
            return "No red-flags encountered.".to_string();
        }

        let mut analysis = String::from("Red-Flag Analysis:\n");
        let mut sorted: Vec<_> = self.red_flag_breakdown.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (reason, count) in sorted {
            let percentage = (*count as f64 / self.red_flagged_samples.max(1) as f64) * 100.0;
            analysis.push_str(&format!("  - {}: {} ({:.1}%)\n", reason, count, percentage));
        }

        analysis
    }

    /// Get metrics suitable for SEAL learning
    pub fn to_seal_metrics(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();

        map.insert(
            "execution_id".to_string(),
            serde_json::json!(self.execution_id),
        );
        map.insert(
            "total_steps".to_string(),
            serde_json::json!(self.total_steps),
        );
        map.insert(
            "completed_steps".to_string(),
            serde_json::json!(self.completed_steps),
        );
        map.insert(
            "success_rate".to_string(),
            serde_json::json!(self.actual_success_rate),
        );
        map.insert(
            "average_votes".to_string(),
            serde_json::json!(self.average_votes_per_step),
        );
        map.insert(
            "red_flag_rate".to_string(),
            serde_json::json!(self.red_flagged_samples as f64 / self.total_samples.max(1) as f64),
        );
        map.insert(
            "cost_per_step".to_string(),
            serde_json::json!(self.cost_per_step),
        );
        map.insert(
            "time_per_step_ms".to_string(),
            serde_json::json!(self.average_time_per_step_ms),
        );
        map.insert(
            "final_success".to_string(),
            serde_json::json!(self.final_success),
        );

        if let Some(ref model) = self.model {
            map.insert("model".to_string(), serde_json::json!(model));
        }

        map
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Aggregated metrics across multiple MDAP executions
#[derive(Clone, Debug, Default)]
pub struct AggregatedMetrics {
    /// Number of executions
    pub execution_count: u64,
    /// Total steps across all executions
    pub total_steps: u64,
    /// Total completed steps
    pub completed_steps: u64,
    /// Total cost
    pub total_cost_usd: f64,
    /// Average success rate
    pub average_success_rate: f64,
    /// Average votes per step
    pub average_votes_per_step: f64,
    /// Average red-flag rate
    pub average_red_flag_rate: f64,
    /// Most common red-flag reasons
    pub common_red_flags: HashMap<String, u64>,
}

impl AggregatedMetrics {
    /// Add metrics from an execution
    pub fn add(&mut self, metrics: &MdapMetrics) {
        self.execution_count += 1;
        self.total_steps += metrics.total_steps;
        self.completed_steps += metrics.completed_steps;
        self.total_cost_usd += metrics.actual_cost_usd;

        // Running averages
        let n = self.execution_count as f64;
        self.average_success_rate =
            (self.average_success_rate * (n - 1.0) + metrics.actual_success_rate) / n;
        self.average_votes_per_step =
            (self.average_votes_per_step * (n - 1.0) + metrics.average_votes_per_step) / n;

        let red_flag_rate =
            metrics.red_flagged_samples as f64 / metrics.total_samples.max(1) as f64;
        self.average_red_flag_rate = (self.average_red_flag_rate * (n - 1.0) + red_flag_rate) / n;

        // Aggregate red-flag reasons
        for (reason, count) in &metrics.red_flag_breakdown {
            *self.common_red_flags.entry(reason.clone()).or_insert(0) += count;
        }
    }

    /// Generate summary
    pub fn summary(&self) -> String {
        format!(
            "Aggregated MDAP Metrics ({} executions):\n\
             - Steps: {}/{} ({:.1}% success)\n\
             - Total cost: ${:.4}\n\
             - Avg success rate: {:.1}%\n\
             - Avg votes/step: {:.1}\n\
             - Avg red-flag rate: {:.1}%",
            self.execution_count,
            self.completed_steps,
            self.total_steps,
            (self.completed_steps as f64 / self.total_steps.max(1) as f64) * 100.0,
            self.total_cost_usd,
            self.average_success_rate * 100.0,
            self.average_votes_per_step,
            self.average_red_flag_rate * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_subtask_metric(id: &str, succeeded: bool) -> SubtaskMetric {
        SubtaskMetric {
            subtask_id: id.to_string(),
            description: format!("Test subtask {}", id),
            samples_needed: 5,
            red_flags_hit: 1,
            red_flag_reasons: vec!["ResponseTooLong".to_string()],
            final_confidence: 0.9,
            execution_time_ms: 500,
            winner_votes: 3,
            total_votes: 4,
            succeeded,
            input_tokens: 100,
            output_tokens: 50,
            complexity_estimate: 0.5,
        }
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = MdapMetrics::new("test_exec_001");
        assert_eq!(metrics.execution_id, "test_exec_001");
        assert!(metrics.start_time.is_some());
    }

    #[test]
    fn test_record_subtask() {
        let mut metrics = MdapMetrics::new("test");
        metrics.total_steps = 2;

        metrics.record_subtask(make_subtask_metric("1", true));
        metrics.record_subtask(make_subtask_metric("2", true));

        assert_eq!(metrics.completed_steps, 2);
        assert_eq!(metrics.total_samples, 10);
        assert_eq!(metrics.red_flagged_samples, 2);
    }

    #[test]
    fn test_finalize() {
        let mut metrics = MdapMetrics::new("test");
        metrics.total_steps = 2;
        metrics.record_subtask(make_subtask_metric("1", true));
        metrics.record_subtask(make_subtask_metric("2", true));
        metrics.finalize(true);

        assert!(metrics.end_time.is_some());
        assert!(metrics.final_success);
        assert!(metrics.average_votes_per_step > 0.0);
        assert!(metrics.average_time_per_step_ms > 0.0);
    }

    #[test]
    fn test_summary() {
        let mut metrics = MdapMetrics::new("test");
        metrics.total_steps = 2;
        metrics.record_subtask(make_subtask_metric("1", true));
        metrics.finalize(true);

        let summary = metrics.summary();
        assert!(summary.contains("Steps:"));
        assert!(summary.contains("Samples:"));
        assert!(summary.contains("Cost:"));
    }

    #[test]
    fn test_red_flag_analysis() {
        let mut metrics = MdapMetrics::new("test");
        metrics.record_subtask(make_subtask_metric("1", true));
        metrics.red_flagged_samples = 5;

        let analysis = metrics.red_flag_analysis();
        assert!(analysis.contains("ResponseTooLong"));
    }

    #[test]
    fn test_to_seal_metrics() {
        let mut metrics = MdapMetrics::new("test");
        metrics.total_steps = 10;
        metrics.completed_steps = 9;
        metrics.model = Some("claude-3-sonnet".to_string());

        let seal_metrics = metrics.to_seal_metrics();
        assert_eq!(seal_metrics.get("execution_id").unwrap(), "test");
        assert_eq!(seal_metrics.get("total_steps").unwrap(), 10);
    }

    #[test]
    fn test_json_serialization() {
        let mut metrics = MdapMetrics::new("test");
        metrics.total_steps = 5;
        metrics.finalize(true);

        let json = metrics.to_json().unwrap();
        let restored = MdapMetrics::from_json(&json).unwrap();

        assert_eq!(restored.execution_id, metrics.execution_id);
        assert_eq!(restored.total_steps, metrics.total_steps);
    }

    #[test]
    fn test_aggregated_metrics() {
        let mut agg = AggregatedMetrics::default();

        let mut metrics1 = MdapMetrics::new("exec1");
        metrics1.total_steps = 10;
        metrics1.completed_steps = 10;
        metrics1.actual_success_rate = 1.0;
        metrics1.average_votes_per_step = 3.0;
        metrics1.actual_cost_usd = 0.01;
        agg.add(&metrics1);

        let mut metrics2 = MdapMetrics::new("exec2");
        metrics2.total_steps = 10;
        metrics2.completed_steps = 8;
        metrics2.actual_success_rate = 0.8;
        metrics2.average_votes_per_step = 5.0;
        metrics2.actual_cost_usd = 0.02;
        agg.add(&metrics2);

        assert_eq!(agg.execution_count, 2);
        assert_eq!(agg.total_steps, 20);
        assert_eq!(agg.completed_steps, 18);
        assert!((agg.average_success_rate - 0.9).abs() < 0.01);
        assert!((agg.average_votes_per_step - 4.0).abs() < 0.01);
    }
}
