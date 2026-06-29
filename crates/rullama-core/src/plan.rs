use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Status of a plan
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum PlanStatus {
    /// Plan is in draft state (not yet started).
    #[default]
    Draft,
    /// Plan is actively being executed.
    Active,
    /// Plan execution is paused.
    Paused,
    /// Plan has been completed successfully.
    Completed,
    /// Plan has been abandoned.
    Abandoned,
}

impl std::fmt::Display for PlanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanStatus::Draft => write!(f, "draft"),
            PlanStatus::Active => write!(f, "active"),
            PlanStatus::Paused => write!(f, "paused"),
            PlanStatus::Completed => write!(f, "completed"),
            PlanStatus::Abandoned => write!(f, "abandoned"),
        }
    }
}

impl std::str::FromStr for PlanStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "draft" => Ok(PlanStatus::Draft),
            "active" => Ok(PlanStatus::Active),
            "paused" => Ok(PlanStatus::Paused),
            "completed" => Ok(PlanStatus::Completed),
            "abandoned" => Ok(PlanStatus::Abandoned),
            _ => Err(format!("Unknown plan status: {}", s)),
        }
    }
}

/// Metadata for a persisted execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanMetadata {
    /// Unique plan identifier.
    pub plan_id: String,
    /// Conversation this plan belongs to.
    pub conversation_id: String,
    /// Short title derived from the task description.
    pub title: String,
    /// Full task description the plan was created for.
    pub task_description: String,
    /// The plan content (steps, instructions).
    pub plan_content: String,
    /// Model used to generate the plan, if known.
    pub model_id: Option<String>,
    /// Current status of the plan.
    pub status: PlanStatus,
    /// Whether the plan has been executed.
    pub executed: bool,
    /// Number of iterations used during execution.
    pub iterations_used: u32,
    /// Unix timestamp when the plan was created.
    pub created_at: i64,
    /// Unix timestamp when the plan was last updated.
    pub updated_at: i64,
    /// File path if the plan was exported to disk.
    pub file_path: Option<String>,
    /// Optional embedding vector for similarity search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Parent plan ID for branched plans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_plan_id: Option<String>,
    /// IDs of child (branched) plans.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub child_plan_ids: Vec<String>,
    /// Branch name for branched plans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_name: Option<String>,
    /// Whether this branch has been merged back.
    #[serde(default)]
    pub merged: bool,
    /// Nesting depth in the plan tree.
    #[serde(default)]
    pub depth: u32,
}

impl PlanMetadata {
    /// Create a new plan with the given task and content
    pub fn new(conversation_id: String, task_description: String, plan_content: String) -> Self {
        let now = Utc::now().timestamp();
        let plan_id = uuid::Uuid::new_v4().to_string();

        let title = task_description
            .lines()
            .next()
            .unwrap_or(&task_description)
            .chars()
            .take(50)
            .collect::<String>();

        Self {
            plan_id,
            conversation_id,
            title,
            task_description,
            plan_content,
            model_id: None,
            status: PlanStatus::Draft,
            executed: false,
            iterations_used: 0,
            created_at: now,
            updated_at: now,
            file_path: None,
            embedding: None,
            parent_plan_id: None,
            child_plan_ids: Vec::new(),
            branch_name: None,
            merged: false,
            depth: 0,
        }
    }

    /// Create a branch (sub-plan) from this plan
    pub fn create_branch(
        &self,
        branch_name: String,
        task_description: String,
        plan_content: String,
    ) -> Self {
        let mut branch = Self::new(self.conversation_id.clone(), task_description, plan_content);
        branch.parent_plan_id = Some(self.plan_id.clone());
        branch.branch_name = Some(branch_name);
        branch.depth = self.depth + 1;
        branch
    }

    /// Add a child plan ID
    pub fn add_child(&mut self, child_id: String) {
        if !self.child_plan_ids.contains(&child_id) {
            self.child_plan_ids.push(child_id);
            self.updated_at = Utc::now().timestamp();
        }
    }

    /// Mark as merged
    pub fn mark_merged(&mut self) {
        self.merged = true;
        self.status = PlanStatus::Completed;
        self.updated_at = Utc::now().timestamp();
    }

    /// Check if this is a root plan
    pub fn is_root(&self) -> bool {
        self.parent_plan_id.is_none()
    }

    /// Check if this plan has children
    pub fn has_children(&self) -> bool {
        !self.child_plan_ids.is_empty()
    }

    /// Set the model used
    pub fn with_model(mut self, model_id: String) -> Self {
        self.model_id = Some(model_id);
        self
    }

    /// Set iterations used
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations_used = iterations;
        self
    }

    /// Mark as executed
    pub fn mark_executed(&mut self) {
        self.executed = true;
        self.status = PlanStatus::Completed;
        self.updated_at = Utc::now().timestamp();
    }

    /// Update status
    pub fn set_status(&mut self, status: PlanStatus) {
        self.status = status;
        self.updated_at = Utc::now().timestamp();
    }

    /// Set file path after export
    pub fn set_file_path(&mut self, path: String) {
        self.file_path = Some(path);
        self.updated_at = Utc::now().timestamp();
    }

    /// Get created_at as DateTime
    pub fn created_at_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp(self.created_at, 0).unwrap_or_else(Utc::now)
    }

    /// Generate markdown export with YAML frontmatter
    pub fn to_markdown(&self) -> String {
        let created = self.created_at_datetime().format("%Y-%m-%dT%H:%M:%SZ");
        let model = self.model_id.as_deref().unwrap_or("unknown");

        format!(
            r#"---
plan_id: {}
conversation_id: {}
title: "{}"
status: {}
executed: {}
iterations: {}
created_at: {}
model: {}
---

# Execution Plan: {}

## Original Task

{}

## Plan

{}

---
*Generated by Brainwires Agent Framework*
"#,
            self.plan_id,
            self.conversation_id,
            self.title.replace('"', r#"\""#),
            self.status,
            self.executed,
            self.iterations_used,
            created,
            model,
            self.title,
            self.task_description,
            self.plan_content
        )
    }
}

/// A single step in a serializable pre-execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Sequential step index, 1-based.
    pub step_number: u32,
    /// Short human-readable description of what this step will do.
    pub description: String,
    /// Name of the tool this step is expected to invoke, if known.
    pub tool_hint: Option<String>,
    /// Estimated tokens this step will consume (prompt + completion combined).
    pub estimated_tokens: u64,
}

/// Budget constraints that a serializable plan must satisfy before execution
/// begins. Used in `TaskAgentConfig::plan_budget`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanBudget {
    /// Reject plans with more steps than this limit.
    pub max_steps: Option<u32>,
    /// Reject plans whose total estimated tokens exceed this ceiling.
    pub max_estimated_tokens: Option<u64>,
    /// Reject plans whose estimated cost (USD) exceeds this ceiling.
    pub max_estimated_cost_usd: Option<f64>,
    /// Cost per token used for the USD estimate. Default: 0.000003 ($3/M).
    pub cost_per_token: f64,
}

impl Default for PlanBudget {
    fn default() -> Self {
        Self {
            max_steps: None,
            max_estimated_tokens: None,
            max_estimated_cost_usd: None,
            cost_per_token: 0.000003,
        }
    }
}

impl PlanBudget {
    /// Create a new budget with no limits (accepts any plan).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum allowed step count.
    pub fn with_max_steps(mut self, max: u32) -> Self {
        self.max_steps = Some(max);
        self
    }

    /// Set the maximum allowed estimated token count.
    pub fn with_max_tokens(mut self, max: u64) -> Self {
        self.max_estimated_tokens = Some(max);
        self
    }

    /// Set the maximum allowed estimated cost in USD.
    pub fn with_max_cost_usd(mut self, max: f64) -> Self {
        self.max_estimated_cost_usd = Some(max);
        self
    }

    /// Check a plan against this budget.
    ///
    /// Returns `Ok(())` when the plan is within budget, or `Err(reason)` with
    /// a human-readable explanation when any limit is exceeded.
    pub fn check(&self, plan: &SerializablePlan) -> Result<(), String> {
        let step_count = plan.steps.len() as u32;
        let total_tokens = plan.total_estimated_tokens();
        let total_cost = total_tokens as f64 * self.cost_per_token;

        if let Some(max) = self.max_steps
            && step_count > max
        {
            return Err(format!(
                "plan has {} steps but limit is {}",
                step_count, max
            ));
        }

        if let Some(max) = self.max_estimated_tokens
            && total_tokens > max
        {
            return Err(format!(
                "plan estimates {} tokens but limit is {}",
                total_tokens, max
            ));
        }

        if let Some(max) = self.max_estimated_cost_usd
            && total_cost > max
        {
            return Err(format!(
                "plan estimates ${:.6} USD but limit is ${:.6}",
                total_cost, max
            ));
        }

        Ok(())
    }
}

/// A serializable execution plan produced by the agent *before* any side
/// effects occur.  When `TaskAgentConfig::plan_budget` is set, the agent
/// generates this plan in a separate provider call and validates it against the
/// budget; if the budget is exceeded the run fails immediately with
/// `FailureCategory::PlanBudgetExceeded`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePlan {
    /// Unique identifier for this plan.
    pub plan_id: String,
    /// The original task description the plan was built for.
    pub task_description: String,
    /// Ordered steps in the plan.
    pub steps: Vec<PlanStep>,
    /// Unix timestamp (seconds) when the plan was generated.
    pub created_at: i64,
}

impl SerializablePlan {
    /// Create a new plan with a generated ID and the current timestamp.
    pub fn new(task_description: String, steps: Vec<PlanStep>) -> Self {
        Self {
            plan_id: uuid::Uuid::new_v4().to_string(),
            task_description,
            steps,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Sum of all `estimated_tokens` across steps.
    pub fn total_estimated_tokens(&self) -> u64 {
        self.steps.iter().map(|s| s.estimated_tokens).sum()
    }

    /// Number of steps in this plan.
    pub fn step_count(&self) -> u32 {
        self.steps.len() as u32
    }

    /// Parse a plan from model-generated text that contains an embedded JSON
    /// object with a `"steps"` array.
    ///
    /// Finds the first `{` … last `}` span in `text`, parses the JSON, and
    /// extracts the steps array.  Returns `None` when no valid plan JSON is
    /// found or the steps array is empty.
    pub fn parse_from_text(task_description: String, text: &str) -> Option<Self> {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        if start > end {
            return None;
        }
        let json_str = &text[start..=end];
        let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let steps_array = value.get("steps")?.as_array()?;

        let steps: Vec<PlanStep> = steps_array
            .iter()
            .enumerate()
            .filter_map(|(i, step)| {
                let description = step.get("description")?.as_str()?.to_string();
                let estimated_tokens = step
                    .get("estimated_tokens")
                    .or_else(|| step.get("tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(500);
                let tool_hint = step
                    .get("tool")
                    .or_else(|| step.get("tool_hint"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                Some(PlanStep {
                    step_number: (i + 1) as u32,
                    description,
                    tool_hint,
                    estimated_tokens,
                })
            })
            .collect();

        if steps.is_empty() {
            return None;
        }

        Some(Self::new(task_description, steps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_metadata_new() {
        let plan = PlanMetadata::new(
            "conv-123".to_string(),
            "Implement auth".to_string(),
            "Step 1".to_string(),
        );
        assert!(!plan.plan_id.is_empty());
        assert_eq!(plan.status, PlanStatus::Draft);
        assert!(plan.is_root());
    }

    #[test]
    fn test_plan_branching() {
        let parent = PlanMetadata::new(
            "conv-123".to_string(),
            "Main".to_string(),
            "Plan".to_string(),
        );
        let branch = parent.create_branch(
            "feature-x".to_string(),
            "Feature X".to_string(),
            "Branch plan".to_string(),
        );
        assert_eq!(branch.parent_plan_id, Some(parent.plan_id));
        assert_eq!(branch.depth, 1);
        assert!(!branch.is_root());
    }

    #[test]
    fn test_plan_budget_check_no_limits() {
        let budget = PlanBudget::new();
        let plan = SerializablePlan::new(
            "task".into(),
            vec![PlanStep {
                step_number: 1,
                description: "do thing".into(),
                tool_hint: None,
                estimated_tokens: 9_000_000,
            }],
        );
        // No limits set — always passes
        assert!(budget.check(&plan).is_ok());
    }

    #[test]
    fn test_plan_budget_check_step_limit_exceeded() {
        let budget = PlanBudget::new().with_max_steps(2);
        let steps: Vec<PlanStep> = (1..=3)
            .map(|i| PlanStep {
                step_number: i,
                description: format!("step {i}"),
                tool_hint: None,
                estimated_tokens: 100,
            })
            .collect();
        let plan = SerializablePlan::new("task".into(), steps);
        let result = budget.check(&plan);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("3 steps"));
    }

    #[test]
    fn test_plan_budget_check_token_limit_exceeded() {
        let budget = PlanBudget::new().with_max_tokens(500);
        let steps: Vec<PlanStep> = (1..=3)
            .map(|i| PlanStep {
                step_number: i,
                description: format!("step {i}"),
                tool_hint: None,
                estimated_tokens: 300,
            })
            .collect();
        let plan = SerializablePlan::new("task".into(), steps);
        let result = budget.check(&plan);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("900 tokens"));
    }

    #[test]
    fn test_serializable_plan_parse_from_text() {
        let text = r#"Here is my plan:
{"steps":[{"description":"Read the file","tool":"read_file","estimated_tokens":300},{"description":"Write changes","tool":"write_file","estimated_tokens":500}]}
That's the plan."#;
        let plan = SerializablePlan::parse_from_text("task".into(), text).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].step_number, 1);
        assert_eq!(plan.steps[0].description, "Read the file");
        assert_eq!(plan.steps[0].tool_hint, Some("read_file".to_string()));
        assert_eq!(plan.total_estimated_tokens(), 800);
    }

    #[test]
    fn test_serializable_plan_parse_empty_steps_returns_none() {
        let text = r#"{"steps":[]}"#;
        assert!(SerializablePlan::parse_from_text("task".into(), text).is_none());
    }

    #[test]
    fn test_serializable_plan_parse_no_json_returns_none() {
        assert!(SerializablePlan::parse_from_text("task".into(), "no json here").is_none());
    }
}
