//! Planner Agent - LLM-powered dynamic task planner
//!
//! [`PlannerAgent`] wraps a [`TaskAgent`] with a planner-specific system prompt.
//! It explores the codebase using read-only tools and outputs structured JSON
//! describing tasks for worker agents to execute.
//!
//! The planner never directly mutates the task graph — it produces a
//! [`PlannerOutput`] that the [`CycleOrchestrator`](super::cycle_orchestrator)
//! interprets.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use brainwires_core::{Provider, Task, TaskPriority};

use crate::context::AgentContext;
use crate::system_prompts::planner_agent_prompt;
use crate::task_agent::{TaskAgent, TaskAgentConfig, TaskAgentResult};

// ── Public types ────────────────────────────────────────────────────────────

/// Priority level for dynamically created tasks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DynamicTaskPriority {
    /// Must be done first.
    Urgent,
    /// Important but not blocking.
    High,
    /// Default priority.
    Normal,
    /// Nice to have.
    Low,
}

impl From<DynamicTaskPriority> for TaskPriority {
    fn from(p: DynamicTaskPriority) -> Self {
        match p {
            DynamicTaskPriority::Urgent => TaskPriority::Urgent,
            DynamicTaskPriority::High => TaskPriority::High,
            DynamicTaskPriority::Normal => TaskPriority::Normal,
            DynamicTaskPriority::Low => TaskPriority::Low,
        }
    }
}

/// A task specification created dynamically by the planner at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTaskSpec {
    /// Unique identifier (typically a UUID assigned by the planner).
    pub id: String,
    /// Clear description of what the worker should do.
    pub description: String,
    /// File paths the task is expected to touch (hints for worktree scope).
    #[serde(default)]
    pub files_involved: Vec<String>,
    /// IDs of other specs this task depends on.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Task priority.
    #[serde(default = "default_priority")]
    pub priority: DynamicTaskPriority,
    /// Estimated iterations the worker will need.
    #[serde(default)]
    pub estimated_iterations: Option<u32>,
    /// Optional per-task agent config override.
    #[serde(skip)]
    pub agent_config_override: Option<TaskAgentConfig>,
}

fn default_priority() -> DynamicTaskPriority {
    DynamicTaskPriority::Normal
}

/// Request to spawn a sub-planner for a specific focus area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubPlannerRequest {
    /// Area of the codebase to focus on.
    pub focus_area: String,
    /// Additional context for the sub-planner.
    pub context: String,
    /// Maximum recursion depth remaining.
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,
}

fn default_max_depth() -> u32 {
    1
}

/// Output produced by a planner agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerOutput {
    /// Tasks to execute in this cycle.
    pub tasks: Vec<DynamicTaskSpec>,
    /// Optional sub-planners to spawn for deeper analysis.
    #[serde(default)]
    pub sub_planners: Vec<SubPlannerRequest>,
    /// Brief explanation of the overall plan.
    #[serde(default)]
    pub rationale: String,
}

/// Configuration for the planner agent.
#[derive(Debug, Clone)]
pub struct PlannerAgentConfig {
    /// LLM call budget for planning.
    pub max_iterations: u32,
    /// Maximum number of tasks per cycle.
    pub max_tasks: usize,
    /// Maximum number of sub-planners to spawn.
    pub max_sub_planners: usize,
    /// Maximum recursion depth for sub-planners.
    pub planning_depth: u32,
    /// Temperature for the planning LLM call.
    pub temperature: f32,
    /// Max tokens per LLM response.
    pub max_tokens: u32,
}

impl Default for PlannerAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_tasks: 15,
            max_sub_planners: 3,
            planning_depth: 2,
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

// ── PlannerAgent ────────────────────────────────────────────────────────────

/// An LLM-powered planner that explores the codebase and produces task plans.
///
/// Wraps a [`TaskAgent`] with a planner-specific system prompt. The agent runs
/// with read-only tools and its final output is parsed as structured JSON.
pub struct PlannerAgent {
    agent: Arc<TaskAgent>,
    config: PlannerAgentConfig,
}

impl PlannerAgent {
    /// Create a new planner agent.
    ///
    /// # Parameters
    /// - `id`: Unique agent identifier.
    /// - `goal`: The high-level objective to plan for.
    /// - `hints`: Guidance from previous cycles (empty on first cycle).
    /// - `provider`: AI provider for LLM calls.
    /// - `context`: Agent context (working directory, tools, etc.).
    /// - `config`: Planner-specific configuration.
    pub fn new(
        id: String,
        goal: &str,
        hints: &[String],
        provider: Arc<dyn Provider>,
        context: Arc<AgentContext>,
        config: PlannerAgentConfig,
    ) -> Self {
        let system_prompt = planner_agent_prompt(&id, &context.working_directory, goal, hints);

        let agent_config = TaskAgentConfig {
            max_iterations: config.max_iterations,
            system_prompt: Some(system_prompt),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            validation_config: None, // Planners don't need validation
            ..Default::default()
        };

        let task = Task::new(
            format!("planner-{}", uuid::Uuid::new_v4()),
            format!("Plan tasks for: {}", goal),
        );

        let agent = Arc::new(TaskAgent::new(id, task, provider, context, agent_config));

        Self { agent, config }
    }

    /// Execute the planner and return the parsed output.
    pub async fn execute(&self) -> Result<(PlannerOutput, TaskAgentResult)> {
        let result = self.agent.execute().await?;

        if !result.success {
            return Err(anyhow!("Planner agent failed: {}", result.summary));
        }

        let output = Self::parse_output(&result.summary, &self.config)?;
        Ok((output, result))
    }

    /// Parse planner output from the agent's summary text.
    ///
    /// Extracts JSON from markdown code fences or raw JSON in the text.
    pub fn parse_output(text: &str, config: &PlannerAgentConfig) -> Result<PlannerOutput> {
        let json_str = extract_json_block(text)
            .ok_or_else(|| anyhow!("No JSON block found in planner output"))?;

        let mut output: PlannerOutput = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse planner JSON: {}", e))?;

        // Enforce limits
        output.tasks.truncate(config.max_tasks);
        output.sub_planners.truncate(config.max_sub_planners);

        // Assign IDs to tasks that don't have them
        for task in &mut output.tasks {
            if task.id.is_empty() {
                task.id = uuid::Uuid::new_v4().to_string();
            }
        }

        // Validate: no circular dependencies within the plan
        validate_task_graph(&output.tasks)?;

        Ok(output)
    }

    /// Get a reference to the underlying agent.
    pub fn agent(&self) -> &Arc<TaskAgent> {
        &self.agent
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Extract a JSON block from text, looking for ```json fences first, then raw JSON.
fn extract_json_block(text: &str) -> Option<String> {
    // Try ```json ... ``` fences
    if let Some(start) = text.find("```json") {
        let content_start = start + "```json".len();
        if let Some(end) = text[content_start..].find("```") {
            return Some(text[content_start..content_start + end].trim().to_string());
        }
    }

    // Try ``` ... ``` fences (without json tag)
    if let Some(start) = text.find("```") {
        let content_start = start + "```".len();
        // Skip any language tag on the same line
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

    // Try to find raw JSON object
    if let Some(start) = text.find('{') {
        // Find matching closing brace
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

/// Validate that a set of task specs has no circular dependencies.
fn validate_task_graph(tasks: &[DynamicTaskSpec]) -> Result<()> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let id_set: HashSet<&str> = tasks.iter().map(|t| t.id.as_str()).collect();

    // Kahn's algorithm for topological sort / cycle detection
    let mut in_degree: HashMap<&str, usize> = tasks.iter().map(|t| (t.id.as_str(), 0)).collect();
    // in_degree[task] = number of deps task has within this plan
    for task in tasks {
        let count = task
            .depends_on
            .iter()
            .filter(|d| id_set.contains(d.as_str()))
            .count();
        in_degree.insert(task.id.as_str(), count);
    }

    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        // Find tasks that depend on this node and decrement their in-degree
        for task in tasks {
            if task.depends_on.iter().any(|d| d == node) && id_set.contains(task.id.as_str()) {
                let deg = in_degree.get_mut(task.id.as_str()).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(task.id.as_str());
                }
            }
        }
    }

    if visited < tasks.len() {
        return Err(anyhow!(
            "Circular dependency detected in planner task graph"
        ));
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_block_fenced() {
        let text = r#"Here is the plan:

```json
{"tasks": [], "rationale": "nothing to do"}
```

Done."#;
        let json = extract_json_block(text).unwrap();
        assert!(json.contains("tasks"));
    }

    #[test]
    fn test_extract_json_block_raw() {
        let text = r#"I think the plan is {"tasks": [], "rationale": "test"} and that's it."#;
        let json = extract_json_block(text).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["tasks"].is_array());
    }

    #[test]
    fn test_parse_planner_output() {
        let text = r#"```json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "Add error handling to parser",
      "files_involved": ["src/parser.rs"],
      "depends_on": [],
      "priority": "high",
      "estimated_iterations": 10
    },
    {
      "id": "task-2",
      "description": "Add tests for parser",
      "files_involved": ["tests/parser_test.rs"],
      "depends_on": ["task-1"],
      "priority": "normal",
      "estimated_iterations": 5
    }
  ],
  "sub_planners": [],
  "rationale": "Parser needs error handling before tests can be written"
}
```"#;

        let config = PlannerAgentConfig::default();
        let output = PlannerAgent::parse_output(text, &config).unwrap();
        assert_eq!(output.tasks.len(), 2);
        assert_eq!(output.tasks[0].id, "task-1");
        assert_eq!(output.tasks[1].depends_on, vec!["task-1"]);
        assert_eq!(
            output.rationale,
            "Parser needs error handling before tests can be written"
        );
    }

    #[test]
    fn test_validate_task_graph_no_cycle() {
        let tasks = vec![
            DynamicTaskSpec {
                id: "a".into(),
                description: "A".into(),
                files_involved: vec![],
                depends_on: vec![],
                priority: DynamicTaskPriority::Normal,
                estimated_iterations: None,
                agent_config_override: None,
            },
            DynamicTaskSpec {
                id: "b".into(),
                description: "B".into(),
                files_involved: vec![],
                depends_on: vec!["a".into()],
                priority: DynamicTaskPriority::Normal,
                estimated_iterations: None,
                agent_config_override: None,
            },
        ];
        assert!(validate_task_graph(&tasks).is_ok());
    }

    #[test]
    fn test_validate_task_graph_cycle() {
        let tasks = vec![
            DynamicTaskSpec {
                id: "a".into(),
                description: "A".into(),
                files_involved: vec![],
                depends_on: vec!["b".into()],
                priority: DynamicTaskPriority::Normal,
                estimated_iterations: None,
                agent_config_override: None,
            },
            DynamicTaskSpec {
                id: "b".into(),
                description: "B".into(),
                files_involved: vec![],
                depends_on: vec!["a".into()],
                priority: DynamicTaskPriority::Normal,
                estimated_iterations: None,
                agent_config_override: None,
            },
        ];
        assert!(validate_task_graph(&tasks).is_err());
    }

    #[test]
    fn test_truncate_limits() {
        let text = r#"```json
{
  "tasks": [
    {"id": "1", "description": "t1"},
    {"id": "2", "description": "t2"},
    {"id": "3", "description": "t3"}
  ],
  "sub_planners": [
    {"focus_area": "a", "context": "c", "max_depth": 1},
    {"focus_area": "b", "context": "c", "max_depth": 1}
  ],
  "rationale": "test"
}
```"#;

        let config = PlannerAgentConfig {
            max_tasks: 2,
            max_sub_planners: 1,
            ..Default::default()
        };
        let output = PlannerAgent::parse_output(text, &config).unwrap();
        assert_eq!(output.tasks.len(), 2);
        assert_eq!(output.sub_planners.len(), 1);
    }

    #[test]
    fn test_dynamic_task_priority_conversion() {
        assert_eq!(
            TaskPriority::from(DynamicTaskPriority::Urgent),
            TaskPriority::Urgent
        );
        assert_eq!(
            TaskPriority::from(DynamicTaskPriority::Normal),
            TaskPriority::Normal
        );
    }
}
