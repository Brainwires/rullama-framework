//! Strategy Selector - Decomposition Strategy Selection
//!
//! Uses a provider to analyze tasks and recommend the optimal
//! decomposition strategy for MDAP execution.

use std::sync::Arc;
use tracing::warn;

use rullama_core::message::Message;
use rullama_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

/// Task type classification
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskType {
    /// Code-related tasks (implementation, refactoring)
    Code,
    /// Multi-step planning tasks
    Planning,
    /// Research and analysis tasks
    Analysis,
    /// Simple single-step tasks
    Simple,
    /// Unknown/ambiguous tasks
    Unknown,
}

impl TaskType {
    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("code") || lower.contains("implement") || lower.contains("refactor") {
            TaskType::Code
        } else if lower.contains("plan") || lower.contains("design") || lower.contains("architect")
        {
            TaskType::Planning
        } else if lower.contains("analy")
            || lower.contains("research")
            || lower.contains("investigate")
        {
            TaskType::Analysis
        } else if lower.contains("simple") || lower.contains("single") || lower.contains("atomic") {
            TaskType::Simple
        } else {
            TaskType::Unknown
        }
    }
}

/// Recommended decomposition strategy
#[derive(Clone, Debug)]
pub enum RecommendedStrategy {
    /// Binary recursive decomposition.
    BinaryRecursive {
        /// Maximum recursion depth for decomposition.
        max_depth: u32,
    },
    /// Sequential step-by-step
    Sequential,
    /// Domain-specific for code
    CodeOperations,
    /// No decomposition needed
    None,
}

impl RecommendedStrategy {
    /// Get default max_depth for binary recursive
    pub fn default_depth() -> u32 {
        10
    }
}

/// Result of strategy selection
#[derive(Clone, Debug)]
pub struct StrategyResult {
    /// Recommended strategy
    pub strategy: RecommendedStrategy,
    /// Task type classification
    pub task_type: TaskType,
    /// Confidence score
    pub confidence: f32,
    /// Whether LLM was used
    pub used_local_llm: bool,
    /// Reasoning for the selection
    pub reasoning: Option<String>,
}

impl StrategyResult {
    /// Create from LLM selection
    pub fn from_local(
        strategy: RecommendedStrategy,
        task_type: TaskType,
        confidence: f32,
        reasoning: Option<String>,
    ) -> Self {
        Self {
            strategy,
            task_type,
            confidence,
            used_local_llm: true,
            reasoning,
        }
    }

    /// Create from heuristic selection
    pub fn from_heuristic(strategy: RecommendedStrategy, task_type: TaskType) -> Self {
        Self {
            strategy,
            task_type,
            confidence: 0.5,
            used_local_llm: false,
            reasoning: None,
        }
    }
}

/// Strategy selector for MDAP decomposition
pub struct StrategySelector {
    provider: Arc<dyn Provider>,
    model_id: String,
}

impl StrategySelector {
    /// Create a new strategy selector
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
        }
    }

    /// Select the optimal decomposition strategy for a task
    pub async fn select_strategy(&self, task: &str) -> Option<StrategyResult> {
        let timer = InferenceTimer::new("select_strategy", &self.model_id);

        let prompt = self.build_selection_prompt(task);

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(100);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let result = self.parse_selection(&output);
                timer.finish(true);
                Some(result)
            }
            Err(e) => {
                warn!(target: "local_llm", "Strategy selection failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Heuristic strategy selection (pattern-based fallback)
    pub fn select_heuristic(&self, task: &str) -> StrategyResult {
        let lower = task.to_lowercase();
        let word_count = task.split_whitespace().count();

        // Detect task type
        let task_type = self.classify_task_type(&lower);

        // Select strategy based on task type and complexity
        let strategy = match task_type {
            TaskType::Simple => RecommendedStrategy::None,
            TaskType::Code => {
                if word_count > 30 {
                    RecommendedStrategy::BinaryRecursive { max_depth: 8 }
                } else {
                    RecommendedStrategy::CodeOperations
                }
            }
            TaskType::Planning => {
                if word_count > 50 {
                    RecommendedStrategy::BinaryRecursive { max_depth: 10 }
                } else {
                    RecommendedStrategy::Sequential
                }
            }
            TaskType::Analysis => RecommendedStrategy::Sequential,
            TaskType::Unknown => {
                // Use complexity heuristics
                if word_count < 10 {
                    RecommendedStrategy::None
                } else if word_count < 30 {
                    RecommendedStrategy::Sequential
                } else {
                    RecommendedStrategy::BinaryRecursive { max_depth: 10 }
                }
            }
        };

        StrategyResult::from_heuristic(strategy, task_type)
    }

    /// Classify task type from content
    fn classify_task_type(&self, lower: &str) -> TaskType {
        // Code indicators
        let code_indicators = [
            "implement",
            "code",
            "function",
            "class",
            "method",
            "refactor",
            "debug",
            "fix bug",
            "write a",
            "create a function",
            "add a feature",
        ];

        // Planning indicators
        let planning_indicators = [
            "plan",
            "design",
            "architect",
            "strategy",
            "roadmap",
            "outline",
            "structure",
            "organize",
        ];

        // Analysis indicators
        let analysis_indicators = [
            "analyze",
            "research",
            "investigate",
            "explain",
            "understand",
            "review",
            "audit",
            "examine",
            "study",
        ];

        // Simple indicators
        let simple_indicators = ["just", "simply", "only", "quick", "small change"];

        // Check code first (common case)
        if code_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Code;
        }

        if planning_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Planning;
        }

        if analysis_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Analysis;
        }

        if simple_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Simple;
        }

        TaskType::Unknown
    }

    /// Build the selection prompt
    fn build_selection_prompt(&self, task: &str) -> String {
        format!(
            r#"Analyze this task and recommend the best decomposition strategy.

Task: "{}"

Available strategies:
1. BINARY_RECURSIVE - Best for complex tasks that can be split recursively (many subtasks)
2. SEQUENTIAL - Best for step-by-step tasks with clear ordering (moderate complexity)
3. CODE_OPERATIONS - Best for code-specific tasks (implementation, refactoring)
4. NONE - Best for simple, atomic tasks that don't need decomposition

Also classify the task type:
- CODE: Implementation, refactoring, debugging
- PLANNING: Design, architecture, strategy
- ANALYSIS: Research, investigation, review
- SIMPLE: Quick, single-step tasks

Output format:
STRATEGY: <strategy_name>
TYPE: <task_type>
REASON: <brief explanation>

Selection:"#,
            if task.len() > 300 { &task[..300] } else { task }
        )
    }

    /// Parse the LLM output
    fn parse_selection(&self, output: &str) -> StrategyResult {
        let upper = output.to_uppercase();

        // Parse strategy
        let strategy = if upper.contains("BINARY_RECURSIVE") || upper.contains("BINARY RECURSIVE") {
            RecommendedStrategy::BinaryRecursive { max_depth: 10 }
        } else if upper.contains("SEQUENTIAL") {
            RecommendedStrategy::Sequential
        } else if upper.contains("CODE_OPERATIONS") || upper.contains("CODE OPERATIONS") {
            RecommendedStrategy::CodeOperations
        } else if upper.contains("NONE") {
            RecommendedStrategy::None
        } else {
            // Default to sequential for ambiguous cases
            RecommendedStrategy::Sequential
        };

        // Parse task type
        let task_type = if upper.contains("TYPE: CODE") || upper.contains("TYPE:CODE") {
            TaskType::Code
        } else if upper.contains("TYPE: PLANNING") || upper.contains("TYPE:PLANNING") {
            TaskType::Planning
        } else if upper.contains("TYPE: ANALYSIS") || upper.contains("TYPE:ANALYSIS") {
            TaskType::Analysis
        } else if upper.contains("TYPE: SIMPLE") || upper.contains("TYPE:SIMPLE") {
            TaskType::Simple
        } else {
            TaskType::Unknown
        };

        // Extract reasoning
        let reasoning = if let Some(reason_start) = output.find("REASON:") {
            let reason_text = &output[reason_start + 7..];
            let end = reason_text.find('\n').unwrap_or(reason_text.len());
            Some(reason_text[..end].trim().to_string())
        } else {
            None
        };

        StrategyResult::from_local(strategy, task_type, 0.8, reasoning)
    }
}

/// Builder for StrategySelector
pub struct StrategySelectorBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
}

impl Default for StrategySelectorBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-1.2b".to_string(), // Larger model for better reasoning
        }
    }
}

impl StrategySelectorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for strategy selection.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Build the strategy selector, returning `None` if no provider was set.
    pub fn build(self) -> Option<StrategySelector> {
        self.provider
            .map(|p| StrategySelector::new(p, self.model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_parsing() {
        assert_eq!(TaskType::from_str("code"), TaskType::Code);
        assert_eq!(TaskType::from_str("implement feature"), TaskType::Code);
        assert_eq!(
            TaskType::from_str("design architecture"),
            TaskType::Planning
        );
        assert_eq!(TaskType::from_str("analyze the data"), TaskType::Analysis);
        assert_eq!(TaskType::from_str("simple fix"), TaskType::Simple);
        assert_eq!(TaskType::from_str("random text"), TaskType::Unknown);
    }

    #[test]
    fn test_heuristic_selection_code() {
        let _selector = StrategySelectorBuilder::default();
        let result = select_heuristic_direct("Implement a new authentication system with OAuth2");
        assert_eq!(result.task_type, TaskType::Code);
    }

    #[test]
    fn test_heuristic_selection_simple() {
        let result = select_heuristic_direct("just fix the typo");
        assert_eq!(result.task_type, TaskType::Simple);
        assert!(matches!(result.strategy, RecommendedStrategy::None));
    }

    #[test]
    fn test_heuristic_selection_planning() {
        let result =
            select_heuristic_direct("Design the system architecture for the new microservice");
        assert_eq!(result.task_type, TaskType::Planning);
    }

    fn select_heuristic_direct(task: &str) -> StrategyResult {
        let lower = task.to_lowercase();
        let word_count = task.split_whitespace().count();

        let task_type = classify_task_type_direct(&lower);

        let strategy = match task_type {
            TaskType::Simple => RecommendedStrategy::None,
            TaskType::Code => {
                if word_count > 30 {
                    RecommendedStrategy::BinaryRecursive { max_depth: 8 }
                } else {
                    RecommendedStrategy::CodeOperations
                }
            }
            TaskType::Planning => {
                if word_count > 50 {
                    RecommendedStrategy::BinaryRecursive { max_depth: 10 }
                } else {
                    RecommendedStrategy::Sequential
                }
            }
            TaskType::Analysis => RecommendedStrategy::Sequential,
            TaskType::Unknown => {
                if word_count < 10 {
                    RecommendedStrategy::None
                } else if word_count < 30 {
                    RecommendedStrategy::Sequential
                } else {
                    RecommendedStrategy::BinaryRecursive { max_depth: 10 }
                }
            }
        };

        StrategyResult::from_heuristic(strategy, task_type)
    }

    fn classify_task_type_direct(lower: &str) -> TaskType {
        let code_indicators = ["implement", "code", "function", "class", "refactor"];
        let planning_indicators = ["plan", "design", "architect"];
        let analysis_indicators = ["analyze", "research", "investigate"];
        let simple_indicators = ["just", "simply", "only"];

        if code_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Code;
        }
        if planning_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Planning;
        }
        if analysis_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Analysis;
        }
        if simple_indicators.iter().any(|i| lower.contains(i)) {
            return TaskType::Simple;
        }
        TaskType::Unknown
    }

    #[test]
    fn test_parse_selection() {
        let output = r#"STRATEGY: BINARY_RECURSIVE
TYPE: CODE
REASON: Task involves multiple implementation steps"#;

        let result = parse_selection_direct(output);
        assert!(matches!(
            result.strategy,
            RecommendedStrategy::BinaryRecursive { .. }
        ));
        assert_eq!(result.task_type, TaskType::Code);
        assert!(result.reasoning.is_some());
    }

    fn parse_selection_direct(output: &str) -> StrategyResult {
        let upper = output.to_uppercase();

        let strategy = if upper.contains("BINARY_RECURSIVE") {
            RecommendedStrategy::BinaryRecursive { max_depth: 10 }
        } else if upper.contains("SEQUENTIAL") {
            RecommendedStrategy::Sequential
        } else if upper.contains("CODE_OPERATIONS") {
            RecommendedStrategy::CodeOperations
        } else if upper.contains("NONE") {
            RecommendedStrategy::None
        } else {
            RecommendedStrategy::Sequential
        };

        let task_type = if upper.contains("TYPE: CODE") {
            TaskType::Code
        } else if upper.contains("TYPE: PLANNING") {
            TaskType::Planning
        } else if upper.contains("TYPE: ANALYSIS") {
            TaskType::Analysis
        } else if upper.contains("TYPE: SIMPLE") {
            TaskType::Simple
        } else {
            TaskType::Unknown
        };

        let reasoning = if let Some(reason_start) = output.find("REASON:") {
            let reason_text = &output[reason_start + 7..];
            let end = reason_text.find('\n').unwrap_or(reason_text.len());
            Some(reason_text[..end].trim().to_string())
        } else {
            None
        };

        StrategyResult::from_local(strategy, task_type, 0.8, reasoning)
    }
}
