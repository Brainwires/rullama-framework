//! Task Decomposition Module
//!
//! Implements task decomposition strategies for MDAP, breaking complex tasks
//! into minimal subtasks that can be executed by microagents.
//!
//! The paper's approach is Maximal Agentic Decomposition (MAD), where
//! each subtask should be as small as possible (m=1).

pub mod recursive;

// Re-export commonly used types from recursive
pub use recursive::{BinaryRecursiveDecomposer, SimpleRecursiveDecomposer};

use super::error::{DecompositionError, MdapResult};
use super::microagent::Subtask;

/// Context for task decomposition
#[derive(Clone, Debug)]
pub struct DecomposeContext {
    /// Working directory for file operations
    pub working_directory: String,
    /// Available tools for the agent
    pub available_tools: Vec<String>,
    /// Maximum decomposition depth
    pub max_depth: u32,
    /// Current depth in recursive decomposition
    pub current_depth: u32,
    /// Additional context/constraints
    pub additional_context: Option<String>,
}

impl Default for DecomposeContext {
    fn default() -> Self {
        Self {
            working_directory: ".".to_string(),
            available_tools: Vec::new(),
            max_depth: 10,
            current_depth: 0,
            additional_context: None,
        }
    }
}

impl DecomposeContext {
    /// Create a new context
    pub fn new(working_directory: impl Into<String>) -> Self {
        Self {
            working_directory: working_directory.into(),
            ..Default::default()
        }
    }

    /// Add available tools
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    /// Set max depth
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Add additional context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.additional_context = Some(context.into());
        self
    }

    /// Create a child context (increment depth)
    pub fn child(&self) -> Self {
        Self {
            working_directory: self.working_directory.clone(),
            available_tools: self.available_tools.clone(),
            max_depth: self.max_depth,
            current_depth: self.current_depth + 1,
            additional_context: self.additional_context.clone(),
        }
    }

    /// Check if we've exceeded max depth
    pub fn at_max_depth(&self) -> bool {
        self.current_depth >= self.max_depth
    }
}

/// Result of task decomposition
#[derive(Clone, Debug)]
pub struct DecompositionResult {
    /// The subtasks resulting from decomposition
    pub subtasks: Vec<Subtask>,
    /// How to combine results from subtasks
    pub composition_function: CompositionFunction,
    /// Whether the task is already minimal (cannot decompose further)
    pub is_minimal: bool,
    /// Estimated total complexity (sum of subtask complexities)
    pub total_complexity: f32,
}

impl DecompositionResult {
    /// Create a minimal (atomic) result
    pub fn atomic(subtask: Subtask) -> Self {
        let complexity = subtask.complexity_estimate;
        Self {
            subtasks: vec![subtask],
            composition_function: CompositionFunction::Identity,
            is_minimal: true,
            total_complexity: complexity,
        }
    }

    /// Create a result with multiple subtasks
    pub fn composite(subtasks: Vec<Subtask>, composition: CompositionFunction) -> Self {
        let total_complexity: f32 = subtasks.iter().map(|s| s.complexity_estimate).sum();
        Self {
            subtasks,
            composition_function: composition,
            is_minimal: false,
            total_complexity,
        }
    }
}

/// How to combine results from subtasks
#[derive(Clone, Debug)]
pub enum CompositionFunction {
    /// Single result, no composition needed
    Identity,
    /// Concatenate all results
    Concatenate,
    /// Merge as a sequence
    Sequence,
    /// Combine into an object with subtask IDs as keys
    ObjectMerge,
    /// Take the last result only
    LastOnly,
    /// Custom composition (described as a prompt)
    Custom(String),
    /// Reduce with an operation
    Reduce {
        /// The reduce operation to apply.
        operation: String,
    },
}

impl CompositionFunction {
    /// Get a description of this composition function
    pub fn description(&self) -> String {
        match self {
            CompositionFunction::Identity => "identity (single result)".to_string(),
            CompositionFunction::Concatenate => "concatenate all results".to_string(),
            CompositionFunction::Sequence => "merge as sequence".to_string(),
            CompositionFunction::ObjectMerge => "merge into object".to_string(),
            CompositionFunction::LastOnly => "take last result".to_string(),
            CompositionFunction::Custom(desc) => format!("custom: {}", desc),
            CompositionFunction::Reduce { operation } => format!("reduce with {}", operation),
        }
    }
}

/// Task decomposition strategy
#[derive(Clone, Debug)]
pub enum DecompositionStrategy {
    /// Binary recursive decomposition (paper's approach for multiplication)
    BinaryRecursive {
        /// Maximum recursion depth.
        max_depth: u32,
    },
    /// Simple text-based decomposition (for testing)
    Simple {
        /// Maximum recursion depth.
        max_depth: u32,
    },
    /// Sequential step-by-step decomposition
    Sequential,
    /// Domain-specific decomposition for code operations
    CodeOperations,
    /// AI-driven decomposition with discriminator voting
    AIDriven {
        /// Number of discriminator votes (k).
        discriminator_k: u32,
    },
    /// No decomposition (execute as single task)
    None,
}

impl Default for DecompositionStrategy {
    fn default() -> Self {
        DecompositionStrategy::BinaryRecursive { max_depth: 10 }
    }
}

/// Trait for task decomposers
#[async_trait::async_trait]
pub trait TaskDecomposer: Send + Sync {
    /// Decompose a task into subtasks
    async fn decompose(
        &self,
        task: &str,
        context: &DecomposeContext,
    ) -> MdapResult<DecompositionResult>;

    /// Check if a task is already minimal (cannot decompose further)
    fn is_minimal(&self, task: &str) -> bool;

    /// Get the decomposition strategy
    fn strategy(&self) -> DecompositionStrategy;
}

/// Simple sequential decomposer that breaks tasks into numbered steps
pub struct SequentialDecomposer {
    max_steps: u32,
}

impl SequentialDecomposer {
    /// Create a new sequential decomposer with the given step limit.
    pub fn new(max_steps: u32) -> Self {
        Self { max_steps }
    }
}

impl Default for SequentialDecomposer {
    fn default() -> Self {
        Self::new(20)
    }
}

#[async_trait::async_trait]
impl TaskDecomposer for SequentialDecomposer {
    async fn decompose(
        &self,
        task: &str,
        context: &DecomposeContext,
    ) -> MdapResult<DecompositionResult> {
        // Simple heuristic: if task has numbered steps, extract them
        let lines: Vec<&str> = task.lines().collect();
        let mut subtasks = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Check if line starts with a number
            let is_numbered = trimmed
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);

            if is_numbered || subtasks.is_empty() {
                let subtask = Subtask::new(
                    format!("step_{}", i + 1),
                    trimmed.to_string(),
                    serde_json::json!({
                        "step": i + 1,
                        "context": context.additional_context
                    }),
                )
                .with_complexity(1.0 / lines.len() as f32);

                subtasks.push(subtask);
            }

            if subtasks.len() >= self.max_steps as usize {
                break;
            }
        }

        if subtasks.is_empty() {
            // Treat as single task
            let subtask = Subtask::atomic(task);
            return Ok(DecompositionResult::atomic(subtask));
        }

        // Add dependencies (each step depends on previous)
        for i in 1..subtasks.len() {
            let prev_id = subtasks[i - 1].id.clone();
            subtasks[i].depends_on.push(prev_id);
        }

        Ok(DecompositionResult::composite(
            subtasks,
            CompositionFunction::Sequence,
        ))
    }

    fn is_minimal(&self, task: &str) -> bool {
        // Consider minimal if single line and short
        !task.contains('\n') && task.len() < 200
    }

    fn strategy(&self) -> DecompositionStrategy {
        DecompositionStrategy::Sequential
    }
}

/// No-op decomposer that treats everything as atomic
pub struct AtomicDecomposer;

#[async_trait::async_trait]
impl TaskDecomposer for AtomicDecomposer {
    async fn decompose(
        &self,
        task: &str,
        _context: &DecomposeContext,
    ) -> MdapResult<DecompositionResult> {
        Ok(DecompositionResult::atomic(Subtask::atomic(task)))
    }

    fn is_minimal(&self, _task: &str) -> bool {
        true
    }

    fn strategy(&self) -> DecompositionStrategy {
        DecompositionStrategy::None
    }
}

/// Validate decomposition result
pub fn validate_decomposition(result: &DecompositionResult) -> MdapResult<()> {
    if result.subtasks.is_empty() {
        return Err(DecompositionError::EmptyResult(
            "Decomposition produced no subtasks".to_string(),
        )
        .into());
    }

    // Check for circular dependencies
    let mut visited = std::collections::HashSet::new();
    for subtask in &result.subtasks {
        visited.insert(subtask.id.clone());
    }

    for subtask in &result.subtasks {
        for dep in &subtask.depends_on {
            if !visited.contains(dep) {
                return Err(DecompositionError::InvalidDependency {
                    subtask: subtask.id.clone(),
                    dependency: dep.clone(),
                }
                .into());
            }
        }
    }

    Ok(())
}

/// Order subtasks by dependencies (topological sort)
pub fn topological_sort(subtasks: &[Subtask]) -> MdapResult<Vec<Subtask>> {
    use std::collections::{HashMap, VecDeque};

    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();

    // Initialize
    for subtask in subtasks {
        in_degree.insert(subtask.id.clone(), subtask.depends_on.len());
        graph.insert(subtask.id.clone(), Vec::new());
    }

    // Build reverse graph (who depends on whom)
    for subtask in subtasks {
        for dep in &subtask.depends_on {
            if let Some(dependents) = graph.get_mut(dep) {
                dependents.push(subtask.id.clone());
            }
        }
    }

    // Find all subtasks with no dependencies
    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(id, _)| id.clone())
        .collect();

    let mut result = Vec::new();
    let subtask_map: HashMap<_, _> = subtasks.iter().map(|s| (s.id.clone(), s.clone())).collect();

    while let Some(id) = queue.pop_front() {
        if let Some(subtask) = subtask_map.get(&id) {
            result.push(subtask.clone());
        }

        if let Some(dependents) = graph.get(&id) {
            for dependent in dependents {
                if let Some(deg) = in_degree.get_mut(dependent) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dependent.clone());
                    }
                }
            }
        }
    }

    if result.len() != subtasks.len() {
        return Err(DecompositionError::CircularDependency(
            "Circular dependency detected in subtasks".to_string(),
        )
        .into());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_context() {
        let ctx = DecomposeContext::new("/home/user/project")
            .with_tools(vec!["read".to_string(), "write".to_string()])
            .with_max_depth(5);

        assert_eq!(ctx.working_directory, "/home/user/project");
        assert_eq!(ctx.available_tools.len(), 2);
        assert_eq!(ctx.max_depth, 5);
    }

    #[test]
    fn test_context_child() {
        let parent = DecomposeContext::new("/home").with_max_depth(5);
        let child = parent.child();

        assert_eq!(child.current_depth, 1);
        assert_eq!(child.max_depth, 5);
    }

    #[test]
    fn test_decomposition_result_atomic() {
        let subtask = Subtask::atomic("Test");
        let result = DecompositionResult::atomic(subtask);

        assert!(result.is_minimal);
        assert_eq!(result.subtasks.len(), 1);
    }

    #[test]
    fn test_topological_sort_simple() {
        let subtasks = vec![
            Subtask::new("a", "Task A", serde_json::Value::Null),
            Subtask::new("b", "Task B", serde_json::Value::Null).depends_on(vec!["a".to_string()]),
            Subtask::new("c", "Task C", serde_json::Value::Null).depends_on(vec!["b".to_string()]),
        ];

        let sorted = topological_sort(&subtasks).unwrap();
        assert_eq!(sorted[0].id, "a");
        assert_eq!(sorted[1].id, "b");
        assert_eq!(sorted[2].id, "c");
    }

    #[test]
    fn test_topological_sort_parallel() {
        let subtasks = vec![
            Subtask::new("a", "Task A", serde_json::Value::Null),
            Subtask::new("b", "Task B", serde_json::Value::Null),
            Subtask::new("c", "Task C", serde_json::Value::Null)
                .depends_on(vec!["a".to_string(), "b".to_string()]),
        ];

        let sorted = topological_sort(&subtasks).unwrap();
        // a and b should come before c
        let c_pos = sorted.iter().position(|s| s.id == "c").unwrap();
        let a_pos = sorted.iter().position(|s| s.id == "a").unwrap();
        let b_pos = sorted.iter().position(|s| s.id == "b").unwrap();
        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_topological_sort_circular() {
        let subtasks = vec![
            Subtask::new("a", "Task A", serde_json::Value::Null).depends_on(vec!["c".to_string()]),
            Subtask::new("b", "Task B", serde_json::Value::Null).depends_on(vec!["a".to_string()]),
            Subtask::new("c", "Task C", serde_json::Value::Null).depends_on(vec!["b".to_string()]),
        ];

        let result = topological_sort(&subtasks);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_atomic_decomposer() {
        let decomposer = AtomicDecomposer;
        let result = decomposer
            .decompose("Test task", &DecomposeContext::default())
            .await
            .unwrap();

        assert!(result.is_minimal);
        assert_eq!(result.subtasks.len(), 1);
    }

    #[tokio::test]
    async fn test_sequential_decomposer() {
        let decomposer = SequentialDecomposer::new(10);
        let task = "1. First step\n2. Second step\n3. Third step";
        let result = decomposer
            .decompose(task, &DecomposeContext::default())
            .await
            .unwrap();

        assert_eq!(result.subtasks.len(), 3);
        assert!(!result.is_minimal);
    }

    #[test]
    fn test_validate_decomposition_valid() {
        let result = DecompositionResult::composite(
            vec![
                Subtask::new("a", "Task A", serde_json::Value::Null),
                Subtask::new("b", "Task B", serde_json::Value::Null)
                    .depends_on(vec!["a".to_string()]),
            ],
            CompositionFunction::Sequence,
        );

        assert!(validate_decomposition(&result).is_ok());
    }

    #[test]
    fn test_validate_decomposition_invalid_dep() {
        let result = DecompositionResult::composite(
            vec![
                Subtask::new("a", "Task A", serde_json::Value::Null)
                    .depends_on(vec!["nonexistent".to_string()]),
            ],
            CompositionFunction::Sequence,
        );

        assert!(validate_decomposition(&result).is_err());
    }
}
