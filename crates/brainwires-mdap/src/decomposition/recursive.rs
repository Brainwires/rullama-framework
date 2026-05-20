//! Binary Recursive Task Decomposition
//!
//! Implements Algorithm 4 from the MAKER paper appendix for recursive
//! task decomposition with voting at each level.
//!
//! The approach:
//! 1. Sample N decompositions of the task
//! 2. Vote via discriminator to select best decomposition
//! 3. Recursively decompose subtasks
//! 4. Compose results with voting

use std::sync::Arc;

use super::super::error::{DecompositionError, MdapResult};
use super::super::microagent::{MicroagentProvider, Subtask};
use super::super::red_flags::{RedFlagConfig, StandardRedFlagValidator};
use super::super::voting::{FirstToAheadByKVoter, ResponseMetadata, SampledResponse};
use super::{
    CompositionFunction, DecomposeContext, DecompositionResult, DecompositionStrategy,
    TaskDecomposer,
};

/// Binary recursive decomposer implementing the paper's approach
///
/// Decomposes tasks by repeatedly splitting them in half until
/// reaching atomic subtasks (m=1).
pub struct BinaryRecursiveDecomposer<P> {
    provider: Arc<P>,
    max_depth: u32,
    k: u32,
    max_samples: u32,
}

impl<P: MicroagentProvider + 'static> BinaryRecursiveDecomposer<P> {
    /// Create a new binary recursive decomposer
    pub fn new(provider: Arc<P>, max_depth: u32, k: u32) -> Self {
        Self {
            provider,
            max_depth,
            k,
            max_samples: 50,
        }
    }

    /// Set max samples for voting
    pub fn with_max_samples(mut self, max_samples: u32) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Sample a decomposition of the task (static version for use with voting)
    async fn sample_decomposition_static(
        provider: Arc<P>,
        task: &str,
        context: &DecomposeContext,
        max_depth: u32,
    ) -> MdapResult<SampledResponse<DecompositionResult>> {
        let system_prompt = Self::build_decomposition_prompt_static(context, max_depth);
        let user_prompt = format!(
            "Decompose this task into exactly 2 smaller subtasks:\n\n{}\n\n\
             Output format:\n\
             SUBTASK_1: <description of first subtask>\n\
             SUBTASK_2: <description of second subtask>\n\
             COMPOSE: <how to combine results: sequence, concatenate, or merge>",
            task
        );

        let start = std::time::Instant::now();
        let response = provider
            .chat(&system_prompt, &user_prompt, 0.3, 500)
            .await?;
        let elapsed = start.elapsed();

        let metadata = ResponseMetadata {
            token_count: response.output_tokens,
            response_time_ms: elapsed.as_millis() as u64,
            format_valid: true,
            finish_reason: response.finish_reason,
            model: None,
        };

        let decomposition = Self::parse_decomposition_static(&response.text, task)?;

        Ok(SampledResponse {
            value: decomposition,
            metadata,
            raw_response: response.text,
            confidence: 0.75, // Default confidence
        })
    }

    /// Build the system prompt for decomposition
    fn build_decomposition_prompt_static(context: &DecomposeContext, max_depth: u32) -> String {
        let tools_str = if context.available_tools.is_empty() {
            "No specific tools available".to_string()
        } else {
            format!("Available tools: {}", context.available_tools.join(", "))
        };

        format!(
            "You are a task decomposition agent. Your job is to break down complex tasks \
             into exactly 2 smaller, independent subtasks.\n\n\
             Rules:\n\
             1. Each subtask should be roughly half the complexity of the original\n\
             2. Subtasks should be as independent as possible\n\
             3. The combination of subtasks should fully cover the original task\n\
             4. Be specific and actionable\n\n\
             Context:\n\
             - Working directory: {}\n\
             - {}\n\
             - Max decomposition depth: {}",
            context.working_directory, tools_str, max_depth
        )
    }

    /// Parse the decomposition response
    fn parse_decomposition_static(
        response: &str,
        original_task: &str,
    ) -> MdapResult<DecompositionResult> {
        let lines: Vec<&str> = response.lines().collect();
        let mut subtask_1_desc: Option<String> = None;
        let mut subtask_2_desc: Option<String> = None;
        let mut compose_method = CompositionFunction::Sequence;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("SUBTASK_1:") {
                subtask_1_desc = Some(trimmed.trim_start_matches("SUBTASK_1:").trim().to_string());
            } else if trimmed.starts_with("SUBTASK_2:") {
                subtask_2_desc = Some(trimmed.trim_start_matches("SUBTASK_2:").trim().to_string());
            } else if trimmed.starts_with("COMPOSE:") {
                let method = trimmed.trim_start_matches("COMPOSE:").trim().to_lowercase();
                compose_method = match method.as_str() {
                    "concatenate" | "concat" => CompositionFunction::Concatenate,
                    "merge" | "object" => CompositionFunction::ObjectMerge,
                    _ => CompositionFunction::Sequence,
                };
            }
        }

        match (subtask_1_desc, subtask_2_desc) {
            (Some(desc1), Some(desc2)) if !desc1.is_empty() && !desc2.is_empty() => {
                let subtask_1 = Subtask::new(
                    uuid::Uuid::new_v4().to_string(),
                    desc1,
                    serde_json::Value::Null,
                )
                .with_complexity(0.5);

                let subtask_2 = Subtask::new(
                    uuid::Uuid::new_v4().to_string(),
                    desc2,
                    serde_json::Value::Null,
                )
                .with_complexity(0.5)
                .depends_on(vec![subtask_1.id.clone()]);

                Ok(DecompositionResult::composite(
                    vec![subtask_1, subtask_2],
                    compose_method,
                ))
            }
            _ => {
                // Couldn't parse decomposition, treat as atomic
                Ok(DecompositionResult::atomic(Subtask::atomic(original_task)))
            }
        }
    }

    /// Check if a task description is minimal (cannot decompose further)
    fn is_task_minimal(&self, task: &str) -> bool {
        // Heuristics for minimal tasks:
        // 1. Very short descriptions
        if task.len() < 50 {
            return true;
        }

        // 2. Single action verbs
        let single_action_prefixes = [
            "return",
            "print",
            "output",
            "calculate",
            "compute",
            "add",
            "subtract",
            "multiply",
            "divide",
            "get",
            "set",
            "read",
            "write",
            "check",
            "verify",
        ];

        let lower = task.to_lowercase();
        for prefix in &single_action_prefixes {
            if lower.starts_with(prefix) && !task.contains(" and ") && !task.contains(" then ") {
                return true;
            }
        }

        // 3. No conjunctions suggesting multiple steps
        !task.contains(" and then ")
            && !task.contains(" followed by ")
            && !task.contains(". Then ")
            && task.matches('.').count() <= 1
    }

    /// Recursively decompose a task
    pub async fn decompose_recursive(
        &self,
        task: &str,
        context: &DecomposeContext,
    ) -> MdapResult<DecompositionResult> {
        // Check depth limit
        if context.at_max_depth() {
            return Err(DecompositionError::MaxDepthExceeded {
                depth: context.current_depth,
                max_depth: context.max_depth,
            }
            .into());
        }

        // Check if task is already minimal
        if self.is_task_minimal(task) {
            return Ok(DecompositionResult::atomic(Subtask::atomic(task)));
        }

        // Create voter for decomposition
        let voter = FirstToAheadByKVoter::new(self.k, self.max_samples);
        let validator = StandardRedFlagValidator::new(RedFlagConfig::relaxed(), None);

        // Clone data needed by the sampler closure
        let provider = self.provider.clone();
        let task_owned = task.to_string();
        let context_owned = context.clone();
        let max_depth = self.max_depth;

        // Vote on decomposition
        let vote_result = voter
            .vote(
                move || {
                    let provider = provider.clone();
                    let task = task_owned.clone();
                    let context = context_owned.clone();
                    async move {
                        Self::sample_decomposition_static(provider, &task, &context, max_depth)
                            .await
                    }
                },
                &validator,
                |result: &DecompositionResult| {
                    // Key by subtask descriptions
                    result
                        .subtasks
                        .iter()
                        .map(|s| s.description.clone())
                        .collect::<Vec<_>>()
                        .join("|")
                },
            )
            .await?;

        let decomposition = vote_result.winner;

        // If decomposition is minimal, return it
        if decomposition.is_minimal {
            return Ok(decomposition);
        }

        // Recursively decompose each subtask
        let child_context = context.child();
        let mut final_subtasks = Vec::new();

        for subtask in decomposition.subtasks {
            let sub_result =
                Box::pin(self.decompose_recursive(&subtask.description, &child_context)).await?;

            // Add all resulting subtasks
            for mut sub_subtask in sub_result.subtasks {
                // Update dependencies if needed
                if !subtask.depends_on.is_empty() && final_subtasks.is_empty() {
                    // First subtask of this branch inherits original dependencies
                    sub_subtask.depends_on.extend(subtask.depends_on.clone());
                }
                final_subtasks.push(sub_subtask);
            }
        }

        Ok(DecompositionResult::composite(
            final_subtasks,
            decomposition.composition_function,
        ))
    }
}

#[async_trait::async_trait]
impl<P: MicroagentProvider + 'static> TaskDecomposer for BinaryRecursiveDecomposer<P> {
    async fn decompose(
        &self,
        task: &str,
        context: &DecomposeContext,
    ) -> MdapResult<DecompositionResult> {
        self.decompose_recursive(task, context).await
    }

    fn is_minimal(&self, task: &str) -> bool {
        self.is_task_minimal(task)
    }

    fn strategy(&self) -> DecompositionStrategy {
        DecompositionStrategy::BinaryRecursive {
            max_depth: self.max_depth,
        }
    }
}

/// Simple decomposer that doesn't use LLM calls (for testing)
pub struct SimpleRecursiveDecomposer {
    max_depth: u32,
    min_task_length: usize,
}

impl SimpleRecursiveDecomposer {
    /// Create a new simple recursive decomposer with the given depth limit.
    pub fn new(max_depth: u32) -> Self {
        Self {
            max_depth,
            min_task_length: 50,
        }
    }

    fn decompose_by_sentences(
        &self,
        task: &str,
        context: &DecomposeContext,
    ) -> DecompositionResult {
        if context.at_max_depth() || task.len() < self.min_task_length {
            return DecompositionResult::atomic(Subtask::atomic(task));
        }

        // Split by sentences
        let sentences: Vec<&str> = task
            .split(['.', ';'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.len() <= 1 {
            return DecompositionResult::atomic(Subtask::atomic(task));
        }

        // Split roughly in half
        let mid = sentences.len() / 2;
        let first_half = sentences[..mid].join(". ");
        let second_half = sentences[mid..].join(". ");

        let subtask_1 = Subtask::new(
            uuid::Uuid::new_v4().to_string(),
            first_half,
            serde_json::Value::Null,
        )
        .with_complexity(0.5);

        let subtask_2 = Subtask::new(
            uuid::Uuid::new_v4().to_string(),
            second_half,
            serde_json::Value::Null,
        )
        .with_complexity(0.5)
        .depends_on(vec![subtask_1.id.clone()]);

        DecompositionResult::composite(vec![subtask_1, subtask_2], CompositionFunction::Sequence)
    }
}

#[async_trait::async_trait]
impl TaskDecomposer for SimpleRecursiveDecomposer {
    async fn decompose(
        &self,
        task: &str,
        context: &DecomposeContext,
    ) -> MdapResult<DecompositionResult> {
        Ok(self.decompose_by_sentences(task, context))
    }

    fn is_minimal(&self, task: &str) -> bool {
        task.len() < self.min_task_length || !task.contains('.')
    }

    fn strategy(&self) -> DecompositionStrategy {
        DecompositionStrategy::BinaryRecursive {
            max_depth: self.max_depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::microagent::MicroagentResponse;
    use super::*;

    struct MockProvider;

    #[async_trait::async_trait]
    impl MicroagentProvider for MockProvider {
        async fn chat(
            &self,
            _system: &str,
            _user: &str,
            _temperature: f32,
            _max_tokens: u32,
        ) -> MdapResult<MicroagentResponse> {
            Ok(MicroagentResponse {
                text: "SUBTASK_1: First part\nSUBTASK_2: Second part\nCOMPOSE: sequence"
                    .to_string(),
                input_tokens: 100,
                output_tokens: 50,
                finish_reason: Some("stop".to_string()),
                response_time_ms: 100,
            })
        }
    }

    #[test]
    fn test_is_task_minimal() {
        let decomposer = BinaryRecursiveDecomposer::new(Arc::new(MockProvider), 10, 3);

        // Short tasks are minimal
        assert!(decomposer.is_task_minimal("Return 42"));
        assert!(decomposer.is_task_minimal("Calculate 2+2"));

        // Tasks with conjunctions are not minimal
        assert!(
            !decomposer.is_task_minimal("First do this and then do that. Then verify the result.")
        );
    }

    #[test]
    fn test_parse_decomposition() {
        let response =
            "SUBTASK_1: Read the file\nSUBTASK_2: Process the content\nCOMPOSE: sequence";
        let result = BinaryRecursiveDecomposer::<MockProvider>::parse_decomposition_static(
            response,
            "test task",
        )
        .unwrap();

        assert_eq!(result.subtasks.len(), 2);
        assert!(!result.is_minimal);
    }

    #[test]
    fn test_parse_decomposition_invalid() {
        let response = "Invalid response without proper format";
        let result = BinaryRecursiveDecomposer::<MockProvider>::parse_decomposition_static(
            response,
            "test task",
        )
        .unwrap();

        // Should fall back to atomic
        assert!(result.is_minimal);
        assert_eq!(result.subtasks.len(), 1);
    }

    #[tokio::test]
    async fn test_simple_recursive_decomposer() {
        let decomposer = SimpleRecursiveDecomposer::new(5);
        // Task must be >= 50 chars to be decomposed (min_task_length)
        let task = "First sentence with some content here. Second sentence also with content. Third sentence.";
        let context = DecomposeContext::default();

        let result = decomposer.decompose(task, &context).await.unwrap();

        assert!(!result.is_minimal);
        assert_eq!(result.subtasks.len(), 2);
    }

    #[tokio::test]
    async fn test_simple_recursive_minimal() {
        let decomposer = SimpleRecursiveDecomposer::new(5);
        let task = "Short task";
        let context = DecomposeContext::default();

        let result = decomposer.decompose(task, &context).await.unwrap();

        assert!(result.is_minimal);
        assert_eq!(result.subtasks.len(), 1);
    }

    #[test]
    fn test_decomposition_strategy() {
        let decomposer = SimpleRecursiveDecomposer::new(5);
        assert!(matches!(
            decomposer.strategy(),
            DecompositionStrategy::BinaryRecursive { max_depth: 5 }
        ));
    }
}
