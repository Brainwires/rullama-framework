//! MDAP - Massively Decomposed Agentic Processes
//!
//! Implementation of the MAKER framework from "Solving a Million-Step LLM Task with Zero Errors".
//!
//! All core MDAP modules are re-exported from the brainwires-mdap framework crate.
//! CLI-specific modules: tool_executor (local), plus MdapConfig/MdapExecutor (inline).

// Re-export all framework modules
pub use brainwires_mdap::composer;
pub use brainwires_mdap::decomposition;
pub use brainwires_mdap::error;
pub use brainwires_mdap::metrics;
pub use brainwires_mdap::microagent;
pub use brainwires_mdap::red_flags;
pub use brainwires_mdap::scaling;
pub use brainwires_mdap::tool_intent;
pub use brainwires_mdap::voting;

// CLI-specific module (kept local)
pub mod tool_executor;

// Re-exports from framework modules for convenience
pub use composer::{Composer, CompositionBuilder};
pub use decomposition::{
    AtomicDecomposer, CompositionFunction, DecomposeContext, DecompositionResult,
    DecompositionStrategy, SequentialDecomposer, TaskDecomposer,
};
pub use error::{MdapError, MdapResult};
pub use metrics::{ConfigSummary, MdapMetrics, SubtaskMetric, VotingRoundMetric};
pub use microagent::{
    Microagent, MicroagentConfig, MicroagentConfigBuilder, MicroagentProvider, MicroagentResponse,
    Subtask, SubtaskOutput,
};
pub use red_flags::{
    AcceptAllValidator, OutputFormat, RedFlagConfig, RedFlagConfigBuilder, RedFlagReason,
    RedFlagResult, RedFlagValidator, StandardRedFlagValidator,
};
pub use scaling::{
    MdapEstimate, ModelCosts, calculate_expected_cost, calculate_k_min, calculate_p_full,
    estimate_mdap, estimate_per_step_success, estimate_valid_response_rate,
};
pub use tool_intent::{
    IntentParseResult, SubtaskOutputWithIntent, ToolCategory, ToolIntent, ToolSchema,
    parse_tool_intent,
};
pub use voting::{
    FirstToAheadByKVoter, ResponseMetadata, SampledResponse, VoteResult, VoterBuilder,
};

// Re-exports from CLI-specific module
pub use tool_executor::{
    MicroagentToolConfig, MicroagentToolExecutor, MicroagentToolExecutorBuilder,
};

use std::collections::HashMap;
use std::sync::Arc;

use brainwires::reasoning::{ComplexityScorer, RecommendedStrategy, StrategySelector};

/// Main MDAP configuration
#[derive(Clone, Debug)]
pub struct MdapConfig {
    /// Vote margin threshold (k in paper)
    pub k: u32,
    /// Target success probability (t in paper, e.g., 0.95)
    pub target_success_rate: f64,
    /// Maximum parallel sampling threads (capped at 4)
    pub parallel_samples: u32,
    /// Maximum samples before giving up on a subtask
    pub max_samples_per_subtask: u32,
    /// Red-flagging configuration
    pub red_flags: RedFlagConfig,
    /// Decomposition strategy
    pub decomposition: DecompositionStrategy,
    /// Whether to use SEAL integration
    pub seal_integration: bool,
    /// Maximum decomposition depth
    pub max_decomposition_depth: u32,
    /// Estimated cost per sample in USD (for cost estimation)
    pub cost_per_sample_usd: Option<f64>,
    /// Whether to fail fast on first subtask failure
    pub fail_fast: bool,
    /// Enable difficulty-aware k adjustment (arxiv:2509.11079v1)
    /// When enabled, k is dynamically adjusted based on task complexity
    pub adaptive_k: bool,
    /// Minimum k value when using adaptive k
    pub k_min: u32,
    /// Maximum k value when using adaptive k
    pub k_max: u32,
}

impl Default for MdapConfig {
    fn default() -> Self {
        Self {
            k: 3,
            target_success_rate: 0.95,
            parallel_samples: 4, // Max 4 threads per user requirement
            max_samples_per_subtask: 50,
            red_flags: RedFlagConfig::strict(), // Paper's approach
            decomposition: DecompositionStrategy::BinaryRecursive { max_depth: 10 },
            seal_integration: true,
            max_decomposition_depth: 10,
            cost_per_sample_usd: Some(0.0001), // Default estimate
            fail_fast: false,
            adaptive_k: true, // Enable by default for cost savings
            k_min: 2,         // Minimum k for easy tasks
            k_max: 5,         // Maximum k for hard tasks
        }
    }
}

impl MdapConfig {
    /// Create a new configuration builder
    pub fn builder() -> MdapConfigBuilder {
        MdapConfigBuilder::default()
    }

    /// Create configuration for high-reliability tasks
    pub fn high_reliability() -> Self {
        Self {
            k: 5,
            target_success_rate: 0.99,
            max_samples_per_subtask: 100,
            adaptive_k: true,
            k_min: 4,
            k_max: 7,
            ..Default::default()
        }
    }

    /// Create configuration for cost-optimized tasks
    pub fn cost_optimized() -> Self {
        Self {
            k: 2,
            target_success_rate: 0.90,
            max_samples_per_subtask: 30,
            parallel_samples: 2,
            adaptive_k: true,
            k_min: 1,
            k_max: 3,
            ..Default::default()
        }
    }

    /// Calculate adaptive k based on task complexity (arxiv:2509.11079v1)
    pub fn adaptive_k_for_subtask(&self, complexity: f32, observed_variance: Option<f64>) -> u32 {
        if !self.adaptive_k {
            return self.k;
        }

        let complexity_clamped = complexity.clamp(0.0, 1.0);
        let base_k = self.k_min as f32 + complexity_clamped * (self.k_max - self.k_min) as f32;

        let variance_adjustment = observed_variance
            .map(|v| (v * 2.0).min(1.0) as f32)
            .unwrap_or(0.0);

        let final_k = base_k + variance_adjustment;
        (final_k.ceil() as u32).clamp(self.k_min, self.k_max)
    }

    pub async fn adaptive_k_with_local_scorer(
        &self,
        task_description: &str,
        scorer: Option<&ComplexityScorer>,
        observed_variance: Option<f64>,
    ) -> u32 {
        if !self.adaptive_k {
            return self.k;
        }

        let complexity = if let Some(s) = scorer {
            match s.score(task_description).await {
                Some(result) => result.score,
                None => s.score_heuristic(task_description).score,
            }
        } else {
            0.5
        };

        self.adaptive_k_for_subtask(complexity, observed_variance)
    }

    pub async fn with_auto_strategy(
        mut self,
        task: &str,
        selector: Option<&StrategySelector>,
    ) -> Self {
        let result = if let Some(s) = selector {
            match s.select_strategy(task).await {
                Some(r) => r,
                None => s.select_heuristic(task),
            }
        } else {
            let _selector = brainwires::reasoning::StrategySelectorBuilder::default();
            self.select_strategy_heuristic(task)
        };

        self.decomposition = match result.strategy {
            RecommendedStrategy::BinaryRecursive { max_depth } => {
                DecompositionStrategy::BinaryRecursive { max_depth }
            }
            RecommendedStrategy::Sequential => DecompositionStrategy::Sequential,
            RecommendedStrategy::CodeOperations => DecompositionStrategy::CodeOperations,
            RecommendedStrategy::None => DecompositionStrategy::None,
        };

        self
    }

    fn select_strategy_heuristic(&self, task: &str) -> brainwires::reasoning::StrategyResult {
        use brainwires::reasoning::{RecommendedStrategy, StrategyResult, TaskType};

        let lower = task.to_lowercase();
        let word_count = task.split_whitespace().count();

        let task_type = if lower.contains("implement")
            || lower.contains("code")
            || lower.contains("function")
        {
            TaskType::Code
        } else if lower.contains("plan") || lower.contains("design") {
            TaskType::Planning
        } else if lower.contains("analyze") || lower.contains("research") {
            TaskType::Analysis
        } else if lower.contains("just") || lower.contains("simply") {
            TaskType::Simple
        } else {
            TaskType::Unknown
        };

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
            _ => {
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

    pub fn adaptive_k_with_heuristic(
        &self,
        task_description: &str,
        scorer: &ComplexityScorer,
        observed_variance: Option<f64>,
    ) -> u32 {
        if !self.adaptive_k {
            return self.k;
        }

        let result = scorer.score_heuristic(task_description);
        self.adaptive_k_for_subtask(result.score, observed_variance)
    }

    /// Validate the configuration
    pub fn validate(&self) -> MdapResult<()> {
        use error::MdapConfigError;

        if self.k < 1 {
            return Err(MdapConfigError::InvalidK(self.k).into());
        }

        if self.target_success_rate <= 0.0 || self.target_success_rate >= 1.0 {
            return Err(MdapConfigError::InvalidTargetSuccessRate(self.target_success_rate).into());
        }

        if self.parallel_samples < 1 || self.parallel_samples > 4 {
            return Err(MdapConfigError::InvalidParallelSamples(self.parallel_samples).into());
        }

        if self.max_samples_per_subtask < 1 {
            return Err(MdapConfigError::InvalidMaxSamples(self.max_samples_per_subtask).into());
        }

        if self.max_decomposition_depth < 1 {
            return Err(MdapConfigError::InvalidMaxDepth(self.max_decomposition_depth).into());
        }

        Ok(())
    }

    /// Convert to configuration summary for metrics
    pub fn to_summary(&self) -> ConfigSummary {
        ConfigSummary {
            k: self.k,
            target_success_rate: self.target_success_rate,
            parallel_samples: self.parallel_samples,
            max_samples_per_subtask: self.max_samples_per_subtask,
            decomposition_strategy: format!("{:?}", self.decomposition),
        }
    }
}

/// Builder for MdapConfig
#[derive(Default)]
pub struct MdapConfigBuilder {
    config: MdapConfig,
}

impl MdapConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn k(mut self, k: u32) -> Self {
        self.config.k = k;
        self
    }

    pub fn target_success_rate(mut self, rate: f64) -> Self {
        self.config.target_success_rate = rate;
        self
    }

    pub fn parallel_samples(mut self, samples: u32) -> Self {
        self.config.parallel_samples = samples.clamp(1, 4);
        self
    }

    pub fn max_samples_per_subtask(mut self, max: u32) -> Self {
        self.config.max_samples_per_subtask = max;
        self
    }

    pub fn red_flags(mut self, config: RedFlagConfig) -> Self {
        self.config.red_flags = config;
        self
    }

    pub fn decomposition(mut self, strategy: DecompositionStrategy) -> Self {
        self.config.decomposition = strategy;
        self
    }

    pub fn seal_integration(mut self, enabled: bool) -> Self {
        self.config.seal_integration = enabled;
        self
    }

    pub fn max_decomposition_depth(mut self, depth: u32) -> Self {
        self.config.max_decomposition_depth = depth;
        self
    }

    pub fn cost_per_sample_usd(mut self, cost: f64) -> Self {
        self.config.cost_per_sample_usd = Some(cost);
        self
    }

    pub fn fail_fast(mut self, enabled: bool) -> Self {
        self.config.fail_fast = enabled;
        self
    }

    pub fn build(self) -> MdapResult<MdapConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    pub fn build_unchecked(self) -> MdapConfig {
        self.config
    }
}

/// Result from MDAP execution
#[derive(Clone, Debug)]
pub struct MdapResult2 {
    /// The final output value
    pub output: serde_json::Value,
    /// Execution metrics
    pub metrics: MdapMetrics,
    /// Whether execution was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Main MDAP executor - orchestrates the full pipeline
pub struct MdapExecutor<P: MicroagentProvider> {
    provider: Arc<P>,
    config: MdapConfig,
    decomposer: Box<dyn TaskDecomposer>,
    metrics: MdapMetrics,
}

impl<P: MicroagentProvider + 'static> MdapExecutor<P> {
    /// Create a new executor
    pub fn new(provider: Arc<P>, config: MdapConfig) -> Self {
        let decomposer: Box<dyn TaskDecomposer> = match &config.decomposition {
            DecompositionStrategy::BinaryRecursive { max_depth } => {
                Box::new(decomposition::recursive::BinaryRecursiveDecomposer::new(
                    provider.clone(),
                    *max_depth,
                    config.k,
                ))
            }
            DecompositionStrategy::Sequential => Box::new(SequentialDecomposer::default()),
            DecompositionStrategy::None => Box::new(AtomicDecomposer),
            _ => Box::new(SequentialDecomposer::default()),
        };

        Self {
            provider,
            config,
            decomposer,
            metrics: MdapMetrics::default(),
        }
    }

    /// Execute a task using the full MDAP pipeline
    pub async fn execute(
        &mut self,
        task: &str,
        working_directory: &str,
    ) -> MdapResult<MdapResult2> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        self.metrics = MdapMetrics::with_config(execution_id.clone(), self.config.to_summary());
        self.metrics.start();

        // 1. Decompose task
        let decompose_context = DecomposeContext::new(working_directory)
            .with_max_depth(self.config.max_decomposition_depth);

        let decomposition = self.decomposer.decompose(task, &decompose_context).await?;
        self.metrics.total_steps = decomposition.subtasks.len() as u64;

        // 2. Estimate cost
        let estimate = estimate_mdap(
            decomposition.subtasks.len() as u64,
            0.99,
            0.95,
            0.0001,
            self.config.target_success_rate,
        )?;
        self.metrics.estimated_cost_usd = estimate.expected_cost_usd;
        self.metrics.estimated_success_probability = estimate.success_probability;

        let k = self.config.k.max(estimate.recommended_k);

        // 3. Execute subtasks in dependency order
        let voter = FirstToAheadByKVoter::new(k, self.config.max_samples_per_subtask);
        let ordered_subtasks = decomposition::topological_sort(&decomposition.subtasks)?;
        let mut results: HashMap<String, SubtaskOutput> = HashMap::new();

        for subtask in ordered_subtasks {
            let input = self.gather_inputs(&subtask, &results)?;

            let microagent = Microagent::new(
                self.provider.clone(),
                subtask.clone(),
                MicroagentConfig::default(),
            );

            let start = std::time::Instant::now();
            let vote_result = microagent.execute_with_voting(&input, &voter).await?;
            let elapsed = start.elapsed();

            self.metrics.record_subtask(SubtaskMetric {
                subtask_id: subtask.id.clone(),
                description: subtask.description.clone(),
                samples_needed: vote_result.total_samples,
                red_flags_hit: vote_result.red_flagged_count,
                red_flag_reasons: vote_result.red_flag_reasons.clone(),
                final_confidence: vote_result.confidence,
                execution_time_ms: elapsed.as_millis() as u64,
                winner_votes: vote_result.winner_votes,
                total_votes: vote_result.total_votes,
                succeeded: true,
                input_tokens: 0,
                output_tokens: 0,
                complexity_estimate: subtask.complexity_estimate,
            });

            results.insert(subtask.id.clone(), vote_result.winner);
        }

        // 4. Compose final result
        let outputs: Vec<SubtaskOutput> = decomposition
            .subtasks
            .iter()
            .filter_map(|s| results.get(&s.id).cloned())
            .collect();

        let composer = Composer::new();
        let final_output = composer.compose(&outputs, &decomposition.composition_function)?;

        // 5. Finalize metrics
        self.metrics.finalize(true);

        Ok(MdapResult2 {
            output: final_output,
            metrics: self.metrics.clone(),
            success: true,
            error: None,
        })
    }

    fn gather_inputs(
        &self,
        subtask: &Subtask,
        results: &HashMap<String, SubtaskOutput>,
    ) -> MdapResult<serde_json::Value> {
        let mut input = subtask.input_state.clone();

        if input.is_null() {
            input = serde_json::json!({});
        }

        if let Some(obj) = input.as_object_mut() {
            for dep_id in &subtask.depends_on {
                if let Some(dep_result) = results.get(dep_id) {
                    obj.insert(format!("dep_{}", dep_id), dep_result.value.clone());
                }
            }
        }

        Ok(input)
    }

    pub fn metrics(&self) -> &MdapMetrics {
        &self.metrics
    }

    pub fn config(&self) -> &MdapConfig {
        &self.config
    }
}

// =============================================================================
// Local LLM Provider Implementation for Microagents
// =============================================================================

#[cfg(feature = "llama-cpp-2")]
mod local_llm_mdap {
    use super::*;
    use crate::providers::local_llm::{LocalInferenceParams, LocalLlmProvider};

    /// Newtype wrapper around `LocalLlmProvider` to satisfy Rust's orphan
    /// rules — both `MicroagentProvider` (defined in `brainwires_agent`) and
    /// `LocalLlmProvider` (re-exported from `brainwires::providers::local_llm`)
    /// are foreign to this crate, so the impl has to hang off a local type.
    pub struct LocalLlmMicroagent(pub Arc<LocalLlmProvider>);

    #[async_trait::async_trait]
    impl MicroagentProvider for LocalLlmMicroagent {
        async fn chat(
            &self,
            system: &str,
            user: &str,
            temperature: f32,
            max_tokens: u32,
        ) -> MdapResult<MicroagentResponse> {
            use brainwires::reasoning::InferenceTimer;

            let timer = InferenceTimer::new("microagent_chat", self.0.config().name.as_str());
            let start = std::time::Instant::now();

            let prompt = format!("{}\n\n{}\n", system, user);

            let params = LocalInferenceParams {
                temperature,
                max_tokens,
                top_p: 0.9,
                top_k: if temperature < 0.2 { 10 } else { 40 },
                repeat_penalty: 1.1,
                stop_sequences: vec![],
            };

            let result = self.0.generate(&prompt, &params).await;
            let elapsed = start.elapsed();

            match result {
                Ok(text) => {
                    timer.finish(true);
                    let input_tokens = ((system.len() + user.len()) / 4) as u32;
                    let output_tokens = (text.len() / 4) as u32;
                    Ok(MicroagentResponse {
                        text,
                        input_tokens,
                        output_tokens,
                        finish_reason: Some("stop".to_string()),
                        response_time_ms: elapsed.as_millis() as u64,
                    })
                }
                Err(e) => {
                    timer.finish(false);
                    Err(error::MicroagentError::ProviderError(e.to_string()).into())
                }
            }
        }

        fn available_tools(&self) -> Vec<ToolSchema> {
            // The model's `supports_tools` flag (see `LocalLlmProviderConfig` /
            // `KnownModel`) describes whether the underlying weights were
            // trained for function calling — it's informational metadata used
            // by CLI listings (`local-models list`, etc.), not a dispatch
            // switch. The actual tool-call wiring lives at the orchestrator
            // layer, and `LocalLlmProvider::generate` exposes a plain
            // text-completion API with no tool-schema parameter. Until that
            // changes upstream, microagents driven by the local backend run
            // tool-free regardless of model capability.
            Vec::new()
        }
    }

    /// Helper to create a local microagent for simple subtasks
    pub fn create_local_microagent(
        provider: Arc<LocalLlmProvider>,
        subtask: Subtask,
    ) -> Microagent<LocalLlmMicroagent> {
        let config = MicroagentConfigBuilder::new()
            .max_output_tokens(512)
            .temperature(0.1)
            .timeout_ms(10000)
            .build();
        Microagent::new(Arc::new(LocalLlmMicroagent(provider)), subtask, config)
    }

    /// Determine if a subtask is suitable for local execution
    pub fn is_suitable_for_local(subtask: &Subtask) -> bool {
        if subtask.complexity_estimate >= 0.4 {
            return false;
        }
        if let Some(OutputFormat::JsonWithFields(_) | OutputFormat::Custom { .. }) =
            subtask.expected_output_format.as_ref()
            && subtask.complexity_estimate >= 0.3
        {
            return false;
        }
        if subtask.description.len() > 500 {
            return false;
        }
        true
    }
}

#[cfg(feature = "llama-cpp-2")]
pub use local_llm_mdap::{create_local_microagent, is_suitable_for_local};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MdapConfig::default();
        assert_eq!(config.k, 3);
        assert_eq!(config.parallel_samples, 4);
        assert!(config.target_success_rate > 0.9);
    }

    #[test]
    fn test_config_builder() {
        let config = MdapConfig::builder()
            .k(5)
            .target_success_rate(0.99)
            .parallel_samples(2)
            .build()
            .unwrap();

        assert_eq!(config.k, 5);
        assert_eq!(config.target_success_rate, 0.99);
        assert_eq!(config.parallel_samples, 2);
    }

    #[test]
    fn test_config_validation() {
        let result = MdapConfig::builder().k(0).build();
        assert!(result.is_err());

        let result = MdapConfig::builder().target_success_rate(1.5).build();
        assert!(result.is_err());

        let config = MdapConfig::builder().parallel_samples(10).build_unchecked();
        assert_eq!(config.parallel_samples, 4);
    }

    #[test]
    fn test_config_presets() {
        let high_rel = MdapConfig::high_reliability();
        assert_eq!(high_rel.k, 5);
        assert_eq!(high_rel.target_success_rate, 0.99);

        let cost_opt = MdapConfig::cost_optimized();
        assert_eq!(cost_opt.k, 2);
        assert_eq!(cost_opt.target_success_rate, 0.90);
    }

    #[test]
    fn test_config_to_summary() {
        let config = MdapConfig::default();
        let summary = config.to_summary();

        assert_eq!(summary.k, config.k);
        assert_eq!(summary.target_success_rate, config.target_success_rate);
    }
}
