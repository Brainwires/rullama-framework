#![deny(missing_docs)]
//! # Brainwires Reasoning
//!
//! Layer 3 — Intelligence. Provider-agnostic reasoning primitives for the
//! Brainwires Agent Framework.
//!
//! This crate owns:
//!
//! - **`plan_parser`** — extract numbered task steps from LLM plan output.
//! - **`output_parser`** — parse structured data (JSON, regex) from raw LLM
//!   text.
//! - **Local inference scorers** — provider-agnostic TIER 1/2 components for
//!   routing, validation, complexity scoring, summarisation, retrieval
//!   gating, relevance ranking, strategy selection, and entity enhancement.
//!   All accept `Arc<dyn Provider>` and fall back to pattern-based logic
//!   when the provider is unavailable.
//!
//! ## Scope note — `prompting` stays in `brainwires-knowledge`
//!
//! The original architectural plan (`sleepy-popping-falcon.md`) called for
//! `prompting` to move here too. It didn't — `prompting` is tightly coupled
//! to `bks_pks` inside `brainwires-knowledge` and a move would have pulled
//! an entire knowledge-store dependency into this crate. The deviation is
//! intentional; tests and consumers of prompting should continue to target
//! `brainwires_prompting`.
//!
//! ## Configuration
//!
//! Enable via `LocalInferenceConfig`:
//! ```toml
//! [local_llm]
//! enabled = true
//! use_for_routing = true
//! use_for_validation = true
//! use_for_complexity = true
//! use_for_summarization = true
//! ```

// ── Parsers (moved from brainwires-core) ─────────────────────────────────
/// Structured output parsers for LLM responses.
pub mod output_parser;
/// Plan text parser for extracting steps from LLM output.
pub mod plan_parser;

// Flat re-exports for convenience.
pub use output_parser::{JsonListParser, JsonOutputParser, OutputParser, RegexOutputParser};
pub use plan_parser::{ParsedStep, parse_plan_steps, steps_to_tasks};

// ── Scorers (moved from brainwires-agent::reasoning) ────────────────────
mod complexity;
mod entity_enhancer;
mod relevance_scorer;
mod retrieval_classifier;
mod router;
/// Reasoning strategies: CoT, ReAct, Reflexion, Tree-of-Thoughts.
pub mod strategies;
mod strategy_selector;
mod summarizer;
mod validator;

pub use complexity::{ComplexityResult, ComplexityScorer, ComplexityScorerBuilder};
pub use entity_enhancer::{
    EnhancedEntity, EnhancedRelationship, EnhancementResult, EntityEnhancer, EntityEnhancerBuilder,
    RelationType, SemanticEntityType,
};
pub use relevance_scorer::{RelevanceResult, RelevanceScorer, RelevanceScorerBuilder};
pub use retrieval_classifier::{
    ClassificationResult, RetrievalClassifier, RetrievalClassifierBuilder,
    RetrievalNeed as LocalRetrievalNeed,
};
pub use router::{LocalRouter, LocalRouterBuilder, RouteResult};
pub use strategies::{
    ChainOfThoughtStrategy, ReActStrategy, ReasoningStrategy, ReflexionStrategy, StrategyPreset,
    StrategyStep, TreeOfThoughtsStrategy,
};
pub use strategy_selector::{
    RecommendedStrategy, StrategyResult, StrategySelector, StrategySelectorBuilder, TaskType,
};
pub use summarizer::{
    ExtractedFact, FactCategory, LocalSummarizer, LocalSummarizerBuilder, SummarizationResult,
};
pub use validator::{LocalValidator, LocalValidatorBuilder, ValidationResult};

use std::time::Instant;
use tracing::{info, warn};

/// Configuration for local inference components.
#[derive(Clone, Debug)]
pub struct LocalInferenceConfig {
    // TIER 1: Quick Wins
    /// Enable local routing
    pub routing_enabled: bool,
    /// Enable local validation
    pub validation_enabled: bool,
    /// Enable complexity scoring
    pub complexity_enabled: bool,

    // TIER 2: Context & Retrieval
    /// Enable local summarization for tiered memory
    pub summarization_enabled: bool,
    /// Enable local retrieval gating
    pub retrieval_gating_enabled: bool,
    /// Enable local relevance scoring
    pub relevance_scoring_enabled: bool,
    /// Enable local strategy selection
    pub strategy_selection_enabled: bool,
    /// Enable local entity enhancement
    pub entity_enhancement_enabled: bool,

    // Model selection per task
    /// Model ID to use for routing (fast model preferred)
    pub routing_model: Option<String>,
    /// Model ID to use for validation (fast model preferred)
    pub validation_model: Option<String>,
    /// Model ID to use for complexity scoring (fast model preferred)
    pub complexity_model: Option<String>,
    /// Model ID to use for summarization (larger model preferred)
    pub summarization_model: Option<String>,
    /// Model ID to use for retrieval classification (fast model preferred)
    pub retrieval_model: Option<String>,
    /// Model ID to use for relevance scoring (fast model preferred)
    pub relevance_model: Option<String>,
    /// Model ID to use for strategy selection (larger model preferred)
    pub strategy_model: Option<String>,
    /// Model ID to use for entity enhancement (fast model preferred)
    pub entity_model: Option<String>,

    /// Log all local inference calls
    pub log_inference: bool,
}

impl Default for LocalInferenceConfig {
    fn default() -> Self {
        Self {
            // TIER 1
            routing_enabled: false,
            validation_enabled: false,
            complexity_enabled: false,
            // TIER 2
            summarization_enabled: false,
            retrieval_gating_enabled: false,
            relevance_scoring_enabled: false,
            strategy_selection_enabled: false,
            entity_enhancement_enabled: false,
            // Model selection - TIER 1
            routing_model: Some("lfm2-350m".to_string()),
            validation_model: Some("lfm2-350m".to_string()),
            complexity_model: Some("lfm2-350m".to_string()),
            // Model selection - TIER 2
            summarization_model: Some("lfm2-1.2b".to_string()),
            retrieval_model: Some("lfm2-350m".to_string()),
            relevance_model: Some("lfm2-350m".to_string()),
            strategy_model: Some("lfm2-1.2b".to_string()),
            entity_model: Some("lfm2-350m".to_string()),
            log_inference: true,
        }
    }
}

impl LocalInferenceConfig {
    /// Create a config with all TIER 1 features enabled
    pub fn tier1_enabled() -> Self {
        Self {
            routing_enabled: true,
            validation_enabled: true,
            complexity_enabled: true,
            ..Default::default()
        }
    }

    /// Create a config with all TIER 2 features enabled
    pub fn tier2_enabled() -> Self {
        Self {
            summarization_enabled: true,
            retrieval_gating_enabled: true,
            relevance_scoring_enabled: true,
            strategy_selection_enabled: true,
            entity_enhancement_enabled: true,
            ..Default::default()
        }
    }

    /// Create a config with all features enabled
    pub fn all_enabled() -> Self {
        Self {
            routing_enabled: true,
            validation_enabled: true,
            complexity_enabled: true,
            summarization_enabled: true,
            retrieval_gating_enabled: true,
            relevance_scoring_enabled: true,
            strategy_selection_enabled: true,
            entity_enhancement_enabled: true,
            ..Default::default()
        }
    }

    /// Create a config with only routing enabled
    pub fn routing_only() -> Self {
        Self {
            routing_enabled: true,
            ..Default::default()
        }
    }

    /// Create a config with only validation enabled
    pub fn validation_only() -> Self {
        Self {
            validation_enabled: true,
            ..Default::default()
        }
    }

    /// Create a config with only summarization enabled
    pub fn summarization_only() -> Self {
        Self {
            summarization_enabled: true,
            ..Default::default()
        }
    }
}

/// Log a local inference event.
pub fn log_inference(task: &str, model: &str, latency_ms: u64, success: bool) {
    if success {
        info!(
            target: "local_llm",
            task = task,
            model = model,
            latency_ms = latency_ms,
            "Local inference completed"
        );
    } else {
        warn!(
            target: "local_llm",
            task = task,
            model = model,
            latency_ms = latency_ms,
            "Local inference failed, falling back to pattern-based"
        );
    }
}

/// Measure inference latency.
pub struct InferenceTimer {
    start: Instant,
    task: String,
    model: String,
}

impl InferenceTimer {
    /// Create a new inference timer for the given task and model.
    pub fn new(task: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            task: task.into(),
            model: model.into(),
        }
    }

    /// Stop the timer and log the inference event.
    pub fn finish(self, success: bool) {
        let latency_ms = self.start.elapsed().as_millis() as u64;
        log_inference(&self.task, &self.model, latency_ms, success);
    }

    /// Return the elapsed time in milliseconds since the timer was created.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LocalInferenceConfig::default();
        assert!(!config.routing_enabled);
        assert!(!config.validation_enabled);
        assert!(!config.complexity_enabled);
        assert!(!config.summarization_enabled);
        assert!(!config.retrieval_gating_enabled);
        assert!(!config.relevance_scoring_enabled);
    }

    #[test]
    fn test_config_tier1_enabled() {
        let config = LocalInferenceConfig::tier1_enabled();
        assert!(config.routing_enabled);
        assert!(config.validation_enabled);
        assert!(config.complexity_enabled);
        assert!(!config.summarization_enabled);
    }

    #[test]
    fn test_config_tier2_enabled() {
        let config = LocalInferenceConfig::tier2_enabled();
        assert!(!config.routing_enabled);
        assert!(config.summarization_enabled);
        assert!(config.retrieval_gating_enabled);
        assert!(config.relevance_scoring_enabled);
        assert!(config.strategy_selection_enabled);
        assert!(config.entity_enhancement_enabled);
    }

    #[test]
    fn test_config_all_enabled() {
        let config = LocalInferenceConfig::all_enabled();
        assert!(config.routing_enabled);
        assert!(config.validation_enabled);
        assert!(config.complexity_enabled);
        assert!(config.summarization_enabled);
        assert!(config.retrieval_gating_enabled);
        assert!(config.relevance_scoring_enabled);
        assert!(config.strategy_selection_enabled);
        assert!(config.entity_enhancement_enabled);
    }

    #[test]
    fn test_config_summarization_only() {
        let config = LocalInferenceConfig::summarization_only();
        assert!(!config.routing_enabled);
        assert!(config.summarization_enabled);
        assert_eq!(config.summarization_model, Some("lfm2-1.2b".to_string()));
    }

    #[test]
    fn test_inference_timer() {
        let timer = InferenceTimer::new("test_task", "test_model");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }
}
