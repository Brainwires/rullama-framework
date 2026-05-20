//! SEAL (Self-Evolving Agentic Learning) Integration Module
//!
//! This module implements techniques from the SEAL paper to enhance
//! conversational question answering, semantic parsing, and self-evolving
//! agent capabilities in brainwires-cli.
//!
//! ## Components
//!
//! - **Coreference Resolution**: Resolves pronouns and elliptical references
//!   ("it", "the file", "that function") to concrete entities from dialog history.
//!
//! - **Query Core Extraction**: Extracts structured S-expression-like query cores
//!   from natural language for graph traversal and code understanding.
//!
//! - **Self-Evolving Learning**: Learns from successful interactions without
//!   retraining, building a library of effective patterns.
//!
//! - **Reflection Module**: Post-execution analysis and error correction
//!   to improve response quality.
//!
//! ## Architecture
//!
//! ```text
//! User Input
//!     │
//!     ▼
//! ┌─────────────────────────┐
//! │ Coreference Resolution  │──► Resolves "it", "the file", etc.
//! └────────────┬────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────┐
//! │ Query Core Extraction   │──► Creates structured query
//! └────────────┬────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────┐
//! │ Learning Coordinator    │──► Checks for learned patterns
//! └────────────┬────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────┐
//! │   Query Execution       │──► Runs against RelationshipGraph
//! └────────────┬────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────┐
//! │   Reflection Module     │──► Validates and corrects
//! └─────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use brainwires_seal::{SealProcessor, SealConfig};
//!
//! let config = SealConfig::default();
//! let mut processor = SealProcessor::new(config);
//!
//! // Process a user message with SEAL enhancements
//! let result = processor.process(
//!     "What uses it?",  // User's query with unresolved reference
//!     &dialog_state,
//!     &entity_store,
//!     Some(&relationship_graph),
//! )?;
//!
//! // result.resolved_query might be: "What uses [main.rs]?"
//! // result.query_core might be: QueryCore { op: DependsOn, ... }
//! ```

pub mod coreference;
#[cfg(feature = "feedback")]
pub mod feedback_bridge;
#[cfg(feature = "knowledge")]
pub mod knowledge_integration;
pub mod learning;
/// LanceDB-backed pattern store for SEAL self-evolving learning.
///
/// Stores `QueryPattern` rows + their embeddings; consumed by the learning
/// loop to find similar past patterns. Moved here from
/// `extras/brainwires-cli/src/storage/pattern_store.rs` because the
/// `QueryPattern` types it operates on live in this module.
pub mod pattern_store;
pub mod query_core;
pub mod reflection;

pub use coreference::{
    CoreferenceResolver, DialogState, ReferenceType, ResolvedReference, SalienceScore,
    UnresolvedReference,
};
#[cfg(feature = "feedback")]
pub use feedback_bridge::{FeedbackBridge, FeedbackProcessingStats};
#[cfg(feature = "knowledge")]
pub use knowledge_integration::{
    EntityResolutionStrategy, IntegrationConfig, SealKnowledgeCoordinator,
};
pub use learning::{
    GlobalMemory, LearningCoordinator, LocalMemory, PatternHint, QueryPattern, TrackedEntity,
};
pub use query_core::{
    FilterPredicate, QueryCore, QueryCoreExtractor, QueryExpr, QueryOp, QueryResult, QuestionType,
    RelationType, SuperlativeDir,
};
pub use reflection::{
    CorrectionRecord, ErrorType, Issue, ReflectionConfig, ReflectionModule, ReflectionReport,
    Severity, SuggestedFix,
};

use anyhow::Result;
use brainwires_core::graph::{EntityStoreT, RelationshipGraphT};

/// Configuration for the SEAL processor
#[derive(Debug, Clone)]
pub struct SealConfig {
    /// Enable coreference resolution
    pub enable_coreference: bool,
    /// Enable query core extraction
    pub enable_query_cores: bool,
    /// Enable self-evolving learning
    pub enable_learning: bool,
    /// Enable reflection module
    pub enable_reflection: bool,
    /// Maximum retry attempts for reflection correction
    pub max_reflection_retries: u32,
    /// Minimum confidence score for coreference resolution
    pub min_coreference_confidence: f32,
    /// Minimum pattern reliability for learning
    pub min_pattern_reliability: f32,
}

impl Default for SealConfig {
    fn default() -> Self {
        Self {
            enable_coreference: true,
            enable_query_cores: true,
            enable_learning: true,
            enable_reflection: true,
            max_reflection_retries: 2,
            min_coreference_confidence: 0.5,
            min_pattern_reliability: 0.7,
        }
    }
}

/// Result of SEAL processing pipeline
#[derive(Debug, Clone)]
pub struct SealProcessingResult {
    /// Original user query
    pub original_query: String,
    /// Query with resolved coreferences
    pub resolved_query: String,
    /// Extracted query core (if any)
    pub query_core: Option<QueryCore>,
    /// Matched learning pattern (if any)
    pub matched_pattern: Option<String>,
    /// Coreference resolutions made
    pub resolutions: Vec<ResolvedReference>,
    /// Quality score from reflection (0.0-1.0)
    pub quality_score: f32,
    /// Any issues detected by reflection
    pub issues: Vec<Issue>,
}

impl SealProcessingResult {
    /// Create a new SealProcessingResult with the given quality score and resolved query
    pub fn new(quality_score: f32, resolved_query: String) -> Self {
        Self {
            original_query: resolved_query.clone(),
            resolved_query,
            query_core: None,
            matched_pattern: None,
            resolutions: Vec::new(),
            quality_score,
            issues: Vec::new(),
        }
    }
}

/// Main SEAL processor that orchestrates all components
pub struct SealProcessor {
    config: SealConfig,
    coreference_resolver: CoreferenceResolver,
    query_extractor: QueryCoreExtractor,
    learning_coordinator: LearningCoordinator,
    reflection_module: ReflectionModule,
}

impl SealProcessor {
    /// Create a new SEAL processor with the given configuration
    pub fn new(config: SealConfig) -> Self {
        Self {
            coreference_resolver: CoreferenceResolver::new(),
            query_extractor: QueryCoreExtractor::new(),
            learning_coordinator: LearningCoordinator::new(String::new()),
            reflection_module: ReflectionModule::new(ReflectionConfig::default()),
            config,
        }
    }

    /// Create a new SEAL processor with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SealConfig::default())
    }

    /// Initialize the learning coordinator with a conversation ID
    pub fn init_conversation(&mut self, conversation_id: &str) {
        self.learning_coordinator = LearningCoordinator::new(conversation_id.to_string());
    }

    /// Process a user query through the SEAL pipeline
    pub fn process(
        &mut self,
        query: &str,
        dialog_state: &DialogState,
        entity_store: &dyn EntityStoreT,
        graph: Option<&dyn RelationshipGraphT>,
    ) -> Result<SealProcessingResult> {
        let mut result = SealProcessingResult {
            original_query: query.to_string(),
            resolved_query: query.to_string(),
            query_core: None,
            matched_pattern: None,
            resolutions: Vec::new(),
            quality_score: 1.0,
            issues: Vec::new(),
        };

        // Step 1: Coreference Resolution
        if self.config.enable_coreference {
            let references = self.coreference_resolver.detect_references(query);
            if !references.is_empty() {
                let resolutions = self.coreference_resolver.resolve(
                    &references,
                    dialog_state,
                    entity_store,
                    graph,
                );

                // Filter by confidence threshold
                let confident_resolutions: Vec<_> = resolutions
                    .into_iter()
                    .filter(|r| r.confidence >= self.config.min_coreference_confidence)
                    .collect();

                if !confident_resolutions.is_empty() {
                    result.resolved_query = self
                        .coreference_resolver
                        .rewrite_with_resolutions(query, &confident_resolutions);
                    result.resolutions = confident_resolutions;
                }
            }
        }

        // Step 2: Query Core Extraction
        if self.config.enable_query_cores {
            // Get entities from entity store for extraction
            let entities: Vec<_> = entity_store.top_entity_info(50);

            result.query_core = self
                .query_extractor
                .extract(&result.resolved_query, &entities);

            // If coreference resolution changed the query, track both versions
            if let Some(ref mut core) = result.query_core
                && result.resolved_query != query
            {
                core.resolved = Some(result.resolved_query.clone());
            }
        }

        // Step 3: Check Learning Coordinator for patterns
        if self.config.enable_learning
            && let Some(pattern) = self.learning_coordinator.process_query(
                query,
                &result.resolved_query,
                result.query_core.clone(),
                dialog_state.current_turn,
            )
        {
            result.matched_pattern = Some(pattern.id.clone());
        }

        // Step 4: Reflection analysis (if we have a query core)
        if self.config.enable_reflection && result.query_core.is_some() {
            // Reflection will be applied after execution in a follow-up call
            // For now, just validate the query core structure
            if let Some(ref core) = result.query_core {
                result.issues = self.reflection_module.validate_query_core(core);
                result.quality_score = if result.issues.is_empty() {
                    1.0
                } else {
                    0.8 - (result.issues.len() as f32 * 0.1).min(0.5)
                };
            }
        }

        Ok(result)
    }

    /// Record the outcome of a query execution for learning
    pub fn record_outcome(
        &mut self,
        pattern_id: Option<&str>,
        success: bool,
        result_count: usize,
        query_core: Option<&QueryCore>,
        execution_time_ms: u64,
    ) {
        if self.config.enable_learning {
            self.learning_coordinator.record_outcome(
                pattern_id,
                success,
                result_count,
                query_core,
                execution_time_ms,
            );
        }
    }

    /// Analyze execution result with reflection module
    pub fn reflect(
        &mut self,
        query_core: &QueryCore,
        result: &QueryResult,
        graph: &dyn RelationshipGraphT,
    ) -> ReflectionReport {
        self.reflection_module.analyze(query_core, result, graph)
    }

    /// Get learning context for prompt injection
    pub fn get_learning_context(&self) -> String {
        self.learning_coordinator.get_context_for_prompt()
    }

    /// Get access to the coreference resolver
    pub fn coreference(&self) -> &CoreferenceResolver {
        &self.coreference_resolver
    }

    /// Get access to the query extractor
    pub fn query_extractor(&self) -> &QueryCoreExtractor {
        &self.query_extractor
    }

    /// Get mutable access to the learning coordinator
    pub fn learning_mut(&mut self) -> &mut LearningCoordinator {
        &mut self.learning_coordinator
    }

    /// Get access to the reflection module
    pub fn reflection(&self) -> &ReflectionModule {
        &self.reflection_module
    }

    /// Get the current configuration
    pub fn config(&self) -> &SealConfig {
        &self.config
    }

    /// Record MDAP execution metrics for learning
    ///
    /// This method records MDAP execution patterns to help improve
    /// future task decomposition and voting strategies.
    #[cfg(feature = "mdap")]
    pub fn record_mdap_metrics(&mut self, metrics: &brainwires_mdap::MdapMetrics) {
        if !self.config.enable_learning {
            return;
        }

        // Record overall success/failure pattern
        self.learning_coordinator.record_outcome(
            None,
            metrics.final_success,
            metrics.completed_steps as usize,
            None,
            0, // MDAP metrics don't track individual query timing
        );

        // Could add more sophisticated learning here:
        // - Track which decomposition strategies work best
        // - Learn red-flag patterns that are too strict/relaxed
        // - Optimize k values based on historical data
        // For now, we just record the basic outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seal_processor_creation() {
        let processor = SealProcessor::with_defaults();
        assert!(processor.config.enable_coreference);
        assert!(processor.config.enable_query_cores);
        assert!(processor.config.enable_learning);
        assert!(processor.config.enable_reflection);
    }

    #[test]
    fn test_seal_config_default() {
        let config = SealConfig::default();
        assert!(config.enable_coreference);
        assert_eq!(config.max_reflection_retries, 2);
        assert!(config.min_coreference_confidence > 0.0);
    }

    #[test]
    fn test_init_conversation() {
        let mut processor = SealProcessor::with_defaults();
        processor.init_conversation("test-conv-123");
        // Verify the conversation ID is set
        assert_eq!(
            processor.learning_coordinator.local.conversation_id,
            "test-conv-123"
        );
    }
}
