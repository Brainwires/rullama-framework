//! SEAL + Knowledge System Integration
//!
//! This module bridges SEAL's entity-centric learning with the Knowledge System's
//! behavioral truths (BKS) and personal facts (PKS), enabling bidirectional learning
//! and context-aware entity resolution.
//!
//! ## Key Integration Points
//!
//! 1. **SEAL → PKS**: Entity resolutions trigger lookups of personal facts
//! 2. **SEAL → BKS**: Query context triggers lookups of behavioral truths
//! 3. **SEAL → BKS Promotion**: High-reliability patterns promoted to shared knowledge
//! 4. **BKS → SEAL**: Behavioral truths loaded into SEAL's global memory
//! 5. **Quality-Aware Context**: SEAL quality scores adjust retrieval thresholds
//!
//! ## Architecture
//!
//! ```text
//! User Input
//!     │
//!     ▼
//! ┌────────────────────────────────────────────┐
//! │   SealKnowledgeCoordinator                 │
//! │                                            │
//! │   ┌─────────────────────────────────────┐ │
//! │   │ SEAL Preprocessing                   │ │
//! │   │ • Coreference: "it" → "main.rs"     │ │
//! │   │ • Query extraction: S-expressions   │ │
//! │   │ • Quality score: 0.0-1.0            │ │
//! │   └──────────┬──────────────────────────┘ │
//! │              │                             │
//! │              ▼                             │
//! │   ┌─────────────────────────────────────┐ │
//! │   │ Knowledge Lookup (PARALLEL)         │ │
//! │   │ • PKS: entity facts (main.rs)       │ │
//! │   │ • BKS: behavioral truths (rust)     │ │
//! │   └──────────┬──────────────────────────┘ │
//! │              │                             │
//! │              ▼                             │
//! │   ┌─────────────────────────────────────┐ │
//! │   │ Confidence Harmonization            │ │
//! │   │ • Combine: SEAL + BKS + PKS         │ │
//! │   │ • Adjust thresholds by quality      │ │
//! │   └──────────┬──────────────────────────┘ │
//! └──────────────┼──────────────────────────────┘
//!                │
//!                ▼
//!     Enhanced Context → OrchestratorAgent
//! ```

use super::{QueryPattern, ResolvedReference, SealProcessingResult};
use anyhow::Result;
use brainwires_knowledge::knowledge::bks_pks::{
    BehavioralKnowledgeCache, BehavioralTruth, PersonalKnowledgeCache, TruthCategory, TruthSource,
};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Default minimum pattern reliability for BKS promotion (80% success rate).
const DEFAULT_PATTERN_PROMOTION_THRESHOLD: f32 = 0.8;

/// Configuration for SEAL + Knowledge integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Master toggle for the integration
    pub enabled: bool,

    /// Enable SEAL patterns → BKS promotion
    pub seal_to_knowledge: bool,

    /// Enable BKS truths → SEAL pattern loading
    pub knowledge_to_seal: bool,

    /// Minimum SEAL quality score to boost with BKS context (0.0-1.0)
    /// Default: 0.7 (only high-quality SEAL results get BKS boost)
    pub min_seal_quality_for_bks_boost: f32,

    /// Minimum SEAL quality score to boost with PKS context (0.0-1.0)
    /// Default: 0.5 (medium-quality SEAL results get PKS boost)
    pub min_seal_quality_for_pks_boost: f32,

    /// Minimum pattern reliability for BKS promotion (0.0-1.0)
    /// Default: 0.8 (80% success rate required)
    pub pattern_promotion_threshold: f32,

    /// Minimum pattern uses before considering promotion
    /// Default: 5 (need statistical significance)
    pub min_pattern_uses: u32,

    /// Cache BKS truths in SEAL's global memory
    pub cache_bks_in_seal: bool,

    /// Entity resolution strategy
    pub entity_resolution_strategy: EntityResolutionStrategy,

    /// Weight for SEAL quality in confidence harmonization (0.0-1.0)
    /// Default: 0.5 (50% weight)
    pub seal_weight: f32,

    /// Weight for BKS confidence in harmonization (0.0-1.0)
    /// Default: 0.3 (30% weight)
    pub bks_weight: f32,

    /// Weight for PKS confidence in harmonization (0.0-1.0)
    /// Default: 0.2 (20% weight)
    pub pks_weight: f32,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            seal_to_knowledge: true,
            knowledge_to_seal: true,
            min_seal_quality_for_bks_boost: 0.7,
            min_seal_quality_for_pks_boost: 0.5,
            pattern_promotion_threshold: DEFAULT_PATTERN_PROMOTION_THRESHOLD,
            min_pattern_uses: 5,
            cache_bks_in_seal: true,
            entity_resolution_strategy: EntityResolutionStrategy::Hybrid {
                seal_weight: 0.6,
                pks_weight: 0.4,
            },
            seal_weight: 0.5,
            bks_weight: 0.3,
            pks_weight: 0.2,
        }
    }
}

impl IntegrationConfig {
    /// Create config with all features enabled (recommended)
    pub fn full() -> Self {
        Self::default()
    }

    /// Create config with only SEAL → Knowledge enabled (no BKS → SEAL loading)
    pub fn seal_to_knowledge_only() -> Self {
        Self {
            knowledge_to_seal: false,
            ..Self::default()
        }
    }

    /// Create config with integration disabled
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.min_seal_quality_for_bks_boost < 0.0 || self.min_seal_quality_for_bks_boost > 1.0 {
            anyhow::bail!("min_seal_quality_for_bks_boost must be between 0.0 and 1.0");
        }

        if self.pattern_promotion_threshold < 0.0 || self.pattern_promotion_threshold > 1.0 {
            anyhow::bail!("pattern_promotion_threshold must be between 0.0 and 1.0");
        }

        let weight_sum = self.seal_weight + self.bks_weight + self.pks_weight;
        if (weight_sum - 1.0).abs() > 0.01 {
            anyhow::bail!(
                "Confidence weights must sum to 1.0 (got: {:.2})",
                weight_sum
            );
        }

        Ok(())
    }
}

/// Strategy for resolving entity references when both SEAL and PKS have candidates
#[derive(Debug, Clone)]
pub enum EntityResolutionStrategy {
    /// Always prefer SEAL's resolution
    SealFirst,

    /// Always prefer PKS context-based resolution
    PksContextFirst,

    /// Weighted combination of SEAL and PKS confidence
    Hybrid {
        /// Weight given to SEAL confidence scores.
        seal_weight: f32,
        /// Weight given to PKS confidence scores.
        pks_weight: f32,
    },
}

/// Bridges SEAL's entity-centric learning with Knowledge System's behavioral truths
pub struct SealKnowledgeCoordinator {
    /// BKS cache for behavioral truths
    bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>,

    /// PKS cache for personal facts
    pks_cache: Arc<Mutex<PersonalKnowledgeCache>>,

    /// Integration settings
    config: IntegrationConfig,
}

impl SealKnowledgeCoordinator {
    /// Create a new coordinator
    pub fn new(
        bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>,
        pks_cache: Arc<Mutex<PersonalKnowledgeCache>>,
        config: IntegrationConfig,
    ) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            bks_cache,
            pks_cache,
            config,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(
        bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>,
        pks_cache: Arc<Mutex<PersonalKnowledgeCache>>,
    ) -> Result<Self> {
        Self::new(bks_cache, pks_cache, IntegrationConfig::default())
    }

    /// Get PKS context for SEAL entity resolutions
    ///
    /// Given SEAL's entity resolutions, look up relevant personal facts
    /// that provide context about those entities.
    ///
    /// Example:
    /// - SEAL resolves "it" → "main.rs"
    /// - PKS has: "main.rs is entry point for brainwires-cli"
    /// - Returns: Formatted context string with relevant facts
    pub async fn get_pks_context(
        &self,
        seal_result: &SealProcessingResult,
    ) -> Result<Option<String>> {
        if !self.config.enabled {
            return Ok(None);
        }

        // Check if SEAL quality is high enough for PKS boost
        if seal_result.quality_score < self.config.min_seal_quality_for_pks_boost {
            return Ok(None);
        }

        // Extract entities from SEAL resolutions
        let entities: Vec<&str> = seal_result
            .resolutions
            .iter()
            .map(|r| r.antecedent.as_str())
            .collect();

        if entities.is_empty() {
            return Ok(None);
        }

        // Look up facts for each entity
        let pks = self.pks_cache.lock().await;
        let mut context_parts = Vec::new();

        for entity in entities {
            // Get facts related to this entity by looking for keys containing the entity name
            // This is a simple heuristic - could be improved with fuzzy matching
            let all_facts: Vec<_> = pks
                .get_all_facts()
                .into_iter()
                .filter(|f| !f.deleted && (f.key.contains(entity) || f.value.contains(entity)))
                .collect();

            if !all_facts.is_empty() {
                context_parts.push(format!("\n**{}:**", entity));

                for fact in all_facts {
                    // Filter by confidence (only include reliable facts)
                    if fact.confidence >= 0.5 {
                        context_parts.push(format!(
                            "  - {} (confidence: {:.2})",
                            fact.value, fact.confidence
                        ));
                    }
                }
            }
        }

        if context_parts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(format!(
                "# PERSONAL CONTEXT\n\nRelevant facts about entities mentioned:\n{}",
                context_parts.join("\n")
            )))
        }
    }

    /// Get BKS context for query
    ///
    /// Given the user's query (after SEAL processing), look up relevant
    /// behavioral truths that might help with execution.
    ///
    /// Example:
    /// - Query: "How do I run the Rust project?"
    /// - BKS has: "For Rust projects, use 'cargo run' not 'rustc main.rs'"
    /// - Returns: Formatted context string with relevant truths
    pub async fn get_bks_context(&self, query: &str) -> Result<Option<String>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let bks = self.bks_cache.lock().await;

        // Get truths matching the query context with scores
        let truths = bks.get_matching_truths_with_scores(query, 0.5, 5)?;

        if truths.is_empty() {
            return Ok(None);
        }

        let mut context_parts = vec!["# BEHAVIORAL KNOWLEDGE\n".to_string()];
        context_parts.push("Learned patterns that may be relevant:\n".to_string());

        for (truth, score) in truths {
            context_parts.push(format!(
                "\n**{}** (confidence: {:.2}, relevance: {:.2}):",
                truth.context_pattern, truth.confidence, score
            ));
            context_parts.push(format!("  Rule: {}", truth.rule));
            context_parts.push(format!("  Why: {}", truth.rationale));
        }

        Ok(Some(context_parts.join("\n")))
    }

    /// Harmonize confidence from multiple sources
    ///
    /// Combines SEAL quality score with BKS and PKS confidence values
    /// using weighted averaging to produce a unified confidence score.
    pub fn harmonize_confidence(
        &self,
        seal_quality: f32,
        bks_confidence: Option<f32>,
        pks_confidence: Option<f32>,
    ) -> f32 {
        let mut weighted_sum = seal_quality * self.config.seal_weight;
        let mut total_weight = self.config.seal_weight;

        if let Some(bks) = bks_confidence {
            weighted_sum += bks * self.config.bks_weight;
            total_weight += self.config.bks_weight;
        }

        if let Some(pks) = pks_confidence {
            weighted_sum += pks * self.config.pks_weight;
            total_weight += self.config.pks_weight;
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            (weighted_sum / total_weight).min(1.0)
        } else {
            seal_quality
        }
    }

    /// Adjust retrieval threshold based on SEAL quality
    ///
    /// When SEAL quality is low, we need more context to compensate,
    /// so we lower the retrieval threshold to include more historical messages.
    ///
    /// When SEAL quality is high, we can be more selective with context.
    pub fn adjust_retrieval_threshold(&self, base_threshold: f32, seal_quality: f32) -> f32 {
        // Quality adjustment factor: lower quality → lower threshold (more context)
        // Formula: adjusted = base * (0.7 + 0.3 * quality)
        // - quality = 0.0 → 70% of base threshold
        // - quality = 1.0 → 100% of base threshold
        base_threshold * (0.7 + 0.3 * seal_quality).max(0.5)
    }

    /// Check if a SEAL pattern should be promoted to BKS
    ///
    /// Patterns are promoted when they:
    /// 1. Have reliability above threshold (default: 0.8)
    /// 2. Have been used enough times (default: 5)
    /// 3. Integration is enabled
    pub async fn check_and_promote_pattern(
        &mut self,
        pattern: &QueryPattern,
        execution_context: &str,
    ) -> Result<Option<BehavioralTruth>> {
        if !self.config.enabled || !self.config.seal_to_knowledge {
            return Ok(None);
        }

        // Check promotion criteria
        if pattern.reliability() < self.config.pattern_promotion_threshold {
            tracing::debug!(
                "Pattern '{}' reliability ({:.2}) below threshold ({:.2})",
                pattern.template,
                pattern.reliability(),
                self.config.pattern_promotion_threshold
            );
            return Ok(None);
        }

        let total_uses = pattern.success_count + pattern.failure_count;
        if total_uses < self.config.min_pattern_uses {
            tracing::debug!(
                "Pattern '{}' uses ({}) below minimum ({})",
                pattern.template,
                total_uses,
                self.config.min_pattern_uses
            );
            return Ok(None);
        }

        // Create BKS truth from SEAL pattern
        let category = self.infer_category(&pattern.question_type);
        let rule = self.generalize_pattern_to_rule(pattern);
        let rationale = format!(
            "Learned from {} successful executions with {:.1}% reliability (SEAL pattern)",
            pattern.success_count,
            pattern.reliability() * 100.0
        );

        let truth = BehavioralTruth::new(
            category,
            execution_context.to_string(),
            rule,
            rationale,
            TruthSource::SuccessPattern, // Pattern emerged from successful usage
            None,                        // Anonymous
        );

        // Submit to BKS
        let mut bks = self.bks_cache.lock().await;
        match bks.queue_submission(truth.clone()) {
            Ok(_) => {
                tracing::info!(
                    "✓ Promoted SEAL pattern to BKS: '{}' (reliability: {:.2}, uses: {})",
                    pattern.template,
                    pattern.reliability(),
                    total_uses
                );
                Ok(Some(truth))
            }
            Err(e) => {
                tracing::warn!("Failed to promote SEAL pattern to BKS: {}", e);
                Err(e)
            }
        }
    }

    /// Load relevant BKS truths into SEAL's global memory on startup
    ///
    /// This enables SEAL to benefit from collective learning by having
    /// high-confidence behavioral truths available for pattern matching.
    pub async fn sync_bks_to_seal(
        &mut self,
        seal_learning: &mut super::learning::LearningCoordinator,
    ) -> Result<u32> {
        if !self.config.enabled || !self.config.knowledge_to_seal {
            return Ok(0);
        }

        if !self.config.cache_bks_in_seal {
            return Ok(0);
        }

        let bks = self.bks_cache.lock().await;

        // Get high-confidence truths (> 0.7) from last 30 days
        let truths = bks.get_reliable_truths(0.7, 30);

        let mut loaded = 0;
        for truth in truths {
            // Convert BKS truth to structured SEAL pattern hint and store it
            if let Some(hint) = self.truth_to_pattern_hint(truth) {
                seal_learning.global.add_pattern_hint(hint);
                tracing::debug!(
                    "Loaded BKS truth into SEAL: {} -> {}",
                    truth.context_pattern,
                    truth.rule
                );
                loaded += 1;
            }
        }

        tracing::info!("Loaded {} BKS truths into SEAL global memory", loaded);
        Ok(loaded)
    }

    /// Observe SEAL resolutions for PKS learning
    ///
    /// Tracks which entities the user focuses on, allowing PKS to build
    /// a profile of user interests and recently-used entities.
    pub async fn observe_seal_resolutions(
        &mut self,
        resolutions: &[ResolvedReference],
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut pks = self.pks_cache.lock().await;

        for resolution in resolutions {
            // Track entity as context fact
            let key = format!("recent_entity:{}", resolution.antecedent);

            // Upsert fact (will update timestamp if exists)
            pks.upsert_fact_simple(
                &key,
                &resolution.antecedent,
                resolution.confidence,
                true, // local_only (don't sync entity tracking)
            )?;
        }

        Ok(())
    }

    /// Record a tool failure pattern for shared learning
    ///
    /// When validation fails or tools error, record the pattern in BKS
    /// so other users can benefit from this knowledge.
    pub async fn record_tool_failure(
        &mut self,
        tool_name: &str,
        error_message: &str,
        context: &str,
    ) -> Result<()> {
        if !self.config.enabled || !self.config.seal_to_knowledge {
            return Ok(());
        }

        // Create behavioral truth about tool failure pattern
        let truth = BehavioralTruth::new(
            TruthCategory::ErrorRecovery,
            context.to_string(),
            format!(
                "Tool '{}' commonly fails with: {}",
                tool_name, error_message
            ),
            "Observed from validation failures".to_string(),
            TruthSource::FailurePattern,
            None,
        );

        let mut bks = self.bks_cache.lock().await;
        bks.queue_submission(truth)?;

        tracing::debug!(
            "Recorded tool failure pattern: {} in context: {}",
            tool_name,
            context
        );

        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }

    /// Get reference to PKS cache
    pub fn get_pks_cache(&self) -> Arc<Mutex<PersonalKnowledgeCache>> {
        Arc::clone(&self.pks_cache)
    }

    /// Get reference to BKS cache
    pub fn get_bks_cache(&self) -> Arc<Mutex<BehavioralKnowledgeCache>> {
        Arc::clone(&self.bks_cache)
    }

    // --- Private helper methods ---

    /// Infer BKS category from SEAL question type
    fn infer_category(&self, question_type: &super::query_core::QuestionType) -> TruthCategory {
        use super::query_core::QuestionType;

        match question_type {
            QuestionType::Definition => TruthCategory::CommandUsage,
            QuestionType::Dependency => TruthCategory::TaskStrategy,
            QuestionType::Location => TruthCategory::TaskStrategy,
            QuestionType::Count | QuestionType::Superlative => TruthCategory::TaskStrategy,
            _ => TruthCategory::TaskStrategy,
        }
    }

    /// Generalize a SEAL pattern template into a BKS rule
    fn generalize_pattern_to_rule(&self, pattern: &QueryPattern) -> String {
        // Note: QueryPattern has required_types, not entity_types
        let types_str = pattern
            .required_types
            .iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "For '{:?}' queries about {}, use pattern: {}",
            pattern.question_type, types_str, pattern.template
        )
    }

    /// Convert BKS truth to a structured SEAL pattern hint.
    ///
    /// This is a best-effort conversion since BKS and SEAL use different schemas.
    /// Returns None if the truth doesn't map to a SEAL pattern.
    fn truth_to_pattern_hint(
        &self,
        truth: &BehavioralTruth,
    ) -> Option<super::learning::PatternHint> {
        Some(super::learning::PatternHint {
            context_pattern: truth.context_pattern.clone(),
            rule: truth.rule.clone(),
            confidence: truth.confidence as f64,
            source: "bks".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_validation() {
        let mut config = IntegrationConfig::default();
        assert!(config.validate().is_ok());

        // Invalid quality threshold
        config.min_seal_quality_for_bks_boost = 1.5;
        assert!(config.validate().is_err());

        // Invalid weight sum
        config = IntegrationConfig::default();
        config.seal_weight = 0.5;
        config.bks_weight = 0.5;
        config.pks_weight = 0.5; // Sum > 1.0
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_confidence_harmonization() {
        let coordinator = create_test_coordinator();

        // Only SEAL
        let conf = coordinator.harmonize_confidence(0.8, None, None);
        assert!((conf - 0.8).abs() < 0.01);

        // SEAL + BKS + PKS
        let conf = coordinator.harmonize_confidence(0.6, Some(0.9), Some(0.8));
        // Expected: 0.6*0.5 + 0.9*0.3 + 0.8*0.2 = 0.3 + 0.27 + 0.16 = 0.73
        assert!((conf - 0.73).abs() < 0.01);
    }

    #[test]
    fn test_retrieval_threshold_adjustment() {
        let coordinator = create_test_coordinator();

        // Low quality → lower threshold (need more context)
        let adjusted = coordinator.adjust_retrieval_threshold(0.75, 0.0);
        assert!((adjusted - 0.525).abs() < 0.01); // 0.75 * 0.7

        // High quality → higher threshold (can be selective)
        let adjusted = coordinator.adjust_retrieval_threshold(0.75, 1.0);
        assert!((adjusted - 0.75).abs() < 0.01); // 0.75 * 1.0

        // Medium quality
        let adjusted = coordinator.adjust_retrieval_threshold(0.75, 0.5);
        assert!((adjusted - 0.6375).abs() < 0.01); // 0.75 * 0.85
    }

    fn create_test_coordinator() -> SealKnowledgeCoordinator {
        let bks_cache = Arc::new(Mutex::new(
            BehavioralKnowledgeCache::in_memory(100).unwrap(),
        ));
        let pks_cache = Arc::new(Mutex::new(PersonalKnowledgeCache::in_memory(100).unwrap()));

        SealKnowledgeCoordinator::new(bks_cache, pks_cache, IntegrationConfig::default()).unwrap()
    }
}
