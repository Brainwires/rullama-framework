//! Integration tests for SEAL + Knowledge System Integration
//!
//! Tests the bidirectional learning between SEAL (entity-centric learning)
//! and the Knowledge System (BKS + PKS).

use brainwires::knowledge::EntityType;
use brainwires::knowledge::bks_pks::{BehavioralKnowledgeCache, PersonalKnowledgeCache};
use brainwires_seal::{
    EntityResolutionStrategy, IntegrationConfig, ReferenceType, ResolvedReference, SalienceScore,
    SealKnowledgeCoordinator, SealProcessingResult, UnresolvedReference,
};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Create test coordinator with in-memory caches
fn create_test_coordinator() -> SealKnowledgeCoordinator {
    let bks_cache = Arc::new(Mutex::new(
        BehavioralKnowledgeCache::in_memory(100).unwrap(),
    ));
    let pks_cache = Arc::new(Mutex::new(PersonalKnowledgeCache::in_memory(100).unwrap()));

    SealKnowledgeCoordinator::new(bks_cache, pks_cache, IntegrationConfig::default()).unwrap()
}

#[tokio::test]
async fn test_confidence_harmonization() {
    let coordinator = create_test_coordinator();

    // Only SEAL quality
    let conf = coordinator.harmonize_confidence(0.8, None, None);
    assert!((conf - 0.8).abs() < 0.01);

    // SEAL + BKS + PKS
    let conf = coordinator.harmonize_confidence(0.6, Some(0.9), Some(0.8));
    // Expected: 0.6*0.5 + 0.9*0.3 + 0.8*0.2 = 0.3 + 0.27 + 0.16 = 0.73
    assert!((conf - 0.73).abs() < 0.01);

    // SEAL + BKS only
    let conf = coordinator.harmonize_confidence(0.5, Some(0.9), None);
    // Expected: (0.5*0.5 + 0.9*0.3) / (0.5 + 0.3) = 0.52
    assert!((conf - 0.65).abs() < 0.01);
}

#[tokio::test]
async fn test_retrieval_threshold_adjustment() {
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

#[tokio::test]
async fn test_seal_to_pks_entity_observation() {
    let mut coordinator = create_test_coordinator();

    // Manually create SEAL resolution to test PKS observation
    let resolution = ResolvedReference {
        reference: UnresolvedReference {
            text: "it".to_string(),
            ref_type: ReferenceType::SingularNeutral,
            start: 0,
            end: 2,
        },
        antecedent: "main.rs".to_string(),
        entity_type: EntityType::File,
        confidence: 0.85,
        salience: SalienceScore::default(),
    };

    // Observe resolution in PKS
    coordinator
        .observe_seal_resolutions(&[resolution])
        .await
        .unwrap();

    // Verify PKS tracked the entity
    let pks_cache = coordinator.get_pks_cache();
    let pks = pks_cache.lock().await;
    let facts = pks.get_all_facts();

    // Should have recent_entity fact
    let has_entity_fact = facts
        .iter()
        .any(|f| f.key.starts_with("recent_entity:") && f.local_only);

    assert!(
        has_entity_fact,
        "PKS should track observed entity resolutions"
    );
}

#[tokio::test]
async fn test_quality_aware_threshold() {
    let _config = IntegrationConfig::default();

    // High quality SEAL result should use stricter threshold
    let base_threshold = 0.75;

    // High quality (1.0) → no adjustment needed
    let high_quality = 1.0f32;
    let adjustment = 0.7 + 0.3 * high_quality;
    let threshold = base_threshold * adjustment;
    assert!((threshold - 0.75).abs() < 0.01);

    // Low quality (0.0) → need more context, lower threshold
    let low_quality = 0.0f32;
    let adjustment = 0.7 + 0.3 * low_quality;
    let threshold = base_threshold * adjustment;
    assert!((threshold - 0.525).abs() < 0.01);
}

#[tokio::test]
async fn test_entity_resolution_strategies() {
    // Test SealFirst strategy
    let config = IntegrationConfig {
        entity_resolution_strategy: EntityResolutionStrategy::SealFirst,
        ..Default::default()
    };
    assert!(config.validate().is_ok());

    // Test PksContextFirst strategy
    let config = IntegrationConfig {
        entity_resolution_strategy: EntityResolutionStrategy::PksContextFirst,
        ..Default::default()
    };
    assert!(config.validate().is_ok());

    // Test Hybrid strategy
    let config = IntegrationConfig {
        entity_resolution_strategy: EntityResolutionStrategy::Hybrid {
            seal_weight: 0.6,
            pks_weight: 0.4,
        },
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn test_config_validation() {
    let mut config = IntegrationConfig::default();
    assert!(config.validate().is_ok());

    // Invalid quality threshold (> 1.0)
    config.min_seal_quality_for_bks_boost = 1.5;
    assert!(config.validate().is_err());

    // Invalid weight sum (not equal to 1.0)
    config = IntegrationConfig::default();
    config.seal_weight = 0.5;
    config.bks_weight = 0.5;
    config.pks_weight = 0.5; // Sum = 1.5
    assert!(config.validate().is_err());

    // Valid weight sum
    config.seal_weight = 0.5;
    config.bks_weight = 0.3;
    config.pks_weight = 0.2; // Sum = 1.0
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn test_get_bks_context() {
    let coordinator = create_test_coordinator();

    // Empty BKS should return None
    let context = coordinator
        .get_bks_context("How do I run Rust projects?")
        .await
        .unwrap();

    // Should be None for empty knowledge base
    assert!(context.is_none());
}

#[tokio::test]
async fn test_get_pks_context() {
    let coordinator = create_test_coordinator();

    // Create SEAL result with empty resolutions
    let seal_result = SealProcessingResult {
        original_query: "test query".to_string(),
        resolved_query: "test query".to_string(),
        query_core: None,
        matched_pattern: None,
        resolutions: Vec::new(),
        quality_score: 0.85,
        issues: Vec::new(),
    };

    // Empty PKS and no resolutions should return None
    let context = coordinator.get_pks_context(&seal_result).await.unwrap();
    assert!(context.is_none());
}

#[tokio::test]
async fn test_observe_seal_resolutions_empty() {
    let mut coordinator = create_test_coordinator();

    // Empty resolutions should not error
    coordinator.observe_seal_resolutions(&[]).await.unwrap();

    // Verify PKS is still empty
    let pks_cache = coordinator.get_pks_cache();
    let pks = pks_cache.lock().await;
    let facts = pks.get_all_facts();
    assert_eq!(facts.len(), 0);
}

#[tokio::test]
async fn test_integration_config_defaults() {
    let config = IntegrationConfig::default();

    assert!(config.enabled);
    assert!(config.seal_to_knowledge);
    assert!(config.knowledge_to_seal);
    assert_eq!(config.min_seal_quality_for_bks_boost, 0.7);
    assert_eq!(config.min_seal_quality_for_pks_boost, 0.5);
    assert_eq!(config.pattern_promotion_threshold, 0.8);
    assert_eq!(config.min_pattern_uses, 5);
    assert!(config.cache_bks_in_seal);

    // Weights should sum to 1.0
    let weight_sum = config.seal_weight + config.bks_weight + config.pks_weight;
    assert!((weight_sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_coordinator_creation() {
    let bks_cache = Arc::new(Mutex::new(
        BehavioralKnowledgeCache::in_memory(100).unwrap(),
    ));
    let pks_cache = Arc::new(Mutex::new(PersonalKnowledgeCache::in_memory(100).unwrap()));

    // Should create successfully with valid config
    let coordinator = SealKnowledgeCoordinator::new(
        bks_cache.clone(),
        pks_cache.clone(),
        IntegrationConfig::default(),
    );
    assert!(coordinator.is_ok());

    // Should fail with invalid config
    let invalid_config = IntegrationConfig {
        seal_weight: 2.0, // Invalid weight sum
        ..Default::default()
    };
    let coordinator = SealKnowledgeCoordinator::new(bks_cache, pks_cache, invalid_config);
    assert!(coordinator.is_err());
}

#[tokio::test]
async fn test_with_defaults_constructor() {
    let bks_cache = Arc::new(Mutex::new(
        BehavioralKnowledgeCache::in_memory(100).unwrap(),
    ));
    let pks_cache = Arc::new(Mutex::new(PersonalKnowledgeCache::in_memory(100).unwrap()));

    let coordinator = SealKnowledgeCoordinator::with_defaults(bks_cache, pks_cache);
    assert!(coordinator.is_ok());

    let coordinator = coordinator.unwrap();
    let config = coordinator.config();
    assert!(config.enabled);
}
