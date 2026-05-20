//! Learning & Optimization
//!
//! This module tracks technique effectiveness and learns from outcomes,
//! promoting successful patterns to BKS for collective learning.

use super::techniques::PromptingTechnique;
use anyhow::Result;
use brainwires_knowledge::knowledge::bks_pks::{
    BehavioralKnowledgeCache, BehavioralTruth, TruthCategory, TruthSource,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Default minimum reliability threshold for promoting a technique to BKS.
const DEFAULT_PROMOTION_THRESHOLD: f64 = 0.8;

/// Record of technique effectiveness for a specific task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechniqueEffectivenessRecord {
    /// The prompting technique that was used.
    pub technique: PromptingTechnique,
    /// The cluster this task belongs to.
    pub cluster_id: String,
    /// Description of the task that was executed.
    pub task_description: String,
    /// Whether the task completed successfully.
    pub success: bool,
    /// Number of iterations consumed.
    pub iterations_used: u32,
    /// Quality score from 0.0 to 1.0.
    pub quality_score: f32,
    /// Unix timestamp of the execution.
    pub timestamp: i64,
}

/// Statistics for a technique in a specific cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechniqueStats {
    /// Number of successful executions.
    pub success_count: u32,
    /// Number of failed executions.
    pub failure_count: u32,
    /// Average iterations used across executions.
    pub avg_iterations: f32,
    /// Average quality score across executions.
    pub avg_quality: f32,
    /// Unix timestamp of the last execution.
    pub last_used: i64,
}

impl TechniqueStats {
    /// Create new stats with initial values
    pub fn new() -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            avg_iterations: 0.0,
            avg_quality: 0.0,
            last_used: Utc::now().timestamp(),
        }
    }

    /// Calculate reliability (success rate)
    pub fn reliability(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f32 / total as f32
        }
    }

    /// Total uses
    pub fn total_uses(&self) -> u32 {
        self.success_count + self.failure_count
    }

    /// Update stats with new outcome (using EMA for averages)
    pub fn update(&mut self, success: bool, iterations: u32, quality: f32) {
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }

        let alpha = 0.3; // EMA weight
        self.avg_iterations = alpha * iterations as f32 + (1.0 - alpha) * self.avg_iterations;
        self.avg_quality = alpha * quality + (1.0 - alpha) * self.avg_quality;
        self.last_used = Utc::now().timestamp();
    }
}

impl Default for TechniqueStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordinates learning and promotion of technique effectiveness
pub struct PromptingLearningCoordinator {
    /// Historical records of technique effectiveness
    records: Vec<TechniqueEffectivenessRecord>,

    /// BKS cache for promoting effective techniques
    bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>,

    /// Aggregated statistics per (cluster_id, technique)
    technique_stats: HashMap<(String, PromptingTechnique), TechniqueStats>,

    /// Minimum reliability for BKS promotion (default: 0.8)
    promotion_threshold: f32,

    /// Minimum uses before promotion (default: 5)
    min_uses_for_promotion: u32,
}

impl PromptingLearningCoordinator {
    /// Create a new learning coordinator
    pub fn new(bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>) -> Self {
        Self {
            records: Vec::new(),
            bks_cache,
            technique_stats: HashMap::new(),
            promotion_threshold: DEFAULT_PROMOTION_THRESHOLD as f32,
            min_uses_for_promotion: 5,
        }
    }

    /// Create with custom promotion thresholds
    pub fn with_thresholds(
        bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>,
        promotion_threshold: f32,
        min_uses: u32,
    ) -> Self {
        Self {
            records: Vec::new(),
            bks_cache,
            technique_stats: HashMap::new(),
            promotion_threshold,
            min_uses_for_promotion: min_uses,
        }
    }

    /// Record outcome of using specific techniques
    ///
    /// This is called after task completion to track which techniques worked.
    ///
    /// # Arguments
    /// * `cluster_id` - The cluster that was matched
    /// * `techniques` - The techniques that were used
    /// * `task_description` - Description of the task
    /// * `success` - Whether the task completed successfully
    /// * `iterations` - Number of iterations used
    /// * `quality_score` - Quality score (0.0-1.0)
    pub fn record_outcome(
        &mut self,
        cluster_id: String,
        techniques: Vec<PromptingTechnique>,
        task_description: String,
        success: bool,
        iterations: u32,
        quality_score: f32,
    ) {
        let timestamp = Utc::now().timestamp();

        for technique in techniques {
            // Create record
            let record = TechniqueEffectivenessRecord {
                technique: technique.clone(),
                cluster_id: cluster_id.clone(),
                task_description: task_description.clone(),
                success,
                iterations_used: iterations,
                quality_score,
                timestamp,
            };

            self.records.push(record);

            // Update aggregated stats
            self.update_stats(&cluster_id, &technique, success, iterations, quality_score);
        }
    }

    /// Update aggregated statistics for a technique
    fn update_stats(
        &mut self,
        cluster_id: &str,
        technique: &PromptingTechnique,
        success: bool,
        iterations: u32,
        quality: f32,
    ) {
        let key = (cluster_id.to_string(), technique.clone());
        let stats = self.technique_stats.entry(key).or_default();
        stats.update(success, iterations, quality);
    }

    /// Check if technique should be promoted to BKS
    ///
    /// Promotion criteria (same as SEAL patterns):
    /// - Reliability > threshold (default: 0.8 / 80%)
    /// - Total uses > min_uses (default: 5)
    ///
    /// # Returns
    /// * `true` if technique qualifies for promotion
    pub fn should_promote(&self, cluster_id: &str, technique: &PromptingTechnique) -> bool {
        if let Some(stats) = self
            .technique_stats
            .get(&(cluster_id.to_string(), technique.clone()))
        {
            let reliability = stats.reliability();
            let uses = stats.total_uses();

            reliability >= self.promotion_threshold && uses >= self.min_uses_for_promotion
        } else {
            false
        }
    }

    /// Promote technique to BKS
    ///
    /// Creates a BehavioralTruth with effectiveness information and submits to BKS.
    /// This allows other users to benefit from the learned effectiveness.
    pub async fn promote_technique_to_bks(
        &mut self,
        cluster_id: &str,
        technique: &PromptingTechnique,
    ) -> Result<bool> {
        if !self.should_promote(cluster_id, technique) {
            return Ok(false);
        }

        let stats = self
            .technique_stats
            .get(&(cluster_id.to_string(), technique.clone()))
            .expect("should_promote verified this entry exists");

        let reliability = stats.reliability();
        let uses = stats.total_uses();

        // Create BehavioralTruth
        let truth = BehavioralTruth::new(
            TruthCategory::PromptingTechnique,
            cluster_id.to_string(), // context_pattern
            format!(
                "Use {:?} for {} tasks (achieves {:.1}% success rate)",
                technique,
                cluster_id,
                reliability * 100.0
            ), // rule
            format!(
                "Learned from {} executions with avg quality {:.2}. \
                Average iterations: {:.1}. \
                This technique has proven effective for this task cluster.",
                uses, stats.avg_quality, stats.avg_iterations
            ), // rationale
            TruthSource::SuccessPattern,
            None, // No specific user attribution
        );

        // Submit to BKS
        let mut bks = self.bks_cache.lock().await;
        let queued = bks.queue_submission(truth)?;

        if queued {
            tracing::debug!(
                ?technique,
                %cluster_id,
                reliability_pct = reliability * 100.0,
                uses,
                "Adaptive Prompting: Promoted technique for cluster"
            );
        }

        Ok(queued)
    }

    /// Check and promote all eligible techniques
    ///
    /// This should be called periodically (e.g., after each task completion)
    /// to promote techniques that have reached the threshold.
    pub async fn check_and_promote_all(&mut self) -> Result<Vec<(String, PromptingTechnique)>> {
        let mut promoted = Vec::new();

        // Collect eligible techniques (to avoid borrowing issues)
        let eligible: Vec<_> = self
            .technique_stats
            .keys()
            .filter(|(cluster_id, technique)| self.should_promote(cluster_id, technique))
            .cloned()
            .collect();

        // Promote each eligible technique
        for (cluster_id, technique) in eligible {
            if self
                .promote_technique_to_bks(&cluster_id, &technique)
                .await?
            {
                promoted.push((cluster_id, technique));
            }
        }

        Ok(promoted)
    }

    /// Get statistics for a specific technique in a cluster
    pub fn get_stats(
        &self,
        cluster_id: &str,
        technique: &PromptingTechnique,
    ) -> Option<&TechniqueStats> {
        self.technique_stats
            .get(&(cluster_id.to_string(), technique.clone()))
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> &HashMap<(String, PromptingTechnique), TechniqueStats> {
        &self.technique_stats
    }

    /// Get recent records (last N)
    pub fn get_recent_records(&self, count: usize) -> Vec<&TechniqueEffectivenessRecord> {
        self.records.iter().rev().take(count).collect()
    }

    /// Get statistics summary for a cluster
    pub fn get_cluster_summary(&self, cluster_id: &str) -> ClusterSummary {
        let mut summary = ClusterSummary {
            cluster_id: cluster_id.to_string(),
            total_executions: 0,
            techniques: HashMap::new(),
        };

        for ((cid, technique), stats) in &self.technique_stats {
            if cid == cluster_id {
                summary.total_executions += stats.total_uses();
                summary.techniques.insert(technique.clone(), stats.clone());
            }
        }

        summary
    }

    /// Clear old records (keep only recent N records)
    pub fn prune_old_records(&mut self, keep_count: usize) {
        if self.records.len() > keep_count {
            let excess = self.records.len() - keep_count;
            self.records.drain(0..excess);
        }
    }

    /// Get promotion thresholds
    pub fn get_thresholds(&self) -> (f32, u32) {
        (self.promotion_threshold, self.min_uses_for_promotion)
    }
}

/// Summary of technique performance for a cluster
#[derive(Debug, Clone)]
pub struct ClusterSummary {
    /// The cluster identifier.
    pub cluster_id: String,
    /// Total number of task executions in this cluster.
    pub total_executions: u32,
    /// Per-technique performance statistics.
    pub techniques: HashMap<PromptingTechnique, TechniqueStats>,
}

impl ClusterSummary {
    /// Get the most effective technique in this cluster
    pub fn best_technique(&self) -> Option<(&PromptingTechnique, &TechniqueStats)> {
        self.techniques
            .iter()
            .filter(|(_, stats)| stats.total_uses() >= 3) // Minimum sample size
            .max_by(|(_, a), (_, b)| {
                // Compare by reliability, then by quality
                a.reliability()
                    .partial_cmp(&b.reliability())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        a.avg_quality
                            .partial_cmp(&b.avg_quality)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            })
    }

    /// Get techniques eligible for promotion
    pub fn promotable_techniques(&self, threshold: f32, min_uses: u32) -> Vec<&PromptingTechnique> {
        self.techniques
            .iter()
            .filter(|(_, stats)| stats.reliability() >= threshold && stats.total_uses() >= min_uses)
            .map(|(technique, _)| technique)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_technique_stats_update() {
        let mut stats = TechniqueStats::new();

        // Record 5 successes
        for _ in 0..5 {
            stats.update(true, 10, 0.9);
        }

        assert_eq!(stats.success_count, 5);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.reliability(), 1.0);
        assert_eq!(stats.total_uses(), 5);
        assert!(stats.avg_quality > 0.7); // EMA with alpha=0.3 from 0.0, 5 updates → ~0.75
    }

    #[test]
    fn test_should_promote() {
        let bks_cache = Arc::new(Mutex::new(
            BehavioralKnowledgeCache::in_memory(100).unwrap(),
        ));
        let mut coordinator = PromptingLearningCoordinator::new(bks_cache);

        let cluster_id = "test_cluster";
        let technique = PromptingTechnique::ChainOfThought;

        // Record 6 successes (meets threshold)
        for _ in 0..6 {
            coordinator.record_outcome(
                cluster_id.to_string(),
                vec![technique.clone()],
                "test task".to_string(),
                true,
                5,
                0.9,
            );
        }

        assert!(coordinator.should_promote(cluster_id, &technique));
    }

    #[test]
    fn test_not_enough_uses() {
        let bks_cache = Arc::new(Mutex::new(
            BehavioralKnowledgeCache::in_memory(100).unwrap(),
        ));
        let mut coordinator = PromptingLearningCoordinator::new(bks_cache);

        let cluster_id = "test_cluster";
        let technique = PromptingTechnique::ChainOfThought;

        // Only 3 uses (below threshold of 5)
        for _ in 0..3 {
            coordinator.record_outcome(
                cluster_id.to_string(),
                vec![technique.clone()],
                "test task".to_string(),
                true,
                5,
                0.9,
            );
        }

        assert!(!coordinator.should_promote(cluster_id, &technique));
    }

    #[test]
    fn test_reliability_too_low() {
        let bks_cache = Arc::new(Mutex::new(
            BehavioralKnowledgeCache::in_memory(100).unwrap(),
        ));
        let mut coordinator = PromptingLearningCoordinator::new(bks_cache);

        let cluster_id = "test_cluster";
        let technique = PromptingTechnique::ChainOfThought;

        // 3 successes, 3 failures = 50% reliability (below 80% threshold)
        for _ in 0..3 {
            coordinator.record_outcome(
                cluster_id.to_string(),
                vec![technique.clone()],
                "test task".to_string(),
                true,
                5,
                0.9,
            );
        }
        for _ in 0..3 {
            coordinator.record_outcome(
                cluster_id.to_string(),
                vec![technique.clone()],
                "test task".to_string(),
                false,
                5,
                0.5,
            );
        }

        assert!(!coordinator.should_promote(cluster_id, &technique));
    }

    #[test]
    fn test_cluster_summary() {
        let bks_cache = Arc::new(Mutex::new(
            BehavioralKnowledgeCache::in_memory(100).unwrap(),
        ));
        let mut coordinator = PromptingLearningCoordinator::new(bks_cache);

        let cluster_id = "test_cluster";

        // Record outcomes for multiple techniques
        coordinator.record_outcome(
            cluster_id.to_string(),
            vec![PromptingTechnique::ChainOfThought],
            "task 1".to_string(),
            true,
            5,
            0.9,
        );
        coordinator.record_outcome(
            cluster_id.to_string(),
            vec![PromptingTechnique::PlanAndSolve],
            "task 2".to_string(),
            true,
            8,
            0.85,
        );

        let summary = coordinator.get_cluster_summary(cluster_id);
        assert_eq!(summary.cluster_id, cluster_id);
        assert_eq!(summary.total_executions, 2);
        assert_eq!(summary.techniques.len(), 2);
    }

    #[tokio::test]
    async fn test_promotion_to_bks() {
        let bks_cache = Arc::new(Mutex::new(
            BehavioralKnowledgeCache::in_memory(100).unwrap(),
        ));
        let mut coordinator = PromptingLearningCoordinator::new(bks_cache.clone());

        let cluster_id = "numerical_reasoning";
        let technique = PromptingTechnique::ChainOfThought;

        // Record 6 successful uses
        for _ in 0..6 {
            coordinator.record_outcome(
                cluster_id.to_string(),
                vec![technique.clone()],
                "calculate primes".to_string(),
                true,
                5,
                0.9,
            );
        }

        // Promote to BKS
        let promoted = coordinator
            .promote_technique_to_bks(cluster_id, &technique)
            .await
            .unwrap();
        assert!(promoted);

        // Verify BKS contains the truth
        let bks = bks_cache.lock().await;
        let _truths = bks.all_truths().collect::<Vec<_>>();

        // Check that at least one truth was queued
        let pending = bks.pending_submissions();
        assert!(!pending.is_empty());

        let truth = &pending[0].truth;
        assert_eq!(truth.category, TruthCategory::PromptingTechnique);
        assert_eq!(truth.context_pattern, cluster_id);
        assert!(truth.rule.contains("ChainOfThought"));
    }
}
