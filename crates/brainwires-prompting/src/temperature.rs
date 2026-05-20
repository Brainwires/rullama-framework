//! Temperature Optimization
//!
//! This module provides adaptive temperature selection per task cluster,
//! based on the paper's findings:
//! - Low temp (0.0): Best for logical tasks (Zebra Puzzles, Web of Lies, Boolean Expressions)
//! - High temp (1.3): Best for linguistic tasks (Hyperbaton - adjective order judgment)
//!
//! Temperature performance is tracked per cluster and can be shared via BKS/PKS.

use super::clustering::TaskCluster;
use anyhow::Result;
use brainwires_knowledge::knowledge::bks_pks::{
    BehavioralKnowledgeCache, BehavioralTruth, TruthCategory, TruthSource,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Tracks performance metrics for a specific temperature setting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperaturePerformance {
    /// Success rate (0.0-1.0) using EMA
    pub success_rate: f32,
    /// Average quality score (0.0-1.0) using EMA
    pub avg_quality: f32,
    /// Number of samples collected
    pub sample_count: u32,
    /// Last updated timestamp
    pub last_updated: i64,
}

impl TemperaturePerformance {
    /// Create a new performance record with neutral defaults.
    pub fn new() -> Self {
        Self {
            success_rate: 0.5, // Neutral starting point
            avg_quality: 0.5,
            sample_count: 0,
            last_updated: chrono::Utc::now().timestamp(),
        }
    }

    /// Update metrics with new outcome using EMA (alpha = 0.3)
    pub fn update(&mut self, success: bool, quality: f32) {
        let alpha = 0.3;
        self.success_rate =
            alpha * (if success { 1.0 } else { 0.0 }) + (1.0 - alpha) * self.success_rate;
        self.avg_quality = alpha * quality + (1.0 - alpha) * self.avg_quality;
        self.sample_count += 1;
        self.last_updated = chrono::Utc::now().timestamp();
    }

    /// Combined score for ranking (60% success rate, 40% quality)
    pub fn score(&self) -> f32 {
        0.6 * self.success_rate + 0.4 * self.avg_quality
    }
}

impl Default for TemperaturePerformance {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages adaptive temperature selection per task cluster
pub struct TemperatureOptimizer {
    /// Maps (cluster_id, temperature_int) → performance stats
    /// Temperature stored as i32 (multiply by 10: 0.0 → 0, 0.2 → 2, 1.3 → 13)
    performance_map: HashMap<(String, i32), TemperaturePerformance>,
    /// BKS cache for querying shared temperature preferences
    bks_cache: Option<Arc<Mutex<BehavioralKnowledgeCache>>>,
    /// Candidate temperatures to test (from paper)
    candidates: Vec<f32>,
    /// Minimum samples before trusting a temperature setting
    min_samples: u32,
}

impl TemperatureOptimizer {
    /// Create a new temperature optimizer
    pub fn new() -> Self {
        Self {
            performance_map: HashMap::new(),
            bks_cache: None,
            candidates: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3],
            min_samples: 5,
        }
    }

    /// Convert temperature f32 to i32 for HashMap key
    fn temp_to_key(temp: f32) -> i32 {
        (temp * 10.0).round() as i32
    }

    /// Set BKS cache for querying shared temperature knowledge
    pub fn with_bks(mut self, bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>) -> Self {
        self.bks_cache = Some(bks_cache);
        self
    }

    /// Set minimum samples required before trusting a temperature
    pub fn with_min_samples(mut self, min_samples: u32) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Get optimal temperature for a cluster
    ///
    /// Selection order:
    /// 1. BKS shared knowledge (if available)
    /// 2. Local learned temperature (if enough samples)
    /// 3. Default heuristic based on cluster characteristics
    pub async fn get_optimal_temperature(&self, cluster: &TaskCluster) -> f32 {
        // Source 1: BKS shared knowledge
        if let Some(bks_temp) = self.query_bks_temperature(&cluster.id).await {
            return bks_temp;
        }

        // Source 2: Local learned temperature
        if let Some(local_temp) = self.get_local_optimal(&cluster.id) {
            return local_temp;
        }

        // Source 3: Default heuristic based on cluster characteristics
        self.get_default_temperature(cluster)
    }

    /// Get locally learned optimal temperature
    fn get_local_optimal(&self, cluster_id: &str) -> Option<f32> {
        let mut best_temp = None;
        let mut best_score = f32::NEG_INFINITY;

        for &temp in &self.candidates {
            let temp_key = Self::temp_to_key(temp);
            if let Some(perf) = self
                .performance_map
                .get(&(cluster_id.to_string(), temp_key))
                && perf.sample_count >= self.min_samples
            {
                let score = perf.score();
                if score > best_score {
                    best_score = score;
                    best_temp = Some(temp);
                }
            }
        }

        best_temp
    }

    /// Query BKS for shared temperature knowledge
    async fn query_bks_temperature(&self, cluster_id: &str) -> Option<f32> {
        if let Some(ref bks_cache) = self.bks_cache {
            let bks = bks_cache.lock().await;

            // Query for temperature truths for this cluster
            // get_matching_truths takes just a context string
            let truths = bks.get_matching_truths(cluster_id);

            // Parse temperature from truth content
            // Example: "For numerical_reasoning, use temperature 0.0 for optimal results"
            for truth in truths {
                // Filter for TaskStrategy category
                if truth.category == TruthCategory::TaskStrategy
                    && let Some(temp) = self.parse_temperature_from_truth(truth)
                {
                    return Some(temp);
                }
            }
        }

        None
    }

    /// Parse temperature value from BKS truth
    fn parse_temperature_from_truth(&self, truth: &BehavioralTruth) -> Option<f32> {
        // Look for "temperature X.X" pattern in rule or rationale
        let text = format!("{} {}", truth.rule, truth.rationale);

        // Simple regex-like parsing
        if let Some(idx) = text.find("temperature") {
            let substr = &text[idx..];
            // Find first number after "temperature"
            let parts: Vec<&str> = substr.split_whitespace().collect();
            for part in parts.iter().skip(1) {
                if let Ok(temp) = part.parse::<f32>()
                    && self.candidates.contains(&temp)
                {
                    return Some(temp);
                }
            }
        }

        None
    }

    /// Get default temperature based on cluster characteristics (heuristic)
    fn get_default_temperature(&self, cluster: &TaskCluster) -> f32 {
        let desc = cluster.description.to_lowercase();

        // Logic/reasoning tasks: Low temperature (0.0)
        if desc.contains("logic")
            || desc.contains("boolean")
            || desc.contains("reasoning")
            || desc.contains("puzzle")
            || desc.contains("deduction")
        {
            return 0.0;
        }

        // Creative/linguistic tasks: High temperature (1.3)
        if desc.contains("creative")
            || desc.contains("linguistic")
            || desc.contains("story")
            || desc.contains("writing")
            || desc.contains("generation")
        {
            return 1.3;
        }

        // Numerical/calculation tasks: Low temperature (0.2)
        if desc.contains("numerical")
            || desc.contains("calculation")
            || desc.contains("math")
            || desc.contains("arithmetic")
        {
            return 0.2;
        }

        // Code generation: Moderate temperature (0.6)
        if desc.contains("code")
            || desc.contains("programming")
            || desc.contains("implementation")
            || desc.contains("algorithm")
        {
            return 0.6;
        }

        // Default: Moderate temperature
        0.7
    }

    /// Record outcome for a temperature setting
    pub fn record_temperature_outcome(
        &mut self,
        cluster_id: String,
        temperature: f32,
        success: bool,
        quality: f32,
    ) {
        let temp_key = Self::temp_to_key(temperature);
        let key = (cluster_id, temp_key);
        let perf = self.performance_map.entry(key).or_default();

        perf.update(success, quality);
    }

    /// Check if temperature should be promoted to BKS
    pub async fn check_and_promote_temperature(
        &self,
        cluster_id: &str,
        temperature: f32,
        min_score: f32,
        min_samples: u32,
    ) -> Result<()> {
        let temp_key = Self::temp_to_key(temperature);
        let key = (cluster_id.to_string(), temp_key);

        if let Some(perf) = self.performance_map.get(&key)
            && perf.sample_count >= min_samples
            && perf.score() >= min_score
        {
            // Promote to BKS
            if let Some(ref bks_cache) = self.bks_cache {
                let truth = BehavioralTruth::new(
                    TruthCategory::TaskStrategy,
                    cluster_id.to_string(),
                    format!(
                        "For {} tasks, use temperature {} for optimal results",
                        cluster_id, temperature
                    ),
                    format!(
                        "Learned from {} executions with {:.1}% success rate and {:.2} avg quality",
                        perf.sample_count,
                        perf.success_rate * 100.0,
                        perf.avg_quality
                    ),
                    TruthSource::SuccessPattern,
                    None,
                );

                let mut bks = bks_cache.lock().await;
                bks.queue_submission(truth)?;
            }
        }

        Ok(())
    }

    /// Get all performance data (for debugging/inspection)
    pub fn get_all_performance(&self) -> &HashMap<(String, i32), TemperaturePerformance> {
        &self.performance_map
    }

    /// Get performance for a specific cluster and temperature
    pub fn get_performance(
        &self,
        cluster_id: &str,
        temperature: f32,
    ) -> Option<&TemperaturePerformance> {
        let temp_key = Self::temp_to_key(temperature);
        self.performance_map
            .get(&(cluster_id.to_string(), temp_key))
    }
}

impl Default for TemperatureOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::techniques::PromptingTechnique;

    #[test]
    fn test_temperature_performance_update() {
        let mut perf = TemperaturePerformance::new();
        assert_eq!(perf.success_rate, 0.5);
        assert_eq!(perf.sample_count, 0);

        // Record success
        perf.update(true, 0.9);
        assert!(perf.success_rate > 0.5); // Should increase
        assert_eq!(perf.sample_count, 1);

        // Record failure
        perf.update(false, 0.3);
        assert_eq!(perf.sample_count, 2);
        assert!(perf.avg_quality < 0.9); // Should decrease
    }

    #[test]
    fn test_temperature_performance_score() {
        let mut perf = TemperaturePerformance::new();
        perf.success_rate = 0.8;
        perf.avg_quality = 0.7;

        let score = perf.score();
        assert!((score - 0.76).abs() < 0.01); // 0.6 * 0.8 + 0.4 * 0.7 = 0.76
    }

    #[test]
    fn test_default_temperature_heuristics() {
        let optimizer = TemperatureOptimizer::new();

        // Logic task: Low temperature
        let logic_cluster = TaskCluster::new(
            "logic_task".to_string(),
            "Boolean logic and reasoning puzzles".to_string(),
            vec![0.5; 768],
            vec![PromptingTechnique::LogicOfThought],
            vec![],
        );
        assert_eq!(optimizer.get_default_temperature(&logic_cluster), 0.0);

        // Creative task: High temperature
        let creative_cluster = TaskCluster::new(
            "creative_task".to_string(),
            "Creative writing and story generation".to_string(),
            vec![0.5; 768],
            vec![PromptingTechnique::RolePlaying],
            vec![],
        );
        assert_eq!(optimizer.get_default_temperature(&creative_cluster), 1.3);

        // Code task: Moderate temperature
        let code_cluster = TaskCluster::new(
            "code_task".to_string(),
            "Code implementation and algorithm design".to_string(),
            vec![0.5; 768],
            vec![PromptingTechnique::PlanAndSolve],
            vec![],
        );
        assert_eq!(optimizer.get_default_temperature(&code_cluster), 0.6);
    }

    #[test]
    fn test_record_and_retrieve_local_optimal() {
        let mut optimizer = TemperatureOptimizer::new();

        // Record outcomes for different temperatures
        for _ in 0..10 {
            optimizer.record_temperature_outcome("test_cluster".to_string(), 0.0, true, 0.9);
            optimizer.record_temperature_outcome("test_cluster".to_string(), 0.6, false, 0.5);
        }

        // Get optimal (should be 0.0 due to high success rate)
        let optimal = optimizer.get_local_optimal("test_cluster");
        assert_eq!(optimal, Some(0.0));
    }

    #[test]
    fn test_min_samples_requirement() {
        let mut optimizer = TemperatureOptimizer::new().with_min_samples(5);

        // Record only 3 samples
        for _ in 0..3 {
            optimizer.record_temperature_outcome("test_cluster".to_string(), 0.0, true, 0.95);
        }

        // Should not return optimal (not enough samples)
        assert_eq!(optimizer.get_local_optimal("test_cluster"), None);

        // Add 2 more samples
        for _ in 0..2 {
            optimizer.record_temperature_outcome("test_cluster".to_string(), 0.0, true, 0.95);
        }

        // Now should return optimal
        assert_eq!(optimizer.get_local_optimal("test_cluster"), Some(0.0));
    }

    #[tokio::test]
    async fn test_get_optimal_temperature_fallback() {
        let optimizer = TemperatureOptimizer::new();

        // No BKS, no local data → should use heuristic
        let cluster = TaskCluster::new(
            "logic_test".to_string(),
            "Boolean logic problems".to_string(),
            vec![0.5; 768],
            vec![PromptingTechnique::LogicOfThought],
            vec![],
        );

        let temp = optimizer.get_optimal_temperature(&cluster).await;
        assert_eq!(temp, 0.0); // Heuristic for logic tasks
    }
}
