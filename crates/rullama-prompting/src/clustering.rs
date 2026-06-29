//! Task Clustering System with SEAL Integration
//!
//! This module implements k-means clustering of tasks by semantic similarity,
//! enhanced with SEAL's query core extraction for better classification.

use super::techniques::{ComplexityLevel, PromptingTechnique};
use crate::seal::SealProcessingResult;
#[cfg(feature = "prompting")]
use anyhow::Context as _;
use anyhow::{Result, anyhow};
#[cfg(feature = "prompting")]
use linfa::prelude::*;
#[cfg(feature = "prompting")]
use linfa_clustering::KMeans;
#[cfg(feature = "prompting")]
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// A task cluster identified by semantic similarity (SEAL-enhanced)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCluster {
    /// Unique cluster identifier.
    pub id: String,
    /// LLM-generated semantic description of this cluster.
    pub description: String,
    /// Embedding vector of the cluster description.
    pub embedding: Vec<f32>,
    /// Prompting techniques mapped to this cluster (typically 3-4).
    pub techniques: Vec<PromptingTechnique>,
    /// Example task descriptions belonging to this cluster.
    pub example_tasks: Vec<String>,

    /// Example query cores from SEAL for tasks in this cluster.
    pub seal_query_cores: Vec<String>,
    /// Average SEAL quality score for tasks in this cluster.
    pub avg_seal_quality: f32,
    /// Recommended complexity level based on average SEAL quality.
    pub recommended_complexity: ComplexityLevel,
}

impl TaskCluster {
    /// Create a new task cluster
    pub fn new(
        id: String,
        description: String,
        embedding: Vec<f32>,
        techniques: Vec<PromptingTechnique>,
        example_tasks: Vec<String>,
    ) -> Self {
        Self {
            id,
            description,
            embedding,
            techniques,
            example_tasks,
            seal_query_cores: Vec::new(),
            avg_seal_quality: 0.5,
            recommended_complexity: ComplexityLevel::Moderate,
        }
    }

    /// Update SEAL-related metrics
    pub fn update_seal_metrics(&mut self, query_cores: Vec<String>, avg_quality: f32) {
        self.seal_query_cores = query_cores;
        self.avg_seal_quality = avg_quality;
        self.recommended_complexity = if avg_quality < 0.5 {
            ComplexityLevel::Simple
        } else if avg_quality < 0.8 {
            ComplexityLevel::Moderate
        } else {
            ComplexityLevel::Advanced
        };
    }
}

/// Manages task clustering
pub struct TaskClusterManager {
    clusters: Vec<TaskCluster>,
    _embedding_dim: usize,
}

impl TaskClusterManager {
    /// Create a new task cluster manager
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
            _embedding_dim: 768, // Default for most embedding models
        }
    }

    /// Create with specific embedding dimension
    pub fn with_embedding_dim(embedding_dim: usize) -> Self {
        Self {
            clusters: Vec::new(),
            _embedding_dim: embedding_dim,
        }
    }

    /// Get all clusters
    pub fn get_clusters(&self) -> &[TaskCluster] {
        &self.clusters
    }

    /// Add a cluster
    pub fn add_cluster(&mut self, cluster: TaskCluster) {
        self.clusters.push(cluster);
    }

    /// Set clusters (replaces existing)
    pub fn set_clusters(&mut self, clusters: Vec<TaskCluster>) {
        self.clusters = clusters;
    }

    /// Find task cluster most similar to a task description (SEAL-enhanced)
    ///
    /// This is the core classification function that:
    /// 1. Uses SEAL's resolved query if available (not original query)
    /// 2. Prefers SEAL's query core for better semantic matching
    /// 3. Boosts similarity if SEAL quality is high
    ///
    /// # Arguments
    /// * `task_embedding` - Pre-computed embedding of the task
    /// * `seal_result` - Optional SEAL processing result for enhancement
    ///
    /// # Returns
    /// * Tuple of (cluster reference, similarity score)
    pub fn find_matching_cluster(
        &self,
        task_embedding: &[f32],
        seal_result: Option<&SealProcessingResult>,
    ) -> Result<(&TaskCluster, f32)> {
        if self.clusters.is_empty() {
            return Err(anyhow!("No clusters available"));
        }

        let mut best_match = None;
        let mut best_similarity = f32::NEG_INFINITY;

        for cluster in &self.clusters {
            let similarity = cosine_similarity(task_embedding, &cluster.embedding);

            // Boost similarity if SEAL quality is high
            let boosted_similarity = if let Some(seal) = seal_result {
                if seal.quality_score > 0.7 {
                    similarity * 1.1 // 10% boost for high-quality SEAL resolutions
                } else {
                    similarity
                }
            } else {
                similarity
            };

            if boosted_similarity > best_similarity {
                best_similarity = boosted_similarity;
                best_match = Some(cluster);
            }
        }

        let cluster = best_match.ok_or_else(|| anyhow!("No matching cluster found"))?;
        Ok((cluster, best_similarity))
    }

    /// Build clusters from a set of task embeddings using k-means (requires prompting feature - linfa)
    #[cfg(feature = "prompting")]
    pub fn build_clusters_from_embeddings(
        &mut self,
        task_embeddings: Array2<f32>,
        task_descriptions: Vec<String>,
        min_clusters: usize,
        max_clusters: usize,
    ) -> Result<Vec<usize>> {
        if task_embeddings.nrows() != task_descriptions.len() {
            return Err(anyhow!(
                "Embeddings and descriptions length mismatch: {} vs {}",
                task_embeddings.nrows(),
                task_descriptions.len()
            ));
        }

        if task_embeddings.nrows() < min_clusters {
            return Err(anyhow!(
                "Not enough tasks ({}) for minimum clusters ({})",
                task_embeddings.nrows(),
                min_clusters
            ));
        }

        // Find optimal K using silhouette scores
        let optimal_k = self.find_optimal_k(&task_embeddings, min_clusters, max_clusters)?;

        // Perform k-means clustering
        let assignments = self.perform_kmeans(&task_embeddings, optimal_k)?;

        // Build cluster objects
        self.build_cluster_objects(
            &task_embeddings,
            &task_descriptions,
            &assignments,
            optimal_k,
        )?;

        Ok(assignments)
    }

    /// Find optimal number of clusters using silhouette scores
    #[cfg(feature = "prompting")]
    fn find_optimal_k(
        &self,
        embeddings: &Array2<f32>,
        min_k: usize,
        max_k: usize,
    ) -> Result<usize> {
        let mut best_k = min_k;
        let mut best_score = f32::NEG_INFINITY;

        let effective_max_k = max_k.min(embeddings.nrows() / 2);

        for k in min_k..=effective_max_k {
            let assignments = self.perform_kmeans(embeddings, k)?;
            let score = self.compute_silhouette_score(embeddings, &assignments, k);

            if score > best_score {
                best_score = score;
                best_k = k;
            }
        }

        Ok(best_k)
    }

    /// Perform k-means clustering
    #[cfg(feature = "prompting")]
    fn perform_kmeans(&self, embeddings: &Array2<f32>, k: usize) -> Result<Vec<usize>> {
        let dataset = DatasetBase::from(embeddings.clone());

        let model = KMeans::params(k)
            .max_n_iterations(100)
            .tolerance(1e-4)
            .fit(&dataset)
            .context("K-means fitting failed")?;

        let assignments: Vec<usize> = model.predict(embeddings).iter().copied().collect();

        Ok(assignments)
    }

    /// Compute silhouette score for clustering quality
    #[cfg(feature = "prompting")]
    fn compute_silhouette_score(
        &self,
        embeddings: &Array2<f32>,
        assignments: &[usize],
        k: usize,
    ) -> f32 {
        let n = embeddings.nrows();
        if n == 0 {
            return 0.0;
        }

        let mut silhouette_sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            let cluster_i = assignments[i];

            let mut a_i = 0.0;
            let mut same_cluster_count = 0;
            for (j, &assignment_j) in assignments.iter().enumerate().take(n) {
                if i != j && assignment_j == cluster_i {
                    a_i += euclidean_distance(
                        &embeddings.row(i).to_vec(),
                        &embeddings.row(j).to_vec(),
                    );
                    same_cluster_count += 1;
                }
            }
            if same_cluster_count > 0 {
                a_i /= same_cluster_count as f32;
            }

            let mut b_i = f32::INFINITY;
            for other_cluster in 0..k {
                if other_cluster == cluster_i {
                    continue;
                }

                let mut dist_sum = 0.0;
                let mut other_count = 0;
                for (j, &assignment_j) in assignments.iter().enumerate().take(n) {
                    if assignment_j == other_cluster {
                        dist_sum += euclidean_distance(
                            &embeddings.row(i).to_vec(),
                            &embeddings.row(j).to_vec(),
                        );
                        other_count += 1;
                    }
                }
                if other_count > 0 {
                    let avg_dist = dist_sum / other_count as f32;
                    b_i = b_i.min(avg_dist);
                }
            }

            if b_i.is_finite() && a_i > 0.0 {
                let s_i = (b_i - a_i) / a_i.max(b_i);
                silhouette_sum += s_i;
                count += 1;
            }
        }

        if count > 0 {
            silhouette_sum / count as f32
        } else {
            0.0
        }
    }

    /// Build cluster objects from assignments
    #[cfg(feature = "prompting")]
    fn build_cluster_objects(
        &mut self,
        embeddings: &Array2<f32>,
        descriptions: &[String],
        assignments: &[usize],
        k: usize,
    ) -> Result<()> {
        let mut clusters = Vec::new();

        for cluster_id in 0..k {
            let mut cluster_tasks = Vec::new();
            let mut cluster_embeddings = Vec::new();

            for (i, &assignment) in assignments.iter().enumerate() {
                if assignment == cluster_id {
                    cluster_tasks.push(descriptions[i].clone());
                    cluster_embeddings.push(embeddings.row(i).to_vec());
                }
            }

            if cluster_tasks.is_empty() {
                continue;
            }

            let centroid = compute_centroid(&cluster_embeddings);

            let cluster = TaskCluster::new(
                format!("cluster_{}", cluster_id),
                format!("Cluster {}", cluster_id),
                centroid,
                Vec::new(),
                cluster_tasks.iter().take(5).cloned().collect(),
            );

            clusters.push(cluster);
        }

        self.clusters = clusters;
        Ok(())
    }

    /// Get cluster count
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Get cluster by ID
    pub fn get_cluster_by_id(&self, id: &str) -> Option<&TaskCluster> {
        self.clusters.iter().find(|c| c.id == id)
    }

    /// Get mutable cluster by ID
    pub fn get_cluster_by_id_mut(&mut self, id: &str) -> Option<&mut TaskCluster> {
        self.clusters.iter_mut().find(|c| c.id == id)
    }
}

impl Default for TaskClusterManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute Euclidean distance between two vectors
#[allow(dead_code)]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute centroid of a set of embeddings
#[allow(dead_code)]
fn compute_centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let mut centroid = vec![0.0; dim];

    for embedding in embeddings {
        for (i, &val) in embedding.iter().enumerate() {
            centroid[i] += val;
        }
    }

    let n = embeddings.len() as f32;
    for val in &mut centroid {
        *val /= n;
    }

    centroid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_centroid() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let centroid = compute_centroid(&embeddings);
        assert_eq!(centroid, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cluster_manager_basic() {
        let mut manager = TaskClusterManager::new();
        assert_eq!(manager.cluster_count(), 0);

        let cluster = TaskCluster::new(
            "test_cluster".to_string(),
            "Test cluster".to_string(),
            vec![0.1, 0.2, 0.3],
            Vec::new(),
            vec!["task1".to_string()],
        );

        manager.add_cluster(cluster);
        assert_eq!(manager.cluster_count(), 1);
    }
}
