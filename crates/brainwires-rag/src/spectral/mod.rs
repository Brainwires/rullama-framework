//! MSS-inspired spectral subset selection for diverse RAG retrieval.
//!
//! Standard top-k retrieval by cosine similarity produces redundant results.
//! This module implements a greedy log-determinant maximization algorithm
//! (inspired by DPP / Marcus-Spielman-Srivastava interlacing polynomials)
//! that selects k items that are both relevant AND collectively diverse.
//!
//! # Algorithm
//!
//! Given n candidate embeddings with relevance scores, we build a kernel matrix:
//! ```text
//! L_ij = (r_i^lambda) * (r_j^lambda) * cos_sim(v_i, v_j)
//! ```
//! and greedily select the subset S of size k that maximizes `log det(L_S)`.
//!
//! The greedy algorithm achieves a (1 - 1/e) ~ 0.63 approximation ratio
//! to the optimal solution, which is the best possible in polynomial time
//! for submodular maximization.
//!
//! # Complexity
//!
//! O(n*k^3) -- trivial for n <= 200, k <= 20 (typical RAG retrieval sizes).
//! With incremental Cholesky updates, the inner loop is O(k^2) per candidate,
//! giving overall O(n*k^2).

pub mod graph_ops;
pub mod kernel;
pub mod linalg;

use brainwires_core::SearchResult;
use kernel::{build_kernel_matrix, cross_column};
use linalg::{cholesky_extend, log_det_incremental};
use ndarray::Array2;

/// Configuration for spectral subset selection.
#[derive(Debug, Clone)]
pub struct SpectralSelectConfig {
    /// Number of items to select. If `None`, uses the query limit.
    pub k: Option<usize>,
    /// Relevance/diversity trade-off parameter.
    /// - 0.0 = pure diversity (ignores relevance scores)
    /// - 1.0 = relevance-dominated (approaches standard top-k)
    /// - 0.5 = balanced (default)
    pub lambda: f32,
    /// Minimum number of candidates to trigger spectral selection.
    /// Below this threshold, results are returned as-is.
    pub min_candidates: usize,
    /// Diagonal regularization epsilon for numerical stability.
    pub regularization: f32,
}

impl Default for SpectralSelectConfig {
    fn default() -> Self {
        Self {
            k: None,
            lambda: 0.5,
            min_candidates: 10,
            regularization: 1e-6,
        }
    }
}

/// Trait for diversity-aware reranking of search results.
pub trait DiversityReranker: Send + Sync {
    /// Rerank candidates, returning indices into `results` in selection order.
    ///
    /// # Arguments
    ///
    /// * `results` - Original search results with scores
    /// * `embeddings` - Embedding vectors corresponding to each result
    /// * `k` - Number of items to select
    ///
    /// # Returns
    ///
    /// Indices into `results`, ordered by selection round (first selected = most valuable).
    fn rerank(&self, results: &[SearchResult], embeddings: &[Vec<f32>], k: usize) -> Vec<usize>;
}

/// Spectral reranker using greedy log-determinant maximization.
pub struct SpectralReranker {
    config: SpectralSelectConfig,
}

impl SpectralReranker {
    /// Create a new spectral reranker with the given configuration.
    pub fn new(config: SpectralSelectConfig) -> Self {
        Self { config }
    }

    /// Create a spectral reranker with default settings.
    pub fn with_defaults() -> Self {
        Self::new(SpectralSelectConfig::default())
    }
}

impl DiversityReranker for SpectralReranker {
    fn rerank(&self, results: &[SearchResult], embeddings: &[Vec<f32>], k: usize) -> Vec<usize> {
        let n = results.len();

        // Edge cases
        if n == 0 {
            return Vec::new();
        }
        if k >= n {
            return (0..n).collect();
        }
        if k == 0 {
            return Vec::new();
        }

        // Skip spectral selection if too few candidates
        if n < self.config.min_candidates {
            return (0..k.min(n)).collect();
        }

        // Build kernel matrix
        let embedding_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let kernel = build_kernel_matrix(
            &embedding_refs,
            &scores,
            self.config.lambda,
            self.config.regularization,
        );

        greedy_log_det_select(&kernel, k)
    }
}

/// Greedy log-determinant maximization with incremental Cholesky updates.
///
/// Selects k indices from the n*n kernel matrix that (approximately) maximize
/// `log det(L_S)`, achieving a (1-1/e) approximation ratio.
fn greedy_log_det_select(kernel: &Array2<f32>, k: usize) -> Vec<usize> {
    let n = kernel.nrows();
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<bool> = vec![true; n];

    // Current Cholesky factor of the selected submatrix (starts empty)
    let mut chol_s: Option<Array2<f32>> = None;
    let mut current_log_det: f32 = 0.0;

    for round in 0..k {
        let mut best_idx = usize::MAX;
        let mut best_gain = f32::NEG_INFINITY;

        for c in 0..n {
            if !remaining[c] {
                continue;
            }

            let gain = if round == 0 {
                // First selection: gain = log(L_{c,c})
                let diag = kernel[[c, c]];
                if diag > 0.0 {
                    diag.ln()
                } else {
                    f32::NEG_INFINITY
                }
            } else {
                // Incremental gain via Cholesky
                let cross = cross_column(kernel, &selected, c);
                let diag_cc = kernel[[c, c]];
                let new_ld = log_det_incremental(
                    chol_s.as_ref().expect(
                        "chol_s is initialized in round 0 before any incremental round runs",
                    ),
                    &cross,
                    diag_cc,
                    current_log_det,
                );
                new_ld - current_log_det
            };

            if gain > best_gain {
                best_gain = gain;
                best_idx = c;
            }
        }

        if best_idx == usize::MAX || best_gain == f32::NEG_INFINITY {
            // No more valid candidates (degenerate kernel)
            break;
        }

        // Update Cholesky factor
        if round == 0 {
            let diag = kernel[[best_idx, best_idx]];
            let mut l = Array2::<f32>::zeros((1, 1));
            l[[0, 0]] = diag.sqrt();
            chol_s = Some(l);
            current_log_det = diag.ln();
        } else {
            let cross = cross_column(kernel, &selected, best_idx);
            let diag_cc = kernel[[best_idx, best_idx]];
            chol_s = Some(
                cholesky_extend(
                    chol_s.as_ref().expect(
                        "chol_s is initialized in round 0 before any incremental round runs",
                    ),
                    &cross,
                    diag_cc,
                )
                .expect("Cholesky extend failed after positive gain check"),
            );
            current_log_det += best_gain;
        }

        selected.push(best_idx);
        remaining[best_idx] = false;
    }

    selected
}

// ── Cross-encoder reranker ────────────────────────────────────────────────

/// Configuration for the query-aware cross-encoder reranker.
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Blend weight between the original retrieval score and the query-document
    /// cosine similarity.
    ///
    /// - `1.0` → use original retrieval score only (no re-ranking)
    /// - `0.0` → use query-document cosine similarity only
    /// - `0.5` → equal blend (default)
    pub alpha: f32,
    /// Pre-computed query embedding used as the "query" side of the joint score.
    ///
    /// If empty, the reranker falls back to the original score order (alpha = 1.0).
    pub query_embedding: Vec<f32>,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            query_embedding: Vec::new(),
        }
    }
}

/// Query-aware reranker that blends the original retrieval score with a
/// query-document cosine similarity for a joint re-scoring pass.
///
/// This is a lightweight, embedding-based approximation of a true cross-encoder
/// that requires no additional model — it reuses the same embeddings already
/// computed during retrieval.
pub struct CrossEncoderReranker {
    config: CrossEncoderConfig,
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker with the given configuration.
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self { config }
    }

    /// Convenience constructor — specify alpha and query embedding directly.
    pub fn with_alpha(alpha: f32, query_embedding: Vec<f32>) -> Self {
        Self::new(CrossEncoderConfig {
            alpha,
            query_embedding,
        })
    }
}

impl DiversityReranker for CrossEncoderReranker {
    fn rerank(&self, results: &[SearchResult], embeddings: &[Vec<f32>], k: usize) -> Vec<usize> {
        let n = results.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        if k >= n {
            return (0..n).collect();
        }

        // If no query embedding, fall back to score-descending order.
        if self.config.query_embedding.is_empty() {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                results[b]
                    .score
                    .partial_cmp(&results[a].score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return indices.into_iter().take(k).collect();
        }

        let query_emb = &self.config.query_embedding;
        let alpha = self.config.alpha.clamp(0.0, 1.0);

        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let cos = if i < embeddings.len() {
                    kernel::cosine_similarity(query_emb, &embeddings[i])
                } else {
                    0.0
                };
                let joint = alpha * results[i].score + (1.0 - alpha) * cos;
                (i, joint)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

/// Select which reranker(s) to apply in [`crate::rag::client::RagClient::query_diverse`].
pub enum RerankerKind {
    /// Greedy log-determinant spectral reranker (diversity-focused).
    Spectral(SpectralSelectConfig),
    /// Query-aware cross-encoder reranker (relevance-focused).
    CrossEncoder(CrossEncoderConfig),
    /// Apply spectral reranking first (for diversity), then cross-encoder on
    /// the selected subset (for final relevance ordering).
    Both {
        /// Config for the spectral (first) pass.
        spectral: SpectralSelectConfig,
        /// Config for the cross-encoder (second) pass.
        cross_encoder: CrossEncoderConfig,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(score: f32) -> SearchResult {
        SearchResult {
            file_path: String::new(),
            root_path: None,
            content: String::new(),
            score,
            vector_score: score,
            keyword_score: None,
            start_line: 0,
            end_line: 0,
            language: String::new(),
            project: None,
            indexed_at: 0,
        }
    }

    #[test]
    fn test_empty_input() {
        let reranker = SpectralReranker::with_defaults();
        let result = reranker.rerank(&[], &[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_k_zero() {
        let reranker = SpectralReranker::with_defaults();
        let results = vec![make_result(0.9)];
        let embeddings = vec![vec![1.0, 0.0]];
        let result = reranker.rerank(&results, &embeddings, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_k_greater_than_n() {
        let reranker = SpectralReranker::with_defaults();
        let results = vec![make_result(0.9), make_result(0.8)];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = reranker.rerank(&results, &embeddings, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_below_min_candidates() {
        let config = SpectralSelectConfig {
            min_candidates: 20,
            ..Default::default()
        };
        let reranker = SpectralReranker::new(config);
        let results: Vec<SearchResult> =
            (0..5).map(|i| make_result(0.9 - i as f32 * 0.1)).collect();
        let embeddings: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 0.0]).collect();
        let result = reranker.rerank(&results, &embeddings, 3);
        // Should return first 3 indices unchanged
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_spectral_prefers_diverse() {
        // Create 10 near-duplicate vectors + 5 diverse vectors
        // The spectral reranker should prefer the diverse ones
        let mut results = Vec::new();
        let mut embeddings = Vec::new();

        // 10 near-duplicates (high score, very similar embeddings)
        for i in 0..10 {
            results.push(make_result(0.95));
            let mut emb = vec![1.0, 0.0, 0.0, 0.0, 0.0];
            emb[0] += i as f32 * 0.01; // tiny variation
            embeddings.push(emb);
        }

        // 5 diverse vectors (slightly lower score, orthogonal embeddings)
        let diverse_dirs = [
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.0, 0.0],
        ];
        for dir in &diverse_dirs {
            results.push(make_result(0.85));
            embeddings.push(dir.clone());
        }

        let reranker = SpectralReranker::new(SpectralSelectConfig {
            min_candidates: 5,
            lambda: 0.3, // favor diversity
            ..Default::default()
        });

        let selected = reranker.rerank(&results, &embeddings, 5);
        assert_eq!(selected.len(), 5);

        // Count how many of the selected are from the diverse set (indices 10-14)
        let diverse_count = selected.iter().filter(|&&idx| idx >= 10).count();
        // With lambda=0.3 (diversity-favoring), we should pick at least 3 diverse items
        assert!(
            diverse_count >= 3,
            "Expected at least 3 diverse items, got {}. Selected: {:?}",
            diverse_count,
            selected
        );
    }

    #[test]
    fn test_lambda_one_approximates_topk() {
        // With lambda=1.0, relevance dominates -- should approximate top-k by score
        let mut results = Vec::new();
        let mut embeddings = Vec::new();

        for i in 0..15 {
            let score = 1.0 - i as f32 * 0.05;
            results.push(make_result(score));
            // Even with diverse embeddings, high lambda should prefer high scores
            let mut emb = vec![0.0; 10];
            emb[i % 10] = 1.0;
            embeddings.push(emb);
        }

        let reranker = SpectralReranker::new(SpectralSelectConfig {
            min_candidates: 5,
            lambda: 1.0,
            ..Default::default()
        });

        let selected = reranker.rerank(&results, &embeddings, 5);
        assert_eq!(selected.len(), 5);

        // The top 5 by score are indices 0..5
        // With lambda=1.0 and diverse embeddings, all top-5 should be selected
        // (since they're all diverse AND high-scoring)
        for &idx in &selected {
            assert!(
                idx < 7,
                "Expected top items, got index {}. Selected: {:?}",
                idx,
                selected
            );
        }
    }

    #[test]
    fn test_k_equals_one() {
        // k=1 should pick the single best item (highest diagonal = highest score * self-sim)
        let results = vec![make_result(0.5), make_result(0.9), make_result(0.7)];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let reranker = SpectralReranker::new(SpectralSelectConfig {
            min_candidates: 2,
            ..Default::default()
        });

        let selected = reranker.rerank(&results, &embeddings, 1);
        assert_eq!(selected.len(), 1);
        // Index 1 has highest score (0.9), should be selected
        assert_eq!(selected[0], 1);
    }

    #[test]
    fn test_greedy_determinism() {
        // Same input should always produce same output
        let results: Vec<SearchResult> = (0..12)
            .map(|i| make_result(0.9 - i as f32 * 0.05))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let mut e = vec![0.0; 5];
                e[i % 5] = 1.0;
                e
            })
            .collect();

        let reranker = SpectralReranker::new(SpectralSelectConfig {
            min_candidates: 5,
            ..Default::default()
        });

        let r1 = reranker.rerank(&results, &embeddings, 4);
        let r2 = reranker.rerank(&results, &embeddings, 4);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_performance_200_candidates() {
        // 200 candidates, 384-dim (all-MiniLM-L6-v2), k=20 should complete quickly
        let n = 200;
        let dim = 384;
        let k = 20;

        let results: Vec<SearchResult> = (0..n)
            .map(|i| make_result(1.0 - i as f32 / n as f32))
            .collect();

        // Create pseudo-random embeddings deterministically
        let embeddings: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                    .collect()
            })
            .collect();

        let reranker = SpectralReranker::new(SpectralSelectConfig {
            min_candidates: 5,
            ..Default::default()
        });

        let start = std::time::Instant::now();
        let selected = reranker.rerank(&results, &embeddings, k);
        let elapsed = start.elapsed();

        assert_eq!(selected.len(), k);
        assert!(
            elapsed.as_millis() < 500,
            "Performance test: took {}ms, expected <500ms",
            elapsed.as_millis()
        );
    }

    // ── CrossEncoderReranker tests ────────────────────────────────────────

    #[test]
    fn test_cross_encoder_empty_input() {
        let r = CrossEncoderReranker::with_alpha(0.5, vec![1.0, 0.0]);
        assert!(r.rerank(&[], &[], 5).is_empty());
    }

    #[test]
    fn test_cross_encoder_k_zero() {
        let r = CrossEncoderReranker::with_alpha(0.5, vec![1.0, 0.0]);
        let results = vec![make_result(0.9)];
        let embeddings = vec![vec![1.0, 0.0]];
        assert!(r.rerank(&results, &embeddings, 0).is_empty());
    }

    #[test]
    fn test_cross_encoder_pure_cosine_alpha_zero() {
        // alpha=0.0 → pure cosine similarity.
        // query = [1, 0]; doc0 = [1, 0] (cos=1.0); doc1 = [0, 1] (cos=0.0)
        let query_emb = vec![1.0_f32, 0.0];
        let r = CrossEncoderReranker::with_alpha(0.0, query_emb);

        let results = vec![make_result(0.5), make_result(0.9)]; // doc1 has higher original score
        let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]; // doc0 aligned, doc1 orthogonal

        let selected = r.rerank(&results, &embeddings, 2);
        // doc0 should rank first (cos=1.0 > cos=0.0)
        assert_eq!(selected[0], 0);
    }

    #[test]
    fn test_cross_encoder_pure_original_alpha_one() {
        // alpha=1.0 → use original scores unchanged.
        let r = CrossEncoderReranker::with_alpha(1.0, vec![1.0, 0.0]);
        let results = vec![make_result(0.3), make_result(0.9), make_result(0.6)];
        let embeddings = vec![vec![0.0_f32, 1.0]; 3];
        let selected = r.rerank(&results, &embeddings, 2);
        // Should be score-descending: indices 1, 2
        assert_eq!(selected[0], 1); // score 0.9
        assert_eq!(selected[1], 2); // score 0.6
    }

    #[test]
    fn test_cross_encoder_blend_changes_ranking() {
        // With alpha=0.5 and a query aligned to doc0, doc0 should beat doc1
        // even though doc1 has a higher original score.
        let query_emb = vec![1.0_f32, 0.0];
        let r = CrossEncoderReranker::with_alpha(0.5, query_emb);
        // doc0: score=0.3, cos=1.0  → joint = 0.5*0.3 + 0.5*1.0 = 0.65
        // doc1: score=0.9, cos=0.0  → joint = 0.5*0.9 + 0.5*0.0 = 0.45
        let results = vec![make_result(0.3), make_result(0.9)];
        let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let selected = r.rerank(&results, &embeddings, 2);
        assert_eq!(selected[0], 0); // doc0 wins with blend
    }

    #[test]
    fn test_cross_encoder_empty_query_embedding_falls_back_to_score_order() {
        let r = CrossEncoderReranker::with_alpha(0.5, Vec::new());
        let results = vec![make_result(0.3), make_result(0.9), make_result(0.6)];
        let embeddings = vec![vec![1.0_f32, 0.0]; 3];
        let selected = r.rerank(&results, &embeddings, 2);
        assert_eq!(selected[0], 1); // highest original score
    }

    #[test]
    fn test_cross_encoder_k_gte_n_returns_all() {
        let r = CrossEncoderReranker::with_alpha(0.5, vec![1.0, 0.0]);
        let results = vec![make_result(0.8), make_result(0.5)];
        let embeddings = vec![vec![1.0_f32, 0.0]; 2];
        let selected = r.rerank(&results, &embeddings, 10);
        assert_eq!(selected.len(), 2);
    }
}
