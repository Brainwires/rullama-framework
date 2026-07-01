//! Relevance-weighted Gram matrix construction for spectral subset selection.
//!
//! Builds the kernel matrix `L` where:
//! ```text
//! L_ij = (r_i^lambda) * (r_j^lambda) * cos_sim(v_i, v_j)
//! ```
//! This encodes both relevance (via score weighting) and diversity (via embedding
//! similarity) into a single positive semi-definite matrix whose log-determinant
//! measures the "volume" spanned by the selected subset.

use ndarray::Array2;

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Build the relevance-weighted kernel matrix.
///
/// # Arguments
///
/// * `embeddings` - Slice of embedding vectors (one per candidate)
/// * `scores` - Relevance scores for each candidate (e.g., fused RRF scores)
/// * `lambda` - Trade-off parameter: 0.0 = pure diversity, 1.0 = relevance-dominated
/// * `regularization` - Small epsilon added to diagonal for numerical stability
///
/// # Returns
///
/// An n*n symmetric positive-definite kernel matrix.
pub fn build_kernel_matrix(
    embeddings: &[&[f32]],
    scores: &[f32],
    lambda: f32,
    regularization: f32,
) -> Array2<f32> {
    let n = embeddings.len();
    debug_assert_eq!(n, scores.len());

    let mut kernel = Array2::<f32>::zeros((n, n));

    // Precompute relevance weights: r_i^lambda
    let weights: Vec<f32> = scores
        .iter()
        .map(|&s| {
            // Clamp score to [0, 1] to avoid NaN from negative bases
            let s_clamped = s.clamp(0.0, 1.0);
            s_clamped.powf(lambda)
        })
        .collect();

    for i in 0..n {
        for j in i..n {
            let sim = cosine_similarity(embeddings[i], embeddings[j]);
            // Clamp similarity to [0, 1] -- negative cosine similarity means
            // the items are anti-correlated, treat as zero similarity for DPP
            let sim_clamped = sim.max(0.0);
            let val = weights[i] * weights[j] * sim_clamped;
            kernel[[i, j]] = val;
            kernel[[j, i]] = val;
        }
        // Diagonal regularization for numerical stability
        kernel[[i, i]] += regularization;
    }

    kernel
}

/// Extract the submatrix `L_S` for a given set of indices.
pub fn submatrix(kernel: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let k = indices.len();
    let mut sub = Array2::<f32>::zeros((k, k));
    for (i, &row) in indices.iter().enumerate() {
        for (j, &col) in indices.iter().enumerate() {
            sub[[i, j]] = kernel[[row, col]];
        }
    }
    sub
}

/// Extract the cross-kernel column between selected set S and candidate c.
///
/// Returns `[L_{s_0, c}, L_{s_1, c}, ..., L_{s_{m-1}, c}]`.
pub fn cross_column(kernel: &Array2<f32>, selected: &[usize], candidate: usize) -> Vec<f32> {
    selected.iter().map(|&s| kernel[[s, candidate]]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_kernel_matrix_shape() {
        let e1 = vec![1.0, 0.0];
        let e2 = vec![0.0, 1.0];
        let e3 = vec![1.0, 1.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2, &e3];
        let scores = vec![0.9, 0.8, 0.7];

        let kernel = build_kernel_matrix(&embeddings, &scores, 0.5, 1e-6);
        assert_eq!(kernel.nrows(), 3);
        assert_eq!(kernel.ncols(), 3);
    }

    #[test]
    fn test_kernel_matrix_symmetric() {
        let e1 = vec![1.0, 2.0, 3.0];
        let e2 = vec![4.0, 5.0, 6.0];
        let e3 = vec![7.0, 8.0, 9.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2, &e3];
        let scores = vec![0.9, 0.8, 0.7];

        let kernel = build_kernel_matrix(&embeddings, &scores, 0.5, 1e-6);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (kernel[[i, j]] - kernel[[j, i]]).abs() < 1e-6,
                    "not symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_kernel_lambda_zero_ignores_relevance() {
        // With lambda=0, all weights become 1.0, so the kernel is purely cosine similarity
        let e1 = vec![1.0, 0.0];
        let e2 = vec![0.0, 1.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2];
        let scores = vec![0.1, 0.9]; // Very different scores

        let kernel = build_kernel_matrix(&embeddings, &scores, 0.0, 0.0);
        // Off-diagonal should be 0 (orthogonal vectors)
        assert!(kernel[[0, 1]].abs() < 1e-6);
        // Diagonal should be 1.0 (cos_sim of vector with itself)
        assert!((kernel[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((kernel[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_submatrix_extraction() {
        let e1 = vec![1.0, 0.0];
        let e2 = vec![0.0, 1.0];
        let e3 = vec![1.0, 1.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2, &e3];
        let scores = vec![0.9, 0.8, 0.7];

        let kernel = build_kernel_matrix(&embeddings, &scores, 0.5, 1e-6);
        let sub = submatrix(&kernel, &[0, 2]);

        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 2);
        assert!((sub[[0, 0]] - kernel[[0, 0]]).abs() < 1e-6);
        assert!((sub[[0, 1]] - kernel[[0, 2]]).abs() < 1e-6);
        assert!((sub[[1, 0]] - kernel[[2, 0]]).abs() < 1e-6);
        assert!((sub[[1, 1]] - kernel[[2, 2]]).abs() < 1e-6);
    }

    #[test]
    fn test_cross_column() {
        let e1 = vec![1.0, 0.0];
        let e2 = vec![0.0, 1.0];
        let e3 = vec![1.0, 1.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2, &e3];
        let scores = vec![0.9, 0.8, 0.7];

        let kernel = build_kernel_matrix(&embeddings, &scores, 0.5, 1e-6);
        let cross = cross_column(&kernel, &[0, 1], 2);

        assert_eq!(cross.len(), 2);
        assert!((cross[0] - kernel[[0, 2]]).abs() < 1e-6);
        assert!((cross[1] - kernel[[1, 2]]).abs() < 1e-6);
    }
}
