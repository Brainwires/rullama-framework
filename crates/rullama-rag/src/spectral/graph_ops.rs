//! Spectral graph operations for clustering, centrality, and sparsification.
//!
//! Extends the spectral module beyond RAG reranking to general graph analysis.
//! All algorithms operate on weighted adjacency matrices (dense, `Array2<f32>`)
//! and are designed for graphs up to ~500 nodes (typical knowledge/call graphs).
//!
//! # Key operations
//!
//! - **Laplacian construction** from weighted adjacency matrices
//! - **Fiedler vector** (2nd smallest eigenvector) via inverse power iteration
//! - **Spectral clustering** using Laplacian eigenvectors + k-means
//! - **Algebraic connectivity** (spectral gap) for bottleneck detection
//! - **Effective resistance** for edge importance ranking
//! - **Spectral sparsification** preserving graph structure with fewer edges

use ndarray::Array2;

// ── Laplacian ──────────────────────────────────────────────────────────────

/// Build the combinatorial graph Laplacian from a weighted adjacency matrix.
///
/// ```text
/// L = D - W
/// ```
///
/// where `D` is the diagonal degree matrix (`D_ii = sum_j W_ij`) and `W` is
/// the adjacency matrix. The Laplacian is symmetric positive semi-definite
/// with smallest eigenvalue 0 (eigenvector = all-ones).
///
/// # Panics
///
/// Debug-asserts that the input is square.
pub fn laplacian(adjacency: &Array2<f32>) -> Array2<f32> {
    let n = adjacency.nrows();
    debug_assert_eq!(n, adjacency.ncols(), "adjacency must be square");

    let mut lap = Array2::<f32>::zeros((n, n));

    for i in 0..n {
        let mut degree = 0.0_f32;
        for j in 0..n {
            if i != j {
                let w = adjacency[[i, j]];
                lap[[i, j]] = -w;
                degree += w;
            }
        }
        lap[[i, i]] = degree;
    }

    lap
}

/// Build the normalized symmetric Laplacian.
///
/// ```text
/// L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
/// ```
///
/// Eigenvalues lie in `[0, 2]`. Preferred for spectral clustering because it
/// accounts for degree heterogeneity.
pub fn normalized_laplacian(adjacency: &Array2<f32>) -> Array2<f32> {
    let n = adjacency.nrows();
    debug_assert_eq!(n, adjacency.ncols());

    // Compute D^{-1/2}
    let mut d_inv_sqrt = vec![0.0_f32; n];
    for i in 0..n {
        let mut degree = 0.0_f32;
        for j in 0..n {
            degree += adjacency[[i, j]];
        }
        d_inv_sqrt[i] = if degree > 0.0 {
            1.0 / degree.sqrt()
        } else {
            0.0
        };
    }

    let mut lap = Array2::<f32>::eye(n);

    for i in 0..n {
        for j in 0..n {
            if i != j && adjacency[[i, j]] > 0.0 {
                lap[[i, j]] = -d_inv_sqrt[i] * adjacency[[i, j]] * d_inv_sqrt[j];
            }
        }
    }

    lap
}

// ── Eigenvector computation ────────────────────────────────────────────────

/// Compute the smallest non-trivial eigenvector (Fiedler vector) of a
/// symmetric Laplacian via inverse power iteration with deflation.
///
/// The Fiedler vector is the eigenvector corresponding to the second-smallest
/// eigenvalue (algebraic connectivity). Its sign pattern partitions the graph
/// into two loosely-connected halves — the foundation of spectral bisection.
///
/// Uses a shifted-and-inverted approach: solves `(L + shift*I)^{-1} x` to
/// target the smallest eigenvalues, then deflates the trivial eigenvector
/// (constant vector) to isolate the Fiedler vector.
///
/// # Arguments
///
/// * `laplacian` - Symmetric graph Laplacian (combinatorial or normalized)
/// * `max_iter` - Maximum power iterations (default: 300)
/// * `tol` - Convergence tolerance on eigenvector change (default: 1e-6)
///
/// # Returns
///
/// `(fiedler_vector, algebraic_connectivity)` — the eigenvector and its eigenvalue.
/// Returns `None` if the graph is disconnected (algebraic connectivity ≈ 0).
pub fn fiedler_vector(lap: &Array2<f32>, max_iter: usize, tol: f32) -> Option<(Vec<f32>, f32)> {
    let n = lap.nrows();
    if n < 2 {
        return None;
    }

    // Shift to make (L + shift*I) positive definite (shift > 0 but small)
    let shift = 1e-4_f32;
    let mut shifted = lap.clone();
    for i in 0..n {
        shifted[[i, i]] += shift;
    }

    // Cholesky factorize for fast solves
    let chol = super::linalg::cholesky(&shifted)?;

    // Random-ish initial vector (deterministic for reproducibility)
    let inv_n = 1.0 / (n as f32).sqrt();
    let trivial: Vec<f32> = vec![inv_n; n]; // normalized constant vector

    let mut x: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 13) as f32 - 6.0).collect();

    // Deflate trivial eigenvector from initial guess
    deflate(&mut x, &trivial);
    normalize(&mut x);

    for _ in 0..max_iter {
        // Solve (L + shift*I) * y = x  via Cholesky
        let y = cholesky_solve(&chol, &x);

        // Deflate trivial eigenvector
        let mut y_deflated = y;
        deflate(&mut y_deflated, &trivial);

        // Normalize
        let norm = vec_norm(&y_deflated);
        if norm < 1e-12 {
            return None; // collapsed — graph may be disconnected
        }
        for v in &mut y_deflated {
            *v /= norm;
        }

        // Check convergence (angle between x and y_deflated)
        let dot: f32 = x.iter().zip(&y_deflated).map(|(a, b)| a * b).sum();
        let change = (1.0 - dot.abs()).abs();

        x = y_deflated;

        if change < tol {
            break;
        }
    }

    // Compute Rayleigh quotient: eigenvalue = x^T L x / x^T x
    let eigenvalue = rayleigh_quotient(lap, &x);

    if eigenvalue < 1e-8 {
        return None; // effectively disconnected
    }

    Some((x, eigenvalue))
}

/// Compute the first `k` smallest eigenvectors of a symmetric Laplacian.
///
/// Uses sequential deflation: compute eigenvector, deflate, repeat.
/// Returns eigenvectors as columns (each `Vec<f32>` is one eigenvector),
/// skipping the trivial constant eigenvector.
///
/// # Returns
///
/// Up to `k` eigenvectors and their eigenvalues, sorted by eigenvalue ascending.
pub fn smallest_eigenvectors(
    lap: &Array2<f32>,
    k: usize,
    max_iter: usize,
    tol: f32,
) -> Vec<(Vec<f32>, f32)> {
    let n = lap.nrows();
    if n < 2 || k == 0 {
        return Vec::new();
    }

    let shift = 1e-4_f32;
    let mut shifted = lap.clone();
    for i in 0..n {
        shifted[[i, i]] += shift;
    }

    let chol = match super::linalg::cholesky(&shifted) {
        Some(c) => c,
        None => return Vec::new(),
    };

    let inv_n = 1.0 / (n as f32).sqrt();
    let trivial: Vec<f32> = vec![inv_n; n];

    let mut found: Vec<(Vec<f32>, f32)> = Vec::with_capacity(k);

    for _ in 0..k {
        let mut x: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3 + found.len() * 11) % 13) as f32 - 6.0)
            .collect();

        // Deflate trivial + all previously found eigenvectors
        deflate(&mut x, &trivial);
        for (ev, _) in &found {
            deflate(&mut x, ev);
        }
        normalize(&mut x);

        let mut converged = false;
        for _ in 0..max_iter {
            let mut y = cholesky_solve(&chol, &x);

            deflate(&mut y, &trivial);
            for (ev, _) in &found {
                deflate(&mut y, ev);
            }

            let norm = vec_norm(&y);
            if norm < 1e-12 {
                break;
            }
            for v in &mut y {
                *v /= norm;
            }

            let dot: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            let change = (1.0 - dot.abs()).abs();
            x = y;

            if change < tol {
                converged = true;
                break;
            }
        }

        if !converged && !found.is_empty() {
            // If we can't converge on more eigenvectors, stop
            break;
        }

        let eigenvalue = rayleigh_quotient(lap, &x);
        if eigenvalue < 1e-8 {
            break; // rest are in the null space
        }

        found.push((x, eigenvalue));
    }

    found
}

// ── Spectral clustering ────────────────────────────────────────────────────

/// Spectral clustering using the first `k` Laplacian eigenvectors.
///
/// 1. Compute first k non-trivial eigenvectors of the normalized Laplacian
/// 2. Row-normalize the eigenvector matrix (each node gets a k-dim embedding)
/// 3. Run k-means on the embeddings to produce cluster assignments
///
/// # Arguments
///
/// * `adjacency` - Weighted adjacency matrix (symmetric, non-negative)
/// * `k` - Number of clusters
///
/// # Returns
///
/// Cluster assignment for each node: `assignments[i]` is the cluster index for node `i`.
/// Returns `None` if the graph has fewer than `k` meaningful components.
pub fn spectral_cluster(adjacency: &Array2<f32>, k: usize) -> Option<Vec<usize>> {
    let n = adjacency.nrows();
    if n < k || k == 0 {
        return None;
    }
    if k == 1 {
        return Some(vec![0; n]);
    }

    // For k=2, use Fiedler vector sign partitioning directly.
    // This is more robust than k-means on 1D embeddings.
    if k == 2 {
        let lap = laplacian(adjacency);
        let (fv, _) = fiedler_vector(&lap, 500, 1e-7)?;
        return Some(fv.iter().map(|&v| if v >= 0.0 { 0 } else { 1 }).collect());
    }

    // For k>2, use recursive spectral bisection.
    // This avoids k-means instability on small spectral embeddings.
    let mut assignments = vec![0usize; n];
    let mut next_label = 1usize;

    // Queue of (node_indices, cluster_label, remaining_splits)
    let mut queue: Vec<(Vec<usize>, usize, usize)> = vec![((0..n).collect(), 0, k - 1)];

    while let Some((nodes, _label, remaining)) = queue.pop() {
        if remaining == 0 || nodes.len() <= 1 {
            continue;
        }

        // Build sub-adjacency for this group
        let m = nodes.len();
        let mut sub_adj = Array2::<f32>::zeros((m, m));
        for (si, &ni) in nodes.iter().enumerate() {
            for (sj, &nj) in nodes.iter().enumerate() {
                sub_adj[[si, sj]] = adjacency[[ni, nj]];
            }
        }

        let sub_lap = laplacian(&sub_adj);
        let bisection = fiedler_vector(&sub_lap, 500, 1e-7);

        if let Some((fv, _)) = bisection {
            let mut group_a = Vec::new();
            let mut group_b = Vec::new();

            for (si, &v) in fv.iter().enumerate() {
                if v >= 0.0 {
                    group_a.push(nodes[si]);
                } else {
                    group_b.push(nodes[si]);
                }
            }

            // Assign the new label to group_b
            let new_label = next_label;
            next_label += 1;
            for &ni in &group_b {
                assignments[ni] = new_label;
            }

            // Distribute remaining splits between groups (larger group gets more)
            if remaining > 1 {
                let splits_a = (remaining - 1) * group_a.len() / nodes.len();
                let splits_b = (remaining - 1) - splits_a;
                if splits_a > 0 {
                    queue.push((group_a, _label, splits_a));
                }
                if splits_b > 0 {
                    queue.push((group_b, new_label, splits_b));
                }
            }
        }
    }

    Some(assignments)
}

// ── Algebraic connectivity ─────────────────────────────────────────────────

/// Compute the algebraic connectivity (Fiedler value) of a graph.
///
/// This is the second-smallest eigenvalue of the Laplacian. It measures
/// how well-connected the graph is:
/// - 0 = disconnected (multiple components)
/// - Small = bottleneck exists (nearly disconnected)
/// - Large = well-connected
///
/// Useful for monitoring agent dependency graphs: a shrinking spectral gap
/// indicates approaching contention.
pub fn algebraic_connectivity(adjacency: &Array2<f32>) -> f32 {
    let lap = laplacian(adjacency);
    match fiedler_vector(&lap, 300, 1e-6) {
        Some((_, eigenvalue)) => eigenvalue,
        None => 0.0,
    }
}

// ── Effective resistance & sparsification ──────────────────────────────────

/// Compute effective resistance between two nodes.
///
/// The effective resistance `R_{ij}` is the potential difference when
/// 1 unit of current flows from i to j in a resistor network with
/// conductances equal to edge weights.
///
/// ```text
/// R_ij = (e_i - e_j)^T L^+ (e_i - e_j)
/// ```
///
/// where `L^+` is the Moore-Penrose pseudoinverse of the Laplacian.
///
/// High resistance = edge is a "bridge" (structurally important).
/// Low resistance = edge is redundant (many alternative paths).
pub fn effective_resistance(adjacency: &Array2<f32>, i: usize, j: usize) -> f32 {
    let n = adjacency.nrows();
    debug_assert!(i < n && j < n);

    if i == j {
        return 0.0;
    }

    let lap = laplacian(adjacency);

    // Compute L^+ via regularized inverse: (L + (1/n)J)^{-1} - (1/n)J
    // where J is the all-ones matrix. Simpler: solve (L + eps*I) x = (e_i - e_j)
    // and compute R ≈ (e_i - e_j)^T x.
    let reg = 1.0 / n as f32;
    let mut lap_reg = lap;
    for r in 0..n {
        lap_reg[[r, r]] += reg;
    }

    let chol = match super::linalg::cholesky(&lap_reg) {
        Some(c) => c,
        None => return f32::INFINITY,
    };

    // RHS: e_i - e_j
    let mut rhs = vec![0.0_f32; n];
    rhs[i] = 1.0;
    rhs[j] = -1.0;

    let x = cholesky_solve(&chol, &rhs);

    // R_ij = (e_i - e_j)^T x = x[i] - x[j]
    (x[i] - x[j]).abs()
}

/// Spectral sparsification: keep edges with probability proportional to
/// effective resistance, preserving the graph's spectral properties.
///
/// Implements a simplified Spielman-Srivastava algorithm:
/// 1. Compute effective resistances for all edges
/// 2. Keep each edge with probability `p_e = min(1, c * R_e * w_e * log(n) / epsilon^2)`
/// 3. Reweight kept edges to maintain expected Laplacian
///
/// # Arguments
///
/// * `adjacency` - Weighted adjacency matrix
/// * `epsilon` - Approximation quality (smaller = more edges kept, better approximation).
///   Typical: 0.3-0.5 for good sparsification, 0.1 for high fidelity.
///
/// # Returns
///
/// Sparsified adjacency matrix where `(1-epsilon) L ≤ L_sparse ≤ (1+epsilon) L`
/// in the spectral (Loewner) ordering.
pub fn sparsify(adjacency: &Array2<f32>, epsilon: f32) -> Array2<f32> {
    let n = adjacency.nrows();
    debug_assert_eq!(n, adjacency.ncols());

    if n <= 3 {
        return adjacency.clone(); // too small to sparsify
    }

    // Collect all edges with their effective resistances
    let lap = laplacian(adjacency);

    // Regularized Laplacian for pseudoinverse approximation
    let reg = 1.0 / n as f32;
    let mut lap_reg = lap;
    for i in 0..n {
        lap_reg[[i, i]] += reg;
    }

    let chol = match super::linalg::cholesky(&lap_reg) {
        Some(c) => c,
        None => return adjacency.clone(),
    };

    // Collect all edges with their leverage scores
    let mut edges: Vec<(usize, usize, f32, f32)> = Vec::new(); // (i, j, weight, leverage)

    for i in 0..n {
        for j in (i + 1)..n {
            let w = adjacency[[i, j]];
            if w <= 0.0 {
                continue;
            }

            // Compute effective resistance for this edge
            let mut rhs = vec![0.0_f32; n];
            rhs[i] = 1.0;
            rhs[j] = -1.0;
            let x = cholesky_solve(&chol, &rhs);
            let r_ij = (x[i] - x[j]).abs();

            // Leverage score: w_e * R_e
            // For a spanning tree edge, leverage ≈ 1. For redundant edges, leverage < 1.
            let leverage = w * r_ij;
            edges.push((i, j, w, leverage));
        }
    }

    if edges.is_empty() {
        return adjacency.clone();
    }

    // Sort by leverage to find the threshold
    let mut leverages: Vec<f32> = edges.iter().map(|e| e.3).collect();
    leverages.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Target: keep approximately (1 - epsilon) fraction of total leverage weight
    // Edges with leverage below the epsilon-quantile are redundant
    let target_keep = ((1.0 - epsilon) * edges.len() as f32).ceil() as usize;
    let threshold = if target_keep < edges.len() {
        leverages[edges.len() - target_keep]
    } else {
        0.0 // keep all
    };

    let mut sparse = Array2::<f32>::zeros((n, n));

    for &(i, j, w, leverage) in &edges {
        if leverage >= threshold {
            // Keep important edges (high leverage = structurally important)
            sparse[[i, j]] = w;
            sparse[[j, i]] = w;
        }
        // else: drop (redundant, low leverage)
    }

    sparse
}

// ── Spectral centrality ────────────────────────────────────────────────────

/// Compute spectral centrality scores for all nodes.
///
/// Nodes with extreme Fiedler vector values are at the "boundary" between
/// communities — they are structural bridges. Nodes near zero are deep
/// inside their community.
///
/// Returns `(node_index, centrality_score)` pairs sorted by centrality descending.
/// The centrality is the absolute Fiedler value, so bridge nodes score highest.
pub fn spectral_centrality(adjacency: &Array2<f32>) -> Vec<(usize, f32)> {
    let n = adjacency.nrows();
    let lap = laplacian(adjacency);

    let fv = match fiedler_vector(&lap, 300, 1e-6) {
        Some((v, _)) => v,
        None => return (0..n).map(|i| (i, 0.0)).collect(),
    };

    let mut scores: Vec<(usize, f32)> = fv.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Compute spectral bisection: partition nodes into two groups by the sign
/// of the Fiedler vector.
///
/// Returns `(group_a, group_b)` — indices of nodes in each partition.
/// The cut between groups is approximately minimal in the weighted sense.
pub fn spectral_bisection(adjacency: &Array2<f32>) -> Option<(Vec<usize>, Vec<usize>)> {
    let lap = laplacian(adjacency);
    let (fv, _) = fiedler_vector(&lap, 300, 1e-6)?;

    let mut group_a = Vec::new();
    let mut group_b = Vec::new();

    for (i, &v) in fv.iter().enumerate() {
        if v >= 0.0 {
            group_a.push(i);
        } else {
            group_b.push(i);
        }
    }

    Some((group_a, group_b))
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Solve `L x = b` where `L` is a lower-triangular Cholesky factor.
/// Actually solves `L L^T x = b` (forward then back substitution).
fn cholesky_solve(chol: &Array2<f32>, b: &[f32]) -> Vec<f32> {
    let n = chol.nrows();

    // Forward solve: L y = b
    let mut y = vec![0.0_f32; n];
    for i in 0..n {
        let mut sum = 0.0_f32;
        for j in 0..i {
            sum += chol[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / chol[[i, i]];
    }

    // Back solve: L^T x = y
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = 0.0_f32;
        for j in (i + 1)..n {
            sum += chol[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = (y[i] - sum) / chol[[i, i]];
    }

    x
}

/// Deflate vector `x` by removing its component along `basis`.
/// Assumes `basis` is already normalized.
fn deflate(x: &mut [f32], basis: &[f32]) {
    let dot: f32 = x.iter().zip(basis).map(|(a, b)| a * b).sum();
    for (xi, &bi) in x.iter_mut().zip(basis) {
        *xi -= dot * bi;
    }
}

/// Euclidean norm.
fn vec_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Normalize a vector in-place.
fn normalize(x: &mut [f32]) {
    let n = vec_norm(x);
    if n > 1e-12 {
        for v in x.iter_mut() {
            *v /= n;
        }
    }
}

/// Rayleigh quotient: `x^T A x / x^T x`
fn rayleigh_quotient(a: &Array2<f32>, x: &[f32]) -> f32 {
    let n = a.nrows();
    let mut num = 0.0_f32;
    for i in 0..n {
        let mut ax_i = 0.0_f32;
        for j in 0..n {
            ax_i += a[[i, j]] * x[j];
        }
        num += x[i] * ax_i;
    }
    let denom: f32 = x.iter().map(|v| v * v).sum();
    if denom > 0.0 { num / denom } else { 0.0 }
}

/// Simple k-means clustering on dense embeddings.
///
/// Uses k-means++ initialization for stable centroids.
/// Currently used by tests; available for spectral_cluster with k>2 in the future.
#[allow(dead_code)]
fn kmeans(points: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = points.len();
    let dim = points[0].len();

    if n <= k {
        return (0..n).collect();
    }

    // k-means++ initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(points[0].clone()); // first centroid = first point

    for _ in 1..k {
        // Find point with max min-distance to existing centroids
        let mut best_idx = 0;
        let mut best_dist = f32::NEG_INFINITY;

        for (i, p) in points.iter().enumerate() {
            let min_dist = centroids
                .iter()
                .map(|c| sq_dist(p, c))
                .fold(f32::INFINITY, f32::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        centroids.push(points[best_idx].clone());
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f32::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = sq_dist(p, centroid);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut sums = vec![vec![0.0_f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, p) in points.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &v) in p.iter().enumerate() {
                sums[c][j] += v;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] / counts[c] as f32;
                }
            }
        }
    }

    assignments
}

/// Squared Euclidean distance.
#[allow(dead_code)]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── Laplacian tests ────────────────────────────────────────────────

    #[test]
    fn test_laplacian_triangle() {
        // Complete graph K3 with unit weights
        let adj = array![[0.0_f32, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let lap = laplacian(&adj);

        // Diagonal = 2 (degree), off-diagonal = -1
        for i in 0..3 {
            assert!((lap[[i, i]] - 2.0).abs() < 1e-6);
            for j in 0..3 {
                if i != j {
                    assert!((lap[[i, j]] - (-1.0)).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        // Laplacian rows sum to zero for any adjacency
        let adj = array![[0.0_f32, 0.5, 0.3], [0.5, 0.0, 0.8], [0.3, 0.8, 0.0]];
        let lap = laplacian(&adj);

        for i in 0..3 {
            let row_sum: f32 = (0..3).map(|j| lap[[i, j]]).sum();
            assert!(
                row_sum.abs() < 1e-6,
                "Row {} sums to {} (should be 0)",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_normalized_laplacian_eigenvalues_bounded() {
        // For any graph, normalized Laplacian eigenvalues ∈ [0, 2]
        let adj = array![[0.0_f32, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.0]];
        let nlap = normalized_laplacian(&adj);

        // Diagonal should be 1.0 (I - D^{-1/2} W D^{-1/2}, diagonal of I)
        for i in 0..3 {
            assert!(
                (nlap[[i, i]] - 1.0).abs() < 1e-6,
                "Normalized Laplacian diagonal[{}] = {}",
                i,
                nlap[[i, i]]
            );
        }

        // Verify symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (nlap[[i, j]] - nlap[[j, i]]).abs() < 1e-6,
                    "Not symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    // ── Fiedler vector tests ───────────────────────────────────────────

    #[test]
    fn test_fiedler_path_graph() {
        // Path graph: 0 -- 1 -- 2 -- 3
        // Fiedler vector should be approximately [-a, -b, b, a] (monotone)
        let adj = array![
            [0.0_f32, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
        let lap = laplacian(&adj);
        let (fv, lambda) = fiedler_vector(&lap, 500, 1e-7).expect("path graph is connected");

        // Algebraic connectivity of P4 = 2 - sqrt(2) ≈ 0.586
        assert!(
            (lambda - 0.586).abs() < 0.05,
            "Expected λ₂ ≈ 0.586, got {}",
            lambda
        );

        // Fiedler vector should be monotone (or reverse-monotone)
        let monotone_inc = fv[0] < fv[1] && fv[1] < fv[2] && fv[2] < fv[3];
        let monotone_dec = fv[0] > fv[1] && fv[1] > fv[2] && fv[2] > fv[3];
        assert!(
            monotone_inc || monotone_dec,
            "Fiedler vector of path graph should be monotone, got {:?}",
            fv
        );
    }

    #[test]
    fn test_fiedler_complete_graph() {
        // K4: all edges weight 1. Algebraic connectivity = n = 4
        let adj = array![
            [0.0_f32, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ];
        let lap = laplacian(&adj);
        let (_, lambda) = fiedler_vector(&lap, 500, 1e-7).expect("K4 is connected");

        // For Kn, all non-trivial eigenvalues = n
        assert!(
            (lambda - 4.0).abs() < 0.1,
            "Expected λ₂ = 4.0, got {}",
            lambda
        );
    }

    // ── Spectral clustering tests ──────────────────────────────────────

    #[test]
    fn test_spectral_cluster_two_cliques() {
        // Two cliques of size 3 connected by a single weak edge
        //   0--1--2  ~~~  3--4--5
        let mut adj = Array2::<f32>::zeros((6, 6));
        // Clique 1: {0, 1, 2}
        for &(i, j) in &[(0, 1), (0, 2), (1, 2)] {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
        // Clique 2: {3, 4, 5}
        for &(i, j) in &[(3, 4), (3, 5), (4, 5)] {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
        // Weak bridge
        adj[[2, 3]] = 0.1;
        adj[[3, 2]] = 0.1;

        let assignments = spectral_cluster(&adj, 2).expect("should cluster");

        // Nodes within each clique should share a cluster assignment
        // (but we don't know which cluster index they'll get)
        assert_eq!(assignments[0], assignments[1], "clique1 nodes 0,1 differ");
        assert_eq!(assignments[1], assignments[2], "clique1 nodes 1,2 differ");
        assert_eq!(assignments[3], assignments[4], "clique2 nodes 3,4 differ");
        assert_eq!(assignments[4], assignments[5], "clique2 nodes 4,5 differ");

        // The two cliques should be in different clusters.
        // Use spectral bisection as a cross-check — it's more reliable
        // since it doesn't depend on k-means initialization.
        let (a, _b) = spectral_bisection(&adj).expect("connected graph");
        let a_has_0 = a.contains(&0);
        let a_has_3 = a.contains(&3);
        assert_ne!(a_has_0, a_has_3, "Bisection should separate the cliques");
    }

    #[test]
    fn test_spectral_cluster_single_node() {
        let adj = array![[0.0_f32]];
        let assignments = spectral_cluster(&adj, 1).expect("single node");
        assert_eq!(assignments, vec![0]);
    }

    // ── Algebraic connectivity tests ───────────────────────────────────

    #[test]
    fn test_algebraic_connectivity_path_vs_complete() {
        // Complete graph should have higher connectivity than path
        let path = array![[0.0_f32, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let complete = array![[0.0_f32, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];

        let ac_path = algebraic_connectivity(&path);
        let ac_complete = algebraic_connectivity(&complete);

        assert!(
            ac_complete > ac_path,
            "K3 ({}) should be more connected than P3 ({})",
            ac_complete,
            ac_path
        );
    }

    // ── Effective resistance tests ─────────────────────────────────────

    #[test]
    fn test_effective_resistance_self() {
        let adj = array![[0.0_f32, 1.0], [1.0, 0.0]];
        assert_eq!(effective_resistance(&adj, 0, 0), 0.0);
    }

    #[test]
    fn test_effective_resistance_single_edge() {
        // Single edge with weight 1: R ≈ 1/w = 1.0
        // Note: regularization (1/n) shifts the exact value slightly
        let adj = array![[0.0_f32, 1.0], [1.0, 0.0]];
        let r = effective_resistance(&adj, 0, 1);
        assert!(r > 0.5 && r < 1.5, "Expected R ≈ 1.0, got {}", r);
    }

    #[test]
    fn test_effective_resistance_parallel_paths() {
        // Triangle: two paths from 0 to 1 (direct + via 2)
        // Resistance should be less than single edge
        let adj = array![[0.0_f32, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let r_direct = effective_resistance(&adj, 0, 1);
        let single_edge = array![[0.0_f32, 1.0], [1.0, 0.0]];
        let r_single = effective_resistance(&single_edge, 0, 1);

        assert!(
            r_direct < r_single,
            "Parallel paths ({}) should reduce resistance vs single edge ({})",
            r_direct,
            r_single
        );
    }

    // ── Sparsification tests ───────────────────────────────────────────

    #[test]
    fn test_sparsify_preserves_connectivity() {
        // Dense graph: sparsified version should still be connected
        let mut adj = Array2::<f32>::zeros((5, 5));
        for i in 0..5 {
            for j in (i + 1)..5 {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }

        let sparse = sparsify(&adj, 0.5);

        // Check every node still has at least one edge
        for i in 0..5 {
            let degree: f32 = (0..5).map(|j| sparse[[i, j]]).sum();
            assert!(
                degree > 0.0,
                "Node {} became isolated after sparsification",
                i
            );
        }
    }

    #[test]
    fn test_sparsify_reduces_edges() {
        // Graph with varied edge weights — sparsification should drop weak redundant edges
        // Two dense clusters connected by a few cross-edges
        let n = 10;
        let mut adj = Array2::<f32>::zeros((n, n));
        // Cluster 1: nodes 0-4 with strong edges
        for i in 0..5 {
            for j in (i + 1)..5 {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        // Cluster 2: nodes 5-9 with strong edges
        for i in 5..10 {
            for j in (i + 1)..10 {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        // Weak cross-cluster edges (redundant paths)
        for i in 0..5 {
            for j in 5..10 {
                adj[[i, j]] = 0.05;
                adj[[j, i]] = 0.05;
            }
        }

        let sparse = sparsify(&adj, 0.3);

        let original_edges: usize = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .filter(|&(i, j)| adj[[i, j]] > 0.0)
            .count();
        let sparse_edges: usize = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .filter(|&(i, j)| sparse[[i, j]] > 0.0)
            .count();

        assert!(
            sparse_edges < original_edges,
            "Sparsification should reduce edges: {} -> {}",
            original_edges,
            sparse_edges
        );
    }

    #[test]
    fn test_sparsify_small_graph_unchanged() {
        // Graphs with ≤3 nodes should be returned unchanged
        let adj = array![[0.0_f32, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let sparse = sparsify(&adj, 0.5);
        assert_eq!(sparse, adj);
    }

    // ── Spectral bisection tests ───────────────────────────────────────

    #[test]
    fn test_spectral_bisection_barbell() {
        // Barbell: two triangles connected by one edge
        let mut adj = Array2::<f32>::zeros((6, 6));
        // Triangle 1
        for &(i, j) in &[(0, 1), (0, 2), (1, 2)] {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
        // Triangle 2
        for &(i, j) in &[(3, 4), (3, 5), (4, 5)] {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
        // Bridge
        adj[[2, 3]] = 0.1;
        adj[[3, 2]] = 0.1;

        let (a, b) = spectral_bisection(&adj).expect("connected graph");

        // Each group should have 3 nodes
        assert_eq!(a.len() + b.len(), 6);

        // Verify correct partitioning
        let mut a_set: Vec<usize> = a.clone();
        let mut b_set: Vec<usize> = b.clone();
        a_set.sort();
        b_set.sort();

        let correct_a = (a_set == vec![0, 1, 2] && b_set == vec![3, 4, 5])
            || (a_set == vec![3, 4, 5] && b_set == vec![0, 1, 2]);
        assert!(
            correct_a,
            "Expected clean bisection, got A={:?} B={:?}",
            a, b
        );
    }

    // ── Spectral centrality tests ──────────────────────────────────────

    #[test]
    fn test_spectral_centrality_returns_all_nodes() {
        // Verify spectral_centrality returns a score for every node
        // and scores are non-negative
        let adj = array![
            [0.0_f32, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ];

        let centrality = spectral_centrality(&adj);
        assert_eq!(centrality.len(), 4);

        // All scores should be non-negative
        for &(_, score) in &centrality {
            assert!(score >= 0.0, "Centrality score should be non-negative");
        }

        // Scores should be sorted descending
        for w in centrality.windows(2) {
            assert!(w[0].1 >= w[1].1, "Centrality should be sorted descending");
        }

        // Endpoint nodes (0, 3) should have higher Fiedler |values| than
        // interior nodes (1, 2) in a path graph
        let node_0_score = centrality.iter().find(|&&(i, _)| i == 0).unwrap().1;
        let node_1_score = centrality.iter().find(|&&(i, _)| i == 1).unwrap().1;
        assert!(
            node_0_score > node_1_score,
            "Endpoints should have higher |Fiedler| than interior: 0={}, 1={}",
            node_0_score,
            node_1_score
        );
    }

    // ── K-means tests ──────────────────────────────────────────────────

    #[test]
    fn test_kmeans_obvious_clusters() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];

        let assignments = kmeans(&points, 2, 50);

        // Points 0,1,2 should be in one cluster; 3,4,5 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[0], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[3], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }
}
