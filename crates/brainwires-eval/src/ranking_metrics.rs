//! Ranking quality metrics for information retrieval evaluation.
//!
//! Pure functions operating on scores and ground-truth relevance labels.
//! No async, no external dependencies — safe to call from any eval case.
//!
//! ## Metrics
//!
//! | Function | What it measures |
//! |----------|-----------------|
//! | [`ndcg_at_k`] | Normalized Discounted Cumulative Gain — ranking quality with graded relevance |
//! | [`mrr`] | Mean Reciprocal Rank — rank of the first relevant result |
//! | [`precision_at_k`] | Fraction of top-K results that are relevant |
//!
//! ## Example
//!
//! ```rust
//! use brainwires_eval::{ndcg_at_k, mrr, precision_at_k};
//!
//! // scores[i] = system score for item i (higher = more relevant to system)
//! // relevance[i] = ground-truth relevance label for item i (0 = irrelevant)
//! let scores    = [0.9, 0.4, 0.7, 0.2];
//! let relevance = [3,   1,   2,   0  ];
//!
//! let ndcg = ndcg_at_k(&scores, &relevance, 4);
//! assert!(ndcg > 0.9, "should be near-perfect: highest score == highest relevance");
//! ```

/// Compute NDCG@K (Normalized Discounted Cumulative Gain).
///
/// Measures how well the system's ranking matches ground-truth relevance.
/// Returns 1.0 for a perfect ranking, 0.0 when no relevant items are returned.
///
/// # Arguments
/// * `scores` — system-assigned scores; higher = system considers more relevant
/// * `relevance` — ground-truth relevance labels (0 = irrelevant; higher = more relevant)
/// * `k` — cut-off depth; pass `0` to evaluate all items
///
/// # Panics
/// Panics if `scores.len() != relevance.len()`.
pub fn ndcg_at_k(scores: &[f64], relevance: &[usize], k: usize) -> f64 {
    assert_eq!(
        scores.len(),
        relevance.len(),
        "scores and relevance must have the same length"
    );
    if scores.is_empty() {
        return 0.0;
    }

    let n = scores.len();
    let cut = if k == 0 || k > n { n } else { k };

    // Sort by score descending (system ranking).
    let mut ranked: Vec<(f64, usize)> = scores
        .iter()
        .copied()
        .zip(relevance.iter().copied())
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    // DCG@K of the system ranking.
    let dcg: f64 = ranked[..cut]
        .iter()
        .enumerate()
        .map(|(i, (_, rel))| (2_f64.powi(*rel as i32) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    // IDCG@K — ideal ranking (sort by relevance descending).
    let mut ideal: Vec<usize> = relevance.to_vec();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal[..cut]
        .iter()
        .enumerate()
        .map(|(i, rel)| (2_f64.powi(*rel as i32) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        (dcg / idcg).clamp(0.0, 1.0)
    }
}

/// Compute MRR (Mean Reciprocal Rank).
///
/// Returns the reciprocal of the 1-based rank of the first relevant item
/// (any item with `relevance > 0`). Returns 0.0 if no relevant item exists.
///
/// # Panics
/// Panics if `scores.len() != relevance.len()`.
pub fn mrr(scores: &[f64], relevance: &[usize]) -> f64 {
    assert_eq!(
        scores.len(),
        relevance.len(),
        "scores and relevance must have the same length"
    );
    if scores.is_empty() {
        return 0.0;
    }

    let mut ranked: Vec<(f64, usize)> = scores
        .iter()
        .copied()
        .zip(relevance.iter().copied())
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    for (i, (_, rel)) in ranked.iter().enumerate() {
        if *rel > 0 {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// Compute Precision@K.
///
/// Returns the fraction of the top-K items that have `relevance > 0`.
///
/// # Arguments
/// * `scores` — system-assigned scores
/// * `relevance` — ground-truth relevance labels (0 = irrelevant)
/// * `k` — cut-off depth; pass `0` to evaluate all items
///
/// # Panics
/// Panics if `scores.len() != relevance.len()`.
pub fn precision_at_k(scores: &[f64], relevance: &[usize], k: usize) -> f64 {
    assert_eq!(
        scores.len(),
        relevance.len(),
        "scores and relevance must have the same length"
    );
    if scores.is_empty() {
        return 0.0;
    }

    let n = scores.len();
    let cut = if k == 0 || k > n { n } else { k };

    let mut ranked: Vec<(f64, usize)> = scores
        .iter()
        .copied()
        .zip(relevance.iter().copied())
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let relevant = ranked[..cut].iter().filter(|(_, rel)| *rel > 0).count();
    relevant as f64 / cut as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ndcg_at_k ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ndcg_perfect_ranking() {
        // System scores match relevance perfectly.
        let scores = [0.9, 0.7, 0.4, 0.1];
        let rel = [3, 2, 1, 0];
        let ndcg = ndcg_at_k(&scores, &rel, 4);
        assert!(
            (ndcg - 1.0).abs() < 1e-9,
            "perfect ranking should yield NDCG=1.0, got {ndcg}"
        );
    }

    #[test]
    fn test_ndcg_worst_ranking() {
        // System scores are the reverse of relevance.
        // With graded labels [3,2,1,0] and 4 positions the worst-case NDCG is
        // ~0.548 (not 0) because high-weight items still appear, just late.
        let scores = [0.1, 0.4, 0.7, 0.9];
        let rel = [3, 2, 1, 0];
        let ndcg = ndcg_at_k(&scores, &rel, 4);
        assert!(
            ndcg < 0.65 && ndcg > 0.0,
            "reversed graded ranking should give NDCG in (0, 0.65), got {ndcg}"
        );
        // Confirm it is strictly worse than perfect.
        let perfect = ndcg_at_k(&[0.9, 0.7, 0.4, 0.1], &rel, 4);
        assert!(ndcg < perfect, "reversed ranking must score below perfect");
    }

    #[test]
    fn test_ndcg_all_zero_relevance() {
        let scores = [0.9, 0.5, 0.1];
        let rel = [0, 0, 0];
        assert_eq!(ndcg_at_k(&scores, &rel, 3), 0.0);
    }

    #[test]
    fn test_ndcg_empty() {
        assert_eq!(ndcg_at_k(&[], &[], 0), 0.0);
    }

    #[test]
    fn test_ndcg_k_truncates() {
        // NDCG@K is not monotonic with K.
        //
        // When the top-K are perfectly ranked but deeper positions contain
        // highly-relevant items that were missed, IDCG grows (adding those
        // missed items to the ideal denominator) while DCG stays flat (the
        // system didn't retrieve them there), so NDCG@K > NDCG@(K+n).
        //
        // Scenario: 4 items, top-2 are both rel=2 (perfect). Item 3 is rel=0
        // (irrelevant, correctly ranked low). Item 4 is rel=2 (relevant but
        // missed — scored last).
        //   NDCG@2 = 1.0    (top-2 are ideal)
        //   NDCG@4 < 1.0    (IDCG grows by rel=2 item 4, but DCG misses it)
        let scores = [0.9, 0.7, 0.5, 0.3];
        let rel = [2, 2, 0, 2];
        let ndcg_k2 = ndcg_at_k(&scores, &rel, 2);
        let ndcg_k4 = ndcg_at_k(&scores, &rel, 4);
        assert!(
            (ndcg_k2 - 1.0).abs() < 1e-9,
            "top-2 perfectly ranked → NDCG@2 should be 1.0, got {ndcg_k2}"
        );
        assert!(
            ndcg_k4 < 1.0,
            "missed relevant item at rank 4 should lower NDCG@4, got {ndcg_k4}"
        );
        assert!(
            ndcg_k2 > ndcg_k4,
            "NDCG@2={ndcg_k2} should exceed NDCG@4={ndcg_k4}"
        );
    }

    #[test]
    fn test_ndcg_k_zero_means_all() {
        let scores = [0.9, 0.5];
        let rel = [2, 1];
        assert!((ndcg_at_k(&scores, &rel, 0) - ndcg_at_k(&scores, &rel, 2)).abs() < 1e-9);
    }

    // ── mrr ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mrr_first_is_relevant() {
        let scores = [0.9, 0.5, 0.1];
        let rel = [1, 0, 0];
        assert!((mrr(&scores, &rel) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_second_is_relevant() {
        let scores = [0.9, 0.5, 0.1];
        let rel = [0, 1, 0];
        assert!((mrr(&scores, &rel) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_no_relevant() {
        assert_eq!(mrr(&[0.9, 0.5], &[0, 0]), 0.0);
    }

    #[test]
    fn test_mrr_empty() {
        assert_eq!(mrr(&[], &[]), 0.0);
    }

    // ── precision_at_k ─────────────────────────────────────────────────────────

    #[test]
    fn test_precision_at_k_all_relevant() {
        let scores = [0.9, 0.7, 0.5];
        let rel = [1, 1, 1];
        assert!((precision_at_k(&scores, &rel, 3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_at_k_half_relevant() {
        let scores = [0.9, 0.8, 0.5, 0.1];
        let rel = [1, 1, 0, 0];
        assert!((precision_at_k(&scores, &rel, 4) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_precision_at_k_truncates() {
        // Top-2 are both relevant; bottom-2 are not.
        let scores = [0.9, 0.8, 0.3, 0.1];
        let rel = [1, 1, 0, 0];
        assert!((precision_at_k(&scores, &rel, 2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_at_k_zero_k_means_all() {
        let scores = [0.9, 0.5];
        let rel = [1, 0];
        assert!((precision_at_k(&scores, &rel, 0) - precision_at_k(&scores, &rel, 2)).abs() < 1e-9);
    }
}
