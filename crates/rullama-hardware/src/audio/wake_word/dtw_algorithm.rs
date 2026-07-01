//! Dynamic Time Warping over arbitrary feature-vector sequences.
//!
//! The math is the classical Sakoe–Chiba banded DP from the 1970s speech
//! recognition literature. We use Euclidean distance for the local cost,
//! the three-neighbour (up / left / diagonal) accumulator, and a fixed
//! warp window so the comparison is bounded.
//!
//! Length-normalised — the return is the accumulated cost divided by the
//! warping path length so two sequences of different lengths get a fair
//! comparison.

/// Window width used by [`dtw_distance`] for the Sakoe–Chiba band.
///
/// Frames `i` and `j` are only compared when `|i - j| <= SAKOE_CHIBA_WINDOW`.
/// 10 frames at our default 10 ms hop = 100 ms of tolerated time warping,
/// which is roughly the amount a human can shift a short wake word.
pub const SAKOE_CHIBA_WINDOW: usize = 10;

/// Compute the DTW distance between two sequences of feature vectors.
///
/// Returns the length-normalised cost (sum-of-Euclidean-distances divided
/// by the matched-path length). Lower is more similar; identical sequences
/// return `0.0`.
///
/// Edge cases:
/// * If either sequence is empty, returns `f32::INFINITY` so a missing
///   template never accidentally fires.
/// * If both sequences are empty, returns `0.0` (vacuously identical).
/// * If the feature widths differ, returns `f32::INFINITY` — caller bug.
pub fn dtw_distance(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a.is_empty() || b.is_empty() {
        return f32::INFINITY;
    }
    if a[0].len() != b[0].len() {
        return f32::INFINITY;
    }

    let n = a.len();
    let m = b.len();
    let w = SAKOE_CHIBA_WINDOW.max(n.abs_diff(m));

    // `cost[i][j]` = best cumulative cost to reach (i, j) in the warping
    // matrix. We store the path length alongside so we can length-normalise
    // at the end.
    let mut cost = vec![vec![f32::INFINITY; m]; n];
    let mut steps = vec![vec![0usize; m]; n];

    cost[0][0] = euclid(&a[0], &b[0]);
    steps[0][0] = 1;

    // First row: can only come from the left.
    for j in 1..m.min(w + 1) {
        cost[0][j] = cost[0][j - 1] + euclid(&a[0], &b[j]);
        steps[0][j] = j + 1;
    }
    // First column: can only come from above.
    for i in 1..n.min(w + 1) {
        cost[i][0] = cost[i - 1][0] + euclid(&a[i], &b[0]);
        steps[i][0] = i + 1;
    }

    // Body of the matrix, restricted to the Sakoe-Chiba band.
    for i in 1..n {
        let j_lo = i.saturating_sub(w).max(1);
        let j_hi = (i + w + 1).min(m);
        for j in j_lo..j_hi {
            let d = euclid(&a[i], &b[j]);

            // Three predecessors. INFINITY entries can be left alone —
            // they correctly poison any path going through them.
            let (best_prev, best_steps) = best_predecessor(
                cost[i - 1][j],
                steps[i - 1][j],
                cost[i][j - 1],
                steps[i][j - 1],
                cost[i - 1][j - 1],
                steps[i - 1][j - 1],
            );

            if best_prev.is_finite() {
                cost[i][j] = best_prev + d;
                steps[i][j] = best_steps + 1;
            }
        }
    }

    let total = cost[n - 1][m - 1];
    let len = steps[n - 1][m - 1].max(1) as f32;
    if total.is_finite() {
        total / len
    } else {
        f32::INFINITY
    }
}

/// Euclidean distance between two equal-length feature vectors.
#[inline]
fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

/// Pick the cheapest of three predecessors and return its cost + step count.
#[inline]
fn best_predecessor(
    c_up: f32,
    s_up: usize,
    c_left: f32,
    s_left: usize,
    c_diag: f32,
    s_diag: usize,
) -> (f32, usize) {
    let mut best_c = c_up;
    let mut best_s = s_up;
    if c_left < best_c {
        best_c = c_left;
        best_s = s_left;
    }
    if c_diag < best_c {
        best_c = c_diag;
        best_s = s_diag;
    }
    (best_c, best_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(values: &[&[f32]]) -> Vec<Vec<f32>> {
        values.iter().map(|v| v.to_vec()).collect()
    }

    #[test]
    fn dtw_distance_identical_sequences_is_zero() {
        let s = seq(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let d = dtw_distance(&s, &s);
        assert!(
            d.abs() < 1e-5,
            "identical sequences must DTW to ~0, got {d}"
        );
    }

    #[test]
    fn dtw_distance_is_symmetric() {
        let a = seq(&[&[1.0], &[2.0], &[5.0], &[3.0]]);
        let b = seq(&[&[1.0], &[5.0], &[2.0], &[3.0]]);
        let ab = dtw_distance(&a, &b);
        let ba = dtw_distance(&b, &a);
        assert!(
            (ab - ba).abs() < 1e-4,
            "symmetry violated: dtw(a,b)={ab}, dtw(b,a)={ba}"
        );
    }

    #[test]
    fn dtw_distance_grows_with_drift() {
        // Baseline: identical short sequences.
        let base = seq(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let drift1 = seq(&[&[0.0], &[1.5], &[2.0], &[3.0]]);
        let drift2 = seq(&[&[0.0], &[2.0], &[2.5], &[4.0]]);
        let drift3 = seq(&[&[5.0], &[6.0], &[7.0], &[8.0]]);

        let d0 = dtw_distance(&base, &base);
        let d1 = dtw_distance(&base, &drift1);
        let d2 = dtw_distance(&base, &drift2);
        let d3 = dtw_distance(&base, &drift3);

        assert!(d0 < d1, "no-drift should beat slight drift ({d0} < {d1})");
        assert!(
            d1 < d2,
            "slight drift should beat moderate drift ({d1} < {d2})"
        );
        assert!(
            d2 < d3,
            "moderate drift should beat fully shifted ({d2} < {d3})"
        );
    }

    #[test]
    fn dtw_distance_short_warp_finds_match() {
        // A sequence and a 1-frame-shifted copy of itself (insert a dup
        // frame at the front). Per-frame-Euclidean over Hamming-like
        // sequences this should warp neatly and stay small.
        let base = seq(&[
            &[1.0, 0.0],
            &[2.0, 0.0],
            &[3.0, 0.0],
            &[4.0, 0.0],
            &[5.0, 0.0],
        ]);
        let mut shifted = vec![base[0].clone()];
        shifted.extend(base.iter().cloned());

        let warped = dtw_distance(&base, &shifted);
        // Unrelated baseline — completely different feature values.
        let unrelated = seq(&[
            &[10.0, 0.0],
            &[20.0, 0.0],
            &[30.0, 0.0],
            &[40.0, 0.0],
            &[50.0, 0.0],
        ]);
        let bad = dtw_distance(&base, &unrelated);

        // The whole point of DTW: a 1-frame shift should warp away to a near-
        // zero cost. (Mathematically, with a free duplicate-frame match, the
        // result can even hit exactly 0 — so we don't compare against
        // `dtw(base, base)`.)
        assert!(warped < 0.5, "1-frame warp should stay tiny, got {warped}");
        assert!(
            warped < bad / 4.0,
            "warped ({warped}) must be much less than unrelated ({bad})"
        );
    }

    #[test]
    fn dtw_handles_empty_sequences() {
        let empty: Vec<Vec<f32>> = vec![];
        let one = seq(&[&[1.0]]);
        assert_eq!(dtw_distance(&empty, &empty), 0.0);
        assert!(dtw_distance(&empty, &one).is_infinite());
        assert!(dtw_distance(&one, &empty).is_infinite());
    }

    #[test]
    fn dtw_handles_mismatched_widths() {
        let a = seq(&[&[1.0, 2.0]]);
        let b = seq(&[&[1.0, 2.0, 3.0]]);
        assert!(dtw_distance(&a, &b).is_infinite());
    }

    /// Direct DTW with a custom band width — exposed so the band-restriction
    /// test below can dial it down to a single frame and prove the band
    /// actually restricts the path. Mirrors `dtw_distance` exactly except
    /// the constant is parameterised.
    fn dtw_with_window(a: &[Vec<f32>], b: &[Vec<f32>], window: usize) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        if a.is_empty() || b.is_empty() {
            return f32::INFINITY;
        }
        if a[0].len() != b[0].len() {
            return f32::INFINITY;
        }
        let n = a.len();
        let m = b.len();
        let w = window.max(n.abs_diff(m));
        let mut cost = vec![vec![f32::INFINITY; m]; n];
        let mut steps = vec![vec![0usize; m]; n];
        cost[0][0] = euclid(&a[0], &b[0]);
        steps[0][0] = 1;
        for j in 1..m.min(w + 1) {
            cost[0][j] = cost[0][j - 1] + euclid(&a[0], &b[j]);
            steps[0][j] = j + 1;
        }
        for i in 1..n.min(w + 1) {
            cost[i][0] = cost[i - 1][0] + euclid(&a[i], &b[0]);
            steps[i][0] = i + 1;
        }
        for i in 1..n {
            let j_lo = i.saturating_sub(w).max(1);
            let j_hi = (i + w + 1).min(m);
            for j in j_lo..j_hi {
                let d = euclid(&a[i], &b[j]);
                let (best_prev, best_steps) = best_predecessor(
                    cost[i - 1][j],
                    steps[i - 1][j],
                    cost[i][j - 1],
                    steps[i][j - 1],
                    cost[i - 1][j - 1],
                    steps[i - 1][j - 1],
                );
                if best_prev.is_finite() {
                    cost[i][j] = best_prev + d;
                    steps[i][j] = best_steps + 1;
                }
            }
        }
        let total = cost[n - 1][m - 1];
        let len = steps[n - 1][m - 1].max(1) as f32;
        if total.is_finite() {
            total / len
        } else {
            f32::INFINITY
        }
    }

    #[test]
    fn sakoe_chiba_band_actually_restricts_search() {
        // Construct two sequences where the optimal warp requires a wide
        // jump (the matching content sits at different positions in each).
        // With a narrow band, the optimal path is unreachable; with a wide
        // band, it is. The cost difference proves the band is doing work.
        //
        // Sequence `a`: [low, low, low, low, low, HIGH, HIGH, HIGH]
        // Sequence `b`: [HIGH, HIGH, HIGH, low, low, low, low, low]
        // The natural alignment is `a[5..]` ↔ `b[..3]` and `a[..5]` ↔ `b[3..]`
        // — requires a path that drifts |i - j| up to ~5.
        let a = seq(&[
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[100.0],
            &[100.0],
            &[100.0],
        ]);
        let b = seq(&[
            &[100.0],
            &[100.0],
            &[100.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
        ]);
        let narrow = dtw_with_window(&a, &b, 1);
        let wide = dtw_with_window(&a, &b, 6);
        // The wide band can find a measurably better alignment than the
        // narrow band — they must not be equal, which would mean the band
        // wasn't restricting anything. (Magnitude depends on input; what
        // matters is the strict inequality.)
        assert!(
            wide < narrow,
            "wide band should beat narrow band on diagonal-shift inputs: \
             narrow={narrow}, wide={wide}",
        );
        // Sanity: both costs must be finite (the corner must be reachable
        // in both — the algorithm widens the band to at least |n-m| to
        // guarantee this).
        assert!(narrow.is_finite() && wide.is_finite());
    }
}
