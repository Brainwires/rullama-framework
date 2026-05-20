//! Linear algebra utilities for spectral subset selection.
//!
//! Provides Cholesky decomposition and log-determinant computation
//! for small matrices (up to ~20x20), using pure ndarray with no
//! external LAPACK dependency.

use ndarray::Array2;

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor `L` such that `A = L * L^T`.
///
/// # Errors
///
/// Returns `None` if the matrix is not positive definite (a diagonal element
/// of `L` would be non-positive).
pub fn cholesky(a: &Array2<f32>) -> Option<Array2<f32>> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols(), "Cholesky requires a square matrix");

    let mut l = Array2::<f32>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f32;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return None; // not positive definite
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Some(l)
}

/// Compute `log det(A)` via Cholesky decomposition.
///
/// ```text
/// log det(A) = 2 * sum log(diag(chol(A)))
/// ```
///
/// Returns `f32::NEG_INFINITY` if the matrix is not positive definite.
pub fn log_det(a: &Array2<f32>) -> f32 {
    match cholesky(a) {
        Some(l) => {
            let n = l.nrows();
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum += l[[i, i]].ln();
            }
            2.0 * sum
        }
        None => f32::NEG_INFINITY,
    }
}

/// Compute `log det(A_{S union {c}})` incrementally given `log det(A_S)` and the
/// Cholesky factor of `A_S`.
///
/// This avoids recomputing the full Cholesky for every candidate and is the
/// standard rank-1 update trick.  For a selected set S of size m, adding
/// candidate c gives an (m+1)x(m+1) matrix:
///
/// ```text
/// A' = [ A_S    a ]
///      [ a^T    d ]
/// ```
///
/// where `a = L_S^{-1} * cross` (cross = column of kernel values between S and c),
/// and `d = kernel(c,c)`.
///
/// `log det(A') = log det(A_S) + log(d - ||a||^2)`   (the Schur complement).
///
/// Returns `f32::NEG_INFINITY` if the Schur complement is non-positive.
pub fn log_det_incremental(
    chol_s: &Array2<f32>,
    cross: &[f32],
    diag_cc: f32,
    current_log_det: f32,
) -> f32 {
    let m = chol_s.nrows();
    debug_assert_eq!(cross.len(), m);

    // Forward-solve L * a = cross  (L is lower triangular)
    let mut a = vec![0.0_f32; m];
    for i in 0..m {
        let mut sum = 0.0_f32;
        for j in 0..i {
            sum += chol_s[[i, j]] * a[j];
        }
        a[i] = (cross[i] - sum) / chol_s[[i, i]];
    }

    let norm_sq: f32 = a.iter().map(|x| x * x).sum();
    let schur = diag_cc - norm_sq;

    if schur <= 0.0 {
        return f32::NEG_INFINITY;
    }

    current_log_det + schur.ln()
}

/// Extend a Cholesky factor by one row/column (rank-1 update).
///
/// Given `L` (m*m lower-triangular Cholesky of `A_S`), `cross` (kernel column
/// between S and c), and `diag_cc` (kernel(c,c)), returns the (m+1)*(m+1)
/// Cholesky factor of `A_{S union {c}}`.
///
/// Returns `None` if the Schur complement is non-positive.
pub fn cholesky_extend(chol_s: &Array2<f32>, cross: &[f32], diag_cc: f32) -> Option<Array2<f32>> {
    let m = chol_s.nrows();
    debug_assert_eq!(cross.len(), m);

    // Forward-solve L * a = cross
    let mut a = vec![0.0_f32; m];
    for i in 0..m {
        let mut sum = 0.0_f32;
        for j in 0..i {
            sum += chol_s[[i, j]] * a[j];
        }
        a[i] = (cross[i] - sum) / chol_s[[i, i]];
    }

    let norm_sq: f32 = a.iter().map(|x| x * x).sum();
    let schur = diag_cc - norm_sq;

    if schur <= 0.0 {
        return None;
    }

    let new_diag = schur.sqrt();
    let new_size = m + 1;
    let mut new_l = Array2::<f32>::zeros((new_size, new_size));

    // Copy existing factor
    for i in 0..m {
        for j in 0..=i {
            new_l[[i, j]] = chol_s[[i, j]];
        }
    }

    // New row
    for j in 0..m {
        new_l[[m, j]] = a[j];
    }
    new_l[[m, m]] = new_diag;

    Some(new_l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cholesky_identity() {
        let eye = Array2::<f32>::eye(3);
        let l = cholesky(&eye).expect("identity is positive definite");
        // L should also be identity
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((l[[i, j]] - 1.0).abs() < 1e-6);
                } else {
                    assert!(l[[i, j]].abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_cholesky_known_matrix() {
        // A = [[4, 2], [2, 3]]
        // L = [[2, 0], [1, sqrt(2)]]
        let a = array![[4.0_f32, 2.0], [2.0, 3.0]];
        let l = cholesky(&a).expect("known positive definite");
        assert!((l[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((l[[1, 0]] - 1.0).abs() < 1e-6);
        assert!((l[[1, 1]] - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!(l[[0, 1]].abs() < 1e-6);
    }

    #[test]
    fn test_log_det_known() {
        // det([[4, 2], [2, 3]]) = 12 - 4 = 8
        let a = array![[4.0_f32, 2.0], [2.0, 3.0]];
        let ld = log_det(&a);
        assert!((ld - 8.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_log_det_identity() {
        let eye = Array2::<f32>::eye(4);
        let ld = log_det(&eye);
        assert!(ld.abs() < 1e-6); // log(1) = 0
    }

    #[test]
    fn test_not_positive_definite() {
        // Negative diagonal
        let a = array![[-1.0_f32, 0.0], [0.0, 1.0]];
        assert!(cholesky(&a).is_none());
        assert!(log_det(&a) == f32::NEG_INFINITY);
    }

    #[test]
    fn test_cholesky_extend_matches_full() {
        // Build a 3x3 PD matrix, compute Cholesky of 2x2 sub, then extend
        let full = array![[4.0_f32, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];

        let sub = array![[4.0_f32, 2.0], [2.0, 5.0]];
        let chol_sub = cholesky(&sub).unwrap();

        let cross = vec![1.0_f32, 3.0]; // full[2, 0..2]
        let diag_cc = 6.0_f32; // full[2, 2]

        let extended = cholesky_extend(&chol_sub, &cross, diag_cc).unwrap();
        let full_chol = cholesky(&full).unwrap();

        for i in 0..3 {
            for j in 0..=i {
                assert!(
                    (extended[[i, j]] - full_chol[[i, j]]).abs() < 1e-5,
                    "mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    extended[[i, j]],
                    full_chol[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_log_det_incremental_matches_full() {
        let full = array![[4.0_f32, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];

        let sub = array![[4.0_f32, 2.0], [2.0, 5.0]];
        let chol_sub = cholesky(&sub).unwrap();
        let ld_sub = log_det(&sub);

        let cross = vec![1.0_f32, 3.0];
        let diag_cc = 6.0_f32;

        let ld_incr = log_det_incremental(&chol_sub, &cross, diag_cc, ld_sub);
        let ld_full = log_det(&full);

        assert!(
            (ld_incr - ld_full).abs() < 1e-4,
            "incremental {} vs full {}",
            ld_incr,
            ld_full
        );
    }
}
