//! CPU f32 oracles for the image-generation engine's WGSL kernels.
//!
//! Same role as `reference::kokoro::ops` for the TTS path: a slow, obviously-
//! correct reference each new image-gen kernel is parity-tested against
//! (GPU-vs-CPU ≤ ~1e-4). Grows as IM2/IM3 land (adaLN modulation, conv/VAE
//! stages); GroupNorm is the first.

/// GroupNorm over a single image in channel-contiguous `[C, H*W]` layout,
/// matching `kernels/wgsl/groupnorm.wgsl`.
///
/// Channels `C = n_groups * chans_per_grp`; `x.len() == C * hw`. Mean/variance
/// are computed over each group's `chans_per_grp * hw` elements; the optional
/// affine is per-channel (`gamma`/`beta` length `C`). Uses the biased
/// (population) variance, like the GPU kernel.
pub fn group_norm(
    x: &[f32],
    n_groups: usize,
    chans_per_grp: usize,
    hw: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    let grp_elems = chans_per_grp * hw;
    assert_eq!(
        x.len(),
        n_groups * grp_elems,
        "x len {} != n_groups({n_groups}) * chans_per_grp({chans_per_grp}) * hw({hw})",
        x.len()
    );
    let mut out = vec![0.0f32; x.len()];
    for g in 0..n_groups {
        let base = g * grp_elems;
        let block = &x[base..base + grp_elems];

        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for &v in block {
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
        }
        let n = grp_elems as f64;
        let mean = sum / n;
        let var = sumsq / n - mean * mean;
        let inv = 1.0 / (var + eps as f64).sqrt();

        for (k, &v) in block.iter().enumerate() {
            let mut nv = ((v as f64 - mean) * inv) as f32;
            if let (Some(gm), Some(bt)) = (gamma, beta) {
                let c = g * chans_per_grp + k / hw; // absolute channel
                nv = nv * gm[c] + bt[c];
            }
            out[base + k] = nv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_norm_one_group_is_layernorm_over_all() {
        // n_groups=1 → normalize the whole tensor; mean≈0, unit var after.
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let y = group_norm(&x, 1, 2, 8, None, None, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-4, "mean {mean}");
        let var: f32 = y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn group_norm_groups_are_independent() {
        // Two groups with very different scales each normalize to ~unit var.
        let mut x = vec![0.0f32; 2 * 1 * 4]; // 2 groups, 1 chan, hw=4
        x[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        x[4..].copy_from_slice(&[100.0, 200.0, 300.0, 400.0]);
        let y = group_norm(&x, 2, 1, 4, None, None, 1e-5);
        for g in 0..2 {
            let blk = &y[g * 4..g * 4 + 4];
            let m: f32 = blk.iter().sum::<f32>() / 4.0;
            assert!(m.abs() < 1e-4);
        }
    }

    #[test]
    fn affine_is_per_channel() {
        // 1 group, 2 channels, hw=2; gamma scales each channel distinctly.
        let x = vec![0.0, 1.0, 10.0, 11.0];
        let gamma = [2.0, 3.0];
        let beta = [0.5, -0.5];
        let y = group_norm(&x, 1, 2, 2, Some(&gamma), Some(&beta), 1e-5);
        // channel 0 uses gamma[0]/beta[0]; channel 1 uses gamma[1]/beta[1]
        let raw = group_norm(&x, 1, 2, 2, None, None, 1e-5);
        assert!((y[0] - (raw[0] * 2.0 + 0.5)).abs() < 1e-5);
        assert!((y[2] - (raw[2] * 3.0 - 0.5)).abs() < 1e-5);
    }
}
