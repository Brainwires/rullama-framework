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

/// adaLN modulation `y[t,c] = x[t,c] * (1 + scale[c]) + shift[c]`, matching
/// `kernels/wgsl/adaln_modulate.wgsl`. `x` is `[seq, hidden]` flat; `scale`
/// and `shift` are length-`hidden`, broadcast across tokens.
pub fn adaln_modulate(
    x: &[f32],
    seq: usize,
    hidden: usize,
    scale: &[f32],
    shift: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), seq * hidden);
    assert_eq!(scale.len(), hidden);
    assert_eq!(shift.len(), hidden);
    let mut out = vec![0.0f32; x.len()];
    for t in 0..seq {
        for c in 0..hidden {
            let i = t * hidden + c;
            out[i] = x[i] * (1.0 + scale[c]) + shift[c];
        }
    }
    out
}

/// Interleaved (GPT-J) RoPE over `[seq, heads, head_dim]`, in place, with
/// precomputed per-token `cos`/`sin` `[seq, head_dim/2]` (shared across heads).
/// Matches `kernels/wgsl/rope_interleaved.wgsl`.
pub fn rope_interleaved(
    x: &mut [f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * head_dim;
            for i in 0..half {
                let c = cos[t * half + i];
                let s = sin[t * half + i];
                let x1 = x[base + 2 * i];
                let x2 = x[base + 2 * i + 1];
                x[base + 2 * i] = x1 * c - x2 * s;
                x[base + 2 * i + 1] = x1 * s + x2 * c;
            }
        }
    }
}

/// Channel-first f32 conv2d (square kernel, stride 1, zero-pad), matching
/// `kernels/wgsl/conv2d_chw_f32.wgsl`. `x` is `[in_c,in_h,in_w]`, `weight`
/// `[out_c,in_c,k,k]`, `bias` `[out_c]`; returns `[out_c,out_h,out_w]` with
/// `out = in + 2*pad - k + 1`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_chw(
    x: &[f32],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
    k: usize,
    pad: usize,
) -> Vec<f32> {
    let out_h = in_h + 2 * pad - k + 1;
    let out_w = in_w + 2 * pad - k + 1;
    let mut y = vec![0.0f32; out_c * out_h * out_w];
    for co in 0..out_c {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = bias[co];
                let iy0 = oy as isize - pad as isize;
                let ix0 = ox as isize - pad as isize;
                for ci in 0..in_c {
                    let xb = ci * in_h * in_w;
                    let wb = (co * in_c + ci) * k * k;
                    for ky in 0..k {
                        let iy = iy0 + ky as isize;
                        if iy < 0 || iy >= in_h as isize {
                            continue;
                        }
                        for kx in 0..k {
                            let ix = ix0 + kx as isize;
                            if ix < 0 || ix >= in_w as isize {
                                continue;
                            }
                            acc +=
                                x[xb + iy as usize * in_w + ix as usize] * weight[wb + ky * k + kx];
                        }
                    }
                }
                y[(co * out_h + oy) * out_w + ox] = acc;
            }
        }
    }
    y
}

/// DiT gated residual add `x[t,c] += tanh(gate[c]) * branch[t,c]` over
/// `[seq,dim]` (gate length `dim`). Matches `kernels/wgsl/gated_residual_add.wgsl`.
pub fn gated_residual_add(x: &mut [f32], seq: usize, dim: usize, gate: &[f32], branch: &[f32]) {
    for t in 0..seq {
        for c in 0..dim {
            x[t * dim + c] += gate[c].tanh() * branch[t * dim + c];
        }
    }
}

/// 2D nearest 2× upsample, channel-first `[C,H,W] → [C,2H,2W]`.
/// Matches `kernels/wgsl/upsample2x_chw.wgsl`.
pub fn upsample2x_chw(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (h * 2, w * 2);
    let mut out = vec![0.0f32; c * h2 * w2];
    for ch in 0..c {
        for y in 0..h2 {
            for xx in 0..w2 {
                out[(ch * h2 + y) * w2 + xx] = x[(ch * h + y / 2) * w + xx / 2];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample2x_chw_doubles_both_axes() {
        // 1 channel, 1×2 → 2×4, each pixel replicated 2×2.
        let x = vec![1.0f32, 2.0];
        let y = upsample2x_chw(&x, 1, 1, 2);
        assert_eq!(y, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn conv2d_chw_identity_kernel() {
        // 1×1 kernel = 2.0, bias 0 → output = 2·input (pad 0).
        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // [1,2,2]
        let w = vec![2.0f32]; // [1,1,1,1]
        let y = conv2d_chw(&x, 1, 2, 2, &w, &[0.0], 1, 1, 0);
        assert_eq!(y, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn rope_interleaved_zero_angle_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        // seq=2, heads=1, hd=4, half=2; cos=1, sin=0 → identity
        let cos = vec![1.0f32; 4];
        let sin = vec![0.0f32; 4];
        rope_interleaved(&mut x, 2, 1, 4, &cos, &sin);
        assert_eq!(x, orig);
    }

    #[test]
    fn rope_interleaved_quarter_turn() {
        // single token/head, hd=2, angle=π/2 → cos0,sin1: (x1,x2)→(-x2, x1)
        let mut x = vec![1.0f32, 0.0];
        rope_interleaved(&mut x, 1, 1, 2, &[0.0], &[1.0]);
        assert!(
            (x[0] - 0.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
            "{x:?}"
        );
    }

    #[test]
    fn adaln_zero_scale_zero_shift_is_identity() {
        let x = vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0];
        let scale = vec![0.0; 3];
        let shift = vec![0.0; 3];
        assert_eq!(adaln_modulate(&x, 2, 3, &scale, &shift), x);
    }

    #[test]
    fn adaln_broadcasts_per_channel_across_tokens() {
        // 2 tokens, 2 channels. scale/shift differ by channel, same per token.
        let x = vec![1.0, 1.0, 2.0, 2.0];
        let scale = vec![1.0, 0.0]; // ch0 ×2, ch1 ×1
        let shift = vec![0.0, 5.0]; // ch1 +5
        let y = adaln_modulate(&x, 2, 2, &scale, &shift);
        assert_eq!(y, vec![2.0, 6.0, 4.0, 7.0]);
    }

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
