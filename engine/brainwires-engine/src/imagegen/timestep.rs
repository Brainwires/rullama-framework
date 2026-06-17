//! Sinusoidal timestep embedding — the host-side front-end of the DiT's
//! timestep embedder (the learned `mlp.0 → SiLU → mlp.2` that follows runs on
//! the GPU with model weights).
//!
//! Exact port of Ollama `x/imagegen/models/zimage/transformer.go`
//! `TimestepEmbedder.Forward`:
//!
//!   half     = dim / 2                          (dim = 256 for Z-Image)
//!   freqs[i] = exp(-ln(10000) * i / half)       i in 0..half
//!   args[i]  = t * freqs[i]
//!   emb      = concat([cos(args), sin(args)])   (cos FIRST — flip_sin_to_cos)
//!
//! Note the denominator is `half` (not `half-1`) and there is NO
//! downscale-frequency shift — getting this exact matters because the learned
//! MLP weights expect this precise layout.

/// Build the `dim`-length sinusoidal embedding for scalar timestep `t`.
/// `dim` must be even; output is `[cos(args)… , sin(args)…]`.
pub fn sinusoidal_timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    assert!(dim.is_multiple_of(2), "timestep embed dim must be even, got {dim}");
    let half = dim / 2;
    let ln_max = (10000.0f64).ln();
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-ln_max * (i as f64) / (half as f64)).exp();
        let arg = (t as f64) * freq;
        out[i] = arg.cos() as f32; // cos half first
        out[half + i] = arg.sin() as f32; // sin half second
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_zero_is_cos_ones_then_sin_zeros() {
        let e = sinusoidal_timestep_embedding(0.0, 8);
        assert_eq!(e.len(), 8);
        for v in &e[..4] {
            assert!((v - 1.0).abs() < 1e-6); // cos(0) = 1
        }
        for v in &e[4..] {
            assert!(v.abs() < 1e-6); // sin(0) = 0
        }
    }

    #[test]
    fn freq0_is_unit_so_leading_entries_are_cos_sin_t() {
        // freqs[0] = exp(0) = 1 ⇒ out[0]=cos(t), out[half]=sin(t).
        let t = 1.3f32;
        let dim = 16;
        let e = sinusoidal_timestep_embedding(t, dim);
        assert!((e[0] - t.cos()).abs() < 1e-6);
        assert!((e[dim / 2] - t.sin()).abs() < 1e-6);
    }

    #[test]
    fn matches_hand_computed_dim4() {
        // half=2, freqs=[1, 10000^-0.5=0.01], t=2 → args=[2, 0.02]
        let e = sinusoidal_timestep_embedding(2.0, 4);
        let expect = [
            2.0f64.cos() as f32,
            0.02f64.cos() as f32,
            2.0f64.sin() as f32,
            0.02f64.sin() as f32,
        ];
        for (g, x) in e.iter().zip(expect) {
            assert!((g - x).abs() < 1e-6, "{g} vs {x}");
        }
    }
}
