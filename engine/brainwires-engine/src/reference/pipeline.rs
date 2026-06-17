//! End-to-end Z-Image generation pipeline (CPU oracle): wires the three
//! validated component forwards + the flow-match scheduler into one
//! text-features → image path. Native-only (reads disk weights).
//!
//!   cap = Qwen3Encoder(tokens)                          [cap_len, 2560]
//!   latent = seeded N(0,1)                              [16, lh, lw]
//!   sched = FlowMatch(steps, dyn, calculate_shift(img_tokens))
//!   for s in 0..steps: v = DiT(latent, sigma[s], cap); latent += (σ'-σ)·v
//!   rgb = VAE.decode(latent)                            [3, lh·8, lw·8]
//!
//! Unconditional (no CFG): a single DiT forward per step, matching Z-Image with
//! no negative prompt. The Qwen2 tokenizer (prompt→ids) is a separate piece; a
//! caller passes token ids (or synthetic caption features) directly.

use crate::error::Result;
use crate::imagegen::config::{Qwen3Config, TransformerConfig, VaeConfig};
use crate::imagegen::scheduler::{FlowMatchScheduler, calculate_shift};
use crate::imagegen::sharded::ShardedSafetensors;
use crate::reference::dit::DitForward;
use crate::reference::qwen3::Qwen3Encoder;
use crate::reference::vae::VaeDecoder;

/// Per-component weights + configs for the pipeline.
pub struct Components<'a> {
    pub enc_st: &'a ShardedSafetensors,
    pub enc_cfg: &'a Qwen3Config,
    pub dit_st: &'a ShardedSafetensors,
    pub dit_cfg: &'a TransformerConfig,
    pub vae_st: &'a ShardedSafetensors,
    pub vae_cfg: &'a VaeConfig,
}

/// Optional per-stage progress callback `(stage, step, total)`.
pub type Progress<'a> = Option<&'a dyn Fn(&str, usize, usize)>;

/// Generate an RGB image `[3, lh·downscale, lw·downscale]` (values in [0,1])
/// from caption token ids. `lh`/`lw` are the latent dims (image / VAE 8×).
pub fn generate(
    c: &Components,
    tokens: &[u32],
    lh: usize,
    lw: usize,
    steps: usize,
    seed: u64,
    progress: Progress,
) -> Result<Vec<f32>> {
    let report = |stage: &str, i: usize, n: usize| {
        if let Some(p) = progress {
            p(stage, i, n);
        }
    };

    // 1. encode caption
    report("encode", 0, 1);
    let cap = Qwen3Encoder::new(c.enc_st, c.enc_cfg).forward(tokens)?;
    let cap_len = tokens.len();

    // 2. init latent noise (seeded, deterministic)
    let cin = c.dit_cfg.in_channels as usize;
    let mut latent = gaussian_noise(cin * lh * lw, seed);

    // 3. schedule (dynamic mu from image-token count, matching Ollama)
    let p = c.dit_cfg.patch_size() as usize;
    let img_tokens = (lh / p) * (lw / p);
    let sched = FlowMatchScheduler::new(steps, true, calculate_shift(img_tokens));

    // 4. denoise loop
    let dit = DitForward::new(c.dit_st, c.dit_cfg);
    for s in 0..steps {
        report("denoise", s, steps);
        let sigma = sched.sigma(s);
        let v = dit.forward(&latent, lh, lw, sigma, &cap, cap_len)?;
        sched.step_in_place(&mut latent, &v, s);
    }

    // 5. decode
    report("decode", 0, 1);
    VaeDecoder::new(c.vae_st, c.vae_cfg).decode(&latent, lh, lw)
}

/// Generate with classifier-free guidance (Z-Image's default, scale ~4.0):
/// each step runs the DiT on both the prompt and the negative prompt and
/// combines `v = v_neg + scale·(v_pos − v_neg)`. `scale == 1.0` reduces to the
/// unconditional [`generate`]. Doubles per-step DiT cost.
#[allow(clippy::too_many_arguments)]
pub fn generate_cfg(
    c: &Components,
    tokens: &[u32],
    neg_tokens: &[u32],
    cfg_scale: f32,
    lh: usize,
    lw: usize,
    steps: usize,
    seed: u64,
    progress: Progress,
) -> Result<Vec<f32>> {
    let report = |stage: &str, i: usize, n: usize| {
        if let Some(p) = progress {
            p(stage, i, n);
        }
    };
    let enc = Qwen3Encoder::new(c.enc_st, c.enc_cfg);
    report("encode", 0, 2);
    let cap = enc.forward(tokens)?;
    report("encode", 1, 2);
    let ncap = enc.forward(neg_tokens)?;

    let cin = c.dit_cfg.in_channels as usize;
    let mut latent = gaussian_noise(cin * lh * lw, seed);
    let p = c.dit_cfg.patch_size() as usize;
    let img_tokens = (lh / p) * (lw / p);
    let sched = FlowMatchScheduler::new(steps, true, calculate_shift(img_tokens));

    let dit = DitForward::new(c.dit_st, c.dit_cfg);
    for s in 0..steps {
        report("denoise", s, steps);
        let sigma = sched.sigma(s);
        let v_pos = dit.forward(&latent, lh, lw, sigma, &cap, tokens.len())?;
        let v_neg = dit.forward(&latent, lh, lw, sigma, &ncap, neg_tokens.len())?;
        let v = cfg_combine(&v_pos, &v_neg, cfg_scale);
        sched.step_in_place(&mut latent, &v, s);
    }
    report("decode", 0, 1);
    VaeDecoder::new(c.vae_st, c.vae_cfg).decode(&latent, lh, lw)
}

/// CFG combine: `v_neg + scale·(v_pos − v_neg)`. `scale == 1` ⇒ `v_pos`.
pub fn cfg_combine(v_pos: &[f32], v_neg: &[f32], scale: f32) -> Vec<f32> {
    v_pos
        .iter()
        .zip(v_neg)
        .map(|(&p, &n)| n + scale * (p - n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::cfg_combine;

    #[test]
    fn cfg_scale_one_is_positive() {
        let pos = [1.0f32, -2.0, 3.0];
        let neg = [0.5f32, 0.5, 0.5];
        assert_eq!(cfg_combine(&pos, &neg, 1.0), pos);
    }

    #[test]
    fn cfg_scale_zero_is_negative() {
        let pos = [1.0f32, 2.0];
        let neg = [9.0f32, -9.0];
        assert_eq!(cfg_combine(&pos, &neg, 0.0), neg);
    }

    #[test]
    fn cfg_pushes_away_from_negative() {
        // scale 4: v = neg + 4(pos-neg); for pos=1,neg=0 → 4.0
        assert_eq!(cfg_combine(&[1.0], &[0.0], 4.0), vec![4.0]);
    }
}

/// Deterministic `N(0,1)` vector via splitmix64 + Box–Muller (no rng dep, and
/// `Math.random`-free so it ports to wasm).
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let unit = |u: u64| ((u >> 11) as f64) / ((1u64 << 53) as f64); // [0,1)
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = unit(next_u64()).max(1e-12);
        let u2 = unit(next_u64());
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = std::f64::consts::TAU * u2;
        out.push((r * ang.cos()) as f32);
        if out.len() < n {
            out.push((r * ang.sin()) as f32);
        }
    }
    out
}
