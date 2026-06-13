//! Entropy-bound block-diffusion sampler (DiffusionGemma).
//!
//! Mirrors `diffusion_generate_entropy_bound` in llama.cpp PR 24423
//! (`examples/diffusion/diffusion.cpp`) **1:1** — that runner is the parity
//! oracle (Ollama cannot run DiffusionGemma). Per step:
//!
//!   1. forward the whole canvas (bidirectional, prefix-KV) → per-position
//!      logits; the model applies self-conditioning internally from the
//!      previous step's raw logits / previous temperature
//!   2. per position: argmax, entropy of softmax(logits/t), one multinomial
//!      draw (CDF walk against a pre-drawn uniform)
//!   3. acceptance: walk positions from most- to least-confident, accepting
//!      while the sum of STRICTLY-EARLIER entropies ≤ `entropy_bound`
//!   4. renoise: accepted positions keep their sampled token, the rest get a
//!      fresh uniform-random token; the OUTPUT canvas is the argmax canvas
//!   5. stop once the argmax canvas has been stable `stability_threshold`
//!      steps AND mean entropy < `confidence_threshold` — or steps exhaust
//!
//! The temperature schedule is linear-descending: `t = t_min + (t_max-t_min)
//! · cur_step/S` for `cur_step = S..1`. The canvas is RANDOM-initialized
//! (uniform over vocab) — NOT mask-token-filled; `mask_token_id` is unused by
//! this path.
//!
//! Randomness is injected through [`SamplerRng`] so parity harnesses can
//! replay the exact `u`/`renoise` sequences captured from the reference
//! runner (C++ `std::mt19937` + `uniform_*_distribution` streams are not
//! portably reproducible across standard libraries).

use crate::error::Result;

/// Defaults match `diffusion_eb_params` in the PR (and Google's published
/// numbers): 48 steps max, temp 0.8→0.4, entropy budget 0.1, stop on 1 stable
/// step + mean entropy < 0.005.
#[derive(Clone, Debug)]
pub struct EbParams {
    pub max_denoising_steps: u32,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub stability_threshold: u32,
    pub confidence_threshold: f32,
}

impl Default for EbParams {
    fn default() -> Self {
        Self {
            max_denoising_steps: 48,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            stability_threshold: 1,
            confidence_threshold: 0.005,
        }
    }
}

/// Per-step randomness source. `uniform01` feeds the multinomial CDF walk;
/// `token` draws the renoise replacements and the initial canvas.
pub trait SamplerRng {
    fn uniform01(&mut self) -> f32;
    fn token(&mut self, n_vocab: u32) -> u32;
}

/// Deterministic xorshift-based default (NOT bit-compatible with the C++
/// mt19937 — use an injected replay rng for reference parity).
pub struct XorShiftRng(pub u64);

impl SamplerRng for XorShiftRng {
    fn uniform01(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn token(&mut self, n_vocab: u32) -> u32 {
        (self.uniform01() * n_vocab as f32) as u32 % n_vocab
    }
}

/// One denoising step's outcome, handed to the step callback.
pub struct StepInfo<'a> {
    pub step_idx: u32,
    pub total_steps: u32,
    /// The model's current best guess for every canvas position (argmax).
    pub argmax_canvas: &'a [u32],
    pub mean_entropy: f32,
    pub n_accepted: usize,
}

/// The per-step model evaluation the sampler drives. Returns the canvas
/// rows' raw logits, packed `[canvas_len × n_vocab]`. `prev` carries the
/// PREVIOUS step's raw canvas logits + 1/t for self-conditioning (None on
/// the first step — the SC gate is zeroed).
pub trait CanvasForward {
    fn forward(&mut self, canvas: &[u32], prev: Option<(&[f32], f32)>) -> Result<Vec<f32>>;
    fn n_vocab(&self) -> usize;
}

/// Run the entropy-bound denoise loop over one canvas. Returns the final
/// argmax canvas (length = `canvas_len`).
pub fn generate_entropy_bound(
    model: &mut dyn CanvasForward,
    canvas_len: usize,
    params: &EbParams,
    rng: &mut dyn SamplerRng,
    mut step_cb: Option<&mut dyn FnMut(&StepInfo) -> bool>,
) -> Result<Vec<u32>> {
    let n_vocab = model.n_vocab();
    let s_total = params.max_denoising_steps.max(1);

    // Random init — uniform over vocab, NOT the mask token.
    let mut current: Vec<u32> = (0..canvas_len).map(|_| rng.token(n_vocab as u32)).collect();

    let mut argmax_canvas = vec![0u32; canvas_len];
    let mut prev_argmax: Vec<u32> = vec![u32::MAX; canvas_len]; // step 0 is never "stable"
    let mut entropy = vec![0f32; canvas_len];
    let mut denoised = vec![0u32; canvas_len];
    let mut prev_logits: Option<Vec<f32>> = None;
    let mut prev_temp_inv = 1.0f32;
    let mut held = 0u32;

    for cur_step in (1..=s_total).rev() {
        let step_idx = s_total - cur_step;
        let t = params.t_min + (params.t_max - params.t_min) * (cur_step as f32 / s_total as f32);
        let temp_inv = 1.0 / t;

        let logits = model.forward(&current, prev_logits.as_deref().map(|l| (l, prev_temp_inv)))?;
        debug_assert_eq!(logits.len(), canvas_len * n_vocab);

        // Pre-draw the step's randomness in position order (seed-reproducible,
        // matching the reference's single-threaded pre-draw).
        let us: Vec<f32> = (0..canvas_len).map(|_| rng.uniform01()).collect();
        let renoise: Vec<u32> = (0..canvas_len).map(|_| rng.token(n_vocab as u32)).collect();

        // Per position: argmax, entropy of softmax(raw/t), multinomial sample.
        for pos in 0..canvas_len {
            let row = &logits[pos * n_vocab..(pos + 1) * n_vocab];
            let mut m = f32::NEG_INFINITY;
            let mut amax = 0usize;
            for (v, &z) in row.iter().enumerate() {
                let zt = z * temp_inv;
                if zt > m {
                    m = zt;
                    amax = v;
                }
            }
            let mut z_sum = 0f32;
            for &z in row {
                z_sum += (z * temp_inv - m).exp();
            }
            let target = us[pos] * z_sum;
            let mut cum = 0f32;
            let mut h = 0f32;
            let mut sampled = n_vocab - 1;
            let mut picked = false;
            for (v, &z) in row.iter().enumerate() {
                let e = (z * temp_inv - m).exp();
                let p = e / z_sum;
                if p > 0.0 {
                    h -= p * p.ln();
                }
                cum += e;
                if !picked && cum >= target {
                    sampled = v;
                    picked = true;
                }
            }
            entropy[pos] = h;
            argmax_canvas[pos] = amax as u32;
            denoised[pos] = sampled as u32;
        }

        // Acceptance: ascending entropy; accept while the sum of strictly-
        // earlier entropies stays within the bound.
        let mut order: Vec<usize> = (0..canvas_len).collect();
        order.sort_by(|&a, &b| entropy[a].partial_cmp(&entropy[b]).unwrap().then(a.cmp(&b)));
        let mut accepted = vec![false; canvas_len];
        let mut cum_e = 0f64;
        for &pos in &order {
            if cum_e <= params.entropy_bound as f64 {
                accepted[pos] = true;
            }
            cum_e += entropy[pos] as f64;
        }

        // Renoise + output canvas + stats.
        let mut entropy_sum = 0f32;
        let mut n_accepted = 0usize;
        for pos in 0..canvas_len {
            current[pos] = if accepted[pos] {
                n_accepted += 1;
                denoised[pos]
            } else {
                renoise[pos]
            };
            entropy_sum += entropy[pos];
        }

        // Adaptive stop: argmax stable AND confident.
        held = if prev_argmax == argmax_canvas {
            held + 1
        } else {
            0
        };
        let mean_entropy = entropy_sum / canvas_len as f32;
        let confident = mean_entropy < params.confidence_threshold;

        if let Some(cb) = step_cb.as_deref_mut() {
            let info = StepInfo {
                step_idx,
                total_steps: s_total,
                argmax_canvas: &argmax_canvas,
                mean_entropy,
                n_accepted,
            };
            if !cb(&info) {
                break;
            }
        }

        if held >= params.stability_threshold && confident {
            break;
        }
        prev_argmax.copy_from_slice(&argmax_canvas);
        prev_logits = Some(logits);
        prev_temp_inv = temp_inv;
    }

    Ok(argmax_canvas)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake model whose logits strongly prefer a fixed target sequence —
    /// the sampler must converge to it and stop early via the adaptive rule.
    struct FixedTarget {
        target: Vec<u32>,
        n_vocab: usize,
        calls: u32,
        saw_self_cond: bool,
    }
    impl CanvasForward for FixedTarget {
        fn forward(&mut self, canvas: &[u32], prev: Option<(&[f32], f32)>) -> Result<Vec<f32>> {
            assert_eq!(canvas.len(), self.target.len());
            if self.calls == 0 {
                assert!(prev.is_none(), "step 0 must not self-condition");
            } else {
                self.saw_self_cond |= prev.is_some();
            }
            self.calls += 1;
            let mut out = vec![0f32; canvas.len() * self.n_vocab];
            for (pos, &t) in self.target.iter().enumerate() {
                out[pos * self.n_vocab + t as usize] = 50.0; // ~one-hot after softmax
            }
            Ok(out)
        }
        fn n_vocab(&self) -> usize {
            self.n_vocab
        }
    }

    #[test]
    fn converges_to_confident_target_and_stops_early() {
        let target = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut model = FixedTarget {
            target: target.clone(),
            n_vocab: 11,
            calls: 0,
            saw_self_cond: false,
        };
        let mut rng = XorShiftRng(0x5EED);
        let mut steps_seen = 0u32;
        let out = generate_entropy_bound(
            &mut model,
            target.len(),
            &EbParams::default(),
            &mut rng,
            Some(&mut |info: &StepInfo| {
                steps_seen = info.step_idx + 1;
                true
            }),
        )
        .unwrap();
        assert_eq!(out, target, "argmax canvas must converge to the target");
        // Near-one-hot logits ⇒ tiny entropy ⇒ stable+confident at step 2
        // (step 0 can't be stable). Way below the 48-step cap.
        assert!(
            model.calls <= 4,
            "expected early stop, ran {} steps",
            model.calls
        );
        assert!(model.saw_self_cond, "later steps must pass prev logits");
    }

    /// Uniform logits ⇒ max entropy ⇒ nothing confident ⇒ runs to the cap.
    struct UniformModel {
        n_vocab: usize,
        calls: u32,
    }
    impl CanvasForward for UniformModel {
        fn forward(&mut self, canvas: &[u32], _prev: Option<(&[f32], f32)>) -> Result<Vec<f32>> {
            self.calls += 1;
            Ok(vec![0f32; canvas.len() * self.n_vocab])
        }
        fn n_vocab(&self) -> usize {
            self.n_vocab
        }
    }

    #[test]
    fn uniform_logits_run_to_step_cap() {
        let mut model = UniformModel {
            n_vocab: 7,
            calls: 0,
        };
        let mut rng = XorShiftRng(42);
        let params = EbParams {
            max_denoising_steps: 5,
            ..Default::default()
        };
        let _ = generate_entropy_bound(&mut model, 4, &params, &mut rng, None).unwrap();
        assert_eq!(model.calls, 5, "no early stop without confidence");
    }

    /// Acceptance budget: with entropies [0.0, 0.04, 0.05, 0.5] and bound 0.1,
    /// the walk accepts the first three (strictly-earlier sums 0, 0, 0.04+0=
    /// 0.04... then 0.09 ≤ 0.1) and rejects the last (0.09+0.05=0.14 > 0.1
    /// after the third). Verified through the public API by constructing
    /// logits with controlled per-position entropy.
    #[test]
    fn acceptance_respects_entropy_budget_ordering() {
        // Two-symbol vocab: H(p) tunes per-position entropy. p≈1 → H≈0;
        // p=0.5 → H=ln2≈0.69.
        struct TwoTok {
            confidences: Vec<f32>, // P(token 0)
        }
        impl CanvasForward for TwoTok {
            fn forward(&mut self, _c: &[u32], _p: Option<(&[f32], f32)>) -> Result<Vec<f32>> {
                let mut out = Vec::new();
                for &p in &self.confidences {
                    // logits (scaled by 1/temp later — entropies shift but
                    // ordering is preserved, which is what this test pins)
                    let l0 = (p / (1.0 - p)).ln();
                    out.extend_from_slice(&[l0, 0.0]);
                }
                Ok(out)
            }
            fn n_vocab(&self) -> usize {
                2
            }
        }
        let mut model = TwoTok {
            confidences: vec![0.999_999, 0.6, 0.999_99, 0.55],
        };
        let mut rng = XorShiftRng(7);
        let params = EbParams {
            max_denoising_steps: 1, // single step — inspect acceptance only
            ..Default::default()
        };
        let mut accepted_count = 0usize;
        let _ = generate_entropy_bound(
            &mut model,
            4,
            &params,
            &mut rng,
            Some(&mut |info: &StepInfo| {
                accepted_count = info.n_accepted;
                true
            }),
        )
        .unwrap();
        // Positions 0 and 2 are near-zero entropy (always accepted: the
        // strictly-earlier sum stays ~0); positions 1 and 3 carry ~0.6-0.7
        // nats each — the first of them is accepted (earlier sum ≈ 0 ≤ 0.1),
        // the second is not (earlier sum ≈ 0.65 > 0.1).
        assert_eq!(
            accepted_count, 3,
            "entropy budget must cut the least-confident position"
        );
    }
}
