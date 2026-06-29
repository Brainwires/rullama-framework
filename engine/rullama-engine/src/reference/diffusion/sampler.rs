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

/// One ingested step's stats (the loop-driver / JS surface reads these to
/// decide whether to keep going and what to render).
pub struct StepOutcome {
    pub step_idx: u32,
    pub total_steps: u32,
    pub mean_entropy: f32,
    pub n_accepted: usize,
    /// Adaptive-stop fired (argmax stable + confident) OR the step budget is
    /// exhausted — no more forwards needed.
    pub done: bool,
}

/// Resumable entropy-bound denoise state — the loop body of
/// [`generate_entropy_bound`] split so an async / JS-driven caller can run one
/// step at a time (the GPU forward is `async` on wasm, so it can't be hidden
/// behind the sync [`CanvasForward`] trait). Native callers use the trait-based
/// driver below; the wasm `denoiseStep` surface holds a `DenoiseState` and
/// calls [`DenoiseState::forward_inputs`] → its own async forward →
/// [`DenoiseState::ingest`].
pub struct DenoiseState {
    params: EbParams,
    canvas_len: usize,
    n_vocab: usize,
    s_total: u32,
    /// Counts DOWN from `s_total` to 1; reaching 0 means the budget is spent.
    cur_step: u32,
    current: Vec<u32>,
    argmax_canvas: Vec<u32>,
    prev_argmax: Vec<u32>,
    prev_logits: Option<Vec<f32>>,
    prev_temp_inv: f32,
    held: u32,
    finished: bool,
}

impl DenoiseState {
    /// Random-initialize a canvas (uniform over vocab, NOT the mask token) and
    /// arm the step budget.
    pub fn new(
        canvas_len: usize,
        n_vocab: usize,
        params: EbParams,
        rng: &mut dyn SamplerRng,
    ) -> Self {
        let s_total = params.max_denoising_steps.max(1);
        let current: Vec<u32> = (0..canvas_len).map(|_| rng.token(n_vocab as u32)).collect();
        Self {
            params,
            canvas_len,
            n_vocab,
            s_total,
            cur_step: s_total,
            current,
            argmax_canvas: vec![0u32; canvas_len],
            prev_argmax: vec![u32::MAX; canvas_len], // step 0 is never "stable"
            prev_logits: None,
            prev_temp_inv: 1.0,
            held: 0,
            finished: false,
        }
    }

    /// No more forwards needed (budget spent or adaptive stop fired).
    pub fn is_done(&self) -> bool {
        self.finished || self.cur_step == 0
    }

    /// The canvas + self-conditioning `(prev_logits, 1/prev_t)` to feed the
    /// next forward. Borrows `self`; drop the borrow before calling `ingest`.
    pub fn forward_inputs(&self) -> (&[u32], Option<(&[f32], f32)>) {
        (
            &self.current,
            self.prev_logits.as_deref().map(|l| (l, self.prev_temp_inv)),
        )
    }

    /// The current best-guess (argmax) canvas.
    pub fn argmax_canvas(&self) -> &[u32] {
        &self.argmax_canvas
    }

    /// Owned copy of the canvas to feed the next forward (for the async / JS
    /// driver, which can't hold a borrow across `await`).
    pub fn input_canvas(&self) -> Vec<u32> {
        self.current.clone()
    }

    /// Move the previous step's logits out for the next forward's
    /// self-conditioning, paired with `1/prev_t` (`None` on the first step).
    /// `ingest` will install THIS step's logits afterward, so taking the old
    /// ones (rather than cloning the ~256×vocab buffer) is sound.
    pub fn take_prev(&mut self) -> Option<(Vec<f32>, f32)> {
        let ti = self.prev_temp_inv;
        self.prev_logits.take().map(|l| (l, ti))
    }

    /// Ingest this step's raw logits `[canvas_len × n_vocab]`: per-position
    /// argmax + entropy + multinomial draw, entropy-budget acceptance, renoise
    /// of the rejected positions, and the adaptive-stop check. Advances the
    /// state; returns the step's stats + whether the loop is done.
    pub fn ingest(&mut self, logits: Vec<f32>, rng: &mut dyn SamplerRng) -> StepOutcome {
        let canvas_len = self.canvas_len;
        let n_vocab = self.n_vocab;
        debug_assert_eq!(logits.len(), canvas_len * n_vocab);
        let step_idx = self.s_total - self.cur_step;
        let t = self.params.t_min
            + (self.params.t_max - self.params.t_min)
                * (self.cur_step as f32 / self.s_total as f32);
        let temp_inv = 1.0 / t;

        // Pre-draw the step's randomness in position order (seed-reproducible,
        // matching the reference's single-threaded pre-draw).
        let us: Vec<f32> = (0..canvas_len).map(|_| rng.uniform01()).collect();
        let renoise: Vec<u32> = (0..canvas_len).map(|_| rng.token(n_vocab as u32)).collect();

        let mut entropy = vec![0f32; canvas_len];
        let mut denoised = vec![0u32; canvas_len];
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
            self.argmax_canvas[pos] = amax as u32;
            denoised[pos] = sampled as u32;
        }

        // Acceptance: ascending entropy; accept while the sum of strictly-
        // earlier entropies stays within the bound.
        let mut order: Vec<usize> = (0..canvas_len).collect();
        order.sort_by(|&a, &b| entropy[a].partial_cmp(&entropy[b]).unwrap().then(a.cmp(&b)));
        let mut accepted = vec![false; canvas_len];
        let mut cum_e = 0f64;
        for &pos in &order {
            if cum_e <= self.params.entropy_bound as f64 {
                accepted[pos] = true;
            }
            cum_e += entropy[pos] as f64;
        }

        // Renoise + stats.
        let mut entropy_sum = 0f32;
        let mut n_accepted = 0usize;
        for pos in 0..canvas_len {
            self.current[pos] = if accepted[pos] {
                n_accepted += 1;
                denoised[pos]
            } else {
                renoise[pos]
            };
            entropy_sum += entropy[pos];
        }

        // Adaptive stop: argmax stable AND confident.
        self.held = if self.prev_argmax == self.argmax_canvas {
            self.held + 1
        } else {
            0
        };
        let mean_entropy = entropy_sum / canvas_len as f32;
        let confident = mean_entropy < self.params.confidence_threshold;

        self.prev_argmax.copy_from_slice(&self.argmax_canvas);
        self.prev_logits = Some(logits);
        self.prev_temp_inv = temp_inv;
        self.cur_step -= 1;
        if (self.held >= self.params.stability_threshold && confident) || self.cur_step == 0 {
            self.finished = true;
        }

        StepOutcome {
            step_idx,
            total_steps: self.s_total,
            mean_entropy,
            n_accepted,
            done: self.finished,
        }
    }
}

/// Run the entropy-bound denoise loop over one canvas (synchronous, trait-based
/// — native callers). Returns the final argmax canvas (length = `canvas_len`).
pub fn generate_entropy_bound(
    model: &mut dyn CanvasForward,
    canvas_len: usize,
    params: &EbParams,
    rng: &mut dyn SamplerRng,
    mut step_cb: Option<&mut dyn FnMut(&StepInfo) -> bool>,
) -> Result<Vec<u32>> {
    let n_vocab = model.n_vocab();
    let mut state = DenoiseState::new(canvas_len, n_vocab, params.clone(), rng);

    while !state.is_done() {
        let logits = {
            let (canvas, prev) = state.forward_inputs();
            model.forward(canvas, prev)?
        };
        let outcome = state.ingest(logits, rng);

        if let Some(cb) = step_cb.as_deref_mut() {
            let info = StepInfo {
                step_idx: outcome.step_idx,
                total_steps: outcome.total_steps,
                argmax_canvas: state.argmax_canvas(),
                mean_entropy: outcome.mean_entropy,
                n_accepted: outcome.n_accepted,
            };
            if !cb(&info) {
                break;
            }
        }
        if outcome.done {
            break;
        }
    }

    Ok(state.argmax_canvas().to_vec())
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

    /// The manual step loop (`input_canvas` + `take_prev` + `ingest`, what the
    /// async / wasm `denoiseStep` does) must produce the SAME canvas as the
    /// trait-driven `generate_entropy_bound`, given the same seed — i.e. the
    /// take-prev plumbing matches the borrow-based driver bit for bit.
    #[test]
    fn manual_step_loop_matches_driver() {
        let target = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
        let params = EbParams {
            max_denoising_steps: 12,
            ..Default::default()
        };

        let mut m1 = FixedTarget {
            target: target.clone(),
            n_vocab: 11,
            calls: 0,
            saw_self_cond: false,
        };
        let mut rng1 = XorShiftRng(0xABCD);
        let driver =
            generate_entropy_bound(&mut m1, target.len(), &params, &mut rng1, None).unwrap();

        let mut m2 = FixedTarget {
            target: target.clone(),
            n_vocab: 11,
            calls: 0,
            saw_self_cond: false,
        };
        let mut rng2 = XorShiftRng(0xABCD);
        let mut st = DenoiseState::new(target.len(), 11, params.clone(), &mut rng2);
        while !st.is_done() {
            let canvas = st.input_canvas();
            let prev = st.take_prev();
            let logits = m2
                .forward(&canvas, prev.as_ref().map(|(l, t)| (l.as_slice(), *t)))
                .unwrap();
            st.ingest(logits, &mut rng2);
        }
        let manual = st.argmax_canvas().to_vec();

        assert_eq!(driver, manual, "step-driven loop must match the driver");
        assert_eq!(driver, target);
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
