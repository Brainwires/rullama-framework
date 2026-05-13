//! Token sampling: temperature / top-k / top-p / repetition penalty.
//!
//! Sits *after* the forward pass — takes a logit vector, returns a token id. Pure
//! Rust, no GPU, runs identically on native and wasm32.
//!
//! Conventions:
//!   * `temperature <= 0`  → greedy (argmax). Same as Ollama with `temperature 0`.
//!   * `top_k == 0`        → disabled (consider all logits).
//!   * `top_p` ∈ (0, 1)    → nucleus filter. `1.0` disables.
//!   * `repetition_penalty == 1.0` → disabled. Standard llama.cpp formula:
//!     positive logits divided, negative logits multiplied (so the *magnitude* moves
//!     toward zero either way, dampening repeat probability).
//!
//! Random source is a tiny xorshift64; we don't pull `rand` into the WASM bundle for
//! a single u64 RNG.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl SamplingOptions {
    /// Always-pick-highest-probability — useful for parity tests with Ollama at
    /// `temperature 0`.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

pub struct Sampler {
    opts: SamplingOptions,
    rng: u64,
    /// Most recent N tokens we've sampled or fed; used for repetition penalty.
    history: Vec<u32>,
    history_window: usize,
}

impl Sampler {
    pub fn new(opts: SamplingOptions) -> Self {
        let seed = if opts.seed == 0 { 0xC0FFEE_5E7Du64 } else { opts.seed };
        Self {
            opts,
            rng: seed,
            history: Vec::new(),
            history_window: 64,
        }
    }

    pub fn options(&self) -> SamplingOptions { self.opts }

    pub fn set_options(&mut self, opts: SamplingOptions) {
        let seed = if opts.seed == 0 { self.rng } else { opts.seed };
        self.opts = opts;
        self.rng = seed;
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Record that `token` was just fed or sampled. Drives the repetition penalty.
    pub fn observe(&mut self, token: u32) {
        self.history.push(token);
        if self.history.len() > self.history_window {
            self.history.remove(0);
        }
    }

    /// Pick a token from `logits`. Mutates `self.rng`. Caller is responsible for
    /// `observe()`-ing the result if it should count toward future rep-penalty windows.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // Greedy short-circuit (no allocations).
        if self.opts.temperature <= 0.0 {
            return argmax(logits) as u32;
        }

        let mut work: Vec<f32> = logits.to_vec();

        // Repetition penalty.
        if self.opts.repetition_penalty > 1.0 {
            let p = self.opts.repetition_penalty;
            for &id in &self.history {
                let i = id as usize;
                if i < work.len() {
                    let l = work[i];
                    work[i] = if l > 0.0 { l / p } else { l * p };
                }
            }
        }

        // Temperature.
        let temp = self.opts.temperature.max(1e-6);
        for v in work.iter_mut() { *v /= temp; }

        // Build (idx, logit) pairs; we'll trim by top_k via partial select.
        let mut pairs: Vec<(usize, f32)> = work.iter().copied().enumerate().collect();

        let k = if self.opts.top_k == 0 || self.opts.top_k as usize >= pairs.len() {
            pairs.len()
        } else {
            self.opts.top_k as usize
        };
        if k < pairs.len() {
            // Partition so the first k are the top-k by logit (descending),
            // rest are below them. Fast: O(N).
            pairs.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
            });
            pairs.truncate(k);
        }
        // Sort the trimmed set descending (small N, cheap).
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Softmax in numerically stable form.
        let max_l = pairs[0].1;
        let mut probs: Vec<(usize, f32)> = pairs.into_iter()
            .map(|(i, l)| (i, (l - max_l).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for p in probs.iter_mut() { p.1 /= sum; }
        }

        // Top-p (nucleus): keep smallest prefix whose cumulative prob >= top_p.
        if self.opts.top_p > 0.0 && self.opts.top_p < 1.0 {
            let mut cum = 0f32;
            let mut keep = probs.len();
            for (idx, (_, p)) in probs.iter().enumerate() {
                cum += *p;
                if cum >= self.opts.top_p {
                    keep = idx + 1;
                    break;
                }
            }
            probs.truncate(keep);
            let s: f32 = probs.iter().map(|(_, p)| *p).sum();
            if s > 0.0 {
                for p in probs.iter_mut() { p.1 /= s; }
            }
        }

        // Sample.
        let r = self.rand_unit();
        let mut cum = 0f32;
        for (id, p) in &probs {
            cum += *p;
            if r <= cum { return *id as u32; }
        }
        probs.last().map(|(id, _)| *id as u32).unwrap_or(0)
    }

    /// xorshift64 → uniform `[0, 1)`.
    fn rand_unit(&mut self) -> f32 {
        let mut s = self.rng;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.rng = s;
        (s as u32 as f32) / (u32::MAX as f32 + 1.0)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v { best_v = x; best_i = i; }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(n: usize, peak: usize, peak_v: f32, base: f32) -> Vec<f32> {
        let mut v = vec![base; n];
        v[peak] = peak_v;
        v
    }

    #[test]
    fn greedy_picks_argmax() {
        let logits = make_logits(100, 42, 5.0, 0.0);
        let mut s = Sampler::new(SamplingOptions::greedy());
        assert_eq!(s.sample(&logits), 42);
    }

    #[test]
    fn temperature_alone_still_picks_high_prob() {
        let logits = make_logits(100, 7, 10.0, 0.0);
        let mut s = Sampler::new(SamplingOptions {
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
        });
        // Even with sampling, a 10 vs 0 logit gap means peak should win nearly always.
        let mut hits = 0;
        for _ in 0..50 { if s.sample(&logits) == 7 { hits += 1; } }
        assert!(hits >= 45, "expected ≥45/50 hits on dominant token, got {hits}");
    }

    #[test]
    fn top_k_eq_one_is_greedy() {
        let logits = make_logits(100, 17, 1.0, 0.0);
        let mut s = Sampler::new(SamplingOptions {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 1,
        });
        // top_k=1 keeps only the argmax token, so every sample is 17.
        for _ in 0..10 { assert_eq!(s.sample(&logits), 17); }
    }

    #[test]
    fn repetition_penalty_lowers_seen_token_probability() {
        let logits = vec![5.0, 0.0, 0.0];
        let mut s = Sampler::new(SamplingOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 5.0,
            seed: 99,
        });
        s.observe(0);
        let mut hit_zero = 0;
        for _ in 0..200 { if s.sample(&logits) == 0 { hit_zero += 1; } }
        // Without penalty, token 0 dominates (logit 5 vs 0). With strong penalty, its
        // logit is divided by 5 → 1.0, so it still wins but no longer dominates.
        assert!(hit_zero < 180, "rep penalty should reduce token-0 dominance, hit {hit_zero}/200");
    }
}
