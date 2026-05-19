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
        let seed = if opts.seed == 0 {
            0x00C0_FFEE_5E7D_u64
        } else {
            opts.seed
        };
        Self {
            opts,
            rng: seed,
            history: Vec::new(),
            history_window: 64,
        }
    }

    pub fn options(&self) -> SamplingOptions {
        self.opts
    }

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
        for v in work.iter_mut() {
            *v /= temp;
        }

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
        let mut probs: Vec<(usize, f32)> = pairs
            .into_iter()
            .map(|(i, l)| (i, (l - max_l).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                p.1 /= sum;
            }
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
                for p in probs.iter_mut() {
                    p.1 /= s;
                }
            }
        }

        // Sample.
        let r = self.rand_unit();
        let mut cum = 0f32;
        for (id, p) in &probs {
            cum += *p;
            if r <= cum {
                return *id as u32;
            }
        }
        probs.last().map(|(id, _)| *id as u32).unwrap_or(0)
    }

    /// Serialize sampler state for suspend/resume: RNG cursor + options +
    /// repetition-penalty history. Format (little-endian):
    ///
    /// ```text
    ///   [0..8]   rng (u64)
    ///   [8..16]  opts.seed (u64)
    ///   [16..20] opts.temperature (f32)
    ///   [20..24] opts.top_k (u32)
    ///   [24..28] opts.top_p (f32)
    ///   [28..32] opts.repetition_penalty (f32)
    ///   [32..36] history_window (u32)
    ///   [36..40] history_len (u32, ≤ history_window)
    ///   [40..]   history[i] (u32 each, history_len entries)
    /// ```
    pub fn dump_state(&self) -> Vec<u8> {
        let history_len = self.history.len() as u32;
        let mut out = Vec::with_capacity(40 + (history_len as usize) * 4);
        out.extend_from_slice(&self.rng.to_le_bytes());
        out.extend_from_slice(&self.opts.seed.to_le_bytes());
        out.extend_from_slice(&self.opts.temperature.to_le_bytes());
        out.extend_from_slice(&self.opts.top_k.to_le_bytes());
        out.extend_from_slice(&self.opts.top_p.to_le_bytes());
        out.extend_from_slice(&self.opts.repetition_penalty.to_le_bytes());
        out.extend_from_slice(&(self.history_window as u32).to_le_bytes());
        out.extend_from_slice(&history_len.to_le_bytes());
        for &tok in &self.history {
            out.extend_from_slice(&tok.to_le_bytes());
        }
        out
    }

    /// Inverse of [`dump_state`]. Validates header, refuses to clobber if
    /// the byte slice is malformed. Restores RNG cursor, options, history.
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() < 40 {
            return Err(format!("sampler state too short: {} bytes", bytes.len()));
        }
        let rng = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let seed = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let temperature = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let top_k = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let top_p = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let repetition_penalty = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let history_window = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
        let history_len = u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize;
        let expected_total = 40 + history_len * 4;
        if bytes.len() < expected_total {
            return Err(format!(
                "sampler state truncated: have {} bytes, need {} for history_len={}",
                bytes.len(),
                expected_total,
                history_len,
            ));
        }
        let mut history = Vec::with_capacity(history_len);
        for i in 0..history_len {
            let off = 40 + i * 4;
            history.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        }
        self.rng = rng;
        self.opts = SamplingOptions {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
        };
        self.history_window = history_window.max(1);
        self.history = history;
        Ok(())
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
        if x > best_v {
            best_v = x;
            best_i = i;
        }
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
        for _ in 0..50 {
            if s.sample(&logits) == 7 {
                hits += 1;
            }
        }
        assert!(
            hits >= 45,
            "expected ≥45/50 hits on dominant token, got {hits}"
        );
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
        for _ in 0..10 {
            assert_eq!(s.sample(&logits), 17);
        }
    }

    #[test]
    fn dump_load_state_roundtrip() {
        let opts = SamplingOptions {
            temperature: 0.42,
            top_k: 17,
            top_p: 0.87,
            repetition_penalty: 1.3,
            seed: 0xABCDEF,
        };
        let mut s = Sampler::new(opts);
        for tok in [3u32, 7, 11, 13, 19] {
            s.observe(tok);
        }
        // Advance the RNG so the saved cursor isn't the default seed.
        let _ = s.rand_unit();
        let _ = s.rand_unit();
        let rng_before = s.rng;
        let bytes = s.dump_state();

        // Mutate the sampler arbitrarily, then restore.
        let mut s2 = Sampler::new(SamplingOptions::default());
        s2.observe(99);
        let _ = s2.rand_unit();
        s2.load_state(&bytes).expect("load_state");

        assert_eq!(s2.rng, rng_before);
        assert_eq!(s2.opts.temperature, opts.temperature);
        assert_eq!(s2.opts.top_k, opts.top_k);
        assert_eq!(s2.opts.top_p, opts.top_p);
        assert_eq!(s2.opts.repetition_penalty, opts.repetition_penalty);
        assert_eq!(s2.opts.seed, opts.seed);
        assert_eq!(s2.history, vec![3, 7, 11, 13, 19]);

        // Sample-equivalence sanity: same logits → same next token + same rng tail.
        let logits = vec![0.1f32, 5.0, 0.2, 0.3];
        let mut s_a = Sampler::new(opts);
        for tok in [3u32, 7, 11, 13, 19] {
            s_a.observe(tok);
        }
        let _ = s_a.rand_unit();
        let _ = s_a.rand_unit();
        let t_a = s_a.sample(&logits);
        let mut s_b = s2;
        let t_b = s_b.sample(&logits);
        assert_eq!(t_a, t_b);
        assert_eq!(s_a.rng, s_b.rng);
    }

    #[test]
    fn load_state_rejects_short_buffer() {
        let mut s = Sampler::new(SamplingOptions::default());
        assert!(s.load_state(&[]).is_err());
        assert!(s.load_state(&[0u8; 10]).is_err());
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
        for _ in 0..200 {
            if s.sample(&logits) == 0 {
                hit_zero += 1;
            }
        }
        // Without penalty, token 0 dominates (logit 5 vs 0). With strong penalty, its
        // logit is divided by 5 → 1.0, so it still wins but no longer dominates.
        assert!(
            hit_zero < 180,
            "rep penalty should reduce token-0 dominance, hit {hit_zero}/200"
        );
    }
}
