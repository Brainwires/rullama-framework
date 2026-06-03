//! Gradient-free voice training (Kokoro voice cloning), forward-only.
//!
//! Optimizes the 256-d voice/style vector to make the synthesized timbre match a
//! target speaker — the same family of approach as kvoicewalk. No backward kernels:
//! it just runs `synthesize_gpu_fast` and scores the output. The loss is a
//! TIMING-INVARIANT timbre signature (mean log-mel over frames) so that changing the
//! style — which changes Kokoro's predicted durations — doesn't break mel alignment.
#![allow(dead_code)]

use super::gpu_fast::WeightCache;
use super::KokoroModel;
use crate::backend::{Pipelines, WgpuCtx};
use crate::multimodal::audio_features::MelEngine;

/// Timbre signature: mean log-mel over time. Length = MEL_BINS. Timing/text-invariant.
pub fn voice_signature(audio: &[f32]) -> Vec<f32> {
    let (mel, n_frames) = MelEngine::new().log_mel(audio);
    if n_frames == 0 {
        return Vec::new();
    }
    let bins = mel.len() / n_frames;
    let mut sig = vec![0.0f32; bins];
    for f in 0..n_frames {
        for m in 0..bins {
            sig[m] += mel[f * bins + m];
        }
    }
    for v in sig.iter_mut() {
        *v /= n_frames as f32;
    }
    sig
}

fn sig_loss(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len().max(1) as f32
}

/// Deterministic xorshift64 + Box–Muller (no Math.random / Instant in this env).
pub(crate) struct Rng(pub u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Result of a voice-training run.
pub struct VoiceTrainResult {
    pub style: Vec<f32>,
    pub loss_curve: Vec<f32>,
}

impl KokoroModel {
    /// Hill-climb the 256-d voice vector to match `target_sig`, synthesizing `ids`
    /// each evaluation. `step` is the initial Gaussian perturbation scale (annealed on
    /// reject). Returns the best voicepack + the loss curve.
    #[allow(clippy::too_many_arguments)]
    pub async fn train_voice(
        &self, ctx: &WgpuCtx, p: &Pipelines, wc: &mut WeightCache,
        ids: &[i64], target_sig: &[f32], init_style: &[f32], iters: usize, step0: f32, seed: u64,
    ) -> VoiceTrainResult {
        let mut style = init_style.to_vec();
        let mut cur_loss = {
            let audio = self.synthesize_gpu_fast(ctx, p, wc, ids, &style).await;
            sig_loss(&voice_signature(&audio), target_sig)
        };
        let mut curve = vec![cur_loss];
        let mut rng = Rng(seed | 1);
        let mut step = step0;
        for _ in 0..iters {
            let mut cand = style.clone();
            for v in cand.iter_mut() {
                *v += step * rng.gauss();
            }
            let audio = self.synthesize_gpu_fast(ctx, p, wc, ids, &cand).await;
            let loss = sig_loss(&voice_signature(&audio), target_sig);
            if loss < cur_loss {
                style = cand;
                cur_loss = loss;
            } else {
                step *= 0.95; // anneal when a step doesn't help
            }
            curve.push(cur_loss);
        }
        VoiceTrainResult { style, loss_curve: curve }
    }

    /// One hill-climb step (for incremental/UI-driven training). Mutates `style`,
    /// `cur_loss`, `step`, `rng` in place; returns the (possibly unchanged) current loss.
    #[allow(clippy::too_many_arguments)]
    pub async fn voice_train_step(
        &self, ctx: &WgpuCtx, p: &Pipelines, wc: &mut WeightCache, ids: &[i64], target_sig: &[f32],
        style: &mut Vec<f32>, cur_loss: &mut f32, step: &mut f32, rng: &mut u64,
    ) -> f32 {
        let mut r = Rng(*rng | 1);
        let mut cand = style.clone();
        for v in cand.iter_mut() {
            *v += *step * r.gauss();
        }
        *rng = r.0;
        let audio = self.synthesize_gpu_fast(ctx, p, wc, ids, &cand).await;
        let loss = sig_loss(&voice_signature(&audio), target_sig);
        if loss < *cur_loss {
            *style = cand;
            *cur_loss = loss;
        } else {
            *step *= 0.95;
        }
        *cur_loss
    }
}
