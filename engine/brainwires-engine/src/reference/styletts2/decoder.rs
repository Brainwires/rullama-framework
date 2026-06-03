//! StyleTTS2-LibriTTS hifigan Decoder — CPU f32 oracle.
//!
//! Shape-identical to Kokoro's istftnet Decoder for the cat-stack (encode + 4 decode
//! AdainResBlk1d), but the Generator is the net-new **hifigan** path: 4 upsamples
//! [10,5,3,2], a per-stage Snake on the trunk (`generator.alphas`), a single-channel
//! HnNSF source fed straight in (no STFT), and a direct-waveform `conv_post` + tanh.
//!
//! Ported piece-by-piece against the isolation fixtures from
//! `scripts/styletts2_dump_decoder_fixtures.py` (encode/decode0-3/har_source/conv_post/audio).
#![allow(dead_code)]

use std::f32::consts::PI;

const SR: f32 = 24000.0;
const SINE_AMP: f32 = 0.1;
const VOICED_THRESHOLD: f32 = 10.0;

/// `F.interpolate(mode="linear", align_corners=False)` for a 1-D signal.
pub(crate) fn interp_linear(x: &[f32], in_len: usize, out_len: usize) -> Vec<f32> {
    let scale = in_len as f32 / out_len as f32;
    let mut out = vec![0f32; out_len];
    for o in 0..out_len {
        let src = (o as f32 + 0.5) * scale - 0.5;
        if src <= 0.0 {
            out[o] = x[0];
        } else {
            let i0 = src.floor() as usize;
            let frac = src - i0 as f32;
            let i1 = (i0 + 1).min(in_len - 1);
            out[o] = x[i0.min(in_len - 1)] * (1.0 - frac) + x[i1] * frac;
        }
    }
    out
}

/// hifigan HnNSF excitation: `f0_curve[frames]` → `har_source[frames*up]`, deterministic
/// (random phase + noise zeroed, matching the fixtures). `up = prod(upsample_rates)`.
/// Algorithm mirrors StyleTTS2's SineGen: nearest-upsample f0 ×up, take `(f0·h/SR) mod 1`,
/// downsample to `frames`, cumulative phase ×2π·up, upsample back, sin, merge via
/// `l_linear` (+tanh). No STFT (unlike Kokoro's istftnet source).
pub fn source_signal(f0_curve: &[f32], up: usize, harmonics: usize, l_w: &[f32], l_b: f32) -> Vec<f32> {
    let frames_in = f0_curve.len();
    let l = frames_in * up;
    let f0_up: Vec<f32> = (0..l).map(|i| f0_curve[i / up]).collect(); // nearest upsample
    let uv: Vec<f32> = f0_up.iter().map(|&v| if v > VOICED_THRESHOLD { 1.0 } else { 0.0 }).collect();

    let mut har = vec![0f32; l];
    for h in 0..harmonics {
        let mult = (h + 1) as f32;
        let rad: Vec<f32> = f0_up.iter().map(|&f| (f * mult / SR).rem_euclid(1.0)).collect();
        let rad_ds = interp_linear(&rad, l, frames_in);
        let mut cum = vec![0f32; frames_in];
        let mut acc = 0.0f32;
        for i in 0..frames_in {
            acc += rad_ds[i];
            cum[i] = acc * 2.0 * PI * up as f32;
        }
        let phase = interp_linear(&cum, frames_in, l);
        let w = l_w[h];
        for i in 0..l {
            har[i] += phase[i].sin() * SINE_AMP * uv[i] * w;
        }
    }
    for v in har.iter_mut() {
        *v = (*v + l_b).tanh();
    }
    har
}
