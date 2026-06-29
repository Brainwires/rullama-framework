//! HnNSF harmonic source (istftnet.SourceModuleHnNSF / SineGen) + forward STFT →
//! the `har` tensor fed to the generator's noise branch. Deterministic variant:
//! the SineGen random initial phase + Gaussian noise are zeroed (seed = "off"),
//! matching the zeroed-randomness reference fixtures. Production can add seeded jitter.
#![allow(dead_code)]

use std::f32::consts::PI;

use super::KokoroModel;

const SR: f32 = 24000.0;
const SINE_AMP: f32 = 0.1;
const VOICED_THRESHOLD: f32 = 10.0;

/// F.interpolate 1-D linear, align_corners=False (per PyTorch upsample_linear1d).
fn interp_linear(x: &[f32], in_len: usize, out_len: usize) -> Vec<f32> {
    let scale = in_len as f32 / out_len as f32;
    let mut out = vec![0.0f32; out_len];
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

impl KokoroModel {
    /// HnNSF excitation signal `har_source [L]` (pre-STFT), deterministic.
    pub fn source_signal(&self, f0_curve: &[f32]) -> Vec<f32> {
        let up: usize = self.cfg.upsample_rates.iter().product::<usize>() * self.cfg.gen_istft_hop; // 300
        let l = f0_curve.len() * up; // 46800
        let frames_in = f0_curve.len(); // 156
        let harmonics = 9;

        // f0 upsampled (nearest) → [L]
        let f0_up: Vec<f32> = (0..l).map(|i| f0_curve[i / up]).collect();
        let uv: Vec<f32> = f0_up
            .iter()
            .map(|&v| if v > VOICED_THRESHOLD { 1.0 } else { 0.0 })
            .collect();

        // per-harmonic sine, summed via l_linear, tanh
        let lw = self.t("k.decoder.generator.m_source.l_linear.weight"); // [1,9]
        let lb = self.t("k.decoder.generator.m_source.l_linear.bias"); // [1]
        let mut har_source = vec![0.0f32; l];
        for h in 0..harmonics {
            let mult = (h + 1) as f32;
            let rad: Vec<f32> = f0_up
                .iter()
                .map(|&f| ((f * mult) / SR).rem_euclid(1.0))
                .collect();
            let rad_ds = interp_linear(&rad, l, frames_in);
            let mut cum = vec![0.0f32; frames_in];
            let mut acc = 0.0f32;
            for i in 0..frames_in {
                acc += rad_ds[i];
                cum[i] = acc * 2.0 * PI * up as f32;
            }
            let phase = interp_linear(&cum, frames_in, l);
            let w = lw[h];
            for i in 0..l {
                let sine = phase[i].sin() * SINE_AMP * uv[i]; // noise = 0
                har_source[i] += sine * w;
            }
        }
        for v in har_source.iter_mut() {
            *v = (*v + lb[0]).tanh();
        }
        har_source
    }

    /// Build the harmonic source spectrum `har [n_fft+2, frames]` from the F0 curve.
    /// `harmonic_num = 8` → 9 harmonics. Deterministic (no random phase/noise).
    pub fn generator_source(&self, f0_curve: &[f32]) -> (Vec<f32>, usize) {
        let har_source = self.source_signal(f0_curve);
        let l = har_source.len();

        // forward STFT (torch.stft: center reflect pad, hann periodic, onesided) → mag, phase
        let nfft = self.cfg.gen_istft_n_fft; // 20
        let hop = self.cfg.gen_istft_hop; // 5
        let nbins = nfft / 2 + 1; // 11
        let pad = nfft / 2; // 10
        let mut padded = vec![0.0f32; l + 2 * pad];
        for i in 0..pad {
            padded[i] = har_source[pad - i]; // reflect (exclude edge)
            padded[l + pad + i] = har_source[l - 2 - i];
        }
        padded[pad..pad + l].copy_from_slice(&har_source);
        let win: Vec<f32> = (0..nfft)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / nfft as f32).cos())
            .collect();
        let frames = (padded.len() - nfft) / hop + 1;

        let mut har = vec![0.0f32; (nfft + 2) * frames]; // [22, frames]: 0..10 mag, 11..21 phase
        for f in 0..frames {
            for k in 0..nbins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for n in 0..nfft {
                    let s = padded[f * hop + n] * win[n];
                    let ang = 2.0 * PI * (k * n) as f32 / nfft as f32;
                    re += s * ang.cos();
                    im -= s * ang.sin();
                }
                har[k * frames + f] = (re * re + im * im).sqrt();
                har[(k + nbins) * frames + f] = im.atan2(re);
            }
        }
        (har, frames)
    }
}
