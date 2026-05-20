//! Mel-frequency cepstral coefficient extractor for short-frame audio.
//!
//! Standard speech-recognition front-end from the 1980s: pre-emphasis →
//! Hamming-windowed FFT → triangular mel filterbank → log-energy → DCT.
//! Tuned for 16 kHz mono i16 PCM and a 25 ms / 10 ms framing schedule, but
//! all four are constructor parameters.
//!
//! Implementation notes (textbook DSP, none of this is novel):
//! * Pre-emphasis coefficient `α = 0.97`.
//! * Hamming window: `0.54 − 0.46·cos(2π·n / (N−1))`.
//! * FFT size is the next power of two at or above `frame_len` (zero-pad
//!   the windowed frame up to that size).
//! * 26 triangular mel filters spanning 0 Hz to `sample_rate / 2`.
//! * Log of each filter energy uses a floor (`1e-10`) so digital silence
//!   produces a very-negative-but-finite coefficient rather than `-∞`.
//! * DCT-II followed by keeping coefficients `1..=num_coeffs` (the zeroth
//!   coefficient is just total energy and adds nothing to phoneme
//!   discrimination).

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

const NUM_MEL_FILTERS: usize = 26;
const PRE_EMPHASIS: f32 = 0.97;
const LOG_FLOOR: f32 = 1e-10;

/// MFCC feature extractor configured for one audio shape.
///
/// Stateless across calls — the same extractor can be reused for any
/// number of audio chunks. Internally caches the FFT plan, the Hamming
/// window, and the mel filterbank to avoid per-call setup cost.
pub struct MfccExtractor {
    sample_rate: u32,
    frame_len: usize,
    hop: usize,
    num_coeffs: usize,
    fft_size: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// Mel filterbank stored as (start_bin, weights) per filter. Weights are
    /// just the triangle slopes, applied to the power spectrum.
    filters: Vec<(usize, Vec<f32>)>,
}

impl MfccExtractor {
    /// Construct an extractor.
    ///
    /// * `sample_rate` — input PCM rate in Hz (e.g. `16_000`).
    /// * `frame_len` — analysis window in samples (e.g. `400` for 25 ms at 16 kHz).
    /// * `hop` — advance between successive frames in samples (e.g. `160` for 10 ms).
    /// * `num_coeffs` — number of cepstral coefficients to retain (e.g. `12`).
    ///   The 0th coefficient is always dropped — you get coefficients 1..=num_coeffs.
    pub fn new(sample_rate: u32, frame_len: usize, hop: usize, num_coeffs: usize) -> Self {
        let fft_size = next_pow2(frame_len);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = hamming_window(frame_len);
        let filters = build_mel_filterbank(sample_rate, fft_size, NUM_MEL_FILTERS);

        Self {
            sample_rate,
            frame_len,
            hop,
            num_coeffs,
            fft_size,
            fft,
            window,
            filters,
        }
    }

    /// Sample rate this extractor was built for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Frame length (samples per analysis window).
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Hop (samples advanced between successive frames).
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Number of cepstral coefficients retained per frame.
    pub fn num_coeffs(&self) -> usize {
        self.num_coeffs
    }

    /// Extract MFCC frames from a contiguous chunk of PCM audio.
    ///
    /// Returns one `Vec<f32>` of length `num_coeffs` per analysis frame.
    /// Number of frames is `1 + (samples.len() − frame_len) / hop`, clamped
    /// to zero when the input is shorter than a single frame.
    pub fn extract(&self, samples: &[i16]) -> Vec<Vec<f32>> {
        if samples.len() < self.frame_len {
            return Vec::new();
        }

        // Pre-emphasis applied to the whole chunk once. Cheap; avoids
        // re-doing it per overlapping frame.
        let pre = pre_emphasize(samples);

        let n_frames = 1 + (samples.len() - self.frame_len) / self.hop;
        let mut out = Vec::with_capacity(n_frames);

        // Scratch buffers — one FFT-sized complex buffer + one power-spectrum
        // vector, reused per frame.
        let mut buf = vec![Complex32::new(0.0, 0.0); self.fft_size];
        let mut power = vec![0.0f32; self.fft_size / 2 + 1];

        for f in 0..n_frames {
            let start = f * self.hop;
            let end = start + self.frame_len;
            buf.iter_mut().for_each(|c| {
                c.re = 0.0;
                c.im = 0.0;
            });
            // Hamming-window the slice into the front of the buffer; the
            // rest stays zero-padded.
            for (i, &x) in pre[start..end].iter().enumerate() {
                buf[i].re = x * self.window[i];
            }
            self.fft.process(&mut buf);

            // Power spectrum (one-sided).
            for k in 0..power.len() {
                let c = buf[k];
                power[k] = c.re * c.re + c.im * c.im;
            }

            // Mel-filterbank log-energies.
            let mut log_energies = Vec::with_capacity(self.filters.len());
            for (start_bin, weights) in &self.filters {
                let mut e = 0.0f32;
                for (offset, w) in weights.iter().enumerate() {
                    let bin = start_bin + offset;
                    if bin < power.len() {
                        e += power[bin] * w;
                    }
                }
                log_energies.push((e + LOG_FLOOR).ln());
            }

            // DCT-II of the log-energies. We only need coefficients
            // 1..=num_coeffs so we can compute them directly instead of
            // running a full DCT.
            let mut coeffs = Vec::with_capacity(self.num_coeffs);
            let n_filt = log_energies.len();
            for k in 1..=self.num_coeffs {
                let mut s = 0.0f32;
                for (n, &lg) in log_energies.iter().enumerate() {
                    s += lg * ((PI * k as f32 * (n as f32 + 0.5)) / n_filt as f32).cos();
                }
                coeffs.push(s);
            }
            out.push(coeffs);
        }
        out
    }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

fn pre_emphasize(samples: &[i16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let mut prev = 0i16;
    for &s in samples {
        let y = s as f32 - PRE_EMPHASIS * prev as f32;
        out.push(y);
        prev = s;
    }
    out
}

fn hamming_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / denom).cos())
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Build a [`NUM_MEL_FILTERS`]-sized triangular filterbank for the given
/// FFT one-sided spectrum.
fn build_mel_filterbank(
    sample_rate: u32,
    fft_size: usize,
    num_filters: usize,
) -> Vec<(usize, Vec<f32>)> {
    let nyquist = sample_rate as f32 / 2.0;
    let low_mel = hz_to_mel(0.0);
    let high_mel = hz_to_mel(nyquist);
    // num_filters triangles share num_filters+2 edges (low, centers..., high).
    let edges: Vec<f32> = (0..num_filters + 2)
        .map(|i| low_mel + (high_mel - low_mel) * (i as f32) / (num_filters + 1) as f32)
        .map(mel_to_hz)
        .collect();

    let n_bins = fft_size / 2 + 1;
    let bin_freq = |bin: usize| (bin as f32) * (sample_rate as f32) / (fft_size as f32);

    let mut filters = Vec::with_capacity(num_filters);
    for i in 0..num_filters {
        let left = edges[i];
        let center = edges[i + 1];
        let right = edges[i + 2];

        // Find first bin where filter is nonzero. We track the start bin
        // and any nonzero weights in a single Option pair so the post-loop
        // fallback below can be the sole construction point — no separate
        // `.unwrap()` needed.
        let mut found: Option<(usize, Vec<f32>)> = None;
        for bin in 0..n_bins {
            let f = bin_freq(bin);
            let w = if f <= left || f >= right {
                0.0
            } else if f <= center {
                (f - left) / (center - left)
            } else {
                (right - f) / (right - center)
            };
            match (&mut found, w > 0.0) {
                (None, true) => found = Some((bin, vec![w])),
                (Some((_, weights)), true) => weights.push(w),
                (Some(_), false) => break, // past the right edge
                (None, false) => continue,
            }
        }
        // Edge case: filter narrower than one bin → emit at least one bin
        // at the centre with weight 1 so it isn't silently dropped.
        let (start_bin, weights) = found.unwrap_or_else(|| {
            let bin = ((center * fft_size as f32) / (sample_rate as f32)) as usize;
            (bin.min(n_bins.saturating_sub(1)), vec![1.0])
        });
        filters.push((start_bin, weights));
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a sine-wave PCM signal of the given length at 16 kHz.
    fn sine_16k(samples: usize, freq_hz: f32) -> Vec<i16> {
        (0..samples)
            .map(|n| {
                let t = n as f32 / 16_000.0;
                ((2.0 * PI * freq_hz * t).sin() * 10_000.0) as i16
            })
            .collect()
    }

    #[test]
    fn mfcc_extract_silence_yields_low_energy_coefficients() {
        let extractor = MfccExtractor::new(16_000, 400, 160, 12);
        let silence = vec![0i16; 16_000]; // 1 second
        let frames = extractor.extract(&silence);
        assert!(!frames.is_empty(), "silence should still yield frames");

        // Each coefficient is a DCT-II projection of log-energies. With all
        // bands floored to ln(LOG_FLOOR) ≈ -23, the DCT cosine sums are
        // dominated by that constant — the 1st coefficient should be small
        // in magnitude (the floor is the same for every band, so DCT-II
        // projects most of the energy into the 0th coefficient which we
        // dropped). Empirically expect ≤ 1.0 in magnitude.
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.len(), 12);
            assert!(
                frame[0].abs() < 5.0,
                "silence frame {i} 1st coeff = {} should be near zero",
                frame[0]
            );
        }
    }

    #[test]
    fn mfcc_extract_frame_count_matches_hop_arithmetic() {
        let extractor = MfccExtractor::new(16_000, 400, 160, 12);
        // Choose a length that fits a known number of frames cleanly.
        let n_samples = 400 + 160 * 9; // expects 10 frames
        let pcm = vec![1i16; n_samples];
        let frames = extractor.extract(&pcm);
        let expected = 1 + (n_samples - 400) / 160;
        assert_eq!(frames.len(), expected, "framing arithmetic mismatched");

        // Below-frame-length input yields zero frames.
        let short = vec![1i16; 100];
        assert_eq!(extractor.extract(&short).len(), 0);
    }

    #[test]
    fn mfcc_extract_deterministic() {
        let extractor = MfccExtractor::new(16_000, 400, 160, 12);
        let pcm = sine_16k(8_000, 440.0); // 0.5 s of 440 Hz tone
        let a = extractor.extract(&pcm);
        let b = extractor.extract(&pcm);
        assert_eq!(a.len(), b.len());
        for (i, (fa, fb)) in a.iter().zip(b.iter()).enumerate() {
            for (j, (xa, xb)) in fa.iter().zip(fb.iter()).enumerate() {
                assert!(
                    (xa - xb).abs() < 1e-5,
                    "frame {i} coeff {j} drifted: {xa} vs {xb}"
                );
            }
        }
    }

    #[test]
    fn mfcc_sine_distinguishes_from_silence() {
        // Sanity: a tone and silence should produce noticeably different
        // MFCC vectors. This guards against the extractor producing
        // identical coefficients for everything (which would invalidate
        // every downstream DTW test).
        let extractor = MfccExtractor::new(16_000, 400, 160, 12);
        let silence = vec![0i16; 8_000];
        let tone = sine_16k(8_000, 1000.0);
        let s = extractor.extract(&silence);
        let t = extractor.extract(&tone);
        assert_eq!(s.len(), t.len());

        // L2 between the first MFCC frame of each should be substantial.
        let mut d = 0.0f32;
        for (xs, xt) in s[0].iter().zip(t[0].iter()) {
            d += (xs - xt).powi(2);
        }
        d = d.sqrt();
        assert!(
            d > 1.0,
            "silence vs 1 kHz tone should differ in MFCC; got L2 = {d}"
        );
    }
}
