//! Mel-spectrogram pipeline for the Gemma 4 audio tower.
//!
//! Mirrors `process_audio.go` in Ollama:
//!   * 16 kHz mono PCM input (or WAV file bytes — decoded here).
//!   * 20 ms / 10 ms (320-sample / 160-sample) frames, Hann window.
//!   * Power-of-two FFT length (`fft_overdrive=True` → ≥ 2× frame_length).
//!   * 128-bin mel filterbank, log-compressed with floor 1e-3.
//!
//! Hand-rolled Cooley-Tukey radix-2 FFT (≈ 25 LOC) — no realfft dependency,
//! keeps the wasm bundle a few dozen KB smaller.
//!
//! Output: `Vec<f32>` of length `n_frames * 128`, ready to feed AudioForward.

use crate::error::{Result, RullamaError};

pub const SAMPLE_RATE:       usize = 16_000;
pub const MEL_BINS:          usize = 128;
pub const FRAME_LENGTH:      usize = 320;     // 20 ms @ 16 kHz
pub const HOP_LENGTH:        usize = 160;     // 10 ms @ 16 kHz
pub const MIN_FREQUENCY:     f32   = 0.0;
pub const MAX_FREQUENCY:     f32   = 8_000.0;
pub const MEL_FLOOR:         f32   = 1e-3;
pub const MAX_AUDIO_TOKENS:  usize = 750;

/// FFT length used by the mel pipeline. `fft_overdrive=True` per Ollama:
/// next pow2 ≥ FRAME_LENGTH, then doubled. With FRAME_LENGTH=320 → 1024.
pub const FFT_LEN: usize = 1024;
pub const NUM_FREQ_BINS: usize = FFT_LEN / 2 + 1;     // 513

/// Cached pieces (Hann window + mel filterbank + bit-reversal permutation)
/// used to compute every spectrogram. Built once per process.
pub struct MelEngine {
    /// Hann-non-zero window of length FRAME_LENGTH (mirrors Ollama's
    /// 0.5 - 0.5 * cos(2π/N * (i + 0.5))).
    window: Vec<f32>,
    /// `[NUM_FREQ_BINS, MEL_BINS]` row-major: `filters[k * MEL_BINS + m]`.
    filters: Vec<f32>,
    /// Bit-reversal permutation for the in-place radix-2 FFT.
    bit_reverse: Vec<usize>,
    /// Pre-computed FFT twiddle factors.
    twiddles_re: Vec<f32>,
    twiddles_im: Vec<f32>,
}

impl MelEngine {
    pub fn new() -> Self {
        let mut window = vec![0f32; FRAME_LENGTH];
        let arg = std::f32::consts::PI * 2.0 / FRAME_LENGTH as f32;
        for i in 0..FRAME_LENGTH {
            window[i] = 0.5 - 0.5 * (arg * (i as f32 + 0.5)).cos();
        }

        let filters = build_mel_filterbank(
            NUM_FREQ_BINS, MEL_BINS, MIN_FREQUENCY, MAX_FREQUENCY, SAMPLE_RATE,
        );

        // Bit-reversal index for in-place radix-2 FFT.
        let mut bit_reverse = vec![0usize; FFT_LEN];
        let mut j = 0usize;
        for i in 1..FFT_LEN {
            let mut bit = FFT_LEN >> 1;
            while j & bit != 0 { j ^= bit; bit >>= 1; }
            j ^= bit;
            bit_reverse[i] = j;
        }

        // Twiddles: w_k = exp(-2πi k / N) for k = 0..N/2.
        let mut twiddles_re = Vec::with_capacity(FFT_LEN / 2);
        let mut twiddles_im = Vec::with_capacity(FFT_LEN / 2);
        for k in 0..FFT_LEN / 2 {
            let theta = -2.0 * std::f32::consts::PI * k as f32 / FFT_LEN as f32;
            twiddles_re.push(theta.cos());
            twiddles_im.push(theta.sin());
        }

        Self { window, filters, bit_reverse, twiddles_re, twiddles_im }
    }

    /// Compute log-mel spectrogram: returns `(flat [n_frames * MEL_BINS] f32, n_frames)`.
    /// Caps `n_frames` at `MAX_AUDIO_TOKENS` (truncates the input audio if longer).
    pub fn log_mel(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        // Ollama uses (len - (frame_length + 1)) / hop_length for frame count.
        let frame_size_for_unfold = FRAME_LENGTH + 1;
        if samples.len() < frame_size_for_unfold {
            return (Vec::new(), 0);
        }
        let mut n_frames = (samples.len() - frame_size_for_unfold) / HOP_LENGTH;
        if n_frames > MAX_AUDIO_TOKENS { n_frames = MAX_AUDIO_TOKENS; }

        let mut out = vec![0f32; n_frames * MEL_BINS];
        let mut re = vec![0f32; FFT_LEN];
        let mut im = vec![0f32; FFT_LEN];

        for f in 0..n_frames {
            // Window + zero-pad to FFT_LEN.
            let start = f * HOP_LENGTH;
            for i in 0..FRAME_LENGTH {
                re[i] = samples[start + i] * self.window[i];
                im[i] = 0.0;
            }
            for i in FRAME_LENGTH..FFT_LEN { re[i] = 0.0; im[i] = 0.0; }

            self.fft_in_place(&mut re, &mut im);

            // Magnitude → mel → log.
            for m in 0..MEL_BINS {
                let mut mel_val = 0f32;
                for k in 0..NUM_FREQ_BINS {
                    let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
                    mel_val += mag * self.filters[k * MEL_BINS + m];
                }
                if mel_val < MEL_FLOOR { mel_val = MEL_FLOOR; }
                out[f * MEL_BINS + m] = mel_val.ln();
            }
        }
        (out, n_frames)
    }

    fn fft_in_place(&self, re: &mut [f32], im: &mut [f32]) {
        // Bit-reversal permutation.
        for i in 1..FFT_LEN {
            let j = self.bit_reverse[i];
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        // Cooley-Tukey butterflies.
        let mut size = 2usize;
        while size <= FFT_LEN {
            let half = size / 2;
            let twiddle_step = FFT_LEN / size;
            let mut start = 0;
            while start < FFT_LEN {
                for k in 0..half {
                    let tw_idx = k * twiddle_step;
                    let w_re = self.twiddles_re[tw_idx];
                    let w_im = self.twiddles_im[tw_idx];

                    let i1 = start + k;
                    let i2 = start + k + half;
                    let t_re = w_re * re[i2] - w_im * im[i2];
                    let t_im = w_re * im[i2] + w_im * re[i2];
                    re[i2] = re[i1] - t_re;
                    im[i2] = im[i1] - t_im;
                    re[i1] = re[i1] + t_re;
                    im[i1] = im[i1] + t_im;
                }
                start += size;
            }
            size <<= 1;
        }
    }
}

impl Default for MelEngine {
    fn default() -> Self { Self::new() }
}

fn build_mel_filterbank(
    num_freq_bins: usize, num_mels: usize, f_min: f32, f_max: f32, sr: usize,
) -> Vec<f32> {
    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10f32.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mut mel_pts = vec![0f32; num_mels + 2];
    for i in 0..num_mels + 2 {
        mel_pts[i] = mel_min + i as f32 * (mel_max - mel_min) / (num_mels + 1) as f32;
    }
    let filter_freqs: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
    let mut fft_freqs = vec![0f32; num_freq_bins];
    for i in 0..num_freq_bins {
        fft_freqs[i] = i as f32 * sr as f32 / (2 * (num_freq_bins - 1)) as f32;
    }

    let mut filters = vec![0f32; num_freq_bins * num_mels];
    for m in 0..num_mels {
        let f_left   = filter_freqs[m];
        let f_center = filter_freqs[m + 1];
        let f_right  = filter_freqs[m + 2];
        for k in 0..num_freq_bins {
            let f = fft_freqs[k];
            let mut v = 0f32;
            if f >= f_left && f <= f_center && f_center > f_left {
                v = (f - f_left) / (f_center - f_left);
            } else if f > f_center && f <= f_right && f_right > f_center {
                v = (f_right - f) / (f_right - f_center);
            }
            if v > 0.0 {
                filters[k * num_mels + m] = v;
            }
        }
    }
    filters
}

// ---------- WAV decoder ----------

/// Decode a WAV file (RIFF/WAVE container, PCM 8/16/24/32 or IEEE float32) into
/// 16 kHz mono `f32` samples in `[-1, 1]`. Mirrors `process_audio.go::decodeWAV`.
pub fn decode_wav(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() < 12
        || &data[0..4] != b"RIFF"
        || &data[8..12] != b"WAVE"
    {
        return Err(RullamaError::Inference("not a WAV file".into()));
    }

    let mut audio_format: u16 = 0;
    let mut num_channels: usize = 0;
    let mut sample_rate: usize = 0;
    let mut bits_per_sample: usize = 0;
    let mut audio_data: &[u8] = &[];
    let mut found_fmt = false;

    let mut offset = 12;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let chunk_end = (offset + 8 + chunk_size).min(data.len());
        let chunk_data = &data[offset + 8..chunk_end];

        match chunk_id {
            b"fmt " => {
                if chunk_data.len() < 16 {
                    return Err(RullamaError::Inference("fmt chunk too short".into()));
                }
                audio_format = u16::from_le_bytes(chunk_data[0..2].try_into().unwrap());
                num_channels = u16::from_le_bytes(chunk_data[2..4].try_into().unwrap()) as usize;
                sample_rate = u32::from_le_bytes(chunk_data[4..8].try_into().unwrap()) as usize;
                bits_per_sample = u16::from_le_bytes(chunk_data[14..16].try_into().unwrap()) as usize;
                if audio_format == 0xFFFE && chunk_data.len() >= 26 {
                    audio_format = u16::from_le_bytes(chunk_data[24..26].try_into().unwrap());
                }
                found_fmt = true;
            }
            b"data" => audio_data = chunk_data,
            _ => {}
        }

        offset += 8 + chunk_size;
        if chunk_size % 2 != 0 { offset += 1; }
    }

    if !found_fmt {
        return Err(RullamaError::Inference("no fmt chunk".into()));
    }
    if audio_format != 1 && audio_format != 3 {
        return Err(RullamaError::Inference(format!(
            "unsupported WAV format {} (need PCM=1 or float=3)", audio_format
        )));
    }
    if audio_data.is_empty() {
        return Err(RullamaError::Inference("no data chunk".into()));
    }

    let mut mono = decode_wav_samples(audio_data, audio_format, bits_per_sample, num_channels);
    if sample_rate != SAMPLE_RATE {
        mono = resample_linear(&mono, sample_rate, SAMPLE_RATE);
    }
    Ok(mono)
}

fn decode_wav_samples(data: &[u8], format: u16, bits: usize, channels: usize) -> Vec<f32> {
    if channels == 0 || bits == 0 { return Vec::new(); }
    let bytes_per_sample = bits / 8;
    let total_samples = data.len() / (bytes_per_sample * channels);
    let mut mono = vec![0f32; total_samples];

    for i in 0..total_samples {
        let mut sum = 0f64;
        for ch in 0..channels {
            let off = (i * channels + ch) * bytes_per_sample;
            if off + bytes_per_sample > data.len() { break; }
            sum += match (format, bits) {
                (1, 16) => {
                    let v = i16::from_le_bytes(data[off..off + 2].try_into().unwrap());
                    v as f64 / 32768.0
                }
                (1, 32) => {
                    let v = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                    v as f64 / 2_147_483_648.0
                }
                (1, 24) => {
                    let mut v = i32::from(data[off])
                        | (i32::from(data[off + 1]) << 8)
                        | (i32::from(data[off + 2]) << 16);
                    if v & 0x800000 != 0 { v |= !0xFFFFFF; }
                    v as f64 / 8_388_608.0
                }
                (3, 32) => {
                    let v = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                    v as f64
                }
                (1, 8) => (data[off] as f64 - 128.0) / 128.0,
                _ => 0.0,
            };
        }
        mono[i] = (sum / channels as f64) as f32;
    }
    mono
}

fn resample_linear(samples: &[f32], from_rate: usize, to_rate: usize) -> Vec<f32> {
    if samples.is_empty() { return Vec::new(); }
    let n = samples.len() * to_rate / from_rate;
    let mut out = vec![0f32; n];
    if n <= 1 { return out; }
    for i in 0..n {
        let pos = i as f64 * (samples.len() - 1) as f64 / (n - 1) as f64;
        let idx = pos as usize;
        let frac = (pos - idx as f64) as f32;
        out[i] = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else {
            samples[idx]
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip_sanity() {
        // FFT of an impulse [1, 0, ..., 0] is all-ones. Energy should equal N.
        let eng = MelEngine::new();
        let mut re = vec![0f32; FFT_LEN];
        let mut im = vec![0f32; FFT_LEN];
        re[0] = 1.0;
        eng.fft_in_place(&mut re, &mut im);
        for k in 0..FFT_LEN {
            assert!((re[k] - 1.0).abs() < 1e-4, "bin {k} re={}", re[k]);
            assert!(im[k].abs() < 1e-4, "bin {k} im={}", im[k]);
        }
    }

    #[test]
    fn log_mel_silence_is_floor() {
        let eng = MelEngine::new();
        let silence = vec![0f32; SAMPLE_RATE]; // 1 s of silence.
        let (mel, n) = eng.log_mel(&silence);
        assert!(n > 0);
        let expected = MEL_FLOOR.ln();
        for v in &mel {
            assert!((v - expected).abs() < 1e-3, "got {} expected {}", v, expected);
        }
    }
}
