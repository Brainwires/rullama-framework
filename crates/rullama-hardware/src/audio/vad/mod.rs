//! Voice Activity Detection (VAD) for the voice assistant pipeline.
//!
//! Two implementations are provided:
//!
//! - [`EnergyVad`] — pure-Rust RMS energy threshold. Zero extra dependencies.
//!   Always available when the `audio` feature is enabled.
//! - [`WebRtcVad`] — wraps the WebRTC VAD algorithm (three aggressiveness modes).
//!   Enabled by the `vad` feature flag.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use rullama_hardware::audio::vad::{EnergyVad, VoiceActivityDetector};
//! // ... create an AudioBuffer from mic capture, then:
//! let vad = EnergyVad::default();
//! // if vad.is_speech(&buffer) { /* speech detected */ }
//! ```

/// Energy-based VAD implementation.
pub mod energy;
/// WebRTC-based VAD implementation.
#[cfg(feature = "vad")]
pub mod webrtc;

pub use energy::EnergyVad;
#[cfg(feature = "vad")]
pub use webrtc::{VadMode, WebRtcVad};

use crate::audio::types::{AudioBuffer, SampleFormat};

/// A span within an audio buffer classified as speech or silence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechSegment {
    /// Whether this segment contains speech.
    pub is_speech: bool,
    /// Start sample index within the source buffer.
    pub start_sample: usize,
    /// Exclusive end sample index within the source buffer.
    pub end_sample: usize,
}

impl SpeechSegment {
    /// Number of samples in this segment.
    pub fn len(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Returns `true` if this segment contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A voice activity detector that classifies audio frames as speech or silence.
pub trait VoiceActivityDetector: Send + Sync {
    /// Returns `true` if `audio` contains any speech.
    fn is_speech(&self, audio: &AudioBuffer) -> bool;

    /// Segment `audio` into alternating speech / silence spans.
    ///
    /// `frame_ms` controls the granularity of analysis (10, 20, or 30 ms).
    fn detect_segments(&self, audio: &AudioBuffer, frame_ms: u32) -> Vec<SpeechSegment>;
}

/// Divisor for normalising i16 PCM samples to the [-1.0, 1.0] range.
/// Equals 2^15 = 32768 (the absolute value of i16::MIN).
const I16_NORMALIZE_DIVISOR: f32 = 32768.0;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute the RMS energy of a PCM buffer in decibels (dBFS).
/// Returns `f32::NEG_INFINITY` for a silent buffer.
pub(crate) fn rms_db(audio: &AudioBuffer) -> f32 {
    let samples = pcm_to_f32(audio);
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_sq = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    if mean_sq == 0.0 {
        return f32::NEG_INFINITY;
    }
    10.0 * mean_sq.log10()
}

/// Convert a raw PCM `AudioBuffer` to `Vec<f32>` normalised to [-1, 1].
pub(crate) fn pcm_to_f32(audio: &AudioBuffer) -> Vec<f32> {
    match audio.config.sample_format {
        SampleFormat::I16 => audio
            .data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / I16_NORMALIZE_DIVISOR)
            .collect(),
        SampleFormat::F32 => audio
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    }
}

/// Convert a raw PCM `AudioBuffer` to mono `Vec<i16>` (mix down if stereo).
#[allow(dead_code)]
pub fn pcm_to_i16_mono(audio: &AudioBuffer) -> Vec<i16> {
    let channels = audio.config.channels as usize;
    match audio.config.sample_format {
        SampleFormat::I16 => {
            let raw: Vec<i16> = audio
                .data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
            if channels <= 1 {
                raw
            } else {
                raw.chunks(channels)
                    .map(|ch| {
                        let sum: i32 = ch.iter().map(|&s| s as i32).sum();
                        (sum / channels as i32) as i16
                    })
                    .collect()
            }
        }
        SampleFormat::F32 => {
            let raw: Vec<f32> = audio
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let mono: Vec<f32> = if channels <= 1 {
                raw
            } else {
                raw.chunks(channels)
                    .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                    .collect()
            };
            mono.iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::types::{AudioBuffer, AudioConfig, SampleFormat};

    fn i16_mono_buffer(samples: &[i16]) -> AudioBuffer {
        AudioBuffer {
            data: samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
            config: AudioConfig {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::I16,
            },
        }
    }

    fn f32_mono_buffer(samples: &[f32]) -> AudioBuffer {
        AudioBuffer {
            data: samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
            config: AudioConfig {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::F32,
            },
        }
    }

    fn stereo_i16_buffer(samples: &[i16]) -> AudioBuffer {
        AudioBuffer {
            data: samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
            config: AudioConfig {
                sample_rate: 16_000,
                channels: 2,
                sample_format: SampleFormat::I16,
            },
        }
    }

    // --- rms_db ---

    #[test]
    fn rms_db_silence_is_neg_infinity() {
        let buf = i16_mono_buffer(&vec![0i16; 160]);
        let db = rms_db(&buf);
        assert_eq!(db, f32::NEG_INFINITY);
    }

    #[test]
    fn rms_db_empty_is_neg_infinity() {
        let buf = i16_mono_buffer(&[]);
        assert_eq!(rms_db(&buf), f32::NEG_INFINITY);
    }

    #[test]
    fn rms_db_full_scale_i16_is_near_0() {
        // Full-scale i16 = 32767/32768 ≈ 1.0 → RMS of DC = ~0 dB
        let samples = vec![i16::MAX; 160];
        let buf = i16_mono_buffer(&samples);
        let db = rms_db(&buf);
        assert!(db > -1.0, "Full-scale DC should be near 0 dBFS, got {db}");
    }

    #[test]
    fn rms_db_f32_full_scale_near_0() {
        let samples = vec![1.0f32; 160];
        let buf = f32_mono_buffer(&samples);
        let db = rms_db(&buf);
        assert!(db > -1.0, "F32 full-scale should be near 0 dBFS, got {db}");
    }

    // --- pcm_to_f32 ---

    #[test]
    fn pcm_to_f32_i16_zero_gives_zero() {
        let buf = i16_mono_buffer(&[0i16; 4]);
        let samples = pcm_to_f32(&buf);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn pcm_to_f32_i16_max_near_one() {
        let buf = i16_mono_buffer(&[i16::MAX]);
        let samples = pcm_to_f32(&buf);
        assert!((samples[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn pcm_to_f32_passthrough_for_f32() {
        let buf = f32_mono_buffer(&[0.5f32, -0.5, 1.0]);
        let samples = pcm_to_f32(&buf);
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.5).abs() < 1e-6);
        assert!((samples[1] + 0.5).abs() < 1e-6);
    }

    // --- pcm_to_i16_mono ---

    #[test]
    fn pcm_to_i16_mono_preserves_mono() {
        let input = vec![100i16, -200, 300];
        let buf = i16_mono_buffer(&input);
        let out = pcm_to_i16_mono(&buf);
        assert_eq!(out, input);
    }

    #[test]
    fn pcm_to_i16_mono_mixes_stereo_i16() {
        // Two channels: L=1000, R=1000 → mono = 1000
        let stereo = vec![1000i16, 1000i16];
        let buf = stereo_i16_buffer(&stereo);
        let mono = pcm_to_i16_mono(&buf);
        assert_eq!(mono.len(), 1);
        assert_eq!(mono[0], 1000);
    }

    // --- SpeechSegment ---

    #[test]
    fn speech_segment_len() {
        let seg = SpeechSegment {
            is_speech: true,
            start_sample: 0,
            end_sample: 320,
        };
        assert_eq!(seg.len(), 320);
        assert!(!seg.is_empty());
    }

    #[test]
    fn speech_segment_empty_when_equal() {
        let seg = SpeechSegment {
            is_speech: false,
            start_sample: 10,
            end_sample: 10,
        };
        assert_eq!(seg.len(), 0);
        assert!(seg.is_empty());
    }
}
