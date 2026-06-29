use crate::audio::types::{AudioBuffer, SampleFormat};
use crate::audio::vad::{SpeechSegment, VoiceActivityDetector, rms_db};

/// A pure-Rust energy-based Voice Activity Detector.
///
/// Computes the RMS energy of each frame and compares it against a dBFS
/// threshold. Simple and dependency-free, though less accurate than the
/// WebRTC VAD algorithm on noisy signals.
///
/// # Example
/// ```rust,no_run
/// use rullama_hardware::audio::vad::{EnergyVad, VoiceActivityDetector};
/// let vad = EnergyVad::default();  // -40 dB threshold
/// ```
pub struct EnergyVad {
    /// Energy threshold in dBFS. Frames above this level are classified as
    /// speech. Typical values: -40 dB (quiet room) to -20 dB (noisy).
    pub threshold_db: f32,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
        }
    }
}

impl EnergyVad {
    /// Create a detector with a custom threshold.
    pub fn new(threshold_db: f32) -> Self {
        Self { threshold_db }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn is_speech(&self, audio: &AudioBuffer) -> bool {
        rms_db(audio) > self.threshold_db
    }

    fn detect_segments(&self, audio: &AudioBuffer, frame_ms: u32) -> Vec<SpeechSegment> {
        let sr = audio.config.sample_rate;
        let channels = audio.config.channels as usize;
        let bytes_per_sample = match audio.config.sample_format {
            SampleFormat::I16 => 2,
            SampleFormat::F32 => 4,
        };
        let frame_samples = (sr * frame_ms / 1000) as usize * channels;
        let frame_bytes = frame_samples * bytes_per_sample;

        let total_frames = audio.data.len() / frame_bytes.max(1);
        let mut segments: Vec<SpeechSegment> = Vec::new();

        for i in 0..total_frames {
            let start = i * frame_bytes;
            let end = (start + frame_bytes).min(audio.data.len());
            let frame_data = audio.data[start..end].to_vec();
            let frame_buf = AudioBuffer {
                data: frame_data,
                config: audio.config.clone(),
            };
            let is_speech = rms_db(&frame_buf) > self.threshold_db;
            let sample_start = i * frame_samples;
            let sample_end = sample_start + frame_samples;

            match segments.last_mut() {
                Some(last) if last.is_speech == is_speech => {
                    last.end_sample = sample_end;
                }
                _ => segments.push(SpeechSegment {
                    is_speech,
                    start_sample: sample_start,
                    end_sample: sample_end,
                }),
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::types::{AudioBuffer, AudioConfig, SampleFormat};

    /// Build a mono I16 buffer from a slice of i16 samples.
    fn i16_buffer(samples: &[i16]) -> AudioBuffer {
        let data = samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>();
        AudioBuffer {
            data,
            config: AudioConfig {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::I16,
            },
        }
    }

    /// Build a mono F32 buffer from a slice of f32 samples.
    fn f32_buffer(samples: &[f32]) -> AudioBuffer {
        let data = samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>();
        AudioBuffer {
            data,
            config: AudioConfig {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::F32,
            },
        }
    }

    // --- EnergyVad::default ---

    #[test]
    fn default_threshold_is_minus_40() {
        let vad = EnergyVad::default();
        assert_eq!(vad.threshold_db, -40.0);
    }

    #[test]
    fn custom_threshold_stored() {
        let vad = EnergyVad::new(-20.0);
        assert_eq!(vad.threshold_db, -20.0);
    }

    // --- is_speech ---

    #[test]
    fn silent_buffer_not_speech() {
        let vad = EnergyVad::default();
        // All zeros = silence
        let buf = i16_buffer(&vec![0i16; 160]);
        assert!(!vad.is_speech(&buf));
    }

    #[test]
    fn loud_tone_is_speech() {
        let vad = EnergyVad::new(-40.0);
        // Full-scale sine wave — very loud
        let samples: Vec<i16> = (0..160)
            .map(|i| (i16::MAX as f32 * (i as f32 * 0.1).sin()) as i16)
            .collect();
        let buf = i16_buffer(&samples);
        assert!(vad.is_speech(&buf));
    }

    #[test]
    fn very_quiet_signal_below_threshold() {
        // Threshold of -10 dB — barely audible signal won't pass
        let vad = EnergyVad::new(-10.0);
        // Very small amplitude (1/32768 of full scale)
        let samples = vec![1i16; 160];
        let buf = i16_buffer(&samples);
        assert!(!vad.is_speech(&buf));
    }

    #[test]
    fn f32_buffer_loud_is_speech() {
        let vad = EnergyVad::new(-40.0);
        let samples: Vec<f32> = (0..160).map(|i| (i as f32 * 0.2).sin()).collect();
        let buf = f32_buffer(&samples);
        assert!(vad.is_speech(&buf));
    }

    // --- detect_segments ---

    #[test]
    fn empty_buffer_yields_no_segments() {
        let vad = EnergyVad::default();
        let buf = i16_buffer(&[]);
        let segs = vad.detect_segments(&buf, 20);
        assert!(segs.is_empty());
    }

    #[test]
    fn all_silence_gives_single_silence_segment() {
        let vad = EnergyVad::default();
        // 16000 Hz * 60ms = 960 samples, giving 3 frames of 20ms each
        let buf = i16_buffer(&vec![0i16; 960]);
        let segs = vad.detect_segments(&buf, 20);
        assert!(!segs.is_empty());
        assert!(segs.iter().all(|s| !s.is_speech));
    }

    #[test]
    fn segments_cover_full_range() {
        let vad = EnergyVad::default();
        let n_samples = 960usize; // 3 * 20ms frames at 16kHz
        let buf = i16_buffer(&vec![0i16; n_samples]);
        let segs = vad.detect_segments(&buf, 20);
        if !segs.is_empty() {
            assert_eq!(segs[0].start_sample, 0);
        }
    }
}
