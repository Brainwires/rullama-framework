use std::time::Instant;

use super::{WakeWordDetection, WakeWordDetector};
use crate::audio::types::{AudioBuffer, AudioConfig, SampleFormat};
use crate::audio::vad::rms_db;

/// A simple energy-based wake trigger — fires when audio energy stays above a
/// threshold for a minimum number of consecutive frames.
///
/// Useful for "tap-to-wake" or "clap-to-wake" style activation, or as a
/// zero-dependency fallback when a trained wake word model is not available.
///
/// Always available with the `wake-word` feature.
///
/// # Example
/// ```rust,no_run
/// use rullama_hardware::audio::wake_word::{EnergyTriggerDetector, WakeWordDetector};
/// // Triggers when audio exceeds -20 dB for 2+ consecutive frames.
/// let mut detector = EnergyTriggerDetector::new(-20.0, 2, 16000);
/// ```
pub struct EnergyTriggerDetector {
    /// Energy threshold in dBFS. Frames above this level count as "active".
    pub threshold_db: f32,
    /// Minimum consecutive active frames required to fire. Default: 2.
    pub min_consecutive: u32,
    /// Sample rate (must match the audio being fed). Default: 16 000 Hz.
    pub sample_rate: u32,
    /// Name returned in the detection event.
    pub trigger_name: String,

    consecutive: u32,
    start: Instant,
}

impl EnergyTriggerDetector {
    /// Create a new energy trigger detector.
    pub fn new(threshold_db: f32, min_consecutive: u32, sample_rate: u32) -> Self {
        Self {
            threshold_db,
            min_consecutive,
            sample_rate,
            trigger_name: "energy_trigger".to_string(),
            consecutive: 0,
            start: Instant::now(),
        }
    }

    /// Set the trigger name returned in detection events.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.trigger_name = name.into();
        self
    }
}

impl Default for EnergyTriggerDetector {
    fn default() -> Self {
        Self::new(-30.0, 3, 16_000)
    }
}

impl WakeWordDetector for EnergyTriggerDetector {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn frame_size(&self) -> usize {
        // 30 ms at the configured sample rate
        (self.sample_rate as usize * 30 / 1000).max(1)
    }

    fn process_frame(&mut self, samples: &[i16]) -> Option<WakeWordDetection> {
        // Convert samples to an AudioBuffer for rms_db
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let buf = AudioBuffer {
            data,
            config: AudioConfig {
                sample_rate: self.sample_rate,
                channels: 1,
                sample_format: SampleFormat::I16,
            },
        };

        let db = rms_db(&buf);
        if db > self.threshold_db {
            self.consecutive += 1;
            if self.consecutive >= self.min_consecutive {
                self.consecutive = 0;
                return Some(WakeWordDetection {
                    keyword: self.trigger_name.clone(),
                    score: (db - self.threshold_db).clamp(0.0, 40.0) / 40.0,
                    timestamp_ms: self.start.elapsed().as_millis() as u64,
                });
            }
        } else {
            self.consecutive = 0;
        }

        None
    }
}
