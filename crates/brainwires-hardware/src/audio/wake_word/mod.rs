//! Wake word detection for the voice assistant pipeline.
//!
//! A wake word is a short phrase (e.g. "hey assistant", "ok computer") that
//! activates the voice assistant from idle. Two backends are supported:
//!
//! - [`EnergyTriggerDetector`] — zero-dependency RMS-burst fallback. Fires
//!   on any sustained audio energy above a threshold. Feature: `wake-word`.
//! - [`DtwWakeWordDetector`] — in-house DTW (Dynamic Time Warping) over
//!   MFCC features. Pure-Rust DSP, no ML framework, no third-party
//!   wake-word library. Speaker-dependent — callers enroll 3+ reference
//!   recordings before inference. Feature: `wake-word-dtw`.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! # #[cfg(feature = "wake-word-dtw")]
//! # fn _example() -> Result<(), anyhow::Error> {
//! use brainwires_hardware::audio::wake_word::{DtwWakeWordDetector, WakeWordDetector};
//!
//! let mut detector = DtwWakeWordDetector::new();
//! // Enroll 3+ recordings of the user saying the wake phrase
//! // (each 200 ms–2 s of 16 kHz mono i16 PCM):
//! // detector.enroll_template(&recording1)?;
//! // detector.enroll_template(&recording2)?;
//! // detector.enroll_template(&recording3)?;
//! // Then feed live audio in chunks and watch for `Some(...)` returns.
//! # Ok(())
//! # }
//! ```

/// In-house DTW wake-word detector implementation.
#[cfg(feature = "wake-word-dtw")]
pub mod dtw;
/// Pure DTW math over feature-vector sequences (no audio types).
#[cfg(feature = "wake-word-dtw")]
pub mod dtw_algorithm;
/// Energy-burst wake trigger — zero-dependency fallback.
pub mod energy_trigger;
/// MFCC (mel-frequency cepstral coefficient) feature extractor.
#[cfg(feature = "wake-word-dtw")]
pub mod mfcc;

#[cfg(feature = "wake-word-dtw")]
pub use self::dtw::DtwWakeWordDetector;
pub use self::energy_trigger::EnergyTriggerDetector;
#[cfg(feature = "wake-word-dtw")]
pub use self::mfcc::MfccExtractor;

/// A wake word detection event.
#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    /// The name of the matched keyword (from the model file).
    pub keyword: String,
    /// Detection confidence score, 0.0–1.0.
    pub score: f32,
    /// Milliseconds since the detector was created.
    pub timestamp_ms: u64,
}

/// A stateful detector that processes fixed-size audio frames and fires on
/// keyword detection.
///
/// All implementations expect 16 kHz mono i16 PCM. Use
/// [`crate::audio::vad::pcm_to_i16_mono`] to convert an `AudioBuffer` first.
pub trait WakeWordDetector: Send + Sync {
    /// The sample rate this detector expects (always 16 000 Hz).
    fn sample_rate(&self) -> u32;

    /// Number of i16 samples that must be provided per `process_frame` call.
    fn frame_size(&self) -> usize;

    /// Process one audio frame. Returns `Some(detection)` when the wake word
    /// is detected in the provided frame.
    fn process_frame(&mut self, samples: &[i16]) -> Option<WakeWordDetection>;
}
