//! Wake word detection for the voice assistant pipeline.
//!
//! A wake word is a short phrase (e.g. "hey assistant", "ok computer") that
//! activates the voice assistant from idle. Two backends are supported:
//!
//! - `RustpotterDetector` — pure-Rust, Apache 2.0, no native deps.
//!   Uses DTW or ONNX neural models (`.rpw` files). Feature: `wake-word`.
//!
//! ## Quick start
//!
//! ```rust,no_run,ignore
//! use brainwires_hardware::audio::wake_word::{RustpotterDetector, WakeWordDetector};
//!
//! // Load a .rpw model (record samples with rustpotter-cli to create one)
//! let mut detector = RustpotterDetector::from_model_file("hey_assistant.rpw", 0.5).unwrap();
//! println!("Feed {} i16 samples per frame", detector.frame_size());
//! ```

/// Energy-burst wake trigger — zero-dependency fallback.
pub mod energy_trigger;
/// Rustpotter-backed wake word detector (pure Rust, `.rpw` models).
#[cfg(feature = "wake-word-rustpotter")]
pub mod rustpotter;

pub use self::energy_trigger::EnergyTriggerDetector;
#[cfg(feature = "wake-word-rustpotter")]
pub use self::rustpotter::RustpotterDetector;

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
