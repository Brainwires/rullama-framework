use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use rustpotter::{Rustpotter, RustpotterConfig};
use tracing::debug;

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::wake_word::{WakeWordDetection, WakeWordDetector};

/// Wake word detector backed by [`rustpotter`] — pure Rust, Apache 2.0.
///
/// Requires one or more `.rpw` keyword model files (record samples with
/// `rustpotter-cli` to create them). Supports both DTW and ONNX neural models.
///
/// Feature: `wake-word-rustpotter`
///
/// # Example
/// ```rust,no_run
/// use brainwires_hardware::audio::wake_word::{RustpotterDetector, WakeWordDetector};
/// let mut d = RustpotterDetector::from_model_file("hey_assistant.rpw", 0.5).unwrap();
/// // Feed detector.frame_size() i16 samples per call:
/// // if let Some(det) = d.process_frame(&samples) { println!("Wake word: {}", det.keyword); }
/// ```
pub struct RustpotterDetector {
    // `Rustpotter` is `Send` but not `Sync` (it stores `Box<dyn WakewordDetector>`,
    // which is `Send`-only). Wrap it in `Mutex` so the surrounding type satisfies
    // the `WakeWordDetector: Send + Sync` bound. The mutex is uncontended in
    // practice — `process_frame` is called from a single audio thread.
    inner: Mutex<Rustpotter>,
    frame_size: usize,
    start: Instant,
}

impl RustpotterDetector {
    /// Load a single `.rpw` model file.
    pub fn from_model_file(path: impl AsRef<Path>, threshold: f32) -> AudioResult<Self> {
        Self::from_model_files(&[path.as_ref()], threshold)
    }

    /// Load multiple `.rpw` model files.
    pub fn from_model_files(paths: &[impl AsRef<Path>], threshold: f32) -> AudioResult<Self> {
        let mut config = RustpotterConfig::default();
        config.detector.threshold = threshold;

        let mut inner = Rustpotter::new(&config)
            .map_err(|e| AudioError::Device(format!("rustpotter init failed: {e}")))?;

        for (idx, path) in paths.iter().enumerate() {
            let p = path.as_ref();
            // `add_wakeword_from_file` now requires a unique `key`. Derive one
            // from the file stem (falling back to the index) so multiple models
            // with overlapping internal names still load.
            let key = p
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_owned())
                .unwrap_or_else(|| format!("wakeword-{idx}"));
            inner
                .add_wakeword_from_file(&key, p.to_str().unwrap_or_default())
                .map_err(|e| {
                    AudioError::Device(format!(
                        "failed to load wake word model {}: {e}",
                        p.display()
                    ))
                })?;
        }

        let frame_size = inner.get_samples_per_frame();
        debug!("RustpotterDetector ready — frame_size={frame_size}");

        Ok(Self {
            inner: Mutex::new(inner),
            frame_size,
            start: Instant::now(),
        })
    }
}

impl WakeWordDetector for RustpotterDetector {
    fn sample_rate(&self) -> u32 {
        16_000
    }

    fn frame_size(&self) -> usize {
        self.frame_size
    }

    fn process_frame(&mut self, samples: &[i16]) -> Option<WakeWordDetection> {
        // Slice-based API on the Brainwires fork avoids the per-frame
        // `Vec<i16>` allocation that the upstream by-value
        // `process_samples` would have required (`i16` implements
        // rustpotter's `Sample` trait, which already implies `Copy`).
        let result = self.inner.get_mut().ok()?.process_samples_slice(samples)?;
        let timestamp_ms = self.start.elapsed().as_millis() as u64;
        debug!(
            keyword = %result.name,
            score = result.score,
            "Wake word detected"
        );
        Some(WakeWordDetection {
            keyword: result.name,
            score: result.score,
            timestamp_ms,
        })
    }
}
