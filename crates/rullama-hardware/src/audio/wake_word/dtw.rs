//! In-house DTW + MFCC wake-word detector.
//!
//! Pure-Rust DSP pipeline, no ML framework, no third-party wake-word
//! library. Speaker-dependent: the caller enrolls 3+ reference recordings
//! of the wake phrase via [`DtwWakeWordDetector::enroll_template`], then
//! pumps live audio into [`WakeWordDetector::process_frame`].
//!
//! Sliding-window inference: the detector keeps a circular buffer of the
//! most recent ~1.5× longest-template's worth of audio samples. On every
//! call it MFCC-extracts the buffer, computes DTW against each stored
//! template, and fires when the minimum distance dips below the
//! configured threshold.

use std::collections::VecDeque;
use std::time::Instant;

use super::dtw_algorithm::dtw_distance;
use super::mfcc::MfccExtractor;
use super::{WakeWordDetection, WakeWordDetector};
use crate::audio::error::{AudioError, AudioResult};

/// Detector defaults — chosen to work out of the box for 16 kHz mono speech.
const DEFAULT_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_FRAME_LEN: usize = 400; // 25 ms at 16 kHz
const DEFAULT_HOP: usize = 160; // 10 ms at 16 kHz
const DEFAULT_NUM_COEFFS: usize = 12;
const DEFAULT_THRESHOLD: f32 = 30.0;

/// Minimum / maximum enrollable recording length, in samples at 16 kHz.
const MIN_ENROLL_SAMPLES: usize = 3_200; // 200 ms
const MAX_ENROLL_SAMPLES: usize = 32_000; // 2 s

/// In-house wake-word detector using DTW over MFCC features.
///
/// Speaker-dependent — enroll the user's wake word 3+ times before
/// inference. No model file, no ML framework, no third-party wake-word
/// crate.
///
/// Feed audio via the [`WakeWordDetector::process_frame`] trait method
/// (any chunk size; the detector buffers internally).
///
/// # Example
/// ```rust,no_run
/// use rullama_hardware::audio::wake_word::{DtwWakeWordDetector, WakeWordDetector};
///
/// let mut detector = DtwWakeWordDetector::new();
/// // Enroll three recordings of the wake word (each 200 ms–2 s @ 16 kHz):
/// // detector.enroll_template(&recording1)?;
/// // detector.enroll_template(&recording2)?;
/// // detector.enroll_template(&recording3)?;
/// // Then feed live audio in chunks:
/// // if let Some(det) = detector.process_frame(&samples) { /* fire */ }
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct DtwWakeWordDetector {
    threshold: f32,
    extractor: MfccExtractor,
    templates: Vec<Vec<Vec<f32>>>,
    /// Longest template length in MFCC frames. Drives the rolling-window
    /// length below.
    longest_template_frames: usize,
    /// Rolling buffer of recent PCM samples. Capped at ~1.5× the longest
    /// enrolled template (in samples) so we never look further back than
    /// the wake phrase could possibly span.
    rolling: VecDeque<i16>,
    start: Instant,
}

impl Default for DtwWakeWordDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DtwWakeWordDetector {
    /// Construct a detector with the defaults documented at the type level
    /// (16 kHz, 25 ms frames, 10 ms hop, 12 MFCC coefficients, threshold
    /// 30.0).
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            extractor: MfccExtractor::new(
                DEFAULT_SAMPLE_RATE,
                DEFAULT_FRAME_LEN,
                DEFAULT_HOP,
                DEFAULT_NUM_COEFFS,
            ),
            templates: Vec::new(),
            longest_template_frames: 0,
            rolling: VecDeque::new(),
            start: Instant::now(),
        }
    }

    /// Override the DTW detection threshold. Lower is stricter; the default
    /// works for clearly-enunciated 1–2-word wake phrases recorded in a
    /// quiet room.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Enroll one reference recording of the wake word.
    ///
    /// `samples` must be 16-bit PCM at 16 kHz. Returns an error if the
    /// recording is shorter than 200 ms or longer than 2 s — the typical
    /// wake phrase fits comfortably in 250 ms–1.5 s.
    pub fn enroll_template(&mut self, samples: &[i16]) -> AudioResult<()> {
        if samples.len() < MIN_ENROLL_SAMPLES {
            return Err(AudioError::Format(format!(
                "wake-word enrollment too short: {} samples (need ≥{})",
                samples.len(),
                MIN_ENROLL_SAMPLES
            )));
        }
        if samples.len() > MAX_ENROLL_SAMPLES {
            return Err(AudioError::Format(format!(
                "wake-word enrollment too long: {} samples (max {})",
                samples.len(),
                MAX_ENROLL_SAMPLES
            )));
        }
        let mfcc = self.extractor.extract(samples);
        if mfcc.is_empty() {
            return Err(AudioError::Format(
                "wake-word enrollment yielded zero MFCC frames".into(),
            ));
        }
        if mfcc.len() > self.longest_template_frames {
            self.longest_template_frames = mfcc.len();
        }
        self.templates.push(mfcc);
        Ok(())
    }

    /// Clear the rolling sample buffer. Typically called after a confirmed
    /// detection so the next match has to be a fresh utterance.
    pub fn reset_window(&mut self) {
        self.rolling.clear();
    }

    /// Number of enrolled reference templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Configured detection threshold (DTW distance — lower fires stricter).
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Internal helper: append samples, advance the rolling buffer, return
    /// the best DTW distance across all templates (or `None` if no
    /// templates are enrolled or the buffer is too short).
    fn dtw_best(&mut self, samples: &[i16]) -> Option<f32> {
        if self.templates.is_empty() {
            return None;
        }

        // Append, then trim the front so the buffer never exceeds
        // ~1.5× the longest template's audio span. (longest_template_frames
        // is in MFCC frames; convert to samples via hop and add one frame
        // worth of front-pad for the first MFCC window.)
        for &s in samples {
            self.rolling.push_back(s);
        }
        let max_samples = (self.longest_template_frames as f32 * 1.5) as usize
            * self.extractor.hop()
            + self.extractor.frame_len();
        while self.rolling.len() > max_samples {
            self.rolling.pop_front();
        }

        if self.rolling.len() < self.extractor.frame_len() {
            return None;
        }

        // Snapshot the rolling buffer into a contiguous slice for MFCC.
        let buf: Vec<i16> = self.rolling.iter().copied().collect();
        let live = self.extractor.extract(&buf);
        if live.is_empty() {
            return None;
        }

        // Compare against each template. For sliding-window inference we
        // take the last `template_len` frames of the live buffer so the
        // DTW path is properly aligned.
        let mut best: Option<f32> = None;
        for template in &self.templates {
            let tlen = template.len();
            if tlen == 0 {
                continue;
            }
            let window: &[Vec<f32>] = if live.len() >= tlen {
                &live[live.len() - tlen..]
            } else {
                &live[..]
            };
            let d = dtw_distance(window, template);
            if d.is_finite() && best.is_none_or(|b| d < b) {
                best = Some(d);
            }
        }
        best
    }
}

impl WakeWordDetector for DtwWakeWordDetector {
    fn sample_rate(&self) -> u32 {
        self.extractor.sample_rate()
    }

    fn frame_size(&self) -> usize {
        // For the trait interface we report the MFCC frame length — callers
        // wiring this into the voice-assistant pipeline still feed chunks
        // sized to this. Internally `process_frame` accepts any chunk size.
        self.extractor.frame_len()
    }

    fn process_frame(&mut self, samples: &[i16]) -> Option<WakeWordDetection> {
        let d = self.dtw_best(samples)?;
        if d >= self.threshold {
            return None;
        }
        let timestamp_ms = self.start.elapsed().as_millis() as u64;
        Some(WakeWordDetection {
            keyword: "wake".to_string(),
            // Map DTW distance into a 0..1 confidence: 0 = at threshold,
            // 1 = perfect match.
            score: (1.0 - d / self.threshold).clamp(0.0, 1.0),
            timestamp_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_16k(samples: usize, freq_hz: f32) -> Vec<i16> {
        (0..samples)
            .map(|n| {
                let t = n as f32 / 16_000.0;
                ((2.0 * PI * freq_hz * t).sin() * 10_000.0) as i16
            })
            .collect()
    }

    fn white_noise_16k(samples: usize, seed: u32) -> Vec<i16> {
        // Tiny LCG so tests stay deterministic and stdlib-only.
        let mut state = seed as u64;
        (0..samples)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as i32 % 16_000) as i16
            })
            .collect()
    }

    #[test]
    fn detector_template_count_matches_enrollments() {
        let mut d = DtwWakeWordDetector::new();
        d.enroll_template(&sine_16k(8_000, 400.0)).unwrap();
        d.enroll_template(&sine_16k(8_000, 500.0)).unwrap();
        d.enroll_template(&sine_16k(8_000, 600.0)).unwrap();
        assert_eq!(d.template_count(), 3);
    }

    #[test]
    fn detector_rejects_too_short_recording() {
        let mut d = DtwWakeWordDetector::new();
        let too_short = sine_16k(1_600, 440.0); // 100 ms
        let err = d.enroll_template(&too_short).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("too short"),
            "expected 'too short' in error message, got: {msg}"
        );
    }

    #[test]
    fn detector_rejects_too_long_recording() {
        let mut d = DtwWakeWordDetector::new();
        let too_long = sine_16k(48_000, 440.0); // 3 s
        let err = d.enroll_template(&too_long).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("too long"),
            "expected 'too long' in error message, got: {msg}"
        );
    }

    #[test]
    fn detector_fires_on_identical_audio() {
        // Enroll a template, then feed the same audio in 10 ms chunks.
        // Permissive threshold so we don't false-fail on float drift.
        let mut d = DtwWakeWordDetector::new().with_threshold(100.0);
        let template_audio = sine_16k(8_000, 440.0); // 0.5 s of 440 Hz
        d.enroll_template(&template_audio).unwrap();

        let mut fired = false;
        for chunk in template_audio.chunks(160) {
            if d.process_frame(chunk).is_some() {
                fired = true;
                break;
            }
        }
        assert!(fired, "detector should fire when fed the enrolled audio");
    }

    #[test]
    fn detector_does_not_fire_on_unrelated_noise() {
        // Enroll a clean tone with a tight threshold; feed white noise.
        let mut d = DtwWakeWordDetector::new().with_threshold(20.0);
        d.enroll_template(&sine_16k(8_000, 440.0)).unwrap();

        let noise = white_noise_16k(32_000, 0xCAFEBABE);
        for chunk in noise.chunks(160) {
            if let Some(det) = d.process_frame(chunk) {
                panic!(
                    "unexpected wake-word fire on white noise; score = {}, keyword = {}",
                    det.score, det.keyword
                );
            }
        }
    }

    #[test]
    fn detector_with_no_templates_never_fires() {
        let mut d = DtwWakeWordDetector::new();
        let audio = sine_16k(16_000, 440.0);
        for chunk in audio.chunks(160) {
            assert!(
                d.process_frame(chunk).is_none(),
                "no-template detector must never fire"
            );
        }
    }

    #[test]
    fn detector_reset_window_clears_state() {
        // Permissive threshold so the same template-audio replay fires.
        let mut d = DtwWakeWordDetector::new().with_threshold(100.0);
        let template_audio = sine_16k(8_000, 440.0);
        d.enroll_template(&template_audio).unwrap();

        // Prime the rolling buffer all the way through with template audio.
        for chunk in template_audio.chunks(160) {
            let _ = d.process_frame(chunk);
        }
        // At this point the rolling buffer has the full template's worth
        // of samples. Reset it.
        d.reset_window();

        // After reset, the buffer is empty. Feeding 3 × 80 = 240 sub-frame-
        // length samples keeps the buffer below `frame_len` (400), so MFCC
        // can't even extract a frame and `dtw_best` short-circuits to None.
        // That's the strict invariant: a freshly reset detector must not
        // fire on less-than-one-frame of new input, regardless of what it
        // was primed with.
        let tiny = vec![0i16; 80];
        for i in 0..3 {
            assert!(
                d.process_frame(&tiny).is_none(),
                "after reset_window, sub-frame input must not fire (iteration {i})"
            );
        }
    }

    #[test]
    fn detector_with_threshold_low_does_not_fire_on_identical_audio() {
        // Stricter mirror of `fires_on_identical_audio`: with a threshold of
        // 0.0, even a perfect-match DTW distance would have to *also* be
        // exactly zero to fire (`d < threshold` => never on float DTW).
        // Proves the threshold knob is actually wired up — otherwise the
        // identical-audio test would still pass here.
        let mut d = DtwWakeWordDetector::new().with_threshold(0.0);
        let template_audio = sine_16k(8_000, 440.0);
        d.enroll_template(&template_audio).unwrap();

        for chunk in template_audio.chunks(160) {
            assert!(
                d.process_frame(chunk).is_none(),
                "threshold=0.0 must never fire (even identical audio has non-zero DTW)"
            );
        }
    }

    #[test]
    fn detector_with_threshold_high_fires_on_unrelated_audio() {
        // Inverse: with an absurdly high threshold, even weakly-similar
        // audio should fire. Combined with the previous test, this proves
        // `with_threshold` controls the firing behaviour.
        let mut d = DtwWakeWordDetector::new().with_threshold(1.0e9);
        d.enroll_template(&sine_16k(8_000, 440.0)).unwrap();

        // Feed completely unrelated noise. With a high enough threshold,
        // even that should "fire" since DTW distance, however large, is
        // below 1e9.
        let mut fired = false;
        let noise = white_noise_16k(16_000, 0xDEADBEEF);
        for chunk in noise.chunks(160) {
            if d.process_frame(chunk).is_some() {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "threshold=1e9 must accept anything once the rolling buffer fills"
        );
    }

    #[test]
    fn detector_threshold_getter_reflects_with_threshold() {
        let d = DtwWakeWordDetector::new().with_threshold(42.5);
        assert!(
            (d.threshold() - 42.5).abs() < 1e-6,
            "with_threshold must update the field exposed by threshold()"
        );
    }

    #[test]
    fn detector_second_enrollment_of_same_audio_does_not_regress() {
        // Sanity: enrolling the same audio twice should not lower the
        // detection rate. Compare the score(s) seen with 1 template vs
        // with 2 identical templates — the second template provides another
        // matching path, so the best-DTW (minimum across templates) can
        // only stay the same or improve.
        let mut d1 = DtwWakeWordDetector::new().with_threshold(200.0);
        let template_audio = sine_16k(8_000, 440.0);
        d1.enroll_template(&template_audio).unwrap();

        let mut best_score_1: f32 = 0.0;
        for chunk in template_audio.chunks(160) {
            if let Some(det) = d1.process_frame(chunk) {
                best_score_1 = best_score_1.max(det.score);
            }
        }

        let mut d2 = DtwWakeWordDetector::new().with_threshold(200.0);
        d2.enroll_template(&template_audio).unwrap();
        d2.enroll_template(&template_audio).unwrap();
        let mut best_score_2: f32 = 0.0;
        for chunk in template_audio.chunks(160) {
            if let Some(det) = d2.process_frame(chunk) {
                best_score_2 = best_score_2.max(det.score);
            }
        }

        // Score is `1.0 - d/threshold`; the higher score reflects the lower
        // (better) DTW distance. With two identical templates the best DTW
        // distance can only be ≤ what one template alone produced.
        assert!(
            best_score_2 >= best_score_1 - 1e-4,
            "two identical templates regressed score: 1-tpl={best_score_1}, 2-tpl={best_score_2}"
        );
    }
}
