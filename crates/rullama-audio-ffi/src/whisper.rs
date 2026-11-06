//! whisper.cpp transcription via whisper-rs.
//!
//! rullama-hardware's `WhisperStt` reloads the model on every call and is
//! async; the C ABI loads once (`rla_stt_load`) and transcribes synchronously,
//! so this wraps a persistent `WhisperContext` directly.

use std::ffi::c_void;
use std::path::Path;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::dsp::TARGET_RATE;
use crate::engine::Transcriber;

/// whisper.cpp refuses input shorter than 1 s; pad with silence past that.
const MIN_INPUT_SAMPLES: usize = TARGET_RATE as usize * 5 / 4;
/// Upper bound on inference threads.
const MAX_THREADS: usize = 4;

/// Route whisper.cpp / ggml logging into whisper-rs's hooks, which (without
/// the `log_backend` / `tracing_backend` features) discard it.
fn silence_whisper_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(whisper_rs::install_logging_hooks);
}

/// ggml abort callback: `data` is the `&AtomicBool` cancel flag passed to
/// [`WhisperTranscriber::transcribe`], which outlives the `full()` call.
///
/// whisper-rs's `set_abort_callback_safe` is not used: in 0.15 it leaks the
/// boxed closure on every call and its trampoline reads the box as the wrong
/// type.
unsafe extern "C" fn abort_if_cancelled(data: *mut c_void) -> bool {
    // SAFETY: `data` is the `&AtomicBool` installed in `transcribe`, alive for
    // the whole inference that calls back into here.
    let flag = unsafe { &*data.cast::<AtomicBool>() };
    flag.load(Ordering::Relaxed)
}

/// A loaded ggml whisper model.
pub struct WhisperTranscriber {
    ctx: WhisperContext,
    threads: i32,
}

impl WhisperTranscriber {
    /// Load a ggml model file.
    pub fn load(path: &Path) -> Result<Self, String> {
        silence_whisper_logs();
        if !path.is_file() {
            return Err(format!("model file not found: {}", path.display()));
        }
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("model path is not UTF-8: {}", path.display()))?;
        let ctx = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .map_err(|e| format!("failed to load whisper model {}: {e}", path.display()))?;
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(MAX_THREADS);
        Ok(Self {
            ctx,
            threads: threads as i32,
        })
    }
}

/// `[BLANK_AUDIO]`, `[MUSIC]` and similar non-speech markers.
fn is_annotation(s: &str) -> bool {
    s.starts_with('[') && s.ends_with(']')
}

/// Join segment texts, dropping non-speech annotations such as `[BLANK_AUDIO]`.
fn join_segments<'a>(segments: impl Iterator<Item = &'a str>) -> String {
    let words: Vec<&str> = segments
        .map(str::trim)
        .filter(|s| !s.is_empty() && !is_annotation(s))
        .collect();
    words
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

impl Transcriber for WhisperTranscriber {
    fn transcribe(&self, samples: &[f32], cancel: &AtomicBool) -> Result<String, String> {
        let mut padded;
        let input = if samples.len() < MIN_INPUT_SAMPLES {
            padded = samples.to_vec();
            padded.resize(MIN_INPUT_SAMPLES, 0.0);
            &padded[..]
        } else {
            samples
        };

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("whisper: cannot create state: {e}"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_n_threads(self.threads);
        params.set_no_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        // SAFETY: the callback only reads the flag through `data`, and `cancel`
        // outlives `state.full` (the only place whisper.cpp invokes it).
        unsafe {
            params.set_abort_callback(Some(abort_if_cancelled));
            params.set_abort_callback_user_data(std::ptr::from_ref(cancel).cast_mut().cast());
        }
        let ran = state.full(params, input);
        if cancel.load(Ordering::Relaxed) {
            return Err("transcription cancelled".to_string());
        }
        ran.map_err(|e| format!("whisper inference failed: {e}"))?;

        let texts: Vec<String> = (0..state.full_n_segments())
            .filter_map(|i| state.get_segment(i))
            .map(|seg| {
                seg.to_str_lossy()
                    .map(|s| s.into_owned())
                    .map_err(|e| format!("whisper: bad segment text: {e}"))
            })
            .collect::<Result<_, _>>()?;
        Ok(join_segments(texts.iter().map(String::as_str)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_and_cleans_segments() {
        assert_eq!(
            join_segments([" Run the tests", "  in the  second tile. "].into_iter()),
            "Run the tests in the second tile."
        );
        assert_eq!(join_segments(["[BLANK_AUDIO]", " "].into_iter()), "");
    }

    #[test]
    fn missing_model_is_an_error() {
        let err = WhisperTranscriber::load(Path::new("/nonexistent/model.bin"))
            .err()
            .unwrap();
        assert!(err.contains("not found"), "{err}");
    }
}
