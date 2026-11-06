//! # rullama-audio-ffi
//!
//! A small C ABI over rullama-hardware's local audio stack — cpal microphone
//! capture, whisper.cpp speech-to-text, WebRTC VAD silence trimming and cpal
//! playback — for hosts that load a shared library (e.g. Deno's
//! `Deno.dlopen`). The header is `include/rullama_audio.h`.
//!
//! All strings are NUL-terminated UTF-8. Return codes: `>= 0` ok, `-1` error
//! (message via [`rla_last_error`]), `-2` wrong state, `-3` no model loaded,
//! `-4` no audio device. Every entry point catches panics and all state sits
//! behind mutexes, so the functions may be called from any thread.
//!
//! Ownership: the library never hands out memory for the caller to free.
//! [`rla_version`] returns a static string; every other output is copied into
//! a caller-owned `(buf, cap)` buffer, and the return value is the full byte
//! length (without the NUL), so `len >= cap` means it was truncated.
//!
//! The last-error slot is process-wide (not per thread), so a host that calls
//! from several threads should read it right after the failing call and
//! serialize its calls if it needs exact attribution. Call [`rla_shutdown`]
//! before unloading the library.

use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

pub mod cpal_io;
pub mod dsp;
pub mod engine;
pub mod error;
pub mod whisper;

use engine::Engine;
use error::ErrorSlot;

static LAST_ERROR: ErrorSlot = ErrorSlot::new_const();
static ENGINE: LazyLock<Engine> =
    LazyLock::new(|| Engine::new(Box::new(cpal_io::CpalSource), &LAST_ERROR));
static PLAYER: LazyLock<cpal_io::Player> = LazyLock::new(cpal_io::Player::default);

static VERSION: &str = concat!("rullama-audio-ffi ", env!("CARGO_PKG_VERSION"), "\0");

/// Run `f`, turning a panic into `on_panic` plus a last-error message.
fn guard<T>(on_panic: T, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(p) => {
            LAST_ERROR.set(format!("internal panic: {}", error::panic_message(&p)));
            on_panic
        }
    }
}

fn fail(code: i32, msg: impl Into<String>) -> i32 {
    LAST_ERROR.set(msg);
    code
}

/// # Safety
/// `p` must be null or a NUL-terminated string.
unsafe fn path_arg(p: *const c_char) -> Result<PathBuf, i32> {
    if p.is_null() {
        return Err(fail(error::ERROR, "path is null"));
    }
    // SAFETY: non-null and NUL-terminated per the caller's contract.
    let s = unsafe { CStr::from_ptr(p) };
    s.to_str()
        .map(PathBuf::from)
        .map_err(|_| fail(error::ERROR, "path is not valid UTF-8"))
}

/// Static version string, e.g. `"rullama-audio-ffi 0.12.0"`.
#[unsafe(no_mangle)]
pub extern "C" fn rla_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

/// Copy the last error message into `buf` (truncated, NUL-terminated) and
/// return its full byte length (0 when there is none).
///
/// # Safety
/// `buf` must be null or valid for `cap` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rla_last_error(buf: *mut u8, cap: usize) -> i32 {
    // SAFETY: forwarded from the caller's contract.
    guard(error::ERROR, || unsafe {
        error::copy_out(&LAST_ERROR.get(), buf, cap)
    })
}

/// Load (or replace) the whisper ggml model. 0 ok, -1 error.
///
/// # Safety
/// `model_path` must be null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rla_stt_load(model_path: *const c_char) -> i32 {
    guard(error::ERROR, || {
        // SAFETY: forwarded from the caller's contract.
        let path = match unsafe { path_arg(model_path) } {
            Ok(p) => p,
            Err(code) => return code,
        };
        match whisper::WhisperTranscriber::load(&path) {
            Ok(t) => {
                ENGINE.set_transcriber(Arc::new(t));
                error::OK
            }
            Err(e) => fail(error::ERROR, e),
        }
    })
}

/// 1 if a model is loaded, else 0.
#[unsafe(no_mangle)]
pub extern "C" fn rla_stt_loaded() -> i32 {
    guard(0, || i32::from(ENGINE.has_transcriber()))
}

/// Open the default input device and start buffering. 0 ok, -2 not idle or
/// done, -4 no input device, -1 other failure.
#[unsafe(no_mangle)]
pub extern "C" fn rla_capture_start() -> i32 {
    guard(error::ERROR, || ENGINE.capture_start())
}

/// Stop capturing and transcribe on a worker thread (state -> 2). -2 not
/// capturing, -3 no model (the capture keeps running).
#[unsafe(no_mangle)]
pub extern "C" fn rla_capture_stop() -> i32 {
    guard(error::ERROR, || ENGINE.capture_stop())
}

/// Stop capturing without transcribing: write the VAD-trimmed audio as a
/// 16 kHz mono 16-bit WAV to `wav_path` (state -> 0). Returns the ms written,
/// -2 not capturing, -1 write error. Needs no model.
///
/// # Safety
/// `wav_path` must be null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rla_capture_stop_wav(wav_path: *const c_char) -> i32 {
    guard(error::ERROR, || {
        // SAFETY: forwarded from the caller's contract.
        let path = match unsafe { path_arg(wav_path) } {
            Ok(p) => p,
            Err(code) => return code,
        };
        ENGINE.capture_stop_wav(&path)
    })
}

/// Stop capture / abandon a transcription in flight, drop audio, state -> 0. Always 0.
#[unsafe(no_mangle)]
pub extern "C" fn rla_cancel() -> i32 {
    guard(error::OK, || ENGINE.cancel())
}

/// 0 idle, 1 capturing, 2 transcribing, 3 done, 4 error.
#[unsafe(no_mangle)]
pub extern "C" fn rla_state() -> i32 {
    guard(engine::STATE_ERROR, || ENGINE.state())
}

/// Input RMS (0..1) of the last ~100 ms while capturing, else 0.
#[unsafe(no_mangle)]
pub extern "C" fn rla_level() -> f32 {
    guard(0.0, || ENGINE.level())
}

/// Milliseconds captured in this (or the last stopped) capture.
#[unsafe(no_mangle)]
pub extern "C" fn rla_captured_ms() -> u32 {
    guard(0, || ENGINE.captured_ms())
}

/// In state 3: copy the transcript and return its byte length; it is consumed
/// (state -> 0) only if it fit (`len < cap`). -2 outside state 3; in state 4
/// returns -1 and resets to 0.
///
/// # Safety
/// `buf` must be null or valid for `cap` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rla_take_transcript(buf: *mut u8, cap: usize) -> i32 {
    // SAFETY: forwarded from the caller's contract.
    guard(error::ERROR, || unsafe { ENGINE.take_transcript(buf, cap) })
}

/// Synchronously transcribe a WAV file (any rate, mono/stereo, 16-bit int or
/// 32-bit float). Copies the text like [`rla_take_transcript`] and returns its
/// byte length, or -1 / -3. Does not touch the capture state.
///
/// # Safety
/// `wav_path` must be null or a NUL-terminated string; `buf` must be null or
/// valid for `cap` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rla_transcribe_wav(
    wav_path: *const c_char,
    buf: *mut u8,
    cap: usize,
) -> i32 {
    guard(error::ERROR, || {
        // SAFETY: forwarded from the caller's contract.
        let path = match unsafe { path_arg(wav_path) } {
            Ok(p) => p,
            Err(code) => return code,
        };
        if !ENGINE.has_transcriber() {
            return fail(
                error::NO_MODEL,
                "no speech model loaded (call rla_stt_load)",
            );
        }
        let pcm = match dsp::read_wav_pcm(&path) {
            Ok(p) => p,
            Err(e) => return fail(error::ERROR, e),
        };
        let mono = dsp::downmix(&pcm.samples, pcm.channels);
        match ENGINE.transcribe_now(&mono, pcm.rate) {
            // SAFETY: forwarded from the caller's contract.
            Ok(text) => unsafe { error::copy_out(&text, buf, cap) },
            Err(code) => code,
        }
    })
}

/// Play a WAV on the default output device, stopping any playback in
/// progress. Returns once the device is open: 0 ok, -1 error, -4 no device.
///
/// # Safety
/// `wav_path` must be null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rla_play_wav(wav_path: *const c_char) -> i32 {
    guard(error::ERROR, || {
        // SAFETY: forwarded from the caller's contract.
        let path = match unsafe { path_arg(wav_path) } {
            Ok(p) => p,
            Err(code) => return code,
        };
        match PLAYER.play(&path) {
            Ok(()) => error::OK,
            Err((code, msg)) => fail(code, msg),
        }
    })
}

/// Stop playback. Always 0.
#[unsafe(no_mangle)]
pub extern "C" fn rla_play_stop() -> i32 {
    guard(error::OK, || {
        PLAYER.stop();
        error::OK
    })
}

/// 1 while playing, else 0.
#[unsafe(no_mangle)]
pub extern "C" fn rla_playing() -> i32 {
    guard(0, || i32::from(PLAYER.playing()))
}

/// Release everything before the host unloads the library: stop capture and
/// playback (their streams are closed and threads joined), abort and join any
/// transcription worker, abort `rla_transcribe_wav` calls in flight on other
/// threads (they return -1), and unload the model. The library stays usable
/// afterwards (load a model again). Always 0.
#[unsafe(no_mangle)]
pub extern "C" fn rla_shutdown() -> i32 {
    guard(error::OK, || {
        PLAYER.stop();
        ENGINE.shutdown()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_static_and_nul_terminated() {
        let v = unsafe { CStr::from_ptr(rla_version()) }.to_str().unwrap();
        assert_eq!(v, concat!("rullama-audio-ffi ", env!("CARGO_PKG_VERSION")));
    }

    // One test so nothing else writes LAST_ERROR between the call and the check.
    #[test]
    fn panics_and_bad_paths_become_error_codes() {
        let last = || {
            let mut buf = [0u8; 256];
            let n = unsafe { rla_last_error(buf.as_mut_ptr(), buf.len()) } as usize;
            String::from_utf8(buf[..n].to_vec()).unwrap()
        };
        assert_eq!(guard(-1, || panic!("kaboom")), -1);
        assert_eq!(last(), "internal panic: kaboom");
        assert_eq!(guard(-1, || 5), 5);

        assert_eq!(unsafe { rla_stt_load(std::ptr::null()) }, error::ERROR);
        assert_eq!(last(), "path is null");
        let bad = [0xFFu8, 0xFE, 0];
        assert_eq!(unsafe { rla_play_wav(bad.as_ptr().cast()) }, error::ERROR);
        assert_eq!(last(), "path is not valid UTF-8");
        let missing = c"/nonexistent/model.bin";
        assert_eq!(unsafe { rla_stt_load(missing.as_ptr()) }, error::ERROR);
        assert!(last().contains("not found"));
        assert_eq!(rla_stt_loaded(), 0);
    }
}
