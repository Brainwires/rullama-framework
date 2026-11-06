//! C ABI behaviour that needs neither a model nor an audio device.
//!
//! Nothing here opens the microphone: only calls that are rejected before any
//! device is touched (wrong state, no model, bad path) are made.

use std::ffi::CStr;

use rullama_audio_ffi::*;

fn last_error() -> String {
    let mut buf = [0u8; 512];
    let n = unsafe { rla_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(n >= 0);
    String::from_utf8(buf[..n as usize].to_vec()).unwrap()
}

#[test]
fn idle_abi_contract() {
    let v = unsafe { CStr::from_ptr(rla_version()) }.to_str().unwrap();
    assert!(v.starts_with("rullama-audio-ffi "), "{v}");

    assert_eq!(rla_state(), 0);
    assert_eq!(rla_stt_loaded(), 0);
    assert_eq!(rla_level(), 0.0);
    assert_eq!(rla_captured_ms(), 0);
    assert_eq!(rla_playing(), 0);

    assert_eq!(rla_capture_stop(), -2);
    assert!(last_error().contains("not capturing"));
    let out = c"/tmp/rla-should-not-exist.wav";
    assert_eq!(unsafe { rla_capture_stop_wav(out.as_ptr()) }, -2);

    let mut buf = [0u8; 16];
    assert_eq!(
        unsafe { rla_take_transcript(buf.as_mut_ptr(), buf.len()) },
        -2
    );
    assert_eq!(rla_cancel(), 0);
    assert_eq!(rla_state(), 0);
    assert_eq!(rla_play_stop(), 0);

    // No model: -3 (the WAV is never read).
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phrase.wav\0");
    let n = unsafe { rla_transcribe_wav(fixture.as_ptr().cast(), buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n, -3);
    assert!(last_error().contains("no speech model"));

    // Missing file for playback is an error, raised before any device access.
    let missing = c"/nonexistent/rla.wav";
    assert_eq!(unsafe { rla_play_wav(missing.as_ptr()) }, -1);
    assert!(last_error().contains("/nonexistent/rla.wav"));
    assert_eq!(rla_playing(), 0);

    // rla_last_error truncates and reports the full length.
    let full = last_error();
    let mut small = [0xAAu8; 5];
    let n = unsafe { rla_last_error(small.as_mut_ptr(), small.len()) };
    assert_eq!(n as usize, full.len());
    assert_eq!(&small[..4], &full.as_bytes()[..4]);
    assert_eq!(small[4], 0);
}
