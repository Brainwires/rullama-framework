//! End-to-end whisper transcription through the C ABI.
//!
//! Skipped unless `RLA_WHISPER_MODEL` points at a ggml whisper model, e.g.
//! `RLA_WHISPER_MODEL=/path/ggml-base.en-q5_1.bin cargo test -p rullama-audio-ffi`.

use std::ffi::CString;

use rullama_audio_ffi::*;

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn transcribes_fixture_wav() {
    let Ok(model) = std::env::var("RLA_WHISPER_MODEL") else {
        eprintln!(
            "skipping transcribes_fixture_wav: set RLA_WHISPER_MODEL to a ggml whisper model"
        );
        return;
    };
    let model = CString::new(model).unwrap();
    assert_eq!(
        unsafe { rla_stt_load(model.as_ptr()) },
        0,
        "rla_stt_load failed"
    );
    assert_eq!(rla_stt_loaded(), 1);

    let wav = CString::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/phrase.wav"
    ))
    .unwrap();
    let mut buf = vec![0u8; 1024];
    let n = unsafe { rla_transcribe_wav(wav.as_ptr(), buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0, "rla_transcribe_wav returned {n}");
    let n = n as usize;
    assert_eq!(buf[n], 0, "not NUL-terminated");
    let text = std::str::from_utf8(&buf[..n]).unwrap();
    eprintln!("transcript: {text:?}");
    let norm = normalize(text);
    assert!(norm.contains("run the tests"), "{text:?}");
    assert!(norm.contains("second tile"), "{text:?}");
    assert_eq!(text, text.trim());

    // Too-small buffer: truncated, NUL-terminated, full length returned.
    let mut small = [0xAAu8; 8];
    let m = unsafe { rla_transcribe_wav(wav.as_ptr(), small.as_mut_ptr(), small.len()) };
    assert_eq!(m as usize, n);
    assert_eq!(&small[..7], &buf[..7]);
    assert_eq!(small[7], 0);

    // Transcription does not touch the capture state machine.
    assert_eq!(rla_state(), 0);

    // A bad path is -1.
    let missing = CString::new("/nonexistent/rla.wav").unwrap();
    assert_eq!(
        unsafe { rla_transcribe_wav(missing.as_ptr(), buf.as_mut_ptr(), buf.len()) },
        -1
    );
}

#[test]
fn cancel_flag_aborts_whisper_inference() {
    use rullama_audio_ffi::engine::Transcriber;
    use rullama_audio_ffi::whisper::WhisperTranscriber;
    use std::sync::atomic::AtomicBool;

    let Ok(model) = std::env::var("RLA_WHISPER_MODEL") else {
        eprintln!(
            "skipping cancel_flag_aborts_whisper_inference: set RLA_WHISPER_MODEL to a ggml whisper model"
        );
        return;
    };
    let t = WhisperTranscriber::load(std::path::Path::new(&model)).unwrap();
    let pcm = rullama_audio_ffi::dsp::read_wav_pcm(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/phrase.wav"
    )))
    .unwrap();
    let mono = rullama_audio_ffi::dsp::to_mono_16k(&pcm);

    let cancelled = AtomicBool::new(true);
    assert_eq!(
        t.transcribe(&mono, &cancelled),
        Err("transcription cancelled".to_string())
    );
    // The same context still works afterwards.
    let text = t.transcribe(&mono, &AtomicBool::new(false)).unwrap();
    assert!(normalize(&text).contains("second tile"), "{text:?}");
}
