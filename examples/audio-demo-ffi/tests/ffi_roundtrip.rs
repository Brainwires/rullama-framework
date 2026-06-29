//! Handle-map invariant tests for the `audio-demo-ffi` bridge.
//!
//! These tests stay entirely on the Rust side of the FFI. No network calls
//! are made: every real provider constructor needs a valid API key, so the
//! "create + drop" positive path is covered via a synthetic `openai` handle
//! (the OpenAI TTS/STT constructors only validate on request, not on
//! construction). If that ever changes, the positive test should be replaced
//! with a `#[cfg(test)]` fake provider.

use audio_demo_ffi::{
    FfiAudioBuffer, FfiAudioError, FfiOutputFormat, FfiSampleFormat, FfiSttOptions, FfiTtsOptions,
    create_provider, drop_provider, tts_list_voices, tts_synthesize,
};

fn dummy_tts_options() -> FfiTtsOptions {
    FfiTtsOptions {
        voice_id: "alloy".into(),
        speed: None,
        output_format: FfiOutputFormat::Wav,
    }
}

fn dummy_audio_buffer() -> FfiAudioBuffer {
    FfiAudioBuffer {
        data: vec![],
        sample_rate: 16_000,
        channels: 1,
        sample_format: FfiSampleFormat::I16,
    }
}

#[test]
fn invalid_handle_returns_typed_error() {
    // A handle that was never issued must produce a typed `InvalidHandle`
    // error, NOT a panic. Exercise both TTS and STT paths through the
    // bridge.
    let err = tts_synthesize(0xdead_beef, "hello".into(), dummy_tts_options())
        .expect_err("synthesize with bogus handle should fail");
    assert!(
        matches!(err, FfiAudioError::InvalidHandle { .. }),
        "expected InvalidHandle, got {err:?}"
    );

    let err = tts_list_voices(0xdead_beef).expect_err("list_voices with bogus handle should fail");
    assert!(
        matches!(err, FfiAudioError::InvalidHandle { .. }),
        "expected InvalidHandle, got {err:?}"
    );

    let err = audio_demo_ffi::stt_transcribe(
        0xdead_beef,
        dummy_audio_buffer(),
        FfiSttOptions {
            language: None,
            timestamps: false,
            prompt: None,
        },
    )
    .expect_err("stt_transcribe with bogus handle should fail");
    assert!(
        matches!(err, FfiAudioError::InvalidHandle { .. }),
        "expected InvalidHandle, got {err:?}"
    );
}

#[test]
fn drop_invalid_handle_is_safe() {
    // `drop_provider` is documented to be a no-op for unknown handles and
    // MUST NOT panic. We run it twice to make sure repeated drops are
    // likewise harmless.
    drop_provider(0xdead_beef);
    drop_provider(0xdead_beef);
}

#[test]
fn create_then_drop_empties_map() {
    // OpenAI TTS / STT constructors only record the API key; they do not
    // make a network request until `synthesize` / `transcribe` is called.
    // That makes them safe to use here without real credentials.
    let handle = create_provider("openai".into(), "sk-test-not-real".into(), None)
        .expect("openai provider should construct offline");
    assert_ne!(handle, 0, "handle must be non-zero");

    drop_provider(handle);

    // After drop, the handle is no longer registered.
    let err = tts_synthesize(handle, "hi".into(), dummy_tts_options())
        .expect_err("synthesize after drop must fail");
    assert!(
        matches!(err, FfiAudioError::InvalidHandle { .. }),
        "expected InvalidHandle after drop, got {err:?}"
    );
}

#[test]
fn unknown_provider_returns_typed_error() {
    let err = create_provider("not-a-real-provider".into(), "key".into(), None)
        .expect_err("bogus provider name must error");
    assert!(
        matches!(err, FfiAudioError::UnknownProvider { .. }),
        "expected UnknownProvider, got {err:?}"
    );
}
