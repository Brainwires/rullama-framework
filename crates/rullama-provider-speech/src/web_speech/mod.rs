//! Browser-native speech providers (TTS via `window.speechSynthesis`, STT via
//! `SpeechRecognition` / `webkitSpeechRecognition`).
//!
//! # Platform support
//!
//! This module is **wasm32-only**. The entire module is gated behind
//! `#[cfg(all(target_arch = "wasm32", feature = "web-speech"))]`, so on a
//! native target enabling `--features web-speech` is a no-op (the module is
//! empty rather than a compile error). All public types here wrap
//! [`web_sys`] objects that only exist in a browser-like JS environment.
//!
//! # Trait convention
//!
//! The cloud speech providers in this crate (`deepgram`, `google_tts`,
//! `elevenlabs`, `cartesia`, `azure_speech`, `fish`, `murf`) currently expose
//! their own concrete struct APIs and do not share a unified TTS / STT trait.
//! Rather than refactor those existing modules, this module defines small
//! local traits ([`TtsSink`], [`SttSource`]) that capture the shape of a
//! browser-native fire-and-forget TTS sink and an event-driven STT source.
//!
//! # Notes on `web-sys` features
//!
//! All Speech\* bindings used here are stable in `web-sys` 0.3.95 — they do
//! NOT require `RUSTFLAGS=--cfg=web_sys_unstable_apis`. If a future bump of
//! `web-sys` moves any of them behind `web_sys_unstable_apis`, callers will
//! need to set that rustflag in their build configuration.

#![cfg(all(target_arch = "wasm32", feature = "web-speech"))]

pub mod stt;
pub mod tts;

pub use stt::{WebSpeechStt, WebSpeechSttError, WebSpeechSttOptions, WebSpeechSttResult};
pub use tts::{VoiceInfo, WebSpeechTts, WebSpeechTtsOptions};

/// Fire-and-forget TTS sink — speech is rendered by the host environment
/// (browser / OS speech engine) and there is no audio output stream returned
/// to the caller.
pub trait TtsSink {
    /// Options accepted by [`TtsSink::speak`].
    type Options;
    /// Error type produced by sink operations.
    type Error;

    /// Queue `text` for speech synthesis using the given options.
    fn speak(&self, text: &str, opts: Self::Options) -> Result<(), Self::Error>;
    /// Cancel any pending and currently-spoken utterances.
    fn cancel(&self);
    /// Pause the current utterance (resumable via [`TtsSink::resume`]).
    fn pause(&self);
    /// Resume a previously-paused utterance.
    fn resume(&self);
}

/// Event-driven STT source — recognition results arrive asynchronously via a
/// caller-supplied callback closure rather than a polled stream.
pub trait SttSource {
    /// Options accepted by [`SttSource::start`].
    type Options;
    /// Single recognition result delivered to the result callback.
    type Result;
    /// Error type produced by source operations.
    type Error;

    /// Start recognition with the given options.
    fn start(&self, opts: Self::Options) -> Result<(), Self::Error>;
    /// Stop recognition gracefully (flushes any pending final result).
    fn stop(&self);
    /// Abort recognition immediately, discarding any pending result.
    fn abort(&self);
}
