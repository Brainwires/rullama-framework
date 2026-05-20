#![deny(missing_docs)]
//! # brainwires-hardware — audio module
//!
//! Audio capture, playback, speech-to-text, and text-to-speech for the
//! Brainwires Agent Framework.

/// Ring buffer for streaming audio data.
pub mod buffer;
/// Audio capture trait and implementations.
pub mod capture;
/// Audio device enumeration and selection.
pub mod device;
/// Error types for audio operations.
pub mod error;
/// Audio playback trait and implementations.
pub mod playback;
/// Speech-to-text trait and implementations.
pub mod stt;
/// Text-to-speech trait and implementations.
pub mod tts;
/// Core audio types, configs, and data structures.
pub mod types;
/// WAV encoding and decoding utilities.
pub mod wav;

/// Native hardware audio backends using cpal.
#[cfg(feature = "audio")]
pub mod hardware;

/// Cloud API integrations (STT/TTS providers).
#[cfg(feature = "audio")]
pub mod api;

/// FLAC encoding utilities.
#[cfg(feature = "flac")]
pub mod flac;

/// Local inference backends (whisper.cpp via whisper-rs).
#[cfg(feature = "local-stt")]
pub mod local;

/// Voice Activity Detection — EnergyVad (always available) + WebRtcVad (`vad` feature).
pub mod vad;

/// Wake word detection — `EnergyTriggerDetector` (`wake-word`), `RustpotterDetector` (`wake-word-rustpotter`).
#[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
pub mod wake_word;

/// Voice Assistant pipeline — orchestrates capture → wake word → VAD → STT → handler → TTS.
#[cfg(feature = "voice-assistant")]
pub mod assistant;

// Re-exports
pub use buffer::AudioRingBuffer;
pub use capture::AudioCapture;
pub use device::{AudioDevice, DeviceDirection};
pub use error::{AudioError, AudioResult};
pub use playback::AudioPlayback;
pub use stt::SpeechToText;
pub use tts::TextToSpeech;
pub use types::{
    AudioBuffer, AudioConfig, OutputFormat, SampleFormat, SttOptions, Transcript,
    TranscriptSegment, TtsOptions, Voice,
};
pub use wav::{decode_wav, encode_wav};

#[cfg(feature = "audio")]
pub use api::{
    AzureStt, AzureTts, CartesiaTts, DeepgramStt, DeepgramTts, ElevenLabsStt, ElevenLabsTts,
    FishStt, FishTts, GoogleTts, MurfTts, OpenAiResponsesStt, OpenAiResponsesTts, OpenAiStt,
    OpenAiTts,
};
#[cfg(feature = "flac")]
pub use flac::{decode_flac, encode_flac};
#[cfg(feature = "audio")]
pub use hardware::{CpalCapture, CpalPlayback};
#[cfg(feature = "local-stt")]
pub use local::WhisperStt;

// ── VAD re-exports ────────────────────────────────────────────────────────────
pub use vad::{EnergyVad, SpeechSegment, VoiceActivityDetector};
#[cfg(feature = "vad")]
pub use vad::{VadMode, WebRtcVad};

// ── Wake word re-exports ──────────────────────────────────────────────────────
#[cfg(feature = "wake-word-rustpotter")]
pub use wake_word::RustpotterDetector;
#[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
pub use wake_word::{EnergyTriggerDetector, WakeWordDetection, WakeWordDetector};

// ── Voice assistant re-exports ────────────────────────────────────────────────
#[cfg(feature = "voice-assistant")]
pub use assistant::{
    AssistantState, VoiceAssistant, VoiceAssistantBuilder, VoiceAssistantConfig,
    VoiceAssistantHandler,
};
