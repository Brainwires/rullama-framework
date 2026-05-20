#![deny(missing_docs)]
//! Speech (TTS / STT) provider clients for the Brainwires Agent Framework.
//!
//! Standalone so consumers (typically `brainwires-hardware`'s audio surface
//! and the chat-pwa wasm bridge) can pull just the speech clients without
//! dragging in the LLM provider stack (candle / llama.cpp / huggingface,
//! aws-sigv4, gcp_auth, …) that lives in `brainwires-provider`.
//!
//! ## Native cloud providers (`native` feature)
//! - [`azure_speech`] — Microsoft Azure Cognitive Services Speech.
//! - [`cartesia`] — Cartesia TTS.
//! - [`deepgram`] — Deepgram TTS / STT.
//! - [`elevenlabs`] — ElevenLabs TTS / STT.
//! - [`fish`] — Fish Audio TTS / ASR.
//! - [`google_tts`] — Google Cloud Text-to-Speech.
//! - [`murf`] — Murf AI TTS.
//!
//! ## Browser-native (`web-speech` feature, `wasm32` only)
//! - `web_speech` — `speechSynthesis` (TTS) and `SpeechRecognition` (STT).

/// Token-bucket rate limiter shared by every native provider.
///
/// Duplicated from `brainwires-provider::rate_limiter` rather than
/// imported across crates — both copies are 146 lines of standalone
/// stdlib-only code, and avoiding the cross-crate edge keeps this crate
/// independent of the LLM-providers stack.
#[cfg(feature = "native")]
pub mod rate_limiter;

#[cfg(feature = "native")]
pub mod azure_speech;
#[cfg(feature = "native")]
pub mod cartesia;
#[cfg(feature = "native")]
pub mod deepgram;
#[cfg(feature = "native")]
pub mod elevenlabs;
#[cfg(feature = "native")]
pub mod fish;
#[cfg(feature = "native")]
pub mod google_tts;
#[cfg(feature = "native")]
pub mod murf;

/// Browser-native TTS (`speechSynthesis`) and STT (`SpeechRecognition`).
///
/// Compiled only on `wasm32` with the `web-speech` feature enabled.
#[cfg(all(target_arch = "wasm32", feature = "web-speech"))]
pub mod web_speech;

// ── Re-exports ─────────────────────────────────────────────────────────

#[cfg(feature = "native")]
pub use azure_speech::AzureSpeechClient;
#[cfg(feature = "native")]
pub use cartesia::CartesiaClient;
#[cfg(feature = "native")]
pub use deepgram::DeepgramClient;
#[cfg(feature = "native")]
pub use elevenlabs::ElevenLabsClient;
#[cfg(feature = "native")]
pub use fish::FishClient;
#[cfg(feature = "native")]
pub use google_tts::GoogleTtsClient;
#[cfg(feature = "native")]
pub use murf::MurfClient;
