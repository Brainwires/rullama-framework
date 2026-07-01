//! Audio-related request/response types for the OpenAI API.

use serde::{Deserialize, Serialize};

/// Request body for the text-to-speech endpoint (`/v1/audio/speech`).
#[derive(Debug, Clone, Serialize)]
pub struct CreateSpeechRequest {
    /// TTS model id (e.g. `"tts-1"`, `"tts-1-hd"`).
    pub model: String,
    /// The text to synthesise.
    pub input: String,
    /// Voice to use (e.g. `"alloy"`, `"echo"`, `"fable"`, `"onyx"`, `"nova"`, `"shimmer"`).
    pub voice: String,
    /// Audio format -- `"mp3"`, `"opus"`, `"aac"`, `"flac"`, `"wav"`, `"pcm"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Playback speed (0.25 -- 4.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

/// Request parameters for the transcription endpoint (`/v1/audio/transcriptions`).
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// Whisper model id (e.g. `"whisper-1"`).
    pub model: String,
    /// BCP-47 language code of the input audio (e.g. `"en"`).
    pub language: Option<String>,
    /// An optional prompt to guide the model's style.
    pub prompt: Option<String>,
    /// Whether to include word-level timestamps.
    pub timestamps: Option<bool>,
}

/// Response from the transcription endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
    /// Detected or supplied language.
    #[serde(default)]
    pub language: Option<String>,
    /// Duration of the audio in seconds.
    #[serde(default)]
    pub duration: Option<f64>,
    /// Word- or segment-level timestamps (when requested).
    #[serde(default)]
    pub segments: Option<Vec<TranscriptionSegment>>,
}

/// A single segment returned by the transcription endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment index.
    pub id: Option<u32>,
    /// Start time in seconds.
    pub start: Option<f64>,
    /// End time in seconds.
    pub end: Option<f64>,
    /// Transcribed text for this segment.
    pub text: Option<String>,
}
