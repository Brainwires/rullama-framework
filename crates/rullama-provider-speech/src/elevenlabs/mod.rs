//! ElevenLabs API client for text-to-speech and speech-to-text.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

const ELEVENLABS_API_BASE: &str = "https://api.elevenlabs.io/v1";

/// ElevenLabs API client.
pub struct ElevenLabsClient {
    api_key: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl ElevenLabsClient {
    /// Create a new ElevenLabs client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: ELEVENLABS_API_BASE.to_string(),
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set rate limiting.
    pub fn with_rate_limit(mut self, requests_per_minute: u32) -> Self {
        self.rate_limiter = Some(std::sync::Arc::new(RateLimiter::new(requests_per_minute)));
        self
    }

    async fn acquire_rate_limit(&self) {
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }
    }

    /// Text-to-speech synthesis. Returns raw audio bytes (mp3 by default).
    pub async fn text_to_speech(
        &self,
        voice_id: &str,
        req: &ElevenLabsTtsRequest,
    ) -> Result<Vec<u8>> {
        self.acquire_rate_limit().await;

        let url = format!("{}/text-to-speech/{}", self.base_url, voice_id);

        let response = self
            .http_client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send ElevenLabs TTS request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("ElevenLabs TTS API error ({}): {}", status, body);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read ElevenLabs TTS response")?;
        Ok(bytes.to_vec())
    }

    /// Speech-to-text transcription.
    pub async fn speech_to_text(
        &self,
        audio_data: Vec<u8>,
        req: &ElevenLabsSttRequest,
    ) -> Result<ElevenLabsSttResponse> {
        self.acquire_rate_limit().await;

        let url = format!("{}/speech-to-text", self.base_url);

        let file_part = reqwest::multipart::Part::bytes(audio_data)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .context("Failed to create multipart")?;

        let mut form = reqwest::multipart::Form::new().part("audio", file_part);

        if let Some(ref model) = req.model {
            form = form.text("model_id", model.clone());
        }
        if let Some(ref lang) = req.language_code {
            form = form.text("language_code", lang.clone());
        }

        let response = self
            .http_client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .multipart(form)
            .send()
            .await
            .context("Failed to send ElevenLabs STT request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("ElevenLabs STT API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse ElevenLabs STT response")
    }

    /// List available voices.
    pub async fn list_voices(&self) -> Result<ElevenLabsVoicesResponse> {
        self.acquire_rate_limit().await;

        let url = format!("{}/voices", self.base_url);

        let response = self
            .http_client
            .get(&url)
            .header("xi-api-key", &self.api_key)
            .send()
            .await
            .context("Failed to list ElevenLabs voices")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("ElevenLabs voices API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse ElevenLabs voices response")
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// TTS request body.
#[derive(Debug, Clone, Serialize)]
pub struct ElevenLabsTtsRequest {
    /// The text to synthesize.
    pub text: String,
    /// Model ID (e.g., "eleven_multilingual_v2").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Voice settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_settings: Option<ElevenLabsVoiceSettings>,
    /// Output format (e.g., "mp3_44100_128", "pcm_16000").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

/// Voice settings for fine-tuning synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevenLabsVoiceSettings {
    /// Stability (0.0 - 1.0).
    pub stability: f32,
    /// Similarity boost (0.0 - 1.0).
    pub similarity_boost: f32,
    /// Style (0.0 - 1.0, optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f32>,
    /// Use speaker boost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
}

/// STT request parameters.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsSttRequest {
    /// Model ID for transcription.
    pub model: Option<String>,
    /// Language code hint.
    pub language_code: Option<String>,
}

/// STT response.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsSttResponse {
    /// Transcribed text.
    pub text: String,
    /// Detected language.
    #[serde(default)]
    pub language_code: Option<String>,
}

/// Voices list response.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoicesResponse {
    /// Available voices.
    pub voices: Vec<ElevenLabsVoice>,
}

/// A single voice entry.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoice {
    /// Voice ID.
    pub voice_id: String,
    /// Voice name.
    pub name: String,
    /// Available labels/tags.
    #[serde(default)]
    pub labels: std::collections::HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_request_serialization() {
        let req = ElevenLabsTtsRequest {
            text: "Hello world".to_string(),
            model_id: Some("eleven_multilingual_v2".to_string()),
            voice_settings: Some(ElevenLabsVoiceSettings {
                stability: 0.5,
                similarity_boost: 0.75,
                style: None,
                use_speaker_boost: None,
            }),
            output_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["text"], "Hello world");
        assert_eq!(json["model_id"], "eleven_multilingual_v2");
        assert!(json.get("output_format").is_none());
    }

    #[test]
    fn test_client_creation() {
        let client = ElevenLabsClient::new("test-key");
        assert_eq!(client.base_url, ELEVENLABS_API_BASE);
    }

    #[test]
    fn test_voices_response_deserialization() {
        let json = r#"{
            "voices": [
                {"voice_id": "abc123", "name": "Rachel", "labels": {"accent": "american"}}
            ]
        }"#;
        let resp: ElevenLabsVoicesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.voices.len(), 1);
        assert_eq!(resp.voices[0].name, "Rachel");
    }
}
