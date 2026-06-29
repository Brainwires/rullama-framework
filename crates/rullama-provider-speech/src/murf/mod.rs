//! Murf AI API client for text-to-speech.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

const MURF_API_BASE: &str = "https://api.murf.ai/v1";

/// Murf AI API client.
pub struct MurfClient {
    api_key: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl MurfClient {
    /// Create a new Murf client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: MURF_API_BASE.to_string(),
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

    /// Generate speech from text. Returns a URL to the generated audio.
    pub async fn generate_speech(&self, req: &MurfGenerateRequest) -> Result<MurfGenerateResponse> {
        self.acquire_rate_limit().await;

        let url = format!("{}/speech/generate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send Murf generate request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Murf API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Murf generate response")
    }

    /// Download audio from a URL returned by generate_speech.
    pub async fn download_audio(&self, audio_url: &str) -> Result<Vec<u8>> {
        let response = self
            .http_client
            .get(audio_url)
            .send()
            .await
            .context("Failed to download Murf audio")?;

        if !response.status().is_success() {
            let status = response.status();
            anyhow::bail!("Murf download error ({})", status);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read Murf audio bytes")?;
        Ok(bytes.to_vec())
    }

    /// List available voices.
    pub async fn list_voices(&self) -> Result<MurfVoicesResponse> {
        self.acquire_rate_limit().await;

        let url = format!("{}/speech/voices", self.base_url);

        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to list Murf voices")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Murf voices API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Murf voices response")
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// Generate speech request.
#[derive(Debug, Clone, Serialize)]
pub struct MurfGenerateRequest {
    /// Voice ID.
    #[serde(rename = "voiceId")]
    pub voice_id: String,
    /// Text to synthesize.
    pub text: String,
    /// Output format ("WAV", "MP3", "FLAC").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Speaking rate (0.5 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<f32>,
    /// Pitch adjustment (-50 to 50).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<i32>,
    /// Sample rate (8000, 16000, 22050, 24000, 44100, 48000).
    #[serde(rename = "sampleRate", skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
}

/// Generate speech response.
#[derive(Debug, Clone, Deserialize)]
pub struct MurfGenerateResponse {
    /// URL to download the audio file.
    #[serde(rename = "audioFile")]
    pub audio_file: Option<String>,
    /// Duration in seconds.
    #[serde(rename = "audioDuration")]
    pub audio_duration: Option<f64>,
}

/// Voices list response.
#[derive(Debug, Clone, Deserialize)]
pub struct MurfVoicesResponse {
    /// Available voices.
    #[serde(default)]
    pub voices: Vec<MurfVoice>,
}

/// A single Murf voice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MurfVoice {
    /// Voice ID.
    #[serde(rename = "voiceId")]
    pub voice_id: String,
    /// Display name.
    pub name: String,
    /// Gender ("Male", "Female").
    #[serde(default)]
    pub gender: Option<String>,
    /// Language code.
    #[serde(rename = "languageCode", default)]
    pub language_code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = MurfClient::new("test-key");
        assert_eq!(client.base_url, MURF_API_BASE);
    }

    #[test]
    fn test_generate_request_serialization() {
        let req = MurfGenerateRequest {
            voice_id: "en-US-natalie".to_string(),
            text: "Hello world".to_string(),
            format: Some("WAV".to_string()),
            rate: None,
            pitch: None,
            sample_rate: Some(24000),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["voiceId"], "en-US-natalie");
        assert_eq!(json["text"], "Hello world");
    }

    #[test]
    fn test_generate_response_deserialization() {
        let json = r#"{
            "audioFile": "https://cdn.murf.ai/audio/123.wav",
            "audioDuration": 2.5
        }"#;
        let resp: MurfGenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.audio_file,
            Some("https://cdn.murf.ai/audio/123.wav".to_string())
        );
        assert_eq!(resp.audio_duration, Some(2.5));
    }
}
