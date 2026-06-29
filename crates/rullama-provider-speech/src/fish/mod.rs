//! Fish Audio API client for text-to-speech and speech recognition.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

const FISH_API_BASE: &str = "https://api.fish.audio/v1";

/// Fish Audio API client.
pub struct FishClient {
    api_key: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl FishClient {
    /// Create a new Fish Audio client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: FISH_API_BASE.to_string(),
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

    /// Text-to-speech. Returns raw audio bytes.
    pub async fn tts(&self, req: &FishTtsRequest) -> Result<Vec<u8>> {
        self.acquire_rate_limit().await;

        let url = format!("{}/tts", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send Fish TTS request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Fish TTS API error ({}): {}", status, body);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read Fish TTS response")?;
        Ok(bytes.to_vec())
    }

    /// Automatic speech recognition. Returns transcription.
    pub async fn asr(&self, audio_data: Vec<u8>, req: &FishAsrRequest) -> Result<FishAsrResponse> {
        self.acquire_rate_limit().await;

        let url = format!("{}/asr", self.base_url);

        let file_part = reqwest::multipart::Part::bytes(audio_data)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .context("Failed to create multipart")?;

        let mut form = reqwest::multipart::Form::new().part("audio", file_part);

        if let Some(ref lang) = req.language {
            form = form.text("language", lang.clone());
        }

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .context("Failed to send Fish ASR request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Fish ASR API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Fish ASR response")
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// TTS request.
#[derive(Debug, Clone, Serialize)]
pub struct FishTtsRequest {
    /// Text to synthesize.
    pub text: String,
    /// Reference audio ID / voice ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_id: Option<String>,
    /// Output format (e.g., "wav", "mp3").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Speaking speed (0.5 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

/// ASR request parameters.
#[derive(Debug, Clone, Default)]
pub struct FishAsrRequest {
    /// Language hint.
    pub language: Option<String>,
}

/// ASR response.
#[derive(Debug, Clone, Deserialize)]
pub struct FishAsrResponse {
    /// Transcribed text.
    pub text: String,
    /// Duration in seconds.
    #[serde(default)]
    pub duration: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = FishClient::new("test-key");
        assert_eq!(client.base_url, FISH_API_BASE);
    }

    #[test]
    fn test_tts_request_serialization() {
        let req = FishTtsRequest {
            text: "Hello".to_string(),
            reference_id: Some("voice-123".to_string()),
            format: Some("wav".to_string()),
            speed: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["text"], "Hello");
        assert_eq!(json["reference_id"], "voice-123");
    }
}
