//! Google Cloud Text-to-Speech API client.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

const GOOGLE_TTS_API_BASE: &str = "https://texttospeech.googleapis.com/v1";

/// Google Cloud TTS API client.
pub struct GoogleTtsClient {
    api_key: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl GoogleTtsClient {
    /// Create a new Google TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: GOOGLE_TTS_API_BASE.to_string(),
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

    /// Synthesize speech from text. Returns base64-encoded audio content.
    pub async fn synthesize(
        &self,
        req: &GoogleTtsSynthesizeRequest,
    ) -> Result<GoogleTtsSynthesizeResponse> {
        self.acquire_rate_limit().await;

        let url = format!("{}/text:synthesize", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .header("X-Goog-Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send Google TTS request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Google TTS API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Google TTS response")
    }

    /// List available voices.
    pub async fn list_voices(
        &self,
        language_code: Option<&str>,
    ) -> Result<GoogleTtsVoicesResponse> {
        self.acquire_rate_limit().await;

        let mut url = format!("{}/voices", self.base_url);
        if let Some(lang) = language_code {
            url = format!("{}?languageCode={}", url, lang);
        }

        let response = self
            .http_client
            .get(&url)
            .header("X-Goog-Api-Key", &self.api_key)
            .send()
            .await
            .context("Failed to list Google TTS voices")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Google TTS voices API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Google TTS voices response")
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// Synthesize request.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleTtsSynthesizeRequest {
    /// The text input.
    pub input: GoogleTtsInput,
    /// Voice selection.
    pub voice: GoogleTtsVoiceSelection,
    /// Audio config.
    #[serde(rename = "audioConfig")]
    pub audio_config: GoogleTtsAudioConfig,
}

/// Text input for synthesis.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleTtsInput {
    /// Plain text to synthesize.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// SSML input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssml: Option<String>,
}

/// Voice selection parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleTtsVoiceSelection {
    /// Language code (e.g., "en-US").
    #[serde(rename = "languageCode")]
    pub language_code: String,
    /// Voice name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Gender ("MALE", "FEMALE", "NEUTRAL").
    #[serde(rename = "ssmlGender", skip_serializing_if = "Option::is_none")]
    pub ssml_gender: Option<String>,
}

/// Audio configuration.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleTtsAudioConfig {
    /// Audio encoding ("LINEAR16", "MP3", "OGG_OPUS", "MULAW", "ALAW").
    #[serde(rename = "audioEncoding")]
    pub audio_encoding: String,
    /// Speaking rate (0.25 - 4.0).
    #[serde(rename = "speakingRate", skip_serializing_if = "Option::is_none")]
    pub speaking_rate: Option<f32>,
    /// Pitch (-20.0 - 20.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f32>,
    /// Sample rate in Hz.
    #[serde(rename = "sampleRateHertz", skip_serializing_if = "Option::is_none")]
    pub sample_rate_hertz: Option<u32>,
}

/// Synthesize response.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleTtsSynthesizeResponse {
    /// Base64-encoded audio content.
    #[serde(rename = "audioContent")]
    pub audio_content: String,
}

/// Voices list response.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleTtsVoicesResponse {
    /// Available voices.
    #[serde(default)]
    pub voices: Vec<GoogleTtsVoiceEntry>,
}

/// A single voice entry.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleTtsVoiceEntry {
    /// Language codes this voice supports.
    #[serde(rename = "languageCodes", default)]
    pub language_codes: Vec<String>,
    /// Voice name.
    pub name: String,
    /// Gender.
    #[serde(rename = "ssmlGender")]
    pub ssml_gender: Option<String>,
    /// Sample rate in Hz.
    #[serde(rename = "naturalSampleRateHertz")]
    pub natural_sample_rate_hertz: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = GoogleTtsClient::new("test-key");
        assert_eq!(client.base_url, GOOGLE_TTS_API_BASE);
    }

    #[test]
    fn test_synthesize_request_serialization() {
        let req = GoogleTtsSynthesizeRequest {
            input: GoogleTtsInput {
                text: Some("Hello world".to_string()),
                ssml: None,
            },
            voice: GoogleTtsVoiceSelection {
                language_code: "en-US".to_string(),
                name: Some("en-US-Neural2-A".to_string()),
                ssml_gender: None,
            },
            audio_config: GoogleTtsAudioConfig {
                audio_encoding: "LINEAR16".to_string(),
                speaking_rate: None,
                pitch: None,
                sample_rate_hertz: Some(24000),
            },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["input"]["text"], "Hello world");
        assert_eq!(json["voice"]["languageCode"], "en-US");
    }
}
