//! Cartesia API client for text-to-speech.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

const CARTESIA_API_BASE: &str = "https://api.cartesia.ai";
const CARTESIA_VERSION: &str = "2024-06-10";

/// Cartesia API client.
pub struct CartesiaClient {
    api_key: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl CartesiaClient {
    /// Create a new Cartesia client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: CARTESIA_API_BASE.to_string(),
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

    /// Text-to-speech synthesis. Returns raw audio bytes.
    pub async fn tts_bytes(&self, req: &CartesiaTtsRequest) -> Result<Vec<u8>> {
        self.acquire_rate_limit().await;

        let url = format!("{}/tts/bytes", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Cartesia-Version", CARTESIA_VERSION)
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send Cartesia TTS request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Cartesia TTS API error ({}): {}", status, body);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read Cartesia TTS response")?;
        Ok(bytes.to_vec())
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// TTS request.
#[derive(Debug, Clone, Serialize)]
pub struct CartesiaTtsRequest {
    /// Model ID.
    pub model_id: String,
    /// Transcript to synthesize.
    pub transcript: String,
    /// Voice configuration.
    pub voice: CartesiaVoice,
    /// Output format.
    pub output_format: CartesiaOutputFormat,
    /// Language (e.g., "en").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Voice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartesiaVoice {
    /// Voice mode ("id" for pre-built voices).
    pub mode: String,
    /// Voice ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Output format configuration.
#[derive(Debug, Clone, Serialize)]
pub struct CartesiaOutputFormat {
    /// Container ("raw", "wav").
    pub container: String,
    /// Encoding ("pcm_f32le", "pcm_s16le", "pcm_mulaw").
    pub encoding: String,
    /// Sample rate.
    pub sample_rate: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = CartesiaClient::new("test-key");
        assert_eq!(client.base_url, CARTESIA_API_BASE);
    }

    #[test]
    fn test_tts_request_serialization() {
        let req = CartesiaTtsRequest {
            model_id: "sonic-english".to_string(),
            transcript: "Hello world".to_string(),
            voice: CartesiaVoice {
                mode: "id".to_string(),
                id: Some("a0e99841-438c-4a64-b679-ae501e7d6091".to_string()),
            },
            output_format: CartesiaOutputFormat {
                container: "raw".to_string(),
                encoding: "pcm_s16le".to_string(),
                sample_rate: 24000,
            },
            language: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model_id"], "sonic-english");
        assert_eq!(json["transcript"], "Hello world");
    }
}
