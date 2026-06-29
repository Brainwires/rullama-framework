//! Deepgram API client for text-to-speech and speech-to-text.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

const DEEPGRAM_API_BASE: &str = "https://api.deepgram.com/v1";

/// Deepgram API client.
pub struct DeepgramClient {
    api_key: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl DeepgramClient {
    /// Create a new Deepgram client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEEPGRAM_API_BASE.to_string(),
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

    /// Text-to-speech (Aura). Returns raw audio bytes.
    pub async fn speak(&self, req: &DeepgramSpeakRequest) -> Result<Vec<u8>> {
        self.acquire_rate_limit().await;

        let url = format!("{}/speak", self.base_url);

        let mut builder = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(ref model) = req.model {
            builder = builder.query(&[("model", model.as_str())]);
        }
        if let Some(ref encoding) = req.encoding {
            builder = builder.query(&[("encoding", encoding.as_str())]);
        }
        if let Some(sample_rate) = req.sample_rate {
            builder = builder.query(&[("sample_rate", &sample_rate.to_string())]);
        }

        let body = serde_json::json!({ "text": req.text });

        let response = builder
            .json(&body)
            .send()
            .await
            .context("Failed to send Deepgram speak request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Deepgram speak API error ({}): {}", status, body);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read Deepgram speak response")?;
        Ok(bytes.to_vec())
    }

    /// Speech-to-text (Listen). Transcribes audio data.
    pub async fn listen(
        &self,
        audio_data: Vec<u8>,
        req: &DeepgramListenRequest,
    ) -> Result<DeepgramListenResponse> {
        self.acquire_rate_limit().await;

        let mut url = format!("{}/listen", self.base_url);
        let mut params = Vec::new();

        if let Some(ref model) = req.model {
            params.push(format!("model={model}"));
        }
        if let Some(ref lang) = req.language {
            params.push(format!("language={lang}"));
        }
        if req.punctuate {
            params.push("punctuate=true".to_string());
        }
        if req.diarize {
            params.push("diarize=true".to_string());
        }
        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let content_type = req.content_type.as_deref().unwrap_or("audio/wav");

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", content_type)
            .body(audio_data)
            .send()
            .await
            .context("Failed to send Deepgram listen request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Deepgram listen API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Deepgram listen response")
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// Speak (TTS) request.
#[derive(Debug, Clone, Default)]
pub struct DeepgramSpeakRequest {
    /// Text to synthesize.
    pub text: String,
    /// Model name (e.g., "aura-asteria-en").
    pub model: Option<String>,
    /// Output encoding (e.g., "linear16", "mp3").
    pub encoding: Option<String>,
    /// Sample rate for output audio.
    pub sample_rate: Option<u32>,
}

/// Listen (STT) request parameters.
#[derive(Debug, Clone, Default)]
pub struct DeepgramListenRequest {
    /// Model to use (e.g., "nova-2").
    pub model: Option<String>,
    /// Language code (e.g., "en-US").
    pub language: Option<String>,
    /// Add punctuation.
    pub punctuate: bool,
    /// Enable speaker diarization.
    pub diarize: bool,
    /// Content type of the audio (e.g., "audio/wav").
    pub content_type: Option<String>,
}

/// Listen (STT) response.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepgramListenResponse {
    /// Results from transcription.
    pub results: DeepgramResults,
}

/// Transcription results container.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepgramResults {
    /// Channel results.
    pub channels: Vec<DeepgramChannel>,
}

/// A single channel's transcription.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepgramChannel {
    /// Alternative transcriptions.
    pub alternatives: Vec<DeepgramAlternative>,
}

/// A transcription alternative.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepgramAlternative {
    /// The transcribed text.
    pub transcript: String,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Word-level details.
    #[serde(default)]
    pub words: Vec<DeepgramWord>,
}

/// A single word with timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepgramWord {
    /// The word text.
    pub word: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Confidence.
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = DeepgramClient::new("test-key");
        assert_eq!(client.base_url, DEEPGRAM_API_BASE);
    }

    #[test]
    fn test_listen_response_deserialization() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "hello world",
                        "confidence": 0.99,
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
                            {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.98}
                        ]
                    }]
                }]
            }
        }"#;
        let resp: DeepgramListenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.results.channels[0].alternatives[0].transcript,
            "hello world"
        );
    }
}
