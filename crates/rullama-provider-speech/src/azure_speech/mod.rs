//! Azure Cognitive Services Speech API client.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

/// Azure Speech API client.
pub struct AzureSpeechClient {
    subscription_key: String,
    region: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl AzureSpeechClient {
    /// Create a new Azure Speech client.
    pub fn new(subscription_key: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            subscription_key: subscription_key.into(),
            region: region.into(),
            http_client: Client::new(),
            rate_limiter: None,
        }
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

    fn tts_endpoint(&self) -> String {
        format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/v1",
            self.region
        )
    }

    fn stt_endpoint(&self) -> String {
        format!(
            "https://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
            self.region
        )
    }

    fn voices_endpoint(&self) -> String {
        format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/voices/list",
            self.region
        )
    }

    /// Synthesize speech from SSML. Returns raw audio bytes.
    pub async fn synthesize(&self, ssml: &str, output_format: &str) -> Result<Vec<u8>> {
        self.acquire_rate_limit().await;

        let response = self
            .http_client
            .post(self.tts_endpoint())
            .header("Ocp-Apim-Subscription-Key", &self.subscription_key)
            .header("Content-Type", "application/ssml+xml")
            .header("X-Microsoft-OutputFormat", output_format)
            .body(ssml.to_string())
            .send()
            .await
            .context("Failed to send Azure TTS request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Azure TTS API error ({}): {}", status, body);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read Azure TTS response")?;
        Ok(bytes.to_vec())
    }

    /// Synthesize from plain text by wrapping in SSML.
    pub async fn synthesize_text(
        &self,
        text: &str,
        voice_name: &str,
        output_format: &str,
    ) -> Result<Vec<u8>> {
        let ssml = format!(
            r#"<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="{voice_name}">{text}</voice>
</speak>"#,
            voice_name = voice_name,
            text = text,
        );
        self.synthesize(&ssml, output_format).await
    }

    /// Recognize speech from audio data.
    pub async fn recognize(
        &self,
        audio_data: Vec<u8>,
        req: &AzureSttRequest,
    ) -> Result<AzureSttResponse> {
        self.acquire_rate_limit().await;

        let mut url = self.stt_endpoint();
        let lang = req.language.as_deref().unwrap_or("en-US");
        url = format!("{}?language={}", url, lang);

        let content_type = req
            .content_type
            .as_deref()
            .unwrap_or("audio/wav; codecs=audio/pcm; samplerate=16000");

        let response = self
            .http_client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", &self.subscription_key)
            .header("Content-Type", content_type)
            .body(audio_data)
            .send()
            .await
            .context("Failed to send Azure STT request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Azure STT API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Azure STT response")
    }

    /// List available voices.
    pub async fn list_voices(&self) -> Result<Vec<AzureVoice>> {
        self.acquire_rate_limit().await;

        let response = self
            .http_client
            .get(self.voices_endpoint())
            .header("Ocp-Apim-Subscription-Key", &self.subscription_key)
            .send()
            .await
            .context("Failed to list Azure voices")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Azure voices API error ({}): {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse Azure voices response")
    }
}

// ── Request/Response types ──────────────────────────────────────────────

/// STT request parameters.
#[derive(Debug, Clone, Default)]
pub struct AzureSttRequest {
    /// Language (e.g., "en-US").
    pub language: Option<String>,
    /// Content type header value.
    pub content_type: Option<String>,
}

/// STT response.
#[derive(Debug, Clone, Deserialize)]
pub struct AzureSttResponse {
    /// Recognition status.
    #[serde(rename = "RecognitionStatus")]
    pub recognition_status: String,
    /// Recognized text.
    #[serde(rename = "DisplayText")]
    pub display_text: Option<String>,
    /// Offset in ticks.
    #[serde(rename = "Offset")]
    pub offset: Option<u64>,
    /// Duration in ticks.
    #[serde(rename = "Duration")]
    pub duration: Option<u64>,
}

/// An Azure voice entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureVoice {
    /// Voice name.
    #[serde(rename = "Name")]
    pub name: String,
    /// Display name.
    #[serde(rename = "DisplayName")]
    pub display_name: String,
    /// Short name (e.g., "en-US-JennyNeural").
    #[serde(rename = "ShortName")]
    pub short_name: String,
    /// Gender.
    #[serde(rename = "Gender")]
    pub gender: String,
    /// Locale.
    #[serde(rename = "Locale")]
    pub locale: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = AzureSpeechClient::new("test-key", "eastus");
        assert!(client.tts_endpoint().contains("eastus"));
    }

    #[test]
    fn test_stt_response_deserialization() {
        let json = r#"{
            "RecognitionStatus": "Success",
            "DisplayText": "Hello world.",
            "Offset": 0,
            "Duration": 10000000
        }"#;
        let resp: AzureSttResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.recognition_status, "Success");
        assert_eq!(resp.display_text, Some("Hello world.".to_string()));
    }
}
