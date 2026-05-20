//! OpenAI (and OpenAI-compatible) API client, wire types, and submodules.

/// OpenAI audio (TTS / STT) types.
pub mod audio;
/// OpenAI chat provider implementation.
pub mod chat;
/// OpenAI model listing types and implementation.
pub mod models;

pub use audio::*;
pub use chat::*;
pub use models::*;

use anyhow::{Context, Result};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::rate_limiter::RateLimiter;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

// ---------------------------------------------------------------------------
// Request options
// ---------------------------------------------------------------------------

/// Options for chat completion requests sent to the OpenAI API.
#[derive(Debug, Clone, Default)]
pub struct OpenAiRequestOptions {
    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Up to 4 sequences where the API will stop generating further tokens.
    pub stop: Option<Vec<String>>,
    /// System message prepended to the conversation.
    pub system: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAI API client
// ---------------------------------------------------------------------------

/// Low-level HTTP client for the OpenAI (and OpenAI-compatible) REST API.
///
/// This struct is provider-type agnostic: it speaks only in OpenAI wire
/// types (`OpenAIMessage`, `OpenAITool`, etc.) and leaves higher-level
/// concerns such as converting from `brainwires_core::Message` to callers.
pub struct OpenAiClient {
    api_key: String,
    #[allow(dead_code)] // stored as default; callers pass model per-request
    model: String,
    base_url: String,
    http_client: Client,
    organization_id: Option<String>,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl OpenAiClient {
    /// Create a new OpenAI client with the given API key and default model.
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            base_url: OPENAI_API_URL.to_string(),
            http_client: Client::new(),
            organization_id: None,
            rate_limiter: None,
        }
    }

    /// Create a client with rate limiting (requests per minute).
    pub fn with_rate_limit(api_key: String, model: String, requests_per_minute: u32) -> Self {
        Self {
            api_key,
            model,
            base_url: OPENAI_API_URL.to_string(),
            http_client: Client::new(),
            organization_id: None,
            rate_limiter: Some(std::sync::Arc::new(RateLimiter::new(requests_per_minute))),
        }
    }

    /// Set a custom base URL (for OpenAI-compatible APIs like Groq).
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Set the organization ID for API requests.
    pub fn with_organization(mut self, org_id: String) -> Self {
        self.organization_id = Some(org_id);
        self
    }

    /// Wait for rate-limit clearance (no-op if not configured).
    async fn acquire_rate_limit(&self) {
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }
    }

    /// Check if the default model is an O1/O3 model (no streaming, no system messages).
    pub fn is_o1_model(model: &str) -> bool {
        model.starts_with("o1-") || model.starts_with("o3-")
    }

    // -------------------------------------------------------------------
    // Raw API methods
    // -------------------------------------------------------------------

    /// Send a non-streaming chat completion request and return the raw
    /// provider response.
    #[tracing::instrument(name = "openai_client.chat_completions", skip_all, fields(model = %model))]
    pub async fn chat_completions(
        &self,
        messages: &[OpenAIMessage],
        model: &str,
        tools: Option<&[OpenAITool]>,
        options: &OpenAiRequestOptions,
    ) -> Result<OpenAIResponse> {
        let mut request_body = json!({
            "model": model,
            "messages": messages,
        });

        if !Self::is_o1_model(model) {
            if let Some(max_tokens) = options.max_tokens {
                request_body["max_tokens"] = json!(max_tokens);
            }
            if let Some(temp) = options.temperature {
                request_body["temperature"] = json!(temp);
            }
            if let Some(top_p) = options.top_p {
                request_body["top_p"] = json!(top_p);
            }
            if let Some(ref stop) = options.stop {
                request_body["stop"] = json!(stop);
            }
        }

        if let Some(tools_list) = tools
            && !tools_list.is_empty()
        {
            request_body["tools"] = json!(tools_list);
        }

        let mut request = self
            .http_client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(org_id) = &self.organization_id {
            request = request.header("OpenAI-Organization", org_id);
        }

        self.acquire_rate_limit().await;
        let response = request
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }

        let openai_response: OpenAIResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        Ok(openai_response)
    }

    /// Open a streaming chat completion and return a stream of raw
    /// provider-specific chunks.
    pub fn stream_chat_completions<'a>(
        &'a self,
        messages: &'a [OpenAIMessage],
        model: &'a str,
        tools: Option<&'a [OpenAITool]>,
        options: &'a OpenAiRequestOptions,
    ) -> BoxStream<'a, Result<OpenAIStreamChunk>> {
        Box::pin(async_stream::stream! {
            let mut request_body = json!({
                "model": model,
                "messages": messages,
                "stream": true,
            });

            if !Self::is_o1_model(model) {
                if let Some(max_tokens) = options.max_tokens {
                    request_body["max_tokens"] = json!(max_tokens);
                }
                if let Some(temp) = options.temperature {
                    request_body["temperature"] = json!(temp);
                }
                if let Some(top_p) = options.top_p {
                    request_body["top_p"] = json!(top_p);
                }
                if let Some(ref stop) = options.stop {
                    request_body["stop"] = json!(stop);
                }
            }

            if let Some(tools_list) = tools
                && !tools_list.is_empty()
            {
                request_body["tools"] = json!(tools_list);
            }

            let mut request = self
                .http_client
                .post(&self.base_url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json");

            if let Some(org_id) = &self.organization_id {
                request = request.header("OpenAI-Organization", org_id);
            }

            self.acquire_rate_limit().await;
            let response = match request.json(&request_body).send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                yield Err(anyhow::anyhow!("OpenAI API error ({}): {}", status, error_text));
                return;
            }

            // Parse SSE stream
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(e.into());
                        continue;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete events (delimited by \n\n)
                while let Some(pos) = buffer.find("\n\n") {
                    let event_data = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    // Parse SSE event
                    if let Some(data) = event_data.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            // Signal end-of-stream via None in the
                            // stream (the stream simply ends).
                            return;
                        }

                        match serde_json::from_str::<OpenAIStreamChunk>(data) {
                            Ok(parsed) => {
                                yield Ok(parsed);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to parse OpenAI stream chunk: {}", e);
                            }
                        }
                    }
                }
            }
        })
    }

    /// Synthesise speech from text via the `/v1/audio/speech` endpoint.
    ///
    /// Returns the raw audio bytes in the requested format.
    pub async fn create_speech(&self, req: &CreateSpeechRequest) -> Result<Vec<u8>> {
        let speech_url = if self.base_url.ends_with("/chat/completions") {
            self.base_url.replace("/chat/completions", "/audio/speech")
        } else {
            format!("{}/audio/speech", self.base_url.trim_end_matches('/'))
        };

        let mut request = self
            .http_client
            .post(&speech_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(org_id) = &self.organization_id {
            request = request.header("OpenAI-Organization", org_id);
        }

        self.acquire_rate_limit().await;
        let response = request
            .json(req)
            .send()
            .await
            .context("Failed to send speech request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI speech API error ({}): {}", status, error_text);
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read speech response body")?;

        Ok(bytes.to_vec())
    }

    /// Transcribe audio via the `/v1/audio/transcriptions` endpoint.
    ///
    /// `audio_wav` should contain the raw bytes of the audio file (WAV,
    /// MP3, etc.).
    pub async fn create_transcription(
        &self,
        audio_wav: Vec<u8>,
        req: &TranscriptionRequest,
    ) -> Result<TranscriptionResponse> {
        let transcription_url = if self.base_url.ends_with("/chat/completions") {
            self.base_url
                .replace("/chat/completions", "/audio/transcriptions")
        } else {
            format!(
                "{}/audio/transcriptions",
                self.base_url.trim_end_matches('/')
            )
        };

        let file_part = reqwest::multipart::Part::bytes(audio_wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let mut form = reqwest::multipart::Form::new()
            .text("model", req.model.clone())
            .part("file", file_part);

        if let Some(ref lang) = req.language {
            form = form.text("language", lang.clone());
        }
        if let Some(ref prompt) = req.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(true) = req.timestamps {
            form = form.text("response_format", "verbose_json");
            form = form.text("timestamp_granularities[]", "segment");
        }

        let mut request = self
            .http_client
            .post(&transcription_url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        if let Some(org_id) = &self.organization_id {
            request = request.header("OpenAI-Organization", org_id);
        }

        self.acquire_rate_limit().await;
        let response = request
            .multipart(form)
            .send()
            .await
            .context("Failed to send transcription request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "OpenAI transcription API error ({}): {}",
                status,
                error_text
            );
        }

        let transcription: TranscriptionResponse = response
            .json()
            .await
            .context("Failed to parse transcription response")?;

        Ok(transcription)
    }

    /// List available models via the `/v1/models` endpoint.
    pub async fn list_models(&self) -> Result<OpenAIListModelsResponse> {
        let models_url = if self.base_url.ends_with("/chat/completions") {
            self.base_url.replace("/chat/completions", "/models")
        } else {
            format!("{}/models", self.base_url.trim_end_matches('/'))
        };

        let mut request = self
            .http_client
            .get(&models_url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        if let Some(org_id) = &self.organization_id {
            request = request.header("OpenAI-Organization", org_id);
        }

        self.acquire_rate_limit().await;
        let response = request
            .send()
            .await
            .context("Failed to list OpenAI models")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI models API returned {}: {}", status, body);
        }

        let list: OpenAIListModelsResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI models response")?;

        Ok(list)
    }
}

// ---------------------------------------------------------------------------
// OpenAI API serde types (all pub)
// ---------------------------------------------------------------------------

/// A single message in an OpenAI chat completion request or response.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIMessage {
    /// The role of the message author (e.g. `"user"`, `"assistant"`, `"system"`, `"tool"`).
    pub role: String,
    /// The content of the message.
    pub content: OpenAIContent,
    /// Optional name for the participant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    /// The ID of the tool call this message is responding to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message content in an OpenAI request/response.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OpenAIContent {
    /// A simple text string.
    Text(String),
    /// An array of content parts (text, images, etc.).
    Array(Vec<OpenAIContentPart>),
}

/// A typed content part within an OpenAI message.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    /// Text content.
    Text {
        /// The text string.
        text: String,
    },
    /// An image referenced by URL.
    ImageUrl {
        /// The image URL details.
        image_url: OpenAIImageUrl,
    },
}

/// An image URL reference for multimodal content.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIImageUrl {
    /// The URL of the image (can be a data URI or HTTP URL).
    pub url: String,
}

/// A tool definition for the OpenAI function-calling API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAITool {
    /// The tool type (currently always `"function"`).
    pub r#type: String,
    /// The function definition.
    pub function: OpenAIFunction,
}

/// A function definition exposed to the OpenAI model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunction {
    /// The function name.
    pub name: String,
    /// A description of what the function does.
    pub description: String,
    /// JSON Schema describing the function parameters.
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

/// A tool call produced by the OpenAI model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIToolCall {
    /// Unique identifier for this tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The type of tool call (currently always `"function"`).
    pub r#type: String,
    /// The function call details.
    pub function: OpenAIFunctionCall,
}

/// Details of a function call within an OpenAI tool call.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunctionCall {
    /// The name of the function to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The JSON-encoded arguments string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Response from the OpenAI chat completions endpoint.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIResponse {
    /// Completion choices returned by the model.
    pub choices: Vec<OpenAIChoice>,
    /// Token usage statistics.
    pub usage: OpenAIUsage,
}

/// A single completion choice in an OpenAI response.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIChoice {
    /// The message generated by the model.
    pub message: OpenAIResponseMessage,
    /// The reason the model stopped generating (e.g. `"stop"`, `"tool_calls"`).
    pub finish_reason: String,
}

/// The assistant message within an OpenAI response choice.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIResponseMessage {
    /// The content of the message.
    pub content: OpenAIContent,
    /// Tool calls generated by the model, if any.
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

/// Token usage statistics from an OpenAI response.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total number of tokens used.
    pub total_tokens: u32,
}

/// A single chunk from a streaming OpenAI chat completion.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIStreamChunk {
    /// Streaming choices in this chunk.
    pub choices: Vec<OpenAIStreamChoice>,
    /// Token usage statistics (typically present in the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

/// A single streaming choice within an OpenAI stream chunk.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIStreamChoice {
    /// The delta (incremental update) for this choice.
    pub delta: Option<OpenAIStreamDelta>,
}

/// An incremental delta within a streaming OpenAI response.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIStreamDelta {
    /// Incremental text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Incremental tool call data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_openai_client_new() {
        let client = OpenAiClient::new("test-key".to_string(), "gpt-4".to_string());
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gpt-4");
        assert!(client.organization_id.is_none());
    }

    #[test]
    fn test_openai_client_with_organization() {
        let client = OpenAiClient::new("test-key".to_string(), "gpt-4".to_string())
            .with_organization("org-123".to_string());
        assert!(client.organization_id.is_some());
        assert_eq!(client.organization_id.unwrap(), "org-123");
    }

    #[test]
    fn test_is_o1_model_true() {
        assert!(OpenAiClient::is_o1_model("o1-preview"));
    }

    #[test]
    fn test_is_o1_model_false() {
        assert!(!OpenAiClient::is_o1_model("gpt-4"));
    }

    #[test]
    fn test_is_o3_model_true() {
        assert!(OpenAiClient::is_o1_model("o3-preview"));
    }

    #[test]
    fn test_is_o1_mini_model_true() {
        assert!(OpenAiClient::is_o1_model("o1-mini"));
    }

    #[test]
    fn test_openai_client_new_with_different_models() {
        let models = vec!["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"];
        for model in models {
            let client = OpenAiClient::new("test-key".to_string(), model.to_string());
            assert_eq!(client.model, model);
            assert_eq!(client.api_key, "test-key");
        }
    }

    #[test]
    fn test_organization_id_chaining() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string())
            .with_organization("org-abc".to_string());

        assert_eq!(client.organization_id, Some("org-abc".to_string()));
        assert_eq!(client.api_key, "key");
        assert_eq!(client.model, "gpt-4");
    }

    #[test]
    fn test_empty_api_key() {
        let client = OpenAiClient::new("".to_string(), "gpt-4".to_string());
        assert_eq!(client.api_key, "");
    }

    #[test]
    fn test_empty_model() {
        let client = OpenAiClient::new("key".to_string(), "".to_string());
        assert_eq!(client.model, "");
    }

    #[test]
    fn test_openai_content_text_serialization() {
        let content = OpenAIContent::Text("Hello".to_string());
        let serialized = serde_json::to_string(&content).unwrap();
        assert_eq!(serialized, "\"Hello\"");
    }

    #[test]
    fn test_openai_content_array_serialization() {
        let content = OpenAIContent::Array(vec![OpenAIContentPart::Text {
            text: "Test".to_string(),
        }]);
        let serialized = serde_json::to_string(&content).unwrap();
        assert!(serialized.contains("Test"));
    }

    #[test]
    fn test_openai_message_serialization_without_optional_fields() {
        let msg = OpenAIMessage {
            role: "user".to_string(),
            content: OpenAIContent::Text("Hello".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let serialized = serde_json::to_value(&msg).unwrap();
        assert!(serialized.get("name").is_none());
        assert!(serialized.get("tool_calls").is_none());
        assert!(serialized.get("tool_call_id").is_none());
    }

    #[test]
    fn test_openai_message_serialization_with_optional_fields() {
        let msg = OpenAIMessage {
            role: "user".to_string(),
            content: OpenAIContent::Text("Hello".to_string()),
            name: Some("user_1".to_string()),
            tool_calls: None,
            tool_call_id: Some("tc-123".to_string()),
        };

        let serialized = serde_json::to_value(&msg).unwrap();
        assert_eq!(serialized["name"], "user_1");
        assert_eq!(serialized["tool_call_id"], "tc-123");
    }

    #[test]
    fn test_openai_tool_serialization() {
        let tool = OpenAITool {
            r#type: "function".to_string(),
            function: OpenAIFunction {
                name: "test_fn".to_string(),
                description: "Test function".to_string(),
                parameters: HashMap::new(),
            },
        };

        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized["type"], "function");
        assert_eq!(serialized["function"]["name"], "test_fn");
    }

    #[test]
    fn test_openai_response_deserialization() {
        let json = r#"{
            "choices": [{
                "message": {
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: OpenAIResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn test_openai_stream_chunk_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "content": "Hello"
                }
            }]
        }"#;

        let chunk: OpenAIStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert!(chunk.choices[0].delta.is_some());
    }

    #[test]
    fn test_openai_stream_chunk_with_usage() {
        let json = r#"{
            "choices": [],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }"#;

        let chunk: OpenAIStreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 20);
        assert_eq!(usage.completion_tokens, 10);
    }

    #[test]
    fn test_openai_content_part_image_deserialization() {
        let json = r#"{
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,abc123"
            }
        }"#;

        let part: OpenAIContentPart = serde_json::from_str(json).unwrap();
        match part {
            OpenAIContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/png;base64,abc123");
            }
            _ => panic!("Expected image url part"),
        }
    }

    #[test]
    fn test_openai_tool_call_deserialization() {
        let json = r#"{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"London\"}"
            }
        }"#;

        let tool_call: OpenAIToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.id, Some("call_123".to_string()));
        assert_eq!(tool_call.r#type, "function");
        assert_eq!(tool_call.function.name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_is_o1_model_with_various_names() {
        let o1_models = vec!["o1-preview", "o1-mini", "o1-turbo", "o3-preview", "o3-mini"];
        let non_o1_models = vec![
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4-turbo",
            "o1",
            "o3",
        ];

        for model in o1_models {
            assert!(
                OpenAiClient::is_o1_model(model),
                "Expected {} to be detected as o1 model",
                model
            );
        }

        for model in non_o1_models {
            assert!(
                !OpenAiClient::is_o1_model(model),
                "Expected {} to not be detected as o1 model",
                model
            );
        }
    }

    #[test]
    fn test_openai_list_models_response_deserialization() {
        let json = r#"{
            "data": [
                {"id": "gpt-4o", "owned_by": "openai", "created": 1700000000},
                {"id": "gpt-3.5-turbo", "owned_by": "openai", "created": 1690000000}
            ]
        }"#;

        let resp: OpenAIListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4o");
        assert_eq!(resp.data[1].id, "gpt-3.5-turbo");
    }

    #[test]
    fn test_request_options_default() {
        let opts = OpenAiRequestOptions::default();
        assert!(opts.temperature.is_none());
        assert!(opts.max_tokens.is_none());
        assert!(opts.top_p.is_none());
        assert!(opts.stop.is_none());
        assert!(opts.system.is_none());
    }
}
