use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::rate_limiter::RateLimiter;

pub mod chat;
pub use chat::*;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Low-level Google Gemini API client.
///
/// Handles HTTP communication, rate limiting, and (de)serialisation of the
/// Gemini wire types.  Does **not** know about `brainwires_core` domain types;
/// see the chat-layer wrapper (`GoogleChatProvider`) for that.
pub struct GoogleClient {
    api_key: String,
    model: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl GoogleClient {
    /// Create a new Google client with the given API key and model.
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Create a client with rate limiting (requests per minute).
    pub fn with_rate_limit(api_key: String, model: String, requests_per_minute: u32) -> Self {
        Self {
            api_key,
            model,
            http_client: Client::new(),
            rate_limiter: Some(std::sync::Arc::new(RateLimiter::new(requests_per_minute))),
        }
    }

    /// Return the model name this client was created with.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the API key this client was created with.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Wait for rate-limit clearance (no-op if not configured).
    async fn acquire_rate_limit(&self) {
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }
    }

    // -----------------------------------------------------------------
    // Raw API methods
    // -----------------------------------------------------------------

    /// Non-streaming `generateContent` call.
    pub async fn generate_content(&self, req: &GeminiRequest) -> Result<GeminiResponse> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            GEMINI_API_BASE, self.model, self.api_key
        );

        self.acquire_rate_limit().await;
        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send request to Google Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Google Gemini API error ({}): {}", status, error_text);
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .context("Failed to parse Google Gemini response")?;

        Ok(gemini_response)
    }

    /// Non-streaming `generateContent` call using an explicit model name.
    ///
    /// Used when `ChatOptions::model` overrides the provider's default model.
    pub async fn generate_content_for_model(
        &self,
        model: &str,
        req: &GeminiRequest,
    ) -> Result<GeminiResponse> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            GEMINI_API_BASE, model, self.api_key
        );
        self.acquire_rate_limit().await;
        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .context("Failed to send request to Google Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Google Gemini API error ({}): {}", status, error_text);
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .context("Failed to parse Google Gemini response")?;
        Ok(gemini_response)
    }

    /// Streaming `streamGenerateContent` call using an explicit model name.
    pub fn stream_generate_content_for_model<'a>(
        &'a self,
        model: String,
        req: &'a GeminiRequest,
    ) -> BoxStream<'a, Result<GeminiStreamChunk>> {
        use futures::stream::StreamExt;
        let url = format!(
            "{}/models/{}:streamGenerateContent?key={}",
            GEMINI_API_BASE, model, self.api_key
        );
        Box::pin(async_stream::stream! {
            let url = url;
            self.acquire_rate_limit().await;
            let response = match self
                .http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => { yield Err(e.into()); return; }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                yield Err(anyhow::anyhow!("Google Gemini API error ({}): {}", status, error_text));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => { yield Err(e.into()); continue; }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() { continue; }
                    match serde_json::from_str::<GeminiStreamChunk>(&line) {
                        Ok(chunk) => { yield Ok(chunk); }
                        Err(e) => { tracing::warn!("Failed to parse Gemini stream chunk: {}", e); }
                    }
                }
            }
        })
    }

    /// Streaming `streamGenerateContent` call.
    ///
    /// Returns a stream of [`GeminiStreamChunk`]s parsed from the
    /// newline-delimited JSON the Gemini API produces.
    pub fn stream_generate_content<'a>(
        &'a self,
        req: &'a GeminiRequest,
    ) -> BoxStream<'a, Result<GeminiStreamChunk>> {
        use futures::stream::StreamExt;

        Box::pin(async_stream::stream! {
            let url = format!(
                "{}/models/{}:streamGenerateContent?key={}",
                GEMINI_API_BASE, self.model, self.api_key
            );

            self.acquire_rate_limit().await;
            let response = match self
                .http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                yield Err(anyhow::anyhow!(
                    "Google Gemini API error ({}): {}",
                    status,
                    error_text
                ));
                return;
            }

            // Parse streaming response (newline-delimited JSON)
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

                // Process complete JSON objects (delimited by newlines)
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    match serde_json::from_str::<GeminiStreamChunk>(&line) {
                        Ok(chunk) => {
                            yield Ok(chunk);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse Gemini stream chunk: {}", e);
                        }
                    }
                }
            }
        })
    }

    /// Paginated model listing.
    pub async fn list_models(&self) -> Result<Vec<GoogleModelEntry>> {
        let mut all_models = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut url = format!(
                "{}/models?key={}&pageSize=1000",
                GEMINI_API_BASE, self.api_key
            );
            if let Some(ref token) = page_token {
                url.push_str(&format!("&pageToken={}", token));
            }

            let resp = self
                .http_client
                .get(&url)
                .send()
                .await
                .context("Failed to list Google models")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "Google models API returned {}: {}",
                    status,
                    body
                ));
            }

            let page: GoogleListResponse = resp
                .json()
                .await
                .context("Failed to parse Google models response")?;

            all_models.extend(page.models);

            match page.next_page_token {
                Some(token) if !token.is_empty() => page_token = Some(token),
                _ => break,
            }
        }

        Ok(all_models)
    }
}

// =========================================================================
// Gemini API wire types  (all pub for reuse in the chat-layer crate)
// =========================================================================

/// A request body for the Gemini `generateContent` endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiRequest {
    /// Conversation messages.
    pub contents: Vec<GeminiMessage>,
    /// Optional system-level instruction.
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiSystemInstruction>,
    /// Optional generation configuration (temperature, etc.).
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    /// Optional tool declarations available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiToolSet>>,
}

/// A system-level instruction for the Gemini API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiSystemInstruction {
    /// The parts that make up the system instruction.
    pub parts: Vec<GeminiPart>,
}

/// Generation configuration for a Gemini request.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiGenerationConfig {
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// A set of function declarations that the model may call.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiToolSet {
    /// The function declarations in this tool set.
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

/// A single message in a Gemini conversation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiMessage {
    /// The role of the message author (e.g. `"user"`, `"model"`).
    pub role: String,
    /// The content parts of this message.
    pub parts: Vec<GeminiPart>,
}

/// A single content part within a Gemini message.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum GeminiPart {
    /// Plain text content.
    Text {
        /// The text string.
        text: String,
    },
    /// Inline binary data (e.g. an image).
    InlineData {
        /// The inline data payload.
        inline_data: GeminiInlineData,
    },
    /// A function call requested by the model.
    FunctionCall {
        /// The function call details.
        function_call: GeminiFunctionCall,
    },
    /// A response to a previous function call.
    FunctionResponse {
        /// The function response details.
        function_response: GeminiFunctionResponse,
    },
}

/// Inline binary data embedded in a Gemini message part.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiInlineData {
    /// The MIME type of the data (e.g. `"image/png"`).
    pub mime_type: String,
    /// Base64-encoded data.
    pub data: String,
}

/// A function call produced by the Gemini model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiFunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function.
    pub args: serde_json::Value,
}

/// A response to a Gemini function call.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiFunctionResponse {
    /// The name of the function that was called.
    pub name: String,
    /// The response payload from the function.
    pub response: serde_json::Value,
}

/// A function declaration exposed to the Gemini model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiFunctionDeclaration {
    /// The function name.
    pub name: String,
    /// A description of what the function does.
    pub description: String,
    /// JSON Schema describing the function parameters.
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

/// Response from the Gemini `generateContent` endpoint.
#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    /// Candidate completions returned by the model.
    pub candidates: Vec<GeminiCandidate>,
    /// Token usage metadata.
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

/// A single candidate completion from the Gemini model.
#[derive(Debug, Deserialize)]
pub struct GeminiCandidate {
    /// The generated content.
    pub content: GeminiContent,
    /// The reason the model stopped generating (e.g. `"STOP"`).
    #[serde(rename = "finishReason")]
    pub finish_reason: String,
}

/// The content of a Gemini candidate response.
#[derive(Debug, Deserialize)]
pub struct GeminiContent {
    /// The parts that make up this content.
    pub parts: Vec<GeminiPart>,
}

/// Token usage metadata from a Gemini response.
#[derive(Debug, Deserialize, Clone)]
pub struct GeminiUsageMetadata {
    /// Number of tokens in the prompt.
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    /// Number of tokens in the generated candidates.
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: u32,
    /// Total token count (prompt + candidates).
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: u32,
}

/// A single chunk from a streaming Gemini response.
#[derive(Debug, Deserialize)]
pub struct GeminiStreamChunk {
    /// Candidate completions in this chunk.
    pub candidates: Vec<GeminiCandidate>,
    /// Token usage metadata (typically present in the final chunk).
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

// =========================================================================
// Model listing types
// =========================================================================

/// A single model entry returned by the Google `models` endpoint.
#[derive(Debug, Deserialize, Clone)]
pub struct GoogleModelEntry {
    /// The model resource name, e.g. `"models/gemini-2.0-flash"`.
    pub name: String,
    /// Human-readable display name.
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    /// Maximum number of input tokens the model supports.
    #[serde(rename = "inputTokenLimit")]
    pub input_token_limit: Option<u32>,
    /// Maximum number of output tokens the model supports.
    #[serde(rename = "outputTokenLimit")]
    pub output_token_limit: Option<u32>,
    /// Generation methods supported by this model (e.g. `"generateContent"`).
    #[serde(rename = "supportedGenerationMethods", default)]
    pub supported_generation_methods: Vec<String>,
}

/// Paginated response from the Google `models` endpoint.
#[derive(Debug, Deserialize)]
pub struct GoogleListResponse {
    /// The model entries in this page.
    #[serde(default)]
    pub models: Vec<GoogleModelEntry>,
    /// Token for fetching the next page, if any.
    #[serde(rename = "nextPageToken")]
    pub next_page_token: Option<String>,
}

// =========================================================================
// GoogleModelLister  (implements the `ModelLister` trait)
// =========================================================================

use crate::model_listing::{AvailableModel, ModelCapability, ModelLister};

/// Lists models available from the Google Gemini API.
pub struct GoogleModelLister {
    api_key: String,
    #[allow(dead_code)] // reserved for direct HTTP calls in future
    http_client: Client,
}

impl GoogleModelLister {
    /// Create a new model lister with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            http_client: Client::new(),
        }
    }

    /// Infer capabilities from `supportedGenerationMethods`.
    fn infer_capabilities(methods: &[String]) -> Vec<ModelCapability> {
        let mut caps = Vec::new();

        let has_generate = methods.iter().any(|m| m == "generateContent");
        let has_embed = methods.iter().any(|m| m == "embedContent");

        if has_generate {
            caps.push(ModelCapability::Chat);
            caps.push(ModelCapability::ToolUse);
            caps.push(ModelCapability::Vision);
        }
        if has_embed {
            caps.push(ModelCapability::Embedding);
        }

        if caps.is_empty() {
            // Fallback: at least mark as Chat if we can't determine
            caps.push(ModelCapability::Chat);
        }

        caps
    }
}

#[async_trait]
impl ModelLister for GoogleModelLister {
    async fn list_models(&self) -> Result<Vec<AvailableModel>> {
        let client = GoogleClient::new(self.api_key.clone(), String::new());
        let entries = client.list_models().await?;

        let mut all_models = Vec::new();
        for entry in &entries {
            // Strip "models/" prefix from name
            let id = entry
                .name
                .strip_prefix("models/")
                .unwrap_or(&entry.name)
                .to_string();

            all_models.push(AvailableModel {
                id,
                display_name: entry.display_name.clone(),
                provider: crate::ProviderType::Google,
                capabilities: Self::infer_capabilities(&entry.supported_generation_methods),
                owned_by: Some("google".to_string()),
                context_window: entry.input_token_limit,
                max_output_tokens: entry.output_token_limit,
                created_at: None,
            });
        }

        Ok(all_models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_google_client_new() {
        let client = GoogleClient::new("test-key".to_string(), "gemini-pro".to_string());
        assert_eq!(client.api_key(), "test-key");
        assert_eq!(client.model(), "gemini-pro");
    }

    #[test]
    fn test_google_client_new_empty_api_key() {
        let client = GoogleClient::new("".to_string(), "gemini-pro".to_string());
        assert_eq!(client.api_key(), "");
        assert_eq!(client.model(), "gemini-pro");
    }

    #[test]
    fn test_google_client_new_empty_model() {
        let client = GoogleClient::new("test-key".to_string(), "".to_string());
        assert_eq!(client.api_key(), "test-key");
        assert_eq!(client.model(), "");
    }

    #[test]
    fn test_google_client_new_special_chars() {
        let client = GoogleClient::new(
            "key-with-special-!@#$%".to_string(),
            "model-1.5-pro".to_string(),
        );
        assert_eq!(client.api_key(), "key-with-special-!@#$%");
        assert_eq!(client.model(), "model-1.5-pro");
    }

    #[test]
    fn test_gemini_request_serialization() {
        let req = GeminiRequest {
            contents: vec![GeminiMessage {
                role: "user".to_string(),
                parts: vec![GeminiPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["contents"][0]["role"], "user");
    }

    #[test]
    fn test_gemini_request_with_system_instruction() {
        let req = GeminiRequest {
            contents: vec![],
            system_instruction: Some(GeminiSystemInstruction {
                parts: vec![GeminiPart::Text {
                    text: "You are helpful".to_string(),
                }],
            }),
            generation_config: None,
            tools: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("systemInstruction").is_some());
    }

    #[test]
    fn test_gemini_request_with_generation_config() {
        let req = GeminiRequest {
            contents: vec![],
            system_instruction: None,
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(0.5),
                max_output_tokens: Some(1024),
                top_p: None,
            }),
            tools: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["generationConfig"]["temperature"], 0.5);
        assert_eq!(json["generationConfig"]["maxOutputTokens"], 1024);
    }

    #[test]
    fn test_gemini_request_with_tools() {
        let mut params = std::collections::HashMap::new();
        params.insert("arg1".to_string(), json!({"type": "string"}));

        let req = GeminiRequest {
            contents: vec![],
            system_instruction: None,
            generation_config: None,
            tools: Some(vec![GeminiToolSet {
                function_declarations: vec![GeminiFunctionDeclaration {
                    name: "test_tool".to_string(),
                    description: "A test tool".to_string(),
                    parameters: params,
                }],
            }]),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["tools"][0]["function_declarations"][0]["name"],
            "test_tool"
        );
    }

    #[test]
    fn test_gemini_part_text_roundtrip() {
        let part = GeminiPart::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["text"], "Hello");
    }

    #[test]
    fn test_gemini_part_inline_data_roundtrip() {
        let part = GeminiPart::InlineData {
            inline_data: GeminiInlineData {
                mime_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["inline_data"]["mime_type"], "image/png");
        assert_eq!(json["inline_data"]["data"], "base64data");
    }

    #[test]
    fn test_gemini_part_function_call_roundtrip() {
        let part = GeminiPart::FunctionCall {
            function_call: GeminiFunctionCall {
                name: "my_func".to_string(),
                args: json!({"key": "value"}),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["function_call"]["name"], "my_func");
        assert_eq!(json["function_call"]["args"]["key"], "value");
    }

    #[test]
    fn test_gemini_part_function_response_roundtrip() {
        let part = GeminiPart::FunctionResponse {
            function_response: GeminiFunctionResponse {
                name: "my_func".to_string(),
                response: json!({"result": "ok"}),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["function_response"]["name"], "my_func");
    }

    #[test]
    fn test_gemini_generation_config_skip_none_fields() {
        let config = GeminiGenerationConfig {
            temperature: Some(0.7),
            max_output_tokens: None,
            top_p: None,
        };
        let json = serde_json::to_value(&config).unwrap();
        assert!(json.get("temperature").is_some());
        assert!(json.get("maxOutputTokens").is_none());
        assert!(json.get("topP").is_none());
    }

    #[test]
    fn test_google_model_entry_deserialization() {
        let json = r#"{
            "name": "models/gemini-2.0-flash",
            "displayName": "Gemini 2.0 Flash",
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": ["generateContent", "countTokens"]
        }"#;
        let entry: GoogleModelEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.name, "models/gemini-2.0-flash");
        assert_eq!(entry.display_name, Some("Gemini 2.0 Flash".to_string()));
        assert_eq!(entry.input_token_limit, Some(1048576));
        assert_eq!(entry.output_token_limit, Some(8192));
        assert_eq!(entry.supported_generation_methods.len(), 2);
    }

    #[test]
    fn test_google_list_response_deserialization() {
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-pro",
                    "displayName": "Gemini Pro",
                    "supportedGenerationMethods": ["generateContent"]
                }
            ],
            "nextPageToken": "abc123"
        }"#;
        let resp: GoogleListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 1);
        assert_eq!(resp.next_page_token, Some("abc123".to_string()));
    }

    #[test]
    fn test_google_list_response_no_next_page() {
        let json = r#"{
            "models": []
        }"#;
        let resp: GoogleListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 0);
        assert!(resp.next_page_token.is_none());
    }
}
