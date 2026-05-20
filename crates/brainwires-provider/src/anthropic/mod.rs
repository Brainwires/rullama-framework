/// Anthropic chat provider implementation.
pub mod chat;
/// Anthropic model listing types and implementation.
pub mod models;

#[cfg(feature = "bedrock")]
pub mod bedrock;
#[cfg(feature = "vertex-ai")]
pub mod vertex;

pub use chat::*;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::rate_limiter::RateLimiter;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// Auth strategy
// ---------------------------------------------------------------------------

/// Authentication strategy for the Anthropic Messages protocol.
///
/// All three variants speak the same wire protocol but use different
/// endpoints and auth mechanisms.
pub enum AuthStrategy {
    /// Direct Anthropic API — `x-api-key` header.
    Anthropic {
        /// The Anthropic API key.
        api_key: String,
    },
    /// Amazon Bedrock — AWS SigV4 signed requests.
    #[cfg(feature = "bedrock")]
    Bedrock {
        /// Bedrock SigV4 auth context.
        auth: bedrock::BedrockAuth,
    },
    /// Google Vertex AI — OAuth2 Bearer token.
    #[cfg(feature = "vertex-ai")]
    VertexAI {
        /// Vertex AI OAuth2 auth context.
        auth: vertex::VertexAuth,
    },
}

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

/// Low-level Anthropic (Claude) API client.
///
/// This struct handles authentication, rate-limiting, and HTTP transport.
/// It exposes raw API methods that return Anthropic-native types; higher-level
/// abstractions (e.g. the `Provider` trait) live in [`chat`].
///
/// Supports three backends via [`AuthStrategy`]:
/// - Direct Anthropic API
/// - Amazon Bedrock (feature `bedrock`)
/// - Google Vertex AI (feature `vertex-ai`)
pub struct AnthropicClient {
    auth_strategy: AuthStrategy,
    model: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl AnthropicClient {
    /// Create a new Anthropic client with the given API key and model.
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            auth_strategy: AuthStrategy::Anthropic { api_key },
            model,
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Create a client with rate limiting (requests per minute).
    pub fn with_rate_limit(api_key: String, model: String, requests_per_minute: u32) -> Self {
        Self {
            auth_strategy: AuthStrategy::Anthropic { api_key },
            model,
            http_client: Client::new(),
            rate_limiter: Some(std::sync::Arc::new(RateLimiter::new(requests_per_minute))),
        }
    }

    /// Create a Bedrock-backed client.
    #[cfg(feature = "bedrock")]
    pub fn bedrock(auth: bedrock::BedrockAuth, model: String) -> Self {
        Self {
            auth_strategy: AuthStrategy::Bedrock { auth },
            model,
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Create a Vertex AI-backed client.
    #[cfg(feature = "vertex-ai")]
    pub fn vertex(auth: vertex::VertexAuth, model: String) -> Self {
        Self {
            auth_strategy: AuthStrategy::VertexAI { auth },
            model,
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Return the model name this client was created with.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Return the API key, if using direct Anthropic auth.
    ///
    /// Returns `None` for Bedrock and Vertex AI backends.
    pub fn api_key(&self) -> Option<&str> {
        match &self.auth_strategy {
            AuthStrategy::Anthropic { api_key } => Some(api_key),
            #[cfg(feature = "bedrock")]
            AuthStrategy::Bedrock { .. } => None,
            #[cfg(feature = "vertex-ai")]
            AuthStrategy::VertexAI { .. } => None,
        }
    }

    /// Wait for rate-limit clearance (no-op if not configured).
    async fn acquire_rate_limit(&self) {
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }
    }

    // -----------------------------------------------------------------------
    // URL resolution
    // -----------------------------------------------------------------------

    /// Resolve the endpoint URL for the given streaming mode.
    #[allow(unused_variables)]
    fn resolve_url(&self, streaming: bool) -> String {
        match &self.auth_strategy {
            AuthStrategy::Anthropic { .. } => ANTHROPIC_API_URL.to_string(),
            #[cfg(feature = "bedrock")]
            AuthStrategy::Bedrock { auth } => {
                if streaming {
                    bedrock::bedrock_stream_url(auth.region(), &self.model)
                } else {
                    bedrock::bedrock_invoke_url(auth.region(), &self.model)
                }
            }
            #[cfg(feature = "vertex-ai")]
            AuthStrategy::VertexAI { auth } => {
                if streaming {
                    auth.stream_url(&self.model)
                } else {
                    auth.raw_predict_url(&self.model)
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Request body building
    // -----------------------------------------------------------------------

    /// Build the JSON request body, adapting for backend differences.
    ///
    /// - **Anthropic**: includes `model`, `anthropic-version` as header
    /// - **Bedrock**: omits `model` (in URL), `anthropic_version` set by SigV4 signer
    /// - **Vertex AI**: includes `model`, adds `anthropic_version` to body
    fn build_body(&self, req: &AnthropicRequest, streaming: bool) -> serde_json::Value {
        let mut body = json!({
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "stream": streaming,
        });

        // Model field: omitted for Bedrock (model is in URL)
        match &self.auth_strategy {
            #[cfg(feature = "bedrock")]
            AuthStrategy::Bedrock { .. } => {}
            _ => {
                body["model"] = json!(req.model);
            }
        }

        // Vertex AI: anthropic_version in body
        #[cfg(feature = "vertex-ai")]
        if matches!(&self.auth_strategy, AuthStrategy::VertexAI { .. }) {
            body["anthropic_version"] = json!(ANTHROPIC_VERSION);
        }

        // Prompt caching is only supported on the direct Anthropic API.
        // Bedrock and Vertex AI have their own (proprietary, different-shaped)
        // caching mechanisms, and some SDK versions reject unknown top-level
        // fields — emitting `cache_control` there risks a 400 regression.
        let cache_prompt =
            req.cache_prompt && matches!(&self.auth_strategy, AuthStrategy::Anthropic { .. });

        if let Some(ref sys) = req.system {
            body["system"] = if cache_prompt {
                json!([{
                    "type": "text",
                    "text": sys,
                    "cache_control": { "type": "ephemeral" }
                }])
            } else {
                json!(sys)
            };
        }
        if let Some(temp) = req.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = req.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(ref stop) = req.stop_sequences {
            body["stop_sequences"] = json!(stop);
        }
        if let Some(ref tools) = req.tools {
            body["tools"] = if cache_prompt && !tools.is_empty() {
                let mut tools_arr: Vec<serde_json::Value> = tools
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or(serde_json::Value::Null))
                    .collect();
                if let Some(last) = tools_arr.last_mut()
                    && let Some(obj) = last.as_object_mut()
                {
                    obj.insert("cache_control".to_string(), json!({ "type": "ephemeral" }));
                }
                json!(tools_arr)
            } else {
                json!(tools)
            };
        }

        body
    }

    // -----------------------------------------------------------------------
    // Auth application + send
    // -----------------------------------------------------------------------

    /// Build, authenticate, and send a request. Returns the raw response.
    async fn send_request(&self, url: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
        match &self.auth_strategy {
            AuthStrategy::Anthropic { api_key } => self
                .http_client
                .post(url)
                .header("x-api-key", api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(body)
                .send()
                .await
                .context("Failed to send request to Anthropic"),
            #[cfg(feature = "bedrock")]
            AuthStrategy::Bedrock { auth } => {
                let body_bytes =
                    serde_json::to_vec(body).context("Failed to serialize request body")?;
                let mut request = self
                    .http_client
                    .post(url)
                    .header("content-type", "application/json")
                    .body(body_bytes)
                    .build()
                    .context("Failed to build Bedrock request")?;

                auth.sign_request(&mut request)
                    .await
                    .context("Failed to sign Bedrock request with SigV4")?;

                self.http_client
                    .execute(request)
                    .await
                    .context("Failed to send request to Bedrock")
            }
            #[cfg(feature = "vertex-ai")]
            AuthStrategy::VertexAI { auth } => {
                let token = auth
                    .get_token()
                    .await
                    .context("Failed to get Vertex AI OAuth2 token")?;

                self.http_client
                    .post(url)
                    .header("Authorization", format!("Bearer {}", token))
                    .header("content-type", "application/json")
                    .json(body)
                    .send()
                    .await
                    .context("Failed to send request to Vertex AI")
            }
        }
    }

    /// Return a label for the current backend (used in error messages).
    fn backend_label(&self) -> &'static str {
        match &self.auth_strategy {
            AuthStrategy::Anthropic { .. } => "Anthropic",
            #[cfg(feature = "bedrock")]
            AuthStrategy::Bedrock { .. } => "Bedrock",
            #[cfg(feature = "vertex-ai")]
            AuthStrategy::VertexAI { .. } => "Vertex AI",
        }
    }

    // -----------------------------------------------------------------------
    // Raw API methods
    // -----------------------------------------------------------------------

    /// Send a non-streaming request and return the parsed response.
    pub async fn messages(&self, req: &AnthropicRequest) -> Result<AnthropicResponse> {
        let url = self.resolve_url(false);
        let body = self.build_body(req, false);

        self.acquire_rate_limit().await;

        let response = self.send_request(&url, &body).await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "{} API error ({}): {}",
                self.backend_label(),
                status,
                error_text
            );
        }

        let anthropic_response: AnthropicResponse =
            response.json().await.context("Failed to parse response")?;

        Ok(anthropic_response)
    }

    /// Send a streaming request and return a stream of raw SSE events.
    pub fn stream_messages<'a>(
        &'a self,
        req: &'a AnthropicRequest,
    ) -> BoxStream<'a, Result<AnthropicStreamEvent>> {
        Box::pin(async_stream::stream! {
            let url = self.resolve_url(true);
            let body = self.build_body(req, true);

            self.acquire_rate_limit().await;

            let response = match self.send_request(&url, &body).await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                yield Err(anyhow::anyhow!("{} API error ({}): {}", self.backend_label(), status, error_text));
                return;
            }

            // Parse SSE stream — identical format for all three backends
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
                            continue;
                        }

                        match serde_json::from_str::<AnthropicStreamEvent>(data) {
                            Ok(event) => {
                                yield Ok(event);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to parse stream event: {}", e);
                            }
                        }
                    }
                }
            }
        })
    }

    /// Paginated listing of models available via `/v1/models`.
    ///
    /// Only supported for direct Anthropic API. Returns an empty list for
    /// Bedrock and Vertex AI backends.
    pub async fn list_models(&self) -> Result<Vec<AnthropicModelEntry>> {
        // Model listing is only available on direct Anthropic API
        let api_key = match self.api_key() {
            Some(key) => key,
            None => return Ok(Vec::new()),
        };

        let mut all_models = Vec::new();
        let mut after_id: Option<String> = None;

        loop {
            let mut url = format!("{}?limit=1000", ANTHROPIC_MODELS_URL);
            if let Some(ref cursor) = after_id {
                url.push_str(&format!("&after_id={}", cursor));
            }

            let resp = self
                .http_client
                .get(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .send()
                .await
                .context("Failed to list Anthropic models")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "Anthropic models API returned {}: {}",
                    status,
                    body
                ));
            }

            let page: AnthropicListResponse = resp
                .json()
                .await
                .context("Failed to parse Anthropic models response")?;

            for entry in page.data {
                all_models.push(entry);
            }

            if !page.has_more {
                break;
            }
            after_id = page.last_id;
        }

        Ok(all_models)
    }
}

// ---------------------------------------------------------------------------
// Request type
// ---------------------------------------------------------------------------

/// A request to the Anthropic `/v1/messages` endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    /// Model identifier (e.g. `"claude-sonnet-4-20250514"`).
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<AnthropicMessage>,
    /// Optional system prompt (sent as a top-level field, not a message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
    /// Emit `cache_control: ephemeral` breakpoints on the system prompt and
    /// the final tool definition. Anthropic silently no-ops when the cached
    /// prefix is below the model's minimum cacheable size, so it's always
    /// safe to leave on; disable for debugging or deterministic replay.
    #[serde(skip)]
    pub cache_prompt: bool,
}

// ---------------------------------------------------------------------------
// Anthropic API serde types
// ---------------------------------------------------------------------------

/// A single message in an Anthropic conversation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessage {
    /// The role of the message author (e.g. `"user"`, `"assistant"`).
    pub role: String,
    /// The content blocks of this message.
    pub content: Vec<AnthropicContentBlock>,
}

/// A typed content block within an Anthropic message.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    /// Plain text content.
    Text {
        /// The text string.
        text: String,
    },
    /// Base64-encoded image for vision-capable models.
    Image {
        /// Anthropic-native source envelope (`{type, media_type, data}`).
        source: AnthropicImageSource,
    },
    /// A tool use request from the model.
    ToolUse {
        /// Unique identifier for this tool use.
        id: String,
        /// The name of the tool to invoke.
        name: String,
        /// The JSON input to the tool.
        input: serde_json::Value,
    },
    /// The result of a tool invocation.
    ToolResult {
        /// The ID of the tool use this result corresponds to.
        tool_use_id: String,
        /// The textual result content.
        content: String,
    },
}

/// Image source envelope used inside `AnthropicContentBlock::Image`.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    /// Inline base64-encoded image data.
    Base64 {
        /// MIME type, e.g. `"image/png"` or `"image/jpeg"`.
        media_type: String,
        /// Base64-encoded image bytes.
        data: String,
    },
}

/// A tool definition for the Anthropic function-calling API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicTool {
    /// The tool name.
    pub name: String,
    /// A description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool input parameters.
    pub input_schema: std::collections::HashMap<String, serde_json::Value>,
}

/// Response from the Anthropic `/v1/messages` endpoint.
#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicResponse {
    /// The content blocks generated by the model.
    pub content: Vec<AnthropicContentBlock>,
    /// The reason the model stopped generating (e.g. `"end_turn"`, `"tool_use"`).
    pub stop_reason: String,
    /// Token usage statistics.
    pub usage: AnthropicUsage,
}

/// Token usage statistics from an Anthropic response.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct AnthropicUsage {
    /// Number of input tokens consumed.
    pub input_tokens: u32,
    /// Number of output tokens generated.
    pub output_tokens: u32,
    /// Tokens written to the prompt cache on this request.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    /// Tokens read from the prompt cache (charged at 10% of input rate).
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

/// A single server-sent event from a streaming Anthropic response.
#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicStreamEvent {
    /// The event type (e.g. `"content_block_delta"`, `"message_delta"`).
    #[serde(rename = "type")]
    pub event_type: String,
    /// The delta payload, if present.
    pub delta: Option<AnthropicDelta>,
    /// Token usage statistics, if present.
    pub usage: Option<AnthropicUsage>,
    /// Summary text from a `context_window_management_event`.
    ///
    /// Present only when `event_type == "context_window_management_event"`.
    /// Contains the model-generated summary that replaces the compacted history.
    #[serde(default)]
    pub summary: Option<String>,
    /// Approximate tokens freed by context compaction.
    ///
    /// Present only when `event_type == "context_window_management_event"`.
    #[serde(default)]
    pub tokens_freed: Option<u32>,
}

/// An incremental delta within a streaming Anthropic event.
#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicDelta {
    /// Incremental text content.
    pub text: Option<String>,
}

// ---------------------------------------------------------------------------
// Model listing
// ---------------------------------------------------------------------------

use crate::model_listing::{
    AnthropicListResponse, AnthropicModelEntry, AvailableModel, ModelCapability, ModelLister,
};

const ANTHROPIC_MODELS_URL: &str = "https://api.anthropic.com/v1/models";

/// Lists models available from the Anthropic API.
pub struct AnthropicModelLister {
    api_key: String,
    http_client: Client,
}

impl AnthropicModelLister {
    /// Create a new model lister with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            http_client: Client::new(),
        }
    }
}

#[async_trait]
impl ModelLister for AnthropicModelLister {
    async fn list_models(&self) -> Result<Vec<AvailableModel>> {
        let mut all_models = Vec::new();
        let mut after_id: Option<String> = None;

        loop {
            let mut url = format!("{}?limit=1000", ANTHROPIC_MODELS_URL);
            if let Some(ref cursor) = after_id {
                url.push_str(&format!("&after_id={}", cursor));
            }

            let resp = self
                .http_client
                .get(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .send()
                .await
                .context("Failed to list Anthropic models")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "Anthropic models API returned {}: {}",
                    status,
                    body
                ));
            }

            let page: AnthropicListResponse = resp
                .json()
                .await
                .context("Failed to parse Anthropic models response")?;

            for entry in &page.data {
                all_models.push(AvailableModel {
                    id: entry.id.clone(),
                    display_name: Some(entry.display_name.clone()),
                    provider: crate::ProviderType::Anthropic,
                    capabilities: vec![
                        ModelCapability::Chat,
                        ModelCapability::ToolUse,
                        ModelCapability::Vision,
                    ],
                    owned_by: Some("anthropic".to_string()),
                    context_window: None,
                    max_output_tokens: None,
                    created_at: None,
                });
            }

            if !page.has_more {
                break;
            }
            after_id = page.last_id;
        }

        Ok(all_models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_client_new() {
        let client = AnthropicClient::new("test-key".to_string(), "claude-sonnet-4-6".to_string());
        assert_eq!(client.api_key(), Some("test-key"));
        assert_eq!(client.model, "claude-sonnet-4-6");
    }

    #[test]
    fn test_client_model_accessor() {
        let client = AnthropicClient::new("test-key".to_string(), "claude-sonnet-4-6".to_string());
        assert_eq!(client.model(), "claude-sonnet-4-6");
    }

    #[test]
    fn test_client_api_key_accessor() {
        let client = AnthropicClient::new("test-key".to_string(), "claude-sonnet-4-6".to_string());
        assert_eq!(client.api_key(), Some("test-key"));
    }

    #[test]
    fn test_anthropic_client_with_empty_api_key() {
        let client = AnthropicClient::new("".to_string(), "claude-sonnet-4-6".to_string());
        assert_eq!(client.api_key(), Some(""));
        assert_eq!(client.model, "claude-sonnet-4-6");
    }

    #[test]
    fn test_anthropic_client_with_special_characters_in_api_key() {
        let api_key = "sk-ant-api03-!@#$%^&*()_+-=[]{}|;':\",./<>?".to_string();
        let client = AnthropicClient::new(api_key.clone(), "claude-opus-4-6".to_string());
        assert_eq!(client.api_key(), Some(api_key.as_str()));
    }

    #[test]
    fn test_anthropic_client_with_various_model_names() {
        let models = vec![
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "custom-model-123",
            // Claude 4.6 generation
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ];

        for model in models {
            let client = AnthropicClient::new("test-key".to_string(), model.to_string());
            assert_eq!(client.model, model);
        }
    }

    #[test]
    fn test_anthropic_constants() {
        assert_eq!(ANTHROPIC_API_URL, "https://api.anthropic.com/v1/messages");
        assert_eq!(ANTHROPIC_VERSION, "2023-06-01");
    }

    #[test]
    fn test_resolve_url_anthropic() {
        let client = AnthropicClient::new("key".to_string(), "claude-sonnet-4-6".to_string());
        assert_eq!(client.resolve_url(false), ANTHROPIC_API_URL);
        assert_eq!(client.resolve_url(true), ANTHROPIC_API_URL);
    }

    #[test]
    fn test_backend_label() {
        let client = AnthropicClient::new("key".to_string(), "claude-sonnet-4-6".to_string());
        assert_eq!(client.backend_label(), "Anthropic");
    }

    #[cfg(feature = "bedrock")]
    #[test]
    fn test_bedrock_client() {
        let auth = bedrock::BedrockAuth::new(
            "us-west-2".to_string(),
            "AKID".to_string(),
            "secret".to_string(),
            None,
        );
        let client = AnthropicClient::bedrock(auth, "anthropic.claude-sonnet-4-6-v1:0".to_string());
        assert_eq!(client.api_key(), None);
        assert_eq!(client.backend_label(), "Bedrock");
        assert!(client.resolve_url(false).contains("us-west-2"));
        assert!(client.resolve_url(false).contains("/invoke"));
        assert!(
            client
                .resolve_url(true)
                .contains("invoke-with-response-stream")
        );
    }

    #[cfg(feature = "bedrock")]
    #[test]
    fn test_bedrock_from_environment() {
        // Don't rely on actual env vars in test — just test with explicit credentials
        let auth = bedrock::BedrockAuth::new(
            "eu-west-1".to_string(),
            "test-access-key".to_string(),
            "test-secret-key".to_string(),
            Some("test-session-token".to_string()),
        );
        assert_eq!(auth.region(), "eu-west-1");
    }

    #[cfg(feature = "vertex-ai")]
    #[test]
    fn test_vertex_client() {
        let auth = vertex::VertexAuth::new("my-project".to_string(), "us-central1".to_string());
        let client = AnthropicClient::vertex(auth, "claude-sonnet-4-6".to_string());
        assert_eq!(client.api_key(), None);
        assert_eq!(client.backend_label(), "Vertex AI");
        assert!(client.resolve_url(false).contains("rawPredict"));
        assert!(!client.resolve_url(false).contains("stream"));
        assert!(client.resolve_url(true).contains("streamRawPredict"));
    }

    #[test]
    fn test_anthropic_request_serialization() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: vec![AnthropicContentBlock::Text {
                    text: "Hello".to_string(),
                }],
            }],
            system: Some("You are helpful".to_string()),
            max_tokens: 4096,
            temperature: Some(0.7),
            top_p: None,
            stop_sequences: None,
            tools: None,
            stream: false,
            cache_prompt: false,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-sonnet-4-6");
        assert_eq!(json["max_tokens"], 4096);
        let temp = json["temperature"].as_f64().unwrap();
        assert!(
            (temp - 0.7).abs() < 1e-6,
            "temperature {temp} not close to 0.7"
        );
        assert!(json.get("top_p").is_none());
        assert!(json.get("stop_sequences").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn test_anthropic_content_block_text_serde() {
        let block = AnthropicContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
    }

    #[test]
    fn test_anthropic_content_block_tool_use_serde() {
        let block = AnthropicContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "search".to_string(),
            input: serde_json::json!({"query": "test"}),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["id"], "tool-1");
        assert_eq!(json["name"], "search");
        assert_eq!(json["input"]["query"], "test");
    }

    #[test]
    fn test_anthropic_content_block_tool_result_serde() {
        let block = AnthropicContentBlock::ToolResult {
            tool_use_id: "tool-1".to_string(),
            content: "result text".to_string(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "tool-1");
        assert_eq!(json["content"], "result text");
    }

    #[test]
    fn test_anthropic_message_serialization() {
        let msg = AnthropicMessage {
            role: "user".to_string(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Look at this".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({}),
                },
            ],
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_anthropic_response_deserialization() {
        let json = r#"{
            "content": [
                {"type": "text", "text": "Hello!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.stop_reason, "end_turn");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.content.len(), 1);
    }

    #[test]
    fn test_anthropic_stream_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "delta": {"text": "Hi"},
            "usage": null
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(event.delta.unwrap().text.unwrap(), "Hi");
    }

    #[test]
    fn test_anthropic_stream_event_message_delta() {
        let json = r#"{
            "type": "message_delta",
            "delta": null,
            "usage": {"input_tokens": 0, "output_tokens": 42}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "message_delta");
        assert_eq!(event.usage.unwrap().output_tokens, 42);
    }

    #[test]
    fn test_anthropic_tool_serialization() {
        let mut schema = std::collections::HashMap::new();
        schema.insert(
            "query".to_string(),
            serde_json::json!({"type": "string", "description": "Search query"}),
        );
        let tool = AnthropicTool {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            input_schema: schema,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "search");
        assert!(json["input_schema"]["query"].is_object());
    }

    #[test]
    fn test_build_body_anthropic() {
        let client = AnthropicClient::new("key".to_string(), "claude-sonnet-4-6".to_string());
        let req = AnthropicRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![],
            system: None,
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            stream: false,
            cache_prompt: false,
        };
        let body = client.build_body(&req, false);
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_build_body_system_cache_control_when_enabled() {
        let client = AnthropicClient::new("key".to_string(), "claude-sonnet-4-6".to_string());
        let req = AnthropicRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![],
            system: Some("You are helpful".to_string()),
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            stream: false,
            cache_prompt: true,
        };
        let body = client.build_body(&req, false);
        // When caching is on, system must be an array with a cache_control marker
        let system = &body["system"];
        assert!(system.is_array(), "system should be array when caching");
        let first = &system[0];
        assert_eq!(first["type"], "text");
        assert_eq!(first["text"], "You are helpful");
        assert_eq!(first["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_build_body_system_plain_when_cache_disabled() {
        let client = AnthropicClient::new("key".to_string(), "claude-sonnet-4-6".to_string());
        let req = AnthropicRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![],
            system: Some("Plain system".to_string()),
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            stream: false,
            cache_prompt: false,
        };
        let body = client.build_body(&req, false);
        // When caching is off, system stays a plain string
        assert_eq!(body["system"], "Plain system");
    }

    #[cfg(feature = "bedrock")]
    #[test]
    fn test_build_body_bedrock_never_emits_cache_control_even_when_cache_prompt_true() {
        let auth = bedrock::BedrockAuth::new(
            "us-west-2".to_string(),
            "AKID".to_string(),
            "secret".to_string(),
            None,
        );
        let client = AnthropicClient::bedrock(auth, "anthropic.claude-sonnet-4-6-v1:0".to_string());
        let req = AnthropicRequest {
            model: "anthropic.claude-sonnet-4-6-v1:0".to_string(),
            messages: vec![],
            system: Some("You are helpful".to_string()),
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            stream: false,
            // Even with cache_prompt=true, the build must suppress cache_control
            // for Bedrock to avoid 400 regressions on its proprietary caching.
            cache_prompt: true,
        };
        let body = client.build_body(&req, false);
        // system must serialise as a plain string, not an array-with-cache-control.
        assert_eq!(body["system"], "You are helpful");
    }

    #[test]
    fn test_build_body_tools_cache_control_on_last() {
        let client = AnthropicClient::new("key".to_string(), "claude-sonnet-4-6".to_string());
        let tools = vec![
            AnthropicTool {
                name: "first".to_string(),
                description: "First tool".to_string(),
                input_schema: std::collections::HashMap::new(),
            },
            AnthropicTool {
                name: "second".to_string(),
                description: "Second tool".to_string(),
                input_schema: std::collections::HashMap::new(),
            },
        ];
        let req = AnthropicRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![],
            system: None,
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: Some(tools),
            stream: false,
            cache_prompt: true,
        };
        let body = client.build_body(&req, false);
        let tools_arr = body["tools"].as_array().expect("tools must be array");
        assert_eq!(tools_arr.len(), 2);
        // Only the last tool gets cache_control
        assert!(tools_arr[0].get("cache_control").is_none());
        assert_eq!(tools_arr[1]["cache_control"]["type"], "ephemeral");
    }
}
