//! Model listing and validation for AI providers.
//!
//! Each provider implements [`ModelLister`] to query available models from its API.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::ProviderType;

/// Capabilities a model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    /// Text chat / completions.
    Chat,
    /// Tool / function calling.
    ToolUse,
    /// Image / vision understanding.
    Vision,
    /// Text embedding generation.
    Embedding,
    /// Audio processing.
    Audio,
    /// Image generation.
    ImageGeneration,
}

impl std::fmt::Display for ModelCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "chat"),
            Self::ToolUse => write!(f, "tool_use"),
            Self::Vision => write!(f, "vision"),
            Self::Embedding => write!(f, "embedding"),
            Self::Audio => write!(f, "audio"),
            Self::ImageGeneration => write!(f, "image_generation"),
        }
    }
}

/// A model available from a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModel {
    /// Model identifier (e.g. "claude-sonnet-4-20250514", "gpt-4o").
    pub id: String,
    /// Human-readable name, if provided by the API.
    pub display_name: Option<String>,
    /// Which provider owns this model.
    pub provider: ProviderType,
    /// What the model can do.
    pub capabilities: Vec<ModelCapability>,
    /// Organization/owner string from the API.
    pub owned_by: Option<String>,
    /// Maximum input context window (tokens).
    pub context_window: Option<u32>,
    /// Maximum output tokens the model can produce.
    pub max_output_tokens: Option<u32>,
    /// Unix timestamp (seconds) when the model was created.
    pub created_at: Option<i64>,
}

impl AvailableModel {
    /// Whether this model supports chat completions.
    pub fn is_chat_capable(&self) -> bool {
        self.capabilities.contains(&ModelCapability::Chat)
    }
}

/// Trait for querying a provider's model catalogue.
#[async_trait]
pub trait ModelLister: Send + Sync {
    /// Fetch all models available for this provider.
    async fn list_models(&self) -> Result<Vec<AvailableModel>>;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Infer capabilities for an OpenAI-format model ID.
///
/// Shared by the OpenAI and Groq listers.
pub fn infer_openai_capabilities(model_id: &str) -> Vec<ModelCapability> {
    let id = model_id.to_lowercase();

    // Embedding models
    if id.contains("embedding") || id.starts_with("text-embedding") {
        return vec![ModelCapability::Embedding];
    }

    // Audio models
    if id.starts_with("whisper") || id.starts_with("tts") {
        return vec![ModelCapability::Audio];
    }

    // Image generation
    if id.starts_with("dall-e") {
        return vec![ModelCapability::ImageGeneration];
    }

    // Chat-capable models get Chat + ToolUse by default
    let mut caps = vec![ModelCapability::Chat, ModelCapability::ToolUse];

    // Vision-capable models
    if id.contains("vision")
        || id.contains("gpt-4o")
        || id.contains("gpt-4-turbo")
        || id.contains("gpt-5")
        || (id.starts_with("o") && !id.starts_with("omni"))
    {
        caps.push(ModelCapability::Vision);
    }

    caps
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a [`ModelLister`] for the given provider.
///
/// * `api_key` — required for cloud providers, ignored for Ollama.
/// * `base_url` — optional override (used for Ollama or custom endpoints).
pub fn create_model_lister(
    provider_type: ProviderType,
    api_key: Option<&str>,
    base_url: Option<&str>,
) -> Result<Box<dyn ModelLister>> {
    match provider_type {
        ProviderType::Anthropic => {
            let key = api_key
                .ok_or_else(|| anyhow::anyhow!("Anthropic requires an API key"))?
                .to_string();
            Ok(Box::new(super::anthropic::AnthropicModelLister::new(key)))
        }
        ProviderType::OpenAI => {
            let key = api_key
                .ok_or_else(|| anyhow::anyhow!("OpenAI requires an API key"))?
                .to_string();
            Ok(Box::new(super::openai_chat::OpenAIModelLister::new(
                key,
                base_url.map(|s| s.to_string()),
            )))
        }
        ProviderType::Google => {
            let key = api_key
                .ok_or_else(|| anyhow::anyhow!("Google requires an API key"))?
                .to_string();
            Ok(Box::new(super::gemini::GoogleModelLister::new(key)))
        }
        ProviderType::Groq
        | ProviderType::Together
        | ProviderType::Fireworks
        | ProviderType::Anyscale => {
            // All OpenAI-compatible: reuse OpenAI model lister with the registry's models URL
            let key = api_key
                .ok_or_else(|| anyhow::anyhow!("{} requires an API key", provider_type))?
                .to_string();
            let registry_url = super::registry::lookup(provider_type).and_then(|e| e.models_url);
            let url = base_url
                .or(registry_url)
                .unwrap_or("https://api.openai.com/v1/models");
            Ok(Box::new(super::openai_chat::OpenAIModelLister::new(
                key,
                Some(url.to_string()),
            )))
        }
        ProviderType::Ollama => Ok(Box::new(super::ollama::OllamaModelLister::new(
            base_url.map(|s| s.to_string()),
        ))),
        ProviderType::OpenAiResponses => {
            // Shares the same models endpoint as OpenAI Chat Completions
            let key = api_key
                .ok_or_else(|| anyhow::anyhow!("OpenAI Responses requires an API key"))?
                .to_string();
            Ok(Box::new(super::openai_chat::OpenAIModelLister::new(
                key,
                base_url.map(|s| s.to_string()),
            )))
        }
        ProviderType::Brainwires
        | ProviderType::Custom
        | ProviderType::MiniMax
        | ProviderType::Bedrock
        | ProviderType::VertexAI
        | ProviderType::ElevenLabs
        | ProviderType::Deepgram
        | ProviderType::Azure
        | ProviderType::Fish
        | ProviderType::Cartesia
        | ProviderType::Murf => Err(anyhow::anyhow!(
            "Model listing is not supported for {} provider via this interface",
            provider_type
        )),
    }
}

// ---------------------------------------------------------------------------
// Response types shared across listers
// ---------------------------------------------------------------------------

/// Anthropic `/v1/models` list response.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicListResponse {
    pub data: Vec<AnthropicModelEntry>,
    pub has_more: bool,
    #[serde(default)]
    pub last_id: Option<String>,
}

/// A model entry from the Anthropic API.
#[derive(Debug, Deserialize)]
pub struct AnthropicModelEntry {
    /// Model identifier (e.g. `"claude-sonnet-4-20250514"`).
    pub id: String,
    /// Human-readable name.
    pub display_name: String,
    /// Resource type (always `"model"`).
    #[serde(rename = "type")]
    pub _type: Option<String>,
    /// ISO 8601 creation timestamp.
    pub created_at: Option<String>,
}

/// OpenAI `/v1/models` response.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIListResponse {
    pub data: Vec<OpenAIModelEntry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIModelEntry {
    pub id: String,
    pub owned_by: Option<String>,
    pub created: Option<i64>,
}

/// Google `models` list response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct GoogleListResponse {
    #[serde(default)]
    pub models: Vec<GoogleModelEntry>,
    #[serde(rename = "nextPageToken")]
    pub next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct GoogleModelEntry {
    /// e.g. "models/gemini-2.0-flash"
    pub name: String,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(rename = "inputTokenLimit")]
    pub input_token_limit: Option<u32>,
    #[serde(rename = "outputTokenLimit")]
    pub output_token_limit: Option<u32>,
    #[serde(rename = "supportedGenerationMethods", default)]
    pub supported_generation_methods: Vec<String>,
}

/// Ollama `/api/tags` response.
#[derive(Debug, Deserialize)]
pub(crate) struct OllamaTagsResponse {
    pub models: Vec<OllamaModelEntry>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OllamaModelEntry {
    pub name: String,
    pub modified_at: Option<String>,
    pub size: Option<u64>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_openai_capabilities_chat() {
        let caps = infer_openai_capabilities("gpt-4o");
        assert!(caps.contains(&ModelCapability::Chat));
        assert!(caps.contains(&ModelCapability::ToolUse));
        assert!(caps.contains(&ModelCapability::Vision));
    }

    #[test]
    fn test_infer_openai_capabilities_embedding() {
        let caps = infer_openai_capabilities("text-embedding-3-small");
        assert!(caps.contains(&ModelCapability::Embedding));
        assert!(!caps.contains(&ModelCapability::Chat));
    }

    #[test]
    fn test_infer_openai_capabilities_audio() {
        let caps = infer_openai_capabilities("whisper-1");
        assert!(caps.contains(&ModelCapability::Audio));
        assert!(!caps.contains(&ModelCapability::Chat));
    }

    #[test]
    fn test_infer_openai_capabilities_image_gen() {
        let caps = infer_openai_capabilities("dall-e-3");
        assert!(caps.contains(&ModelCapability::ImageGeneration));
        assert!(!caps.contains(&ModelCapability::Chat));
    }

    #[test]
    fn test_infer_openai_capabilities_basic_chat() {
        let caps = infer_openai_capabilities("gpt-3.5-turbo");
        assert!(caps.contains(&ModelCapability::Chat));
        assert!(caps.contains(&ModelCapability::ToolUse));
        assert!(!caps.contains(&ModelCapability::Vision));
    }

    #[test]
    fn test_available_model_is_chat_capable() {
        let model = AvailableModel {
            id: "test".to_string(),
            display_name: None,
            provider: ProviderType::OpenAI,
            capabilities: vec![ModelCapability::Chat],
            owned_by: None,
            context_window: None,
            max_output_tokens: None,
            created_at: None,
        };
        assert!(model.is_chat_capable());

        let embedding_model = AvailableModel {
            id: "embed".to_string(),
            display_name: None,
            provider: ProviderType::OpenAI,
            capabilities: vec![ModelCapability::Embedding],
            owned_by: None,
            context_window: None,
            max_output_tokens: None,
            created_at: None,
        };
        assert!(!embedding_model.is_chat_capable());
    }

    #[test]
    fn test_parse_anthropic_response() {
        let json = r#"{
            "data": [
                {"id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4", "type": "model", "created_at": "2025-05-14T00:00:00Z"},
                {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku", "type": "model"}
            ],
            "has_more": false
        }"#;
        let resp: AnthropicListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "claude-sonnet-4-20250514");
        assert!(!resp.has_more);
    }

    #[test]
    fn test_parse_openai_response() {
        let json = r#"{
            "data": [
                {"id": "gpt-4o", "owned_by": "openai", "created": 1715367049},
                {"id": "text-embedding-3-small", "owned_by": "openai", "created": 1705948997}
            ]
        }"#;
        let resp: OpenAIListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4o");
    }

    #[test]
    fn test_parse_google_response() {
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                    "inputTokenLimit": 1048576,
                    "outputTokenLimit": 8192,
                    "supportedGenerationMethods": ["generateContent", "countTokens"]
                }
            ]
        }"#;
        let resp: GoogleListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 1);
        assert_eq!(resp.models[0].input_token_limit, Some(1048576));
    }

    #[test]
    fn test_parse_ollama_response() {
        let json = r#"{
            "models": [
                {"name": "llama3.1:latest", "modified_at": "2024-08-01T00:00:00Z", "size": 4000000000},
                {"name": "codellama:7b", "modified_at": "2024-07-15T00:00:00Z", "size": 3800000000}
            ]
        }"#;
        let resp: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 2);
        assert_eq!(resp.models[0].name, "llama3.1:latest");
    }

    #[test]
    fn test_model_capability_display() {
        assert_eq!(ModelCapability::Chat.to_string(), "chat");
        assert_eq!(ModelCapability::ToolUse.to_string(), "tool_use");
        assert_eq!(ModelCapability::Vision.to_string(), "vision");
    }

    #[test]
    fn test_create_model_lister_no_key() {
        let result = create_model_lister(ProviderType::Anthropic, None, None);
        assert!(result.is_err());
        let err = result.map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("API key"));
    }

    #[test]
    fn test_create_model_lister_ollama_no_key() {
        let result = create_model_lister(ProviderType::Ollama, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_model_lister_brainwires_unsupported() {
        let result = create_model_lister(ProviderType::Brainwires, Some("key"), None);
        assert!(result.is_err());
    }
}
