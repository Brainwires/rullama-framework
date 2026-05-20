#![deny(missing_docs)]
//! Provider layer for the Brainwires Agent Framework.
//!
//! Contains both low-level API client structs (HTTP transport, auth, rate
//! limiting, serde) and high-level chat provider implementations that wrap
//! them with the `brainwires_core::Provider` trait.

// Re-export core traits for convenience
pub use brainwires_core::provider::{ChatOptions, Provider};

// Rate limiting and HTTP client
#[cfg(feature = "native")]
pub mod http_client;
#[cfg(feature = "native")]
pub mod rate_limiter;

#[cfg(feature = "native")]
pub use http_client::RateLimitedClient;
#[cfg(feature = "native")]
pub use rate_limiter::RateLimiter;

// ── Protocol directories ──────────────────────────────────────────────

/// OpenAI Chat Completions protocol (also used by Groq, Together, Fireworks, Anyscale).
#[cfg(feature = "native")]
pub mod openai_chat;

/// OpenAI Responses API protocol (`/v1/responses`).
#[cfg(feature = "native")]
pub mod openai_responses;

/// Anthropic Messages protocol (also used by Bedrock, Vertex AI).
#[cfg(feature = "native")]
pub mod anthropic;

/// Google Gemini generateContent protocol.
#[cfg(feature = "native")]
pub mod gemini;

/// Ollama native chat protocol.
#[cfg(feature = "native")]
pub mod ollama;

/// Brainwires HTTP relay protocol.
#[cfg(feature = "native")]
pub mod brainwires_http;
#[cfg(feature = "native")]
pub use brainwires_http::{DEFAULT_BACKEND_URL, DEV_BACKEND_URL, get_backend_from_api_key};

// Speech (TTS / STT) provider clients live in `brainwires-provider-speech`.

// ── Registry ──────────────────────────────────────────────────────────

/// Provider registry — protocol, auth, and endpoint metadata for all known providers.
pub mod registry;

// ── Model listing ─────────────────────────────────────────────────────

/// Model listing — query available models from provider APIs.
#[cfg(feature = "native")]
pub mod model_listing;

/// Chat provider factory — registry-driven protocol dispatch.
#[cfg(feature = "native")]
pub mod chat_factory;

// ── Local LLM ─────────────────────────────────────────────────────────

/// Local LLM inference (always compiled, llama.cpp behind feature flag).
pub mod local_llm;

// Browser-native `web_speech` lives in `brainwires-provider-speech`.

// ── Re-exports ────────────────────────────────────────────────────────

// Chat-capable API clients
#[cfg(feature = "native")]
pub use anthropic::AnthropicClient;
#[cfg(feature = "native")]
pub use brainwires_http::BrainwiresHttpProvider;
#[cfg(feature = "native")]
pub use gemini::GoogleClient;
#[cfg(feature = "native")]
pub use ollama::OllamaProvider;
#[cfg(feature = "native")]
pub use openai_chat::OpenAiClient;

// Chat providers
#[cfg(feature = "native")]
pub use anthropic::chat::AnthropicChatProvider;
#[cfg(feature = "native")]
pub use gemini::chat::GoogleChatProvider;
#[cfg(feature = "native")]
pub use ollama::chat::OllamaChatProvider;
#[cfg(feature = "native")]
pub use openai_chat::chat::OpenAiChatProvider;
#[cfg(feature = "native")]
pub use openai_responses::OpenAiResponsesProvider;

// Audio/speech client re-exports live on `brainwires-provider-speech`.

// Model listing
#[cfg(feature = "native")]
pub use model_listing::{AvailableModel, ModelCapability, ModelLister, create_model_lister};

// Factory
#[cfg(feature = "native")]
pub use chat_factory::ChatProviderFactory;

// Local LLM
pub use local_llm::*;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// AI provider types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// Anthropic (Claude).
    Anthropic,
    /// OpenAI (GPT).
    OpenAI,
    /// Google (Gemini).
    Google,
    /// Groq inference.
    Groq,
    /// Ollama local models.
    Ollama,
    /// Brainwires HTTP relay.
    Brainwires,
    /// Together AI.
    Together,
    /// Fireworks AI.
    Fireworks,
    /// Anyscale.
    Anyscale,
    /// Amazon Bedrock (Anthropic Messages via AWS SigV4).
    Bedrock,
    /// Google Vertex AI (Anthropic Messages via OAuth2).
    VertexAI,
    /// ElevenLabs.
    ElevenLabs,
    /// Deepgram.
    Deepgram,
    /// Azure Speech.
    Azure,
    /// Fish Audio.
    Fish,
    /// Cartesia.
    Cartesia,
    /// Murf AI.
    Murf,
    /// OpenAI Responses API.
    OpenAiResponses,
    /// MiniMax AI.
    MiniMax,
    /// Custom / user-defined provider.
    Custom,
}

impl ProviderType {
    /// Get the default model for this provider
    pub fn default_model(&self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-4-6",
            Self::OpenAI => "gpt-5-mini",
            Self::Google => "gemini-2.5-flash",
            Self::Groq => "llama-3.3-70b-versatile",
            Self::Ollama => "llama3.3",
            Self::Brainwires => "gpt-5-mini",
            Self::Together => "meta-llama/Llama-3.1-8B-Instruct",
            Self::Fireworks => "accounts/fireworks/models/llama-v3p1-8b-instruct",
            Self::Anyscale => "meta-llama/Meta-Llama-3.1-8B-Instruct",
            Self::Bedrock => "anthropic.claude-sonnet-4-6-v1:0",
            Self::VertexAI => "claude-sonnet-4-6",
            Self::ElevenLabs => "eleven_multilingual_v2",
            Self::Deepgram => "nova-2",
            Self::Azure => "en-US-JennyNeural",
            Self::Fish => "default",
            Self::Cartesia => "sonic-english",
            Self::Murf => "en-US-natalie",
            Self::OpenAiResponses => "gpt-5-mini",
            Self::MiniMax => "MiniMax-M2.7",
            Self::Custom => "claude-sonnet-4-6",
        }
    }

    /// Parse from string
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "anthropic" => Some(Self::Anthropic),
            "openai" => Some(Self::OpenAI),
            "google" | "gemini" => Some(Self::Google),
            "groq" => Some(Self::Groq),
            "ollama" => Some(Self::Ollama),
            "brainwires" => Some(Self::Brainwires),
            "together" => Some(Self::Together),
            "fireworks" => Some(Self::Fireworks),
            "anyscale" => Some(Self::Anyscale),
            "bedrock" => Some(Self::Bedrock),
            "vertex-ai" | "vertexai" | "vertex_ai" => Some(Self::VertexAI),
            "elevenlabs" => Some(Self::ElevenLabs),
            "deepgram" => Some(Self::Deepgram),
            "azure" => Some(Self::Azure),
            "fish" => Some(Self::Fish),
            "cartesia" => Some(Self::Cartesia),
            "murf" => Some(Self::Murf),
            "openai-responses" | "openai_responses" => Some(Self::OpenAiResponses),
            "minimax" => Some(Self::MiniMax),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAI => "openai",
            Self::Google => "google",
            Self::Groq => "groq",
            Self::Ollama => "ollama",
            Self::Brainwires => "brainwires",
            Self::Together => "together",
            Self::Fireworks => "fireworks",
            Self::Anyscale => "anyscale",
            Self::Bedrock => "bedrock",
            Self::VertexAI => "vertex-ai",
            Self::ElevenLabs => "elevenlabs",
            Self::Deepgram => "deepgram",
            Self::Azure => "azure",
            Self::Fish => "fish",
            Self::Cartesia => "cartesia",
            Self::Murf => "murf",
            Self::OpenAiResponses => "openai-responses",
            Self::MiniMax => "minimax",
            Self::Custom => "custom",
        }
    }

    /// Whether this provider requires an API key
    pub fn requires_api_key(&self) -> bool {
        !matches!(self, Self::Ollama | Self::Bedrock | Self::VertexAI)
    }
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ProviderType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_opt(s).ok_or_else(|| anyhow::anyhow!("Unknown provider: {}", s))
    }
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type
    pub provider: ProviderType,
    /// Model name
    pub model: String,
    /// API key (if required)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL (for custom endpoints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Additional provider-specific options
    #[serde(flatten)]
    pub options: std::collections::HashMap<String, serde_json::Value>,
    /// Analytics collector — not serialized, threaded through at runtime.
    #[cfg(feature = "telemetry")]
    #[serde(skip)]
    pub analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl ProviderConfig {
    /// Create a new provider config
    pub fn new(provider: ProviderType, model: String) -> Self {
        Self {
            provider,
            model,
            api_key: None,
            base_url: None,
            options: std::collections::HashMap::new(),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Set API key
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set base URL
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set a provider-specific option.
    pub fn with_option(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.options.insert(key.into(), value);
        self
    }

    /// Set the AWS region (for Bedrock) or GCP region (for Vertex AI).
    pub fn with_region(self, region: impl Into<String>) -> Self {
        self.with_option("region", serde_json::Value::String(region.into()))
    }

    /// Set the GCP project ID (for Vertex AI).
    pub fn with_project_id(self, project_id: impl Into<String>) -> Self {
        self.with_option("project_id", serde_json::Value::String(project_id.into()))
    }

    /// Attach an analytics collector — called by the factory layer before provider construction.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<brainwires_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_default_model() {
        assert_eq!(ProviderType::Anthropic.default_model(), "claude-sonnet-4-6");
        assert_eq!(ProviderType::OpenAI.default_model(), "gpt-5-mini");
        assert_eq!(ProviderType::Google.default_model(), "gemini-2.5-flash");
        assert_eq!(
            ProviderType::Groq.default_model(),
            "llama-3.3-70b-versatile"
        );
        assert_eq!(ProviderType::Ollama.default_model(), "llama3.3");
        assert_eq!(ProviderType::Brainwires.default_model(), "gpt-5-mini");
        assert_eq!(ProviderType::MiniMax.default_model(), "MiniMax-M2.7");
    }

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!(
            ProviderType::from_str_opt("anthropic"),
            Some(ProviderType::Anthropic)
        );
        assert_eq!(
            ProviderType::from_str_opt("openai"),
            Some(ProviderType::OpenAI)
        );
        assert_eq!(
            ProviderType::from_str_opt("google"),
            Some(ProviderType::Google)
        );
        assert_eq!(
            ProviderType::from_str_opt("gemini"),
            Some(ProviderType::Google)
        );
        assert_eq!(ProviderType::from_str_opt("groq"), Some(ProviderType::Groq));
        assert_eq!(
            ProviderType::from_str_opt("ollama"),
            Some(ProviderType::Ollama)
        );
        assert_eq!(
            ProviderType::from_str_opt("brainwires"),
            Some(ProviderType::Brainwires)
        );
        assert_eq!(
            ProviderType::from_str_opt("together"),
            Some(ProviderType::Together)
        );
        assert_eq!(
            ProviderType::from_str_opt("fireworks"),
            Some(ProviderType::Fireworks)
        );
        assert_eq!(
            ProviderType::from_str_opt("anyscale"),
            Some(ProviderType::Anyscale)
        );
        assert_eq!(
            ProviderType::from_str_opt("elevenlabs"),
            Some(ProviderType::ElevenLabs)
        );
        assert_eq!(
            ProviderType::from_str_opt("deepgram"),
            Some(ProviderType::Deepgram)
        );
        assert_eq!(
            ProviderType::from_str_opt("custom"),
            Some(ProviderType::Custom)
        );
        assert_eq!(
            ProviderType::from_str_opt("minimax"),
            Some(ProviderType::MiniMax)
        );
        assert_eq!(ProviderType::from_str_opt("unknown"), None);
    }

    #[test]
    fn test_provider_type_requires_api_key() {
        assert!(ProviderType::Anthropic.requires_api_key());
        assert!(ProviderType::OpenAI.requires_api_key());
        assert!(!ProviderType::Ollama.requires_api_key());
        assert!(ProviderType::ElevenLabs.requires_api_key());
        assert!(ProviderType::MiniMax.requires_api_key());
    }

    #[test]
    fn test_provider_config() {
        let config = ProviderConfig::new(ProviderType::Anthropic, "claude-3".to_string())
            .with_api_key("sk-test")
            .with_base_url("https://api.example.com");
        assert_eq!(config.provider, ProviderType::Anthropic);
        assert_eq!(config.api_key, Some("sk-test".to_string()));
        assert_eq!(config.base_url, Some("https://api.example.com".to_string()));
    }
}
