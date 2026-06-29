//! Provider registry — connection details for all known providers.
//!
//! Maps each [`ProviderType`] to its wire protocol, default endpoint, auth
//! scheme, and model-listing URL. The `ChatProviderFactory`
//! uses the registry for protocol dispatch, eliminating per-provider boilerplate.

use super::ProviderType;

/// Wire protocol spoken by a chat provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatProtocol {
    /// OpenAI Chat Completions (`POST /v1/chat/completions`).
    OpenAiChatCompletions,
    /// OpenAI Responses API (`POST /v1/responses`).
    OpenAiResponses,
    /// Anthropic Messages (`POST /v1/messages`).
    AnthropicMessages,
    /// Google Gemini `generateContent`.
    GeminiGenerateContent,
    /// Ollama native chat (`POST /api/chat`).
    OllamaChat,
    /// Brainwires HTTP relay.
    BrainwiresRelay,
}

/// Authentication scheme used to authorize requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthScheme {
    /// Standard `Authorization: Bearer <token>` header.
    BearerToken,
    /// Custom header name (e.g. Anthropic uses `x-api-key`).
    CustomHeader {
        /// Header name.
        header: &'static str,
    },
    /// AWS SigV4 request signing (Amazon Bedrock).
    AwsSigV4,
    /// Google OAuth2 service-account token (Vertex AI).
    GoogleOAuth,
    /// No authentication required (e.g. local Ollama).
    None,
}

/// Static metadata for a known provider.
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    /// Which provider this entry describes.
    pub provider_type: ProviderType,
    /// Chat wire protocol.
    pub chat_protocol: ChatProtocol,
    /// Default base URL for API requests.
    pub default_base_url: &'static str,
    /// Default model identifier.
    pub default_model: &'static str,
    /// Authentication scheme.
    pub auth: AuthScheme,
    /// Whether the provider supports listing available models.
    pub supports_model_listing: bool,
    /// URL for the models endpoint (if supported).
    pub models_url: Option<&'static str>,
}

/// Static table of all known chat providers.
///
/// Audio-only providers (ElevenLabs, Deepgram, etc.) are not included here
/// because they don't implement the `Provider` (chat) trait.
pub static PROVIDER_REGISTRY: &[ProviderEntry] = &[
    ProviderEntry {
        provider_type: ProviderType::OpenAI,
        chat_protocol: ChatProtocol::OpenAiChatCompletions,
        default_base_url: "https://api.openai.com/v1/chat/completions",
        default_model: "gpt-4o",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://api.openai.com/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Groq,
        chat_protocol: ChatProtocol::OpenAiChatCompletions,
        default_base_url: "https://api.groq.com/openai/v1/chat/completions",
        default_model: "llama-3.3-70b-versatile",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://api.groq.com/openai/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Together,
        chat_protocol: ChatProtocol::OpenAiChatCompletions,
        default_base_url: "https://api.together.xyz/v1/chat/completions",
        default_model: "meta-llama/Llama-3.1-8B-Instruct",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://api.together.xyz/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Fireworks,
        chat_protocol: ChatProtocol::OpenAiChatCompletions,
        default_base_url: "https://api.fireworks.ai/inference/v1/chat/completions",
        default_model: "accounts/fireworks/models/llama-v3p1-8b-instruct",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://api.fireworks.ai/inference/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Anyscale,
        chat_protocol: ChatProtocol::OpenAiChatCompletions,
        default_base_url: "https://api.endpoints.anyscale.com/v1/chat/completions",
        default_model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://api.endpoints.anyscale.com/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Anthropic,
        chat_protocol: ChatProtocol::AnthropicMessages,
        default_base_url: "https://api.anthropic.com/v1/messages",
        default_model: "claude-sonnet-4-6",
        auth: AuthScheme::CustomHeader {
            header: "x-api-key",
        },
        supports_model_listing: true,
        models_url: Some("https://api.anthropic.com/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Bedrock,
        chat_protocol: ChatProtocol::AnthropicMessages,
        default_base_url: "https://bedrock-runtime.us-east-1.amazonaws.com",
        default_model: "anthropic.claude-sonnet-4-6-v1:0",
        auth: AuthScheme::AwsSigV4,
        supports_model_listing: false,
        models_url: None,
    },
    ProviderEntry {
        provider_type: ProviderType::VertexAI,
        chat_protocol: ChatProtocol::AnthropicMessages,
        default_base_url: "https://us-central1-aiplatform.googleapis.com",
        default_model: "claude-sonnet-4-6",
        auth: AuthScheme::GoogleOAuth,
        supports_model_listing: false,
        models_url: None,
    },
    ProviderEntry {
        provider_type: ProviderType::Google,
        chat_protocol: ChatProtocol::GeminiGenerateContent,
        default_base_url: "https://generativelanguage.googleapis.com",
        default_model: "gemini-2.0-flash-exp",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://generativelanguage.googleapis.com/v1beta/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Ollama,
        chat_protocol: ChatProtocol::OllamaChat,
        default_base_url: "http://localhost:11434",
        default_model: "llama3.1",
        auth: AuthScheme::None,
        supports_model_listing: true,
        models_url: Some("http://localhost:11434/api/tags"),
    },
    ProviderEntry {
        provider_type: ProviderType::OpenAiResponses,
        chat_protocol: ChatProtocol::OpenAiResponses,
        default_base_url: "https://api.openai.com/v1/responses",
        default_model: "gpt-5-mini",
        auth: AuthScheme::BearerToken,
        supports_model_listing: true,
        models_url: Some("https://api.openai.com/v1/models"),
    },
    ProviderEntry {
        provider_type: ProviderType::Brainwires,
        chat_protocol: ChatProtocol::BrainwiresRelay,
        default_base_url: "https://brainwires.studio",
        default_model: "gpt-5-mini",
        auth: AuthScheme::BearerToken,
        supports_model_listing: false,
        models_url: None,
    },
    ProviderEntry {
        provider_type: ProviderType::MiniMax,
        chat_protocol: ChatProtocol::OpenAiChatCompletions,
        default_base_url: "https://api.minimax.io/v1/chat/completions",
        default_model: "MiniMax-M2.7",
        auth: AuthScheme::BearerToken,
        supports_model_listing: false,
        models_url: None,
    },
];

/// Look up the registry entry for a given provider type.
pub fn lookup(provider_type: ProviderType) -> Option<&'static ProviderEntry> {
    PROVIDER_REGISTRY
        .iter()
        .find(|e| e.provider_type == provider_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_known_providers() {
        assert!(lookup(ProviderType::OpenAI).is_some());
        assert!(lookup(ProviderType::Groq).is_some());
        assert!(lookup(ProviderType::Together).is_some());
        assert!(lookup(ProviderType::Fireworks).is_some());
        assert!(lookup(ProviderType::Anyscale).is_some());
        assert!(lookup(ProviderType::Anthropic).is_some());
        assert!(lookup(ProviderType::Bedrock).is_some());
        assert!(lookup(ProviderType::VertexAI).is_some());
        assert!(lookup(ProviderType::Google).is_some());
        assert!(lookup(ProviderType::Ollama).is_some());
        assert!(lookup(ProviderType::OpenAiResponses).is_some());
        assert!(lookup(ProviderType::Brainwires).is_some());
        assert!(lookup(ProviderType::MiniMax).is_some());
    }

    #[test]
    fn test_lookup_audio_only_returns_none() {
        assert!(lookup(ProviderType::ElevenLabs).is_none());
        assert!(lookup(ProviderType::Deepgram).is_none());
    }

    #[test]
    fn test_openai_compat_aliases_share_protocol() {
        let groq = lookup(ProviderType::Groq).unwrap();
        let together = lookup(ProviderType::Together).unwrap();
        let fireworks = lookup(ProviderType::Fireworks).unwrap();
        let anyscale = lookup(ProviderType::Anyscale).unwrap();

        assert_eq!(groq.chat_protocol, ChatProtocol::OpenAiChatCompletions);
        assert_eq!(together.chat_protocol, ChatProtocol::OpenAiChatCompletions);
        assert_eq!(fireworks.chat_protocol, ChatProtocol::OpenAiChatCompletions);
        assert_eq!(anyscale.chat_protocol, ChatProtocol::OpenAiChatCompletions);
    }

    #[test]
    fn test_anthropic_protocol_variants() {
        let anthropic = lookup(ProviderType::Anthropic).unwrap();
        let bedrock = lookup(ProviderType::Bedrock).unwrap();
        let vertex = lookup(ProviderType::VertexAI).unwrap();

        assert_eq!(anthropic.chat_protocol, ChatProtocol::AnthropicMessages);
        assert_eq!(bedrock.chat_protocol, ChatProtocol::AnthropicMessages);
        assert_eq!(vertex.chat_protocol, ChatProtocol::AnthropicMessages);

        // But different auth schemes
        assert_eq!(
            anthropic.auth,
            AuthScheme::CustomHeader {
                header: "x-api-key"
            }
        );
        assert_eq!(bedrock.auth, AuthScheme::AwsSigV4);
        assert_eq!(vertex.auth, AuthScheme::GoogleOAuth);
    }

    #[test]
    fn test_ollama_no_auth() {
        let ollama = lookup(ProviderType::Ollama).unwrap();
        assert_eq!(ollama.auth, AuthScheme::None);
    }
}
