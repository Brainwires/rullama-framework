//! Chat provider factory — registry-driven protocol dispatch.
//!
//! Creates `Arc<dyn Provider>` from a [`ProviderConfig`] by looking up the
//! provider in the registry and dispatching to the
//! appropriate protocol handler.

use std::sync::Arc;

use anyhow::{Result, anyhow};

use super::ProviderConfig;
#[cfg(any(feature = "bedrock", feature = "vertex-ai"))]
use super::ProviderType;
use super::registry::{self, ChatProtocol};
use brainwires_core::Provider;

/// Pure chat provider factory — creates provider instances from config.
///
/// No CLI dependencies (no SessionManager, no keyring, no file I/O).
/// The caller is responsible for resolving API keys and base URLs
/// before calling `create()`.
pub struct ChatProviderFactory;

impl ChatProviderFactory {
    /// Create a chat provider from a fully-resolved config.
    ///
    /// All fields (api_key, base_url, model) must already be populated.
    pub fn create(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        let entry = registry::lookup(config.provider)
            .ok_or_else(|| anyhow!("Provider type '{}' is not a chat provider", config.provider))?;

        match entry.chat_protocol {
            ChatProtocol::OpenAiChatCompletions => {
                Self::create_openai_compat(config, entry.default_base_url)
            }
            ChatProtocol::OpenAiResponses => Self::create_openai_responses(config),
            ChatProtocol::AnthropicMessages => Self::create_anthropic(config),
            ChatProtocol::GeminiGenerateContent => Self::create_gemini(config),
            ChatProtocol::OllamaChat => Self::create_ollama(config),
            ChatProtocol::BrainwiresRelay => Self::create_brainwires(config),
        }
    }

    // -----------------------------------------------------------------------
    // Protocol-specific constructors
    // -----------------------------------------------------------------------

    fn create_openai_compat(
        config: &ProviderConfig,
        default_base_url: &str,
    ) -> Result<Arc<dyn Provider>> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("{} provider requires an API key", config.provider))?;
        let mut client = super::openai_chat::OpenAiClient::new(api_key, config.model.clone());
        let base_url = config.base_url.as_deref().unwrap_or(default_base_url);
        client = client.with_base_url(base_url.to_string());
        let client = Arc::new(client);
        let provider =
            super::openai_chat::chat::OpenAiChatProvider::new(client, config.model.clone())
                .with_provider_name(config.provider.as_str());
        #[cfg(feature = "telemetry")]
        let provider = match config.analytics_collector.as_ref() {
            Some(c) => provider.with_analytics(Arc::clone(c)),
            None => provider,
        };
        Ok(Arc::new(provider))
    }

    fn create_openai_responses(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("OpenAI Responses provider requires an API key"))?;
        let mut client = super::openai_responses::ResponsesClient::new(api_key);
        if let Some(ref base_url) = config.base_url {
            client = client.with_base_url(base_url.clone());
        }
        let client = Arc::new(client);
        let provider =
            super::openai_responses::OpenAiResponsesProvider::new(client, config.model.clone())
                .with_provider_name(config.provider.as_str());
        #[cfg(feature = "telemetry")]
        let provider = match config.analytics_collector.as_ref() {
            Some(c) => provider.with_analytics(Arc::clone(c)),
            None => provider,
        };
        Ok(Arc::new(provider))
    }

    fn create_anthropic(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        match config.provider {
            #[cfg(feature = "bedrock")]
            ProviderType::Bedrock => {
                return Self::create_bedrock(config);
            }
            #[cfg(feature = "vertex-ai")]
            ProviderType::VertexAI => {
                return Self::create_vertex(config);
            }
            _ => {}
        }

        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("{} provider requires an API key", config.provider))?;
        let client = Arc::new(super::anthropic::AnthropicClient::new(
            api_key,
            config.model.clone(),
        ));
        let provider =
            super::anthropic::chat::AnthropicChatProvider::new(client, config.model.clone())
                .with_provider_name(config.provider.as_str());
        #[cfg(feature = "telemetry")]
        let provider = match config.analytics_collector.as_ref() {
            Some(c) => provider.with_analytics(Arc::clone(c)),
            None => provider,
        };
        Ok(Arc::new(provider))
    }

    #[cfg(feature = "bedrock")]
    fn create_bedrock(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        use super::anthropic::bedrock::BedrockAuth;

        let region = config
            .options
            .get("region")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Try explicit credentials from options, fall back to environment
        let auth = if let (Some(access_key), Some(secret_key)) = (
            config.options.get("access_key_id").and_then(|v| v.as_str()),
            config
                .options
                .get("secret_access_key")
                .and_then(|v| v.as_str()),
        ) {
            let session_token = config
                .options
                .get("session_token")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            BedrockAuth::new(
                region.unwrap_or_else(|| "us-east-1".to_string()),
                access_key.to_string(),
                secret_key.to_string(),
                session_token,
            )
        } else {
            BedrockAuth::from_environment(region)?
        };

        let client = Arc::new(super::anthropic::AnthropicClient::bedrock(
            auth,
            config.model.clone(),
        ));
        let mut provider =
            super::anthropic::chat::AnthropicChatProvider::new(client, config.model.clone())
                .with_provider_name("bedrock");
        #[cfg(feature = "telemetry")]
        if let Some(ref c) = config.analytics_collector {
            provider = provider.with_analytics(Arc::clone(c));
        }
        Ok(Arc::new(provider))
    }

    #[cfg(feature = "vertex-ai")]
    fn create_vertex(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        use super::anthropic::vertex::VertexAuth;

        let project_id = config.options.get("project_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or_else(|| anyhow!(
                "Vertex AI requires a project_id. Set it via config options or GOOGLE_CLOUD_PROJECT env var."
            ))?;

        let region = config
            .options
            .get("region")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "us-central1".to_string());

        let auth = VertexAuth::new(project_id, region);

        let client = Arc::new(super::anthropic::AnthropicClient::vertex(
            auth,
            config.model.clone(),
        ));
        let mut provider =
            super::anthropic::chat::AnthropicChatProvider::new(client, config.model.clone())
                .with_provider_name("vertex-ai");
        #[cfg(feature = "telemetry")]
        if let Some(ref c) = config.analytics_collector {
            provider = provider.with_analytics(Arc::clone(c));
        }
        Ok(Arc::new(provider))
    }

    fn create_gemini(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("Google provider requires an API key"))?;
        let client = Arc::new(super::gemini::GoogleClient::new(
            api_key,
            config.model.clone(),
        ));
        let provider = super::gemini::chat::GoogleChatProvider::new(client, config.model.clone());
        #[cfg(feature = "telemetry")]
        let provider = match config.analytics_collector.as_ref() {
            Some(c) => provider.with_analytics(Arc::clone(c)),
            None => provider,
        };
        Ok(Arc::new(provider))
    }

    fn create_ollama(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        let provider = super::ollama::chat::OllamaChatProvider::new(
            config.model.clone(),
            config.base_url.clone(),
        );
        #[cfg(feature = "telemetry")]
        let provider = match config.analytics_collector.as_ref() {
            Some(c) => provider.with_analytics(Arc::clone(c)),
            None => provider,
        };
        Ok(Arc::new(provider))
    }

    fn create_brainwires(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("Brainwires provider requires an API key"))?;
        let backend_url = config.base_url.clone().unwrap_or_else(|| {
            super::brainwires_http::get_backend_from_api_key(&api_key).to_string()
        });
        Ok(Arc::new(
            super::brainwires_http::BrainwiresHttpProvider::new(
                api_key,
                backend_url,
                config.model.clone(),
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProviderType;

    #[test]
    fn test_create_ollama_no_key_required() {
        let config = ProviderConfig::new(ProviderType::Ollama, "llama3.1".to_string());
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "ollama");
    }

    #[test]
    fn test_create_anthropic_requires_key() {
        let config = ProviderConfig::new(ProviderType::Anthropic, "claude-3".to_string());
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_groq_with_key() {
        let config = ProviderConfig::new(ProviderType::Groq, "llama-3.3-70b-versatile".to_string())
            .with_api_key("gsk_test");
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "groq");
    }

    #[test]
    fn test_create_together_with_key() {
        let config = ProviderConfig::new(
            ProviderType::Together,
            "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        )
        .with_api_key("tok_test");
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "together");
    }

    #[test]
    fn test_create_fireworks_with_key() {
        let config = ProviderConfig::new(
            ProviderType::Fireworks,
            "llama-v3p1-8b-instruct".to_string(),
        )
        .with_api_key("fw_test");
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "fireworks");
    }

    #[test]
    fn test_create_minimax_with_key() {
        let config = ProviderConfig::new(ProviderType::MiniMax, "MiniMax-M2.7".to_string())
            .with_api_key("minimax_test_key");
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "minimax");
    }

    #[test]
    fn test_create_minimax_requires_key() {
        let config = ProviderConfig::new(ProviderType::MiniMax, "MiniMax-M2.7".to_string());
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_only_provider_rejected() {
        let config = ProviderConfig::new(
            ProviderType::ElevenLabs,
            "eleven_multilingual_v2".to_string(),
        )
        .with_api_key("key");
        let result = ChatProviderFactory::create(&config);
        assert!(result.is_err());
    }

    #[cfg(feature = "bedrock")]
    #[test]
    fn test_create_bedrock_with_explicit_credentials() {
        let config = ProviderConfig::new(
            ProviderType::Bedrock,
            "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        )
        .with_option("region", serde_json::json!("us-west-2"))
        .with_option("access_key_id", serde_json::json!("AKID_TEST"))
        .with_option("secret_access_key", serde_json::json!("SECRET_TEST"));

        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "bedrock");
    }

    #[cfg(feature = "vertex-ai")]
    #[test]
    fn test_create_vertex_requires_project_id() {
        // No project_id set and no GOOGLE_CLOUD_PROJECT env var
        let config = ProviderConfig::new(
            ProviderType::VertexAI,
            "claude-3-5-sonnet-v2@20241022".to_string(),
        );

        // This should fail unless GOOGLE_CLOUD_PROJECT is set in the test environment
        let result = ChatProviderFactory::create(&config);
        if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
            assert!(result.is_err());
        }
    }

    #[cfg(feature = "vertex-ai")]
    #[test]
    fn test_create_vertex_with_project_id() {
        let config = ProviderConfig::new(
            ProviderType::VertexAI,
            "claude-3-5-sonnet-v2@20241022".to_string(),
        )
        .with_project_id("my-test-project")
        .with_region("europe-west1");

        let result = ChatProviderFactory::create(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "vertex-ai");
    }
}
