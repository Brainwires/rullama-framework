//! Provider types.
//!
//! Re-exports ChatOptions from brainwires-core and ProviderType/ProviderConfig from brainwires-provider.

use serde::{Deserialize, Serialize};

// Re-export ChatOptions from framework
pub use brainwires::core::provider::ChatOptions;

// Re-export from providers crate
pub use brainwires::providers::ProviderConfig;
pub use brainwires::providers::ProviderType;

/// Brainwires backend configuration (CLI-specific)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainwiresConfig {
    /// Model name
    pub model: String,
    /// API key (always required for Brainwires)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Backend URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_url: Option<String>,
    /// Additional configuration options
    #[serde(flatten)]
    pub options: std::collections::HashMap<String, serde_json::Value>,
}

impl BrainwiresConfig {
    /// Create a new Brainwires config
    pub fn new(model: String) -> Self {
        Self {
            model,
            api_key: None,
            backend_url: None,
            options: std::collections::HashMap::new(),
        }
    }

    /// Set API key
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set backend URL
    pub fn with_backend_url<S: Into<String>>(mut self, backend_url: S) -> Self {
        self.backend_url = Some(backend_url.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_options_default() {
        let opts = ChatOptions::default();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.max_tokens, Some(4096));
        assert!(opts.system.is_none());
        assert!(opts.top_p.is_none());
    }

    #[test]
    fn test_brainwires_config() {
        let config = BrainwiresConfig::new("claude-3-5-sonnet-20241022".to_string())
            .with_api_key("test-key");
        assert_eq!(config.model, "claude-3-5-sonnet-20241022");
        assert!(config.api_key.is_some());
    }

    #[test]
    fn test_provider_type_default_model() {
        assert_eq!(ProviderType::Anthropic.default_model(), "claude-sonnet-4-6");
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
            ProviderType::from_str_opt("ollama"),
            Some(ProviderType::Ollama)
        );
    }

    #[test]
    fn test_provider_type_as_str() {
        assert_eq!(ProviderType::Anthropic.as_str(), "anthropic");
        assert_eq!(ProviderType::OpenAI.as_str(), "openai");
    }
}
