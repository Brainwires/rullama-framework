use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelMapping {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelSlots {
    pub opus: ModelMapping,
    pub sonnet: ModelMapping,
    pub haiku: ModelMapping,
}

#[derive(Debug, Deserialize)]
pub struct AdapterConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub models: ModelSlots,
    pub port: Option<u16>,
}

#[derive(Debug)]
pub struct ResolvedModel<'a> {
    pub provider: &'a ProviderConfig,
    pub target_model: &'a str,
}

impl AdapterConfig {
    pub fn load_from(path: PathBuf) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path.display(), e))?;
        let config: Self = serde_json::from_str(&data)
            .map_err(|e| anyhow::anyhow!("failed to parse {}: {}", path.display(), e))?;
        Ok(config)
    }

    /// Resolve an incoming Anthropic model name to a provider + target model.
    ///
    /// Matches "opus", "sonnet", or "haiku" anywhere in the model string.
    /// Defaults to the sonnet slot for unrecognized models.
    pub fn resolve_model(&self, incoming: &str) -> anyhow::Result<ResolvedModel<'_>> {
        let lower = incoming.to_lowercase();
        let mapping = if lower.contains("opus") {
            &self.models.opus
        } else if lower.contains("haiku") {
            &self.models.haiku
        } else {
            // Default: sonnet (covers "sonnet" and any unrecognized model)
            &self.models.sonnet
        };

        let provider = self.providers.get(&mapping.provider).ok_or_else(|| {
            anyhow::anyhow!("provider '{}' not found in config", mapping.provider)
        })?;

        Ok(ResolvedModel {
            provider,
            target_model: &mapping.model,
        })
    }
}
