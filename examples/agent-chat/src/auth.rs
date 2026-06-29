use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

use crate::config::ChatConfig;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ApiKeys {
    #[serde(default)]
    pub keys: HashMap<String, String>,
}

impl ApiKeys {
    fn path() -> std::path::PathBuf {
        ChatConfig::config_dir().join("api_keys.toml")
    }

    pub fn load() -> Result<Self> {
        let path = Self::path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        toml::from_str(&content).with_context(|| "Failed to parse api_keys.toml")
    }

    pub fn save(&self) -> Result<()> {
        let dir = ChatConfig::config_dir();
        fs::create_dir_all(&dir)?;
        let content = toml::to_string_pretty(self)?;
        fs::write(Self::path(), content)?;
        Ok(())
    }

    pub fn set(&mut self, provider: &str, key: String) -> Result<()> {
        self.keys.insert(provider.to_lowercase(), key);
        self.save()
    }

    pub fn remove(&mut self, provider: &str) -> Result<()> {
        self.keys.remove(&provider.to_lowercase());
        self.save()
    }
}

/// Resolve API key for a provider: CLI flag > env var > api_keys.toml
pub fn resolve_api_key(provider: &str, cli_key: Option<&str>) -> Result<Option<String>> {
    if let Some(key) = cli_key {
        return Ok(Some(key.to_string()));
    }

    let env_var = match provider {
        "anthropic" => "ANTHROPIC_API_KEY",
        "openai" | "openai_responses" => "OPENAI_API_KEY",
        "google" => "GOOGLE_API_KEY",
        "groq" => "GROQ_API_KEY",
        "together" => "TOGETHER_API_KEY",
        "fireworks" => "FIREWORKS_API_KEY",
        "anyscale" => "ANYSCALE_API_KEY",
        "brainwires" => "BRAINWIRES_API_KEY",
        _ => "",
    };

    if !env_var.is_empty()
        && let Ok(key) = std::env::var(env_var)
        && !key.is_empty()
    {
        return Ok(Some(key));
    }

    let keys = ApiKeys::load()?;
    Ok(keys.keys.get(&provider.to_lowercase()).cloned())
}
