use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    #[serde(default = "default_provider")]
    pub default_provider: String,
    #[serde(default = "default_model")]
    pub default_model: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default = "default_permission_mode")]
    pub permission_mode: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_provider() -> String {
    "anthropic".into()
}
fn default_model() -> String {
    "claude-sonnet-4-20250514".into()
}
fn default_permission_mode() -> String {
    "ask".into()
}
fn default_max_tokens() -> u32 {
    4096
}
fn default_temperature() -> f32 {
    0.7
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            default_provider: default_provider(),
            default_model: default_model(),
            system_prompt: None,
            permission_mode: default_permission_mode(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
    }
}

impl ChatConfig {
    pub fn config_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".brainwires")
            .join("chat")
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.toml")
    }

    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if !path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        toml::from_str(&content).with_context(|| "Failed to parse config.toml")
    }

    pub fn save(&self) -> Result<()> {
        let dir = Self::config_dir();
        fs::create_dir_all(&dir)?;
        let content = toml::to_string_pretty(self)?;
        fs::write(Self::config_path(), content)?;
        Ok(())
    }

    /// Load a config from an explicit path. Returns `Ok(Default)` when the
    /// file is missing. Unlike [`load`], this does NOT write anything to disk.
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        toml::from_str(&content).with_context(|| "Failed to parse config.toml")
    }

    /// Save this config to an explicit path, creating parent directories as
    /// needed.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "default_provider" => Some(self.default_provider.clone()),
            "default_model" => Some(self.default_model.clone()),
            "system_prompt" => self.system_prompt.clone(),
            "permission_mode" => Some(self.permission_mode.clone()),
            "max_tokens" => Some(self.max_tokens.to_string()),
            "temperature" => Some(self.temperature.to_string()),
            _ => None,
        }
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "default_provider" => self.default_provider = value.to_string(),
            "default_model" => self.default_model = value.to_string(),
            "system_prompt" => self.system_prompt = Some(value.to_string()),
            "permission_mode" => self.permission_mode = value.to_string(),
            "max_tokens" => self.max_tokens = value.parse()?,
            "temperature" => self.temperature = value.parse()?,
            _ => anyhow::bail!("Unknown config key: {key}"),
        }
        self.save()
    }
}
