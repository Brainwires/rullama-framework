use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::auth::SessionManager;
use crate::utils::paths::PlatformPaths;

/// Model information (CLI-friendly format)
/// All models are accessed through the Brainwires backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub ai_vendor: String, // The actual AI provider (anthropic, openai, google, etc.)
    pub context_window: u32,
    pub is_default: bool,
}

/// Backend model structure (matches database schema)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendModel {
    pub model_id: String,
    pub model_name: String,
    #[serde(alias = "provider_id")] // Support both names during transition
    pub ai_vendor: String, // The AI vendor (anthropic, openai, google, etc.)
    pub hosted_id: String,
    pub image_input: bool,
    pub abilities: String,
    pub is_premium: bool,
    pub available: String, // AVAILABLE, PREMIUM, TESTING, UNAVAILABLE
    pub pricing_currency: Option<String>,
    pub pricing_unit: Option<String>,
    pub pricing_input_cost: Option<f64>,
    pub pricing_output_cost: Option<f64>,
    pub max_content_length: Option<i64>,
    pub max_token_output_length: Option<i64>,
    pub min_temperature: f64,
    pub max_temperature: f64,
    pub api_adapter: String,
    pub updated_at: String,
}

impl BackendModel {
    /// Convert backend model to ModelInfo
    pub fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.model_id.clone(),
            name: self.model_name.clone(),
            ai_vendor: self.ai_vendor.clone(),
            context_window: self.max_content_length.unwrap_or(200_000) as u32,
            is_default: self.model_id == "claude-3-5-sonnet-20241022",
        }
    }
}

/// Cached models with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCache {
    pub models: Vec<BackendModel>,
    pub cached_at: DateTime<Utc>,
}

impl ModelCache {
    /// Check if cache is still valid (24 hours)
    pub fn is_valid(&self) -> bool {
        let now = Utc::now();
        let age = now.signed_duration_since(self.cached_at);
        age < Duration::hours(24)
    }
}

/// Model client for fetching models from Brainwires backend
pub struct ModelClient {
    http_client: Client,
    backend_url: String,
}

impl ModelClient {
    /// Create a new model client
    pub fn new(backend_url: String) -> Self {
        Self {
            http_client: Client::new(),
            backend_url,
        }
    }

    /// Get cache file path
    fn cache_path() -> Result<PathBuf> {
        Ok(PlatformPaths::brainwires_data_dir()?.join("models_cache.json"))
    }

    /// Load cached models
    fn load_cache() -> Option<ModelCache> {
        let path = Self::cache_path().ok()?;
        if !path.exists() {
            return None;
        }

        let content = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Save models to cache
    fn save_cache(models: &[BackendModel]) -> Result<()> {
        let cache = ModelCache {
            models: models.to_vec(),
            cached_at: Utc::now(),
        };

        let path = Self::cache_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create cache directory")?;
        }

        let content = serde_json::to_string_pretty(&cache)?;
        fs::write(&path, content).context("Failed to write models cache")?;

        Ok(())
    }

    /// Fetch models from backend API
    pub async fn fetch_models(&self, api_key: &str) -> Result<Vec<BackendModel>> {
        let url = format!("{}/api/models", self.backend_url);

        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch models from backend")?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            return Err(anyhow::anyhow!(
                "Failed to fetch models (status {}): {}",
                status,
                error_text
            ));
        }

        // Get the raw response text first for debugging
        let response_text = response
            .text()
            .await
            .context("Failed to read response body")?;

        let models: Vec<BackendModel> =
            serde_json::from_str(&response_text).with_context(|| {
                format!(
                    "Failed to parse models response. Response was: {}",
                    &response_text[..response_text.len().min(200)]
                )
            })?;

        // Save to cache
        Self::save_cache(&models)?;

        Ok(models)
    }

    /// Get models with caching
    pub async fn get_models(&self, use_cache: bool, api_key: &str) -> Result<Vec<BackendModel>> {
        // Try cache first if allowed
        if use_cache
            && let Some(cache) = Self::load_cache()
            && cache.is_valid()
        {
            return Ok(cache.models);
        }

        // Fetch from backend
        self.fetch_models(api_key).await
    }

    /// Get models or fall back to hardcoded defaults
    pub async fn get_models_with_fallback(&self, api_key: &str) -> Vec<BackendModel> {
        // Try cached models first
        if let Some(cache) = Self::load_cache()
            && cache.is_valid()
        {
            return cache.models;
        }

        // Try fetching from backend
        match self.fetch_models(api_key).await {
            Ok(models) => models,
            Err(_) => {
                // Fall back to hardcoded default
                vec![BackendModel {
                    model_id: "claude-3-5-sonnet-20241022".to_string(),
                    model_name: "Claude 3.5 Sonnet".to_string(),
                    ai_vendor: "anthropic".to_string(),
                    hosted_id: "claude-3-5-sonnet-20241022".to_string(),
                    image_input: true,
                    abilities: "text,vision".to_string(),
                    is_premium: false,
                    available: "AVAILABLE".to_string(),
                    pricing_currency: Some("USD".to_string()),
                    pricing_unit: Some("per_million_tokens".to_string()),
                    pricing_input_cost: Some(3.0),
                    pricing_output_cost: Some(15.0),
                    max_content_length: Some(200_000),
                    max_token_output_length: Some(8_192),
                    min_temperature: 0.0,
                    max_temperature: 1.0,
                    api_adapter: "anthropic".to_string(),
                    updated_at: Utc::now().to_rfc3339(),
                }]
            }
        }
    }
}

/// Model registry for managing available models
/// All models are fetched from the Brainwires backend
pub struct ModelRegistry;

impl ModelRegistry {
    /// Get default model from backend API
    /// Falls back to a hardcoded default if backend is unavailable
    /// Note: This is only used as a last resort. The config file should have the default.
    pub async fn default_model() -> String {
        match Self::fetch_models().await {
            Ok(models) => {
                // Return the first model marked as default, or the first model
                models
                    .iter()
                    .find(|m| m.is_default)
                    .or_else(|| models.first())
                    .map(|m| m.id.clone())
                    .unwrap_or_else(|| "claude-haiku-4-5-20251001".to_string())
            }
            Err(_) => {
                // Fallback to a small, cheap model the backend actually lists.
                "claude-haiku-4-5-20251001".to_string()
            }
        }
    }

    /// Get default model synchronously (fallback only)
    /// This should only be used where async is not possible
    /// Note: This is only used as a last resort. The config file should have the default.
    pub fn default_model_sync() -> String {
        "claude-haiku-4-5-20251001".to_string()
    }

    /// Fetch models from backend API
    pub async fn fetch_models() -> Result<Vec<ModelInfo>> {
        // Get session to determine backend URL
        let session = SessionManager::get_session()?.ok_or_else(|| {
            anyhow::anyhow!("Not authenticated. Please run 'brainwires auth' first.")
        })?;

        // Get API key from secure storage
        let api_key = SessionManager::get_api_key()?.ok_or_else(|| {
            anyhow::anyhow!("No API key found. Please re-authenticate with: brainwires auth")
        })?;

        let client = ModelClient::new(session.backend.clone());
        let backend_models = client.get_models(true, api_key.as_str()).await?;

        // Convert to ModelInfo
        let models: Vec<ModelInfo> = backend_models.iter().map(|m| m.to_model_info()).collect();

        Ok(models)
    }

    /// Get all models
    pub async fn get_all_models() -> Result<Vec<ModelInfo>> {
        Self::fetch_models().await
    }

    /// Find a model by ID
    pub async fn find_model(model_id: &str) -> Result<Option<ModelInfo>> {
        let models = Self::fetch_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    /// Find AI vendor by model ID (e.g., "anthropic", "openai", etc.)
    pub async fn find_ai_vendor_by_model(model_id: &str) -> Result<Option<String>> {
        let models = Self::fetch_models().await?;
        for model in models {
            if model.id == model_id {
                return Ok(Some(model.ai_vendor));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_sync() {
        let model = ModelRegistry::default_model_sync();
        assert_eq!(model, "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_backend_model_to_model_info() {
        let backend_model = BackendModel {
            model_id: "claude-3-5-sonnet-20241022".to_string(),
            model_name: "Claude 3.5 Sonnet".to_string(),
            ai_vendor: "anthropic".to_string(),
            hosted_id: "claude-3-5-sonnet-20241022".to_string(),
            image_input: true,
            abilities: "text,vision".to_string(),
            is_premium: false,
            available: "AVAILABLE".to_string(),
            pricing_currency: Some("USD".to_string()),
            pricing_unit: Some("per_million_tokens".to_string()),
            pricing_input_cost: Some(3.0),
            pricing_output_cost: Some(15.0),
            max_content_length: Some(200_000),
            max_token_output_length: Some(8_192),
            min_temperature: 0.0,
            max_temperature: 1.0,
            api_adapter: "anthropic".to_string(),
            updated_at: Utc::now().to_rfc3339(),
        };

        let model_info = backend_model.to_model_info();
        assert_eq!(model_info.id, "claude-3-5-sonnet-20241022");
        assert_eq!(model_info.name, "Claude 3.5 Sonnet");
        assert_eq!(model_info.ai_vendor, "anthropic");
        assert_eq!(model_info.context_window, 200_000);
        assert!(model_info.is_default);
    }

    #[test]
    fn test_model_cache_validity() {
        let cache = ModelCache {
            models: vec![],
            cached_at: Utc::now(),
        };
        assert!(cache.is_valid());

        let old_cache = ModelCache {
            models: vec![],
            cached_at: Utc::now() - Duration::hours(25),
        };
        assert!(!old_cache.is_valid());
    }
}
