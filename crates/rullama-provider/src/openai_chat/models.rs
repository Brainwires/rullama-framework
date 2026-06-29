//! Model listing types and the `OpenAIModelLister` implementation.

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::model_listing::{
    AvailableModel, ModelLister, OpenAIListResponse, infer_openai_capabilities,
};

// ---------------------------------------------------------------------------
// Raw response types (from the `/v1/models` endpoint)
// ---------------------------------------------------------------------------

/// Response from the `/v1/models` endpoint.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIListModelsResponse {
    /// List of available models.
    pub data: Vec<OpenAIModelEntry>,
}

/// A single entry in the models list.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIModelEntry {
    /// Model identifier (e.g. `"gpt-4o"`).
    pub id: String,
    /// Organization that owns the model.
    pub owned_by: Option<String>,
    /// Unix timestamp when the model was created.
    pub created: Option<i64>,
}

// ---------------------------------------------------------------------------
// High-level model lister
// ---------------------------------------------------------------------------

const OPENAI_MODELS_LIST_URL: &str = "https://api.openai.com/v1/models";

/// Lists models available from the OpenAI API (or any OpenAI-compatible endpoint).
pub struct OpenAIModelLister {
    api_key: String,
    base_url: String,
    http_client: Client,
}

impl OpenAIModelLister {
    /// Create a new model lister with the given API key and optional base URL.
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| OPENAI_MODELS_LIST_URL.to_string()),
            http_client: Client::new(),
        }
    }
}

#[async_trait]
impl ModelLister for OpenAIModelLister {
    async fn list_models(&self) -> Result<Vec<AvailableModel>> {
        // If the caller passed a chat-completions URL, derive the models URL
        let models_url = if self.base_url.ends_with("/chat/completions") {
            self.base_url.replace("/chat/completions", "/models")
        } else {
            self.base_url.clone()
        };

        let resp = self
            .http_client
            .get(&models_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to list OpenAI models")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI models API returned {}: {}",
                status,
                body
            ));
        }

        let list: OpenAIListResponse = resp
            .json()
            .await
            .context("Failed to parse OpenAI models response")?;

        let models = list
            .data
            .into_iter()
            .map(|entry| AvailableModel {
                id: entry.id.clone(),
                display_name: None,
                provider: crate::ProviderType::OpenAI,
                capabilities: infer_openai_capabilities(&entry.id),
                owned_by: entry.owned_by,
                context_window: None,
                max_output_tokens: None,
                created_at: entry.created,
            })
            .collect();

        Ok(models)
    }
}
