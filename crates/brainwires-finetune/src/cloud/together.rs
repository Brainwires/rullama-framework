use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::debug;

use crate::datasets::DataFormat;

use super::{CloudFineTuneConfig, FineTuneProvider};
use crate::error::TrainingError;
use crate::types::{
    DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary, TrainingProgress,
};

const TOGETHER_API_URL: &str = "https://api.together.xyz/v1";

/// Together AI fine-tuning provider.
///
/// Supports SFT and DPO fine-tuning with OpenAI-compatible format.
pub struct TogetherFineTune {
    api_key: String,
    client: Client,
    base_url: String,
}

impl TogetherFineTune {
    /// Create a new Together AI fine-tune provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            base_url: TOGETHER_API_URL.to_string(),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Extract error message from API response body.
    fn extract_error(body: &serde_json::Value) -> String {
        body.get("error")
            .and_then(|e| {
                // Try {"error": {"message": "..."}} first, then {"error": "..."}
                e.get("message")
                    .and_then(|m| m.as_str())
                    .or_else(|| e.as_str())
            })
            .unwrap_or("Unknown error")
            .to_string()
    }

    /// Parse job status from API response.
    fn parse_job_status(body: &serde_json::Value) -> TrainingJobStatus {
        let status_str = body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("pending");

        match status_str {
            "pending" | "queued" => TrainingJobStatus::Queued,
            "running" | "processing" => TrainingJobStatus::Running {
                progress: TrainingProgress::default(),
            },
            "completed" | "succeeded" => {
                let model_id = body
                    .get("output_name")
                    .or_else(|| body.get("fine_tuned_model"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                TrainingJobStatus::Succeeded { model_id }
            }
            "failed" | "error" => TrainingJobStatus::Failed {
                error: Self::extract_error(body),
            },
            "cancelled" => TrainingJobStatus::Cancelled,
            _ => TrainingJobStatus::Pending,
        }
    }
}

#[async_trait]
impl FineTuneProvider for TogetherFineTune {
    fn name(&self) -> &str {
        "together"
    }

    fn supported_base_models(&self) -> Vec<String> {
        vec![
            "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
            "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
            "Qwen/Qwen2.5-7B-Instruct".to_string(),
        ]
    }

    fn supports_dpo(&self) -> bool {
        true
    }

    async fn upload_dataset(
        &self,
        data: &[u8],
        _format: DataFormat,
    ) -> Result<DatasetId, TrainingError> {
        debug!("Uploading dataset to Together AI ({} bytes)", data.len());

        let part = reqwest::multipart::Part::bytes(data.to_vec()).file_name("training_data.jsonl");

        let form = reqwest::multipart::Form::new()
            .text("purpose", "fine-tune")
            .part("file", part);

        let response = self
            .client
            .post(format!("{}/files", self.base_url))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?;

        let status = response.status();
        let body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            return Err(TrainingError::Api {
                message: Self::extract_error(&body),
                status_code: status.as_u16(),
            });
        }

        let file_id = body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Upload("Missing file ID".to_string()))?
            .to_string();

        Ok(DatasetId(file_id))
    }

    async fn create_job(
        &self,
        config: CloudFineTuneConfig,
    ) -> Result<TrainingJobId, TrainingError> {
        debug!(
            "Creating Together AI fine-tuning job for: {}",
            config.base_model
        );

        let mut body = json!({
            "training_file": config.training_dataset.0,
            "model": config.base_model,
            "n_epochs": config.hyperparams.epochs,
            "learning_rate": config.hyperparams.learning_rate,
            "batch_size": config.hyperparams.batch_size,
        });

        if let Some(ref suffix) = config.suffix {
            body["suffix"] = json!(suffix);
        }

        let response = self
            .client
            .post(format!("{}/fine-tunes", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let response_body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            return Err(TrainingError::Api {
                message: Self::extract_error(&response_body),
                status_code: status.as_u16(),
            });
        }

        let job_id = response_body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Provider("Missing job ID".to_string()))?
            .to_string();

        Ok(TrainingJobId(job_id))
    }

    async fn get_job_status(
        &self,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        let url = format!("{}/fine-tunes/{}", self.base_url, job_id.0);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let status = response.status();
        let body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            return Err(TrainingError::Api {
                message: Self::extract_error(&body),
                status_code: status.as_u16(),
            });
        }

        Ok(Self::parse_job_status(&body))
    }

    async fn cancel_job(&self, job_id: &TrainingJobId) -> Result<(), TrainingError> {
        let url = format!("{}/fine-tunes/{}/cancel", self.base_url, job_id.0);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let body: serde_json::Value = response.json().await.unwrap_or_default();
            return Err(TrainingError::Provider(Self::extract_error(&body)));
        }

        Ok(())
    }

    async fn list_jobs(&self) -> Result<Vec<TrainingJobSummary>, TrainingError> {
        let response = self
            .client
            .get(format!("{}/fine-tunes", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let body: serde_json::Value = response.json().await?;
        let data = body.get("data").and_then(|v| v.as_array());

        Ok(data
            .map(|jobs| {
                jobs.iter()
                    .filter_map(|j| {
                        Some(TrainingJobSummary {
                            job_id: TrainingJobId(j.get("id")?.as_str()?.to_string()),
                            provider: "together".to_string(),
                            base_model: j.get("model")?.as_str()?.to_string(),
                            status: Self::parse_job_status(j),
                            created_at: chrono::Utc::now(),
                            metrics: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn delete_model(&self, model_id: &str) -> Result<(), TrainingError> {
        let url = format!("{}/models/{}", self.base_url, model_id);

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let body: serde_json::Value = response.json().await.unwrap_or_default();
            return Err(TrainingError::Provider(Self::extract_error(&body)));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_together_supported_models() {
        let provider = TogetherFineTune::new("test");
        assert!(provider.supports_dpo());
        assert!(!provider.supported_base_models().is_empty());
    }
}
