use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::debug;

use brainwires_datasets::DataFormat;

use super::{CloudFineTuneConfig, FineTuneProvider};
use crate::error::TrainingError;
use crate::types::{
    DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary, TrainingProgress,
};

const FIREWORKS_API_URL: &str = "https://api.fireworks.ai/v1";

/// Fireworks AI fine-tuning provider.
///
/// Supports SFT V2 and Reinforced Fine-Tuning (RFT).
pub struct FireworksFineTune {
    api_key: String,
    client: Client,
    base_url: String,
    account_id: Option<String>,
}

impl FireworksFineTune {
    /// Create a new Fireworks AI fine-tune provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            base_url: FIREWORKS_API_URL.to_string(),
            account_id: None,
        }
    }

    /// Set the Fireworks account ID.
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Extract error message from API response body.
    fn extract_error(body: &serde_json::Value) -> String {
        body.get("error")
            .and_then(|e| {
                e.get("message")
                    .and_then(|m| m.as_str())
                    .or_else(|| e.as_str())
            })
            .unwrap_or("Unknown error")
            .to_string()
    }

    /// Parse job status from API response. Fireworks uses "state" field with UPPER_CASE values.
    fn parse_job_status(body: &serde_json::Value) -> TrainingJobStatus {
        let status_str = body
            .get("state")
            .or_else(|| body.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("pending");

        match status_str {
            "PENDING" | "pending" => TrainingJobStatus::Pending,
            "RUNNING" | "running" => TrainingJobStatus::Running {
                progress: TrainingProgress::default(),
            },
            "COMPLETED" | "completed" => {
                let model_id = body
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                TrainingJobStatus::Succeeded { model_id }
            }
            "FAILED" | "failed" => TrainingJobStatus::Failed {
                error: Self::extract_error(body),
            },
            "CANCELLED" | "cancelled" => TrainingJobStatus::Cancelled,
            _ => TrainingJobStatus::Pending,
        }
    }
}

#[async_trait]
impl FineTuneProvider for FireworksFineTune {
    fn name(&self) -> &str {
        "fireworks"
    }

    fn supported_base_models(&self) -> Vec<String> {
        vec![
            "accounts/fireworks/models/llama-v3p1-8b-instruct".to_string(),
            "accounts/fireworks/models/llama-v3p1-70b-instruct".to_string(),
            "accounts/fireworks/models/mixtral-8x7b-instruct".to_string(),
            "accounts/fireworks/models/qwen2p5-7b-instruct".to_string(),
        ]
    }

    fn supports_dpo(&self) -> bool {
        false // Fireworks uses RFT, not DPO
    }

    async fn upload_dataset(
        &self,
        data: &[u8],
        _format: DataFormat,
    ) -> Result<DatasetId, TrainingError> {
        debug!("Uploading dataset to Fireworks AI ({} bytes)", data.len());

        let part = reqwest::multipart::Part::bytes(data.to_vec()).file_name("training_data.jsonl");

        let form = reqwest::multipart::Form::new().part("file", part);

        let response = self
            .client
            .post(format!("{}/datasets", self.base_url))
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

        let dataset_id = body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Upload("Missing dataset ID".to_string()))?
            .to_string();

        Ok(DatasetId(dataset_id))
    }

    async fn create_job(
        &self,
        config: CloudFineTuneConfig,
    ) -> Result<TrainingJobId, TrainingError> {
        debug!(
            "Creating Fireworks fine-tuning job for: {}",
            config.base_model
        );

        let body = json!({
            "dataset": config.training_dataset.0,
            "model": config.base_model,
            "epochs": config.hyperparams.epochs,
            "learning_rate": config.hyperparams.learning_rate,
            "batch_size": config.hyperparams.batch_size,
        });

        let response = self
            .client
            .post(format!("{}/fine-tuning/jobs", self.base_url))
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
        let url = format!("{}/fine-tuning/jobs/{}", self.base_url, job_id.0);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let body: serde_json::Value = response.json().await?;
        Ok(Self::parse_job_status(&body))
    }

    async fn cancel_job(&self, job_id: &TrainingJobId) -> Result<(), TrainingError> {
        let url = format!("{}/fine-tuning/jobs/{}/cancel", self.base_url, job_id.0);
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
            .get(format!("{}/fine-tuning/jobs", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let body: serde_json::Value = response.json().await?;
        let jobs = body.get("data").and_then(|v| v.as_array());

        Ok(jobs
            .map(|arr| {
                arr.iter()
                    .filter_map(|j| {
                        Some(TrainingJobSummary {
                            job_id: TrainingJobId(j.get("id")?.as_str()?.to_string()),
                            provider: "fireworks".to_string(),
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
