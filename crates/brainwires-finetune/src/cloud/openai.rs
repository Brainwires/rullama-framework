use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::{debug, warn};

use crate::datasets::DataFormat;

use super::{CloudFineTuneConfig, FineTuneProvider};
use crate::error::TrainingError;
use crate::types::{
    DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary, TrainingProgress,
};

const OPENAI_FILES_URL: &str = "https://api.openai.com/v1/files";
const OPENAI_FINETUNE_URL: &str = "https://api.openai.com/v1/fine_tuning/jobs";

/// OpenAI fine-tuning provider.
pub struct OpenAiFineTune {
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAiFineTune {
    /// Create a new OpenAI fine-tune provider.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn files_url(&self) -> String {
        if self.base_url == "https://api.openai.com/v1" {
            OPENAI_FILES_URL.to_string()
        } else {
            format!("{}/files", self.base_url)
        }
    }

    fn finetune_url(&self) -> String {
        if self.base_url == "https://api.openai.com/v1" {
            OPENAI_FINETUNE_URL.to_string()
        } else {
            format!("{}/fine_tuning/jobs", self.base_url)
        }
    }

    fn parse_job_status(status_str: &str, response: &serde_json::Value) -> TrainingJobStatus {
        match status_str {
            "validating_files" => TrainingJobStatus::Validating,
            "queued" => TrainingJobStatus::Queued,
            "running" => {
                // Extract progress from trained_tokens and similar fields
                let step = response
                    .get("trained_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                TrainingJobStatus::Running {
                    progress: TrainingProgress {
                        step,
                        ..Default::default()
                    },
                }
            }
            "succeeded" => {
                let model_id = response
                    .get("fine_tuned_model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                TrainingJobStatus::Succeeded { model_id }
            }
            "failed" => {
                let error = response
                    .get("error")
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error")
                    .to_string();
                TrainingJobStatus::Failed { error }
            }
            "cancelled" => TrainingJobStatus::Cancelled,
            _ => TrainingJobStatus::Pending,
        }
    }
}

#[async_trait]
impl FineTuneProvider for OpenAiFineTune {
    fn name(&self) -> &str {
        "openai"
    }

    fn supported_base_models(&self) -> Vec<String> {
        vec![
            "gpt-4o-mini-2024-07-18".to_string(),
            "gpt-4o-2024-08-06".to_string(),
            "gpt-4-0613".to_string(),
            "gpt-3.5-turbo-0125".to_string(),
            "gpt-3.5-turbo-1106".to_string(),
        ]
    }

    fn supports_dpo(&self) -> bool {
        true // OpenAI supports DPO via preference data format
    }

    async fn upload_dataset(
        &self,
        data: &[u8],
        _format: DataFormat,
    ) -> Result<DatasetId, TrainingError> {
        debug!("Uploading dataset to OpenAI ({} bytes)", data.len());

        let part = reqwest::multipart::Part::bytes(data.to_vec()).file_name("training_data.jsonl");

        let form = reqwest::multipart::Form::new()
            .text("purpose", "fine-tune")
            .part("file", part);

        let response = self
            .client
            .post(self.files_url())
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?;

        let status = response.status();
        let body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown upload error")
                .to_string();
            return Err(TrainingError::Api {
                message,
                status_code: status.as_u16(),
            });
        }

        let file_id = body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Upload("Missing file ID in response".to_string()))?
            .to_string();

        debug!("Dataset uploaded: {}", file_id);
        Ok(DatasetId(file_id))
    }

    async fn create_job(
        &self,
        config: CloudFineTuneConfig,
    ) -> Result<TrainingJobId, TrainingError> {
        debug!(
            "Creating OpenAI fine-tuning job for model: {}",
            config.base_model
        );

        let mut body = json!({
            "training_file": config.training_dataset.0,
            "model": config.base_model,
            "hyperparameters": {
                "n_epochs": config.hyperparams.epochs,
                "batch_size": config.hyperparams.batch_size,
                "learning_rate_multiplier": config.hyperparams.learning_rate / 2e-5,
            },
        });

        if let Some(ref val_dataset) = config.validation_dataset {
            body["validation_file"] = json!(val_dataset.0);
        }

        if let Some(ref suffix) = config.suffix {
            body["suffix"] = json!(suffix);
        }

        let response = self
            .client
            .post(self.finetune_url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let response_body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            let message = response_body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error")
                .to_string();
            return Err(TrainingError::Api {
                message,
                status_code: status.as_u16(),
            });
        }

        let job_id = response_body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Provider("Missing job ID in response".to_string()))?
            .to_string();

        debug!("Fine-tuning job created: {}", job_id);
        Ok(TrainingJobId(job_id))
    }

    async fn get_job_status(
        &self,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        let url = format!("{}/{}", self.finetune_url(), job_id.0);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let status = response.status();
        let body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            if status.as_u16() == 404 {
                return Err(TrainingError::JobNotFound(job_id.0.clone()));
            }
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error")
                .to_string();
            return Err(TrainingError::Api {
                message,
                status_code: status.as_u16(),
            });
        }

        let status_str = body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        Ok(Self::parse_job_status(status_str, &body))
    }

    async fn cancel_job(&self, job_id: &TrainingJobId) -> Result<(), TrainingError> {
        let url = format!("{}/{}/cancel", self.finetune_url(), job_id.0);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body: serde_json::Value = response.json().await?;
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Failed to cancel job")
                .to_string();
            return Err(TrainingError::Api {
                message,
                status_code: status.as_u16(),
            });
        }

        debug!("Job {} cancelled", job_id);
        Ok(())
    }

    async fn list_jobs(&self) -> Result<Vec<TrainingJobSummary>, TrainingError> {
        let response = self
            .client
            .get(self.finetune_url())
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let status = response.status();
        let body: serde_json::Value = response.json().await?;

        if !status.is_success() {
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Failed to list jobs")
                .to_string();
            return Err(TrainingError::Api {
                message,
                status_code: status.as_u16(),
            });
        }

        let data = body
            .get("data")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
            .clone();

        let summaries: Vec<TrainingJobSummary> = data
            .iter()
            .filter_map(|job| {
                let job_id = job.get("id")?.as_str()?.to_string();
                let model = job.get("model")?.as_str()?.to_string();
                let status_str = job.get("status")?.as_str()?;
                let created_at_ts = job.get("created_at")?.as_i64()?;
                let created_at = chrono::DateTime::from_timestamp(created_at_ts, 0)?;

                Some(TrainingJobSummary {
                    job_id: TrainingJobId(job_id),
                    provider: "openai".to_string(),
                    base_model: model,
                    status: Self::parse_job_status(status_str, job),
                    created_at,
                    metrics: None,
                })
            })
            .collect();

        Ok(summaries)
    }

    async fn delete_model(&self, model_id: &str) -> Result<(), TrainingError> {
        let url = format!("{}/models/{}", self.base_url, model_id);

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body: serde_json::Value = response.json().await?;
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Failed to delete model")
                .to_string();
            return Err(TrainingError::Api {
                message,
                status_code: status.as_u16(),
            });
        }

        warn!("Deleted fine-tuned model: {}", model_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_job_status() {
        let response = json!({"status": "running", "trained_tokens": 1000});
        let status = OpenAiFineTune::parse_job_status("running", &response);
        assert!(matches!(status, TrainingJobStatus::Running { .. }));

        let response = json!({"status": "succeeded", "fine_tuned_model": "ft:gpt-4o-mini:abc"});
        let status = OpenAiFineTune::parse_job_status("succeeded", &response);
        assert!(
            matches!(status, TrainingJobStatus::Succeeded { model_id } if model_id == "ft:gpt-4o-mini:abc")
        );
    }

    #[test]
    fn test_supported_models() {
        let provider = OpenAiFineTune::new("test-key");
        let models = provider.supported_base_models();
        assert!(models.iter().any(|m| m.contains("gpt-4o")));
    }
}
