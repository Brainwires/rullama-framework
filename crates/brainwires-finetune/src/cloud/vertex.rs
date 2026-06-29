use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::{debug, info};

use brainwires_datasets::DataFormat;

use super::{CloudFineTuneConfig, FineTuneProvider};
use crate::error::TrainingError;
use crate::types::{
    DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary, TrainingProgress,
};

/// Google Vertex AI fine-tuning provider.
///
/// Supports Gemini model tuning.
/// Requires GCP service account credentials or an explicit access token.
///
/// **Note**: Vertex AI requires training data in GCS. Use `DatasetId::from_gcs_uri()`
/// to pass GCS URIs directly rather than uploading through this API.
pub struct VertexFineTune {
    project_id: String,
    location: String,
    client: Client,
    access_token: Option<String>,
    #[cfg(feature = "vertex")]
    token_provider: Option<std::sync::Arc<dyn gcp_auth::TokenProvider>>,
}

impl VertexFineTune {
    /// Create a new Google Vertex AI fine-tune provider.
    pub fn new(project_id: impl Into<String>, location: impl Into<String>) -> Self {
        Self {
            project_id: project_id.into(),
            location: location.into(),
            client: Client::new(),
            access_token: None,
            #[cfg(feature = "vertex")]
            token_provider: None,
        }
    }

    /// Set an explicit access token.
    pub fn with_access_token(mut self, token: impl Into<String>) -> Self {
        self.access_token = Some(token.into());
        self
    }

    /// Set up authentication from a GCP service account JSON file.
    #[cfg(feature = "vertex")]
    pub async fn from_service_account(
        project_id: impl Into<String>,
        location: impl Into<String>,
        service_account_path: &std::path::Path,
    ) -> Result<Self, TrainingError> {
        let sa_json = std::fs::read_to_string(service_account_path).map_err(|e| {
            TrainingError::Config(format!("Failed to read service account file: {}", e))
        })?;
        let credentials = gcp_auth::CustomServiceAccount::from_json(&sa_json)
            .map_err(|e| TrainingError::Config(format!("Invalid service account: {}", e)))?;

        Ok(Self {
            project_id: project_id.into(),
            location: location.into(),
            client: Client::new(),
            access_token: None,
            token_provider: Some(std::sync::Arc::new(credentials)),
        })
    }

    fn base_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}",
            self.location, self.project_id, self.location
        )
    }

    async fn get_token(&self) -> Result<String, TrainingError> {
        if let Some(ref token) = self.access_token {
            return Ok(token.clone());
        }

        #[cfg(feature = "vertex")]
        if let Some(ref provider) = self.token_provider {
            let token = provider
                .token(&["https://www.googleapis.com/auth/cloud-platform"])
                .await
                .map_err(|e| TrainingError::Config(format!("GCP token error: {}", e)))?;
            return Ok(token.as_str().to_string());
        }

        Err(TrainingError::Config(
            "No Vertex AI credentials configured. Use with_access_token() or from_service_account()".to_string()
        ))
    }

    async fn api_request(
        &self,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, TrainingError> {
        let token = self.get_token().await?;
        let url = format!("{}{}", self.base_url(), path);

        let mut request = match method {
            "POST" => {
                let mut r = self.client.post(&url);
                if let Some(b) = body {
                    r = r.json(&b);
                }
                r
            }
            "GET" => self.client.get(&url),
            "DELETE" => self.client.delete(&url),
            _ => {
                return Err(TrainingError::Config(format!(
                    "Unsupported method: {}",
                    method
                )));
            }
        };

        request = request
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json");

        let response = request
            .send()
            .await
            .map_err(|e| TrainingError::Provider(format!("Vertex AI request failed: {}", e)))?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            TrainingError::Provider(format!("Failed to read Vertex AI response: {}", e))
        })?;

        if !status.is_success() {
            return Err(TrainingError::Api {
                message: format!("Vertex AI API error: {}", text),
                status_code: status.as_u16(),
            });
        }

        if text.is_empty() {
            return Ok(json!({}));
        }

        serde_json::from_str(&text).map_err(|e| {
            TrainingError::Provider(format!("Failed to parse Vertex AI response: {}", e))
        })
    }
}

#[async_trait]
impl FineTuneProvider for VertexFineTune {
    fn name(&self) -> &str {
        "vertex"
    }

    fn supported_base_models(&self) -> Vec<String> {
        vec![
            "gemini-1.5-flash-002".to_string(),
            "gemini-1.5-pro-002".to_string(),
        ]
    }

    fn supports_dpo(&self) -> bool {
        false // Vertex uses RLHF, not DPO
    }

    async fn upload_dataset(
        &self,
        data: &[u8],
        _format: DataFormat,
    ) -> Result<DatasetId, TrainingError> {
        debug!(
            "Vertex AI fine-tuning requires data in GCS. Dataset size: {} bytes",
            data.len()
        );
        Err(TrainingError::Config(
            "Vertex AI requires dataset in GCS. Use DatasetId::from_gcs_uri() and pass directly to create_job".to_string()
        ))
    }

    async fn create_job(
        &self,
        config: CloudFineTuneConfig,
    ) -> Result<TrainingJobId, TrainingError> {
        info!("Creating Vertex AI tuning job for: {}", config.base_model);

        let body = json!({
            "baseModel": config.base_model,
            "supervisedTuningSpec": {
                "trainingDatasetUri": config.training_dataset.0,
                "validationDatasetUri": config.validation_dataset.as_ref().map(|d| d.0.as_str()),
                "hyperParameters": {
                    "epochCount": config.hyperparams.epochs,
                    "learningRateMultiplier": config.hyperparams.learning_rate / 0.001,
                }
            },
            "tunedModelDisplayName": config.suffix.as_deref().unwrap_or("brainwires-ft"),
        });

        let response = self.api_request("POST", "/tuningJobs", Some(body)).await?;

        let job_name = response
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Provider("Missing 'name' in response".to_string()))?;

        info!("Created Vertex AI job: {}", job_name);
        Ok(TrainingJobId(job_name.to_string()))
    }

    async fn get_job_status(
        &self,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        debug!("Checking Vertex AI job status: {}", job_id);

        // Job IDs are full resource names like projects/X/locations/Y/tuningJobs/Z
        // Extract the relative path
        let path = if job_id.0.starts_with("projects/") {
            // Already a full resource path, need to reconstruct URL
            format!(
                "/{}",
                job_id
                    .0
                    .rsplit("locations/")
                    .next()
                    .map(|s| format!("tuningJobs/{}", s.rsplit('/').next().unwrap_or("")))
                    .unwrap_or_default()
            )
        } else {
            format!("/tuningJobs/{}", job_id.0)
        };

        let response = self.api_request("GET", &path, None).await?;

        let state = response
            .get("state")
            .and_then(|v| v.as_str())
            .unwrap_or("STATE_UNSPECIFIED");

        match state {
            "JOB_STATE_SUCCEEDED" => {
                let model_id = response
                    .get("tunedModel")
                    .and_then(|v| v.get("model"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(TrainingJobStatus::Succeeded { model_id })
            }
            "JOB_STATE_FAILED" => {
                let error = response
                    .get("error")
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error")
                    .to_string();
                Ok(TrainingJobStatus::Failed { error })
            }
            "JOB_STATE_CANCELLED" => Ok(TrainingJobStatus::Cancelled),
            "JOB_STATE_RUNNING" => Ok(TrainingJobStatus::Running {
                progress: TrainingProgress::default(),
            }),
            "JOB_STATE_PENDING" | "JOB_STATE_QUEUED" => Ok(TrainingJobStatus::Pending),
            _ => Ok(TrainingJobStatus::Pending),
        }
    }

    async fn cancel_job(&self, job_id: &TrainingJobId) -> Result<(), TrainingError> {
        info!("Cancelling Vertex AI job: {}", job_id);
        let path = format!("/tuningJobs/{}:cancel", job_id.0);
        self.api_request("POST", &path, Some(json!({}))).await?;
        Ok(())
    }

    async fn list_jobs(&self) -> Result<Vec<TrainingJobSummary>, TrainingError> {
        let response = self.api_request("GET", "/tuningJobs", None).await?;

        let jobs = response
            .get("tuningJobs")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|job| {
                let name = job.get("name")?.as_str()?;
                let base_model = job.get("baseModel")?.as_str()?;
                let state = job.get("state")?.as_str()?;
                let status = match state {
                    "JOB_STATE_SUCCEEDED" => TrainingJobStatus::Succeeded {
                        model_id: job
                            .get("tunedModel")
                            .and_then(|v| v.get("model"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    },
                    "JOB_STATE_FAILED" => TrainingJobStatus::Failed {
                        error: job
                            .get("error")
                            .and_then(|v| v.get("message"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    },
                    "JOB_STATE_CANCELLED" => TrainingJobStatus::Cancelled,
                    "JOB_STATE_RUNNING" => TrainingJobStatus::Running {
                        progress: TrainingProgress::default(),
                    },
                    _ => TrainingJobStatus::Pending,
                };

                Some(TrainingJobSummary {
                    job_id: TrainingJobId(name.to_string()),
                    provider: "vertex".to_string(),
                    base_model: base_model.to_string(),
                    status,
                    created_at: chrono::Utc::now(),
                    metrics: None,
                })
            })
            .collect();

        Ok(jobs)
    }

    async fn delete_model(&self, model_id: &str) -> Result<(), TrainingError> {
        info!("Deleting Vertex AI model: {}", model_id);
        let path = format!("/models/{}", model_id);
        self.api_request("DELETE", &path, None).await?;
        Ok(())
    }
}
