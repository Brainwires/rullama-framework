use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::{debug, info};

use crate::datasets::DataFormat;

use super::{CloudFineTuneConfig, FineTuneProvider};
use crate::error::TrainingError;
use crate::types::{
    DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary, TrainingProgress,
};

/// AWS Bedrock fine-tuning provider.
///
/// Supports Claude Haiku fine-tuning and other Bedrock foundation models.
/// Requires AWS credentials (access key + secret key).
///
/// **Note**: Bedrock requires training data in S3. Use `DatasetId::from_s3_uri()`
/// to pass S3 URIs directly rather than uploading through this API.
pub struct BedrockFineTune {
    region: String,
    client: Client,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
}

impl BedrockFineTune {
    /// Create a new AWS Bedrock fine-tune provider.
    pub fn new(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            client: Client::new(),
            access_key_id: None,
            secret_access_key: None,
        }
    }

    /// Set explicit AWS credentials.
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    fn base_url(&self) -> String {
        format!("https://bedrock.{}.amazonaws.com", self.region)
    }

    fn get_credentials(&self) -> Result<(&str, &str), TrainingError> {
        let access_key = self.access_key_id.as_deref().ok_or_else(|| {
            TrainingError::Config(
                "AWS access key ID not set. Use with_credentials() or set AWS_ACCESS_KEY_ID"
                    .to_string(),
            )
        })?;
        let secret_key = self.secret_access_key.as_deref().ok_or_else(|| {
            TrainingError::Config("AWS secret access key not set. Use with_credentials() or set AWS_SECRET_ACCESS_KEY".to_string())
        })?;
        Ok((access_key, secret_key))
    }

    #[cfg(feature = "bedrock")]
    fn sign_request(
        &self,
        method: &str,
        uri: &str,
        body: &[u8],
        access_key: &str,
        secret_key: &str,
    ) -> Result<Vec<(String, String)>, TrainingError> {
        use aws_credential_types::Credentials;
        use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
        use aws_sigv4::sign::v4;
        use std::time::SystemTime;

        let credentials =
            Credentials::new(access_key, secret_key, None, None, "brainwires-training");
        let identity = credentials.into();

        let mut settings = SigningSettings::default();
        settings.signature_location = aws_sigv4::http_request::SignatureLocation::Headers;

        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| TrainingError::Config(format!("SigV4 params error: {}", e)))?;

        let signable_request = SignableRequest::new(
            method,
            uri,
            std::iter::once(("content-type", "application/json")),
            SignableBody::Bytes(body),
        )
        .map_err(|e| TrainingError::Config(format!("Signable request error: {}", e)))?;

        let (signing_instructions, _signature) = sign(signable_request, &signing_params.into())
            .map_err(|e| TrainingError::Config(format!("SigV4 signing error: {}", e)))?
            .into_parts();

        // Build a dummy HTTP request, apply signing headers, then extract them
        let mut http_request = http::Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(())
            .map_err(|e| TrainingError::Config(format!("HTTP request build error: {}", e)))?;

        signing_instructions.apply_to_request_http1x(&mut http_request);

        let headers: Vec<(String, String)> = http_request
            .headers()
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap_or("").to_string()))
            .collect();

        Ok(headers)
    }

    #[cfg(not(feature = "bedrock"))]
    fn sign_request(
        &self,
        _method: &str,
        _uri: &str,
        _body: &[u8],
        _access_key: &str,
        _secret_key: &str,
    ) -> Result<Vec<(String, String)>, TrainingError> {
        Err(TrainingError::Config(
            "Bedrock feature not enabled. Build with --features bedrock".to_string(),
        ))
    }

    async fn signed_request(
        &self,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, TrainingError> {
        let (access_key, secret_key) = self.get_credentials()?;
        let url = format!("{}{}", self.base_url(), path);
        let body_bytes = body
            .as_ref()
            .map(|b| serde_json::to_vec(b).unwrap_or_default())
            .unwrap_or_default();

        let headers = self.sign_request(method, &url, &body_bytes, access_key, secret_key)?;

        let mut request = match method {
            "POST" => self.client.post(&url).body(body_bytes),
            "GET" => self.client.get(&url),
            "DELETE" => self.client.delete(&url),
            _ => {
                return Err(TrainingError::Config(format!(
                    "Unsupported method: {}",
                    method
                )));
            }
        };

        request = request.header("content-type", "application/json");
        for (name, value) in headers {
            request = request.header(&name, &value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| TrainingError::Provider(format!("Bedrock request failed: {}", e)))?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            TrainingError::Provider(format!("Failed to read Bedrock response: {}", e))
        })?;

        if !status.is_success() {
            return Err(TrainingError::Api {
                message: format!("Bedrock API error: {}", text),
                status_code: status.as_u16(),
            });
        }

        serde_json::from_str(&text).map_err(|e| {
            TrainingError::Provider(format!("Failed to parse Bedrock response: {}", e))
        })
    }
}

#[async_trait]
impl FineTuneProvider for BedrockFineTune {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn supported_base_models(&self) -> Vec<String> {
        vec![
            "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            "meta.llama3-1-8b-instruct-v1:0".to_string(),
            "amazon.titan-text-lite-v1".to_string(),
        ]
    }

    fn supports_dpo(&self) -> bool {
        false
    }

    async fn upload_dataset(
        &self,
        data: &[u8],
        _format: DataFormat,
    ) -> Result<DatasetId, TrainingError> {
        debug!(
            "Bedrock fine-tuning requires data in S3. Dataset size: {} bytes",
            data.len()
        );
        Err(TrainingError::Config(
            "Bedrock requires dataset in S3. Use DatasetId::from_s3_uri() and pass directly to create_job".to_string()
        ))
    }

    async fn create_job(
        &self,
        config: CloudFineTuneConfig,
    ) -> Result<TrainingJobId, TrainingError> {
        info!(
            "Creating Bedrock fine-tuning job for: {}",
            config.base_model
        );

        let mut body = json!({
            "baseModelIdentifier": config.base_model,
            "customModelName": config.suffix.as_deref().unwrap_or("brainwires-ft"),
            "roleArn": "arn:aws:iam::role/bedrock-training",
            "trainingDataConfig": {
                "s3Uri": config.training_dataset.0
            },
            "hyperParameters": {
                "epochCount": config.hyperparams.epochs.to_string(),
                "batchSize": config.hyperparams.batch_size.to_string(),
                "learningRate": config.hyperparams.learning_rate.to_string(),
            }
        });

        if let Some(ref val_dataset) = config.validation_dataset {
            body["validationDataConfig"] = json!({ "s3Uri": val_dataset.0 });
        }

        let response = self
            .signed_request("POST", "/model-customization-jobs", Some(body))
            .await?;

        let job_arn = response
            .get("jobArn")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainingError::Provider("Missing jobArn in response".to_string()))?;

        info!("Created Bedrock job: {}", job_arn);
        Ok(TrainingJobId(job_arn.to_string()))
    }

    async fn get_job_status(
        &self,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        debug!("Checking Bedrock job status: {}", job_id);

        let path = format!("/model-customization-jobs/{}", job_id.0);
        let response = self.signed_request("GET", &path, None).await?;

        let status = response
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        match status {
            "InProgress" | "Training" => Ok(TrainingJobStatus::Running {
                progress: TrainingProgress::default(),
            }),
            "Completed" => {
                let model_id = response
                    .get("outputModelArn")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(TrainingJobStatus::Succeeded { model_id })
            }
            "Failed" => {
                let error = response
                    .get("failureMessage")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error")
                    .to_string();
                Ok(TrainingJobStatus::Failed { error })
            }
            "Stopping" | "Stopped" => Ok(TrainingJobStatus::Cancelled),
            "Validating" => Ok(TrainingJobStatus::Validating),
            _ => Ok(TrainingJobStatus::Pending),
        }
    }

    async fn cancel_job(&self, job_id: &TrainingJobId) -> Result<(), TrainingError> {
        info!("Cancelling Bedrock job: {}", job_id);
        let path = format!("/model-customization-jobs/{}/stop", job_id.0);
        self.signed_request("POST", &path, Some(json!({}))).await?;
        Ok(())
    }

    async fn list_jobs(&self) -> Result<Vec<TrainingJobSummary>, TrainingError> {
        let response = self
            .signed_request("GET", "/model-customization-jobs", None)
            .await?;

        let jobs = response
            .get("modelCustomizationJobSummaries")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|job| {
                let job_id = job.get("jobArn")?.as_str()?;
                let model = job.get("baseModelIdentifier")?.as_str()?;
                let status_str = job.get("status")?.as_str()?;
                let status = match status_str {
                    "Completed" => TrainingJobStatus::Succeeded {
                        model_id: job
                            .get("outputModelArn")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    },
                    "Failed" => TrainingJobStatus::Failed {
                        error: job
                            .get("failureMessage")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    },
                    "InProgress" | "Training" => TrainingJobStatus::Running {
                        progress: TrainingProgress::default(),
                    },
                    "Stopping" | "Stopped" => TrainingJobStatus::Cancelled,
                    _ => TrainingJobStatus::Pending,
                };

                Some(TrainingJobSummary {
                    job_id: TrainingJobId(job_id.to_string()),
                    provider: "bedrock".to_string(),
                    base_model: model.to_string(),
                    status,
                    created_at: chrono::Utc::now(),
                    metrics: None,
                })
            })
            .collect();

        Ok(jobs)
    }

    async fn delete_model(&self, model_id: &str) -> Result<(), TrainingError> {
        info!("Deleting Bedrock custom model: {}", model_id);
        let path = format!("/custom-models/{}", model_id);
        self.signed_request("DELETE", &path, None).await?;
        Ok(())
    }
}
