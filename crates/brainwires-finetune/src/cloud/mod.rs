/// Anyscale fine-tuning provider.
pub mod anyscale;
/// AWS Bedrock fine-tuning provider.
pub mod bedrock;
/// Cost estimation utilities.
pub mod cost;
/// Fireworks AI fine-tuning provider.
pub mod fireworks;
/// OpenAI fine-tuning provider.
pub mod openai;
/// Job polling utilities.
pub mod polling;
/// Together AI fine-tuning provider.
pub mod together;
/// Google Vertex AI fine-tuning provider.
pub mod vertex;

use crate::datasets::DataFormat;
use async_trait::async_trait;

use crate::config::{AlignmentMethod, LoraConfig, TrainingHyperparams};
use crate::error::TrainingError;
use crate::types::{DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary};

/// Configuration for a cloud fine-tuning job.
#[derive(Debug, Clone)]
pub struct CloudFineTuneConfig {
    /// Base model to fine-tune (provider-specific ID).
    pub base_model: String,
    /// Uploaded training dataset ID.
    pub training_dataset: DatasetId,
    /// Optional validation dataset ID.
    pub validation_dataset: Option<DatasetId>,
    /// Training hyperparameters.
    pub hyperparams: TrainingHyperparams,
    /// LoRA config (if provider supports PEFT).
    pub lora: Option<LoraConfig>,
    /// Alignment method (DPO/ORPO if provider supports it).
    pub alignment: AlignmentMethod,
    /// Suffix appended to fine-tuned model name.
    pub suffix: Option<String>,
}

impl CloudFineTuneConfig {
    /// Create a new cloud fine-tune config with default hyperparameters.
    pub fn new(base_model: impl Into<String>, training_dataset: DatasetId) -> Self {
        Self {
            base_model: base_model.into(),
            training_dataset,
            validation_dataset: None,
            hyperparams: TrainingHyperparams::default(),
            lora: None,
            alignment: AlignmentMethod::None,
            suffix: None,
        }
    }

    /// Set the validation dataset.
    pub fn with_validation(mut self, dataset: DatasetId) -> Self {
        self.validation_dataset = Some(dataset);
        self
    }

    /// Set training hyperparameters.
    pub fn with_hyperparams(mut self, h: TrainingHyperparams) -> Self {
        self.hyperparams = h;
        self
    }

    /// Set LoRA configuration.
    pub fn with_lora(mut self, lora: LoraConfig) -> Self {
        self.lora = Some(lora);
        self
    }

    /// Set alignment method.
    pub fn with_alignment(mut self, alignment: AlignmentMethod) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set model name suffix.
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }
}

/// Trait for cloud fine-tuning providers.
#[async_trait]
pub trait FineTuneProvider: Send + Sync {
    /// Provider name.
    fn name(&self) -> &str;

    /// List base models available for fine-tuning.
    fn supported_base_models(&self) -> Vec<String>;

    /// Whether this provider supports DPO/preference optimization.
    fn supports_dpo(&self) -> bool;

    /// Upload a dataset (JSONL bytes) and get a dataset ID.
    async fn upload_dataset(
        &self,
        data: &[u8],
        format: DataFormat,
    ) -> Result<DatasetId, TrainingError>;

    /// Create a fine-tuning job.
    async fn create_job(&self, config: CloudFineTuneConfig)
    -> Result<TrainingJobId, TrainingError>;

    /// Get the current status of a training job.
    async fn get_job_status(
        &self,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError>;

    /// Cancel a running training job.
    async fn cancel_job(&self, job_id: &TrainingJobId) -> Result<(), TrainingError>;

    /// List all training jobs.
    async fn list_jobs(&self) -> Result<Vec<TrainingJobSummary>, TrainingError>;

    /// Delete a fine-tuned model.
    async fn delete_model(&self, model_id: &str) -> Result<(), TrainingError>;
}

/// Factory for creating cloud fine-tune providers.
pub struct FineTuneProviderFactory;

impl FineTuneProviderFactory {
    /// Create an OpenAI fine-tune provider.
    pub fn openai(api_key: impl Into<String>) -> openai::OpenAiFineTune {
        openai::OpenAiFineTune::new(api_key)
    }

    /// Create a Together AI fine-tune provider.
    pub fn together(api_key: impl Into<String>) -> together::TogetherFineTune {
        together::TogetherFineTune::new(api_key)
    }

    /// Create a Fireworks AI fine-tune provider.
    pub fn fireworks(api_key: impl Into<String>) -> fireworks::FireworksFineTune {
        fireworks::FireworksFineTune::new(api_key)
    }

    /// Create an Anyscale fine-tune provider.
    pub fn anyscale(api_key: impl Into<String>) -> anyscale::AnyscaleFineTune {
        anyscale::AnyscaleFineTune::new(api_key)
    }

    /// Create an AWS Bedrock fine-tune provider.
    pub fn bedrock(region: impl Into<String>) -> bedrock::BedrockFineTune {
        bedrock::BedrockFineTune::new(region)
    }

    /// Create a Google Vertex AI fine-tune provider.
    pub fn vertex(
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> vertex::VertexFineTune {
        vertex::VertexFineTune::new(project_id, location)
    }
}

pub use self::anyscale::AnyscaleFineTune;
pub use self::bedrock::BedrockFineTune;
pub use self::cost::CostEstimator;
pub use self::fireworks::FireworksFineTune;
pub use self::openai::OpenAiFineTune;
pub use self::polling::JobPoller;
pub use self::together::TogetherFineTune;
pub use self::vertex::VertexFineTune;
