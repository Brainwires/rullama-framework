use std::collections::HashMap;
use tracing::info;

use crate::error::TrainingError;
use crate::types::{TrainingJobId, TrainingJobStatus, TrainingJobSummary};

#[cfg(feature = "cloud")]
use crate::cloud::{CloudFineTuneConfig, FineTuneProvider, JobPoller};

// Local training backend lives in the sibling `rullama` workspace
// (`rullama-finetune`). Consumers wire that crate's `TrainingBackend` impl
// into their own orchestrator (this manager covers cloud only).

/// High-level training orchestrator.
///
/// Provides a unified API for cloud fine-tuning across providers. Local
/// training backends (LoRA / QLoRA / DoRA) live in `rullama-finetune` (sibling
/// workspace) — drive them from your own loop.
pub struct TrainingManager {
    #[cfg(feature = "cloud")]
    cloud_providers: HashMap<String, Box<dyn FineTuneProvider>>,
}

impl TrainingManager {
    /// Create a new training manager.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cloud")]
            cloud_providers: HashMap::new(),
        }
    }

    /// Register a cloud fine-tuning provider.
    #[cfg(feature = "cloud")]
    pub fn add_cloud_provider(&mut self, provider: Box<dyn FineTuneProvider>) {
        let name = provider.name().to_string();
        info!("Registered cloud fine-tune provider: {}", name);
        self.cloud_providers.insert(name, provider);
    }

    /// List registered cloud providers.
    #[cfg(feature = "cloud")]
    pub fn cloud_providers(&self) -> Vec<&str> {
        self.cloud_providers.keys().map(|s| s.as_str()).collect()
    }

    /// Get a cloud provider by name.
    #[cfg(feature = "cloud")]
    pub fn get_cloud_provider(&self, name: &str) -> Option<&dyn FineTuneProvider> {
        self.cloud_providers.get(name).map(|p| p.as_ref())
    }

    /// Start a cloud fine-tuning job.
    #[cfg(feature = "cloud")]
    pub async fn start_cloud_job(
        &self,
        provider_name: &str,
        config: CloudFineTuneConfig,
    ) -> Result<TrainingJobId, TrainingError> {
        let provider = self.cloud_providers.get(provider_name).ok_or_else(|| {
            TrainingError::Provider(format!(
                "Unknown provider: {}. Available: {:?}",
                provider_name,
                self.cloud_providers.keys().collect::<Vec<_>>()
            ))
        })?;

        info!(
            "Starting cloud fine-tuning job on {} with model {}",
            provider_name, config.base_model
        );

        provider.create_job(config).await
    }

    /// Poll a cloud job until completion.
    #[cfg(feature = "cloud")]
    pub async fn wait_for_cloud_job(
        &self,
        provider_name: &str,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        let provider = self.cloud_providers.get(provider_name).ok_or_else(|| {
            TrainingError::Provider(format!("Unknown provider: {}", provider_name))
        })?;

        let poller = JobPoller::default();
        poller.poll_with_logging(provider.as_ref(), job_id).await
    }

    /// Check status of a cloud job.
    #[cfg(feature = "cloud")]
    pub async fn check_cloud_job(
        &self,
        provider_name: &str,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        let provider = self.cloud_providers.get(provider_name).ok_or_else(|| {
            TrainingError::Provider(format!("Unknown provider: {}", provider_name))
        })?;

        provider.get_job_status(job_id).await
    }

    /// Cancel a cloud job.
    #[cfg(feature = "cloud")]
    pub async fn cancel_cloud_job(
        &self,
        provider_name: &str,
        job_id: &TrainingJobId,
    ) -> Result<(), TrainingError> {
        let provider = self.cloud_providers.get(provider_name).ok_or_else(|| {
            TrainingError::Provider(format!("Unknown provider: {}", provider_name))
        })?;

        provider.cancel_job(job_id).await
    }

    /// List all jobs across all cloud providers.
    #[cfg(feature = "cloud")]
    pub async fn list_all_cloud_jobs(&self) -> Result<Vec<TrainingJobSummary>, TrainingError> {
        let mut all_jobs = Vec::new();
        for provider in self.cloud_providers.values() {
            match provider.list_jobs().await {
                Ok(jobs) => all_jobs.extend(jobs),
                Err(e) => {
                    tracing::warn!("Failed to list jobs from {}: {}", provider.name(), e);
                }
            }
        }
        Ok(all_jobs)
    }

    // train_local removed — local fine-tuning lives in the sibling
    // `rullama` workspace as `rullama-finetune`. Drive it directly
    // from that crate's `TrainingBackend` impl.
}

impl Default for TrainingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_manager_creation() {
        let manager = TrainingManager::new();

        #[cfg(feature = "cloud")]
        assert!(manager.cloud_providers().is_empty());
    }
}
