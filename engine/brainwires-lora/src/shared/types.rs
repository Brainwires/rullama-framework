use serde::{Deserialize, Serialize};

/// Unique identifier for a training job.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrainingJobId(pub String);

impl std::fmt::Display for TrainingJobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<S: Into<String>> From<S> for TrainingJobId {
    fn from(s: S) -> Self {
        Self(s.into())
    }
}

/// Unique identifier for an uploaded dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatasetId(pub String);

impl std::fmt::Display for DatasetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<S: Into<String>> From<S> for DatasetId {
    fn from(s: S) -> Self {
        Self(s.into())
    }
}

impl DatasetId {
    /// Create a DatasetId from an S3 URI (for Bedrock).
    pub fn from_s3_uri(uri: &str) -> Self {
        Self(uri.to_string())
    }

    /// Create a DatasetId from a GCS URI (for Vertex AI).
    pub fn from_gcs_uri(uri: &str) -> Self {
        Self(uri.to_string())
    }
}

/// Status of a training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TrainingJobStatus {
    /// Job is pending.
    Pending,
    /// Job is validating inputs.
    Validating,
    /// Job is queued for execution.
    Queued,
    /// Job is actively running.
    Running {
        /// Current training progress.
        progress: TrainingProgress,
    },
    /// Job completed successfully.
    Succeeded {
        /// ID of the fine-tuned model.
        model_id: String,
    },
    /// Job failed with an error.
    Failed {
        /// Error description.
        error: String,
    },
    /// Job was cancelled.
    Cancelled,
}

impl TrainingJobStatus {
    /// Whether the job has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Succeeded { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }

    /// Whether the job is currently running.
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }

    /// Whether the job succeeded.
    pub fn is_succeeded(&self) -> bool {
        matches!(self, Self::Succeeded { .. })
    }
}

/// Progress information for a running training job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingProgress {
    /// Current epoch.
    pub epoch: u32,
    /// Total number of epochs.
    pub total_epochs: u32,
    /// Current training step.
    pub step: u64,
    /// Total training steps.
    pub total_steps: u64,
    /// Training loss.
    pub train_loss: Option<f64>,
    /// Evaluation loss.
    pub eval_loss: Option<f64>,
    /// Current learning rate.
    pub learning_rate: Option<f64>,
    /// Elapsed time in seconds.
    pub elapsed_secs: u64,
}

impl TrainingProgress {
    /// Fraction of training completed (0.0-1.0).
    pub fn completion_fraction(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.step as f64 / self.total_steps as f64
    }
}

/// Metrics from a completed training job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Final training loss.
    pub final_train_loss: Option<f64>,
    /// Final evaluation loss.
    pub final_eval_loss: Option<f64>,
    /// Total training steps completed.
    pub total_steps: u64,
    /// Total epochs completed.
    pub total_epochs: u32,
    /// Total tokens processed.
    pub total_tokens_trained: Option<u64>,
    /// Total training duration in seconds.
    pub duration_secs: u64,
    /// Estimated cost in USD.
    pub estimated_cost_usd: Option<f64>,
}

/// Summary of a training job for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobSummary {
    /// Job identifier.
    pub job_id: TrainingJobId,
    /// Provider name.
    pub provider: String,
    /// Base model being fine-tuned.
    pub base_model: String,
    /// Current job status.
    pub status: TrainingJobStatus,
    /// Job creation time.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Training metrics (if available).
    pub metrics: Option<TrainingMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status_terminal() {
        assert!(!TrainingJobStatus::Pending.is_terminal());
        assert!(!TrainingJobStatus::Queued.is_terminal());
        assert!(
            TrainingJobStatus::Succeeded {
                model_id: "m".into()
            }
            .is_terminal()
        );
        assert!(
            TrainingJobStatus::Failed {
                error: "err".into()
            }
            .is_terminal()
        );
        assert!(TrainingJobStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_progress_completion() {
        let p = TrainingProgress {
            step: 50,
            total_steps: 100,
            ..Default::default()
        };
        assert!((p.completion_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_id_from_string() {
        let id: TrainingJobId = "ft-abc123".into();
        assert_eq!(id.0, "ft-abc123");
        assert_eq!(id.to_string(), "ft-abc123");
    }
}
