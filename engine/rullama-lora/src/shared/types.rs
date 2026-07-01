use serde::{Deserialize, Serialize};

/// Progress information emitted during a training run.
///
/// Pre-rewrite, this type also carried cloud-job lifecycle fields
/// (`TrainingJobId`, `TrainingJobStatus`, etc.) inherited from
/// `rullama-finetune`'s cloud API. Native LoRA training has no jobs and no
/// remote scheduler; those types were dropped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingProgress {
    /// Current epoch (1-based once started).
    pub epoch: u32,
    /// Total number of epochs.
    pub total_epochs: u32,
    /// Current training step.
    pub step: u64,
    /// Total training steps.
    pub total_steps: u64,
    /// Training loss for the current step.
    pub train_loss: Option<f64>,
    /// Evaluation loss (if the trainer ran an eval pass at this step).
    pub eval_loss: Option<f64>,
    /// Current learning rate (post-scheduler).
    pub learning_rate: Option<f64>,
    /// Elapsed wall-clock time in seconds.
    pub elapsed_secs: u64,
}

impl TrainingProgress {
    /// Fraction of training completed, in `[0.0, 1.0]`.
    pub fn completion_fraction(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.step as f64 / self.total_steps as f64).clamp(0.0, 1.0)
    }
}

/// Metrics from a completed training run.
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_progress_completion_clamped() {
        let p = TrainingProgress {
            step: 110,
            total_steps: 100,
            ..Default::default()
        };
        assert_eq!(p.completion_fraction(), 1.0);
    }
}
