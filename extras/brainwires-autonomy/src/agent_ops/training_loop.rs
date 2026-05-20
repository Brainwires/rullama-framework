//! Autonomous training loop — orchestrates model training cycles.

use serde::{Deserialize, Serialize};

/// Configuration for the autonomous training loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLoopConfig {
    /// Maximum training rounds.
    pub max_rounds: u32,
    /// Maximum cost per round in USD.
    pub max_cost_per_round: f64,
    /// Minimum improvement threshold to continue training.
    pub min_improvement: f64,
    /// Dataset size limit per round.
    pub max_dataset_size: usize,
    /// Whether to auto-evaluate after each round.
    pub auto_evaluate: bool,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            max_rounds: 5,
            max_cost_per_round: 10.0,
            min_improvement: 0.01,
            max_dataset_size: 10_000,
            auto_evaluate: true,
        }
    }
}

/// Result of a single training round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRoundResult {
    /// Round number (1-based).
    pub round: u32,
    /// Number of examples in the training dataset.
    pub dataset_size: usize,
    /// Final training loss for this round.
    pub training_loss: f64,
    /// Validation loss, if a validation set was used.
    pub validation_loss: Option<f64>,
    /// Evaluation score from the eval suite, if available.
    pub eval_score: Option<f64>,
    /// Cost of this round in USD.
    pub cost: f64,
    /// Duration of this round in seconds.
    pub duration_secs: f64,
}

/// Report for a complete training loop execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLoopReport {
    /// Results from each training round.
    pub rounds: Vec<TrainingRoundResult>,
    /// Total duration of the training loop in seconds.
    pub total_duration_secs: f64,
    /// Total cost across all rounds in USD.
    pub total_cost: f64,
    /// Whether training converged before reaching the max round limit.
    pub converged: bool,
    /// Final evaluation score from the last round, if available.
    pub final_eval_score: Option<f64>,
}

/// Orchestrates autonomous training cycles with evaluation checkpoints.
///
/// The actual training and evaluation implementations are provided by
/// `brainwires-finetune` and `brainwires-eval` respectively. This struct
/// manages the loop logic, convergence detection, and reporting.
pub struct AutonomousTrainingLoop {
    config: TrainingLoopConfig,
}

impl AutonomousTrainingLoop {
    /// Create a new autonomous training loop with the given configuration.
    pub fn new(config: TrainingLoopConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TrainingLoopConfig {
        &self.config
    }

    /// Check if training should continue based on improvement between rounds.
    pub fn should_continue(
        &self,
        current_round: u32,
        prev_score: Option<f64>,
        current_score: Option<f64>,
    ) -> bool {
        if current_round >= self.config.max_rounds {
            return false;
        }

        match (prev_score, current_score) {
            (Some(prev), Some(curr)) => {
                let improvement = curr - prev;
                improvement >= self.config.min_improvement
            }
            _ => true, // Continue if we don't have scores yet
        }
    }

    /// Generate a report from collected round results.
    pub fn generate_report(
        &self,
        rounds: Vec<TrainingRoundResult>,
        total_duration_secs: f64,
        converged: bool,
    ) -> TrainingLoopReport {
        let total_cost: f64 = rounds.iter().map(|r| r.cost).sum();
        let final_eval_score = rounds.last().and_then(|r| r.eval_score);

        TrainingLoopReport {
            rounds,
            total_duration_secs,
            total_cost,
            converged,
            final_eval_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_continue_false_when_max_rounds_reached() {
        let tl = AutonomousTrainingLoop::new(TrainingLoopConfig {
            max_rounds: 3,
            ..Default::default()
        });
        assert!(!tl.should_continue(3, Some(0.8), Some(0.9)));
    }

    #[test]
    fn should_continue_true_when_no_scores() {
        let tl = AutonomousTrainingLoop::new(TrainingLoopConfig::default());
        assert!(tl.should_continue(1, None, None));
        assert!(tl.should_continue(1, Some(0.5), None));
    }

    #[test]
    fn should_continue_false_when_improvement_below_threshold() {
        let tl = AutonomousTrainingLoop::new(TrainingLoopConfig {
            min_improvement: 0.05,
            max_rounds: 10,
            ..Default::default()
        });
        // Improvement of 0.01 < 0.05 threshold
        assert!(!tl.should_continue(2, Some(0.80), Some(0.81)));
        // Improvement of 0.10 >= 0.05 threshold
        assert!(tl.should_continue(2, Some(0.80), Some(0.90)));
    }

    #[test]
    fn generate_report_computes_totals() {
        let tl = AutonomousTrainingLoop::new(TrainingLoopConfig::default());
        let rounds = vec![
            TrainingRoundResult {
                round: 1,
                dataset_size: 100,
                training_loss: 0.5,
                validation_loss: Some(0.6),
                eval_score: Some(0.7),
                cost: 1.0,
                duration_secs: 10.0,
            },
            TrainingRoundResult {
                round: 2,
                dataset_size: 200,
                training_loss: 0.3,
                validation_loss: Some(0.4),
                eval_score: Some(0.85),
                cost: 2.0,
                duration_secs: 20.0,
            },
        ];

        let report = tl.generate_report(rounds, 30.0, true);
        assert!((report.total_cost - 3.0).abs() < f64::EPSILON);
        assert!((report.total_duration_secs - 30.0).abs() < f64::EPSILON);
        assert!(report.converged);
        assert_eq!(report.final_eval_score, Some(0.85));
        assert_eq!(report.rounds.len(), 2);
    }
}
