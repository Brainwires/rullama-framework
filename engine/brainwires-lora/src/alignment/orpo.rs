/// ORPO (Odds Ratio Preference Optimization) loss.
///
/// ORPO combines SFT and preference alignment in a single pass,
/// eliminating the need for a separate reference model:
///
/// L_ORPO = L_SFT + λ * L_OR
///
/// Where L_OR is the odds ratio loss that contrasts chosen vs rejected:
///   L_OR = -log σ(log(odds(y_w|x) / odds(y_l|x)))
///   odds(y|x) = P(y|x) / (1 - P(y|x))
///
/// Advantages over DPO:
/// - No reference model needed (saves 50% VRAM)
/// - Single training pass (faster)
/// - Simpler implementation
#[derive(Debug, Clone)]
pub struct OrpoLoss {
    /// Weight of the odds ratio loss relative to SFT loss.
    pub lambda: f64,
}

impl Default for OrpoLoss {
    fn default() -> Self {
        Self { lambda: 0.5 }
    }
}

impl OrpoLoss {
    /// Create a new ORPO loss with the given alignment weight lambda.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Compute odds from probability.
    fn odds(prob: f64) -> f64 {
        let p = prob.clamp(1e-10, 1.0 - 1e-10);
        p / (1.0 - p)
    }

    /// Compute ORPO alignment loss component.
    ///
    /// - `chosen_prob`: Average token probability for chosen response
    /// - `rejected_prob`: Average token probability for rejected response
    pub fn compute_alignment_loss(&self, chosen_prob: f64, rejected_prob: f64) -> f64 {
        let chosen_odds = Self::odds(chosen_prob);
        let rejected_odds = Self::odds(rejected_prob);

        let log_odds_ratio = (chosen_odds / rejected_odds).ln();

        // -log σ(log_odds_ratio)
        -log_sigmoid(log_odds_ratio)
    }

    /// Compute full ORPO loss (SFT + alignment).
    ///
    /// - `sft_loss`: Standard cross-entropy loss on chosen response
    /// - `chosen_prob`: Average token probability for chosen response
    /// - `rejected_prob`: Average token probability for rejected response
    pub fn compute(&self, sft_loss: f64, chosen_prob: f64, rejected_prob: f64) -> f64 {
        let alignment_loss = self.compute_alignment_loss(chosen_prob, rejected_prob);
        sft_loss + self.lambda * alignment_loss
    }

    /// Compute average ORPO loss over a batch.
    pub fn compute_batch(
        &self,
        sft_losses: &[f64],
        chosen_probs: &[f64],
        rejected_probs: &[f64],
    ) -> f64 {
        if sft_losses.is_empty() {
            return 0.0;
        }

        let sum: f64 = sft_losses
            .iter()
            .zip(chosen_probs.iter())
            .zip(rejected_probs.iter())
            .map(|((sft, cp), rp)| self.compute(*sft, *cp, *rp))
            .sum();

        sum / sft_losses.len() as f64
    }
}

/// Numerically stable log-sigmoid.
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -((1.0 + (-x).exp()).ln())
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orpo_loss() {
        let loss = OrpoLoss::new(0.5);
        // When chosen has higher probability, alignment loss should be lower
        let l1 = loss.compute_alignment_loss(0.8, 0.3);
        let l2 = loss.compute_alignment_loss(0.5, 0.5);
        assert!(l1 < l2); // Better preference = lower loss
    }

    #[test]
    fn test_orpo_full_loss() {
        let loss = OrpoLoss::new(0.5);
        let total = loss.compute(2.0, 0.7, 0.3);
        assert!(total > 2.0); // SFT loss + positive alignment loss
    }

    #[test]
    fn test_orpo_batch() {
        let loss = OrpoLoss::new(0.5);
        let batch_loss = loss.compute_batch(&[2.0, 1.5], &[0.7, 0.8], &[0.3, 0.4]);
        let individual_avg = (loss.compute(2.0, 0.7, 0.3) + loss.compute(1.5, 0.8, 0.4)) / 2.0;
        assert!((batch_loss - individual_avg).abs() < 1e-10);
    }

    #[test]
    fn test_odds() {
        assert!((OrpoLoss::odds(0.5) - 1.0).abs() < 1e-10);
        assert!(OrpoLoss::odds(0.8) > 1.0);
        assert!(OrpoLoss::odds(0.2) < 1.0);
    }
}
