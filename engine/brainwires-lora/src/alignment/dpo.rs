/// Direct Preference Optimization (DPO) loss.
///
/// DPO eliminates the need for a reward model by directly optimizing
/// the policy with a simple binary cross-entropy loss:
///
/// L_DPO = -log σ(β * (log π(y_w|x) / π_ref(y_w|x) - log π(y_l|x) / π_ref(y_l|x)))
///
/// Where:
/// - y_w = preferred/chosen response
/// - y_l = dispreferred/rejected response
/// - π = current policy
/// - π_ref = reference policy (frozen base model)
/// - β = temperature parameter controlling deviation from reference
#[derive(Debug, Clone)]
pub struct DpoLoss {
    /// Temperature parameter. Higher β = closer to reference policy.
    pub beta: f64,
    /// Label smoothing for robustness.
    pub label_smoothing: f64,
}

impl Default for DpoLoss {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
        }
    }
}

impl DpoLoss {
    /// Create a new DPO loss with the given temperature parameter.
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            label_smoothing: 0.0,
        }
    }

    /// Set the label smoothing factor for robustness.
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    /// Compute DPO loss from log-probabilities.
    ///
    /// - `chosen_logps`: Log-probability of chosen response under current policy
    /// - `rejected_logps`: Log-probability of rejected response under current policy
    /// - `ref_chosen_logps`: Log-probability of chosen response under reference policy
    /// - `ref_rejected_logps`: Log-probability of rejected response under reference policy
    pub fn compute(
        &self,
        chosen_logps: f64,
        rejected_logps: f64,
        ref_chosen_logps: f64,
        ref_rejected_logps: f64,
    ) -> f64 {
        let chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps);
        let rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps);

        let logits = chosen_rewards - rejected_rewards;

        // Binary cross-entropy with label smoothing
        if self.label_smoothing > 0.0 {
            let smooth = self.label_smoothing;
            -(smooth * log_sigmoid(-logits) + (1.0 - smooth) * log_sigmoid(logits))
        } else {
            -log_sigmoid(logits)
        }
    }

    /// Compute average DPO loss over a batch of preference pairs.
    pub fn compute_batch(
        &self,
        chosen_logps: &[f64],
        rejected_logps: &[f64],
        ref_chosen_logps: &[f64],
        ref_rejected_logps: &[f64],
    ) -> f64 {
        if chosen_logps.is_empty() {
            return 0.0;
        }

        let sum: f64 = chosen_logps
            .iter()
            .zip(rejected_logps.iter())
            .zip(ref_chosen_logps.iter())
            .zip(ref_rejected_logps.iter())
            .map(|(((c, r), rc), rr)| self.compute(*c, *r, *rc, *rr))
            .sum();

        sum / chosen_logps.len() as f64
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
    fn test_dpo_loss_basic() {
        let loss = DpoLoss::new(0.1);
        // When chosen has higher log-prob ratio, loss should be low
        let l = loss.compute(-1.0, -3.0, -1.5, -1.5);
        assert!(l > 0.0); // Loss is always positive
        assert!(l < 1.0); // Should be reasonable with these inputs
    }

    #[test]
    fn test_dpo_loss_symmetry() {
        let loss = DpoLoss::new(0.1);
        // If chosen and rejected are equally likely, loss should be log(2)
        let l = loss.compute(-2.0, -2.0, -2.0, -2.0);
        assert!((l - 2.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_dpo_batch() {
        let loss = DpoLoss::new(0.1);
        let batch_loss =
            loss.compute_batch(&[-1.0, -1.5], &[-3.0, -2.5], &[-1.5, -1.5], &[-1.5, -1.5]);
        let individual_avg =
            (loss.compute(-1.0, -3.0, -1.5, -1.5) + loss.compute(-1.5, -2.5, -1.5, -1.5)) / 2.0;
        assert!((batch_loss - individual_avg).abs() < 1e-10);
    }

    #[test]
    fn test_log_sigmoid() {
        assert!((log_sigmoid(0.0) - (-2.0_f64.ln())).abs() < 1e-10);
        assert!(log_sigmoid(100.0) > -1e-10); // Should be close to 0
        assert!(log_sigmoid(-100.0) < -99.0); // Should be very negative
    }
}
