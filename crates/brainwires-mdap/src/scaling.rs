//! Scaling Laws for MDAP
//!
//! Implements equations 13-18 from the MAKER paper for cost and probability estimation.
//! These allow predicting the cost and success probability of MDAP execution.
//!
//! Key equations:
//! - Success probability: `p_full = (1 + ((1-p)/p)^k)^(-s)` (Eq. 13)
//! - Minimum k: `k_min = ceil(ln(t^(-1/s) - 1) / ln((1-p)/p))` (Eq. 14)
//! - Expected cost: `E[cost] ≈ c·s·k_min / (v·(2p-1))` (Eq. 19)

use super::error::{MdapResult, ScalingError};

/// Cost and probability estimation result
#[derive(Clone, Debug)]
pub struct MdapEstimate {
    /// Expected cost in USD
    pub expected_cost_usd: f64,
    /// Expected number of API calls
    pub expected_api_calls: u64,
    /// Probability of full task success
    pub success_probability: f64,
    /// Recommended k value for target success rate
    pub recommended_k: u32,
    /// Estimated execution time in seconds
    pub estimated_time_seconds: f64,
    /// Per-step success probability used in calculation
    pub per_step_success: f64,
    /// Number of steps in the task
    pub num_steps: u64,
}

/// Estimate MDAP execution cost and success probability
///
/// Implements equations 13-18 from the MAKER paper.
///
/// # Arguments
/// * `num_steps` - Number of subtasks (s in paper)
/// * `per_step_success_rate` - Probability each step succeeds (p in paper, must be > 0.5)
/// * `valid_response_rate` - Rate of valid (non-red-flagged) responses (v in paper)
/// * `cost_per_sample_usd` - Cost per API call in USD (c in paper)
/// * `target_success_rate` - Desired overall success probability (t in paper)
///
/// # Returns
/// * `Ok(MdapEstimate)` - Estimation results
/// * `Err(ScalingError)` - If parameters are invalid (e.g., p <= 0.5)
pub fn estimate_mdap(
    num_steps: u64,
    per_step_success_rate: f64,
    valid_response_rate: f64,
    cost_per_sample_usd: f64,
    target_success_rate: f64,
) -> MdapResult<MdapEstimate> {
    // Validate inputs
    if num_steps == 0 {
        return Err(ScalingError::InvalidStepCount(0).into());
    }

    if per_step_success_rate <= 0.5 {
        return Err(ScalingError::VotingCannotConverge {
            p: per_step_success_rate,
        }
        .into());
    }

    if per_step_success_rate >= 1.0 {
        return Err(ScalingError::InvalidSuccessProbability(per_step_success_rate).into());
    }

    if target_success_rate <= 0.0 || target_success_rate >= 1.0 {
        return Err(ScalingError::InvalidTargetProbability(target_success_rate).into());
    }

    let p = per_step_success_rate;
    let s = num_steps as f64;
    let t = target_success_rate;
    let v = valid_response_rate.clamp(0.01, 1.0); // Avoid division by zero
    let c = cost_per_sample_usd;

    // Equation 14: k_min = ceil(ln(t^(-1/s) - 1) / ln((1-p)/p))
    let k_min = calculate_k_min(s, p, t);

    // Equation 13: p_full = (1 + ((1-p)/p)^k)^(-s)
    let p_full = calculate_p_full(s, p, k_min);

    // Equation 19: E[cost] ≈ c·s·k_min / (v·(2p-1))
    let expected_cost = (c * s * k_min as f64) / (v * (2.0 * p - 1.0));

    // Expected API calls (accounting for red-flagged samples)
    let expected_calls = (s * k_min as f64 / v).ceil() as u64;

    // Rough time estimate (assuming 500ms per call, parallelized by 4)
    let time_per_step = 0.5 * (k_min as f64 / 4.0).ceil();
    let estimated_time = s * time_per_step;

    Ok(MdapEstimate {
        expected_cost_usd: expected_cost,
        expected_api_calls: expected_calls,
        success_probability: p_full,
        recommended_k: k_min,
        estimated_time_seconds: estimated_time,
        per_step_success: p,
        num_steps,
    })
}

/// Calculate minimum k for target success probability
///
/// Implements Equation 14 from the paper:
/// k_min = ceil(ln(t^(-1/s) - 1) / ln((1-p)/p))
///
/// # Arguments
/// * `num_steps` - Number of steps (s)
/// * `p` - Per-step success probability
/// * `target` - Target overall success probability (t)
///
/// # Returns
/// The minimum k value needed to achieve the target
pub fn calculate_k_min(num_steps: f64, p: f64, target: f64) -> u32 {
    if p <= 0.5 {
        return u32::MAX; // Voting won't converge
    }

    let ratio = (1.0 - p) / p;

    // t^(-1/s) - 1
    // Handle edge case where target is very close to 1
    if target >= 0.9999 {
        // For very high targets, use approximation
        let a = target.powf(-1.0 / num_steps) - 1.0;
        if a <= 0.0 || ratio <= 0.0 {
            return 10; // Default high k for extreme cases
        }
        let k = (a.ln() / ratio.ln()).ceil() as u32;
        return k.clamp(1, 100); // Cap at reasonable values
    }

    let a = target.powf(-1.0 / num_steps) - 1.0;

    if a <= 0.0 {
        return 1; // Target is easily achievable
    }

    if ratio <= 0.0 || ratio >= 1.0 {
        // Edge case: handle numerical issues
        return 1;
    }

    // k = ceil(ln(a) / ln(ratio))
    let k = (a.ln() / ratio.ln()).ceil() as u32;

    k.max(1) // Minimum k=1
}

/// Calculate full-task success probability
///
/// Implements Equation 13 from the paper:
/// p_full = (1 + ((1-p)/p)^k)^(-s)
///
/// # Arguments
/// * `num_steps` - Number of steps (s)
/// * `p` - Per-step success probability
/// * `k` - Vote margin threshold
///
/// # Returns
/// The probability of completing all steps successfully
pub fn calculate_p_full(num_steps: f64, p: f64, k: u32) -> f64 {
    if p <= 0.5 {
        return 0.0; // Voting won't converge
    }

    let ratio = (1.0 - p) / p;

    // Handle numerical stability for large k
    let ratio_k = if k > 50 {
        // For very large k, ratio^k approaches 0 (since ratio < 1 when p > 0.5)
        0.0
    } else {
        ratio.powi(k as i32)
    };

    // p_sub = 1 / (1 + ratio^k)
    let p_sub = 1.0 / (1.0 + ratio_k);

    // p_full = p_sub^s
    p_sub.powf(num_steps)
}

/// Calculate expected number of votes needed per step
///
/// Based on paper's analysis of first-to-ahead-by-k voting.
///
/// # Arguments
/// * `p` - Per-step success probability
/// * `k` - Vote margin threshold
///
/// # Returns
/// Expected number of votes before a winner emerges
pub fn calculate_expected_votes(p: f64, k: u32) -> f64 {
    if p <= 0.5 {
        return f64::INFINITY;
    }

    // Approximation: E[votes] ≈ k / (2p - 1)
    k as f64 / (2.0 * p - 1.0)
}

/// Estimate per-step success rate from sample data
///
/// # Arguments
/// * `total_samples` - Total number of samples taken
/// * `correct_samples` - Number of correct samples
/// * `red_flagged_samples` - Number of red-flagged samples
///
/// # Returns
/// Estimated per-step success rate
pub fn estimate_per_step_success(
    total_samples: u64,
    correct_samples: u64,
    red_flagged_samples: u64,
) -> f64 {
    let valid = total_samples.saturating_sub(red_flagged_samples);
    if valid == 0 {
        return 0.5; // Default to neutral when no valid samples
    }
    (correct_samples as f64 / valid as f64).clamp(0.0, 1.0)
}

/// Estimate valid response rate from sample data
///
/// # Arguments
/// * `total_samples` - Total number of samples taken
/// * `red_flagged_samples` - Number of red-flagged samples
///
/// # Returns
/// Estimated valid response rate (v in paper)
pub fn estimate_valid_response_rate(total_samples: u64, red_flagged_samples: u64) -> f64 {
    if total_samples == 0 {
        return 0.95; // Default assumption
    }
    let valid = total_samples.saturating_sub(red_flagged_samples);
    (valid as f64 / total_samples as f64).clamp(0.01, 1.0)
}

/// Calculate cost for a specific configuration
///
/// # Arguments
/// * `num_steps` - Number of steps
/// * `k` - Vote margin threshold
/// * `valid_rate` - Valid response rate
/// * `per_step_success` - Per-step success probability
/// * `cost_per_call` - Cost per API call
///
/// # Returns
/// Expected total cost in USD
pub fn calculate_expected_cost(
    num_steps: u64,
    k: u32,
    valid_rate: f64,
    per_step_success: f64,
    cost_per_call: f64,
) -> f64 {
    let s = num_steps as f64;
    let v = valid_rate.clamp(0.01, 1.0);
    let p = per_step_success.clamp(0.51, 0.999);

    // E[cost] ≈ c·s·k / (v·(2p-1))
    (cost_per_call * s * k as f64) / (v * (2.0 * p - 1.0))
}

/// Suggest optimal k for budget constraint
///
/// # Arguments
/// * `num_steps` - Number of steps
/// * `per_step_success` - Per-step success probability
/// * `valid_rate` - Valid response rate
/// * `cost_per_call` - Cost per API call
/// * `budget_usd` - Maximum budget in USD
///
/// # Returns
/// Maximum k that fits within budget
pub fn suggest_k_for_budget(
    num_steps: u64,
    per_step_success: f64,
    valid_rate: f64,
    cost_per_call: f64,
    budget_usd: f64,
) -> u32 {
    let s = num_steps as f64;
    let v = valid_rate.clamp(0.01, 1.0);
    let p = per_step_success.clamp(0.51, 0.999);
    let c = cost_per_call;

    // From E[cost] = c·s·k / (v·(2p-1))
    // Solving for k: k = budget · v · (2p-1) / (c · s)
    let k = (budget_usd * v * (2.0 * p - 1.0)) / (c * s);

    (k.floor() as u32).max(1)
}

/// Model-specific cost estimates (per 1000 tokens)
#[derive(Clone, Debug)]
pub struct ModelCosts {
    /// Cost per 1000 input tokens
    pub input_per_1k: f64,
    /// Cost per 1000 output tokens
    pub output_per_1k: f64,
}

impl ModelCosts {
    /// Claude 3.5 Sonnet pricing
    pub fn claude_sonnet() -> Self {
        Self {
            input_per_1k: 0.003,
            output_per_1k: 0.015,
        }
    }

    /// Claude 3.5 Haiku pricing
    pub fn claude_haiku() -> Self {
        Self {
            input_per_1k: 0.00025,
            output_per_1k: 0.00125,
        }
    }

    /// GPT-4o pricing
    pub fn gpt4o() -> Self {
        Self {
            input_per_1k: 0.0025,
            output_per_1k: 0.01,
        }
    }

    /// GPT-4o-mini pricing
    pub fn gpt4o_mini() -> Self {
        Self {
            input_per_1k: 0.00015,
            output_per_1k: 0.0006,
        }
    }

    /// Estimate cost for a single call
    pub fn estimate_call_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        (input_tokens as f64 / 1000.0 * self.input_per_1k)
            + (output_tokens as f64 / 1000.0 * self.output_per_1k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_k_min_basic() {
        // With p=0.99 and s=100, what k do we need for t=0.95?
        let k = calculate_k_min(100.0, 0.99, 0.95);
        assert!(k >= 1);
        assert!(k <= 10); // Should be achievable with small k for high p
    }

    #[test]
    fn test_calculate_k_min_low_p() {
        // With p=0.6 (just above 0.5), need higher k
        let k = calculate_k_min(10.0, 0.6, 0.95);
        assert!(k > 1);
    }

    #[test]
    fn test_calculate_k_min_edge_cases() {
        // p <= 0.5 should return MAX
        let k = calculate_k_min(10.0, 0.5, 0.95);
        assert_eq!(k, u32::MAX);

        let k = calculate_k_min(10.0, 0.4, 0.95);
        assert_eq!(k, u32::MAX);
    }

    #[test]
    fn test_calculate_p_full() {
        // With high p and reasonable k, should achieve high success
        let p_full = calculate_p_full(10.0, 0.99, 5);
        assert!(p_full > 0.99);

        // With lower p, success drops
        let p_full_low = calculate_p_full(10.0, 0.7, 3);
        assert!(p_full_low < p_full);
    }

    #[test]
    fn test_calculate_p_full_convergence() {
        // p <= 0.5 should return 0
        let p_full = calculate_p_full(10.0, 0.5, 5);
        assert_eq!(p_full, 0.0);
    }

    #[test]
    fn test_estimate_mdap_valid() {
        let estimate = estimate_mdap(100, 0.99, 0.95, 0.001, 0.95).unwrap();

        assert!(estimate.success_probability > 0.9);
        assert!(estimate.recommended_k >= 1);
        assert!(estimate.expected_cost_usd > 0.0);
        assert!(estimate.expected_api_calls > 0);
    }

    #[test]
    fn test_estimate_mdap_invalid_p() {
        let result = estimate_mdap(100, 0.4, 0.95, 0.001, 0.95);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_mdap_invalid_steps() {
        let result = estimate_mdap(0, 0.99, 0.95, 0.001, 0.95);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_per_step_success() {
        // 80 correct out of 100 total, 10 red-flagged = 80/90 = ~0.89
        let p = estimate_per_step_success(100, 80, 10);
        assert!((p - 0.889).abs() < 0.01);

        // All red-flagged should return 0.5
        let p_all_flagged = estimate_per_step_success(100, 0, 100);
        assert_eq!(p_all_flagged, 0.5);
    }

    #[test]
    fn test_estimate_valid_response_rate() {
        // 90 valid out of 100
        let v = estimate_valid_response_rate(100, 10);
        assert_eq!(v, 0.9);

        // Zero samples
        let v_zero = estimate_valid_response_rate(0, 0);
        assert_eq!(v_zero, 0.95);
    }

    #[test]
    fn test_calculate_expected_cost() {
        let cost = calculate_expected_cost(100, 3, 0.95, 0.99, 0.001);
        assert!(cost > 0.0);

        // Higher k = higher cost
        let cost_high_k = calculate_expected_cost(100, 10, 0.95, 0.99, 0.001);
        assert!(cost_high_k > cost);
    }

    #[test]
    fn test_suggest_k_for_budget() {
        let k = suggest_k_for_budget(100, 0.99, 0.95, 0.001, 1.0);
        assert!(k >= 1);

        // Smaller budget = smaller k
        let k_small = suggest_k_for_budget(100, 0.99, 0.95, 0.001, 0.1);
        assert!(k_small <= k);
    }

    #[test]
    fn test_model_costs() {
        let sonnet = ModelCosts::claude_sonnet();
        let cost = sonnet.estimate_call_cost(1000, 500);
        // 1000 input = $0.003, 500 output = $0.0075
        assert!((cost - 0.0105).abs() < 0.001);
    }

    #[test]
    fn test_calculate_expected_votes() {
        // With p=0.99, expected votes ≈ k / 0.98
        let votes = calculate_expected_votes(0.99, 3);
        assert!((votes - 3.06).abs() < 0.1);

        // p=0.5 should return infinity
        let votes_half = calculate_expected_votes(0.5, 3);
        assert!(votes_half.is_infinite());
    }

    #[test]
    fn test_high_step_count() {
        // Paper claims million-step tasks are feasible
        let estimate = estimate_mdap(1_000_000, 0.99, 0.95, 0.0001, 0.95).unwrap();

        // Should still achieve high success with reasonable k
        assert!(estimate.success_probability > 0.9);
    }
}
