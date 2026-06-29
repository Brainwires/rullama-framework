use crate::config::TrainingHyperparams;

/// Per-provider cost estimation for fine-tuning jobs.
pub struct CostEstimator;

/// Cost breakdown for a fine-tuning job.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Provider name.
    pub provider: String,
    /// Estimated cost in USD.
    pub estimated_cost_usd: f64,
    /// Cost per 1M training tokens.
    pub cost_per_million_tokens: f64,
    /// Total estimated tokens.
    pub total_tokens: u64,
    /// Number of epochs.
    pub epochs: u32,
}

impl CostEstimator {
    /// Estimate cost for OpenAI fine-tuning.
    pub fn openai(
        model: &str,
        total_tokens: u64,
        hyperparams: &TrainingHyperparams,
    ) -> CostEstimate {
        // OpenAI pricing (per 1M training tokens, as of 2025)
        let cost_per_million = match model {
            m if m.contains("gpt-4o-mini") => 3.00,
            m if m.contains("gpt-4o") => 25.00,
            m if m.contains("gpt-4") => 30.00,
            m if m.contains("gpt-3.5") => 8.00,
            _ => 10.00, // default estimate
        };

        let total = total_tokens as f64 * hyperparams.epochs as f64;
        let cost = total / 1_000_000.0 * cost_per_million;

        CostEstimate {
            provider: "openai".to_string(),
            estimated_cost_usd: cost,
            cost_per_million_tokens: cost_per_million,
            total_tokens,
            epochs: hyperparams.epochs,
        }
    }

    /// Estimate cost for Together AI fine-tuning.
    pub fn together(
        model: &str,
        total_tokens: u64,
        hyperparams: &TrainingHyperparams,
    ) -> CostEstimate {
        // Together AI pricing (per 1M tokens, approximate)
        let cost_per_million = match model {
            m if m.contains("8B") || m.contains("8b") => 0.50,
            m if m.contains("70B") || m.contains("70b") => 3.00,
            m if m.contains("Mixtral") => 2.00,
            _ => 1.00,
        };

        let total = total_tokens as f64 * hyperparams.epochs as f64;
        let cost = total / 1_000_000.0 * cost_per_million;

        CostEstimate {
            provider: "together".to_string(),
            estimated_cost_usd: cost,
            cost_per_million_tokens: cost_per_million,
            total_tokens,
            epochs: hyperparams.epochs,
        }
    }

    /// Estimate cost for Fireworks AI fine-tuning.
    pub fn fireworks(
        model: &str,
        total_tokens: u64,
        hyperparams: &TrainingHyperparams,
    ) -> CostEstimate {
        let cost_per_million = match model {
            m if m.contains("7b") || m.contains("7B") || m.contains("8b") || m.contains("8B") => {
                0.40
            }
            m if m.contains("13b") || m.contains("13B") => 0.80,
            m if m.contains("70b") || m.contains("70B") => 3.00,
            m if m.contains("Mixtral") || m.contains("mixtral") => 2.00,
            _ => 1.00,
        };
        let total = total_tokens as f64 * hyperparams.epochs as f64;
        let cost = total / 1_000_000.0 * cost_per_million;

        CostEstimate {
            provider: "fireworks".to_string(),
            estimated_cost_usd: cost,
            cost_per_million_tokens: cost_per_million,
            total_tokens,
            epochs: hyperparams.epochs,
        }
    }

    /// Estimate cost for Anyscale fine-tuning.
    pub fn anyscale(
        model: &str,
        total_tokens: u64,
        hyperparams: &TrainingHyperparams,
    ) -> CostEstimate {
        let cost_per_million = match model {
            m if m.contains("7b") || m.contains("7B") || m.contains("8b") || m.contains("8B") => {
                0.30
            }
            m if m.contains("13b") || m.contains("13B") => 0.60,
            m if m.contains("70b") || m.contains("70B") => 2.50,
            m if m.contains("Mixtral") || m.contains("mixtral") => 1.50,
            _ => 0.80,
        };
        let total = total_tokens as f64 * hyperparams.epochs as f64;
        let cost = total / 1_000_000.0 * cost_per_million;

        CostEstimate {
            provider: "anyscale".to_string(),
            estimated_cost_usd: cost,
            cost_per_million_tokens: cost_per_million,
            total_tokens,
            epochs: hyperparams.epochs,
        }
    }

    /// Estimate cost for AWS Bedrock fine-tuning.
    pub fn bedrock(
        model: &str,
        total_tokens: u64,
        hyperparams: &TrainingHyperparams,
    ) -> CostEstimate {
        let cost_per_million = match model {
            m if m.contains("claude") || m.contains("Claude") => 20.00,
            m if m.contains("llama") || m.contains("Llama") => 1.00,
            m if m.contains("titan") || m.contains("Titan") => 0.80,
            _ => 5.00,
        };
        let total = total_tokens as f64 * hyperparams.epochs as f64;
        let cost = total / 1_000_000.0 * cost_per_million;

        CostEstimate {
            provider: "bedrock".to_string(),
            estimated_cost_usd: cost,
            cost_per_million_tokens: cost_per_million,
            total_tokens,
            epochs: hyperparams.epochs,
        }
    }

    /// Estimate cost for Google Vertex AI fine-tuning.
    pub fn vertex(
        model: &str,
        total_tokens: u64,
        hyperparams: &TrainingHyperparams,
    ) -> CostEstimate {
        let cost_per_million = match model {
            m if m.contains("flash") || m.contains("Flash") => 4.00,
            m if m.contains("pro") || m.contains("Pro") => 16.00,
            _ => 8.00,
        };
        let total = total_tokens as f64 * hyperparams.epochs as f64;
        let cost = total / 1_000_000.0 * cost_per_million;

        CostEstimate {
            provider: "vertex".to_string(),
            estimated_cost_usd: cost,
            cost_per_million_tokens: cost_per_million,
            total_tokens,
            epochs: hyperparams.epochs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_cost_estimation() {
        let hyperparams = TrainingHyperparams::default(); // 3 epochs
        let estimate = CostEstimator::openai("gpt-4o-mini-2024-07-18", 1_000_000, &hyperparams);

        assert_eq!(estimate.provider, "openai");
        assert!((estimate.cost_per_million_tokens - 3.0).abs() < f64::EPSILON);
        // 1M tokens * 3 epochs * $3/1M = $9
        assert!((estimate.estimated_cost_usd - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_together_cost_estimation() {
        let hyperparams = TrainingHyperparams::default();
        let estimate = CostEstimator::together(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            500_000,
            &hyperparams,
        );

        assert_eq!(estimate.provider, "together");
        // 500K tokens * 3 epochs * $0.50/1M = $0.75
        assert!((estimate.estimated_cost_usd - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_fireworks_model_pricing() {
        let hp = TrainingHyperparams::default();
        let e7b = CostEstimator::fireworks("llama-3.1-8B-instruct", 1_000_000, &hp);
        assert!((e7b.cost_per_million_tokens - 0.40).abs() < f64::EPSILON);
        let e70b = CostEstimator::fireworks("llama-3.1-70B-instruct", 1_000_000, &hp);
        assert!((e70b.cost_per_million_tokens - 3.00).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bedrock_cost() {
        let hp = TrainingHyperparams::default();
        let e = CostEstimator::bedrock("anthropic.claude-3-haiku", 1_000_000, &hp);
        assert_eq!(e.provider, "bedrock");
        assert!(e.estimated_cost_usd > 0.0);
    }

    #[test]
    fn test_vertex_cost() {
        let hp = TrainingHyperparams::default();
        let e = CostEstimator::vertex("gemini-1.5-flash-002", 1_000_000, &hp);
        assert_eq!(e.provider, "vertex");
        assert!(e.estimated_cost_usd > 0.0);
    }
}
