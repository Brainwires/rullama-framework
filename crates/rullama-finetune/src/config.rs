use serde::{Deserialize, Serialize};

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparams {
    /// Number of training epochs.
    pub epochs: u32,
    /// Batch size per device.
    pub batch_size: u32,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// Warmup steps for LR scheduler.
    pub warmup_steps: u64,
    /// Weight decay factor.
    pub weight_decay: f64,
    /// Learning rate scheduler type.
    pub lr_scheduler: LrScheduler,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Maximum sequence length (tokens).
    pub max_seq_len: usize,
    /// Gradient accumulation steps (effective batch = batch_size * grad_accum).
    pub gradient_accumulation_steps: u32,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,
}

impl Default for TrainingHyperparams {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 4,
            learning_rate: 2e-5,
            warmup_steps: 100,
            weight_decay: 0.01,
            lr_scheduler: LrScheduler::Cosine,
            seed: 42,
            max_seq_len: 2048,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
        }
    }
}

/// Learning rate scheduler types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LrScheduler {
    /// Constant learning rate.
    Constant,
    /// Linear decay to zero.
    Linear,
    /// Cosine annealing.
    Cosine,
    /// Cosine with warm restarts.
    CosineWarmRestarts,
}

/// LoRA adapter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank (typical: 8, 16, 32, 64).
    pub rank: u32,
    /// LoRA alpha scaling factor (typical: rank * 2).
    pub alpha: f32,
    /// Dropout rate on LoRA layers.
    pub dropout: f32,
    /// Target modules to apply LoRA to (e.g., ["q_proj", "v_proj"]).
    pub target_modules: Vec<String>,
    /// Adapter method variant.
    pub method: AdapterMethod,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            method: AdapterMethod::LoRA,
        }
    }
}

/// Adapter method for parameter-efficient fine-tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterMethod {
    /// Low-Rank Adaptation.
    LoRA,
    /// Quantized LoRA (4-bit or 8-bit base weights).
    QLoRA {
        /// Quantization bit width.
        bits: u8,
    },
    /// Weight-Decomposed Low-Rank Adaptation (direction + magnitude).
    DoRA,
    /// Quantized DoRA.
    QDoRA {
        /// Quantization bit width.
        bits: u8,
    },
}

impl AdapterMethod {
    /// Whether this adapter method uses quantization.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::QLoRA { .. } | Self::QDoRA { .. })
    }

    /// Return quantization bit width, if applicable.
    pub fn quantization_bits(&self) -> Option<u8> {
        match self {
            Self::QLoRA { bits } | Self::QDoRA { bits } => Some(*bits),
            _ => None,
        }
    }
}

/// Alignment training method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum AlignmentMethod {
    /// Direct Preference Optimization.
    DPO {
        /// DPO beta parameter.
        beta: f64,
    },
    /// Odds Ratio Preference Optimization (single-pass).
    ORPO {
        /// ORPO lambda parameter.
        lambda: f64,
    },
    /// No alignment, standard SFT only.
    #[default]
    None,
}

impl AlignmentMethod {
    /// Create DPO alignment with default beta.
    pub fn dpo() -> Self {
        Self::DPO { beta: 0.1 }
    }

    /// Create ORPO alignment with default lambda.
    pub fn orpo() -> Self {
        Self::ORPO { lambda: 0.5 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparams_defaults() {
        let h = TrainingHyperparams::default();
        assert_eq!(h.epochs, 3);
        assert_eq!(h.batch_size, 4);
        assert!((h.learning_rate - 2e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lora_config_defaults() {
        let c = LoraConfig::default();
        assert_eq!(c.rank, 16);
        assert_eq!(c.target_modules.len(), 4);
    }

    #[test]
    fn test_adapter_method_quantized() {
        assert!(!AdapterMethod::LoRA.is_quantized());
        assert!(AdapterMethod::QLoRA { bits: 4 }.is_quantized());
        assert_eq!(
            AdapterMethod::QLoRA { bits: 4 }.quantization_bits(),
            Some(4)
        );
        assert!(AdapterMethod::DoRA.quantization_bits().is_none());
    }

    #[test]
    fn test_alignment_methods() {
        let dpo = AlignmentMethod::dpo();
        assert!(matches!(dpo, AlignmentMethod::DPO { beta } if (beta - 0.1).abs() < f64::EPSILON));

        let orpo = AlignmentMethod::orpo();
        assert!(
            matches!(orpo, AlignmentMethod::ORPO { lambda } if (lambda - 0.5).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = LoraConfig {
            method: AdapterMethod::QLoRA { bits: 4 },
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: LoraConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.method, AdapterMethod::QLoRA { bits: 4 });
    }
}
