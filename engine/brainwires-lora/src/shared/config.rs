use serde::{Deserialize, Serialize};

/// Training hyperparameters.
///
/// All fields are declared knobs that the trainer is intended to honor.
/// Fields that aren't wired yet are documented as such in the
/// `TrainingSession::step` path; nothing here is silently no-op.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparams {
    /// Number of training epochs.
    pub epochs: u32,
    /// Batch size per device.
    pub batch_size: u32,
    /// Initial (post-warmup) learning rate.
    pub learning_rate: f64,
    /// Warmup steps for the LR scheduler.
    pub warmup_steps: u64,
    /// Weight decay coefficient (applied inside Adam).
    pub weight_decay: f64,
    /// Learning rate scheduler type.
    pub lr_scheduler: LrScheduler,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Maximum sequence length (tokens).
    pub max_seq_len: usize,
    /// Gradient accumulation steps. Effective batch = `batch_size * gradient_accumulation_steps`.
    pub gradient_accumulation_steps: u32,
    /// Maximum gradient L2 norm before clipping. `0.0` disables clipping.
    pub max_grad_norm: f64,
    /// Loss objective. See [`LossMode`].
    #[serde(default)]
    pub loss_mode: LossMode,
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
            loss_mode: LossMode::default(),
        }
    }
}

/// Choice of cross-entropy loss objective.
///
/// - [`LossMode::NextToken`] — train on a *single* target token: the
///   first token of the completion given the full prompt. The forward
///   only needs logits at the final prompt position (current path); the
///   backward seeds `dL/d_logits` from one (softmax − one_hot) row.
///   Cheap and reliable; the M0 acceptance pipeline.
/// - [`LossMode::PerPosition`] — train on every position of the
///   completion: logits are emitted at all positions in a configured
///   range, CE is averaged across the completion (mask-aware), and
///   `dL/d_logits` is accumulated for each. Closer to standard
///   causal-LM fine-tuning. Adds an output-projection pass per token
///   in the range plus a per-row CE-backward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossMode {
    /// CE on the first completion token only (M0/M1 default).
    #[default]
    NextToken,
    /// CE averaged across every completion position (M1.4).
    PerPosition,
}

/// Learning rate scheduler types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LrScheduler {
    /// Constant learning rate after warmup.
    Constant,
    /// Linear decay to zero after warmup.
    Linear,
    /// Cosine annealing after warmup.
    Cosine,
    /// Cosine with warm restarts.
    CosineWarmRestarts,
}

/// LoRA adapter configuration.
///
/// Only plain LoRA is supported. QLoRA / DoRA are out of scope for the
/// initial native rewrite; see the migration report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank `r` (typical: 8, 16, 32).
    pub rank: u32,
    /// LoRA scaling factor α. The forward applies `(α / rank)` as the scale.
    pub alpha: f32,
    /// Dropout rate on the LoRA path (applied to the input of the A matmul).
    pub dropout: f32,
    /// Target projections to wrap with LoRA. Names match GGUF tensor stems —
    /// `attn_q`, `attn_k`, `attn_v`, `attn_o`, `ffn_gate`, `ffn_up`, `ffn_down`.
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "attn_q".to_string(),
                "attn_k".to_string(),
                "attn_v".to_string(),
                "attn_o".to_string(),
            ],
        }
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
        assert!(c.target_modules.contains(&"attn_q".to_string()));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = LoraConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: LoraConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rank, config.rank);
        assert_eq!(parsed.target_modules, config.target_modules);
    }
}
