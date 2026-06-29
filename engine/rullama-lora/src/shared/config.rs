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
    /// Trade backward-pass memory for compute. When `true`, the
    /// scratch stores only one capture set shared across layers (+
    /// per-layer `hidden_in`); the per-layer activations are
    /// **recomputed** during backward by replaying that layer's
    /// forward right before its backward. Memory reduction is
    /// proportional to `n_layers - 1` for the non-`hidden_in`
    /// captures; compute cost is roughly +1x the forward.
    ///
    /// For Gemma 4 e2b at seq=1 the saving is ~1.7 MB out of ~50 MB
    /// total scratch — useful when seq grows under future
    /// PerPosition variants, not when running the M0 NextToken
    /// smoke. Default `false`.
    #[serde(default)]
    pub gradient_checkpointing: bool,
    /// Store LoRA A/B in bf16 instead of f32 (Adam state stays
    /// fp32). Memory savings on `save_adapter` are 2×; in-memory
    /// savings on the GPU side are 2× *only* if the bf16 kernel
    /// variants are wired. Currently honored at the safetensors
    /// save path — bf16 adapters round-trip identically.
    #[serde(default)]
    pub mixed_precision: bool,
    /// **Truncated backward** — exit the backward sweep early when the
    /// current layer index is below this floor. `0` (default) means
    /// "backprop through every layer" (the standard case); larger
    /// values progressively narrow the trainable region to just the
    /// top `n_layers - backward_layer_floor` layers.
    ///
    /// Memory + compute win on memory-constrained devices (iPhone):
    /// the per-layer LoRA backward, attention/FFN backward, and per-
    /// layer activation captures BELOW the floor are skipped. The
    /// FORWARD pass still runs every layer (the activations at the
    /// floor are needed to seed the partial backward); only the
    /// backward sweep is truncated.
    ///
    /// Trade-off: the adapter only updates the unfrozen top layers,
    /// so the adapter's expressive range shrinks. Empirically the
    /// last 5-10 layers carry most of the task-specific signal, so
    /// `floor = n_layers - 10` is a reasonable iPhone-safe default.
    ///
    /// Default `0` keeps the production training path unchanged.
    #[serde(default)]
    pub backward_layer_floor: u32,

    /// **Memory-tight mode** — enables the iOS-Safari-WebGPU survival
    /// stack: per-layer weight destroy during forward (MeBP), tiled
    /// head_outproj matmul (8 vocab-tiles), per-step JS event-loop
    /// yields at GPU submit boundaries, backward-kernel pre-warm at
    /// session start, chunked destroy IPC. Each of these trades
    /// compute time for memory pressure relief on iPhone Safari, where
    /// the WebContent process is killed at ~1.4 GiB GPU RSS.
    ///
    /// On Mac browsers / desktop, none of this is needed — turning it
    /// off restores the native-fast path. The total compute cost of
    /// the iOS workarounds is ~3-5× extra training time (MeBP destroy
    /// alone is +30-40% per the MeBP paper §4.2).
    ///
    /// Default `false` keeps the fast desktop path. The JS-side
    /// "Memory-tight" toggle in the PWA's Fine-tune panel sets this
    /// to true when the user opts into the iPhone-safe preset (auto-
    /// applied on mobile UAs).
    #[serde(default)]
    pub memory_tight: bool,
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
            gradient_checkpointing: false,
            mixed_precision: false,
            backward_layer_floor: 0,
            memory_tight: false,
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
    /// Optional per-layer targeting. `None` (default) → wrap LoRA on
    /// every layer for each `target_modules` entry (the standard
    /// fine-tune path). `Some(layers)` → only wrap LoRA on the
    /// specified layer indices. Used by the ROME pipeline to
    /// restrict the rank-1 LoRA on `ffn_down` to one specific layer
    /// (so the edit fires only when that layer's FFN runs — single-
    /// layer fact-locality semantics from the ROME paper).
    #[serde(default)]
    pub target_layers: Option<Vec<u32>>,
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
            target_layers: None,
        }
    }
}

impl LoraConfig {
    /// True iff `layer_idx` should be LoRA-wrapped. Respects the
    /// optional per-layer restriction.
    pub fn includes_layer(&self, layer_idx: u32) -> bool {
        match &self.target_layers {
            Some(layers) => layers.contains(&layer_idx),
            None => true,
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
