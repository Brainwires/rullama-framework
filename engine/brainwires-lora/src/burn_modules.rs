//! Burn-native modules for LoRA fine-tuning.
//!
//! These are actual `burn::module::Module` implementations that run on GPU via WGPU.

use burn_core::module::{Module, Param};
use burn_core::prelude::*;
use burn_core::tensor::activation;
use burn_nn as nn;

/// LoRA adapter module in Burn.
///
/// Wraps a frozen base linear layer with trainable low-rank A/B matrices.
/// Forward: y = W_frozen @ x + (B @ A @ x) * scaling
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    /// Frozen base weight (not updated during training).
    base: nn::Linear<B>,
    /// Down-projection: (rank × in_features).
    lora_a: nn::Linear<B>,
    /// Up-projection: (out_features × rank).
    lora_b: nn::Linear<B>,
    /// Scaling factor: alpha / rank.
    #[module(skip)]
    scaling: f32,
    /// Whether the LoRA adapter is active.
    #[module(skip)]
    active: bool,
}

/// Configuration for creating a LoRA linear layer.
#[derive(Config, Debug)]
pub struct LoraLinearConfig {
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// LoRA rank (bottleneck dimension).
    #[config(default = "16")]
    pub rank: usize,
    /// Alpha scaling factor.
    #[config(default = "32.0")]
    pub alpha: f32,
}

impl LoraLinearConfig {
    /// Initialize LoRA linear layer with random base weights.
    ///
    /// Base weights are initialized from a normal distribution (would be loaded from model).
    /// LoRA A is initialized with Kaiming uniform, B is initialized to zero
    /// (so initial LoRA contribution is zero).
    pub fn init<B: Backend>(&self, device: &B::Device) -> LoraLinear<B> {
        let base = nn::LinearConfig::new(self.in_features, self.out_features)
            .with_bias(false)
            .init(device);

        // A: (in_features → rank) — Kaiming init
        let lora_a = nn::LinearConfig::new(self.in_features, self.rank)
            .with_bias(false)
            .init(device);

        // B: (rank → out_features) — zero init so LoRA starts as identity
        let lora_b_config = nn::LinearConfig::new(self.rank, self.out_features).with_bias(false);
        let mut lora_b = lora_b_config.init(device);
        // Zero-initialize B so the LoRA contribution starts at zero
        lora_b.weight = lora_b.weight.map(|w| w.zeros_like());

        LoraLinear {
            base,
            lora_a,
            lora_b,
            scaling: self.alpha / self.rank as f32,
            active: true,
        }
    }

    /// Initialize LoRA linear layer with pre-loaded base weights.
    ///
    /// Use this when loading real model weights from SafeTensors.
    /// The base weight tensor should have shape `[in_features, out_features]`.
    pub fn init_with_base_weights<B: Backend>(
        &self,
        base_weight: Tensor<B, 2>,
        device: &B::Device,
    ) -> LoraLinear<B> {
        let base = nn::Linear {
            weight: Param::from_tensor(base_weight),
            bias: None,
        };

        let lora_a = nn::LinearConfig::new(self.in_features, self.rank)
            .with_bias(false)
            .init(device);

        let lora_b_config = nn::LinearConfig::new(self.rank, self.out_features).with_bias(false);
        let mut lora_b = lora_b_config.init(device);
        lora_b.weight = lora_b.weight.map(|w| w.zeros_like());

        LoraLinear {
            base,
            lora_a,
            lora_b,
            scaling: self.alpha / self.rank as f32,
            active: true,
        }
    }
}

impl<B: Backend> LoraLinear<B> {
    /// Get LoRA A weight data for serialization.
    pub fn lora_a_weight(&self) -> Tensor<B, 2> {
        self.lora_a.weight.val()
    }

    /// Get LoRA B weight data for serialization.
    pub fn lora_b_weight(&self) -> Tensor<B, 2> {
        self.lora_b.weight.val()
    }

    /// Forward pass: base + LoRA adapter.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let base_out = self.base.forward(input.clone());

        if !self.active {
            return base_out;
        }

        // LoRA path: input → A → B, scaled
        let lora_out = self.lora_b.forward(self.lora_a.forward(input));
        let lora_scaled = lora_out.mul_scalar(self.scaling);

        base_out + lora_scaled
    }

    /// Freeze the base layer (already frozen by design, but explicit).
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Number of trainable parameters (A + B only).
    pub fn trainable_param_count(&self) -> usize {
        let a_shape = self.lora_a.weight.val().shape();
        let a_params = a_shape.dims[0] * a_shape.dims[1];
        let b_shape = self.lora_b.weight.val().shape();
        let b_params = b_shape.dims[0] * b_shape.dims[1];
        a_params + b_params
    }
}

/// RMS Layer Normalization (used in LLaMA-style models).
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter.
    weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability.
    #[module(skip)]
    eps: f64,
}

/// Configuration for RMS normalization.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Epsilon for numerical stability.
    #[config(default = "1e-5")]
    pub eps: f64,
}

impl RmsNormConfig {
    /// Initialize an RMS normalization layer on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let weight = Tensor::ones([self.hidden_size], device);
        RmsNorm {
            weight: Param::from_tensor(weight),
            eps: self.eps,
        }
    }
}

impl<B: Backend> RmsNorm<B> {
    /// Forward pass: normalize input using root mean square.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(1);
        let rms = (variance + self.eps).sqrt();
        let normed = x / rms;
        normed * self.weight.val().unsqueeze_dim(0)
    }
}

/// SwiGLU feed-forward network (LLaMA-style).
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
}

/// Configuration for SwiGLU FFN.
#[derive(Config, Debug)]
pub struct SwiGluFfnConfig {
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
}

impl SwiGluFfnConfig {
    /// Initialize a SwiGLU feed-forward network on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluFfn<B> {
        SwiGluFfn {
            gate_proj: nn::LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> SwiGluFfn<B> {
    /// Forward pass: gate with SiLU activation, element-wise multiply, and project down.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Simple cross-entropy loss for language modeling.
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,       // [batch * seq_len, vocab_size]
    targets: Tensor<B, 1, Int>, // [batch * seq_len]
) -> Tensor<B, 1> {
    let log_softmax = activation::log_softmax(logits, 1);
    let batch_size = targets.dims()[0];

    // Gather the log probabilities at target indices
    let targets_2d = targets.reshape([batch_size, 1]);
    let gathered = log_softmax.gather(1, targets_2d);

    // Negative log likelihood
    gathered.neg().mean()
}

/// Training step output.
#[derive(Debug)]
pub struct TrainStepOutput<B: Backend> {
    /// Loss tensor from the training step.
    pub loss: Tensor<B, 1>,
    /// Number of tokens processed in this step.
    pub num_tokens: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 5: DoRA, DPO/ORPO tensor losses, Transformer block
// ────────────────────────────────────────────────────────────────────────────

/// DoRA (Weight-Decomposed Low-Rank Adaptation) module in Burn.
///
/// Decomposes weight update into direction and magnitude:
///   W' = m * (W₀ + B·A) / ‖W₀ + B·A‖_col
///
/// Where m is a learnable per-output-neuron magnitude vector.
#[derive(Module, Debug)]
pub struct DoraLinear<B: Backend> {
    /// Frozen base weight.
    base: nn::Linear<B>,
    /// Down-projection: (in_features → rank).
    lora_a: nn::Linear<B>,
    /// Up-projection: (rank → out_features).
    lora_b: nn::Linear<B>,
    /// Learnable magnitude vector (one scalar per output neuron).
    magnitude: Param<Tensor<B, 1>>,
    /// Scaling factor: alpha / rank.
    #[module(skip)]
    scaling: f32,
}

/// Configuration for DoRA linear layer.
#[derive(Config, Debug)]
pub struct DoraLinearConfig {
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// LoRA rank for the directional component.
    #[config(default = "16")]
    pub rank: usize,
    /// Alpha scaling factor.
    #[config(default = "32.0")]
    pub alpha: f32,
}

impl DoraLinearConfig {
    /// Initialize a DoRA linear layer with random base weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DoraLinear<B> {
        let base = nn::LinearConfig::new(self.in_features, self.out_features)
            .with_bias(false)
            .init(device);

        let lora_a = nn::LinearConfig::new(self.in_features, self.rank)
            .with_bias(false)
            .init(device);

        let lora_b_config = nn::LinearConfig::new(self.rank, self.out_features).with_bias(false);
        let mut lora_b = lora_b_config.init(device);
        lora_b.weight = lora_b.weight.map(|w| w.zeros_like());

        // Initialize magnitude from base weight column norms.
        let base_w = base.weight.val();
        let col_norms = base_w
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            .squeeze::<1>();
        let magnitude = Param::from_tensor(col_norms);

        DoraLinear {
            base,
            lora_a,
            lora_b,
            magnitude,
            scaling: self.alpha / self.rank as f32,
        }
    }

    /// Initialize a DoRA linear layer with pre-loaded base weights.
    ///
    /// Use this when loading real model weights from SafeTensors.
    /// The base weight tensor should have shape `[in_features, out_features]`.
    pub fn init_with_base_weights<B: Backend>(
        &self,
        base_weight: Tensor<B, 2>,
        device: &B::Device,
    ) -> DoraLinear<B> {
        let col_norms = base_weight
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            .squeeze::<1>();
        let magnitude = Param::from_tensor(col_norms);

        let base = nn::Linear {
            weight: Param::from_tensor(base_weight),
            bias: None,
        };

        let lora_a = nn::LinearConfig::new(self.in_features, self.rank)
            .with_bias(false)
            .init(device);

        let lora_b_config = nn::LinearConfig::new(self.rank, self.out_features).with_bias(false);
        let mut lora_b = lora_b_config.init(device);
        lora_b.weight = lora_b.weight.map(|w| w.zeros_like());

        DoraLinear {
            base,
            lora_a,
            lora_b,
            magnitude,
            scaling: self.alpha / self.rank as f32,
        }
    }
}

impl<B: Backend> DoraLinear<B> {
    /// Get LoRA A weight data for serialization.
    pub fn lora_a_weight(&self) -> Tensor<B, 2> {
        self.lora_a.weight.val()
    }

    /// Get LoRA B weight data for serialization.
    pub fn lora_b_weight(&self) -> Tensor<B, 2> {
        self.lora_b.weight.val()
    }

    /// Get magnitude vector data for serialization.
    pub fn magnitude_data(&self) -> Tensor<B, 1> {
        self.magnitude.val()
    }

    /// Forward pass with direction-magnitude decomposition.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Burn stores Linear weights as [in_features, out_features].
        // LoRA update: A_w @ B_w in Burn convention (opposite of PyTorch's B @ A).
        let lora_a_w = self.lora_a.weight.val(); // [in_features, rank]
        let lora_b_w = self.lora_b.weight.val(); // [rank, out_features]
        let lora_update = lora_a_w.matmul(lora_b_w).mul_scalar(self.scaling); // [in, out]
        let updated_w = self.base.weight.val() + lora_update; // [in, out]

        // Per-output-neuron norms: sum across dim 0 (input dim) since columns = outputs
        let col_norms = updated_w.clone().powf_scalar(2.0).sum_dim(0).sqrt(); // [1, out]
        let eps: f32 = 1e-8;
        let col_norms_safe = col_norms + eps;

        // Normalize: direction = W' / ‖W'‖
        let direction = updated_w / col_norms_safe; // [in, out]

        // Apply magnitude: W_final = m * direction
        let m = self.magnitude.val().unsqueeze_dim(0); // [1, out]
        let final_w = direction * m; // [in, out]

        // Forward: y = input @ W (Burn convention, no transpose needed)
        input.matmul(final_w)
    }

    /// Total trainable parameters: LoRA A + LoRA B + magnitude vector.
    pub fn trainable_param_count(&self) -> usize {
        let a_shape = self.lora_a.weight.val().shape();
        let b_shape = self.lora_b.weight.val().shape();
        let m_shape = self.magnitude.val().shape();
        a_shape.dims[0] * a_shape.dims[1] + b_shape.dims[0] * b_shape.dims[1] + m_shape.dims[0]
    }
}

/// QLoRA adapter module in Burn.
///
/// Like `LoraLinear`, but the base weight was loaded from a quantized source
/// and dequantized at init time. The frozen base linear is identical in structure;
/// memory savings come from the quantized *storage* format (SafeTensors + INT4/INT8).
#[derive(Module, Debug)]
pub struct QLoraLinear<B: Backend> {
    /// Frozen base weight (dequantized from quantized storage).
    base: nn::Linear<B>,
    /// Down-projection: (in_features → rank).
    lora_a: nn::Linear<B>,
    /// Up-projection: (rank → out_features).
    lora_b: nn::Linear<B>,
    /// Scaling factor: alpha / rank.
    #[module(skip)]
    scaling: f32,
    /// Whether the LoRA adapter is active.
    #[module(skip)]
    active: bool,
}

/// Configuration for creating a QLoRA linear layer.
#[derive(Config, Debug)]
pub struct QLoraLinearConfig {
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// LoRA rank (bottleneck dimension).
    #[config(default = "16")]
    pub rank: usize,
    /// Alpha scaling factor.
    #[config(default = "32.0")]
    pub alpha: f32,
    /// Quantization bits (4 or 8).
    #[config(default = "4")]
    pub bits: u8,
}

impl QLoraLinearConfig {
    /// Initialize QLoRA with pre-dequantized base weights.
    ///
    /// The `base_weight_f32` slice contains weights that have already been
    /// through the quantize → dequantize pipeline (i.e., they carry quantization
    /// noise but are in f32 format for training).
    pub fn init_quantized<B: Backend>(
        &self,
        base_weight_f32: &[f32],
        device: &B::Device,
    ) -> QLoraLinear<B> {
        let weight_tensor = Tensor::<B, 1>::from_floats(
            burn_core::tensor::TensorData::new(base_weight_f32.to_vec(), [base_weight_f32.len()]),
            device,
        )
        .reshape([self.in_features, self.out_features]);

        let base = nn::Linear {
            weight: Param::from_tensor(weight_tensor),
            bias: None,
        };

        let lora_a = nn::LinearConfig::new(self.in_features, self.rank)
            .with_bias(false)
            .init(device);

        let lora_b_config = nn::LinearConfig::new(self.rank, self.out_features).with_bias(false);
        let mut lora_b = lora_b_config.init(device);
        lora_b.weight = lora_b.weight.map(|w| w.zeros_like());

        QLoraLinear {
            base,
            lora_a,
            lora_b,
            scaling: self.alpha / self.rank as f32,
            active: true,
        }
    }

    /// Initialize QLoRA with random base weights (fallback when no model file).
    pub fn init<B: Backend>(&self, device: &B::Device) -> QLoraLinear<B> {
        let base = nn::LinearConfig::new(self.in_features, self.out_features)
            .with_bias(false)
            .init(device);

        let lora_a = nn::LinearConfig::new(self.in_features, self.rank)
            .with_bias(false)
            .init(device);

        let lora_b_config = nn::LinearConfig::new(self.rank, self.out_features).with_bias(false);
        let mut lora_b = lora_b_config.init(device);
        lora_b.weight = lora_b.weight.map(|w| w.zeros_like());

        QLoraLinear {
            base,
            lora_a,
            lora_b,
            scaling: self.alpha / self.rank as f32,
            active: true,
        }
    }
}

impl<B: Backend> QLoraLinear<B> {
    /// Forward pass: base + LoRA adapter (identical to LoRA).
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let base_out = self.base.forward(input.clone());

        if !self.active {
            return base_out;
        }

        let lora_out = self.lora_b.forward(self.lora_a.forward(input));
        let lora_scaled = lora_out.mul_scalar(self.scaling);
        base_out + lora_scaled
    }

    /// Get LoRA A weight data for serialization.
    pub fn lora_a_weight(&self) -> Tensor<B, 2> {
        self.lora_a.weight.val()
    }

    /// Get LoRA B weight data for serialization.
    pub fn lora_b_weight(&self) -> Tensor<B, 2> {
        self.lora_b.weight.val()
    }

    /// Number of trainable parameters (A + B only, base is frozen).
    pub fn trainable_param_count(&self) -> usize {
        let a_shape = self.lora_a.weight.val().shape();
        let b_shape = self.lora_b.weight.val().shape();
        a_shape.dims[0] * a_shape.dims[1] + b_shape.dims[0] * b_shape.dims[1]
    }

    /// Set whether the LoRA adapter is active.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

/// DPO loss computed on Burn tensors.
///
/// L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
pub fn dpo_loss<B: Backend>(
    chosen_logps: Tensor<B, 1>,       // Log-prob of chosen under policy
    rejected_logps: Tensor<B, 1>,     // Log-prob of rejected under policy
    ref_chosen_logps: Tensor<B, 1>,   // Log-prob of chosen under reference
    ref_rejected_logps: Tensor<B, 1>, // Log-prob of rejected under reference
    beta: f32,
) -> Tensor<B, 1> {
    let chosen_rewards = (chosen_logps - ref_chosen_logps).mul_scalar(beta);
    let rejected_rewards = (rejected_logps - ref_rejected_logps).mul_scalar(beta);
    let logits = chosen_rewards - rejected_rewards;

    // -log σ(logits) = log(1 + exp(-logits)) = softplus(-logits)
    let neg_logits = logits.neg();
    // softplus: log(1 + exp(x)) — numerically stable
    let loss = (neg_logits.clone().exp() + 1.0).log();
    loss.mean()
}

/// ORPO alignment loss computed on Burn tensors.
///
/// L_OR = -log σ(log(odds(chosen) / odds(rejected)))
/// where odds(p) = p / (1-p)
pub fn orpo_alignment_loss<B: Backend>(
    chosen_probs: Tensor<B, 1>,   // Average token probability for chosen
    rejected_probs: Tensor<B, 1>, // Average token probability for rejected
) -> Tensor<B, 1> {
    let eps: f32 = 1e-10;
    let one_minus_eps: f32 = 1.0 - eps;

    // Clamp probabilities to avoid log(0)
    let chosen_clamped = chosen_probs.clamp(eps, one_minus_eps);
    let rejected_clamped = rejected_probs.clamp(eps, one_minus_eps);

    // odds = p / (1 - p)
    let ones = Tensor::ones_like(&chosen_clamped);
    let chosen_odds = chosen_clamped.clone() / (ones.clone() - chosen_clamped);
    let rejected_odds = rejected_clamped.clone() / (ones - rejected_clamped);

    // log odds ratio
    let log_odds_ratio = (chosen_odds / rejected_odds).log();

    // -log σ(log_odds_ratio) = softplus(-log_odds_ratio)
    let neg_lor = log_odds_ratio.neg();
    let loss = (neg_lor.exp() + 1.0).log();
    loss.mean()
}

/// Full ORPO loss: SFT + lambda * alignment.
pub fn orpo_loss<B: Backend>(
    sft_loss: Tensor<B, 1>,
    chosen_probs: Tensor<B, 1>,
    rejected_probs: Tensor<B, 1>,
    lambda: f32,
) -> Tensor<B, 1> {
    let align = orpo_alignment_loss(chosen_probs, rejected_probs);
    sft_loss + align.mul_scalar(lambda)
}

/// Minimal transformer block as a Burn Module.
///
/// Components: RMSNorm → Attention (simplified) → Residual → RMSNorm → SwiGLU FFN → Residual
#[derive(Module, Debug)]
pub struct BurnTransformerBlock<B: Backend> {
    pre_norm: RmsNorm<B>,
    /// Simplified multi-head attention (Q/K/V projections + output).
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
    post_norm: RmsNorm<B>,
    ffn: SwiGluFfn<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

/// Configuration for a transformer block.
#[derive(Config, Debug)]
pub struct BurnTransformerBlockConfig {
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
}

impl BurnTransformerBlockConfig {
    /// Initialize a transformer block on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTransformerBlock<B> {
        let head_dim = self.hidden_size / self.num_heads;

        BurnTransformerBlock {
            pre_norm: RmsNormConfig::new(self.hidden_size).init(device),
            q_proj: nn::LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            k_proj: nn::LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            v_proj: nn::LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            o_proj: nn::LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            post_norm: RmsNormConfig::new(self.hidden_size).init(device),
            ffn: SwiGluFfnConfig::new(self.hidden_size, self.intermediate_size).init(device),
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

impl<B: Backend> BurnTransformerBlock<B> {
    /// Forward pass through the transformer block.
    ///
    /// Input: [batch_size, hidden_size] (single position, no sequence dim for simplicity).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Pre-norm + attention + residual
        let normed = self.pre_norm.forward(x.clone());
        let q = self.q_proj.forward(normed.clone());
        let k = self.k_proj.forward(normed.clone());
        let v = self.v_proj.forward(normed);

        // Simplified attention: softmax(Q·K^T / sqrt(d)) · V
        let scale = (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(k.transpose()).div_scalar(scale);
        let attn_weights = activation::softmax(attn_weights, 1);
        let attn_out = attn_weights.matmul(v);
        let attn_proj = self.o_proj.forward(attn_out);

        let h = x + attn_proj; // residual

        // Post-norm + FFN + residual
        let normed2 = self.post_norm.forward(h.clone());
        let ffn_out = self.ffn.forward(normed2);

        h + ffn_out // residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_lora_linear_forward() {
        let device = Default::default();
        let config = LoraLinearConfig::new(64, 128);
        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [4, 64],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);

        assert_eq!(output.dims(), [4, 128]);
    }

    #[test]
    fn test_lora_linear_zero_init() {
        let device = Default::default();
        let config = LoraLinearConfig::new(32, 32);
        let layer = config.init::<TestBackend>(&device);

        // With B zero-initialized, LoRA contribution should be zero
        // so output should equal base output
        let input = Tensor::<TestBackend, 2>::random(
            [2, 32],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let base_out = layer.base.forward(input.clone());
        let full_out = layer.forward(input);

        let diff = (full_out - base_out).abs().sum().into_scalar();
        assert!(
            diff < 1e-5,
            "LoRA should contribute zero initially, diff={}",
            diff
        );
    }

    #[test]
    fn test_lora_inactive() {
        let device = Default::default();
        let config = LoraLinearConfig::new(32, 32);
        let mut layer = config.init::<TestBackend>(&device);
        layer.set_active(false);

        let input = Tensor::<TestBackend, 2>::random(
            [2, 32],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let base_out = layer.base.forward(input.clone());
        let full_out = layer.forward(input);

        let diff = (full_out - base_out).abs().sum().into_scalar();
        assert!(diff < 1e-6, "Inactive LoRA should not contribute");
    }

    #[test]
    fn test_rms_norm() {
        let device = Default::default();
        let norm = RmsNormConfig::new(64).init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [4, 64],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = norm.forward(input);

        assert_eq!(output.dims(), [4, 64]);
    }

    #[test]
    fn test_swiglu_ffn() {
        let device = Default::default();
        let ffn = SwiGluFfnConfig::new(64, 128).init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [4, 64],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = ffn.forward(input);

        assert_eq!(output.dims(), [4, 64]);
    }

    #[test]
    fn test_trainable_params() {
        let device = Default::default();
        let config = LoraLinearConfig::new(4096, 4096).with_rank(16);
        let layer = config.init::<TestBackend>(&device);

        let params = layer.trainable_param_count();
        assert_eq!(params, 16 * 4096 + 4096 * 16); // A + B
    }

    // ── Phase 5 tests ──

    #[test]
    fn test_dora_forward() {
        let device = Default::default();
        let config = DoraLinearConfig::new(64, 128).with_rank(8);
        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [4, 64],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        assert_eq!(output.dims(), [4, 128]);
    }

    #[test]
    fn test_dora_trainable_params() {
        let device = Default::default();
        let config = DoraLinearConfig::new(256, 256).with_rank(16);
        let layer = config.init::<TestBackend>(&device);

        let params = layer.trainable_param_count();
        // A: 16*256 + B: 256*16 + magnitude: 256
        assert_eq!(params, 16 * 256 + 256 * 16 + 256);
    }

    #[test]
    fn test_dpo_loss_tensor() {
        let device = Default::default();
        let chosen = Tensor::<TestBackend, 1>::from_floats([-1.0, -0.5, -0.8], &device);
        let rejected = Tensor::<TestBackend, 1>::from_floats([-3.0, -2.5, -2.8], &device);
        let ref_chosen = Tensor::<TestBackend, 1>::from_floats([-1.5, -1.0, -1.2], &device);
        let ref_rejected = Tensor::<TestBackend, 1>::from_floats([-1.5, -1.0, -1.2], &device);

        let loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, 0.1);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "DPO loss should be positive, got {}", val);
        assert!(val < 5.0, "DPO loss should be reasonable, got {}", val);
    }

    #[test]
    fn test_dpo_loss_equal_logps() {
        let device = Default::default();
        // When chosen and rejected are equal, loss should be log(2)
        let logps = Tensor::<TestBackend, 1>::from_floats([-2.0], &device);
        let loss = dpo_loss(logps.clone(), logps.clone(), logps.clone(), logps, 0.1);
        let val: f32 = loss.into_scalar();
        assert!(
            (val - (2.0f32).ln()).abs() < 0.01,
            "Expected ~ln(2), got {}",
            val
        );
    }

    #[test]
    fn test_orpo_alignment_loss() {
        let device = Default::default();
        let chosen = Tensor::<TestBackend, 1>::from_floats([0.8, 0.7], &device);
        let rejected = Tensor::<TestBackend, 1>::from_floats([0.3, 0.2], &device);

        let loss = orpo_alignment_loss(chosen, rejected);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "ORPO alignment loss should be positive");
    }

    #[test]
    fn test_orpo_full_loss() {
        let device = Default::default();
        let sft = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
        let chosen = Tensor::<TestBackend, 1>::from_floats([0.7], &device);
        let rejected = Tensor::<TestBackend, 1>::from_floats([0.3], &device);

        let total = orpo_loss(sft, chosen, rejected, 0.5);
        let val: f32 = total.into_scalar();
        assert!(val > 2.0, "Total should be > SFT loss, got {}", val);
    }

    #[test]
    fn test_transformer_block() {
        let device = Default::default();
        let config = BurnTransformerBlockConfig::new(64, 4, 128);
        let block = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [8, 64],
            burn_core::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let output = block.forward(input);
        assert_eq!(
            output.dims(),
            [8, 64],
            "Transformer block should preserve shape"
        );
    }

    #[test]
    fn test_lora_init_with_base_weights() {
        let device = Default::default();
        let config = LoraLinearConfig::new(32, 64).with_rank(8);

        // Create known base weight
        let base_weight = Tensor::<TestBackend, 2>::random(
            [32, 64],
            burn_core::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let layer = config.init_with_base_weights::<TestBackend>(base_weight.clone(), &device);
        let input = Tensor::<TestBackend, 2>::random(
            [4, 32],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input.clone());
        assert_eq!(output.dims(), [4, 64]);

        // With zero-init B, output should match base @ input
        let expected = input.matmul(base_weight);
        let diff = (output - expected).abs().sum().into_scalar();
        assert!(
            diff < 1e-4,
            "LoRA with base weights should match base output initially, diff={}",
            diff
        );
    }

    #[test]
    fn test_dora_init_with_base_weights() {
        let device = Default::default();
        let config = DoraLinearConfig::new(32, 64).with_rank(8);

        let base_weight = Tensor::<TestBackend, 2>::random(
            [32, 64],
            burn_core::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let layer = config.init_with_base_weights::<TestBackend>(base_weight, &device);
        let input = Tensor::<TestBackend, 2>::random(
            [4, 32],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        assert_eq!(output.dims(), [4, 64]);
    }

    #[test]
    fn test_qlora_forward() {
        let device = Default::default();
        let config = QLoraLinearConfig::new(64, 128).with_rank(8);
        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random(
            [4, 64],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        assert_eq!(output.dims(), [4, 128]);
    }

    #[test]
    fn test_qlora_init_quantized() {
        let device = Default::default();
        let config = QLoraLinearConfig::new(16, 32).with_rank(4).with_bits(4);

        // Simulate dequantized weights
        let base_weights: Vec<f32> = (0..16 * 32).map(|i| (i as f32) * 0.001).collect();
        let layer = config.init_quantized::<TestBackend>(&base_weights, &device);

        let input = Tensor::<TestBackend, 2>::random(
            [2, 16],
            burn_core::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = layer.forward(input);
        assert_eq!(output.dims(), [2, 32]);
        assert_eq!(layer.trainable_param_count(), 4 * 16 + 32 * 4);
    }
}
