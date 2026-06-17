//! Z-Image-Turbo configuration, parsed from the diffusers component
//! `config.json` files (`transformer/`, `text_encoder/`, `vae/`, `scheduler/`).
//!
//! Values are the real Tongyi-MAI/Z-Image-Turbo configs (not the Ollama Go
//! struct zero-values, which are placeholders). The test fixtures below are the
//! verbatim upstream JSON so the parser is pinned to ground truth.

use serde::Deserialize;

use crate::error::{Result, RullamaError};

fn parse<T: for<'de> Deserialize<'de>>(bytes: &[u8], what: &str) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| RullamaError::Image(format!("{what} config: {e}")))
}

/// `transformer/config.json` — the single-stream S3-DiT denoiser.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    /// Model hidden dim (3840).
    pub dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub n_layers: u32,
    pub n_refiner_layers: u32,
    /// Latent channels in/out of the DiT (16).
    pub in_channels: u32,
    /// Text-encoder feature dim the caption embedder consumes (2560 = Qwen3 hidden).
    pub cap_feat_dim: u32,
    /// Per-axis RoPE dims; sum == head_dim (e.g. [32,48,48] → 128).
    pub axes_dims: Vec<u32>,
    pub axes_lens: Vec<u32>,
    #[serde(default = "one_vec")]
    pub all_patch_size: Vec<u32>,
    #[serde(default = "one_vec")]
    pub all_f_patch_size: Vec<u32>,
    pub norm_eps: f32,
    pub qk_norm: bool,
    pub rope_theta: f32,
    pub t_scale: f32,
}

fn one_vec() -> Vec<u32> {
    vec![1]
}

impl TransformerConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let c: Self = parse(bytes, "transformer")?;
        let hd = c.dim / c.n_heads;
        let axes_sum: u32 = c.axes_dims.iter().sum();
        if axes_sum != hd {
            return Err(RullamaError::Image(format!(
                "axes_dims sum {axes_sum} != head_dim {hd} (dim {} / n_heads {})",
                c.dim, c.n_heads
            )));
        }
        Ok(c)
    }
    /// Per-head dim = dim / n_heads.
    pub fn head_dim(&self) -> u32 {
        self.dim / self.n_heads
    }
    /// Spatial patch size (latent → DiT tokens).
    pub fn patch_size(&self) -> u32 {
        *self.all_patch_size.first().unwrap_or(&1)
    }
}

/// `text_encoder/config.json` — Qwen3 (the prompt encoder).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
}

impl Qwen3Config {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        parse(bytes, "text_encoder")
    }
    /// Total query projection width = num_attention_heads * head_dim.
    pub fn q_dim(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }
    /// KV projection width = num_key_value_heads * head_dim (GQA).
    pub fn kv_dim(&self) -> u32 {
        self.num_key_value_heads * self.head_dim
    }
}

/// `vae/config.json` — AutoencoderKL (flux-dev VAE), decoder side.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub block_out_channels: Vec<u32>,
    pub latent_channels: u32,
    pub layers_per_block: u32,
    pub norm_num_groups: u32,
    #[serde(default)]
    pub mid_block_add_attention: bool,
    pub in_channels: u32,
    pub out_channels: u32,
    pub scaling_factor: f32,
    #[serde(default)]
    pub shift_factor: f32,
}

impl VaeConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        parse(bytes, "vae")
    }
    /// Spatial downscale factor = 2^(num_down_blocks - 1) (first block keeps
    /// resolution). For the 4-block flux VAE this is 8×.
    pub fn downscale(&self) -> u32 {
        1 << (self.block_out_channels.len().saturating_sub(1) as u32)
    }
}

/// `scheduler/scheduler_config.json` — FlowMatchEulerDiscreteScheduler.
#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    pub num_train_timesteps: u32,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "default_shift")]
    pub shift: f32,
}

fn default_shift() -> f32 {
    1.0
}

impl SchedulerConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        parse(bytes, "scheduler")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim Tongyi-MAI/Z-Image-Turbo upstream configs (ground truth).
    const TRANSFORMER: &str = r#"{
      "_class_name": "ZImageTransformer2DModel", "all_f_patch_size": [1],
      "all_patch_size": [2], "axes_dims": [32, 48, 48], "axes_lens": [1536, 512, 512],
      "cap_feat_dim": 2560, "dim": 3840, "in_channels": 16, "n_heads": 30,
      "n_kv_heads": 30, "n_layers": 30, "n_refiner_layers": 2, "norm_eps": 1e-05,
      "qk_norm": true, "rope_theta": 256.0, "t_scale": 1000.0 }"#;

    const TEXT_ENCODER: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "head_dim": 128,
      "hidden_act": "silu", "hidden_size": 2560, "intermediate_size": 9728,
      "max_position_embeddings": 40960, "model_type": "qwen3",
      "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06, "rope_theta": 1000000, "tie_word_embeddings": true,
      "torch_dtype": "bfloat16", "vocab_size": 151936 }"#;

    const VAE: &str = r#"{
      "_class_name": "AutoencoderKL", "act_fn": "silu",
      "block_out_channels": [128, 256, 512, 512], "force_upcast": true,
      "in_channels": 3, "latent_channels": 16, "layers_per_block": 2,
      "mid_block_add_attention": true, "norm_num_groups": 32, "out_channels": 3,
      "sample_size": 1024, "scaling_factor": 0.3611, "shift_factor": 0.1159,
      "use_post_quant_conv": false, "use_quant_conv": false }"#;

    const SCHEDULER: &str = r#"{
      "_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000,
      "use_dynamic_shifting": false, "shift": 3.0 }"#;

    #[test]
    fn transformer_config_real() {
        let c = TransformerConfig::parse(TRANSFORMER.as_bytes()).unwrap();
        assert_eq!(c.dim, 3840);
        assert_eq!(c.n_layers, 30);
        assert_eq!(c.head_dim(), 128); // 3840 / 30
        assert_eq!(c.axes_dims.iter().sum::<u32>(), 128); // RoPE split fills head_dim
        assert_eq!(c.cap_feat_dim, 2560);
        assert_eq!(c.in_channels, 16);
        assert_eq!(c.patch_size(), 2);
        assert!(c.qk_norm);
    }

    #[test]
    fn qwen3_config_real_gqa() {
        let c = Qwen3Config::parse(TEXT_ENCODER.as_bytes()).unwrap();
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(c.num_hidden_layers, 36);
        assert_eq!(c.q_dim(), 32 * 128); // 4096
        assert_eq!(c.kv_dim(), 8 * 128); // 1024 (GQA)
        assert!(c.tie_word_embeddings);
        assert!(!c.attention_bias);
        // encoder output width matches the DiT's caption feature dim
        assert_eq!(
            c.hidden_size,
            TransformerConfig::parse(TRANSFORMER.as_bytes())
                .unwrap()
                .cap_feat_dim
        );
    }

    #[test]
    fn vae_config_real() {
        let c = VaeConfig::parse(VAE.as_bytes()).unwrap();
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(c.latent_channels, 16);
        assert_eq!(c.norm_num_groups, 32);
        assert_eq!(c.downscale(), 8); // 2^3
        assert!(c.mid_block_add_attention);
        assert!((c.scaling_factor - 0.3611).abs() < 1e-6);
        assert!((c.shift_factor - 0.1159).abs() < 1e-6);
    }

    #[test]
    fn scheduler_config_real_static_shift() {
        let c = SchedulerConfig::parse(SCHEDULER.as_bytes()).unwrap();
        assert_eq!(c.num_train_timesteps, 1000);
        assert!(!c.use_dynamic_shifting); // ⇒ STATIC shift path
        assert!((c.shift - 3.0).abs() < 1e-6);
    }

    #[test]
    fn transformer_rejects_bad_axes() {
        let bad = TRANSFORMER.replace("[32, 48, 48]", "[32, 48, 64]");
        assert!(TransformerConfig::parse(bad.as_bytes()).is_err());
    }
}
