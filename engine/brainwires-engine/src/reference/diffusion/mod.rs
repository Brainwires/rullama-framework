//! DiffusionGemma — sibling engine for Google's block-diffusion text model
//! (`general.architecture = "diffusion-gemma"`).
//!
//! Same 26B-A4B sparse-MoE backbone as `gemma4:26b` (the MoE FFN code in
//! `reference/moe.rs` + `backend/dispatch/moe.rs` is shared), but ONE weight
//! set runs in two modes:
//!   - encoder: causal attention, WRITES the KV cache (prompt prefill + block
//!     commit; self-conditioning zeroed; per-layer `enc_layer_output_scale`)
//!   - decoder: bidirectional attention over the 256-token canvas, READS the
//!     prefix KV (denoise steps; per-layer `layer_output_scale`)
//! plus a self-conditioning gated MLP (`self_cond_{pre_norm,gate,up,down}`)
//! that feeds softmax(prev step's canvas logits / prev t) — converted to a
//! probability-weighted embedding average — back onto the canvas embeddings.
//!
//! Parity oracle: llama.cpp PR 24423 (`llama-diffusion-cli`); Ollama cannot
//! run this architecture. The sampler below mirrors that runner's
//! entropy-bound denoiser exactly; the forward graph mirrors
//! `src/models/diffusion-gemma.cpp`.

pub mod mask;
pub mod sampler;

use crate::error::{Result, RullamaError};
use crate::gguf::GgufReader;
use crate::model::config::Gemma4Config;

/// DiffusionGemma model configuration: the shared Gemma-4 backbone keys
/// (parsed under the `diffusion-gemma.` prefix) plus the diffusion extras.
#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    /// The transformer backbone — identical key set to `gemma4.*`, including
    /// the MoE expert fields.
    pub base: Gemma4Config,
    /// `tokenizer.ggml.mask_token_id` (4 on the released checkpoints). Unused
    /// by the entropy-bound sampler (random canvas init) but part of the
    /// model's vocab contract.
    pub mask_token_id: Option<u32>,
}

impl DiffusionConfig {
    pub fn from_gguf(r: &GgufReader) -> Result<Self> {
        let arch = r.get("general.architecture")?.as_str()?;
        if arch != "diffusion-gemma" {
            return Err(RullamaError::Config(format!(
                "expected architecture 'diffusion-gemma', got '{arch}'"
            )));
        }
        let base = Gemma4Config::from_gguf_with_prefix(r, "diffusion-gemma")?;
        let mask_token_id = r
            .get_opt("tokenizer.ggml.mask_token_id")
            .map(|v| v.as_u32())
            .transpose()?;
        Ok(Self {
            base,
            mask_token_id,
        })
    }
}
