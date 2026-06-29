//! Pure-Rust f32 oracle for EmbeddingGemma-300M (architecture `gemma3`).
//!
//! EmbeddingGemma is an *encoder-only* Gemma 3: the same RMSNorm / GeGLU /
//! RoPE / QK-norm stack rullama already runs for Gemma 4, but with
//! **bidirectional** attention (`gemma3.attention.causal = false`), no PLE,
//! no KV cache, and a pooling + dense projection head instead of an LM head.
//!
//! Parity target: `llama.cpp`'s `gemma3` embedding graph (the same GGUF, via
//! Ollama's `/api/embed`). Output is a mean-pooled, dense-projected,
//! L2-normalized 768-d vector with Matryoshka truncation support.
//!
//! Reuses the scalar ops in [`crate::reference::ops`] (`rmsnorm`, `geglu_split`,
//! `rope_neox`, `matvec`, `softmax`, `add_into`, `scale`) — identical math to
//! the Gemma 4 oracle, just arranged for a full-sequence bidirectional pass.
#![allow(dead_code)]

pub mod forward;
pub mod gpu;

use std::sync::Arc;

use crate::error::{Result, RullamaError};
use crate::gguf::GgufReader;
use crate::reference::weights::Weights;

/// Per-layer attention kind. SWA layers use a (symmetric, for the encoder)
/// sliding-window mask; Global layers attend across the whole sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    SlidingWindow,
    Global,
}

/// Mean / CLS / last-token pooling over the final hidden states.
/// Matches llama.cpp's `LLAMA_POOLING_TYPE_*` (0=none,1=mean,2=cls,3=last).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingType {
    None,
    Mean,
    Cls,
    Last,
}

/// Parsed `gemma3.*` GGUF metadata for EmbeddingGemma.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    pub n_layers: u32,
    pub d_model: u32,
    pub context_length: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub sliding_window: u32,
    pub layer_kinds: Vec<LayerKind>,
    /// `gemma3.attention.causal` — false for the embedding encoder.
    pub causal: bool,
    pub pooling: PoolingType,
    pub vocab_size: u32,
    /// Output (final-projection) embedding dim after the dense head. 768 for
    /// EmbeddingGemma; Matryoshka can truncate this at inference.
    pub embed_dim: u32,
}

impl EmbedConfig {
    pub fn from_gguf(r: &GgufReader) -> Result<Self> {
        let arch = r.get("general.architecture")?.as_str()?;
        if arch != "gemma3" {
            return Err(RullamaError::Config(format!(
                "embed: expected architecture 'gemma3', got '{arch}'"
            )));
        }
        let n_layers = r.get("gemma3.block_count")?.as_u32()?;
        let d_model = r.get("gemma3.embedding_length")?.as_u32()?;
        let context_length = r.get("gemma3.context_length")?.as_u32()?;
        let n_heads = r.get("gemma3.attention.head_count")?.as_u32()?;
        let n_kv_heads = r.get("gemma3.attention.head_count_kv")?.as_u32()?;
        let head_dim = r.get("gemma3.attention.key_length")?.as_u32()?;
        let ffn = r.get("gemma3.feed_forward_length")?.as_u32()?;
        let rms_eps = r.get("gemma3.attention.layer_norm_rms_epsilon")?.as_f32()?;
        let rope_base = r.get("gemma3.rope.freq_base")?.as_f32()?;
        let sliding_window = r.get("gemma3.attention.sliding_window")?.as_u32()?;
        let causal = r
            .get("gemma3.attention.causal")
            .ok()
            .and_then(|v| v.as_bool().ok())
            .unwrap_or(false);

        // Per-layer SWA/global pattern. Some GGUFs publish a bool array the
        // length of n_layers; if absent, fall back to all-global.
        let layer_kinds: Vec<LayerKind> = match r.get("gemma3.attention.sliding_window_pattern") {
            Ok(v) => {
                let pattern = v.as_bool_array()?;
                if pattern.len() != n_layers as usize {
                    return Err(RullamaError::Config(format!(
                        "embed: sliding_window_pattern length {} != n_layers {}",
                        pattern.len(),
                        n_layers
                    )));
                }
                pattern
                    .iter()
                    .map(|&swa| {
                        if swa {
                            LayerKind::SlidingWindow
                        } else {
                            LayerKind::Global
                        }
                    })
                    .collect()
            }
            Err(_) => vec![LayerKind::Global; n_layers as usize],
        };

        let pooling = match r
            .get("gemma3.pooling_type")
            .ok()
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(1)
        {
            0 => PoolingType::None,
            2 => PoolingType::Cls,
            3 => PoolingType::Last,
            _ => PoolingType::Mean,
        };

        // vocab from the token_embd tensor (col count) — same as Gemma path.
        let vocab_size = r
            .tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| *t.dims.last().unwrap_or(&0) as u32)
            .unwrap_or(0);

        Ok(EmbedConfig {
            n_layers,
            d_model,
            context_length,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn,
            rms_eps,
            rope_base,
            sliding_window,
            layer_kinds,
            causal,
            pooling,
            vocab_size,
            embed_dim: d_model, // dense head preserves 768; Matryoshka truncates later
        })
    }

    pub fn kind(&self, layer: u32) -> LayerKind {
        self.layer_kinds[layer as usize]
    }
}

/// EmbeddingGemma model: parsed config + weight accessor over the GGUF.
pub struct EmbedModel {
    pub cfg: EmbedConfig,
    pub weights: Weights,
}

impl EmbedModel {
    pub fn new(reader: Arc<GgufReader>) -> Result<Self> {
        let cfg = EmbedConfig::from_gguf(&reader)?;
        let weights = Weights::new(reader);
        Ok(Self { cfg, weights })
    }

    fn t(&self, name: &str) -> Result<Vec<f32>> {
        self.weights.load(name)
    }
}
