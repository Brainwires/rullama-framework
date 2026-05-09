//! Gemma 4 model config, parsed from `gemma4.*` GGUF metadata keys.
//!
//! Source of truth: `convert/convert_gemma4.go` and `model/models/gemma4/model_text.go`
//! in the Ollama reference impl at /Users/nightness/Source/ollama.

use crate::error::{Result, RullamaError};
use crate::gguf::{GgufReader, GgufValue};

/// Whether a layer uses sliding-window attention (true) or global causal attention (false).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    SlidingWindow,
    Global,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    /// `gemma4.block_count` — total number of transformer layers.
    pub n_layers: u32,
    /// `gemma4.embedding_length` — hidden / residual stream width.
    pub d_model: u32,
    /// `gemma4.context_length` — max position id.
    pub max_pos: u32,

    // ---- attention ----
    pub n_heads: u32,
    /// SWA layers' KV head count (`gemma4.attention.head_count_kv`). Some GGUFs use a
    /// per-layer array; for now we accept the scalar form.
    pub n_kv_heads_swa: u32,
    /// Global layers' KV head count, if explicitly differentiated. Falls back to
    /// `n_kv_heads_swa` when the optional key is absent.
    pub n_kv_heads_global: u32,
    /// Per-head dimension on global layers (`gemma4.attention.key_length`).
    pub head_dim_global: u32,
    /// Per-head dimension on SWA layers (`gemma4.attention.key_length_swa`).
    pub head_dim_swa: u32,
    pub rms_norm_eps: f32,
    pub sliding_window: u32,
    /// Per-layer kind from `gemma4.attention.sliding_window_pattern` (length = n_layers).
    pub layer_kinds: Vec<LayerKind>,
    /// Number of trailing layers that share KV with an earlier donor layer of the
    /// same kind (`gemma4.attention.shared_kv_layers`).
    pub shared_kv_layers: u32,

    // ---- MLP ----
    /// Per-layer FFN intermediate size (`gemma4.feed_forward_length`). Always length
    /// = n_layers; some Gemma 4 variants use double-wide MLP on KV-shared layers.
    pub ffn_inter: Vec<u32>,

    // ---- RoPE ----
    /// Global layer RoPE base (`gemma4.rope.freq_base`).
    pub rope_freq_base: f32,
    /// SWA layer RoPE base (`gemma4.rope.freq_base_swa`).
    pub rope_freq_base_swa: f32,
    /// Number of dimensions rotated by RoPE on global layers
    /// (`gemma4.rope.dimension_count`). Less than head_dim_global → partial rotation.
    pub rope_dim_global: u32,
    /// Number of dimensions rotated by RoPE on SWA layers (full rotation:
    /// `rope_dim_swa == head_dim_swa`).
    pub rope_dim_swa: u32,

    // ---- output ----
    /// `gemma4.final_logit_softcapping` (typically 30.0).
    pub final_logit_softcap: f32,
    /// Per-layer-input embedding width (`gemma4.embedding_length_per_layer_input`).
    /// Zero ⇒ PLE disabled (large variants); non-zero ⇒ PLE enabled (E2B/E4B).
    pub ple_dim: u32,

    // ---- vocab ----
    pub vocab_size: u32,
    pub bos_id: Option<u32>,
    pub eos_ids: Vec<u32>,
    pub pad_id: Option<u32>,
    pub unk_id: Option<u32>,
}

impl Gemma4Config {
    pub fn from_gguf(r: &GgufReader) -> Result<Self> {
        let arch = r.get("general.architecture")?.as_str()?;
        if arch != "gemma4" {
            return Err(RullamaError::Config(format!("expected architecture 'gemma4', got '{arch}'")));
        }

        let n_layers = r.get("gemma4.block_count")?.as_u32()?;
        let d_model = r.get("gemma4.embedding_length")?.as_u32()?;
        let max_pos = r.get("gemma4.context_length")?.as_u32()?;

        // attention
        let n_heads = r.get("gemma4.attention.head_count")?.as_u32()?;
        let n_kv_heads_swa = r.get("gemma4.attention.head_count_kv")?.as_u32()?;
        let n_kv_heads_global = r
            .get_opt("gemma4.attention.global_head_count_kv")
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(n_kv_heads_swa);
        let head_dim_global = r.get("gemma4.attention.key_length")?.as_u32()?;
        let head_dim_swa = r.get("gemma4.attention.key_length_swa")?.as_u32()?;
        let rms_norm_eps = r.get("gemma4.attention.layer_norm_rms_epsilon")?.as_f32()?;
        let sliding_window = r.get("gemma4.attention.sliding_window")?.as_u32()?;
        let shared_kv_layers = r
            .get_opt("gemma4.attention.shared_kv_layers")
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);

        let pattern = r.get("gemma4.attention.sliding_window_pattern")?.as_bool_array()?;
        if pattern.len() as u32 != n_layers {
            return Err(RullamaError::Config(format!(
                "sliding_window_pattern length {} != n_layers {}",
                pattern.len(), n_layers
            )));
        }
        let layer_kinds: Vec<LayerKind> = pattern.iter().map(|&b| {
            if b { LayerKind::SlidingWindow } else { LayerKind::Global }
        }).collect();

        // FFN intermediate sizes: GGUF stores as either a scalar or a per-layer array.
        let ffn_inter: Vec<u32> = match r.get("gemma4.feed_forward_length")? {
            GgufValue::ArrayU32(v) => v.clone(),
            GgufValue::ArrayU64(v) => v.iter().map(|&x| x as u32).collect(),
            GgufValue::ArrayI32(v) => v.iter().map(|&x| x as u32).collect(),
            GgufValue::ArrayI64(v) => v.iter().map(|&x| x as u32).collect(),
            scalar => {
                let s = scalar.as_u32()?;
                vec![s; n_layers as usize]
            }
        };
        if ffn_inter.len() as u32 != n_layers {
            return Err(RullamaError::Config(format!(
                "feed_forward_length array length {} != n_layers {}", ffn_inter.len(), n_layers
            )));
        }

        // RoPE
        let rope_freq_base = r.get("gemma4.rope.freq_base")?.as_f32()?;
        let rope_freq_base_swa = r.get("gemma4.rope.freq_base_swa")?.as_f32()?;
        let rope_dim_global = r
            .get_opt("gemma4.rope.dimension_count")
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(head_dim_global / 4); // fallback: 25% partial rotation
        let rope_dim_swa = r
            .get_opt("gemma4.rope.dimension_count_swa")
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(head_dim_swa); // fallback: full rotation

        // output
        let final_logit_softcap = r.get("gemma4.final_logit_softcapping")?.as_f32()?;
        let ple_dim = r
            .get_opt("gemma4.embedding_length_per_layer_input")
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);

        // vocab
        let tokens = r.get("tokenizer.ggml.tokens")?.as_string_array()?;
        let vocab_size = tokens.len() as u32;
        let bos_id = r.get_opt("tokenizer.ggml.bos_token_id").map(|v| v.as_u32()).transpose()?;
        let pad_id = r.get_opt("tokenizer.ggml.padding_token_id").map(|v| v.as_u32()).transpose()?;
        let unk_id = r.get_opt("tokenizer.ggml.unknown_token_id").map(|v| v.as_u32()).transpose()?;
        let eos_ids: Vec<u32> = match r.get_opt("tokenizer.ggml.eos_token_ids") {
            Some(v) => v.as_u32_array()?,
            None => match r.get_opt("tokenizer.ggml.eos_token_id") {
                Some(v) => vec![v.as_u32()?],
                None => Vec::new(),
            },
        };

        Ok(Self {
            n_layers, d_model, max_pos,
            n_heads, n_kv_heads_swa, n_kv_heads_global,
            head_dim_global, head_dim_swa,
            rms_norm_eps, sliding_window,
            layer_kinds,
            shared_kv_layers,
            ffn_inter,
            rope_freq_base, rope_freq_base_swa,
            rope_dim_global, rope_dim_swa,
            final_logit_softcap,
            ple_dim,
            vocab_size,
            bos_id, eos_ids, pad_id, unk_id,
        })
    }

    /// True iff this checkpoint uses per-layer-input embeddings (E2B/E4B variants).
    pub fn has_ple(&self) -> bool { self.ple_dim > 0 }

    /// Layer kind for layer `i`.
    pub fn kind(&self, i: u32) -> LayerKind { self.layer_kinds[i as usize] }

    /// FFN intermediate size for layer `i`.
    pub fn ffn(&self, i: u32) -> u32 { self.ffn_inter[i as usize] }

    /// Number of KV heads on layer `i`, depending on its kind.
    pub fn n_kv_heads(&self, i: u32) -> u32 {
        match self.kind(i) {
            LayerKind::SlidingWindow => self.n_kv_heads_swa,
            LayerKind::Global => self.n_kv_heads_global,
        }
    }

    /// Per-head dimension on layer `i`.
    pub fn head_dim(&self, i: u32) -> u32 {
        match self.kind(i) {
            LayerKind::SlidingWindow => self.head_dim_swa,
            LayerKind::Global => self.head_dim_global,
        }
    }
}
