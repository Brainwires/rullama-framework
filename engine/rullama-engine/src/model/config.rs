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
    /// SWA layers' KV head count (`gemma4.attention.head_count_kv`). The 12b stores
    /// this as a per-layer array (8 on SWA layers, 1 on global) — parsed below.
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

    // ---- MoE (gemma4:26b-a4b; zero on dense variants) ----
    /// `gemma4.expert_count` — total routed experts per MoE layer. 0 ⇒ dense model.
    pub expert_count: u32,
    /// `gemma4.expert_used_count` — top-k experts selected per token.
    pub expert_used_count: u32,
    /// `gemma4.expert_feed_forward_length` — each expert's FFN intermediate size
    /// (704 on 26b-a4b vs 2112 for the parallel dense MLP).
    pub expert_ffn: u32,

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
            return Err(RullamaError::Config(format!(
                "expected architecture 'gemma4', got '{arch}'"
            )));
        }
        Self::from_gguf_with_prefix(r, "gemma4")
    }

    /// Parse the backbone config under an alternate metadata prefix.
    /// DiffusionGemma ships the IDENTICAL key set under `diffusion-gemma.*`
    /// — same per-layer arrays, same MoE fields. The caller owns the
    /// `general.architecture` guard.
    pub fn from_gguf_with_prefix(r: &GgufReader, prefix: &str) -> Result<Self> {
        let k = |s: &str| format!("{prefix}.{s}");

        let n_layers = r.get(&k("block_count"))?.as_u32()?;
        let d_model = r.get(&k("embedding_length"))?.as_u32()?;
        let max_pos = r.get(&k("context_length"))?.as_u32()?;

        // attention
        let n_heads = r.get(&k("attention.head_count"))?.as_u32()?;

        // Per-layer sliding-window vs global pattern — parsed before
        // head_count_kv because the 12b stores that as a per-layer array keyed
        // on this pattern.
        let pattern = r
            .get(&k("attention.sliding_window_pattern"))?
            .as_bool_array()?;
        if pattern.len() as u32 != n_layers {
            return Err(RullamaError::Config(format!(
                "sliding_window_pattern length {} != n_layers {}",
                pattern.len(),
                n_layers
            )));
        }
        let layer_kinds: Vec<LayerKind> = pattern
            .iter()
            .map(|&b| {
                if b {
                    LayerKind::SlidingWindow
                } else {
                    LayerKind::Global
                }
            })
            .collect();

        // KV head count. e2b/e4b store a scalar (uniform GQA). The 12b stores a
        // per-layer array ([8,8,8,8,8,1,…] — 8 KV heads on sliding-window
        // layers, 1 on global), which we collapse to the (swa, global) pair the
        // forward path keys on via `cfg.n_kv_heads(layer)`. Mirrors Ollama's
        // numGlobalKVHeads extraction in model_text.go.
        let hckv = r.get(&k("attention.head_count_kv"))?;
        let (n_kv_heads_swa, mut n_kv_heads_global) = match hckv {
            GgufValue::ArrayU32(_)
            | GgufValue::ArrayU64(_)
            | GgufValue::ArrayI32(_)
            | GgufValue::ArrayI64(_) => {
                let per_layer: Vec<u32> = match hckv {
                    GgufValue::ArrayU32(v) => v.clone(),
                    GgufValue::ArrayU64(v) => v.iter().map(|&x| x as u32).collect(),
                    GgufValue::ArrayI32(v) => v.iter().map(|&x| x as u32).collect(),
                    GgufValue::ArrayI64(v) => v.iter().map(|&x| x as u32).collect(),
                    _ => unreachable!(),
                };
                if per_layer.len() as u32 != n_layers {
                    return Err(RullamaError::Config(format!(
                        "head_count_kv array length {} != n_layers {}",
                        per_layer.len(),
                        n_layers
                    )));
                }
                let swa = layer_kinds
                    .iter()
                    .position(|k| matches!(k, LayerKind::SlidingWindow))
                    .map(|i| per_layer[i])
                    .unwrap_or(per_layer[0]);
                let glob = layer_kinds
                    .iter()
                    .position(|k| matches!(k, LayerKind::Global))
                    .map(|i| per_layer[i])
                    .unwrap_or(swa);
                (swa, glob)
            }
            scalar => {
                let s = scalar.as_u32()?;
                (s, s)
            }
        };
        // Optional explicit global-layer KV head override.
        if let Some(v) = r.get_opt(&k("attention.global_head_count_kv")) {
            n_kv_heads_global = v.as_u32()?;
        }

        let head_dim_global = r.get(&k("attention.key_length"))?.as_u32()?;
        let head_dim_swa = r.get(&k("attention.key_length_swa"))?.as_u32()?;
        let rms_norm_eps = r.get(&k("attention.layer_norm_rms_epsilon"))?.as_f32()?;
        let sliding_window = r.get(&k("attention.sliding_window"))?.as_u32()?;
        let shared_kv_layers = r
            .get_opt(&k("attention.shared_kv_layers"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);

        // FFN intermediate sizes: GGUF stores as either a scalar or a per-layer array.
        let ffn_inter: Vec<u32> = match r.get(&k("feed_forward_length"))? {
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
                "feed_forward_length array length {} != n_layers {}",
                ffn_inter.len(),
                n_layers
            )));
        }

        // RoPE
        let rope_freq_base = r.get(&k("rope.freq_base"))?.as_f32()?;
        let rope_freq_base_swa = r.get(&k("rope.freq_base_swa"))?.as_f32()?;
        let rope_dim_global = r
            .get_opt(&k("rope.dimension_count"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(head_dim_global / 4); // fallback: 25% partial rotation
        let rope_dim_swa = r
            .get_opt(&k("rope.dimension_count_swa"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(head_dim_swa); // fallback: full rotation

        // MoE (mirrors Ollama's c.Uint("expert_count", 0) — absent on dense models)
        let expert_count = r
            .get_opt(&k("expert_count"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);
        let expert_used_count = r
            .get_opt(&k("expert_used_count"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);
        let expert_ffn = r
            .get_opt(&k("expert_feed_forward_length"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);

        // output
        let final_logit_softcap = r.get(&k("final_logit_softcapping"))?.as_f32()?;
        let ple_dim = r
            .get_opt(&k("embedding_length_per_layer_input"))
            .map(|v| v.as_u32())
            .transpose()?
            .unwrap_or(0);

        // vocab
        let tokens = r.get("tokenizer.ggml.tokens")?.as_string_array()?;
        let vocab_size = tokens.len() as u32;
        let bos_id = r
            .get_opt("tokenizer.ggml.bos_token_id")
            .map(|v| v.as_u32())
            .transpose()?;
        let pad_id = r
            .get_opt("tokenizer.ggml.padding_token_id")
            .map(|v| v.as_u32())
            .transpose()?;
        let unk_id = r
            .get_opt("tokenizer.ggml.unknown_token_id")
            .map(|v| v.as_u32())
            .transpose()?;
        let eos_ids: Vec<u32> = match r.get_opt("tokenizer.ggml.eos_token_ids") {
            Some(v) => v.as_u32_array()?,
            None => match r.get_opt("tokenizer.ggml.eos_token_id") {
                Some(v) => vec![v.as_u32()?],
                None => Vec::new(),
            },
        };

        Ok(Self {
            n_layers,
            d_model,
            max_pos,
            n_heads,
            n_kv_heads_swa,
            n_kv_heads_global,
            head_dim_global,
            head_dim_swa,
            rms_norm_eps,
            sliding_window,
            layer_kinds,
            shared_kv_layers,
            ffn_inter,
            rope_freq_base,
            rope_freq_base_swa,
            rope_dim_global,
            rope_dim_swa,
            expert_count,
            expert_used_count,
            expert_ffn,
            final_logit_softcap,
            ple_dim,
            vocab_size,
            bos_id,
            eos_ids,
            pad_id,
            unk_id,
        })
    }

    /// True iff this checkpoint uses per-layer-input embeddings (E2B/E4B variants).
    pub fn has_ple(&self) -> bool {
        self.ple_dim > 0
    }

    /// True iff this checkpoint has MoE expert blocks (`gemma4:26b-a4b`).
    /// Which *layers* carry experts is decided by tensor presence
    /// (`blk.N.ffn_gate_inp.weight`), mirroring Ollama's nil-field checks.
    pub fn has_moe(&self) -> bool {
        self.expert_count > 0 && self.expert_used_count > 0
    }

    /// Layer kind for layer `i`.
    pub fn kind(&self, i: u32) -> LayerKind {
        self.layer_kinds[i as usize]
    }

    /// FFN intermediate size for layer `i`.
    pub fn ffn(&self, i: u32) -> u32 {
        self.ffn_inter[i as usize]
    }

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
