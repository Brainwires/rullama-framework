//! CPU f32 forward pass for Gemma 4. Mirrors the Go implementation at
//! `/Users/nightness/Source/ollama/model/models/gemma4/model_text.go`.
//!
//! Single-batch, single-token-at-a-time API. Supports:
//!   - Hybrid SWA / global layers
//!   - Per-layer GQA + per-layer head_dim
//!   - Q-norm, K-norm (weighted RMSNorm), V-norm (unweighted RMSNorm)
//!   - NeoX RoPE with per-layer base; freq_factors on global layers
//!   - KV cache history with sliding-window mask on SWA layers
//!   - Cross-layer KV sharing (donor layers)
//!   - Pre+post norm sandwich (attn and MLP)
//!   - GeGLU MLP
//!   - Per-Layer Embeddings (PLE) injection
//!   - Per-layer output scalar
//!   - Final logit softcap, tied output embedding
//!
//! Not supported (out of v1 scope or out of E2B model coverage):
//!   - Multimodal (vision/audio)
//!   - MoE expert routing
//!   - K=V optimization (we always read the V projection)

use crate::error::{Result, RullamaError};
use crate::model::config::{Gemma4Config, LayerKind};
use super::ops::{add_into, geglu_split, matvec, rmsnorm, rope_neox, scale, softcap, softmax};
use super::weights::Weights;

/// Per-layer KV history. Each `k`/`v` is a flattened `[n_kv_heads, head_dim, pos+1]`
/// tensor stored as a `Vec<f32>` with positions concatenated. Layout chosen to make
/// position-major slicing cheap (`k_at(pos, kv_head)` is contiguous).
#[derive(Default, Clone)]
pub struct LayerKv {
    pub k: Vec<f32>,            // length = (pos+1) * n_kv_heads * head_dim
    pub v: Vec<f32>,            // same shape
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

/// Per-layer KV histories across the whole model.
#[derive(Default)]
pub struct KvState {
    pub layers: Vec<LayerKv>,
}

impl KvState {
    pub fn new(cfg: &Gemma4Config) -> Self {
        Self {
            layers: (0..cfg.n_layers).map(|i| LayerKv {
                n_kv_heads: cfg.n_kv_heads(i),
                head_dim: cfg.head_dim(i),
                ..Default::default()
            }).collect(),
        }
    }
}

/// Public re-export so the GPU forward path can reuse the donor-map logic.
pub fn build_donor_map_pub(cfg: &Gemma4Config) -> Vec<Option<u32>> {
    build_donor_map(cfg)
}

/// Compute donor layer index for KV-shared layers.
///
/// Mirrors the logic in `newTextModel` (model_text.go:127-141): the last
/// `shared_kv_layers` of the model reuse K/V from the most-recent earlier
/// non-shared layer of the same kind (SWA or global).
fn build_donor_map(cfg: &Gemma4Config) -> Vec<Option<u32>> {
    let mut donor = vec![None; cfg.n_layers as usize];
    if cfg.shared_kv_layers == 0 { return donor; }
    let first_shared = cfg.n_layers - cfg.shared_kv_layers;
    for i in first_shared..cfg.n_layers {
        let kind = cfg.kind(i);
        // Find last non-shared layer of same kind.
        let mut j = first_shared as i64 - 1;
        while j >= 0 {
            if cfg.kind(j as u32) == kind {
                donor[i as usize] = Some(j as u32);
                break;
            }
            j -= 1;
        }
    }
    donor
}

/// Run one forward step at `pos` for the single token `token_id`. Returns logits over
/// the full vocab, with the final softcap applied. Mutates `kv_state` to append the
/// new K/V for each non-shared layer.
pub fn forward_token(
    cfg: &Gemma4Config,
    weights: &Weights,
    kv_state: &mut KvState,
    token_id: u32,
    pos: u32,
) -> Result<Vec<f32>> {
    if (token_id as u64) >= cfg.vocab_size as u64 {
        return Err(RullamaError::Inference(format!(
            "token_id {token_id} >= vocab_size {}", cfg.vocab_size
        )));
    }
    let d_model = cfg.d_model as usize;
    let donor_map = build_donor_map(cfg);

    // ---- token embedding ----
    // token_embd is stored [d_model, vocab] in GGUF; the embed for a single token is
    // one row of length d_model. Then scale by sqrt(d_model).
    let mut hidden = weights.load_row("token_embd.weight", token_id as usize)?;
    if hidden.len() != d_model {
        return Err(RullamaError::Inference(format!(
            "token_embd row length {} != d_model {}", hidden.len(), d_model
        )));
    }
    scale(&mut hidden, (d_model as f32).sqrt());

    // ---- per-layer inputs (PLE), if enabled ----
    let per_layer_inputs: Option<Vec<Vec<f32>>> = if cfg.has_ple() {
        Some(prepare_per_layer_inputs(cfg, weights, &hidden, token_id)?)
    } else {
        None
    };

    // ---- transformer layers ----
    for i in 0..cfg.n_layers {
        let pli = per_layer_inputs.as_ref().map(|all| all[i as usize].as_slice());
        layer_forward(cfg, weights, kv_state, &donor_map, i, pos, &mut hidden, pli)?;
    }

    // ---- final norm ----
    let final_norm = weights.load("output_norm.weight")?;
    let mut x = vec![0f32; d_model];
    rmsnorm(&hidden, Some(&final_norm), cfg.rms_norm_eps, &mut x);
    drop(final_norm);

    // ---- output projection (tied to token_embd) ----
    // token_embd is [d_model, vocab]. logits[v] = Σ_i x[i] * token_embd[v*d_model + i].
    // We compute one row at a time to avoid materializing the full vocab × d_model
    // table in f32 memory (1.5 GB).
    let mut logits = vec![0f32; cfg.vocab_size as usize];
    for v in 0..cfg.vocab_size as usize {
        let row = weights.load_row("token_embd.weight", v)?;
        let mut acc = 0f32;
        for k_i in 0..d_model {
            acc += x[k_i] * row[k_i];
        }
        logits[v] = acc;
    }

    // ---- softcap ----
    softcap(&mut logits, cfg.final_logit_softcap);
    Ok(logits)
}

/// Build per-layer input slices for PLE. Output length = n_layers; each slice has
/// length ple_dim. Mirrors `PerLayerProjector.Forward`.
fn prepare_per_layer_inputs(
    cfg: &Gemma4Config,
    weights: &Weights,
    hidden: &[f32],
    token_id: u32,
) -> Result<Vec<Vec<f32>>> {
    let ple_dim = cfg.ple_dim as usize;
    let n_layers = cfg.n_layers as usize;
    let d_model = cfg.d_model as usize;

    // (1) inputsPerLayer: row `token_id` of per_layer_token_embd, shape [n_layers*ple_dim].
    //     Then scale by sqrt(ple_dim), reshape to [ple_dim, n_layers].
    let mut inputs_per_layer = weights.load_row("per_layer_token_embd.weight", token_id as usize)?;
    if inputs_per_layer.len() != n_layers * ple_dim {
        return Err(RullamaError::Inference(format!(
            "per_layer_token_embd row length {} != n_layers*ple_dim {}",
            inputs_per_layer.len(), n_layers * ple_dim
        )));
    }
    scale(&mut inputs_per_layer, (ple_dim as f32).sqrt());

    // (2) perLayerProjection: project hidden to [n_layers*ple_dim] via per_layer_model_proj
    //     (Q4_K weight [d_model, n_layers*ple_dim]).
    let proj_w = weights.load("per_layer_model_proj.weight")?;
    let mut projection = vec![0f32; n_layers * ple_dim];
    matvec(&proj_w, d_model, n_layers * ple_dim, hidden, &mut projection);
    drop(proj_w);
    scale(&mut projection, 1.0 / (d_model as f32).sqrt());

    // (3) RMSNorm projection per layer slice (norm weight is shape [ple_dim], applied
    //     along the ple_dim axis for each layer).
    let proj_norm_w = weights.load("per_layer_proj_norm.weight")?;
    let mut normed = vec![0f32; n_layers * ple_dim];
    for layer in 0..n_layers {
        let off = layer * ple_dim;
        let in_slice = &projection[off..off + ple_dim];
        let out_slice = &mut normed[off..off + ple_dim];
        rmsnorm(in_slice, Some(&proj_norm_w), cfg.rms_norm_eps, out_slice);
    }
    drop(proj_norm_w);
    drop(projection);

    // (4) Add inputsPerLayer and divide by sqrt(2).
    add_into(&mut normed, &inputs_per_layer);
    scale(&mut normed, 1.0 / 2.0_f32.sqrt());

    // (5) Slice per layer.
    Ok((0..n_layers)
        .map(|layer| normed[layer * ple_dim..(layer + 1) * ple_dim].to_vec())
        .collect())
}

/// Run one transformer layer in-place on `hidden`. Updates `kv_state[i]` with the new
/// K/V for non-shared layers. Mirrors `TextLayer.Forward`.
fn layer_forward(
    cfg: &Gemma4Config,
    weights: &Weights,
    kv_state: &mut KvState,
    donor_map: &[Option<u32>],
    i: u32,
    pos: u32,
    hidden: &mut Vec<f32>,
    per_layer_input: Option<&[f32]>,
) -> Result<()> {
    let d_model = cfg.d_model as usize;
    let eps = cfg.rms_norm_eps;
    let prefix = format!("blk.{i}.");

    // ===== ATTENTION BLOCK =====
    let residual = hidden.clone();

    // pre-attn norm
    let attn_norm_w = weights.load(&format!("{prefix}attn_norm.weight"))?;
    let mut x = vec![0f32; d_model];
    rmsnorm(hidden, Some(&attn_norm_w), eps, &mut x);
    drop(attn_norm_w);

    // self-attention
    let attn_out = self_attention(cfg, weights, kv_state, donor_map, i, pos, &x)?;

    // post-attn norm + residual
    let post_attn_w = weights.load(&format!("{prefix}post_attention_norm.weight"))?;
    let mut h2 = vec![0f32; d_model];
    rmsnorm(&attn_out, Some(&post_attn_w), eps, &mut h2);
    drop(post_attn_w);
    add_into(&mut h2, &residual);
    *hidden = h2;

    // ===== MLP BLOCK =====
    let residual = hidden.clone();
    let ffn_n = cfg.ffn(i) as usize;

    // pre-FFN norm
    let mlp_norm_w = weights.load(&format!("{prefix}ffn_norm.weight"))?;
    let mut x = vec![0f32; d_model];
    rmsnorm(hidden, Some(&mlp_norm_w), eps, &mut x);
    drop(mlp_norm_w);

    // gate / up / GeGLU / down
    let gate_w = weights.load(&format!("{prefix}ffn_gate.weight"))?;
    let mut gate = vec![0f32; ffn_n];
    matvec(&gate_w, d_model, ffn_n, &x, &mut gate);
    drop(gate_w);

    let up_w = weights.load(&format!("{prefix}ffn_up.weight"))?;
    let mut up = vec![0f32; ffn_n];
    matvec(&up_w, d_model, ffn_n, &x, &mut up);
    drop(up_w);

    let mut act = vec![0f32; ffn_n];
    geglu_split(&gate, &up, &mut act);
    drop(gate);
    drop(up);

    let down_w = weights.load(&format!("{prefix}ffn_down.weight"))?;
    let mut mlp_out = vec![0f32; d_model];
    matvec(&down_w, ffn_n, d_model, &act, &mut mlp_out);
    drop(down_w);

    // post-FFN norm + residual
    let post_ffw_w = weights.load(&format!("{prefix}post_ffw_norm.weight"))?;
    let mut h3 = vec![0f32; d_model];
    rmsnorm(&mlp_out, Some(&post_ffw_w), eps, &mut h3);
    drop(post_ffw_w);
    add_into(&mut h3, &residual);
    *hidden = h3;

    // ===== PLE INJECTION =====
    if let Some(pli) = per_layer_input {
        let inp_gate_w = weights.load(&format!("{prefix}inp_gate.weight"))?;
        let mut ple_state = vec![0f32; cfg.ple_dim as usize];
        matvec(&inp_gate_w, d_model, cfg.ple_dim as usize, hidden, &mut ple_state);
        drop(inp_gate_w);

        // ple_state = gelu(ple_state) * pli   (GeGLU split, gate=ple_state, up=pli)
        let mut activated = vec![0f32; cfg.ple_dim as usize];
        geglu_split(&ple_state, pli, &mut activated);

        let proj_w = weights.load(&format!("{prefix}proj.weight"))?;
        let mut projected = vec![0f32; d_model];
        matvec(&proj_w, cfg.ple_dim as usize, d_model, &activated, &mut projected);
        drop(proj_w);

        let post_norm_w = weights.load(&format!("{prefix}post_norm.weight"))?;
        let mut normed = vec![0f32; d_model];
        rmsnorm(&projected, Some(&post_norm_w), eps, &mut normed);
        drop(post_norm_w);

        add_into(hidden, &normed);
    }

    // ===== LAYER OUTPUT SCALAR (every layer in this checkpoint) =====
    if let Some(scalar) = weights.load_opt(&format!("{prefix}layer_output_scale.weight"))? {
        if let Some(&s) = scalar.first() { scale(hidden, s); }
    }
    Ok(())
}

/// Self-attention block. Computes Q/K/V, normalizes Q/K/V, applies RoPE, appends to KV
/// cache (or reads donor's), runs softmax-attention with optional sliding window mask,
/// then output projection. Mirrors `TextSelfAttention.Forward`.
fn self_attention(
    cfg: &Gemma4Config,
    weights: &Weights,
    kv_state: &mut KvState,
    donor_map: &[Option<u32>],
    i: u32,
    pos: u32,
    x: &[f32],
) -> Result<Vec<f32>> {
    let prefix = format!("blk.{i}.");
    let d_model = cfg.d_model as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads(i) as usize;
    let head_dim = cfg.head_dim(i) as usize;
    let eps = cfg.rms_norm_eps;

    // ---- Q ----
    let q_w = weights.load(&format!("{prefix}attn_q.weight"))?;
    let mut q = vec![0f32; n_heads * head_dim];
    matvec(&q_w, d_model, n_heads * head_dim, x, &mut q);
    drop(q_w);

    // Q-norm (weighted RMSNorm, applied per head over head_dim)
    let q_norm_w = weights.load(&format!("{prefix}attn_q_norm.weight"))?;
    let mut q_normed = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let off = h * head_dim;
        rmsnorm(&q[off..off + head_dim], Some(&q_norm_w), eps, &mut q_normed[off..off + head_dim]);
    }
    drop(q_norm_w);
    let mut q = q_normed;

    // ---- K, V (skip if KV-shared from donor) ----
    let donor = donor_map[i as usize];
    if donor.is_none() {
        let k_w = weights.load(&format!("{prefix}attn_k.weight"))?;
        let mut k = vec![0f32; n_kv_heads * head_dim];
        matvec(&k_w, d_model, n_kv_heads * head_dim, x, &mut k);
        drop(k_w);

        let v_w = weights.load(&format!("{prefix}attn_v.weight"))?;
        let mut v = vec![0f32; n_kv_heads * head_dim];
        matvec(&v_w, d_model, n_kv_heads * head_dim, x, &mut v);
        drop(v_w);

        // K-norm (weighted, per kv_head over head_dim)
        let k_norm_w = weights.load(&format!("{prefix}attn_k_norm.weight"))?;
        let mut k_normed = vec![0f32; n_kv_heads * head_dim];
        for h in 0..n_kv_heads {
            let off = h * head_dim;
            rmsnorm(&k[off..off + head_dim], Some(&k_norm_w), eps, &mut k_normed[off..off + head_dim]);
        }
        drop(k_norm_w);

        // V-norm (unweighted RMSNorm, per kv_head over head_dim)
        let mut v_normed = vec![0f32; n_kv_heads * head_dim];
        for h in 0..n_kv_heads {
            let off = h * head_dim;
            rmsnorm(&v[off..off + head_dim], None, eps, &mut v_normed[off..off + head_dim]);
        }

        // RoPE on Q and K. K is rotated; V is not.
        apply_rope(cfg, weights, i, pos, head_dim, n_heads, &mut q)?;
        apply_rope(cfg, weights, i, pos, head_dim, n_kv_heads, &mut k_normed)?;

        // Append to layer's KV cache.
        let lkv = &mut kv_state.layers[i as usize];
        lkv.n_kv_heads = n_kv_heads as u32;
        lkv.head_dim = head_dim as u32;
        lkv.k.extend_from_slice(&k_normed);
        lkv.v.extend_from_slice(&v_normed);
    } else {
        // KV is shared from donor — Q is still computed and rotated, but K/V come
        // from donor's history (already rotated and normalized).
        apply_rope(cfg, weights, i, pos, head_dim, n_heads, &mut q)?;
    }

    let kv_layer = donor.unwrap_or(i) as usize;
    let lkv = &kv_state.layers[kv_layer];
    if lkv.head_dim as usize != head_dim || lkv.n_kv_heads as usize != n_kv_heads {
        return Err(RullamaError::Inference(format!(
            "donor layer {kv_layer} kv shape ({}×{}) != current layer {i} ({}×{})",
            lkv.n_kv_heads, lkv.head_dim, n_kv_heads, head_dim
        )));
    }

    // ---- attention ----
    let history_len = lkv.k.len() / (n_kv_heads * head_dim);
    let attn_out = run_attention(cfg, i, pos, head_dim, n_heads, n_kv_heads, history_len, &q, &lkv.k, &lkv.v);

    // ---- output projection ----
    let o_w = weights.load(&format!("{prefix}attn_output.weight"))?;
    let mut out = vec![0f32; d_model];
    matvec(&o_w, n_heads * head_dim, d_model, &attn_out, &mut out);
    drop(o_w);
    Ok(out)
}

/// Apply NeoX RoPE in-place to a `[head_dim, n_heads]` tensor at the given position.
/// On global layers, looks up `rope_freqs.weight` to do proportional rotation.
fn apply_rope(
    cfg: &Gemma4Config,
    weights: &Weights,
    layer: u32,
    pos: u32,
    head_dim: usize,
    n_heads: usize,
    x: &mut [f32],
) -> Result<()> {
    let (base, rope_dims) = match cfg.kind(layer) {
        LayerKind::SlidingWindow => (cfg.rope_freq_base_swa, cfg.rope_dim_swa as usize),
        LayerKind::Global        => (cfg.rope_freq_base,     cfg.rope_dim_global as usize),
    };
    // freq_factors only on global layers, and only if the tensor is present.
    let freqs = if matches!(cfg.kind(layer), LayerKind::Global) {
        weights.load_opt("rope_freqs.weight")?
    } else {
        None
    };
    rope_neox(x, head_dim, n_heads, pos as usize, rope_dims, base, freqs.as_deref());
    Ok(())
}

/// Softmax attention over the cached K/V history. Returns `[n_heads * head_dim]`.
fn run_attention(
    cfg: &Gemma4Config,
    layer: u32,
    pos: u32,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    history_len: usize,
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let window = cfg.sliding_window as usize;
    let is_swa = matches!(cfg.kind(layer), LayerKind::SlidingWindow);

    // For SWA layers we attend only to positions in [max(0, pos - window + 1) .. pos].
    let earliest: usize = if is_swa {
        (pos as usize + 1).saturating_sub(window)
    } else {
        0
    };

    let mut out = vec![0f32; n_heads * head_dim];
    let mut scores = vec![0f32; history_len];

    for qh in 0..n_heads {
        let kvh = qh / heads_per_kv;
        let q_off = qh * head_dim;

        // Scores: q · k(t) for t in valid range; -inf elsewhere (causal + SWA mask).
        for t in 0..history_len {
            if t < earliest || t > pos as usize {
                scores[t] = f32::NEG_INFINITY;
                continue;
            }
            let k_off = (t * n_kv_heads + kvh) * head_dim;
            let mut acc = 0f32;
            for d in 0..head_dim {
                acc += q[q_off + d] * k_history[k_off + d];
            }
            // scale = 1.0 (Gemma 4 absorbs scaling into Q-norm weights).
            scores[t] = acc;
        }

        softmax(&mut scores);

        // Weighted sum of v(t).
        let out_off = qh * head_dim;
        for d in 0..head_dim { out[out_off + d] = 0.0; }
        for t in 0..history_len {
            let w = scores[t];
            if w == 0.0 { continue; }
            let v_off = (t * n_kv_heads + kvh) * head_dim;
            for d in 0..head_dim {
                out[out_off + d] += w * v_history[v_off + d];
            }
        }
    }

    out
}
