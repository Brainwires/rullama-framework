// Per-layer dispatcher fns take many dims (d_model, n_heads, head_dim, ffn,
// n_kv, etc.) — they map 1:1 to Ollama's Go signatures and bundling them
// adds no clarity.
#![allow(clippy::too_many_arguments)]

//! GPU-backed forward pass for Gemma 4. Mirrors `forward.rs` op-for-op but dispatches
//! each weight matmul, normalization, RoPE, attention, and GeGLU through cached wgpu
//! pipelines. Matmul weights live in a [`WeightCache`] so they're uploaded to the GPU
//! exactly once per model session.
//!
//! All weight access is async (M6): on streaming readers each tensor's bytes are
//! fetched from the underlying [`TensorFetcher`] (e.g. an HTTP Range request),
//! uploaded to the GPU, and dropped before the next call. On in-memory readers
//! the same path adds one memcpy per first-touch and is otherwise free.
//!
//! Behavior versus `forward_token`:
//!   * embed lookup, embed scale, vector-add residuals, scalar multiplies, per-layer
//!     slicing — kept on CPU (cheap, would just add GPU roundtrips otherwise).
//!   * Every matmul (Q/K/V/O, gate/up/down, PLE projector + per-layer gate/proj)
//!     runs as a single GPU dispatch over a *cached* weight buffer.
//!   * RMSNorm, RoPE, attention, GeGLU, softcap — go through cached pipelines but
//!     re-upload their (small) f32 inputs each call. M7 fuses these.

use crate::backend::dispatch::{
    attention_cached, geglu_cached, matmul_q4_k_buf, matmul_q6_k_buf, rmsnorm_cached,
    rope_neox_cached, softcap_cached,
};
use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::{Result, RullamaError};
use crate::gguf::{GgmlDtype, dequant_tensor_to_f32_async};
use crate::model::config::{Gemma4Config, LayerKind};
use crate::reference::forward::{KvState, build_donor_map_pub};
use crate::reference::ops::{add_into, scale};
use crate::reference::weights::Weights;

/// Run one forward step at `pos` for the single token `token_id` on the GPU. Returns
/// logits over the vocab, post-softcap. Mutates `kv_state` to append new K/V for
/// non-shared layers.
pub async fn forward_token_gpu(
    cfg: &Gemma4Config,
    weights: &Weights,
    wcache: &WeightCache,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    kv_state: &mut KvState,
    token_id: u32,
    pos: u32,
) -> Result<Vec<f32>> {
    if (token_id as u64) >= cfg.vocab_size as u64 {
        return Err(RullamaError::Inference(format!(
            "token_id {token_id} >= vocab_size {}",
            cfg.vocab_size
        )));
    }
    let d_model = cfg.d_model as usize;

    // ---- token embedding (CPU; one row of Q6_K) ----
    let mut hidden = weights
        .load_row_async("token_embd.weight", token_id as usize)
        .await?;
    scale(&mut hidden, (d_model as f32).sqrt());

    // ---- per-layer inputs (PLE) ----
    let per_layer_inputs: Option<Vec<Vec<f32>>> = if cfg.has_ple() {
        Some(
            prepare_per_layer_inputs_gpu(cfg, weights, wcache, ctx, pipes, &hidden, token_id)
                .await?,
        )
    } else {
        None
    };

    // ---- transformer layers ----
    let donor_map = build_donor_map_pub(cfg);
    for i in 0..cfg.n_layers {
        let pli = per_layer_inputs
            .as_ref()
            .map(|all| all[i as usize].as_slice());
        layer_forward_gpu(
            cfg,
            weights,
            wcache,
            ctx,
            pipes,
            kv_state,
            &donor_map,
            i,
            pos,
            &mut hidden,
            pli,
        )
        .await?;
    }

    // ---- final norm (GPU) ----
    let final_norm_w = weights.load_async("output_norm.weight").await?;
    let x = rmsnorm_cached(ctx, pipes, &hidden, Some(&final_norm_w), cfg.rms_norm_eps).await?;
    drop(final_norm_w);

    // ---- output projection (GPU, tiled) ----
    // token_embd is Q6_K [d_model, vocab=262144] ≈ 330 MB; > 128 MiB binding limit.
    // Split into ~80 MiB tiles along the vocab axis, run matmul per tile, concatenate.
    const MAX_TILE_BYTES: usize = 80 * 1024 * 1024;
    let tiles = wcache
        .buffer_tiles_async("token_embd.weight", MAX_TILE_BYTES)
        .await?;
    let mut logits = vec![0f32; cfg.vocab_size as usize];
    let token_embd_dtype = wcache.dtype("token_embd.weight")?;
    for tile in &tiles {
        let part = match token_embd_dtype {
            GgmlDtype::Q6_K => {
                matmul_q6_k_buf(ctx, pipes, &tile.buffer, &x, d_model, tile.n_rows).await?
            }
            GgmlDtype::Q4_K => {
                matmul_q4_k_buf(ctx, pipes, &tile.buffer, &x, d_model, tile.n_rows).await?
            }
            other => {
                return Err(RullamaError::Inference(format!(
                    "token_embd dtype {other:?} not supported"
                )));
            }
        };
        logits[tile.row_start..tile.row_start + tile.n_rows].copy_from_slice(&part);
    }

    // ---- softcap (GPU) ----
    if cfg.final_logit_softcap > 0.0 {
        logits = softcap_cached(ctx, pipes, &logits, cfg.final_logit_softcap).await?;
    }
    Ok(logits)
}

async fn prepare_per_layer_inputs_gpu(
    cfg: &Gemma4Config,
    weights: &Weights,
    wcache: &WeightCache,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    hidden: &[f32],
    token_id: u32,
) -> Result<Vec<Vec<f32>>> {
    let ple_dim = cfg.ple_dim as usize;
    let n_layers = cfg.n_layers as usize;
    let d_model = cfg.d_model as usize;

    let mut inputs_per_layer = weights
        .load_row_async("per_layer_token_embd.weight", token_id as usize)
        .await?;
    scale(&mut inputs_per_layer, (ple_dim as f32).sqrt());

    if wcache.dtype("per_layer_model_proj.weight")? != GgmlDtype::Q4_K {
        return Err(RullamaError::Inference(
            "per_layer_model_proj expected Q4_K".into(),
        ));
    }
    let proj_buf = wcache.buffer_async("per_layer_model_proj.weight").await?;
    let mut projection =
        matmul_q4_k_buf(ctx, pipes, &proj_buf, hidden, d_model, n_layers * ple_dim).await?;
    scale(&mut projection, 1.0 / (d_model as f32).sqrt());

    let proj_norm_w = weights.load_async("per_layer_proj_norm.weight").await?;
    let mut normed = vec![0f32; n_layers * ple_dim];
    for layer in 0..n_layers {
        let off = layer * ple_dim;
        let slice = &projection[off..off + ple_dim];
        let nslice =
            rmsnorm_cached(ctx, pipes, slice, Some(&proj_norm_w), cfg.rms_norm_eps).await?;
        normed[off..off + ple_dim].copy_from_slice(&nslice);
    }
    drop(proj_norm_w);

    add_into(&mut normed, &inputs_per_layer);
    scale(&mut normed, 1.0 / 2.0_f32.sqrt());

    Ok((0..n_layers)
        .map(|layer| normed[layer * ple_dim..(layer + 1) * ple_dim].to_vec())
        .collect())
}

async fn layer_forward_gpu(
    cfg: &Gemma4Config,
    weights: &Weights,
    wcache: &WeightCache,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
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

    // ===== ATTENTION =====
    let residual = hidden.clone();
    let attn_norm_w = weights
        .load_async(&format!("{prefix}attn_norm.weight"))
        .await?;
    let x = rmsnorm_cached(ctx, pipes, hidden, Some(&attn_norm_w), eps).await?;
    drop(attn_norm_w);

    let attn_out = self_attention_gpu(
        cfg, weights, wcache, ctx, pipes, kv_state, donor_map, i, pos, &x,
    )
    .await?;

    let post_attn_w = weights
        .load_async(&format!("{prefix}post_attention_norm.weight"))
        .await?;
    let mut h2 = rmsnorm_cached(ctx, pipes, &attn_out, Some(&post_attn_w), eps).await?;
    drop(post_attn_w);
    add_into(&mut h2, &residual);
    *hidden = h2;

    // ===== MLP =====
    let residual = hidden.clone();
    let ffn_n = cfg.ffn(i) as usize;

    let mlp_norm_w = weights
        .load_async(&format!("{prefix}ffn_norm.weight"))
        .await?;
    let x = rmsnorm_cached(ctx, pipes, hidden, Some(&mlp_norm_w), eps).await?;
    drop(mlp_norm_w);

    let gate_buf = wcache
        .buffer_async(&format!("{prefix}ffn_gate.weight"))
        .await?;
    let gate = matmul_q4_k_buf(ctx, pipes, &gate_buf, &x, d_model, ffn_n).await?;
    let up_buf = wcache
        .buffer_async(&format!("{prefix}ffn_up.weight"))
        .await?;
    let up = matmul_q4_k_buf(ctx, pipes, &up_buf, &x, d_model, ffn_n).await?;
    let act = geglu_cached(ctx, pipes, &gate, &up).await?;
    drop(gate);
    drop(up);

    let down_name = format!("{prefix}ffn_down.weight");
    let down_buf = wcache.buffer_async(&down_name).await?;
    let mlp_out = match wcache.dtype(&down_name)? {
        GgmlDtype::Q6_K => matmul_q6_k_buf(ctx, pipes, &down_buf, &act, ffn_n, d_model).await?,
        GgmlDtype::Q4_K => matmul_q4_k_buf(ctx, pipes, &down_buf, &act, ffn_n, d_model).await?,
        other => {
            return Err(RullamaError::Inference(format!(
                "ffn_down dtype {other:?} unsupported"
            )));
        }
    };

    let post_ffw_w = weights
        .load_async(&format!("{prefix}post_ffw_norm.weight"))
        .await?;
    let mut h3 = rmsnorm_cached(ctx, pipes, &mlp_out, Some(&post_ffw_w), eps).await?;
    drop(post_ffw_w);
    add_into(&mut h3, &residual);
    *hidden = h3;

    // ===== PLE injection =====
    if let Some(pli) = per_layer_input {
        let inp_gate_buf = wcache
            .buffer_async(&format!("{prefix}inp_gate.weight"))
            .await?;
        let ple_state = matmul_q4_k_buf(
            ctx,
            pipes,
            &inp_gate_buf,
            hidden,
            d_model,
            cfg.ple_dim as usize,
        )
        .await?;
        let activated = geglu_cached(ctx, pipes, &ple_state, pli).await?;
        let proj_buf = wcache.buffer_async(&format!("{prefix}proj.weight")).await?;
        let projected = matmul_q4_k_buf(
            ctx,
            pipes,
            &proj_buf,
            &activated,
            cfg.ple_dim as usize,
            d_model,
        )
        .await?;
        let post_norm_w = weights
            .load_async(&format!("{prefix}post_norm.weight"))
            .await?;
        let normed = rmsnorm_cached(ctx, pipes, &projected, Some(&post_norm_w), eps).await?;
        drop(post_norm_w);
        add_into(hidden, &normed);
    }

    if let Some(scalar) = weights
        .load_opt_async(&format!("{prefix}layer_output_scale.weight"))
        .await?
        && let Some(&s) = scalar.first()
    {
        scale(hidden, s);
    }
    Ok(())
}

async fn self_attention_gpu(
    cfg: &Gemma4Config,
    weights: &Weights,
    wcache: &WeightCache,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
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

    let q_buf = wcache
        .buffer_async(&format!("{prefix}attn_q.weight"))
        .await?;
    let q = matmul_q4_k_buf(ctx, pipes, &q_buf, x, d_model, n_heads * head_dim).await?;
    let q_norm_w = weights
        .load_async(&format!("{prefix}attn_q_norm.weight"))
        .await?;
    let mut q_normed = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let off = h * head_dim;
        let nh = rmsnorm_cached(ctx, pipes, &q[off..off + head_dim], Some(&q_norm_w), eps).await?;
        q_normed[off..off + head_dim].copy_from_slice(&nh);
    }
    drop(q_norm_w);
    let q = q_normed;

    let donor = donor_map[i as usize];
    if donor.is_none() {
        let k_buf = wcache
            .buffer_async(&format!("{prefix}attn_k.weight"))
            .await?;
        let k = matmul_q4_k_buf(ctx, pipes, &k_buf, x, d_model, n_kv_heads * head_dim).await?;
        let k_norm_w = weights
            .load_async(&format!("{prefix}attn_k_norm.weight"))
            .await?;
        let mut k_normed = vec![0f32; n_kv_heads * head_dim];
        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let nh =
                rmsnorm_cached(ctx, pipes, &k[off..off + head_dim], Some(&k_norm_w), eps).await?;
            k_normed[off..off + head_dim].copy_from_slice(&nh);
        }
        drop(k_norm_w);

        let v_name = format!("{prefix}attn_v.weight");
        let v_buf = wcache.buffer_async(&v_name).await?;
        let v = match wcache.dtype(&v_name)? {
            GgmlDtype::Q6_K => {
                matmul_q6_k_buf(ctx, pipes, &v_buf, x, d_model, n_kv_heads * head_dim).await?
            }
            GgmlDtype::Q4_K => {
                matmul_q4_k_buf(ctx, pipes, &v_buf, x, d_model, n_kv_heads * head_dim).await?
            }
            other => {
                return Err(RullamaError::Inference(format!(
                    "attn_v dtype {other:?} unsupported"
                )));
            }
        };
        let mut v_normed = vec![0f32; n_kv_heads * head_dim];
        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let nh = rmsnorm_cached(ctx, pipes, &v[off..off + head_dim], None, eps).await?;
            v_normed[off..off + head_dim].copy_from_slice(&nh);
        }

        let q_rotated =
            apply_rope_gpu(cfg, wcache, ctx, pipes, i, pos, head_dim, n_heads, q).await?;
        let k_rotated = apply_rope_gpu(
            cfg, wcache, ctx, pipes, i, pos, head_dim, n_kv_heads, k_normed,
        )
        .await?;

        let lkv = &mut kv_state.layers[i as usize];
        lkv.n_kv_heads = n_kv_heads as u32;
        lkv.head_dim = head_dim as u32;
        lkv.k.extend_from_slice(&k_rotated);
        lkv.v.extend_from_slice(&v_normed);

        let kv_layer = i as usize;
        let history_len = lkv.k.len() / (n_kv_heads * head_dim);
        let window = if matches!(cfg.kind(i), LayerKind::SlidingWindow) {
            cfg.sliding_window as usize
        } else {
            0
        };
        let attn = attention_cached(
            ctx,
            pipes,
            &q_rotated,
            &kv_state.layers[kv_layer].k,
            &kv_state.layers[kv_layer].v,
            head_dim,
            n_heads,
            n_kv_heads,
            pos as usize,
            history_len,
            window,
        )
        .await?;

        let o_buf = wcache
            .buffer_async(&format!("{prefix}attn_output.weight"))
            .await?;
        return matmul_q4_k_buf(ctx, pipes, &o_buf, &attn, n_heads * head_dim, d_model).await;
    }

    let q_rotated = apply_rope_gpu(cfg, wcache, ctx, pipes, i, pos, head_dim, n_heads, q).await?;
    let donor_idx = donor.unwrap() as usize;
    let lkv = &kv_state.layers[donor_idx];
    if lkv.head_dim as usize != head_dim || lkv.n_kv_heads as usize != n_kv_heads {
        return Err(RullamaError::Inference(format!(
            "donor layer {donor_idx} kv shape ({}×{}) != current layer {i} ({}×{})",
            lkv.n_kv_heads, lkv.head_dim, n_kv_heads, head_dim
        )));
    }
    let history_len = lkv.k.len() / (n_kv_heads * head_dim);
    let window = if matches!(cfg.kind(i), LayerKind::SlidingWindow) {
        cfg.sliding_window as usize
    } else {
        0
    };
    let attn = attention_cached(
        ctx,
        pipes,
        &q_rotated,
        &lkv.k,
        &lkv.v,
        head_dim,
        n_heads,
        n_kv_heads,
        pos as usize,
        history_len,
        window,
    )
    .await?;
    let o_buf = wcache
        .buffer_async(&format!("{prefix}attn_output.weight"))
        .await?;
    matmul_q4_k_buf(ctx, pipes, &o_buf, &attn, n_heads * head_dim, d_model).await
}

async fn apply_rope_gpu(
    cfg: &Gemma4Config,
    wcache: &WeightCache,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    layer: u32,
    pos: u32,
    head_dim: usize,
    n_heads: usize,
    x: Vec<f32>,
) -> Result<Vec<f32>> {
    let (base, rope_dims) = match cfg.kind(layer) {
        LayerKind::SlidingWindow => (cfg.rope_freq_base_swa, cfg.rope_dim_swa as usize),
        LayerKind::Global => (cfg.rope_freq_base, cfg.rope_dim_global as usize),
    };
    let factors = if matches!(cfg.kind(layer), LayerKind::Global) {
        // The RoPE factors tensor is tiny (head_dim/2 f32s); we read it as f32 each
        // call rather than building yet another GPU buffer plumbing path for it.
        dequant_tensor_to_f32_async(wcache.reader(), "rope_freqs.weight")
            .await
            .ok()
    } else {
        None
    };
    rope_neox_cached(
        ctx,
        pipes,
        &x,
        head_dim,
        n_heads,
        pos as usize,
        rope_dims,
        base,
        factors.as_deref(),
    )
    .await
}
