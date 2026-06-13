//! Hybrid streaming GPU forward for DiffusionGemma (Phase C4, increment 1).
//!
//! The non-expert matmuls — attention Q/K/V/O, the dense parallel MLP
//! (gate/up/down), and the lm_head — run on the GPU, batched over ALL `n =
//! prompt + canvas` positions in one dispatch each:
//!
//! - Q4_K / Q5_0 / Q8_0 weights reuse the batched MoE expert kernel
//!   (`moe_expert_matmul_batched`) with `top_k = 1` and an all-zero `ids`
//!   buffer — every position routes to the one resident "expert" (= the dense
//!   weight at slice 0), i.e. a batched dense quant matmul for free.
//! - Q6_K weights (`attn_v` on 13 layers, the tied `token_embd` lm_head) have
//!   no batched kernel yet, so they fall back to a single-row
//!   `matmul_quant_chained` loop over positions (correct-but-slow; a batched
//!   Q6_K kernel is the perf follow-up).
//!
//! Everything else stays in the validated CPU f32 oracle math: token
//! embeddings, the self-conditioning subgraph, all RMSNorms, RoPE, the
//! region-masked bidirectional attention, the router, and the MoE experts
//! (expert-major CPU loop — moving that to the batched GPU path is increment 2,
//! task C4b). The result matches `diffusion_forward` to f32 matmul
//! accumulation-order round-off.
//!
//! Weights flow through the same streaming infra as the AR MoE path: the dense
//! matmul weights are fetched + cached as GPU buffers via
//! [`WeightCache::buffer_async`]; the CPU MoE experts are range-streamed by the
//! synchronous [`Weights`] loaders. wasm peak stays bounded to a tensor at a
//! time, not the 16.8 GB file.

use std::collections::HashMap;

use super::DiffusionConfig;
use super::forward::masked_attention;
use crate::backend::dispatch::{
    make_storage_rw, matmul_quant_chained, moe_expert_matmul_batched_chained, read_back_f32,
};
use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgmlDtype;
use crate::model::config::LayerKind;
use crate::reference::moe::{layer_has_moe, softmax_topk_renorm};
use crate::reference::ops::{
    add_into, geglu_split, matvec, rmsnorm, rope_neox, scale, softcap, softmax,
};
use crate::reference::weights::Weights;

/// GPU context bundle threaded through the forward (keeps the arg lists sane).
struct Gpu<'a> {
    ctx: &'a WgpuCtx,
    pipes: &'a Pipelines,
    wcache: &'a WeightCache,
    /// `n`-long all-zero `u32` ids buffer reused for every dense (top_k=1)
    /// batched matmul; entry `i` routes position `i` to the single resident
    /// weight slice.
    zero_ids: wgpu::Buffer,
}

/// As [`super::forward::diffusion_forward_sc`], but the non-expert matmuls run
/// on the GPU. `wcache` + `weights` must wrap the same GGUF (`wcache` owns the
/// GPU buffers, `weights` the CPU streaming reads).
#[allow(clippy::too_many_arguments)]
pub async fn diffusion_forward_gpu(
    cfg: &DiffusionConfig,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    wcache: &WeightCache,
    weights: &Weights,
    prompt_ids: &[u32],
    canvas_ids: &[u32],
    prev_logits: Option<&[f32]>,
    sc_temp_inv: f32,
) -> Result<Vec<f32>> {
    let base = &cfg.base;
    let d_model = base.d_model as usize;
    let eps = base.rms_norm_eps;
    let n_swa = base.sliding_window as usize;
    let p = prompt_ids.len();
    let c = canvas_ids.len();
    let n = p + c;
    let vocab = base.vocab_size as usize;

    let zero_ids = upload_u32(&ctx.device, "dg.zero_ids", &vec![0u32; n.max(1)]);
    let gpu = Gpu {
        ctx,
        pipes,
        wcache,
        zero_ids,
    };

    // ---- 1. scaled word embedding (CPU per-row range fetch) ----
    let embd_scale = (d_model as f32).sqrt();
    let mut hidden = vec![0f32; n * d_model];
    for (i, &id) in prompt_ids.iter().chain(canvas_ids.iter()).enumerate() {
        let row = weights
            .load_row_async("token_embd.weight", id as usize)
            .await?;
        for (h, &e) in hidden[i * d_model..(i + 1) * d_model]
            .iter_mut()
            .zip(row.iter())
        {
            *h = e * embd_scale;
        }
    }

    // ---- 2. canvas embedding (+ self-conditioning), CPU ----
    apply_canvas_embedding(
        cfg,
        weights,
        &mut hidden,
        prompt_ids.len(),
        c,
        prev_logits,
        sc_temp_inv,
    )
    .await?;

    // ---- 3. transformer layers ----
    for layer in 0..base.n_layers {
        diffusion_layer_gpu(cfg, &gpu, weights, layer, &mut hidden, p, n, n_swa, eps).await?;
    }

    // ---- 4. output norm + lm_head + softcap, CANVAS rows only ----
    let out_norm_w = weights.load_async("output_norm.weight").await?;
    let out_w_name = if weights.has("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight" // tied
    };
    let out_dtype = wcache.reader().tensor(out_w_name)?.dtype;

    // Normalize the canvas rows on CPU, then run the (huge) lm_head on GPU.
    let mut normed = vec![0f32; c * d_model];
    for ci in 0..c {
        let i = p + ci;
        rmsnorm(
            &hidden[i * d_model..(i + 1) * d_model],
            Some(&out_norm_w),
            eps,
            &mut normed[ci * d_model..(ci + 1) * d_model],
        );
    }
    let mut logits = gpu_matmul(&gpu, out_w_name, &normed, d_model, vocab, c, out_dtype).await?;
    if base.final_logit_softcap > 0.0 {
        for ci in 0..c {
            softcap(
                &mut logits[ci * vocab..(ci + 1) * vocab],
                base.final_logit_softcap,
            );
        }
    }
    Ok(logits)
}

/// One transformer layer: attention (GPU projections + CPU masked attention),
/// then the gemma4 parallel dense-MLP + MoE FFN, then the region-aware dual
/// per-layer output scale. In place on `hidden` (`[n, d_model]`).
#[allow(clippy::too_many_arguments)]
async fn diffusion_layer_gpu(
    cfg: &DiffusionConfig,
    gpu: &Gpu<'_>,
    weights: &Weights,
    layer: u32,
    hidden: &mut [f32],
    p: usize,
    n: usize,
    n_swa: usize,
    eps: f32,
) -> Result<()> {
    let base = &cfg.base;
    let d_model = base.d_model as usize;
    let n_heads = base.n_heads as usize;
    let n_kv = base.n_kv_heads(layer) as usize;
    let head_dim = base.head_dim(layer) as usize;
    let swa_layer = matches!(base.kind(layer), LayerKind::SlidingWindow);
    let prefix = format!("blk.{layer}.");
    let residual = hidden.to_vec();

    // ===== attention =====
    // attn_norm (weighted) per row → nx.
    let attn_norm_w = weights
        .load_async(&format!("{prefix}attn_norm.weight"))
        .await?;
    let mut nx = vec![0f32; n * d_model];
    for i in 0..n {
        rmsnorm(
            &hidden[i * d_model..(i + 1) * d_model],
            Some(&attn_norm_w),
            eps,
            &mut nx[i * d_model..(i + 1) * d_model],
        );
    }

    // Q/K/V projections on GPU (batched over all n positions).
    let q_dtype = gpu
        .wcache
        .reader()
        .tensor(&format!("{prefix}attn_q.weight"))?
        .dtype;
    let k_dtype = gpu
        .wcache
        .reader()
        .tensor(&format!("{prefix}attn_k.weight"))?
        .dtype;
    let q_proj = gpu_matmul(
        gpu,
        &format!("{prefix}attn_q.weight"),
        &nx,
        d_model,
        n_heads * head_dim,
        n,
        q_dtype,
    )
    .await?;
    let k_proj = gpu_matmul(
        gpu,
        &format!("{prefix}attn_k.weight"),
        &nx,
        d_model,
        n_kv * head_dim,
        n,
        k_dtype,
    )
    .await?;
    let v_name = format!("{prefix}attn_v.weight");
    let v_proj = if weights.has(&v_name) {
        let v_dtype = gpu.wcache.reader().tensor(&v_name)?.dtype;
        gpu_matmul(gpu, &v_name, &nx, d_model, n_kv * head_dim, n, v_dtype).await?
    } else {
        k_proj.clone() // no-V layers: V := raw K projection (before K-norm)
    };

    // Per-position q/k norms + RoPE; v norm (unweighted, no rope). CPU.
    let q_norm_w = weights
        .load_async(&format!("{prefix}attn_q_norm.weight"))
        .await?;
    let k_norm_w = weights
        .load_async(&format!("{prefix}attn_k_norm.weight"))
        .await?;
    let freqs = if matches!(base.kind(layer), LayerKind::Global) {
        weights.load_opt("rope_freqs.weight")?
    } else {
        None
    };
    let (base_freq, rope_dims) = match base.kind(layer) {
        LayerKind::SlidingWindow => (base.rope_freq_base_swa, base.rope_dim_swa as usize),
        LayerKind::Global => (base.rope_freq_base, base.rope_dim_global as usize),
    };

    let mut q_all = vec![0f32; n * n_heads * head_dim];
    let mut k_all = vec![0f32; n * n_kv * head_dim];
    let mut v_all = vec![0f32; n * n_kv * head_dim];
    for i in 0..n {
        let mut q = q_proj[i * n_heads * head_dim..(i + 1) * n_heads * head_dim].to_vec();
        for h in 0..n_heads {
            let o = h * head_dim;
            let mut t = vec![0f32; head_dim];
            rmsnorm(&q[o..o + head_dim], Some(&q_norm_w), eps, &mut t);
            q[o..o + head_dim].copy_from_slice(&t);
        }
        rope_neox(
            &mut q,
            head_dim,
            n_heads,
            i,
            rope_dims,
            base_freq,
            freqs.as_deref(),
        );
        q_all[i * n_heads * head_dim..(i + 1) * n_heads * head_dim].copy_from_slice(&q);

        let k = &k_proj[i * n_kv * head_dim..(i + 1) * n_kv * head_dim];
        let v = &v_proj[i * n_kv * head_dim..(i + 1) * n_kv * head_dim];
        let mut kn = vec![0f32; n_kv * head_dim];
        let mut vn = vec![0f32; n_kv * head_dim];
        for h in 0..n_kv {
            let o = h * head_dim;
            rmsnorm(
                &k[o..o + head_dim],
                Some(&k_norm_w),
                eps,
                &mut kn[o..o + head_dim],
            );
            rmsnorm(&v[o..o + head_dim], None, eps, &mut vn[o..o + head_dim]);
        }
        rope_neox(
            &mut kn,
            head_dim,
            n_kv,
            i,
            rope_dims,
            base_freq,
            freqs.as_deref(),
        );
        k_all[i * n_kv * head_dim..(i + 1) * n_kv * head_dim].copy_from_slice(&kn);
        v_all[i * n_kv * head_dim..(i + 1) * n_kv * head_dim].copy_from_slice(&vn);
    }

    // Region-masked attention (CPU — cheap vs the matmuls).
    let attn = masked_attention(
        &q_all, &k_all, &v_all, n, n_heads, n_kv, head_dim, p, n_swa, swa_layer,
    );

    // Output projection (GPU) → post_attention_norm + residual (CPU).
    let o_name = format!("{prefix}attn_output.weight");
    let o_dtype = gpu.wcache.reader().tensor(&o_name)?.dtype;
    let o_proj = gpu_matmul(gpu, &o_name, &attn, n_heads * head_dim, d_model, n, o_dtype).await?;
    let post_attn_w = weights
        .load_async(&format!("{prefix}post_attention_norm.weight"))
        .await?;
    for i in 0..n {
        let mut on = vec![0f32; d_model];
        rmsnorm(
            &o_proj[i * d_model..(i + 1) * d_model],
            Some(&post_attn_w),
            eps,
            &mut on,
        );
        add_into(&mut on, &residual[i * d_model..(i + 1) * d_model]);
        hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&on);
    }

    // ===== FFN: dense MLP (GPU matmuls) + MoE experts (CPU) =====
    let is_moe = base.has_moe() && layer_has_moe(weights, layer);
    let ffn_residual = hidden.to_vec();
    ffn_block_gpu(
        cfg,
        gpu,
        weights,
        layer,
        hidden,
        &ffn_residual,
        n,
        eps,
        is_moe,
    )
    .await?;

    // ===== region-aware dual per-layer scale (CPU) =====
    let enc = weights
        .load_opt(&format!("{prefix}enc_layer_output_scale.weight"))?
        .and_then(|v| v.first().copied());
    let dec = weights
        .load_opt(&format!("{prefix}layer_output_scale.weight"))?
        .and_then(|v| v.first().copied());
    if let Some(s) = enc {
        for i in 0..p {
            scale(&mut hidden[i * d_model..(i + 1) * d_model], s);
        }
    }
    if let Some(s) = dec {
        for i in p..n {
            scale(&mut hidden[i * d_model..(i + 1) * d_model], s);
        }
    }
    Ok(())
}

/// gemma4 FFN: parallel dense MLP (GPU gate/up/down) + optional 128-expert MoE
/// (CPU expert-major loop, increment 1). In place on `hidden`.
#[allow(clippy::too_many_arguments)]
async fn ffn_block_gpu(
    cfg: &DiffusionConfig,
    gpu: &Gpu<'_>,
    weights: &Weights,
    layer: u32,
    hidden: &mut [f32],
    residual: &[f32],
    n: usize,
    eps: f32,
    is_moe: bool,
) -> Result<()> {
    let base = &cfg.base;
    let d_model = base.d_model as usize;
    let prefix = format!("blk.{layer}.");
    let ffn_n = base.ffn(layer) as usize;

    let mlp_norm_w = weights
        .load_async(&format!("{prefix}ffn_norm.weight"))
        .await?;
    let post_ffw_w = weights
        .load_async(&format!("{prefix}post_ffw_norm.weight"))
        .await?;

    // Dense MLP, batched on GPU: rmsnorm → gate/up → geglu → down = mlp_out.
    let mut x = vec![0f32; n * d_model];
    for i in 0..n {
        rmsnorm(
            &hidden[i * d_model..(i + 1) * d_model],
            Some(&mlp_norm_w),
            eps,
            &mut x[i * d_model..(i + 1) * d_model],
        );
    }
    let gate_dtype = gpu
        .wcache
        .reader()
        .tensor(&format!("{prefix}ffn_gate.weight"))?
        .dtype;
    let up_dtype = gpu
        .wcache
        .reader()
        .tensor(&format!("{prefix}ffn_up.weight"))?
        .dtype;
    let down_dtype = gpu
        .wcache
        .reader()
        .tensor(&format!("{prefix}ffn_down.weight"))?
        .dtype;
    let gate = gpu_matmul(
        gpu,
        &format!("{prefix}ffn_gate.weight"),
        &x,
        d_model,
        ffn_n,
        n,
        gate_dtype,
    )
    .await?;
    let up = gpu_matmul(
        gpu,
        &format!("{prefix}ffn_up.weight"),
        &x,
        d_model,
        ffn_n,
        n,
        up_dtype,
    )
    .await?;
    let mut act = vec![0f32; n * ffn_n];
    for i in 0..n {
        geglu_split(
            &gate[i * ffn_n..(i + 1) * ffn_n],
            &up[i * ffn_n..(i + 1) * ffn_n],
            &mut act[i * ffn_n..(i + 1) * ffn_n],
        );
    }
    let mlp_out_all = gpu_matmul(
        gpu,
        &format!("{prefix}ffn_down.weight"),
        &act,
        ffn_n,
        d_model,
        n,
        down_dtype,
    )
    .await?;

    if !is_moe {
        for i in 0..n {
            let mut h3 = vec![0f32; d_model];
            rmsnorm(
                &mlp_out_all[i * d_model..(i + 1) * d_model],
                Some(&post_ffw_w),
                eps,
                &mut h3,
            );
            add_into(&mut h3, &residual[i * d_model..(i + 1) * d_model]);
            hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&h3);
        }
        return Ok(());
    }

    // --- MoE (CPU expert-major; increment-2 moves this to the batched GPU path) ---
    let post1 = weights
        .load_opt(&format!("{prefix}post_ffw_norm_1.weight"))?
        .or(weights.load_opt(&format!("{prefix}ffn_post_norm_1.weight"))?)
        .ok_or_else(|| {
            crate::error::RullamaError::Inference(format!(
                "MoE layer {layer}: missing post_ffw_norm_1"
            ))
        })?;
    let pre2 = weights
        .load_opt(&format!("{prefix}pre_ffw_norm_2.weight"))?
        .or(weights.load_opt(&format!("{prefix}ffn_pre_norm_2.weight"))?)
        .ok_or_else(|| {
            crate::error::RullamaError::Inference(format!(
                "MoE layer {layer}: missing pre_ffw_norm_2"
            ))
        })?;
    let post2 = weights
        .load_opt(&format!("{prefix}post_ffw_norm_2.weight"))?
        .or(weights.load_opt(&format!("{prefix}ffn_post_norm_2.weight"))?)
        .ok_or_else(|| {
            crate::error::RullamaError::Inference(format!(
                "MoE layer {layer}: missing post_ffw_norm_2"
            ))
        })?;

    let router_w = weights.load(&format!("{prefix}ffn_gate_inp.weight"))?;
    let router_scale = weights.load_opt(&format!("{prefix}ffn_gate_inp.scale"))?;
    let n_experts = base.expert_count as usize;
    let top_k = base.expert_used_count as usize;
    let inv_sqrt_d = 1.0 / (d_model as f32).sqrt();

    let mut moe_x_all = vec![0f32; n * d_model];
    let mut selected_all: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
    for i in 0..n {
        let h = &hidden[i * d_model..(i + 1) * d_model];
        let mut rx = vec![0f32; d_model];
        rmsnorm(h, None, eps, &mut rx);
        for v in &mut rx {
            *v *= inv_sqrt_d;
        }
        if let Some(s) = &router_scale {
            for (v, sv) in rx.iter_mut().zip(s.iter()) {
                *v *= sv;
            }
        }
        let mut scores = vec![0f32; n_experts];
        matvec(&router_w, d_model, n_experts, &rx, &mut scores);
        selected_all.push(softmax_topk_renorm(&scores, top_k));
        rmsnorm(
            h,
            Some(&pre2),
            eps,
            &mut moe_x_all[i * d_model..(i + 1) * d_model],
        );
    }

    let fused_name = format!("{prefix}ffn_gate_up_exps.weight");
    let fused = weights.has(&fused_name);
    let down_scale = match weights.load_opt(&format!("{prefix}ffn_down_exps.scale"))? {
        Some(s) => Some(s),
        None => weights.load_opt(&format!("{prefix}ffn_gate_inp.per_expert_scale"))?,
    };
    let mut by_expert: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    for (pos, sel) in selected_all.iter().enumerate() {
        for &(e, w) in sel {
            by_expert.entry(e).or_default().push((pos, w));
        }
    }
    let mut moe_acc = vec![0f32; n * d_model];
    for (&e, hits) in by_expert.iter() {
        let scale_e = down_scale.as_ref().map(|s| s[e]).unwrap_or(1.0);
        let down = weights.load_expert(&format!("{prefix}ffn_down_exps.weight"), e)?;
        let n_ff = down.len() / d_model;
        let (gate_e, up_e) = if fused {
            let gu = weights.load_expert(&fused_name, e)?;
            let nf = gu.len() / d_model / 2;
            (gu[..nf * d_model].to_vec(), gu[nf * d_model..].to_vec())
        } else {
            (
                weights.load_expert(&format!("{prefix}ffn_gate_exps.weight"), e)?,
                weights.load_expert(&format!("{prefix}ffn_up_exps.weight"), e)?,
            )
        };
        for &(pos, w) in hits {
            let xp = &moe_x_all[pos * d_model..(pos + 1) * d_model];
            let mut g = vec![0f32; n_ff];
            matvec(&gate_e, d_model, n_ff, xp, &mut g);
            let mut u = vec![0f32; n_ff];
            matvec(&up_e, d_model, n_ff, xp, &mut u);
            let mut a = vec![0f32; n_ff];
            geglu_split(&g, &u, &mut a);
            let mut dvec = vec![0f32; d_model];
            matvec(&down, n_ff, d_model, &a, &mut dvec);
            let acc = &mut moe_acc[pos * d_model..(pos + 1) * d_model];
            for (o, dv) in acc.iter_mut().zip(dvec.iter()) {
                *o += w * scale_e * dv;
            }
        }
    }

    for i in 0..n {
        let mut mlp_normed = vec![0f32; d_model];
        rmsnorm(
            &mlp_out_all[i * d_model..(i + 1) * d_model],
            Some(&post1),
            eps,
            &mut mlp_normed,
        );
        let mut moe_normed = vec![0f32; d_model];
        rmsnorm(
            &moe_acc[i * d_model..(i + 1) * d_model],
            Some(&post2),
            eps,
            &mut moe_normed,
        );
        add_into(&mut mlp_normed, &moe_normed);
        let mut h3 = vec![0f32; d_model];
        rmsnorm(&mlp_normed, Some(&post_ffw_w), eps, &mut h3);
        add_into(&mut h3, &residual[i * d_model..(i + 1) * d_model]);
        hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&h3);
    }
    Ok(())
}

/// Canvas-embedding preamble (CPU): `canvas = rms_norm(canvas + sc_sig)` where
/// the self-conditioning signal is gated off (`prev_logits = None`) on step 0.
/// Mirrors `diffusion_forward_sc`'s canvas block exactly.
async fn apply_canvas_embedding(
    cfg: &DiffusionConfig,
    weights: &Weights,
    hidden: &mut [f32],
    p: usize,
    c: usize,
    prev_logits: Option<&[f32]>,
    sc_temp_inv: f32,
) -> Result<()> {
    let base = &cfg.base;
    let d_model = base.d_model as usize;
    let eps = base.rms_norm_eps;
    let vocab = base.vocab_size as usize;
    let embd_scale = (d_model as f32).sqrt();
    let n = p + c;

    let sc = if let Some(pl) = prev_logits {
        assert_eq!(pl.len(), c * vocab, "prev_logits must be [canvas, vocab]");
        let tok = weights.load_async("token_embd.weight").await?;
        let pre_norm = weights.load_async("self_cond_pre_norm.weight").await?;
        let gate = weights.load_async("self_cond_gate.weight").await?;
        let up = weights.load_async("self_cond_up.weight").await?;
        let down = weights.load_async("self_cond_down.weight").await?;
        let n_ff = gate.len() / d_model;
        Some((tok, pre_norm, gate, up, down, n_ff))
    } else {
        None
    };

    for i in p..n {
        let mut combined = hidden[i * d_model..(i + 1) * d_model].to_vec();
        if let Some((tok, pre_norm, gate, up, down, n_ff)) = &sc {
            let ci = i - p;
            let mut probs: Vec<f32> = prev_logits.unwrap()[ci * vocab..(ci + 1) * vocab]
                .iter()
                .map(|&l| l * sc_temp_inv)
                .collect();
            softmax(&mut probs);
            let mut soft = vec![0f32; d_model];
            for (v, &pv) in probs.iter().enumerate() {
                if pv < 1e-9 {
                    continue;
                }
                let emb = &tok[v * d_model..(v + 1) * d_model];
                for (s, &e) in soft.iter_mut().zip(emb.iter()) {
                    *s += pv * e;
                }
            }
            for s in &mut soft {
                *s *= embd_scale;
            }
            let mut normed = vec![0f32; d_model];
            rmsnorm(&soft, Some(pre_norm), eps, &mut normed);
            let mut g = vec![0f32; *n_ff];
            matvec(gate, d_model, *n_ff, &normed, &mut g);
            let mut u = vec![0f32; *n_ff];
            matvec(up, d_model, *n_ff, &normed, &mut u);
            let mut a = vec![0f32; *n_ff];
            geglu_split(&g, &u, &mut a);
            let mut sc_sig = vec![0f32; d_model];
            matvec(down, *n_ff, d_model, &a, &mut sc_sig);
            add_into(&mut combined, &sc_sig);
        }
        let mut tmp = vec![0f32; d_model];
        rmsnorm(&combined, None, eps, &mut tmp);
        hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&tmp);
    }
    Ok(())
}

/// `y[pos, j] = Σ_i x[pos, i] · dequant(W[j, i])` for all `n_pos` positions.
/// Q4_K / Q5_0 / Q8_0 go through the batched MoE expert kernel (top_k=1, zero
/// ids = the one resident dense weight); other dtypes (Q6_K) fall back to a
/// single-row `matmul_quant_chained` loop. `W` is streamed + cached as a GPU
/// buffer.
async fn gpu_matmul(
    gpu: &Gpu<'_>,
    weight_name: &str,
    x: &[f32],
    k: usize,
    n: usize,
    n_pos: usize,
    dtype: GgmlDtype,
) -> Result<Vec<f32>> {
    let ctx = gpu.ctx;
    let w = gpu.wcache.buffer_async(weight_name).await?;
    let xb = upload_f32(&ctx.device, "dg.x", x);
    let yb = make_storage_rw(&ctx.device, "dg.y", n_pos * n);
    let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dg.read"),
        size: (n_pos * n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dg.mm"),
        });

    match dtype {
        GgmlDtype::Q4_K | GgmlDtype::Q5_0 | GgmlDtype::Q8_0 => {
            // Batched dense matmul: one "expert" (slice 0), every position routes
            // to it via the all-zero ids buffer.
            moe_expert_matmul_batched_chained(
                ctx,
                gpu.pipes,
                &mut enc,
                &w,
                &gpu.zero_ids,
                &xb,
                &yb,
                n_pos,
                k,
                n,
                1,
                dtype,
            )?;
        }
        _ => {
            // Single-row fallback (Q6_K): dispatch one matmul per position into
            // its row of `yb`.
            for pos in 0..n_pos {
                let xrow = upload_f32(&ctx.device, "dg.xrow", &x[pos * k..(pos + 1) * k]);
                let yrow = make_storage_rw(&ctx.device, "dg.yrow", n);
                matmul_quant_chained(ctx, gpu.pipes, &mut enc, &w, &xrow, &yrow, k, n, dtype)?;
                enc.copy_buffer_to_buffer(&yrow, 0, &yb, (pos * n * 4) as u64, (n * 4) as u64);
            }
        }
    }
    enc.copy_buffer_to_buffer(&yb, 0, &read, 0, (n_pos * n * 4) as u64);
    ctx.queue.submit(Some(enc.finish()));
    read_back_f32(&ctx.device, &read).await
}

fn upload_f32(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn upload_u32(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}
