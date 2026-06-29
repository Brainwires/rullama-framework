//! DiffusionGemma CPU forward primitives.
//!
//! The full canvas-forward (wiring the backbone per the PR's
//! `diffusion-gemma.cpp` graph — reusing the validated gemma4 MoE FFN +
//! q/k/v/rope/norm ops, with the region mask in [`super::mask`] and the dual
//! enc/dec per-layer scales) lands once the `llama-diffusion-cli` parity
//! oracle is built. This module starts with the one genuinely-new op that has
//! no gemma4 analogue: **full-sequence masked attention** (the existing CPU
//! attention is strictly causal + per-token KV append; the canvas forward is
//! non-autoregressive — every position attends per a region mask in one pass).

use std::collections::HashMap;

use super::DiffusionConfig;
use super::mask::allowed;
use crate::error::Result;
use crate::model::config::LayerKind;
use crate::reference::moe::{layer_has_moe, softmax_topk_renorm};
use crate::reference::ops::{
    add_into, geglu_split, matvec, rmsnorm, rope_neox, scale, softcap, softmax,
};
use crate::reference::weights::Weights;

/// Non-autoregressive multi-head attention over a full token sequence with a
/// region mask. Layout: `q` is `[n_tokens, n_heads, head_dim]` row-major,
/// `k`/`v` are `[n_tokens, n_kv_heads, head_dim]` (GQA: each query head `qh`
/// reads kv head `qh / (n_heads/n_kv_heads)`). Score scale is 1.0 (Gemma 4
/// folds it into the Q-norm weights — matches PR's `f_attention_scale=1.0`).
/// Returns `[n_tokens, n_heads, head_dim]`.
///
/// `prompt_len` / `n_swa` / `swa_layer` drive the per-edge mask via
/// [`allowed`]; pass `swa_layer=false` + any `n_swa` for a global layer.
#[allow(clippy::too_many_arguments)]
pub fn masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    prompt_len: usize,
    n_swa: usize,
    swa_layer: bool,
) -> Vec<f32> {
    let heads_per_kv = (n_heads / n_kv_heads).max(1);
    let mut out = vec![0f32; n_tokens * n_heads * head_dim];
    let mut scores = vec![0f32; n_tokens];

    for qi in 0..n_tokens {
        for qh in 0..n_heads {
            let kvh = qh / heads_per_kv;
            let q_off = (qi * n_heads + qh) * head_dim;

            for (kj, score) in scores.iter_mut().enumerate() {
                if !allowed(qi, kj, prompt_len, n_swa, swa_layer) {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (kj * n_kv_heads + kvh) * head_dim;
                let mut acc = 0f32;
                for d in 0..head_dim {
                    acc += q[q_off + d] * k[k_off + d];
                }
                *score = acc; // scale 1.0
            }
            softmax(&mut scores);

            let o_off = (qi * n_heads + qh) * head_dim;
            for (kj, &w) in scores.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let v_off = (kj * n_kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    out[o_off + d] += w * v[v_off + d];
                }
            }
        }
    }
    out
}

/// Per-position weighted RMSNorm: norms each `d_model`-length row of a
/// `[n, d_model]` buffer with the same weight, in place.
fn rmsnorm_rows(x: &mut [f32], w: Option<&[f32]>, eps: f32, n: usize, d_model: usize) {
    let mut tmp = vec![0f32; d_model];
    for i in 0..n {
        rmsnorm(&x[i * d_model..(i + 1) * d_model], w, eps, &mut tmp);
        x[i * d_model..(i + 1) * d_model].copy_from_slice(&tmp);
    }
}

/// DiffusionGemma unified [prompt | canvas] CPU forward (zero self-conditioning
/// — the "exactness forward" the eval oracle dumps with no prev_logits). Mirrors
/// `diffusion-gemma.cpp`'s graph: scaled embd → canvas rows rms_norm(no scale)
/// → per layer { attn(region mask) + residual; gemma4 parallel dense+MoE FFN;
/// dual enc/dec scale } → output_norm → lm_head → softcap. Returns the
/// **canvas-position** logits `[canvas_len, vocab]` (what the oracle emits).
///
/// Reuses the validated `route`/`moe_experts` (Phase B) and the masked
/// attention above; the new structure is the full-sequence pass + the
/// region-aware dual per-layer scale.
pub fn diffusion_forward(
    cfg: &DiffusionConfig,
    weights: &Weights,
    prompt_ids: &[u32],
    canvas_ids: &[u32],
) -> Result<Vec<f32>> {
    diffusion_forward_sc(cfg, weights, prompt_ids, canvas_ids, None, 1.0)
}

/// As [`diffusion_forward`], but with optional self-conditioning: `prev_logits`
/// is the previous denoise step's RAW canvas logits `[canvas_len, vocab]`
/// (`None` ⇒ the zero-SC exactness forward; gated off on step 0 in the
/// sampler). `sc_temp_inv` is 1/temperature of the PREVIOUS step. Mirrors
/// `dg_canvas_embed`'s SC subgraph in diffusion-gemma.cpp.
pub fn diffusion_forward_sc(
    cfg: &DiffusionConfig,
    weights: &Weights,
    prompt_ids: &[u32],
    canvas_ids: &[u32],
    prev_logits: Option<&[f32]>,
    sc_temp_inv: f32,
) -> Result<Vec<f32>> {
    let base = &cfg.base;
    let d_model = base.d_model as usize;
    let n_heads = base.n_heads as usize;
    let eps = base.rms_norm_eps;
    let n_swa = base.sliding_window as usize;
    let p = prompt_ids.len();
    let c = canvas_ids.len();
    let n = p + c;
    let vocab = base.vocab_size as usize;

    // ---- 1. scaled word embedding ----
    let embd_scale = (d_model as f32).sqrt();
    let mut hidden = vec![0f32; n * d_model];
    for (i, &id) in prompt_ids.iter().chain(canvas_ids.iter()).enumerate() {
        let row = weights.load_row("token_embd.weight", id as usize)?;
        for (h, &e) in hidden[i * d_model..(i + 1) * d_model]
            .iter_mut()
            .zip(row.iter())
        {
            *h = e * embd_scale;
        }
    }

    // ---- 2. canvas embedding ----
    // SC off: canvas = rms_norm(canvas) (no scale).
    // SC on:  canvas = rms_norm(canvas + sc_sig(prev_logits)), where
    //   sc_sig = sc_mlp( rms_norm( (Σ_v softmax(prev/t)[v]·embd_v)·√d , sc_pre_norm ) )
    //   and sc_mlp is the gated GeGLU sc_gate/sc_up/sc_down.
    let sc = if let Some(pl) = prev_logits {
        assert_eq!(pl.len(), c * vocab, "prev_logits must be [canvas, vocab]");
        // Load the SC weights + the whole embedding table once. GGUF names the
        // self-conditioning MLP `self_cond_*` (gate/up are Q4_K, down is Q5_0).
        let tok = weights.load("token_embd.weight")?; // [vocab, d_model] flat
        let pre_norm = weights.load("self_cond_pre_norm.weight")?;
        let gate = weights.load("self_cond_gate.weight")?;
        let up = weights.load("self_cond_up.weight")?;
        let down = weights.load("self_cond_down.weight")?;
        let n_ff = gate.len() / d_model;
        Some((tok, pre_norm, gate, up, down, n_ff))
    } else {
        None
    };

    for i in p..n {
        let mut combined = hidden[i * d_model..(i + 1) * d_model].to_vec();
        if let Some((tok, pre_norm, gate, up, down, n_ff)) = &sc {
            let ci = i - p;
            // probs = softmax(prev_logits[ci] * sc_temp_inv)
            let mut probs: Vec<f32> = prev_logits.unwrap()[ci * vocab..(ci + 1) * vocab]
                .iter()
                .map(|&l| l * sc_temp_inv)
                .collect();
            softmax(&mut probs);
            // soft[e] = Σ_v probs[v] · embd_v[e]  (weighted average of token embeddings)
            let mut soft = vec![0f32; d_model];
            for (v, &pv) in probs.iter().enumerate() {
                if pv < 1e-9 {
                    continue; // negligible: trims the 262k-token sum to the softmax support
                }
                let emb = &tok[v * d_model..(v + 1) * d_model];
                for (s, &e) in soft.iter_mut().zip(emb.iter()) {
                    *s += pv * e;
                }
            }
            for s in &mut soft {
                *s *= embd_scale;
            }
            // sc_sig = sc_down( gelu(sc_gate·normed) ⊙ (sc_up·normed) )
            let mut normed = vec![0f32; d_model];
            rmsnorm(&soft, Some(pre_norm), eps, &mut normed);
            let mut g = vec![0f32; *n_ff];
            matvec(gate, d_model, *n_ff, &normed, &mut g);
            let mut u = vec![0f32; *n_ff];
            matvec(up, d_model, *n_ff, &normed, &mut u);
            let mut act = vec![0f32; *n_ff];
            geglu_split(&g, &u, &mut act);
            let mut sc_sig = vec![0f32; d_model];
            matvec(down, *n_ff, d_model, &act, &mut sc_sig);
            add_into(&mut combined, &sc_sig);
        }
        let mut tmp = vec![0f32; d_model];
        rmsnorm(&combined, None, eps, &mut tmp);
        hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&tmp);
    }

    // ---- 3. transformer layers ----
    for layer in 0..base.n_layers {
        let prefix = format!("blk.{layer}.");
        let n_kv = base.n_kv_heads(layer) as usize;
        let head_dim = base.head_dim(layer) as usize;
        let swa_layer = matches!(base.kind(layer), LayerKind::SlidingWindow);
        let residual = hidden.clone();

        // attn_norm (weighted) per row
        let attn_norm_w = weights.load(&format!("{prefix}attn_norm.weight"))?;
        let mut nx = hidden.clone();
        rmsnorm_rows(&mut nx, Some(&attn_norm_w), eps, n, d_model);
        drop(attn_norm_w);

        // Q/K/V projections + norms + RoPE, per position.
        let q_w = weights.load(&format!("{prefix}attn_q.weight"))?;
        let q_norm_w = weights.load(&format!("{prefix}attn_q_norm.weight"))?;
        let k_w = weights.load(&format!("{prefix}attn_k.weight"))?;
        let k_norm_w = weights.load(&format!("{prefix}attn_k_norm.weight"))?;
        let v_w = weights.load_opt(&format!("{prefix}attn_v.weight"))?;
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
            let x = &nx[i * d_model..(i + 1) * d_model];
            // Q: proj → per-head q_norm → rope at pos=i
            let mut q = vec![0f32; n_heads * head_dim];
            matvec(&q_w, d_model, n_heads * head_dim, x, &mut q);
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

            // K: proj (kept pre-norm for the no-V V); V := proj or raw K.
            let mut k = vec![0f32; n_kv * head_dim];
            matvec(&k_w, d_model, n_kv * head_dim, x, &mut k);
            let v = match &v_w {
                Some(vw) => {
                    let mut v = vec![0f32; n_kv * head_dim];
                    matvec(vw, d_model, n_kv * head_dim, x, &mut v);
                    v
                }
                None => k.clone(), // no-V layers: V = raw K projection (before K-norm)
            };
            // K-norm (weighted) per head, then rope; V-norm (unweighted) per head, no rope.
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
        drop(q_w);
        drop(q_norm_w);
        drop(k_w);
        drop(k_norm_w);

        // Region-masked attention over the full sequence.
        let attn = masked_attention(
            &q_all, &k_all, &v_all, n, n_heads, n_kv, head_dim, p, n_swa, swa_layer,
        );

        // Output projection + attn_post_norm + residual, per row.
        let o_w = weights.load(&format!("{prefix}attn_output.weight"))?;
        let post_attn_w = weights.load(&format!("{prefix}post_attention_norm.weight"))?;
        for i in 0..n {
            let a = &attn[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            let mut o = vec![0f32; d_model];
            matvec(&o_w, n_heads * head_dim, d_model, a, &mut o);
            let mut on = vec![0f32; d_model];
            rmsnorm(&o, Some(&post_attn_w), eps, &mut on);
            add_into(&mut on, &residual[i * d_model..(i + 1) * d_model]);
            hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&on);
        }
        drop(o_w);
        drop(post_attn_w);

        // FFN: gemma4 parallel dense-MLP + 128-expert MoE (Phase B), per row.
        let is_moe = base.has_moe() && layer_has_moe(weights, layer);
        let ffn_residual = hidden.clone();
        ffn_block(
            weights,
            base,
            layer,
            &mut hidden,
            &ffn_residual,
            n,
            d_model,
            eps,
            is_moe,
        )?;

        // Region-aware dual per-layer scale: prompt rows × enc, canvas rows × dec.
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

        // Optional per-layer dump (canvas rows) for bisecting vs the oracle's
        // l_out-<layer> activations.
        if let Ok(dir) = std::env::var("DG_MINE_LAYERS") {
            let mut buf = Vec::with_capacity(c * d_model * 4);
            for i in p..n {
                for &v in &hidden[i * d_model..(i + 1) * d_model] {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            let _ = std::fs::write(format!("{dir}/mine-{layer}.bin"), &buf);
        }
    }

    // ---- 4. output norm + lm_head + softcap, CANVAS rows only ----
    let out_norm_w = weights.load("output_norm.weight")?;
    let out_w_name = if weights.has("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight" // tied
    };
    let vocab = base.vocab_size as usize;
    // Load the lm_head weight ONCE (vocab × d_model ≈ 1.5 GB dequant) — never
    // per canvas position.
    let ow = weights.load(out_w_name)?;
    let mut logits = vec![0f32; c * vocab];
    let mut normed = vec![0f32; d_model];
    // Optional: dump the pre-lm_head result_norm (canvas rows) for bisecting
    // against the oracle's `llama_get_embeddings_ith`.
    let mut embd_dump = std::env::var("DG_MINE_EMBD").ok().map(|_| Vec::<u8>::new());
    for ci in 0..c {
        let i = p + ci;
        rmsnorm(
            &hidden[i * d_model..(i + 1) * d_model],
            Some(&out_norm_w),
            eps,
            &mut normed,
        );
        if let Some(buf) = embd_dump.as_mut() {
            for &v in &normed {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let row = &mut logits[ci * vocab..(ci + 1) * vocab];
        matvec(&ow, d_model, vocab, &normed, row);
        if base.final_logit_softcap > 0.0 {
            softcap(row, base.final_logit_softcap);
        }
    }
    if let (Ok(path), Some(buf)) = (std::env::var("DG_MINE_EMBD"), embd_dump) {
        let _ = std::fs::write(&path, &buf);
    }
    Ok(logits)
}

/// gemma4 FFN block (dense MLP, optionally + parallel MoE), in place on the
/// per-row `hidden` buffer. Mirrors `reference/forward.rs`'s MoE branch.
#[allow(clippy::too_many_arguments)]
fn ffn_block(
    weights: &Weights,
    base: &crate::model::config::Gemma4Config,
    layer: u32,
    hidden: &mut [f32],
    residual: &[f32],
    n: usize,
    d_model: usize,
    eps: f32,
    is_moe: bool,
) -> Result<()> {
    let prefix = format!("blk.{layer}.");
    let ffn_n = base.ffn(layer) as usize;
    let mlp_norm_w = weights.load(&format!("{prefix}ffn_norm.weight"))?;
    let gate_w = weights.load(&format!("{prefix}ffn_gate.weight"))?;
    let up_w = weights.load(&format!("{prefix}ffn_up.weight"))?;
    let down_w = weights.load(&format!("{prefix}ffn_down.weight"))?;
    let post_ffw_w = weights.load(&format!("{prefix}post_ffw_norm.weight"))?;
    let post1 = if is_moe {
        Some(
            weights
                .load_opt(&format!("{prefix}post_ffw_norm_1.weight"))?
                .or(weights.load_opt(&format!("{prefix}ffn_post_norm_1.weight"))?)
                .ok_or_else(|| {
                    crate::error::RullamaError::Inference(format!(
                        "MoE layer {layer}: missing post_ffw_norm_1"
                    ))
                })?,
        )
    } else {
        None
    };
    let pre2 = if is_moe {
        Some(
            weights
                .load_opt(&format!("{prefix}pre_ffw_norm_2.weight"))?
                .or(weights.load_opt(&format!("{prefix}ffn_pre_norm_2.weight"))?)
                .ok_or_else(|| {
                    crate::error::RullamaError::Inference(format!(
                        "MoE layer {layer}: missing pre_ffw_norm_2"
                    ))
                })?,
        )
    } else {
        None
    };
    let post2 = if is_moe {
        Some(
            weights
                .load_opt(&format!("{prefix}post_ffw_norm_2.weight"))?
                .or(weights.load_opt(&format!("{prefix}ffn_post_norm_2.weight"))?)
                .ok_or_else(|| {
                    crate::error::RullamaError::Inference(format!(
                        "MoE layer {layer}: missing post_ffw_norm_2"
                    ))
                })?,
        )
    } else {
        None
    };

    // Dense MLP for every position (and, when MoE, stash mlp_out + the routed
    // selection + the pre_ffw_norm_2 input so the experts can run EXPERT-MAJOR
    // below — each unique expert loaded once per layer instead of once per
    // (position × slot), the difference between minutes and an hour at 256
    // positions × 8 experts × 30 layers).
    let mut mlp_out_all = vec![0f32; n * d_model];
    let mut moe_x_all = if is_moe {
        vec![0f32; n * d_model]
    } else {
        Vec::new()
    };
    let mut selected_all: Vec<Vec<(usize, f32)>> = if is_moe {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    // router weight + scale loaded ONCE per layer (route() would reload per call)
    let (router_w, router_scale) = if is_moe {
        (
            weights.load(&format!("{prefix}ffn_gate_inp.weight"))?,
            weights.load_opt(&format!("{prefix}ffn_gate_inp.scale"))?,
        )
    } else {
        (Vec::new(), None)
    };
    let n_experts = base.expert_count as usize;
    let top_k = base.expert_used_count as usize;
    let inv_sqrt_d = 1.0 / (d_model as f32).sqrt();

    for i in 0..n {
        let h = &hidden[i * d_model..(i + 1) * d_model];
        let mut x = vec![0f32; d_model];
        rmsnorm(h, Some(&mlp_norm_w), eps, &mut x);
        let mut gate = vec![0f32; ffn_n];
        matvec(&gate_w, d_model, ffn_n, &x, &mut gate);
        let mut up = vec![0f32; ffn_n];
        matvec(&up_w, d_model, ffn_n, &x, &mut up);
        let mut act = vec![0f32; ffn_n];
        geglu_split(&gate, &up, &mut act);
        matvec(
            &down_w,
            ffn_n,
            d_model,
            &act,
            &mut mlp_out_all[i * d_model..(i + 1) * d_model],
        );

        if is_moe {
            // inline route (router weight cached): unweighted rmsnorm → ×1/√d →
            // ×scale → router matvec → softmax+top-k+renorm.
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
            // pre_ffw_norm_2(hidden) is the expert input.
            rmsnorm(
                h,
                Some(pre2.as_ref().unwrap()),
                eps,
                &mut moe_x_all[i * d_model..(i + 1) * d_model],
            );
        }
    }

    if is_moe {
        // Expert-major accumulation.
        let fused_name = format!("{prefix}ffn_gate_up_exps.weight");
        let fused = weights.has(&fused_name);
        let down_scale = match weights.load_opt(&format!("{prefix}ffn_down_exps.scale"))? {
            Some(s) => Some(s),
            None => weights.load_opt(&format!("{prefix}ffn_gate_inp.per_expert_scale"))?,
        };
        // expert -> [(position, routing_weight)]
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
                let x = &moe_x_all[pos * d_model..(pos + 1) * d_model];
                let mut g = vec![0f32; n_ff];
                matvec(&gate_e, d_model, n_ff, x, &mut g);
                let mut u = vec![0f32; n_ff];
                matvec(&up_e, d_model, n_ff, x, &mut u);
                let mut a = vec![0f32; n_ff];
                geglu_split(&g, &u, &mut a);
                let mut d = vec![0f32; d_model];
                matvec(&down, n_ff, d_model, &a, &mut d);
                let acc = &mut moe_acc[pos * d_model..(pos + 1) * d_model];
                for (o, dv) in acc.iter_mut().zip(d.iter()) {
                    *o += w * scale_e * dv;
                }
            }
        }
        // Per position: post-norms + combine + outer post_ffw_norm + residual.
        for i in 0..n {
            let mut mlp_normed = vec![0f32; d_model];
            rmsnorm(
                &mlp_out_all[i * d_model..(i + 1) * d_model],
                Some(post1.as_ref().unwrap()),
                eps,
                &mut mlp_normed,
            );
            let mut moe_normed = vec![0f32; d_model];
            rmsnorm(
                &moe_acc[i * d_model..(i + 1) * d_model],
                Some(post2.as_ref().unwrap()),
                eps,
                &mut moe_normed,
            );
            add_into(&mut mlp_normed, &moe_normed);
            let mut h3 = vec![0f32; d_model];
            rmsnorm(&mlp_normed, Some(&post_ffw_w), eps, &mut h3);
            add_into(&mut h3, &residual[i * d_model..(i + 1) * d_model]);
            hidden[i * d_model..(i + 1) * d_model].copy_from_slice(&h3);
        }
    } else {
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bidirectional (all-canvas, no prompt) attention on a hand-computed
    /// 2-token / 1-head / head_dim-2 case.
    #[test]
    fn masked_attention_bidirectional_hand_case() {
        // q0=[1,0] q1=[0,1]; k0=[1,0] k1=[0,1]; v0=[10,20] v1=[30,40].
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        // prompt_len 0 ⇒ both tokens are canvas ⇒ bidirectional (global).
        let out = masked_attention(&q, &k, &v, 2, 1, 1, 2, 0, 1024, false);

        // token0: softmax([q0·k0, q0·k1]) = softmax([1,0]).
        let (a, b) = {
            let e = 1f32.exp();
            (e / (e + 1.0), 1.0 / (e + 1.0))
        };
        let exp0 = [a * 10.0 + b * 30.0, a * 20.0 + b * 40.0];
        // token1: softmax([0,1]) = (b, a).
        let exp1 = [b * 10.0 + a * 30.0, b * 20.0 + a * 40.0];
        for (i, &e) in exp0.iter().chain(exp1.iter()).enumerate() {
            assert!((out[i] - e).abs() < 1e-5, "out[{i}]={} != {e}", out[i]);
        }
    }

    /// A prompt token (causal) must ignore the canvas; the result equals
    /// attending only over earlier prompt — verifiable by deleting the canvas.
    #[test]
    fn prompt_row_ignores_canvas() {
        // P=1 prompt + 1 canvas. token0 is prompt: sees only itself (causal,
        // never canvas) ⇒ out0 == v0 exactly regardless of canvas content.
        let q = vec![0.5, 0.5, 1.0, 0.0];
        let k = vec![0.3, 0.7, 9.9, 9.9];
        let v = vec![10.0, 20.0, 999.0, 999.0];
        let out = masked_attention(&q, &k, &v, 2, 1, 1, 2, 1, 1024, false);
        assert!((out[0] - 10.0).abs() < 1e-5, "prompt out0[0]={}", out[0]);
        assert!((out[1] - 20.0).abs() < 1e-5, "prompt out0[1]={}", out[1]);
    }

    /// GQA: 2 query heads share 1 kv head — both heads read the same K/V.
    #[test]
    fn gqa_two_query_heads_one_kv() {
        // n_tokens=1 (self-attend), 2 heads, 1 kv head, head_dim=1.
        // Single token attends to itself → out = v for both heads.
        let q = vec![3.0, 7.0]; // [tok0][h0,h1]
        let k = vec![1.0];
        let v = vec![42.0];
        let out = masked_attention(&q, &k, &v, 1, 2, 1, 1, 0, 1024, false);
        assert!((out[0] - 42.0).abs() < 1e-5);
        assert!((out[1] - 42.0).abs() < 1e-5);
    }
}
