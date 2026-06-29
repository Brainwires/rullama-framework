//! Full-sequence bidirectional encoder forward for EmbeddingGemma.
//!
//! Unlike the autoregressive Gemma 4 oracle (`reference/forward.rs`), this
//! processes ALL `T` positions at once with no KV cache and no causal mask.
//! Each layer mirrors the Gemma 3 block: pre-norm → attention (QK-norm + RoPE)
//! → post-attn-norm → residual → pre-FFN-norm → GeGLU → post-FFN-norm →
//! residual. After the stack: output_norm, mean-pool, dense head, L2-normalize.

use super::{EmbedModel, LayerKind, PoolingType};
use crate::error::Result;
use crate::reference::ops::{add_into, geglu_split, matvec, rmsnorm, rope_neox, scale, softmax};

// Attention logit scale = `1/sqrt(head_dim)` (gemma3 `query_pre_attn_scalar`
// = key_length = 256, so 1/16). Confirmed optimal by a parity sweep against
// Ollama's `/api/embed` on the same GGUF: 0.0625 maximized cosine; 1.0
// (the Gemma-4-style "absorbed into Q-norm" scale) was far worse.

impl EmbedModel {
    /// Encode `input_ids` (already BOS/EOS-wrapped by the tokenizer) into a
    /// single pooled + projected + L2-normalized embedding of length
    /// `min(target_dim, embed_dim)`. `target_dim = 0` ⇒ full `embed_dim`.
    pub fn embed_ids(&self, input_ids: &[u32], target_dim: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let t = input_ids.len();
        let d = cfg.d_model as usize;
        let eps = cfg.rms_eps;

        // ---- token embeddings, scaled by sqrt(d_model) ----
        let mut hidden = vec![0f32; t * d];
        let embd_scale = (d as f32).sqrt();
        for (p, &id) in input_ids.iter().enumerate() {
            let row = self.weights.load_row("token_embd.weight", id as usize)?;
            let dst = &mut hidden[p * d..(p + 1) * d];
            for k in 0..d {
                dst[k] = row[k] * embd_scale;
            }
        }

        // ---- transformer layers ----
        for layer in 0..cfg.n_layers {
            self.layer_forward(layer, t, &mut hidden)?;
        }

        // ---- final output norm (per token) ----
        let out_norm = self.t("output_norm.weight")?;
        let mut normed = vec![0f32; t * d];
        for p in 0..t {
            rmsnorm(
                &hidden[p * d..(p + 1) * d],
                Some(&out_norm),
                eps,
                &mut normed[p * d..(p + 1) * d],
            );
        }

        // ---- pooling ----
        let pooled = self.pool(&normed, t);

        // ---- dense projection head: 768 -> 3072 -> 768 ----
        let projected = self.dense_head(&pooled)?;

        // ---- Matryoshka truncation + L2 normalize ----
        let keep = if target_dim == 0 {
            projected.len()
        } else {
            target_dim.min(projected.len())
        };
        let mut out = projected[..keep].to_vec();
        l2_normalize(&mut out);
        Ok(out)
    }

    /// One Gemma 3 encoder block, in place over `hidden` (`[T, d_model]`).
    fn layer_forward(&self, layer: u32, t: usize, hidden: &mut [f32]) -> Result<()> {
        let cfg = &self.cfg;
        let d = cfg.d_model as usize;
        let eps = cfg.rms_eps;
        let prefix = format!("blk.{layer}.");

        // ===== ATTENTION =====
        let residual = hidden.to_vec();

        let attn_norm = self.t(&format!("{prefix}attn_norm.weight"))?;
        let mut x = vec![0f32; t * d];
        for p in 0..t {
            rmsnorm(
                &hidden[p * d..(p + 1) * d],
                Some(&attn_norm),
                eps,
                &mut x[p * d..(p + 1) * d],
            );
        }

        let attn_out = self.attention(layer, t, &x)?;

        let post_attn = self.t(&format!("{prefix}post_attention_norm.weight"))?;
        for p in 0..t {
            let mut h2 = vec![0f32; d];
            rmsnorm(
                &attn_out[p * d..(p + 1) * d],
                Some(&post_attn),
                eps,
                &mut h2,
            );
            add_into(&mut h2, &residual[p * d..(p + 1) * d]);
            hidden[p * d..(p + 1) * d].copy_from_slice(&h2);
        }

        // ===== MLP =====
        let residual = hidden.to_vec();
        let ffn_n = cfg.ffn as usize;

        let ffn_norm = self.t(&format!("{prefix}ffn_norm.weight"))?;
        let gate_w = self.t(&format!("{prefix}ffn_gate.weight"))?;
        let up_w = self.t(&format!("{prefix}ffn_up.weight"))?;
        let down_w = self.t(&format!("{prefix}ffn_down.weight"))?;
        let post_ffw = self.t(&format!("{prefix}post_ffw_norm.weight"))?;

        for p in 0..t {
            let mut xn = vec![0f32; d];
            rmsnorm(&hidden[p * d..(p + 1) * d], Some(&ffn_norm), eps, &mut xn);

            let mut gate = vec![0f32; ffn_n];
            matvec(&gate_w, d, ffn_n, &xn, &mut gate);
            let mut up = vec![0f32; ffn_n];
            matvec(&up_w, d, ffn_n, &xn, &mut up);

            let mut act = vec![0f32; ffn_n];
            geglu_split(&gate, &up, &mut act);

            let mut mlp_out = vec![0f32; d];
            matvec(&down_w, ffn_n, d, &act, &mut mlp_out);

            let mut h3 = vec![0f32; d];
            rmsnorm(&mlp_out, Some(&post_ffw), eps, &mut h3);
            add_into(&mut h3, &residual[p * d..(p + 1) * d]);
            hidden[p * d..(p + 1) * d].copy_from_slice(&h3);
        }

        Ok(())
    }

    /// Bidirectional multi-head attention over the full sequence. Q-norm and
    /// K-norm are weighted RMSNorm per head; RoPE applied per position; SWA
    /// layers use a symmetric sliding window. Returns `[T, d_model]`.
    fn attention(&self, layer: u32, t: usize, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let d = cfg.d_model as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv = cfg.n_kv_heads as usize;
        let hd = cfg.head_dim as usize;
        let eps = cfg.rms_eps;
        let prefix = format!("blk.{layer}.");
        let heads_per_kv = n_heads / n_kv;

        let q_w = self.t(&format!("{prefix}attn_q.weight"))?;
        let k_w = self.t(&format!("{prefix}attn_k.weight"))?;
        let v_w = self.t(&format!("{prefix}attn_v.weight"))?;
        let q_norm = self.t(&format!("{prefix}attn_q_norm.weight"))?;
        let k_norm = self.t(&format!("{prefix}attn_k_norm.weight"))?;
        let o_w = self.t(&format!("{prefix}attn_output.weight"))?;

        // Per-position Q/K/V with QK-norm + RoPE. Layout: [T, heads*hd].
        let mut q_all = vec![0f32; t * n_heads * hd];
        let mut k_all = vec![0f32; t * n_kv * hd];
        let mut v_all = vec![0f32; t * n_kv * hd];

        let base = cfg.rope_base;
        for p in 0..t {
            let xp = &x[p * d..(p + 1) * d];

            // Q
            let mut q = vec![0f32; n_heads * hd];
            matvec(&q_w, d, n_heads * hd, xp, &mut q);
            let mut qn = vec![0f32; n_heads * hd];
            for h in 0..n_heads {
                rmsnorm(
                    &q[h * hd..(h + 1) * hd],
                    Some(&q_norm),
                    eps,
                    &mut qn[h * hd..(h + 1) * hd],
                );
            }
            rope_neox(&mut qn, hd, n_heads, p, hd, base, None);
            q_all[p * n_heads * hd..(p + 1) * n_heads * hd].copy_from_slice(&qn);

            // K
            let mut k = vec![0f32; n_kv * hd];
            matvec(&k_w, d, n_kv * hd, xp, &mut k);
            let mut kn = vec![0f32; n_kv * hd];
            for h in 0..n_kv {
                rmsnorm(
                    &k[h * hd..(h + 1) * hd],
                    Some(&k_norm),
                    eps,
                    &mut kn[h * hd..(h + 1) * hd],
                );
            }
            rope_neox(&mut kn, hd, n_kv, p, hd, base, None);
            k_all[p * n_kv * hd..(p + 1) * n_kv * hd].copy_from_slice(&kn);

            // V (no norm in gemma3)
            let mut v = vec![0f32; n_kv * hd];
            matvec(&v_w, d, n_kv * hd, xp, &mut v);
            v_all[p * n_kv * hd..(p + 1) * n_kv * hd].copy_from_slice(&v);
        }

        let scale_f = 1.0 / (hd as f32).sqrt();
        let is_swa = matches!(cfg.kind(layer), LayerKind::SlidingWindow);
        let window = cfg.sliding_window as usize;

        // Attention output per position. [T, n_heads*hd].
        let mut ctx = vec![0f32; t * n_heads * hd];
        let mut scores = vec![0f32; t];
        for qh in 0..n_heads {
            let kvh = qh / heads_per_kv;
            for i in 0..t {
                // scores over all j (bidirectional), masked by symmetric window.
                for j in 0..t {
                    let within = if cfg.causal {
                        j <= i && (!is_swa || i - j < window)
                    } else if is_swa {
                        i.abs_diff(j) < window
                    } else {
                        true
                    };
                    if !within {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let q_off = i * n_heads * hd + qh * hd;
                    let k_off = j * n_kv * hd + kvh * hd;
                    let mut acc = 0f32;
                    for dd in 0..hd {
                        acc += q_all[q_off + dd] * k_all[k_off + dd];
                    }
                    scores[j] = acc * scale_f;
                }
                softmax(&mut scores);

                let out_off = i * n_heads * hd + qh * hd;
                for dd in 0..hd {
                    ctx[out_off + dd] = 0.0;
                }
                for j in 0..t {
                    let w = scores[j];
                    if w == 0.0 {
                        continue;
                    }
                    let v_off = j * n_kv * hd + kvh * hd;
                    for dd in 0..hd {
                        ctx[out_off + dd] += w * v_all[v_off + dd];
                    }
                }
            }
        }

        // output projection per position
        let mut out = vec![0f32; t * d];
        for p in 0..t {
            matvec(
                &o_w,
                n_heads * hd,
                d,
                &ctx[p * n_heads * hd..(p + 1) * n_heads * hd],
                &mut out[p * d..(p + 1) * d],
            );
        }
        Ok(out)
    }

    /// Pool the per-token final hidden states `[T, d_model]` into `[d_model]`.
    fn pool(&self, normed: &[f32], t: usize) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        match self.cfg.pooling {
            PoolingType::Mean | PoolingType::None => {
                let mut pooled = vec![0f32; d];
                for p in 0..t {
                    for k in 0..d {
                        pooled[k] += normed[p * d + k];
                    }
                }
                let inv = 1.0 / t as f32;
                scale(&mut pooled, inv);
                pooled
            }
            PoolingType::Cls => normed[..d].to_vec(),
            PoolingType::Last => normed[(t - 1) * d..t * d].to_vec(),
        }
    }

    /// Dense projection head: `dense.0` (d→inter) then `dense.1` (inter→d).
    /// EmbeddingGemma's bottleneck is two linear layers with no activation
    /// between (verified against the parity target).
    fn dense_head(&self, pooled: &[f32]) -> Result<Vec<f32>> {
        let d = self.cfg.d_model as usize;
        let w0 = self.t("dense.0.weight")?;
        let inter = w0.len() / d; // [d, inter] in matvec terms
        let mut mid = vec![0f32; inter];
        matvec(&w0, d, inter, pooled, &mut mid);

        let w1 = self.t("dense.1.weight")?;
        let out_d = w1.len() / inter;
        let mut out = vec![0f32; out_d];
        matvec(&w1, inter, out_d, &mid, &mut out);
        Ok(out)
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut sumsq = 0f64;
    for &x in v.iter() {
        sumsq += (x as f64) * (x as f64);
    }
    let norm = sumsq.sqrt() as f32;
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
