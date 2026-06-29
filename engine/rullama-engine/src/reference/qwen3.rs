//! CPU f32 forward for the Qwen3 text encoder (Z-Image's prompt encoder).
//!
//! Exact port of Ollama `x/imagegen/models/qwen3/text_encoder.go`. The oracle
//! for the eventual GPU encoder. Architecture (real Z-Image config):
//! hidden 2560, 36 layers, GQA 32 q / 8 kv heads, head_dim 128, SwiGLU
//! intermediate 9728, RMSNorm eps 1e-6, half-split (NeoX) RoPE θ=1e6, per-head
//! QK-RMSNorm before RoPE, causal attention, pre-norm residual blocks.
//!
//! Weights are read on demand from a [`ShardedSafetensors`] and dequantized to
//! f32 per tensor (slow — it's a reference). The encoder output is the final
//! RMSNorm hidden state `[seq, hidden]`, which becomes the DiT's caption
//! features (`cap_feat_dim == hidden == 2560`).

use crate::error::{Result, RullamaError};
use crate::imagegen::config::Qwen3Config;
use crate::imagegen::sharded::ShardedSafetensors;

/// Qwen3 encoder over a sharded safetensors weight set.
pub struct Qwen3Encoder<'a> {
    st: &'a ShardedSafetensors,
    cfg: &'a Qwen3Config,
}

impl<'a> Qwen3Encoder<'a> {
    pub fn new(st: &'a ShardedSafetensors, cfg: &'a Qwen3Config) -> Self {
        Self { st, cfg }
    }

    /// Encode token ids → final hidden state, row-major `[seq * hidden]`.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let h = cfg.hidden_size as usize;
        let seq = tokens.len();
        if seq == 0 {
            return Err(RullamaError::Image("empty token sequence".into()));
        }
        let hd = cfg.head_dim as usize;
        let nq = cfg.num_attention_heads as usize;
        let nkv = cfg.num_key_value_heads as usize;
        let eps = cfg.rms_norm_eps;

        // ---- embedding lookup (dequant only the needed rows) ----
        let embed_bytes = self.st.tensor_bytes("model.embed_tokens.weight")?;
        let edt = self.st.dtype("model.embed_tokens.weight")?;
        let esz = edt.elem_size();
        let mut x = vec![0.0f32; seq * h];
        for (t, &tok) in tokens.iter().enumerate() {
            let r = tok as usize;
            let row = &embed_bytes[r * h * esz..(r + 1) * h * esz];
            let f = edt.dequant_to_f32(row)?;
            x[t * h..(t + 1) * h].copy_from_slice(&f);
        }
        drop(embed_bytes);

        // ---- transformer layers ----
        for li in 0..cfg.num_hidden_layers as usize {
            let p = format!("model.layers.{li}");
            // attention sub-block
            let normed = rmsnorm(
                &x,
                seq,
                h,
                &self.w(&format!("{p}.input_layernorm.weight"))?,
                eps,
            );
            let attn = self.attention(&normed, seq, &p, nq, nkv, hd)?;
            for i in 0..seq * h {
                x[i] += attn[i];
            }
            // mlp sub-block
            let normed = rmsnorm(
                &x,
                seq,
                h,
                &self.w(&format!("{p}.post_attention_layernorm.weight"))?,
                eps,
            );
            let mlp = self.mlp(&normed, seq, &p)?;
            for i in 0..seq * h {
                x[i] += mlp[i];
            }
        }

        // ---- final norm ----
        let xf = rmsnorm(&x, seq, h, &self.w("model.norm.weight")?, eps);
        Ok(xf)
    }

    fn w(&self, name: &str) -> Result<Vec<f32>> {
        self.st.tensor_f32(name)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        x: &[f32], // [seq, h] (normed)
        seq: usize,
        p: &str,
        nq: usize,
        nkv: usize,
        hd: usize,
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size as usize;
        let qd = nq * hd;
        let kvd = nkv * hd;

        // projections: weight [out, in], y = x · Wᵀ
        let mut q = linear(
            x,
            seq,
            h,
            &self.w(&format!("{p}.self_attn.q_proj.weight"))?,
            qd,
        );
        let mut k = linear(
            x,
            seq,
            h,
            &self.w(&format!("{p}.self_attn.k_proj.weight"))?,
            kvd,
        );
        let v = linear(
            x,
            seq,
            h,
            &self.w(&format!("{p}.self_attn.v_proj.weight"))?,
            kvd,
        );

        // per-head QK RMSNorm (over head_dim) with hardcoded 1e-6, then RoPE.
        let qn = self.w(&format!("{p}.self_attn.q_norm.weight"))?;
        let kn = self.w(&format!("{p}.self_attn.k_norm.weight"))?;
        head_rmsnorm(&mut q, seq, nq, hd, &qn, 1e-6);
        head_rmsnorm(&mut k, seq, nkv, hd, &kn, 1e-6);
        rope_neox(&mut q, seq, nq, hd, self.cfg.rope_theta);
        rope_neox(&mut k, seq, nkv, hd, self.cfg.rope_theta);

        // causal scaled-dot-product attention with GQA, per query head.
        let scale = 1.0f32 / (hd as f32).sqrt();
        let group = nq / nkv;
        let mut ctx = vec![0.0f32; seq * nq * hd];
        for qh in 0..nq {
            let kvh = qh / group;
            for ti in 0..seq {
                // scores over keys 0..=ti (causal)
                let mut scores = vec![f32::NEG_INFINITY; seq];
                let mut maxs = f32::NEG_INFINITY;
                for tj in 0..=ti {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[(ti * nq + qh) * hd + d] * k[(tj * nkv + kvh) * hd + d];
                    }
                    let s = dot * scale;
                    scores[tj] = s;
                    if s > maxs {
                        maxs = s;
                    }
                }
                let mut sum = 0.0f32;
                for tj in 0..=ti {
                    scores[tj] = (scores[tj] - maxs).exp();
                    sum += scores[tj];
                }
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for tj in 0..=ti {
                        acc += scores[tj] * v[(tj * nkv + kvh) * hd + d];
                    }
                    ctx[(ti * nq + qh) * hd + d] = acc / sum;
                }
            }
        }

        // o_proj: [h, qd]
        Ok(linear(
            &ctx,
            seq,
            qd,
            &self.w(&format!("{p}.self_attn.o_proj.weight"))?,
            h,
        ))
    }

    fn mlp(&self, x: &[f32], seq: usize, p: &str) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size as usize;
        let inter = self.cfg.intermediate_size as usize;
        let gate = linear(
            x,
            seq,
            h,
            &self.w(&format!("{p}.mlp.gate_proj.weight"))?,
            inter,
        );
        let up = linear(
            x,
            seq,
            h,
            &self.w(&format!("{p}.mlp.up_proj.weight"))?,
            inter,
        );
        let mut hmid = vec![0.0f32; seq * inter];
        for i in 0..seq * inter {
            let g = gate[i];
            let silu = g / (1.0 + (-g).exp());
            hmid[i] = silu * up[i];
        }
        Ok(linear(
            &hmid,
            seq,
            inter,
            &self.w(&format!("{p}.mlp.down_proj.weight"))?,
            h,
        ))
    }
}

// ---- f32 ops ----

/// Per-row RMSNorm: `y = x / sqrt(mean(x²) + eps) * w`.
fn rmsnorm(x: &[f32], rows: usize, dim: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / dim as f64;
        let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
        for c in 0..dim {
            out[r * dim + c] = row[c] * inv * w[c];
        }
    }
    out
}

/// RMSNorm applied independently to each head slice of `[seq, heads, hd]`
/// (weight length `hd`). Qwen3 QK-norm.
fn head_rmsnorm(x: &mut [f32], seq: usize, heads: usize, hd: usize, w: &[f32], eps: f32) {
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * hd;
            let slice = &x[base..base + hd];
            let ms = slice.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / hd as f64;
            let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
            for d in 0..hd {
                x[base + d] = x[base + d] * inv * w[d];
            }
        }
    }
}

/// Linear `y[r,o] = Σ_i x[r,i] * w[o,i]`; weight is row-major `[out, in]`.
fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += xr[i] * wr[i];
            }
            y[r * out_dim + o] = acc;
        }
    }
    y
}

/// Half-split (NeoX) RoPE over `[seq, heads, hd]`, θ as given. Matches
/// `applyRoPEQwen3`: freqs[i]=exp(-ln(θ)·i/half), rotate (x1,x2) halves.
fn rope_neox(x: &mut [f32], seq: usize, heads: usize, hd: usize, theta: f32) {
    let half = hd / 2;
    let ln_theta = (theta as f64).ln();
    let freqs: Vec<f64> = (0..half)
        .map(|i| (-ln_theta * (i as f64) / (half as f64)).exp())
        .collect();
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * hd;
            for i in 0..half {
                let ang = (t as f64) * freqs[i];
                let (s, c) = (ang.sin() as f32, ang.cos() as f32);
                let x1 = x[base + i];
                let x2 = x[base + half + i];
                x[base + i] = x1 * c - x2 * s;
                x[base + half + i] = x1 * s + x2 * c;
            }
        }
    }
}
