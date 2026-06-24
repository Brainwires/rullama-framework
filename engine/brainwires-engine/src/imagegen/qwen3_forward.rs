//! Async-streaming GPU forward for the Qwen3 text encoder (Z-Image's prompt
//! encoder) — the wasm-facing path; `reference::qwen3::Qwen3Encoder` is its
//! parity oracle.
//!
//! Same hybrid as the DiT path: the dominant projections (q/k/v/o, gate/up/down)
//! dispatch to the GPU bf16 matmul kernel; the sequential glue (RMSNorm, per-head
//! QK-norm, half-split RoPE, causal GQA attention, SwiGLU) stays on the CPU.
//! Runs once per prompt (no KV cache); output is the final RMSNorm hidden state
//! `[seq, hidden]` = the DiT's caption features.
//!
//! Weights stream per tensor from a [`StreamingShards`] over any [`BlobSource`];
//! the embedding table is read one row per token (`tensor_byte_range`) so the
//! ~780 MB table never lands in memory whole.

use wgpu::util::DeviceExt;

use crate::backend::dispatch::{
    add_bias_batched_chained, make_storage_rw, matmul_bf16_batched_chained, read_back_f32,
};
use crate::backend::{Pipelines, WgpuCtx};
use crate::error::{Result, RullamaError};
use crate::imagegen::config::Qwen3Config;
use crate::imagegen::dtype::StDtype;
use crate::imagegen::source::BlobSource;
use crate::imagegen::streaming::StreamingShards;

/// Async GPU Qwen3 encoder streaming weights from a `BlobSource`.
pub struct Qwen3Gpu<'a, S: BlobSource> {
    ctx: &'a WgpuCtx,
    pipes: &'a Pipelines,
    st: &'a StreamingShards<S>,
    cfg: &'a Qwen3Config,
}

impl<'a, S: BlobSource> Qwen3Gpu<'a, S> {
    pub fn new(
        ctx: &'a WgpuCtx,
        pipes: &'a Pipelines,
        st: &'a StreamingShards<S>,
        cfg: &'a Qwen3Config,
    ) -> Self {
        Self {
            ctx,
            pipes,
            st,
            cfg,
        }
    }

    /// Encode token ids → final hidden state, row-major `[seq * hidden]`.
    /// `report(done, total)` (optional) is called per transformer layer so the
    /// UI shows progress during the streamed forward.
    pub async fn forward(
        &self,
        tokens: &[u32],
        report: Option<crate::imagegen::Reporter<'_>>,
    ) -> Result<Vec<f32>> {
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

        // ---- embedding lookup: one row per token (range-read, no bulk table) ----
        let edt = self
            .st
            .dtype("model.embed_tokens.weight")
            .ok_or_else(|| RullamaError::Image("embed_tokens missing".into()))?;
        let esz = edt.elem_size();
        let mut x = vec![0.0f32; seq * h];
        for (t, &tok) in tokens.iter().enumerate() {
            let off = (tok as u64) * (h as u64) * (esz as u64);
            let row = self
                .st
                .tensor_byte_range("model.embed_tokens.weight", off, (h * esz) as u64)
                .await?;
            let f = edt.dequant_to_f32(&row)?;
            x[t * h..(t + 1) * h].copy_from_slice(&f);
        }

        // ---- transformer layers ----
        let n_layers = cfg.num_hidden_layers as usize;
        for li in 0..n_layers {
            if let Some(r) = report {
                r(li, n_layers);
            }
            let p = format!("model.layers.{li}");
            let normed = rmsnorm(
                &x,
                seq,
                h,
                &self.w(&format!("{p}.input_layernorm.weight")).await?,
                eps,
            );
            let attn = self.attention(&normed, seq, &p, nq, nkv, hd).await?;
            for i in 0..seq * h {
                x[i] += attn[i];
            }
            let normed = rmsnorm(
                &x,
                seq,
                h,
                &self
                    .w(&format!("{p}.post_attention_layernorm.weight"))
                    .await?,
                eps,
            );
            let mlp = self.mlp(&normed, seq, &p).await?;
            for i in 0..seq * h {
                x[i] += mlp[i];
            }
        }

        // ---- final norm ----
        Ok(rmsnorm(
            &x,
            seq,
            h,
            &self.w("model.norm.weight").await?,
            eps,
        ))
    }

    async fn w(&self, name: &str) -> Result<Vec<f32>> {
        self.st.tensor_f32(name).await
    }

    async fn attention(
        &self,
        x: &[f32],
        seq: usize,
        p: &str,
        nq: usize,
        nkv: usize,
        hd: usize,
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size as usize;
        let qd = nq * hd;
        let kvd = nkv * hd;

        let mut q = self
            .linear(x, seq, h, &format!("{p}.self_attn.q_proj"), qd)
            .await?;
        let mut k = self
            .linear(x, seq, h, &format!("{p}.self_attn.k_proj"), kvd)
            .await?;
        let v = self
            .linear(x, seq, h, &format!("{p}.self_attn.v_proj"), kvd)
            .await?;

        let qn = self.w(&format!("{p}.self_attn.q_norm.weight")).await?;
        let kn = self.w(&format!("{p}.self_attn.k_norm.weight")).await?;
        head_rmsnorm(&mut q, seq, nq, hd, &qn, 1e-6);
        head_rmsnorm(&mut k, seq, nkv, hd, &kn, 1e-6);
        rope_neox(&mut q, seq, nq, hd, self.cfg.rope_theta);
        rope_neox(&mut k, seq, nkv, hd, self.cfg.rope_theta);

        let scale = 1.0f32 / (hd as f32).sqrt();
        let group = nq / nkv;
        let mut ctx = vec![0.0f32; seq * nq * hd];
        for qh in 0..nq {
            let kvh = qh / group;
            for ti in 0..seq {
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
        self.linear(&ctx, seq, qd, &format!("{p}.self_attn.o_proj"), h)
            .await
    }

    async fn mlp(&self, x: &[f32], seq: usize, p: &str) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size as usize;
        let inter = self.cfg.intermediate_size as usize;
        let gate = self
            .linear(x, seq, h, &format!("{p}.mlp.gate_proj"), inter)
            .await?;
        let up = self
            .linear(x, seq, h, &format!("{p}.mlp.up_proj"), inter)
            .await?;
        let mut hmid = vec![0.0f32; seq * inter];
        for i in 0..seq * inter {
            let g = gate[i];
            hmid[i] = (g / (1.0 + (-g).exp())) * up[i];
        }
        self.linear(&hmid, seq, inter, &format!("{p}.mlp.down_proj"), h)
            .await
    }

    /// GPU bf16 matmul linear `y[r,o]=Σ x[r,i]·W[o,i]`, weight `<p>.weight`
    /// `[out,in]` (no bias in Qwen3 projections). bf16 streamed raw when stored
    /// bf16, else packed from f32.
    async fn linear(
        &self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        p: &str,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let dev = &self.ctx.device;
        let wname = format!("{p}.weight");
        let mut wb: Vec<u8> = if self.st.dtype(&wname) == Some(StDtype::Bf16) {
            self.st.tensor_bytes(&wname).await?
        } else {
            let wf = self.st.tensor_f32(&wname).await?;
            let mut b = Vec::with_capacity(wf.len() * 2);
            for &val in &wf {
                b.extend_from_slice(&half::bf16::from_f32(val).to_bits().to_le_bytes());
            }
            b
        };
        if !wb.len().is_multiple_of(4) {
            wb.extend_from_slice(&[0u8, 0u8]);
        }
        let w_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qwen.w"),
            contents: &wb,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let x_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qwen.x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_buf = make_storage_rw(dev, "qwen.y", rows * out_dim);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen.mm"),
        });
        matmul_bf16_batched_chained(
            self.ctx, self.pipes, &mut enc, &w_buf, &x_buf, &y_buf, in_dim, out_dim, rows,
        );
        let _ = add_bias_batched_chained; // (no bias in Qwen3; kept import shared with DiT)
        let read = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen.read"),
            size: (rows * out_dim * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&y_buf, 0, &read, 0, (rows * out_dim * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));
        read_back_f32(dev, &read).await
    }
}

// ---- free ops (mirror reference::qwen3; the parity test guards drift) ----

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

fn head_rmsnorm(x: &mut [f32], seq: usize, heads: usize, hd: usize, w: &[f32], eps: f32) {
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * hd;
            let ms = x[base..base + hd]
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                / hd as f64;
            let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
            for d in 0..hd {
                x[base + d] = x[base + d] * inv * w[d];
            }
        }
    }
}

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
