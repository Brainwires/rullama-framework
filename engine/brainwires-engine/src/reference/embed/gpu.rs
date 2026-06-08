//! Hybrid GPU forward for EmbeddingGemma.
//!
//! The matmuls (≈90% of the FLOPs) run on the GPU via the existing
//! `matmul_bf16_batched_chained` kernel, batched over all T positions; the
//! norms / RoPE / attention / pooling / dense head stay in CPU f32 (the
//! validated oracle code in `forward.rs`). This is far less code + risk than
//! a full buffer-chained bidirectional WGSL forward, and still moves the
//! dominant cost onto the GPU.
//!
//! Weights are uploaded once and cached for the duration of the call. The
//! method is async because GPU readback is.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use super::{EmbedModel, LayerKind};
use crate::backend::dispatch::matmul_bf16_batched_chained;
use crate::backend::dispatch::{make_storage_rw, read_back_f32, write_storage_f32};
use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::reference::ops::{geglu_split, rmsnorm, rope_neox, scale, softmax};

/// GPU weight-buffer cache (tensor name → uploaded bf16 buffer), valid for
/// one `embed_ids_gpu` call.
type WBufCache = HashMap<String, wgpu::Buffer>;

impl EmbedModel {
    /// GPU-accelerated embedding. Same math as [`EmbedModel::embed_ids`] but
    /// with the linear layers on the GPU. Returns the L2-normalized,
    /// Matryoshka-truncated vector.
    pub async fn embed_ids_gpu(
        &self,
        ctx: &WgpuCtx,
        pipes: &Pipelines,
        input_ids: &[u32],
        target_dim: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let t = input_ids.len();
        let d = cfg.d_model as usize;
        let eps = cfg.rms_eps;
        let mut wc: WBufCache = HashMap::new();

        // ---- token embeddings, scaled by sqrt(d_model) ----
        let embd_scale = (d as f32).sqrt();
        let mut hidden = vec![0f32; t * d];
        for (p, &id) in input_ids.iter().enumerate() {
            let row = self.weights.load_row("token_embd.weight", id as usize)?;
            let dst = &mut hidden[p * d..(p + 1) * d];
            for k in 0..d {
                dst[k] = row[k] * embd_scale;
            }
        }

        for layer in 0..cfg.n_layers {
            self.layer_gpu(ctx, pipes, &mut wc, layer, t, &mut hidden)
                .await?;
        }

        // ---- final output norm (per token) ----
        let out_norm = self.weights.load("output_norm.weight")?;
        let mut normed = vec![0f32; t * d];
        for p in 0..t {
            rmsnorm(
                &hidden[p * d..(p + 1) * d],
                Some(&out_norm),
                eps,
                &mut normed[p * d..(p + 1) * d],
            );
        }

        // ---- mean pool ----
        let mut pooled = vec![0f32; d];
        for p in 0..t {
            for k in 0..d {
                pooled[k] += normed[p * d + k];
            }
        }
        scale(&mut pooled, 1.0 / t as f32);

        // ---- dense head (GPU matmuls), then L2 normalize ----
        let w0_name = "dense.0.weight";
        let inter = self.weights.reader().tensor(w0_name)?.dims[1] as usize;
        let mid = self
            .gpu_matmul(ctx, pipes, &mut wc, w0_name, &pooled, d, inter, 1)
            .await?;
        let w1_name = "dense.1.weight";
        let out_d = self.weights.reader().tensor(w1_name)?.dims[1] as usize;
        let mut projected = self
            .gpu_matmul(ctx, pipes, &mut wc, w1_name, &mid, inter, out_d, 1)
            .await?;

        let keep = if target_dim == 0 {
            projected.len()
        } else {
            target_dim.min(projected.len())
        };
        projected.truncate(keep);
        l2_normalize(&mut projected);
        Ok(projected)
    }

    async fn layer_gpu(
        &self,
        ctx: &WgpuCtx,
        pipes: &Pipelines,
        wc: &mut WBufCache,
        layer: u32,
        t: usize,
        hidden: &mut [f32],
    ) -> Result<()> {
        let cfg = &self.cfg;
        let d = cfg.d_model as usize;
        let eps = cfg.rms_eps;
        let n_heads = cfg.n_heads as usize;
        let n_kv = cfg.n_kv_heads as usize;
        let hd = cfg.head_dim as usize;
        let prefix = format!("blk.{layer}.");

        // ===== attention =====
        let residual = hidden.to_vec();
        let attn_norm = self.weights.load(&format!("{prefix}attn_norm.weight"))?;
        let mut x = vec![0f32; t * d];
        for p in 0..t {
            rmsnorm(
                &hidden[p * d..(p + 1) * d],
                Some(&attn_norm),
                eps,
                &mut x[p * d..(p + 1) * d],
            );
        }

        // Q/K/V projections on GPU (batched over T).
        let q = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}attn_q.weight"),
                &x,
                d,
                n_heads * hd,
                t,
            )
            .await?;
        let k = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}attn_k.weight"),
                &x,
                d,
                n_kv * hd,
                t,
            )
            .await?;
        let v = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}attn_v.weight"),
                &x,
                d,
                n_kv * hd,
                t,
            )
            .await?;

        // QK-norm + RoPE (CPU), then bidirectional attention (CPU).
        let q_norm = self.weights.load(&format!("{prefix}attn_q_norm.weight"))?;
        let k_norm = self.weights.load(&format!("{prefix}attn_k_norm.weight"))?;
        let mut q_all = vec![0f32; t * n_heads * hd];
        let mut k_all = vec![0f32; t * n_kv * hd];
        let base = cfg.rope_base;
        for p in 0..t {
            let mut qn = vec![0f32; n_heads * hd];
            for h in 0..n_heads {
                rmsnorm(
                    &q[p * n_heads * hd + h * hd..p * n_heads * hd + (h + 1) * hd],
                    Some(&q_norm),
                    eps,
                    &mut qn[h * hd..(h + 1) * hd],
                );
            }
            rope_neox(&mut qn, hd, n_heads, p, hd, base, None);
            q_all[p * n_heads * hd..(p + 1) * n_heads * hd].copy_from_slice(&qn);

            let mut kn = vec![0f32; n_kv * hd];
            for h in 0..n_kv {
                rmsnorm(
                    &k[p * n_kv * hd + h * hd..p * n_kv * hd + (h + 1) * hd],
                    Some(&k_norm),
                    eps,
                    &mut kn[h * hd..(h + 1) * hd],
                );
            }
            rope_neox(&mut kn, hd, n_kv, p, hd, base, None);
            k_all[p * n_kv * hd..(p + 1) * n_kv * hd].copy_from_slice(&kn);
        }

        let scale_f = 1.0 / (hd as f32).sqrt();
        let is_swa = matches!(cfg.kind(layer), LayerKind::SlidingWindow);
        let window = cfg.sliding_window as usize;
        let heads_per_kv = n_heads / n_kv;
        let mut ctx_attn = vec![0f32; t * n_heads * hd];
        let mut scores = vec![0f32; t];
        for qh in 0..n_heads {
            let kvh = qh / heads_per_kv;
            for i in 0..t {
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
                    ctx_attn[out_off + dd] = 0.0;
                }
                for j in 0..t {
                    let w = scores[j];
                    if w == 0.0 {
                        continue;
                    }
                    let v_off = j * n_kv * hd + kvh * hd;
                    for dd in 0..hd {
                        ctx_attn[out_off + dd] += w * v[v_off + dd];
                    }
                }
            }
        }

        // output projection (GPU) + post-attn norm + residual (CPU).
        let attn_out = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}attn_output.weight"),
                &ctx_attn,
                n_heads * hd,
                d,
                t,
            )
            .await?;
        let post_attn = self
            .weights
            .load(&format!("{prefix}post_attention_norm.weight"))?;
        for p in 0..t {
            let mut h2 = vec![0f32; d];
            rmsnorm(
                &attn_out[p * d..(p + 1) * d],
                Some(&post_attn),
                eps,
                &mut h2,
            );
            for k in 0..d {
                hidden[p * d + k] = h2[k] + residual[p * d + k];
            }
        }

        // ===== MLP =====
        let residual = hidden.to_vec();
        let ffn_n = cfg.ffn as usize;
        let ffn_norm = self.weights.load(&format!("{prefix}ffn_norm.weight"))?;
        let mut xn = vec![0f32; t * d];
        for p in 0..t {
            rmsnorm(
                &hidden[p * d..(p + 1) * d],
                Some(&ffn_norm),
                eps,
                &mut xn[p * d..(p + 1) * d],
            );
        }
        let gate = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}ffn_gate.weight"),
                &xn,
                d,
                ffn_n,
                t,
            )
            .await?;
        let up = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}ffn_up.weight"),
                &xn,
                d,
                ffn_n,
                t,
            )
            .await?;
        let mut act = vec![0f32; t * ffn_n];
        for p in 0..t {
            geglu_split(
                &gate[p * ffn_n..(p + 1) * ffn_n],
                &up[p * ffn_n..(p + 1) * ffn_n],
                &mut act[p * ffn_n..(p + 1) * ffn_n],
            );
        }
        let mlp_out = self
            .gpu_matmul(
                ctx,
                pipes,
                wc,
                &format!("{prefix}ffn_down.weight"),
                &act,
                ffn_n,
                d,
                t,
            )
            .await?;
        let post_ffw = self
            .weights
            .load(&format!("{prefix}post_ffw_norm.weight"))?;
        for p in 0..t {
            let mut h3 = vec![0f32; d];
            rmsnorm(&mlp_out[p * d..(p + 1) * d], Some(&post_ffw), eps, &mut h3);
            for k in 0..d {
                hidden[p * d + k] = h3[k] + residual[p * d + k];
            }
        }
        Ok(())
    }

    /// `y[batch, n] = x[batch, k] · W[n, k]^T` with W a cached bf16 GPU buffer.
    async fn gpu_matmul(
        &self,
        ctx: &WgpuCtx,
        pipes: &Pipelines,
        wc: &mut WBufCache,
        weight_name: &str,
        x: &[f32],
        k: usize,
        n: usize,
        batch: usize,
    ) -> Result<Vec<f32>> {
        // Upload (and cache) the bf16 weight buffer.
        if !wc.contains_key(weight_name) {
            let bytes = self.weights.reader().tensor_bytes(weight_name)?;
            // Pad to a 4-byte multiple (the bf16 kernel binds array<u32>).
            let mut padded = bytes.to_vec();
            while padded.len() % 4 != 0 {
                padded.push(0);
            }
            let buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(weight_name),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            wc.insert(weight_name.to_string(), buf);
        }
        let w = wc.get(weight_name).unwrap();
        let xb = write_storage_f32(&ctx.device, &ctx.queue, "embed.x", x);
        let yb = make_storage_rw(&ctx.device, "embed.y", batch * n);
        let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed.read"),
            size: (batch * n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("embed.mm"),
            });
        matmul_bf16_batched_chained(ctx, pipes, &mut enc, w, &xb, &yb, k, n, batch);
        enc.copy_buffer_to_buffer(&yb, 0, &read, 0, (batch * n * 4) as u64);
        ctx.queue.submit(Some(enc.finish()));
        read_back_f32(&ctx.device, &read).await
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
