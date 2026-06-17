//! CPU f32 forward for the Z-Image S3-DiT denoiser (the velocity-predicting
//! transformer). Faithful port of Ollama `x/imagegen/models/zimage/
//! transformer.go`. The oracle for the GPU DiT.
//!
//! One denoise step: latent `[16,H,W]` + timestep `t` + caption features
//! `[cap_len, 2560]` → predicted velocity `[16,H,W]`. Pipeline:
//!   patchify → x_embed; cap_embed; t_embed(sinusoidal→MLP)
//!   → noise_refiner×2 (image, adaLN-modulated, ImgRoPE)
//!   → context_refiner×2 (caption, no modulation, CapRoPE)
//!   → concat[img, cap] → 30 main layers (adaLN, unified RoPE)
//!   → take image tokens → final layer (no-affine LN + adaLN scale) → unpatchify
//!
//! Multi-axis interleaved RoPE: θ=256, axes_dims [32,48,48] → 16/24/24 freqs
//! over (t,h,w) position coords. QK-RMSNorm eps 1e-5; block RMSNorm eps = norm_eps;
//! attention scale 1/√head_dim; tanh-gated residuals.

use crate::backend::dispatch::{
    add_bias_batched_chained, make_storage_rw, matmul_bf16_batched_chained, read_back_f32,
};
use crate::backend::{Pipelines, WgpuCtx};
use crate::error::{Result, RullamaError};
use crate::imagegen::config::TransformerConfig;
use crate::imagegen::sharded::ShardedSafetensors;
use crate::imagegen::sinusoidal_timestep_embedding;

const TEMB_DIM: usize = 256;

/// GPU handles for the matmul-accelerated path.
pub struct GpuHandles<'a> {
    pub ctx: &'a WgpuCtx,
    pub pipes: &'a Pipelines,
}

pub struct DitForward<'a> {
    st: &'a ShardedSafetensors,
    cfg: &'a TransformerConfig,
    /// When set, the (dominant) linears run on the GPU bf16 matmul; everything
    /// else stays on the validated CPU path. Output is bit-for-bit the CPU
    /// forward except for bf16-matmul rounding (~1e-2).
    gpu: Option<GpuHandles<'a>>,
}

impl<'a> DitForward<'a> {
    pub fn new(st: &'a ShardedSafetensors, cfg: &'a TransformerConfig) -> Self {
        Self { st, cfg, gpu: None }
    }

    /// Same forward, but linears dispatch to the GPU bf16 matmul kernel.
    pub fn with_gpu(
        st: &'a ShardedSafetensors,
        cfg: &'a TransformerConfig,
        ctx: &'a WgpuCtx,
        pipes: &'a Pipelines,
    ) -> Self {
        Self {
            st,
            cfg,
            gpu: Some(GpuHandles { ctx, pipes }),
        }
    }

    /// Predict velocity for one latent `[in_ch, lh, lw]` at timestep `t`, given
    /// caption features `cap [cap_len, cap_feat_dim]`. Returns `[in_ch, lh, lw]`.
    pub fn forward(
        &self,
        latent: &[f32],
        lh: usize,
        lw: usize,
        t: f32,
        cap: &[f32],
        cap_len: usize,
    ) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let dim = cfg.dim as usize;
        let cin = cfg.in_channels as usize;
        let p = cfg.patch_size() as usize;
        let eps = cfg.norm_eps;
        if latent.len() != cin * lh * lw {
            return Err(RullamaError::Image("latent size mismatch".into()));
        }
        let (ph, pw) = (lh / p, lw / p);
        let img_len = ph * pw;

        // ---- timestep embedding: sinusoidal(t*t_scale,256) → mlp0 → silu → mlp2
        let temb_in = sinusoidal_timestep_embedding(t * cfg.t_scale, TEMB_DIM);
        let h0 = self.linear(&temb_in, 1, TEMB_DIM, "t_embedder.mlp.0", 1024, true)?;
        let mut h0 = h0;
        silu_(&mut h0);
        let temb = self.linear(&h0, 1, 1024, "t_embedder.mlp.2", TEMB_DIM, true)?; // [256]

        // ---- patch embed: patchify → linear(64→dim)
        let patches = patchify(latent, cin, lh, lw, p); // [img_len, cin*p*p]
        let patch_in = cin * p * p;
        let mut x = self.linear(&patches, img_len, patch_in, "all_x_embedder.2-1", dim, true)?;

        // ---- caption embed: rmsnorm(2560,1e-6) → linear(2560→dim)
        let cn = self.w("cap_embedder.0.weight")?;
        let cnormed = rmsnorm(cap, cap_len, cfg.cap_feat_dim as usize, &cn, 1e-6);
        let mut cap_emb = self.linear(
            &cnormed,
            cap_len,
            cfg.cap_feat_dim as usize,
            "cap_embedder.1",
            dim,
            true,
        )?;

        // ---- RoPE cos/sin (unified over [img, cap] positions) ----
        let (ucos, usin) = self.rope_unified(ph, pw, cap_len);
        let half = (cfg.head_dim() / 2) as usize; // 64
        let img_cos = &ucos[..img_len * half];
        let img_sin = &usin[..img_len * half];
        let cap_cos = &ucos[img_len * half..(img_len + cap_len) * half];
        let cap_sin = &usin[img_len * half..(img_len + cap_len) * half];

        // ---- noise refiners (image, modulated) ----
        for i in 0..cfg.n_refiner_layers as usize {
            x = self.block(
                &x,
                img_len,
                Some(&temb),
                img_cos,
                img_sin,
                &format!("noise_refiner.{i}"),
                eps,
            )?;
        }
        // ---- context refiners (caption, no modulation) ----
        for i in 0..cfg.n_refiner_layers as usize {
            cap_emb = self.block(
                &cap_emb,
                cap_len,
                None,
                cap_cos,
                cap_sin,
                &format!("context_refiner.{i}"),
                eps,
            )?;
        }

        // ---- concat [img, cap], run main layers with unified RoPE ----
        let total = img_len + cap_len;
        let mut unified = vec![0.0f32; total * dim];
        unified[..img_len * dim].copy_from_slice(&x);
        unified[img_len * dim..].copy_from_slice(&cap_emb);
        for i in 0..cfg.n_layers as usize {
            unified = self.block(
                &unified,
                total,
                Some(&temb),
                &ucos,
                &usin,
                &format!("layers.{i}"),
                eps,
            )?;
        }

        // ---- take image tokens, final layer, unpatchify ----
        let img_out = &unified[..img_len * dim];
        let out_patches = self.final_layer(img_out, img_len, &temb)?; // [img_len, patch_in]
        Ok(unpatchify(&out_patches, cin, lh, lw, p))
    }

    // ---- transformer block ----
    fn block(
        &self,
        x: &[f32],
        seq: usize,
        temb: Option<&[f32]>,
        cos: &[f32],
        sin: &[f32],
        p: &str,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let dim = self.cfg.dim as usize;
        let mut out = x.to_vec();

        if let Some(temb) = temb {
            // adaLN: temb[256] → [4*dim], split scale_msa/gate_msa/scale_mlp/gate_mlp
            let chunks = self.linear(
                temb,
                1,
                TEMB_DIM,
                &format!("{p}.adaLN_modulation.0"),
                4 * dim,
                true,
            )?;
            let (s_msa, g_msa) = (&chunks[..dim], &chunks[dim..2 * dim]);
            let (s_mlp, g_mlp) = (&chunks[2 * dim..3 * dim], &chunks[3 * dim..4 * dim]);

            // attention: norm1 → ·(1+scale) → attn → norm2 → +tanh(gate)··
            let n1 = self.w(&format!("{p}.attention_norm1.weight"))?;
            let mut normed = rmsnorm(&out, seq, dim, &n1, eps);
            mod_scale(&mut normed, seq, dim, s_msa);
            let attn = self.attention(&normed, seq, cos, sin, p)?;
            let n2 = self.w(&format!("{p}.attention_norm2.weight"))?;
            let attn = rmsnorm(&attn, seq, dim, &n2, eps);
            gated_add(&mut out, seq, dim, g_msa, &attn);

            // ffn: ffn_norm1 → ·(1+scale) → swiglu → ffn_norm2 → +tanh(gate)··
            let f1 = self.w(&format!("{p}.ffn_norm1.weight"))?;
            let mut normed = rmsnorm(&out, seq, dim, &f1, eps);
            mod_scale(&mut normed, seq, dim, s_mlp);
            let ffn = self.feed_forward(&normed, seq, p)?;
            let f2 = self.w(&format!("{p}.ffn_norm2.weight"))?;
            let ffn = rmsnorm(&ffn, seq, dim, &f2, eps);
            gated_add(&mut out, seq, dim, g_mlp, &ffn);
        } else {
            let n1 = self.w(&format!("{p}.attention_norm1.weight"))?;
            let attn = self.attention(&rmsnorm(&out, seq, dim, &n1, eps), seq, cos, sin, p)?;
            let n2 = self.w(&format!("{p}.attention_norm2.weight"))?;
            let attn = rmsnorm(&attn, seq, dim, &n2, eps);
            for i in 0..seq * dim {
                out[i] += attn[i];
            }
            let f1 = self.w(&format!("{p}.ffn_norm1.weight"))?;
            let ffn = self.feed_forward(&rmsnorm(&out, seq, dim, &f1, eps), seq, p)?;
            let f2 = self.w(&format!("{p}.ffn_norm2.weight"))?;
            let ffn = rmsnorm(&ffn, seq, dim, &f2, eps);
            for i in 0..seq * dim {
                out[i] += ffn[i];
            }
        }
        Ok(out)
    }

    fn attention(
        &self,
        x: &[f32],
        seq: usize,
        cos: &[f32],
        sin: &[f32],
        p: &str,
    ) -> Result<Vec<f32>> {
        let dim = self.cfg.dim as usize;
        let nh = self.cfg.n_heads as usize;
        let hd = self.cfg.head_dim() as usize;
        let half = hd / 2;

        let mut q = self.linear(x, seq, dim, &format!("{p}.attention.to_q"), dim, false)?;
        let mut k = self.linear(x, seq, dim, &format!("{p}.attention.to_k"), dim, false)?;
        let v = self.linear(x, seq, dim, &format!("{p}.attention.to_v"), dim, false)?;

        // per-head QK RMSNorm (eps 1e-5) then interleaved 3-axis RoPE
        let qn = self.w(&format!("{p}.attention.norm_q.weight"))?;
        let kn = self.w(&format!("{p}.attention.norm_k.weight"))?;
        head_rmsnorm(&mut q, seq, nh, hd, &qn, 1e-5);
        head_rmsnorm(&mut k, seq, nh, hd, &kn, 1e-5);
        rope_interleaved(&mut q, seq, nh, hd, cos, sin, half);
        rope_interleaved(&mut k, seq, nh, hd, cos, sin, half);

        // full (non-causal) attention per head
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; seq * dim];
        for hh in 0..nh {
            for ti in 0..seq {
                let mut scores = vec![0.0f32; seq];
                let mut mx = f32::NEG_INFINITY;
                for tj in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[(ti * nh + hh) * hd + d] * k[(tj * nh + hh) * hd + d];
                    }
                    scores[tj] = dot * scale;
                    if scores[tj] > mx {
                        mx = scores[tj];
                    }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for tj in 0..seq {
                        acc += scores[tj] * v[(tj * nh + hh) * hd + d];
                    }
                    ctx[(ti * nh + hh) * hd + d] = acc / sum;
                }
            }
        }
        self.linear(
            &ctx,
            seq,
            dim,
            &format!("{p}.attention.to_out.0"),
            dim,
            false,
        )
    }

    fn feed_forward(&self, x: &[f32], seq: usize, p: &str) -> Result<Vec<f32>> {
        let dim = self.cfg.dim as usize;
        let wshape = self.st.shape(&format!("{p}.feed_forward.w1.weight"))?;
        let inter = wshape[0];
        let gate = self.linear(x, seq, dim, &format!("{p}.feed_forward.w1"), inter, false)?;
        let up = self.linear(x, seq, dim, &format!("{p}.feed_forward.w3"), inter, false)?;
        let mut h = vec![0.0f32; seq * inter];
        for i in 0..seq * inter {
            h[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        }
        self.linear(&h, seq, inter, &format!("{p}.feed_forward.w2"), dim, false)
    }

    fn final_layer(&self, x: &[f32], seq: usize, temb: &[f32]) -> Result<Vec<f32>> {
        let dim = self.cfg.dim as usize;
        // scale = adaLN(silu(temb)) [dim]
        let mut s = temb.to_vec();
        silu_(&mut s);
        let scale = self.linear(
            &s,
            1,
            TEMB_DIM,
            "all_final_layer.2-1.adaLN_modulation.1",
            dim,
            true,
        )?;
        // layernorm(no affine) then ·(1+scale)
        let mut h = layernorm_no_affine(x, seq, dim, 1e-6);
        mod_scale(&mut h, seq, dim, &scale);
        let out_dim = self.st.shape("all_final_layer.2-1.linear.weight")?[0];
        self.linear(&h, seq, dim, "all_final_layer.2-1.linear", out_dim, true)
    }

    // ---- multi-axis interleaved RoPE cos/sin over [img(ph×pw), cap] ----
    fn rope_unified(&self, ph: usize, pw: usize, cap_len: usize) -> (Vec<f32>, Vec<f32>) {
        let axes: Vec<usize> = self.cfg.axes_dims.iter().map(|&d| d as usize).collect();
        let theta = self.cfg.rope_theta as f64;
        // per-axis freqs (halves)
        let freqs: Vec<Vec<f64>> = axes
            .iter()
            .map(|&d| {
                let half = d / 2;
                (0..half)
                    .map(|i| (-theta.ln() * (i as f64) / (half as f64)).exp())
                    .collect()
            })
            .collect();
        let half: usize = axes.iter().map(|d| d / 2).sum(); // 64

        // positions: img tokens (t=cap_len+1, h, w) then cap tokens (t=1+i, 0, 0)
        let img_len = ph * pw;
        let total = img_len + cap_len;
        let mut cos = vec![0.0f32; total * half];
        let mut sin = vec![0.0f32; total * half];
        let mut emit = |row: usize, pos: [f64; 3]| {
            let mut off = 0usize;
            for axis in 0..3 {
                for (i, &fr) in freqs[axis].iter().enumerate() {
                    let ang = pos[axis] * fr;
                    cos[row * half + off + i] = ang.cos() as f32;
                    sin[row * half + off + i] = ang.sin() as f32;
                }
                off += freqs[axis].len();
            }
        };
        let mut row = 0;
        for i in 0..ph {
            for j in 0..pw {
                emit(row, [(cap_len + 1) as f64, i as f64, j as f64]);
                row += 1;
            }
        }
        for i in 0..cap_len {
            emit(row, [(1 + i) as f64, 0.0, 0.0]);
            row += 1;
        }
        (cos, sin)
    }

    // ---- weight helpers ----
    fn w(&self, name: &str) -> Result<Vec<f32>> {
        self.st.tensor_f32(name)
    }

    /// Linear `y[r,o]=Σ x[r,i]·W[o,i] (+ b[o])`, weight `<p>.weight` [out,in].
    fn linear(
        &self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        p: &str,
        out_dim: usize,
        bias: bool,
    ) -> Result<Vec<f32>> {
        let w = self.w(&format!("{p}.weight"))?;
        let b = if bias {
            Some(self.w(&format!("{p}.bias"))?)
        } else {
            None
        };
        if let Some(g) = &self.gpu {
            return self.linear_gpu(g, x, rows, in_dim, &w, out_dim, b.as_deref());
        }
        let mut y = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            let xr = &x[r * in_dim..(r + 1) * in_dim];
            for o in 0..out_dim {
                let wr = &w[o * in_dim..(o + 1) * in_dim];
                let mut acc = b.as_ref().map_or(0.0, |b| b[o]);
                for i in 0..in_dim {
                    acc += xr[i] * wr[i];
                }
                y[r * out_dim + o] = acc;
            }
        }
        Ok(y)
    }

    /// GPU bf16 matmul: `y[r,o] = Σ_i x[r,i]·W[o,i] (+ b[o])`. `w`/`b` are the
    /// already-loaded f32 weights; `w` is packed to bf16 for the kernel.
    fn linear_gpu(
        &self,
        g: &GpuHandles,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        w: &[f32],
        out_dim: usize,
        bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;
        let dev = &g.ctx.device;
        // pack weight to bf16 (kernel reads array<u32>, 2 bf16/word; pad to even)
        let mut wb: Vec<u8> = Vec::with_capacity(w.len() * 2 + 2);
        for &v in w {
            wb.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }
        if w.len() % 2 == 1 {
            wb.extend_from_slice(&[0u8, 0u8]);
        }
        let w_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dit.w"),
            contents: &wb,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let x_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dit.x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_buf = make_storage_rw(dev, "dit.y", rows * out_dim);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dit.mm"),
        });
        matmul_bf16_batched_chained(
            g.ctx, g.pipes, &mut enc, &w_buf, &x_buf, &y_buf, in_dim, out_dim, rows,
        );
        if let Some(b) = bias {
            let b_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dit.b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            add_bias_batched_chained(g.ctx, g.pipes, &mut enc, &y_buf, &b_buf, out_dim, rows);
        }
        let read = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dit.read"),
            size: (rows * out_dim * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&y_buf, 0, &read, 0, (rows * out_dim * 4) as u64);
        g.ctx.queue.submit(Some(enc.finish()));
        pollster::block_on(read_back_f32(dev, &read))
    }
}

// ---- free ops ----

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

fn layernorm_no_affine(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|v| (*v as f64 - mean) * (*v as f64 - mean))
            .sum::<f64>()
            / dim as f64;
        let inv = (1.0 / (var + eps as f64).sqrt()) as f32;
        for c in 0..dim {
            out[r * dim + c] = ((row[c] as f64 - mean) as f32) * inv;
        }
    }
    out
}

/// `x[t,c] *= (1 + scale[c])` broadcast over tokens.
fn mod_scale(x: &mut [f32], seq: usize, dim: usize, scale: &[f32]) {
    for t in 0..seq {
        for c in 0..dim {
            x[t * dim + c] *= 1.0 + scale[c];
        }
    }
}

/// `out[t,c] += tanh(gate[c]) * branch[t,c]`.
fn gated_add(out: &mut [f32], seq: usize, dim: usize, gate: &[f32], branch: &[f32]) {
    for t in 0..seq {
        for c in 0..dim {
            out[t * dim + c] += gate[c].tanh() * branch[t * dim + c];
        }
    }
}

fn silu_(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// Interleaved RoPE over `[seq, heads, hd]`: even/odd pairs rotated by per-token
/// cos/sin `[seq, half]` (shared across heads).
fn rope_interleaved(
    x: &mut [f32],
    seq: usize,
    heads: usize,
    hd: usize,
    cos: &[f32],
    sin: &[f32],
    half: usize,
) {
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * hd;
            for i in 0..half {
                let c = cos[t * half + i];
                let s = sin[t * half + i];
                let x1 = x[base + 2 * i];
                let x2 = x[base + 2 * i + 1];
                x[base + 2 * i] = x1 * c - x2 * s;
                x[base + 2 * i + 1] = x1 * s + x2 * c;
            }
        }
    }
}

/// Patchify latent `[C,H,W]` → `[pH*pW, C*p*p]`, feature order `(sy,sx,c)`.
fn patchify(latent: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (ph, pw) = (h / p, w / p);
    let mut out = vec![0.0f32; ph * pw * c * p * p];
    let fpp = c * p * p;
    for r in 0..ph {
        for col in 0..pw {
            let tok = r * pw + col;
            for sy in 0..p {
                for sx in 0..p {
                    for ch in 0..c {
                        let f = (sy * p + sx) * c + ch;
                        out[tok * fpp + f] = latent[ch * h * w + (r * p + sy) * w + (col * p + sx)];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`]: `[pH*pW, C*p*p]` → `[C,H,W]`.
fn unpatchify(patches: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (ph, pw) = (h / p, w / p);
    let mut out = vec![0.0f32; c * h * w];
    let fpp = c * p * p;
    for r in 0..ph {
        for col in 0..pw {
            let tok = r * pw + col;
            for sy in 0..p {
                for sx in 0..p {
                    for ch in 0..c {
                        let f = (sy * p + sx) * c + ch;
                        out[ch * h * w + (r * p + sy) * w + (col * p + sx)] =
                            patches[tok * fpp + f];
                    }
                }
            }
        }
    }
    out
}
