//! Native GPU forward stages built from the Kokoro WGSL kernels, validated
//! stage-by-stage against the same fixtures that proved the CPU oracle. Bring-up
//! toward a full `GpuKokoroForward` (mirroring `multimodal/audio_gpu.rs`).
#![allow(dead_code)]

use super::ops::linear;
use super::KokoroModel;
use crate::backend::dispatch::{
    adain_chained, conv1d_chained, conv_transpose1d_chained, layernorm_affine_chained,
    leaky_relu_chained, make_dummy_storage, make_storage_rw, nearest_upsample2x_chained,
    read_back_f32, residual_add_chained, scale_chained, transpose2d_chained, write_storage_f32,
};
use crate::backend::{Pipelines, WgpuCtx};

const RSQRT2: f32 = 0.707_106_77;

impl KokoroModel {
    /// GPU AdainResBlk1d (slice in, Vec out — for stage validation). Mirrors the CPU
    /// `adain_resblk1d`: residual (adain→leaky→pool→conv1→adain→leaky→conv2) + shortcut
    /// (nearest-2x if upsample, conv1x1 if dim_in≠dim_out), then `(res+sc)*rsqrt(2)`.
    /// gamma/beta come from fc(style) computed on the host (InstanceNorm affine absent).
    #[allow(clippy::too_many_arguments)]
    pub async fn adain_resblk1d_gpu(
        &self, ctx: &WgpuCtx, p: &Pipelines, prefix: &str, x: &[f32],
        dim_in: usize, t: usize, dim_out: usize, upsample: bool, style: &[f32],
    ) -> Vec<f32> {
        let sd = self.cfg.style_dim;
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");
        let learned_sc = dim_in != dim_out;
        let t_out = if upsample { 2 * t } else { t };

        // gamma/beta = chunk(fc(style)) for norm1 (dim_in) and norm2 (dim_out)
        let upload_gb = |which: usize, dim: usize| {
            let fw = self.t(&format!("{prefix}.norm{which}.fc.weight"));
            let fb = self.t(&format!("{prefix}.norm{which}.fc.bias"));
            let gb = linear(style, 1, sd, &fw, Some(&fb), 2 * dim);
            let (g, b) = gb.split_at(dim);
            (write_storage_f32(device, queue, "g", g), write_storage_f32(device, queue, "b", b))
        };
        let (g1, b1) = upload_gb(1, dim_in);
        let (g2, b2) = upload_gb(2, dim_out);

        let c1w = write_storage_f32(device, queue, "c1w", &self.t(&format!("{prefix}.conv1.weight")));
        let c1b = write_storage_f32(device, queue, "c1b", &self.t(&format!("{prefix}.conv1.bias")));
        let c2w = write_storage_f32(device, queue, "c2w", &self.t(&format!("{prefix}.conv2.weight")));
        let c2b = write_storage_f32(device, queue, "c2b", &self.t(&format!("{prefix}.conv2.bias")));

        let xb = write_storage_f32(device, queue, "x", x);
        let h1 = make_storage_rw(device, "h1", dim_in * t);
        let pool = make_storage_rw(device, "pool", dim_in * t_out);
        let cv1 = make_storage_rw(device, "cv1", dim_out * t_out);
        let h3 = make_storage_rw(device, "h3", dim_out * t_out);
        let residual = make_storage_rw(device, "res", dim_out * t_out);
        let sc_up = make_storage_rw(device, "scup", dim_in * t_out);
        let sc = make_storage_rw(device, "sc", dim_out * t_out);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("arb") });
        // residual: adain1 -> leaky -> pool -> conv1 -> adain2 -> leaky -> conv2
        adain_chained(ctx, p, &mut enc, &xb, &g1, &b1, &h1, dim_in, t, 1e-5);
        leaky_relu_chained(ctx, p, &mut enc, &h1, dim_in * t, 0.2);
        if upsample {
            let pw = write_storage_f32(device, queue, "pw", &self.t(&format!("{prefix}.pool.weight")));
            let pb = write_storage_f32(device, queue, "pb", &self.t(&format!("{prefix}.pool.bias")));
            conv_transpose1d_chained(ctx, p, &mut enc, &h1, &pw, Some(&pb), &dummy, &pool, dim_in, t, dim_in, 3, 2, 1, 1, dim_in);
        } else {
            enc.copy_buffer_to_buffer(&h1, 0, &pool, 0, (dim_in * t_out * 4) as u64);
        }
        conv1d_chained(ctx, p, &mut enc, &pool, &c1w, Some(&c1b), &dummy, &cv1, dim_in, t_out, dim_out, 3, 1, 1, 1, 1);
        adain_chained(ctx, p, &mut enc, &cv1, &g2, &b2, &h3, dim_out, t_out, 1e-5);
        leaky_relu_chained(ctx, p, &mut enc, &h3, dim_out * t_out, 0.2);
        conv1d_chained(ctx, p, &mut enc, &h3, &c2w, Some(&c2b), &dummy, &residual, dim_out, t_out, dim_out, 3, 1, 1, 1, 1);

        // shortcut: (nearest-2x if upsample) then conv1x1 if learned_sc.
        // For the identity case, feed the buffer straight into residual_add (no copy —
        // write_storage_f32 buffers lack COPY_SRC).
        let short_in = if upsample {
            nearest_upsample2x_chained(ctx, p, &mut enc, &xb, &sc_up, dim_in, t);
            &sc_up
        } else {
            &xb
        };
        let sc_ref: &wgpu::Buffer = if learned_sc {
            let cw = write_storage_f32(device, queue, "1x1", &self.t(&format!("{prefix}.conv1x1.weight")));
            conv1d_chained(ctx, p, &mut enc, short_in, &cw, None, &dummy, &sc, dim_in, t_out, dim_out, 1, 1, 0, 1, 1);
            &sc
        } else {
            short_in
        };

        // out = (residual + sc) * rsqrt(2)
        residual_add_chained(ctx, p, &mut enc, &residual, sc_ref, dim_out * t_out);
        scale_chained(ctx, p, &mut enc, &residual, dim_out * t_out, RSQRT2);

        let staging = read_staging(device, dim_out * t_out);
        enc.copy_buffer_to_buffer(&residual, 0, &staging, 0, (dim_out * t_out * 4) as u64);
        queue.submit(Some(enc.finish()));
        read_back_f32(device, &staging).await.expect("arb readback")
    }
}

fn read_staging(device: &wgpu::Device, n_floats: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kokoro.read"),
        size: (n_floats * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl KokoroModel {
    /// GPU TextEncoder: embedding (CPU gather) → 3×(conv1d → channel-axis LayerNorm
    /// → LeakyReLU(0.2)) on GPU → BiLSTM (CPU). Returns `[hidden, T]` channel-major,
    /// matching `text_encoder()`. Channel-axis LN = transpose → layernorm_affine → transpose.
    pub async fn text_encoder_gpu(&self, ctx: &WgpuCtx, p: &Pipelines, input_ids: &[i64]) -> Vec<f32> {
        let t = input_ids.len();
        let c = self.cfg.hidden_dim;
        let k = self.cfg.text_encoder_kernel_size;
        let pad = (k - 1) / 2;
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");
        let enc_desc = wgpu::CommandEncoderDescriptor { label: Some("te") };

        // embedding → channel-major [C, T]
        let emb = self.t("k.text_encoder.embedding.weight");
        let mut x_cm = vec![0.0f32; c * t];
        for (ti, &id) in input_ids.iter().enumerate() {
            let row = &emb[id as usize * c..(id as usize + 1) * c];
            for ch in 0..c {
                x_cm[ch * t + ti] = row[ch];
            }
        }
        let mut cur = write_storage_f32(device, queue, "te.x", &x_cm);

        for i in 0..self.cfg.n_layer {
            let cw = write_storage_f32(device, queue, "cw", &self.t(&format!("k.text_encoder.cnn.{i}.0.weight")));
            let cb = write_storage_f32(device, queue, "cb", &self.t(&format!("k.text_encoder.cnn.{i}.0.bias")));
            let gamma = write_storage_f32(device, queue, "g", &self.t(&format!("k.text_encoder.cnn.{i}.1.gamma")));
            let beta = write_storage_f32(device, queue, "b", &self.t(&format!("k.text_encoder.cnn.{i}.1.beta")));
            let conv = make_storage_rw(device, "conv", c * t);
            let tr = make_storage_rw(device, "tr", c * t);
            let ln = make_storage_rw(device, "ln", c * t);
            let back = make_storage_rw(device, "back", c * t);

            let mut enc = device.create_command_encoder(&enc_desc);
            conv1d_chained(ctx, p, &mut enc, &cur, &cw, Some(&cb), &dummy, &conv, c, t, c, k, 1, pad, 1, 1);
            transpose2d_chained(ctx, p, &mut enc, &conv, &tr, c, t); // [C,T] -> [T,C]
            layernorm_affine_chained(ctx, p, &mut enc, &tr, Some(&gamma), Some(&beta), &dummy, &ln, t, c, 1e-5);
            transpose2d_chained(ctx, p, &mut enc, &ln, &back, t, c); // [T,C] -> [C,T]
            leaky_relu_chained(ctx, p, &mut enc, &back, c * t, 0.2);
            queue.submit(Some(enc.finish()));
            cur = back;
        }

        // readback the conv-stack output [C, T]
        let staging = read_staging(device, c * t);
        let mut enc = device.create_command_encoder(&enc_desc);
        enc.copy_buffer_to_buffer(&cur, 0, &staging, 0, (c * t * 4) as u64);
        queue.submit(Some(enc.finish()));
        let conv_out = read_back_f32(device, &staging).await.expect("te readback");

        // CPU BiLSTM (short one-shot seq): [C,T] -> row-major [T,C] -> bilstm -> [C,T]
        let mut x_rm = vec![0.0f32; t * c];
        for ch in 0..c {
            for ti in 0..t {
                x_rm[ti * c + ch] = conv_out[ch * t + ti];
            }
        }
        let lw = self.load_bilstm("k.text_encoder.lstm");
        let lstm = self.run_bilstm(&lw, &x_rm, t, c, c / 2); // [T, C]
        let mut out = vec![0.0f32; c * t];
        for ti in 0..t {
            for ch in 0..c {
                out[ch * t + ti] = lstm[ti * c + ch];
            }
        }
        out
    }
}
