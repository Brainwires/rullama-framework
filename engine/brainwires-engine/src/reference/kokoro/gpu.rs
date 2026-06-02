//! Native GPU forward stages built from the Kokoro WGSL kernels, validated
//! stage-by-stage against the same fixtures that proved the CPU oracle. Bring-up
//! toward a full `GpuKokoroForward` (mirroring `multimodal/audio_gpu.rs`).
#![allow(dead_code)]

use super::KokoroModel;
use crate::backend::dispatch::{
    conv1d_chained, layernorm_affine_chained, leaky_relu_chained, make_dummy_storage,
    make_storage_rw, read_back_f32, transpose2d_chained, write_storage_f32,
};
use crate::backend::{Pipelines, WgpuCtx};

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
