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

impl KokoroModel {
    /// Slice-in/Vec-out GPU conv1d (upload → dispatch → readback). For stage glue.
    #[allow(clippy::too_many_arguments)]
    pub async fn conv1d_gpu(
        &self, ctx: &WgpuCtx, p: &Pipelines, x: &[f32], cin: usize, t: usize,
        w: &[f32], b: Option<&[f32]>, cout: usize, k: usize, stride: usize, pad: usize, dil: usize, groups: usize,
    ) -> Vec<f32> {
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "d");
        let tout = (t + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        let xb = write_storage_f32(device, queue, "x", x);
        let wb = write_storage_f32(device, queue, "w", w);
        let bb = b.map(|bb| write_storage_f32(device, queue, "b", bb));
        let out = make_storage_rw(device, "o", cout * tout);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("c1d") });
        conv1d_chained(ctx, p, &mut enc, &xb, &wb, bb.as_ref(), &dummy, &out, cin, t, cout, k, stride, pad, dil, groups);
        let staging = read_staging(device, cout * tout);
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, (cout * tout * 4) as u64);
        queue.submit(Some(enc.finish()));
        read_back_f32(device, &staging).await.expect("conv1d_gpu rb")
    }

    /// One F0/N branch on GPU: 3× AdainResBlk1d (with a 2× upsample) + 1×1 proj.
    async fn adain_stack_proj_gpu(&self, ctx: &WgpuCtx, p: &Pipelines, which: &str, x_cm: &[f32], f: usize, style: &[f32]) -> Vec<f32> {
        let hid = self.cfg.hidden_dim;
        let half = hid / 2;
        let h = self.adain_resblk1d_gpu(ctx, p, &format!("k.predictor.{which}.0"), x_cm, hid, f, hid, false, style).await;
        let h = self.adain_resblk1d_gpu(ctx, p, &format!("k.predictor.{which}.1"), &h, hid, f, half, true, style).await;
        let h = self.adain_resblk1d_gpu(ctx, p, &format!("k.predictor.{which}.2"), &h, half, 2 * f, half, false, style).await;
        let pw = self.t(&format!("k.predictor.{which}_proj.weight"));
        let pb = self.t(&format!("k.predictor.{which}_proj.bias"));
        self.conv1d_gpu(ctx, p, &h, half, 2 * f, &pw, Some(&pb), 1, 1, 1, 0, 1, 1).await
    }

    /// GPU ProsodyPredictor.F0Ntrain: shared BiLSTM (CPU) + F0/N adain stacks (GPU).
    /// `en [640, F]` channel-major; returns (F0, N) each `[2F]`.
    pub async fn f0_n_gpu(&self, ctx: &WgpuCtx, p: &Pipelines, en: &[f32], f: usize, style: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let cat = self.cfg.hidden_dim + self.cfg.style_dim;
        let hid = self.cfg.hidden_dim;
        let half = hid / 2;
        // shared BiLSTM (CPU): en^T [F,640] -> [F,512] -> channel-major [512,F]
        let mut x_rm = vec![0.0f32; f * cat];
        for ff in 0..f {
            for c in 0..cat {
                x_rm[ff * cat + c] = en[c * f + ff];
            }
        }
        let sw = self.load_bilstm("k.predictor.shared");
        let xs = self.run_bilstm(&sw, &x_rm, f, cat, half);
        let mut x_cm = vec![0.0f32; hid * f];
        for ff in 0..f {
            for c in 0..hid {
                x_cm[c * f + ff] = xs[ff * hid + c];
            }
        }
        let f0 = self.adain_stack_proj_gpu(ctx, p, "F0", &x_cm, f, style).await;
        let n = self.adain_stack_proj_gpu(ctx, p, "N", &x_cm, f, style).await;
        (f0, n)
    }
}

/// Channel-major concat along the channel axis. For `[Ci, T]` row-major buffers this
/// is literal append (each is Ci rows of T), so no GPU kernel is needed.
fn concat_cm(parts: &[&[f32]]) -> Vec<f32> {
    parts.iter().flat_map(|s| s.iter().copied()).collect()
}

impl KokoroModel {
    /// GPU Decoder front (istftnet.Decoder up to the generator). Returns
    /// (`dec_encode [1024, F]`, `x_after_decode [512, 2F]`). `s` = timbre half.
    pub async fn decoder_features_gpu(
        &self, ctx: &WgpuCtx, p: &Pipelines, t_en: &[f32], f0_curve: &[f32], n_curve: &[f32], dur: &[usize], style: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let c = self.cfg.hidden_dim; // 512
        let t = dur.len();
        // asr = expand t_en [512,T] by dur → [512, F] (transpose to row-major, expand)
        let mut t_en_rm = vec![0.0f32; t * c];
        for ch in 0..c {
            for ti in 0..t {
                t_en_rm[ti * c + ch] = t_en[ch * t + ti];
            }
        }
        let (asr, f) = self.expand_by_dur_cm(&t_en_rm, t, c, dur);

        // F0/N stride-2 downsample convs: [2F] → [F]
        let f0d = self
            .conv1d_gpu(ctx, p, f0_curve, 1, f0_curve.len(), &self.t("k.decoder.F0_conv.weight"), Some(&self.t("k.decoder.F0_conv.bias")), 1, 3, 2, 1, 1, 1)
            .await;
        let nd = self
            .conv1d_gpu(ctx, p, n_curve, 1, n_curve.len(), &self.t("k.decoder.N_conv.weight"), Some(&self.t("k.decoder.N_conv.bias")), 1, 3, 2, 1, 1, 1)
            .await;

        // encode: AdainResBlk1d(cat([asr,F0,N]) = 514 → 1024)
        let cat0 = concat_cm(&[&asr, &f0d, &nd]);
        let dec_encode = self.adain_resblk1d_gpu(ctx, p, "k.decoder.encode", &cat0, c + 2, f, 1024, false, style).await;

        // asr_res = Conv1d(512→64, k1)
        let asr_res = self
            .conv1d_gpu(ctx, p, &asr, c, f, &self.t("k.decoder.asr_res.0.weight"), Some(&self.t("k.decoder.asr_res.0.bias")), 64, 1, 1, 0, 1, 1)
            .await;

        // decode stack: cat([x, asr_res, F0, N]) before each block; last upsamples ×2
        let mut x = dec_encode.clone();
        let mut tcur = f;
        for i in 0..4 {
            let xin = concat_cm(&[&x, &asr_res, &f0d, &nd]);
            let dim_in = x.len() / tcur + 64 + 2; // 1090
            let upsample = i == 3;
            let dim_out = if i < 3 { 1024 } else { 512 };
            x = self.adain_resblk1d_gpu(ctx, p, &format!("k.decoder.decode.{i}"), &xin, dim_in, tcur, dim_out, upsample, style).await;
            if upsample {
                tcur *= 2;
            }
        }
        (dec_encode, x)
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
