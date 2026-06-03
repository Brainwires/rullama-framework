//! GPU (WGSL) StyleTTS2 hifigan decoder + generator — the dominant synth cost on GPU.
//!
//! Buffer-chained like Kokoro's `gpu_fast.rs` (activations stay in GPU buffers across a
//! stage, one submit, readback only at the end). Reuses the same chained dispatchers
//! (conv1d/conv_transpose1d with bias+groups+dilation, adain, snake, residual, scale) and
//! the same AdainResBlk1d / AdaINResBlock1 composition — only the Generator wiring is the
//! hifigan variant (4 upsamples, per-stage Snake on the trunk, single-channel HnNSF source,
//! conv_post + tanh; no iSTFT). The acoustic graph (text_encoder/bert/predictor) stays on
//! CPU (small, T≈36) exactly like Kokoro keeps ALBERT/BiLSTM on CPU.
#![allow(dead_code)]

use std::collections::HashMap;

use super::decoder::source_signal;
use crate::backend::dispatch::{
    adain_chained, add_bias_batched_chained, avg_pool2d_half_chained, conv1d_chained, conv2d_chf_chained,
    conv_transpose1d_chained, gelu_exact_chained, layernorm_affine_chained, leaky_relu_chained,
    make_dummy_storage, make_storage_rw, matmul_f16_batched_tiled_chained, nearest_upsample2x_chained,
    read_back_f32, residual_add_chained, scale_chained, snake_chained, vision_attention_chained,
    write_storage_f16, write_storage_f32,
};
use crate::backend::{Pipelines, WgpuCtx};
use crate::reference::kokoro::ops::{leaky_relu as leaky_cpu, linear};

const RSQRT2: f32 = 0.707_106_77;
const STYLE_DIM: usize = 128;

/// Persistent GPU weight cache (name → uploaded f32 buffer).
pub type GpuWeightCache = HashMap<String, wgpu::Buffer>;

pub struct StyleTtsGpu<'a> {
    w: &'a HashMap<String, Vec<f32>>,
    ctx: &'a WgpuCtx,
    p: &'a Pipelines,
    wc: &'a mut GpuWeightCache,
    dummy: wgpu::Buffer,
    scratch: Vec<wgpu::Buffer>,
}

impl<'a> StyleTtsGpu<'a> {
    pub fn new(w: &'a HashMap<String, Vec<f32>>, ctx: &'a WgpuCtx, p: &'a Pipelines, wc: &'a mut GpuWeightCache) -> Self {
        let dummy = make_dummy_storage(&ctx.device, "dummy");
        Self { w, ctx, p, wc, dummy, scratch: Vec::new() }
    }

    fn t(&self, n: &str) -> &[f32] {
        self.w.get(n).unwrap_or_else(|| panic!("missing gpu weight: {n}"))
    }

    /// Debug: readback a buffer + report NaN count / range (env ST2DBG gates the call site).
    async fn dbg(&self, label: &str, buf: &wgpu::Buffer, n: usize) {
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("dbg"), size: (n * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut e = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dbg") });
        e.copy_buffer_to_buffer(buf, 0, &read, 0, (n * 4) as u64);
        self.ctx.queue.submit(Some(e.finish()));
        let v = read_back_f32(&self.ctx.device, &read).await.expect("dbg");
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let (mn, mx) = v.iter().filter(|x| x.is_finite()).fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        eprintln!("[ST2DBG] {label}: n={n} nan={nan} inf={inf} min={mn:.3} max={mx:.3}");
    }

    /// Cached weight buffer (uploaded once).
    fn wt(&mut self, name: &str) -> wgpu::Buffer {
        if let Some(b) = self.wc.get(name) {
            return b.clone();
        }
        let buf = write_storage_f32(&self.ctx.device, &self.ctx.queue, name, self.w.get(name).unwrap_or_else(|| panic!("missing gpu weight: {name}")));
        self.wc.insert(name.to_string(), buf.clone());
        buf
    }

    fn up(&mut self, x: &[f32]) -> wgpu::Buffer {
        let b = make_storage_rw(&self.ctx.device, "up", x.len());
        self.ctx.queue.write_buffer(&b, 0, bytemuck::cast_slice(x));
        self.scratch.push(b.clone());
        b
    }

    fn alloc(&mut self, n: usize) -> wgpu::Buffer {
        let b = make_storage_rw(&self.ctx.device, "scratch", n);
        self.scratch.push(b.clone());
        b
    }

    /// gamma/beta = chunk(fc(style)) for an AdaIN, uploaded.
    fn adain_gb(&mut self, fc_prefix: &str, c: usize, style: &[f32]) -> (wgpu::Buffer, wgpu::Buffer) {
        let fw = self.t(&format!("{fc_prefix}.fc.weight")).to_vec();
        let fb = self.t(&format!("{fc_prefix}.fc.bias")).to_vec();
        let gb = linear(style, 1, STYLE_DIM, &fw, Some(&fb), 2 * c);
        let (g, b) = gb.split_at(c);
        (self.up(g), self.up(b))
    }

    /// AdainResBlk1d (LeakyReLU 0.2), buffer-chained. `upsample` doubles T via the depthwise pool.
    fn adain_resblk1d(&mut self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, dim_in: usize, t: usize, dim_out: usize, upsample: bool, prefix: &str, style: &[f32]) -> (wgpu::Buffer, usize) {
        let (g1, b1) = self.adain_gb(&format!("{prefix}.norm1"), dim_in, style);
        let (g2, b2) = self.adain_gb(&format!("{prefix}.norm2"), dim_out, style);
        let h1 = self.alloc(dim_in * t);
        adain_chained(self.ctx, self.p, enc, x, &g1, &b1, &h1, dim_in, t, 1e-5);
        leaky_relu_chained(self.ctx, self.p, enc, &h1, dim_in * t, 0.2);
        let (h1, t_pool) = if upsample {
            let pw = self.wt(&format!("{prefix}.pool.weight"));
            let pb = self.wt(&format!("{prefix}.pool.bias"));
            let tp = (t - 1) * 2 + (3 - 1) + 1 + 1 - 2; // depthwise convT k3 s2 p1 opad1 → 2t
            let out = self.alloc(dim_in * tp);
            conv_transpose1d_chained(self.ctx, self.p, enc, &h1, &pw, Some(&pb), &self.dummy, &out, dim_in, t, dim_in, 3, 2, 1, 1, dim_in);
            (out, tp)
        } else {
            (h1, t)
        };
        let c1w = self.wt(&format!("{prefix}.conv1.weight"));
        let c1b = self.wt(&format!("{prefix}.conv1.bias"));
        let cv1 = self.alloc(dim_out * t_pool);
        conv1d_chained(self.ctx, self.p, enc, &h1, &c1w, Some(&c1b), &self.dummy, &cv1, dim_in, t_pool, dim_out, 3, 1, 1, 1, 1);
        let h3 = self.alloc(dim_out * t_pool);
        adain_chained(self.ctx, self.p, enc, &cv1, &g2, &b2, &h3, dim_out, t_pool, 1e-5);
        leaky_relu_chained(self.ctx, self.p, enc, &h3, dim_out * t_pool, 0.2);
        let residual = self.alloc(dim_out * t_pool);
        let c2w = self.wt(&format!("{prefix}.conv2.weight"));
        let c2b = self.wt(&format!("{prefix}.conv2.bias"));
        conv1d_chained(self.ctx, self.p, enc, &h3, &c2w, Some(&c2b), &self.dummy, &residual, dim_out, t_pool, dim_out, 3, 1, 1, 1, 1);
        // shortcut
        let sc = if upsample {
            let su = self.alloc(dim_in * t_pool);
            nearest_upsample2x_chained(self.ctx, self.p, enc, x, &su, dim_in, t);
            su
        } else {
            x.clone()
        };
        let sc = if dim_in != dim_out {
            let cw = self.wt(&format!("{prefix}.conv1x1.weight"));
            let out = self.alloc(dim_out * t_pool);
            conv1d_chained(self.ctx, self.p, enc, &sc, &cw, None, &self.dummy, &out, dim_in, t_pool, dim_out, 1, 1, 0, 1, 1);
            out
        } else {
            sc
        };
        residual_add_chained(self.ctx, self.p, enc, &residual, &sc, dim_out * t_pool);
        scale_chained(self.ctx, self.p, enc, &residual, dim_out * t_pool, RSQRT2);
        (residual, t_pool)
    }

    /// AdaINResBlock1 (Snake, 3 dilated conv pairs), buffer-chained. Same length.
    fn adain_resblock1(&mut self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, c: usize, t: usize, k: usize, dil: [usize; 3], prefix: &str, style: &[f32]) -> wgpu::Buffer {
        let xacc = self.alloc(c * t);
        enc.copy_buffer_to_buffer(x, 0, &xacc, 0, (c * t * 4) as u64);
        for j in 0..3 {
            let (g1, b1) = self.adain_gb(&format!("{prefix}.adain1.{j}"), c, style);
            let (g2, b2) = self.adain_gb(&format!("{prefix}.adain2.{j}"), c, style);
            let a1 = self.wt(&format!("{prefix}.alpha1.{j}"));
            let a2 = self.wt(&format!("{prefix}.alpha2.{j}"));
            let c1w = self.wt(&format!("{prefix}.convs1.{j}.weight"));
            let c1b = self.wt(&format!("{prefix}.convs1.{j}.bias"));
            let c2w = self.wt(&format!("{prefix}.convs2.{j}.weight"));
            let c2b = self.wt(&format!("{prefix}.convs2.{j}.bias"));
            let h1 = self.alloc(c * t);
            adain_chained(self.ctx, self.p, enc, &xacc, &g1, &b1, &h1, c, t, 1e-5);
            let h2 = self.alloc(c * t);
            snake_chained(self.ctx, self.p, enc, &h1, &a1, &h2, c, t);
            let h3 = self.alloc(c * t);
            conv1d_chained(self.ctx, self.p, enc, &h2, &c1w, Some(&c1b), &self.dummy, &h3, c, t, c, k, 1, (k * dil[j] - dil[j]) / 2, dil[j], 1);
            let h4 = self.alloc(c * t);
            adain_chained(self.ctx, self.p, enc, &h3, &g2, &b2, &h4, c, t, 1e-5);
            let h5 = self.alloc(c * t);
            snake_chained(self.ctx, self.p, enc, &h4, &a2, &h5, c, t);
            let rb = self.alloc(c * t);
            conv1d_chained(self.ctx, self.p, enc, &h5, &c2w, Some(&c2b), &self.dummy, &rb, c, t, c, k, 1, (k - 1) / 2, 1, 1);
            residual_add_chained(self.ctx, self.p, enc, &xacc, &rb, c * t);
        }
        xacc
    }

    fn concat(&mut self, enc: &mut wgpu::CommandEncoder, parts: &[(&wgpu::Buffer, usize)], t: usize) -> wgpu::Buffer {
        let ctot: usize = parts.iter().map(|(_, c)| *c).sum();
        let out = self.alloc(ctot * t);
        let mut base = 0;
        for (b, c) in parts {
            enc.copy_buffer_to_buffer(b, 0, &out, (base * t * 4) as u64, (c * t * 4) as u64);
            base += c;
        }
        out
    }

    fn enc(&self) -> wgpu::CommandEncoder {
        self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st2.gpu") })
    }
    fn submit(&self, e: wgpu::CommandEncoder) {
        self.ctx.queue.submit(Some(e.finish()));
    }

    /// hifigan Generator on GPU. `x` buffer [512, xt], `har` buffer [1, har_len]. Returns the
    /// pre-tanh waveform buffer + length. **One submit per upsample stage** (project rule —
    /// keeps each command buffer small so large sequences don't trip a GPU timeout).
    async fn generator(&mut self, x: wgpu::Buffer, xt: usize, har: &wgpu::Buffer, har_len: usize, style: &[f32]) -> (wgpu::Buffer, usize) {
        const RATES: [usize; 4] = [10, 5, 3, 2];
        const KERNELS: [usize; 4] = [20, 10, 6, 4];
        const RK: [usize; 3] = [3, 7, 11];
        let rdil = [[1usize, 3, 5]; 3];
        let dbg = std::env::var("ST2DBG").is_ok();
        if dbg {
            self.dbg("gen.har", har, har_len).await;
        }
        let mut cur = x;
        let (mut cin, mut tcur) = (512usize, xt);
        for i in 0..4 {
            let mut enc = self.enc();
            let a = self.wt(&format!("generator.alphas.{i}"));
            let sn = self.alloc(cin * tcur);
            snake_chained(self.ctx, self.p, &mut enc, &cur, &a, &sn, cin, tcur);
            let cout = cin / 2;
            let ncw = self.wt(&format!("generator.noise_convs.{i}.weight"));
            let ncb = self.wt(&format!("generator.noise_convs.{i}.bias"));
            let (xsrc, nres_k, ts) = if i + 1 < 4 {
                let sf: usize = RATES[i + 1..].iter().product();
                let ts = (har_len + 2 * ((sf + 1) / 2) - sf * 2) / sf + 1;
                let o = self.alloc(cout * ts);
                conv1d_chained(self.ctx, self.p, &mut enc, har, &ncw, Some(&ncb), &self.dummy, &o, 1, har_len, cout, sf * 2, sf, (sf + 1) / 2, 1, 1);
                (o, 7usize, ts)
            } else {
                let o = self.alloc(cout * har_len);
                conv1d_chained(self.ctx, self.p, &mut enc, har, &ncw, Some(&ncb), &self.dummy, &o, 1, har_len, cout, 1, 1, 0, 1, 1);
                (o, 11usize, har_len)
            };
            let xsrc = self.adain_resblock1(&mut enc, &xsrc, cout, ts, nres_k, [1, 3, 5], &format!("generator.noise_res.{i}"), style);
            let uw = self.wt(&format!("generator.ups.{i}.weight"));
            let ub = self.wt(&format!("generator.ups.{i}.bias"));
            let u = RATES[i];
            let kk = KERNELS[i];
            let pad = u / 2 + u % 2;
            let tup = (tcur - 1) * u + (kk - 1) + (u % 2) + 1 - 2 * pad;
            let up = self.alloc(cout * tup);
            conv_transpose1d_chained(self.ctx, self.p, &mut enc, &sn, &uw, Some(&ub), &self.dummy, &up, cin, tcur, cout, kk, u, pad, u % 2, 1);
            debug_assert_eq!(tup, ts, "stage {i}: up {tup} != src {ts}");
            residual_add_chained(self.ctx, self.p, &mut enc, &up, &xsrc, cout * tup);
            let acc = self.alloc(cout * tup);
            for (j, (&rk, rd)) in RK.iter().zip(rdil.iter()).enumerate() {
                let rb = self.adain_resblock1(&mut enc, &up, cout, tup, rk, [rd[0], rd[1], rd[2]], &format!("generator.resblocks.{}", i * 3 + j), style);
                residual_add_chained(self.ctx, self.p, &mut enc, &acc, &rb, cout * tup);
            }
            scale_chained(self.ctx, self.p, &mut enc, &acc, cout * tup, 1.0 / 3.0);
            self.submit(enc);
            cur = acc;
            cin = cout;
            tcur = tup;
            if dbg {
                self.dbg(&format!("gen.stage{i}"), &cur, cin * tcur).await;
            }
        }
        let mut enc = self.enc();
        let a = self.wt("generator.alphas.4");
        let sn = self.alloc(cin * tcur);
        snake_chained(self.ctx, self.p, &mut enc, &cur, &a, &sn, cin, tcur);
        let cpw = self.wt("generator.conv_post.weight");
        let cpb = self.wt("generator.conv_post.bias");
        let post = self.alloc(tcur);
        conv1d_chained(self.ctx, self.p, &mut enc, &sn, &cpw, Some(&cpb), &self.dummy, &post, cin, tcur, 1, 7, 1, 3, 1, 1);
        self.submit(enc);
        (post, tcur)
    }

    async fn read(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let rd = self.ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("rd"), size: (n * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut e = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rd") });
        e.copy_buffer_to_buffer(buf, 0, &rd, 0, (n * 4) as u64);
        self.ctx.queue.submit(Some(e.finish()));
        read_back_f32(&self.ctx.device, &rd).await.expect("readback")
    }

    /// GPU StyleEncoder: reference mel `[n_mels, t]` → 128-d style vector. `prefix` =
    /// "acoustic" or "prosodic". The conv-heavy stack runs on GPU; the tiny adaptive-pool +
    /// Linear tail (512→128) runs on CPU after a single readback.
    async fn style_encoder(&mut self, mel_buf: &wgpu::Buffer, n_mels: usize, t: usize, prefix: &str) -> Vec<f32> {
        const BLK: [(usize, usize); 4] = [(64, 128), (128, 256), (256, 512), (512, 512)];
        let pn = |s: &str| format!("{prefix}.{s}");
        let mut enc = self.enc();
        // conv0: 1 → 64, k3 p1
        let c0w = self.wt(&pn("conv0.weight"));
        let c0b = self.wt(&pn("conv0.bias"));
        let mut x = self.alloc(64 * n_mels * t);
        conv2d_chf_chained(self.ctx, self.p, &mut enc, &c0w, mel_buf, Some(&c0b), &self.dummy, &x, 1, n_mels, t, 64, n_mels, t, 3, 3, 1, 1, 1, 1, 1);
        let (mut h, mut w) = (n_mels, t);
        for (i, &(din, dout)) in BLK.iter().enumerate() {
            let (h2, w2) = (h / 2, (w + 1) / 2);
            // shortcut: optional 1×1 conv (when channels change) then avg-pool
            let sc = if din != dout {
                let cw = self.wt(&pn(&format!("blk{i}.sc.weight")));
                let o = self.alloc(dout * h * w);
                conv2d_chf_chained(self.ctx, self.p, &mut enc, &cw, &x, None, &self.dummy, &o, din, h, w, dout, h, w, 1, 1, 1, 1, 0, 0, 1);
                o
            } else {
                x.clone()
            };
            let sc_pool = self.alloc(dout * h2 * w2);
            avg_pool2d_half_chained(self.ctx, self.p, &mut enc, &sc, &sc_pool, dout, h, w, h2, w2);
            // residual: leaky → conv1(k3p1) → strided depthwise down → leaky → conv2(k3p1)
            let r = self.alloc(din * h * w);
            enc.copy_buffer_to_buffer(&x, 0, &r, 0, (din * h * w * 4) as u64);
            leaky_relu_chained(self.ctx, self.p, &mut enc, &r, din * h * w, 0.2);
            let (c1w, c1b) = (self.wt(&pn(&format!("blk{i}.conv1.weight"))), self.wt(&pn(&format!("blk{i}.conv1.bias"))));
            let c1 = self.alloc(din * h * w);
            conv2d_chf_chained(self.ctx, self.p, &mut enc, &c1w, &r, Some(&c1b), &self.dummy, &c1, din, h, w, din, h, w, 3, 3, 1, 1, 1, 1, 1);
            let (dw, db) = (self.wt(&pn(&format!("blk{i}.down.weight"))), self.wt(&pn(&format!("blk{i}.down.bias"))));
            let dn = self.alloc(din * h2 * w2);
            conv2d_chf_chained(self.ctx, self.p, &mut enc, &dw, &c1, Some(&db), &self.dummy, &dn, din, h, w, din, h2, w2, 3, 3, 2, 2, 1, 1, din);
            leaky_relu_chained(self.ctx, self.p, &mut enc, &dn, din * h2 * w2, 0.2);
            let (c2w, c2b) = (self.wt(&pn(&format!("blk{i}.conv2.weight"))), self.wt(&pn(&format!("blk{i}.conv2.bias"))));
            let c2 = self.alloc(dout * h2 * w2);
            conv2d_chf_chained(self.ctx, self.p, &mut enc, &c2w, &dn, Some(&c2b), &self.dummy, &c2, din, h2, w2, dout, h2, w2, 3, 3, 1, 1, 1, 1, 1);
            residual_add_chained(self.ctx, self.p, &mut enc, &c2, &sc_pool, dout * h2 * w2);
            scale_chained(self.ctx, self.p, &mut enc, &c2, dout * h2 * w2, RSQRT2);
            x = c2;
            h = h2;
            w = w2;
        }
        // leaky → conv_out (512→512, k5, no pad) → [512, h-4, w-4]
        leaky_relu_chained(self.ctx, self.p, &mut enc, &x, 512 * h * w, 0.2);
        let (oh, ow) = (h - 4, w - 4);
        let (cow, cob) = (self.wt(&pn("conv_out.weight")), self.wt(&pn("conv_out.bias")));
        let co = self.alloc(512 * oh * ow);
        conv2d_chf_chained(self.ctx, self.p, &mut enc, &cow, &x, Some(&cob), &self.dummy, &co, 512, h, w, 512, oh, ow, 5, 5, 1, 1, 0, 0, 1);
        self.submit(enc);
        // adaptive avg pool (mean over oh·ow) + leaky + Linear(512→128) on CPU (tiny)
        let feat = self.read(&co, 512 * oh * ow).await;
        let mut pooled: Vec<f32> = (0..512).map(|c| feat[c * oh * ow..(c + 1) * oh * ow].iter().sum::<f32>() / (oh * ow) as f32).collect();
        leaky_cpu(&mut pooled, 0.2);
        linear(&pooled, 1, 512, self.t(&pn("linear.weight")), Some(self.t(&pn("linear.bias"))), 128)
    }

    /// Reference mel `[n_mels, t]` → 256-d voice vector (acoustic ‖ prosodic) on the GPU.
    pub async fn encode(&mut self, mel: &[f32], n_mels: usize, t: usize) -> Vec<f32> {
        let mel_buf = self.up(mel);
        let a = self.style_encoder(&mel_buf, n_mels, t, "acoustic").await;
        let pr = self.style_encoder(&mel_buf, n_mels, t, "prosodic").await;
        a.into_iter().chain(pr).collect()
    }

    // ---------------------------------------------------------------------------------------
    // Style-diffusion denoiser on GPU. Mirrors reference/styletts2/diffusion.rs (the validated
    // CPU oracle) but runs the per-eval StyleTransformer1d on the GPU (f16-weight matmuls +
    // layernorm-affine AdaLN + flash attention + exact GELU). The ADPM2 sampler's scalar
    // arithmetic stays on CPU; only net() — the cost — is offloaded. f16 weights are safe here:
    // s_pred is 70% damped by the reference blend before the (exact) decoder.
    // ---------------------------------------------------------------------------------------

    /// f16 weight buffer for a named tensor (cached under "f16:<name>").
    fn wt16(&mut self, name: &str) -> wgpu::Buffer {
        let key = format!("f16:{name}");
        if let Some(b) = self.wc.get(&key) {
            return b.clone();
        }
        let buf = write_storage_f16(&self.ctx.device, &self.ctx.queue, name, self.w.get(name).unwrap_or_else(|| panic!("missing diff weight {name}")));
        self.wc.insert(key, buf.clone());
        buf
    }

    /// f16 weight buffer from an explicit f32 slice (for the to_kv k/v split), cached under `key`.
    fn wt16_slice(&mut self, key: &str, data: &[f32]) -> wgpu::Buffer {
        if let Some(b) = self.wc.get(key) {
            return b.clone();
        }
        let buf = write_storage_f16(&self.ctx.device, &self.ctx.queue, key, data);
        self.wc.insert(key.to_string(), buf.clone());
        buf
    }

    /// y[rows,nout] = x[rows,kin] @ w[nout,kin]ᵀ (+bias), f16 weights.
    fn glin(&mut self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, rows: usize, kin: usize, nout: usize, w: &wgpu::Buffer, bias: Option<&wgpu::Buffer>) -> wgpu::Buffer {
        let y = self.alloc(rows * nout);
        matmul_f16_batched_tiled_chained(self.ctx, self.p, enc, w, x, &y, kin, nout, rows);
        if let Some(b) = bias {
            add_bias_batched_chained(self.ctx, self.p, enc, &y, b, nout, rows);
        }
        y
    }

    /// Style-diffusion sample → s_pred[256] on GPU. `emb` = PLBERT bert_dur `[l,768]` (CPU).
    /// `noise_init`/`noises` are the replayed RNG draws (deterministic given them).
    #[allow(clippy::too_many_arguments)]
    pub async fn diffusion_sample(&mut self, emb: &[f32], l: usize, ref_s: &[f32], noise_init: &[f32], noises: &[Vec<f32>], sigma_data: f32, sigma_min: f32, sigma_max: f32, rho: f32, steps: usize) -> Vec<f32> {
        // Karras schedule
        let inv = 1.0 / rho;
        let (a, b) = (sigma_max.powf(inv), sigma_min.powf(inv));
        let mut sig: Vec<f32> = (0..steps).map(|i| (a + (i as f32 / (steps - 1) as f32) * (b - a)).powf(rho)).collect();
        sig.push(0.0);
        // ADPM2 (scalar math on CPU; the net eval on GPU)
        let mut x: Vec<f32> = noise_init.iter().map(|v| sig[0] * v).collect();
        for i in 0..steps - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            let sigma_up = (sn * sn * (s * s - sn * sn) / (s * s)).sqrt();
            let sigma_down = (sn * sn - sigma_up * sigma_up).sqrt();
            let sigma_mid = (s + sigma_down) * 0.5;
            let dn = self.diff_denoise(&x, s, sigma_data, emb, l, ref_s).await;
            let d: Vec<f32> = (0..256).map(|k| (x[k] - dn[k]) / s).collect();
            let x_mid: Vec<f32> = (0..256).map(|k| x[k] + d[k] * (sigma_mid - s)).collect();
            let dn_mid = self.diff_denoise(&x_mid, sigma_mid, sigma_data, emb, l, ref_s).await;
            let d_mid: Vec<f32> = (0..256).map(|k| (x_mid[k] - dn_mid[k]) / sigma_mid).collect();
            let nz = &noises[i];
            for k in 0..256 {
                x[k] = x[k] + d_mid[k] * (sigma_down - s) + nz[k] * sigma_up;
            }
        }
        x
    }

    /// KDiffusion denoise_fn: c_skip·x + c_out·net(c_in·x, c_noise).
    async fn diff_denoise(&mut self, x: &[f32], sigma: f32, sd: f32, emb: &[f32], l: usize, ref_s: &[f32]) -> Vec<f32> {
        let c_skip = sd * sd / (sigma * sigma + sd * sd);
        let c_out = sigma * sd / (sd * sd + sigma * sigma).sqrt();
        let c_in = 1.0 / (sigma * sigma + sd * sd).sqrt();
        let c_noise = sigma.ln() * 0.25;
        let xin: Vec<f32> = x.iter().map(|v| c_in * v).collect();
        let pred = self.diff_net(&xin, c_noise, emb, l, ref_s).await;
        (0..256).map(|k| c_skip * x[k] + c_out * pred[k]).collect()
    }

    /// Test hook: one isolated GPU denoiser eval (parity vs the CPU oracle's `net_eval`).
    pub async fn diff_net_eval(&mut self, x: &[f32], time: f32, emb: &[f32], l: usize, ref_s: &[f32]) -> Vec<f32> {
        self.diff_net(x, time, emb, l, ref_s).await
    }

    /// One denoiser network eval on GPU. (x[256], time, emb[l,768], ref_s[256]) → [256].
    async fn diff_net(&mut self, x: &[f32], time: f32, emb: &[f32], l: usize, ref_s: &[f32]) -> Vec<f32> {
        const F: usize = 1024;
        const MID: usize = 512;
        // mapping (CPU, tiny) → replicate to [l,1024] → upload
        let mapping = self.diff_mapping(time, ref_s);
        let mut mrep = vec![0f32; l * F];
        for t in 0..l {
            mrep[t * F..(t + 1) * F].copy_from_slice(&mapping);
        }
        let map_buf = self.up(&mrep);
        // h[l,1024] = [ x(256, broadcast) ‖ emb[t](768) ]  (CPU build → upload)
        let mut h = vec![0f32; l * F];
        for t in 0..l {
            h[t * F..t * F + 256].copy_from_slice(&x[..256]);
            h[t * F + 256..t * F + F].copy_from_slice(&emb[t * 768..t * 768 + 768]);
        }
        let hb = self.up(&h);
        let mut enc = self.enc();
        for bi in 0..3 {
            let pfx = format!("diffusion.blocks.{bi}");
            residual_add_chained(self.ctx, self.p, &mut enc, &hb, &map_buf, l * F); // x += mapping
            // AdaLN (norm, norm_context) — affine (1+γ_fc), β_fc from ref_s (constant across rows)
            let (gn, bn) = self.diff_adaln_affine(&format!("{pfx}.attention.norm"), ref_s);
            let (gc, bc) = self.diff_adaln_affine(&format!("{pfx}.attention.norm_context"), ref_s);
            let xn = self.alloc(l * F);
            layernorm_affine_chained(self.ctx, self.p, &mut enc, &hb, Some(&gn), Some(&bn), &self.dummy, &xn, l, F, 1e-5);
            let cn = self.alloc(l * F);
            layernorm_affine_chained(self.ctx, self.p, &mut enc, &hb, Some(&gc), Some(&bc), &self.dummy, &cn, l, F, 1e-5);
            // q = to_q(xn); k,v = split(to_kv)(cn)
            let qw = self.wt16(&format!("{pfx}.attention.to_q.weight"));
            let kvw = self.t(&format!("{pfx}.attention.to_kv.weight")).to_vec(); // [1024,1024]
            let kw = self.wt16_slice(&format!("f16:{pfx}.to_kv.k"), &kvw[..MID * F]);
            let vw = self.wt16_slice(&format!("f16:{pfx}.to_kv.v"), &kvw[MID * F..]);
            let q = self.glin(&mut enc, &xn, l, F, MID, &qw, None);
            let k = self.glin(&mut enc, &cn, l, F, MID, &kw, None);
            let v = self.glin(&mut enc, &cn, l, F, MID, &vw, None);
            // matmul output [l, heads*hd] is already patch-major (PHD) — the layout the flash
            // kernel reads directly (q[(patch*heads+head)*hd+d]); output is PHD too. No transpose.
            let o = self.alloc(MID * l);
            vision_attention_chained(self.ctx, self.p, &mut enc, &q, &k, &v, &o, 64, 8, l);
            let ow = self.wt16(&format!("{pfx}.attention.attention.to_out.weight"));
            let ob = self.wt(&format!("{pfx}.attention.attention.to_out.bias"));
            let attn = self.glin(&mut enc, &o, l, MID, F, &ow, Some(&ob));
            residual_add_chained(self.ctx, self.p, &mut enc, &hb, &attn, l * F); // x += attn
            // FFN: Lin(1024→2048) gelu Lin(2048→1024)
            let f0w = self.wt16(&format!("{pfx}.feed_forward.0.weight"));
            let f0b = self.wt(&format!("{pfx}.feed_forward.0.bias"));
            let ff = self.glin(&mut enc, &hb, l, F, 2 * F, &f0w, Some(&f0b));
            gelu_exact_chained(self.ctx, self.p, &mut enc, &ff, l * 2 * F);
            let f2w = self.wt16(&format!("{pfx}.feed_forward.2.weight"));
            let f2b = self.wt(&format!("{pfx}.feed_forward.2.bias"));
            let ff2 = self.glin(&mut enc, &ff, l, 2 * F, F, &f2w, Some(&f2b));
            residual_add_chained(self.ctx, self.p, &mut enc, &hb, &ff2, l * F); // x += ffn
        }
        self.submit(enc);
        // mean-pool over l + Conv1x1(1024→256) on CPU (tiny)
        let hf = self.read(&hb, l * F).await;
        let mut pooled = vec![0f32; F];
        for t in 0..l {
            for c in 0..F {
                pooled[c] += hf[t * F + c];
            }
        }
        for v in pooled.iter_mut() {
            *v /= l as f32;
        }
        linear(&pooled, 1, F, self.t("diffusion.to_out.1.weight"), Some(self.t("diffusion.to_out.1.bias")), 256)
    }

    /// AdaLayerNorm affine: returns uploaded ((1+γ_fc), β_fc) ∈ ℝ¹⁰²⁴, γ/β = fc(ref_s).
    fn diff_adaln_affine(&mut self, fc_prefix: &str, ref_s: &[f32]) -> (wgpu::Buffer, wgpu::Buffer) {
        let fw = self.t(&format!("{fc_prefix}.fc.weight")).to_vec();
        let fb = self.t(&format!("{fc_prefix}.fc.bias")).to_vec();
        let gb = linear(ref_s, 1, 256, &fw, Some(&fb), 2048);
        let g1: Vec<f32> = gb[..1024].iter().map(|v| 1.0 + v).collect();
        let beta = gb[1024..].to_vec();
        (self.up(&g1), self.up(&beta))
    }

    /// Denoiser time/feature mapping (CPU, tiny): to_mapping(GELU(Lin(time_pos))+GELU(Lin(ref_s))).
    fn diff_mapping(&self, time: f32, ref_s: &[f32]) -> Vec<f32> {
        let gelu = |v: &mut [f32]| {
            for x in v.iter_mut() {
                let z = *x / std::f32::consts::SQRT_2;
                let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
                let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_) * t + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t) * (-z * z).exp();
                *x *= 0.5 * (1.0 + if z >= 0.0 { y } else { -y });
            }
        };
        let mut tpos = vec![0f32; 257];
        tpos[0] = time;
        let tw = self.t("diffusion.to_time.0.0.weights");
        for j in 0..128 {
            let f = time * tw[j] * 2.0 * std::f32::consts::PI;
            tpos[1 + j] = f.sin();
            tpos[1 + 128 + j] = f.cos();
        }
        let mut t_emb = linear(&tpos, 1, 257, self.t("diffusion.to_time.0.1.weight"), Some(self.t("diffusion.to_time.0.1.bias")), 1024);
        gelu(&mut t_emb);
        let mut f_emb = linear(ref_s, 1, 256, self.t("diffusion.to_features.0.weight"), Some(self.t("diffusion.to_features.0.bias")), 1024);
        gelu(&mut f_emb);
        let mut m: Vec<f32> = (0..1024).map(|k| t_emb[k] + f_emb[k]).collect();
        m = linear(&m, 1, 1024, self.t("diffusion.to_mapping.0.weight"), Some(self.t("diffusion.to_mapping.0.bias")), 1024);
        gelu(&mut m);
        m = linear(&m, 1, 1024, self.t("diffusion.to_mapping.2.weight"), Some(self.t("diffusion.to_mapping.2.bias")), 1024);
        gelu(&mut m);
        m
    }

    /// Full hifigan decoder on GPU: `asr [512,f]`, `f0`/`n [2f]` (CPU), `style [128]` → 24 kHz waveform.
    pub async fn decode(&mut self, asr: &[f32], f: usize, f0: &[f32], n: &[f32], style: &[f32]) -> Vec<f32> {
        let dbg = std::env::var("ST2DBG").is_ok();
        let asr_buf = self.up(asr);
        let f0_buf = self.up(f0);
        let n_buf = self.up(n);
        // ---- decoder cat-stack (one submit; small tensors, t ≤ 2f) ----
        let mut enc = self.enc();
        let f0w = self.wt("F0_conv.weight");
        let f0b = self.wt("F0_conv.bias");
        let f0d = self.alloc(f);
        conv1d_chained(self.ctx, self.p, &mut enc, &f0_buf, &f0w, Some(&f0b), &self.dummy, &f0d, 1, 2 * f, 1, 3, 2, 1, 1, 1);
        let nw = self.wt("N_conv.weight");
        let nb = self.wt("N_conv.bias");
        let nd = self.alloc(f);
        conv1d_chained(self.ctx, self.p, &mut enc, &n_buf, &nw, Some(&nb), &self.dummy, &nd, 1, 2 * f, 1, 3, 2, 1, 1, 1);
        let cat0 = self.concat(&mut enc, &[(&asr_buf, 512), (&f0d, 1), (&nd, 1)], f);
        let (mut x, mut tcur) = self.adain_resblk1d(&mut enc, &cat0, 514, f, 1024, false, "encode", style);
        let arw = self.wt("asr_res.0.weight");
        let arb = self.wt("asr_res.0.bias");
        let asr_res = self.alloc(64 * f);
        conv1d_chained(self.ctx, self.p, &mut enc, &asr_buf, &arw, Some(&arb), &self.dummy, &asr_res, 512, f, 64, 1, 1, 0, 1, 1);
        // x is 1024 channels before every decode block (encode → 1024; blocks 0-2 stay
        // 1024; block 3 outputs 512 but its *input* is still 1024).
        for i in 0..4 {
            let xin = self.concat(&mut enc, &[(&x, 1024), (&asr_res, 64), (&f0d, 1), (&nd, 1)], tcur);
            let (nx, nt) = self.adain_resblk1d(&mut enc, &xin, 1090, tcur, if i < 3 { 1024 } else { 512 }, i == 3, &format!("decode.{i}"), style);
            x = nx;
            tcur = nt;
        }
        self.submit(enc);
        if dbg {
            self.dbg("decode_x", &x, 512 * tcur).await;
        }
        // ---- har source (CPU) → generator (one submit per upsample stage) ----
        let lw = self.t("generator.m_source.l_linear.weight").to_vec();
        let lb = self.t("generator.m_source.l_linear.bias")[0];
        let har = source_signal(f0, 300, 9, &lw, lb);
        let har_buf = self.up(&har);
        let (post, tpost) = self.generator(x, tcur, &har_buf, har.len(), style).await;
        if dbg {
            self.dbg("post", &post, tpost).await;
        }
        // ---- readback + tanh on CPU ----
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("rd"), size: (tpost * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut e2 = self.enc();
        e2.copy_buffer_to_buffer(&post, 0, &read, 0, (tpost * 4) as u64);
        self.submit(e2);
        let raw = read_back_f32(&self.ctx.device, &read).await.expect("readback");
        raw.iter().map(|v| v.tanh()).collect()
    }
}
