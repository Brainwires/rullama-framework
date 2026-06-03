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
    adain_chained, conv1d_chained, conv_transpose1d_chained, leaky_relu_chained, make_dummy_storage,
    make_storage_rw, nearest_upsample2x_chained, read_back_f32, residual_add_chained, scale_chained,
    snake_chained, write_storage_f32,
};
use crate::backend::{Pipelines, WgpuCtx};
use crate::reference::kokoro::ops::linear;

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
