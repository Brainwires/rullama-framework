//! Buffer-chained, weight-cached GPU forward — the fast path. Unlike the slice
//! helpers in `gpu.rs` (which upload + dispatch + READ BACK per kernel, ~100 GPU
//! stalls/synth), this keeps activations in GPU buffers across a stage, submits once
//! per contiguous GPU run, and reads back only at the CPU-glue points (BiLSTM, source).
//! Weights are dequantized + uploaded once and cached (big win for repeat synths).
//!
//! Validated against the slice path (`synthesize_gpu`) — see `kokoro_oracle`.
#![allow(dead_code)]

use std::collections::HashMap;

use super::KokoroModel;
use super::ops::linear;
use crate::backend::dispatch::{
    adain_chained, conv_transpose1d_chained, conv1d_chained, istft_chained, leaky_relu_chained,
    make_dummy_storage, make_storage_rw, nearest_upsample2x_chained, read_back_f32,
    residual_add_chained, scale_chained, snake_chained, spec_phase_chained, write_storage_f32,
};
use crate::backend::{Pipelines, WgpuCtx};

const RSQRT2: f32 = 0.707_106_77;

/// Persistent GPU weight cache (name → uploaded buffer), reused across synths.
pub type WeightCache = HashMap<String, wgpu::Buffer>;

/// GPU TTS forward context: borrowed persistent weight cache + per-stage scratch.
pub struct GpuTts<'a> {
    m: &'a KokoroModel,
    ctx: &'a WgpuCtx,
    p: &'a Pipelines,
    wc: &'a mut WeightCache,
    dummy: wgpu::Buffer,
    scratch: Vec<wgpu::Buffer>,
}

impl<'a> GpuTts<'a> {
    pub fn new(
        m: &'a KokoroModel,
        ctx: &'a WgpuCtx,
        p: &'a Pipelines,
        wc: &'a mut WeightCache,
    ) -> Self {
        let dummy = make_dummy_storage(&ctx.device, "dummy");
        Self {
            m,
            ctx,
            p,
            wc,
            dummy,
            scratch: Vec::new(),
        }
    }

    /// Cached weight buffer (dequant + upload once). Returns a cheap clone handle.
    fn w(&mut self, name: &str) -> wgpu::Buffer {
        if let Some(b) = self.wc.get(name) {
            return b.clone();
        }
        let buf = write_storage_f32(&self.ctx.device, &self.ctx.queue, name, &self.m.t(name));
        self.wc.insert(name.to_string(), buf.clone());
        buf
    }

    /// Upload a CPU slice to a fresh read-write buffer (kept alive in scratch).
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
    fn adain_gb(
        &mut self,
        fc_prefix: &str,
        c: usize,
        style: &[f32],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let sd = self.m.cfg.style_dim;
        let fw = self.m.t(&format!("{fc_prefix}.fc.weight"));
        let fb = self.m.t(&format!("{fc_prefix}.fc.bias"));
        let gb = linear(style, 1, sd, &fw, Some(&fb), 2 * c);
        let (g, b) = gb.split_at(c);
        (self.up(g), self.up(b))
    }

    /// AdainResBlk1d (LeakyReLU), buffer-chained. `x` is a buffer; returns (out, tout).
    #[allow(clippy::too_many_arguments)]
    fn adain_resblk1d(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        dim_in: usize,
        t: usize,
        dim_out: usize,
        upsample: bool,
        prefix: &str,
        style: &[f32],
    ) -> (wgpu::Buffer, usize) {
        let learned_sc = dim_in != dim_out;
        let t_out = if upsample { 2 * t } else { t };
        let (g1, b1) = self.adain_gb(&format!("{prefix}.norm1"), dim_in, style);
        let (g2, b2) = self.adain_gb(&format!("{prefix}.norm2"), dim_out, style);

        let h1 = self.alloc(dim_in * t);
        adain_chained(self.ctx, self.p, enc, x, &g1, &b1, &h1, dim_in, t, 1e-5);
        leaky_relu_chained(self.ctx, self.p, enc, &h1, dim_in * t, 0.2);
        let (pool, t_pool) = if upsample {
            let pw = self.w(&format!("{prefix}.pool.weight"));
            let pb = self.w(&format!("{prefix}.pool.bias"));
            let out = self.alloc(dim_in * t_out);
            conv_transpose1d_chained(
                self.ctx,
                self.p,
                enc,
                &h1,
                &pw,
                Some(&pb),
                &self.dummy,
                &out,
                dim_in,
                t,
                dim_in,
                3,
                2,
                1,
                1,
                dim_in,
            );
            (out, t_out)
        } else {
            (h1, t)
        };
        let c1w = self.w(&format!("{prefix}.conv1.weight"));
        let c1b = self.w(&format!("{prefix}.conv1.bias"));
        let cv1 = self.alloc(dim_out * t_pool);
        conv1d_chained(
            self.ctx,
            self.p,
            enc,
            &pool,
            &c1w,
            Some(&c1b),
            &self.dummy,
            &cv1,
            dim_in,
            t_pool,
            dim_out,
            3,
            1,
            1,
            1,
            1,
        );
        let h3 = self.alloc(dim_out * t_pool);
        adain_chained(
            self.ctx, self.p, enc, &cv1, &g2, &b2, &h3, dim_out, t_pool, 1e-5,
        );
        leaky_relu_chained(self.ctx, self.p, enc, &h3, dim_out * t_pool, 0.2);
        let c2w = self.w(&format!("{prefix}.conv2.weight"));
        let c2b = self.w(&format!("{prefix}.conv2.bias"));
        let residual = self.alloc(dim_out * t_pool);
        conv1d_chained(
            self.ctx,
            self.p,
            enc,
            &h3,
            &c2w,
            Some(&c2b),
            &self.dummy,
            &residual,
            dim_out,
            t_pool,
            dim_out,
            3,
            1,
            1,
            1,
            1,
        );

        let short_in = if upsample {
            let su = self.alloc(dim_in * t_out);
            nearest_upsample2x_chained(self.ctx, self.p, enc, x, &su, dim_in, t);
            su
        } else {
            x.clone()
        };
        let sc = if learned_sc {
            let cw = self.w(&format!("{prefix}.conv1x1.weight"));
            let out = self.alloc(dim_out * t_pool);
            conv1d_chained(
                self.ctx,
                self.p,
                enc,
                &short_in,
                &cw,
                None,
                &self.dummy,
                &out,
                dim_in,
                t_pool,
                dim_out,
                1,
                1,
                0,
                1,
                1,
            );
            out
        } else {
            short_in
        };
        residual_add_chained(self.ctx, self.p, enc, &residual, &sc, dim_out * t_pool);
        scale_chained(self.ctx, self.p, enc, &residual, dim_out * t_pool, RSQRT2);
        (residual, t_pool)
    }

    /// AdaINResBlock1 (Snake, 3 dilated conv pairs), buffer-chained. Same length.
    #[allow(clippy::too_many_arguments)]
    fn adain_resblock1(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        c: usize,
        t: usize,
        k: usize,
        dil: [usize; 3],
        prefix: &str,
        style: &[f32],
    ) -> wgpu::Buffer {
        let xacc = self.alloc(c * t);
        enc.copy_buffer_to_buffer(x, 0, &xacc, 0, (c * t * 4) as u64);
        for j in 0..3 {
            let (g1, b1) = self.adain_gb(&format!("{prefix}.adain1.{j}"), c, style);
            let (g2, b2) = self.adain_gb(&format!("{prefix}.adain2.{j}"), c, style);
            let a1 = self.w(&format!("{prefix}.alpha1.{j}"));
            let a2 = self.w(&format!("{prefix}.alpha2.{j}"));
            let c1w = self.w(&format!("{prefix}.convs1.{j}.weight"));
            let c1b = self.w(&format!("{prefix}.convs1.{j}.bias"));
            let c2w = self.w(&format!("{prefix}.convs2.{j}.weight"));
            let c2b = self.w(&format!("{prefix}.convs2.{j}.bias"));
            let (h1, h2, h3, h4, h5, rb) = (
                self.alloc(c * t),
                self.alloc(c * t),
                self.alloc(c * t),
                self.alloc(c * t),
                self.alloc(c * t),
                self.alloc(c * t),
            );
            let pad1 = (k * dil[j] - dil[j]) / 2;
            adain_chained(self.ctx, self.p, enc, &xacc, &g1, &b1, &h1, c, t, 1e-5);
            snake_chained(self.ctx, self.p, enc, &h1, &a1, &h2, c, t);
            conv1d_chained(
                self.ctx,
                self.p,
                enc,
                &h2,
                &c1w,
                Some(&c1b),
                &self.dummy,
                &h3,
                c,
                t,
                c,
                k,
                1,
                pad1,
                dil[j],
                1,
            );
            adain_chained(self.ctx, self.p, enc, &h3, &g2, &b2, &h4, c, t, 1e-5);
            snake_chained(self.ctx, self.p, enc, &h4, &a2, &h5, c, t);
            conv1d_chained(
                self.ctx,
                self.p,
                enc,
                &h5,
                &c2w,
                Some(&c2b),
                &self.dummy,
                &rb,
                c,
                t,
                c,
                k,
                1,
                (k - 1) / 2,
                1,
                1,
            );
            residual_add_chained(self.ctx, self.p, enc, &xacc, &rb, c * t);
        }
        xacc
    }

    /// Channel-major concat of buffers (each [Ci, t]) into one [sum Ci, t] via copies.
    fn concat(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        parts: &[(&wgpu::Buffer, usize)],
        t: usize,
    ) -> wgpu::Buffer {
        let ctot: usize = parts.iter().map(|(_, c)| *c).sum();
        let out = self.alloc(ctot * t);
        let mut base = 0usize;
        for (b, c) in parts {
            enc.copy_buffer_to_buffer(b, 0, &out, (base * t * 4) as u64, (c * t * 4) as u64);
            base += c;
        }
        out
    }

    /// GPU generator (buffer-chained). `x` buffer [512, xt_len], `har` buffer [22, har_len].
    /// Returns the waveform buffer + length. One encoder; caller submits + reads back.
    fn generator(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: wgpu::Buffer,
        xt_len: usize,
        har: &wgpu::Buffer,
        har_len: usize,
        style: &[f32],
    ) -> (wgpu::Buffer, usize) {
        let rates = self.m.cfg.upsample_rates.clone();
        let rkernels = self.m.cfg.resblock_kernel_sizes.clone();
        let rdil = self.m.cfg.resblock_dilation_sizes.clone();
        let nfft = self.m.cfg.gen_istft_n_fft;
        let nbins = nfft / 2 + 1;
        let mut cur = x;
        let mut cin = self.m.cfg.upsample_initial_channel;
        let mut tcur = xt_len;

        for i in 0..rates.len() {
            leaky_relu_chained(self.ctx, self.p, enc, &cur, cin * tcur, 0.1);
            let cout = cin / 2;
            let ncw = self.w(&format!("k.decoder.generator.noise_convs.{i}.weight"));
            let ncb = self.w(&format!("k.decoder.generator.noise_convs.{i}.bias"));
            let (xsrc, nres_k, ts) = if i + 1 < rates.len() {
                let sf: usize = rates[i + 1..].iter().product();
                let ts = (har_len + 2 * ((sf + 1) / 2) - sf * 2) / sf + 1;
                let o = self.alloc(cout * ts);
                conv1d_chained(
                    self.ctx,
                    self.p,
                    enc,
                    har,
                    &ncw,
                    Some(&ncb),
                    &self.dummy,
                    &o,
                    nfft + 2,
                    har_len,
                    cout,
                    sf * 2,
                    sf,
                    (sf + 1) / 2,
                    1,
                    1,
                );
                (o, 7usize, ts)
            } else {
                let o = self.alloc(cout * har_len);
                conv1d_chained(
                    self.ctx,
                    self.p,
                    enc,
                    har,
                    &ncw,
                    Some(&ncb),
                    &self.dummy,
                    &o,
                    nfft + 2,
                    har_len,
                    cout,
                    1,
                    1,
                    0,
                    1,
                    1,
                );
                (o, 11usize, har_len)
            };
            let xsrc = self.adain_resblock1(
                enc,
                &xsrc,
                cout,
                ts,
                nres_k,
                [1, 3, 5],
                &format!("k.decoder.generator.noise_res.{i}"),
                style,
            );

            let uw = self.w(&format!("k.decoder.generator.ups.{i}.weight"));
            let ub = self.w(&format!("k.decoder.generator.ups.{i}.bias"));
            let kk = self.m.cfg.upsample_kernel_sizes[i];
            let tup0 = (tcur - 1) * rates[i] + (kk - 1) + 1 - 2 * ((kk - rates[i]) / 2);
            let up0 = self.alloc(cout * tup0);
            conv_transpose1d_chained(
                self.ctx,
                self.p,
                enc,
                &cur,
                &uw,
                Some(&ub),
                &self.dummy,
                &up0,
                cin,
                tcur,
                cout,
                kk,
                rates[i],
                (kk - rates[i]) / 2,
                0,
                1,
            );
            // reflection pad (1,0) on the last stage, via per-channel copies
            let (up, tup) = if i == rates.len() - 1 {
                let padded = self.alloc(cout * (tup0 + 1));
                for ch in 0..cout {
                    let src = (ch * tup0) as u64 * 4;
                    let dst = (ch * (tup0 + 1)) as u64 * 4;
                    enc.copy_buffer_to_buffer(&up0, src + 4, &padded, dst, 4); // out[c,0]=in[c,1]
                    enc.copy_buffer_to_buffer(&up0, src, &padded, dst + 4, (tup0 * 4) as u64); // out[c,1..]=in[c,..]
                }
                (padded, tup0 + 1)
            } else {
                (up0, tup0)
            };
            residual_add_chained(self.ctx, self.p, enc, &up, &xsrc, cout * tup);

            // 3 resblocks summed, then /num_kernels
            let acc = self.alloc(cout * tup);
            // acc starts zero (fresh); add each resblock
            for (j, (&rk, rd)) in rkernels.iter().zip(rdil.iter()).enumerate() {
                let rb = self.adain_resblock1(
                    enc,
                    &up,
                    cout,
                    tup,
                    rk,
                    [rd[0], rd[1], rd[2]],
                    &format!("k.decoder.generator.resblocks.{}", i * rkernels.len() + j),
                    style,
                );
                residual_add_chained(self.ctx, self.p, enc, &acc, &rb, cout * tup);
            }
            scale_chained(
                self.ctx,
                self.p,
                enc,
                &acc,
                cout * tup,
                1.0 / rkernels.len() as f32,
            );
            cur = acc;
            cin = cout;
            tcur = tup;
        }

        leaky_relu_chained(self.ctx, self.p, enc, &cur, cin * tcur, 0.01);
        let cpw = self.w("k.decoder.generator.conv_post.weight");
        let cpb = self.w("k.decoder.generator.conv_post.bias");
        let post = self.alloc((nfft + 2) * tcur);
        conv1d_chained(
            self.ctx,
            self.p,
            enc,
            &cur,
            &cpw,
            Some(&cpb),
            &self.dummy,
            &post,
            cin,
            tcur,
            nfft + 2,
            7,
            1,
            3,
            1,
            1,
        );
        let tpost = tcur;
        let spec = self.alloc(nbins * tpost);
        let phase = self.alloc(nbins * tpost);
        spec_phase_chained(self.ctx, self.p, enc, &post, &spec, &phase, nbins, tpost);
        let hop = self.m.cfg.gen_istft_hop;
        let out_len = (tpost - 1) * hop + nfft - 2 * (nfft / 2);
        let audio = self.alloc(out_len);
        istft_chained(
            self.ctx, self.p, enc, &spec, &phase, &audio, nbins, tpost, nfft, hop,
        );
        (audio, out_len)
    }

    async fn read(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rd"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rd") });
        enc.copy_buffer_to_buffer(buf, 0, &read, 0, (n * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));
        read_back_f32(&self.ctx.device, &read)
            .await
            .expect("readback")
    }
}

impl KokoroModel {
    /// Fast hybrid GPU forward (buffer-chained, weight-cached). Same result as
    /// `synthesize_gpu` (corr ~1.0) with ~3 readbacks instead of ~100.
    pub async fn synthesize_gpu_fast(
        &self,
        ctx: &WgpuCtx,
        p: &Pipelines,
        wc: &mut WeightCache,
        ids: &[i64],
        ref_s: &[f32],
    ) -> Vec<f32> {
        let mut g = GpuTts::new(self, ctx, p, wc);
        let sd = self.cfg.style_dim;
        let (timbre, prosodic) = (&ref_s[..sd], &ref_s[sd..2 * sd]);
        let cat = self.cfg.hidden_dim + sd;
        let c = self.cfg.hidden_dim;

        // CPU prologue: ALBERT + DurationEncoder + duration (BiLSTM stays CPU)
        let bert = self.bert(ids);
        let be = self.bert_encoder(&bert, ids.len());
        let d = self.duration_encode(&be, ids.len(), prosodic);
        let (_logits, dur) = self.predict_duration(&d, ids.len());
        let (en, f) = self.expand_by_dur_cm(&d, ids.len(), cat, &dur);

        // f0_n: shared BiLSTM (CPU) → adain stacks (GPU) → readback F0/N
        let mut x_rm = vec![0.0f32; f * cat];
        for ff in 0..f {
            for cc in 0..cat {
                x_rm[ff * cat + cc] = en[cc * f + ff];
            }
        }
        let sw = self.load_bilstm("k.predictor.shared");
        let xs = self.run_bilstm(&sw, &x_rm, f, cat, c / 2);
        let mut x_cm = vec![0.0f32; c * f];
        for ff in 0..f {
            for cc in 0..c {
                x_cm[cc * f + ff] = xs[ff * c + cc];
            }
        }
        let half = c / 2;
        let run_branch = |g: &mut GpuTts, which: &str| -> (wgpu::Buffer, usize) {
            let xb = g.up(&x_cm);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("f0n") });
            let (h, t1) = g.adain_resblk1d(
                &mut enc,
                &xb,
                c,
                f,
                c,
                false,
                &format!("k.predictor.{which}.0"),
                prosodic,
            );
            let (h, t2) = g.adain_resblk1d(
                &mut enc,
                &h,
                c,
                t1,
                half,
                true,
                &format!("k.predictor.{which}.1"),
                prosodic,
            );
            let (h, t3) = g.adain_resblk1d(
                &mut enc,
                &h,
                half,
                t2,
                half,
                false,
                &format!("k.predictor.{which}.2"),
                prosodic,
            );
            let pw = g.w(&format!("k.predictor.{which}_proj.weight"));
            let pb = g.w(&format!("k.predictor.{which}_proj.bias"));
            let out = g.alloc(t3);
            conv1d_chained(
                ctx,
                p,
                &mut enc,
                &h,
                &pw,
                Some(&pb),
                &g.dummy,
                &out,
                half,
                t3,
                1,
                1,
                1,
                0,
                1,
                1,
            );
            ctx.queue.submit(Some(enc.finish()));
            (out, t3)
        };
        let (f0b, fl) = run_branch(&mut g, "F0");
        let f0 = g.read(&f0b, fl).await;
        let (nb, nl) = run_branch(&mut g, "N");
        let n = g.read(&nb, nl).await;
        g.scratch.clear();

        // text_encoder: conv stack (GPU) + BiLSTM (CPU)
        let t_en = self.text_encoder_gpu(ctx, p, ids).await;

        // source (CPU) → har buffer
        let (har, frames) = self.generator_source(&f0);

        // decoder + generator: one GPU chain, single readback
        // asr = expand t_en by dur (CPU), F0/N convs + cat + encode + decode + generator (GPU)
        let mut t_en_rm = vec![0.0f32; ids.len() * c];
        for ch in 0..c {
            for ti in 0..ids.len() {
                t_en_rm[ti * c + ch] = t_en[ch * ids.len() + ti];
            }
        }
        let (asr, ff) = self.expand_by_dur_cm(&t_en_rm, ids.len(), c, &dur);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dec+gen"),
            });
        let asr_b = g.up(&asr);
        let f0_b = g.up(&f0);
        let n_b = g.up(&n);
        let f0w = g.w("k.decoder.F0_conv.weight");
        let f0cb = g.w("k.decoder.F0_conv.bias");
        let nw = g.w("k.decoder.N_conv.weight");
        let ncb = g.w("k.decoder.N_conv.bias");
        let fl2 = f0.len();
        let f0d_t = (fl2 + 2 - 2 - 1) / 2 + 1;
        let f0d = g.alloc(f0d_t);
        conv1d_chained(
            ctx,
            p,
            &mut enc,
            &f0_b,
            &f0w,
            Some(&f0cb),
            &g.dummy,
            &f0d,
            1,
            fl2,
            1,
            3,
            2,
            1,
            1,
            1,
        );
        let nd = g.alloc(f0d_t);
        conv1d_chained(
            ctx,
            p,
            &mut enc,
            &n_b,
            &nw,
            Some(&ncb),
            &g.dummy,
            &nd,
            1,
            fl2,
            1,
            3,
            2,
            1,
            1,
            1,
        );
        let cat0 = g.concat(&mut enc, &[(&asr_b, c), (&f0d, 1), (&nd, 1)], ff);
        let (dec_encode, _) = g.adain_resblk1d(
            &mut enc,
            &cat0,
            c + 2,
            ff,
            1024,
            false,
            "k.decoder.encode",
            timbre,
        );
        let arw = g.w("k.decoder.asr_res.0.weight");
        let arb = g.w("k.decoder.asr_res.0.bias");
        let asr_res = g.alloc(64 * ff);
        conv1d_chained(
            ctx,
            p,
            &mut enc,
            &asr_b,
            &arw,
            Some(&arb),
            &g.dummy,
            &asr_res,
            c,
            ff,
            64,
            1,
            1,
            0,
            1,
            1,
        );
        let mut x = dec_encode;
        let mut tcur = ff;
        let mut xc = 1024usize;
        for i in 0..4 {
            let dim_in = xc + 64 + 2;
            let xin = g.concat(
                &mut enc,
                &[(&x, xc), (&asr_res, 64), (&f0d, 1), (&nd, 1)],
                tcur,
            );
            let upsample = i == 3;
            let dim_out = if i < 3 { 1024 } else { 512 };
            let (nx, nt) = g.adain_resblk1d(
                &mut enc,
                &xin,
                dim_in,
                tcur,
                dim_out,
                upsample,
                &format!("k.decoder.decode.{i}"),
                timbre,
            );
            x = nx;
            tcur = nt;
            xc = dim_out;
        }
        let har_b = g.up(&har);
        let (audio_b, alen) = g.generator(&mut enc, x, tcur, &har_b, frames, timbre);
        ctx.queue.submit(Some(enc.finish()));
        let audio = g.read(&audio_b, alen).await;
        g.scratch.clear();
        audio
    }
}
