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
    adain_chained, add_bias_batched_chained, avg_pool2d_half_chained, conv_transpose1d_chained,
    conv_transpose1d_f16_chained, conv1d_chained, conv1d_f16_chained, conv2d_chf_chained,
    conv2d_chf_f16_chained, gelu_exact_chained, layernorm_affine_chained, leaky_relu_chained,
    make_dummy_storage, make_storage_rw, matmul_f16_batched_tiled_chained,
    nearest_upsample2x_chained, read_back_f32, residual_add_chained, scale_chained, snake_chained,
    vision_attention_chained, write_storage_f16, write_storage_f16_bits, write_storage_f32,
};
use crate::backend::{Pipelines, WgpuCtx};
use crate::reference::kokoro::ops::{leaky_relu as leaky_cpu, linear};

const RSQRT2: f32 = 0.707_106_77;
const STYLE_DIM: usize = 128;

/// Native dev/bench aid: when `ST2_GPU_THROTTLE_MS` is set, drain the queue and sleep after every
/// stage submit, so the OS reclaims the GPU between stages (the cursor moves, the compositor runs)
/// instead of one long GPU monopoly. This is what keeps a weak integrated GPU from tripping the
/// macOS watchdog during a long synth. Unset (production) → zero overhead. No-op on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn gpu_yield(ctx: &WgpuCtx) {
    if let Some(ms) = std::env::var("ST2_GPU_THROTTLE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}
#[cfg(target_arch = "wasm32")]
fn gpu_yield(_ctx: &WgpuCtx) {}

/// Persistent GPU weight cache (name → uploaded f32 buffer).
pub type GpuWeightCache = HashMap<String, wgpu::Buffer>;

pub struct StyleTtsGpu<'a> {
    w: &'a HashMap<String, Vec<f32>>,
    /// f16 conv weights (raw bits) for the memory-tight variant. Empty for the
    /// f32 variant; when a conv weight is present here, the conv dispatch routes
    /// to the f16 kernel + an f16 GPU buffer instead of f32.
    w16: &'a HashMap<String, Vec<u16>>,
    ctx: &'a WgpuCtx,
    p: &'a Pipelines,
    wc: &'a mut GpuWeightCache,
    dummy: wgpu::Buffer,
    scratch: Vec<wgpu::Buffer>,
}

impl Drop for StyleTtsGpu<'_> {
    /// Each call allocates fresh scratch buffers with new ids, so the shared bind-group cache (and
    /// the GPU descriptor table behind it) grows on *every* synth/encode. Left unbounded this leaks
    /// until a long session — or a tight loop like the fidelity harness — exhausts the GPU and hard-
    /// locks the machine. Evict this call's scratch from the cache and free its GPU memory on drop.
    fn drop(&mut self) {
        let ids: Vec<u64> = self.scratch.iter().map(crate::backend::buf_id).collect();
        self.ctx.bind_cache.invalidate_buffers(&ids);
        for b in &self.scratch {
            b.destroy();
        }
    }
}

impl<'a> StyleTtsGpu<'a> {
    pub fn new(
        w: &'a HashMap<String, Vec<f32>>,
        w16: &'a HashMap<String, Vec<u16>>,
        ctx: &'a WgpuCtx,
        p: &'a Pipelines,
        wc: &'a mut GpuWeightCache,
    ) -> Self {
        let dummy = make_dummy_storage(&ctx.device, "dummy");
        Self {
            w,
            w16,
            ctx,
            p,
            wc,
            dummy,
            scratch: Vec::new(),
        }
    }

    fn t(&self, n: &str) -> &[f32] {
        self.w
            .get(n)
            .unwrap_or_else(|| panic!("missing gpu weight: {n}"))
    }

    /// Debug: readback a buffer + report NaN count / range (env ST2DBG gates the call site).
    async fn dbg(&self, label: &str, buf: &wgpu::Buffer, n: usize) {
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dbg"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dbg") });
        e.copy_buffer_to_buffer(buf, 0, &read, 0, (n * 4) as u64);
        self.ctx.queue.submit(Some(e.finish()));
        let v = read_back_f32(&self.ctx.device, &read).await.expect("dbg");
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let (mn, mx) = v
            .iter()
            .filter(|x| x.is_finite())
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        eprintln!("[ST2DBG] {label}: n={n} nan={nan} inf={inf} min={mn:.3} max={mx:.3}");
    }

    /// Cached f32 weight buffer (uploaded once). Falls back to dequantizing an
    /// f16-resident weight (`w16`) to f32 when one is fetched through the f32
    /// path — so a conv call site not yet routed to the f16 kernel still works
    /// correctly (just without the GPU-side f16 saving). Makes the f16 routing
    /// an incremental, always-correct migration.
    fn wt(&mut self, name: &str) -> wgpu::Buffer {
        if let Some(b) = self.wc.get(name) {
            return b.clone();
        }
        let buf = if let Some(f32data) = self.w.get(name) {
            write_storage_f32(&self.ctx.device, &self.ctx.queue, name, f32data)
        } else if let Some(bits) = self.w16.get(name) {
            let f32data: Vec<f32> = bits
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            write_storage_f32(&self.ctx.device, &self.ctx.queue, name, &f32data)
        } else {
            panic!("missing gpu weight: {name}");
        };
        self.wc.insert(name.to_string(), buf.clone());
        buf
    }

    /// Cached f16 conv-weight buffer (uploaded once) from the raw f16 bits in
    /// `w16`. Keyed `f16c:<name>` to coexist with any f32 entry.
    fn wt16c(&mut self, name: &str) -> wgpu::Buffer {
        let key = format!("f16c:{name}");
        if let Some(b) = self.wc.get(&key) {
            return b.clone();
        }
        let bits = self
            .w16
            .get(name)
            .unwrap_or_else(|| panic!("missing f16 conv weight: {name}"));
        let buf = write_storage_f16_bits(&self.ctx.device, &self.ctx.queue, name, bits);
        self.wc.insert(key, buf.clone());
        buf
    }

    /// conv1d, routed to the f16 kernel + f16 weight buffer when the weight is
    /// f16-resident (present in `w16`); otherwise the f32 path. Identical math.
    #[allow(clippy::too_many_arguments)]
    fn conv1d_w(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        wname: &str,
        bias: Option<&wgpu::Buffer>,
        y: &wgpu::Buffer,
        cin: usize,
        tin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
    ) -> usize {
        if self.w16.contains_key(wname) {
            let wb = self.wt16c(wname);
            conv1d_f16_chained(
                self.ctx,
                self.p,
                enc,
                x,
                &wb,
                bias,
                &self.dummy,
                y,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                dilation,
                groups,
            )
        } else {
            let wb = self.wt(wname);
            conv1d_chained(
                self.ctx,
                self.p,
                enc,
                x,
                &wb,
                bias,
                &self.dummy,
                y,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                dilation,
                groups,
            )
        }
    }

    /// conv_transpose1d with the same f16/f32 routing as [`Self::conv1d_w`].
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d_w(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        wname: &str,
        bias: Option<&wgpu::Buffer>,
        y: &wgpu::Buffer,
        cin: usize,
        tin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        output_padding: usize,
        groups: usize,
    ) -> usize {
        if self.w16.contains_key(wname) {
            let wb = self.wt16c(wname);
            conv_transpose1d_f16_chained(
                self.ctx,
                self.p,
                enc,
                x,
                &wb,
                bias,
                &self.dummy,
                y,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                output_padding,
                groups,
            )
        } else {
            let wb = self.wt(wname);
            conv_transpose1d_chained(
                self.ctx,
                self.p,
                enc,
                x,
                &wb,
                bias,
                &self.dummy,
                y,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                output_padding,
                groups,
            )
        }
    }

    /// conv2d_chf with the same f16/f32 routing as [`Self::conv1d_w`].
    #[allow(clippy::too_many_arguments)]
    fn conv2d_chf_w(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        wname: &str,
        x: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
        y: &wgpu::Buffer,
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        groups: usize,
    ) {
        if self.w16.contains_key(wname) {
            let wb = self.wt16c(wname);
            conv2d_chf_f16_chained(
                self.ctx,
                self.p,
                enc,
                &wb,
                x,
                bias,
                &self.dummy,
                y,
                in_c,
                in_h,
                in_w,
                out_c,
                out_h,
                out_w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                groups,
            );
        } else {
            let wb = self.wt(wname);
            conv2d_chf_chained(
                self.ctx,
                self.p,
                enc,
                &wb,
                x,
                bias,
                &self.dummy,
                y,
                in_c,
                in_h,
                in_w,
                out_c,
                out_h,
                out_w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                groups,
            );
        }
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
    fn adain_gb(
        &mut self,
        fc_prefix: &str,
        c: usize,
        style: &[f32],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let fw = self.t(&format!("{fc_prefix}.fc.weight")).to_vec();
        let fb = self.t(&format!("{fc_prefix}.fc.bias")).to_vec();
        let gb = linear(style, 1, STYLE_DIM, &fw, Some(&fb), 2 * c);
        let (g, b) = gb.split_at(c);
        (self.up(g), self.up(b))
    }

    /// AdainResBlk1d (LeakyReLU 0.2), buffer-chained. `upsample` doubles T via the depthwise pool.
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
        let (g1, b1) = self.adain_gb(&format!("{prefix}.norm1"), dim_in, style);
        let (g2, b2) = self.adain_gb(&format!("{prefix}.norm2"), dim_out, style);
        let h1 = self.alloc(dim_in * t);
        adain_chained(self.ctx, self.p, enc, x, &g1, &b1, &h1, dim_in, t, 1e-5);
        leaky_relu_chained(self.ctx, self.p, enc, &h1, dim_in * t, 0.2);
        let (h1, t_pool) = if upsample {
            let pb = self.wt(&format!("{prefix}.pool.bias"));
            let tp = (t - 1) * 2 + (3 - 1) + 1 + 1 - 2; // depthwise convT k3 s2 p1 opad1 → 2t
            let out = self.alloc(dim_in * tp);
            self.conv_transpose1d_w(
                enc,
                &h1,
                &format!("{prefix}.pool.weight"),
                Some(&pb),
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
            (out, tp)
        } else {
            (h1, t)
        };
        let c1b = self.wt(&format!("{prefix}.conv1.bias"));
        let cv1 = self.alloc(dim_out * t_pool);
        self.conv1d_w(
            enc,
            &h1,
            &format!("{prefix}.conv1.weight"),
            Some(&c1b),
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
        let residual = self.alloc(dim_out * t_pool);
        let c2b = self.wt(&format!("{prefix}.conv2.bias"));
        self.conv1d_w(
            enc,
            &h3,
            &format!("{prefix}.conv2.weight"),
            Some(&c2b),
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
        // shortcut
        let sc = if upsample {
            let su = self.alloc(dim_in * t_pool);
            nearest_upsample2x_chained(self.ctx, self.p, enc, x, &su, dim_in, t);
            su
        } else {
            x.clone()
        };
        let sc = if dim_in != dim_out {
            let out = self.alloc(dim_out * t_pool);
            self.conv1d_w(
                enc,
                &sc,
                &format!("{prefix}.conv1x1.weight"),
                None,
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
            sc
        };
        residual_add_chained(self.ctx, self.p, enc, &residual, &sc, dim_out * t_pool);
        scale_chained(self.ctx, self.p, enc, &residual, dim_out * t_pool, RSQRT2);
        (residual, t_pool)
    }

    /// AdaINResBlock1 (Snake, 3 dilated conv pairs), buffer-chained. Same length.
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
            let a1 = self.wt(&format!("{prefix}.alpha1.{j}"));
            let a2 = self.wt(&format!("{prefix}.alpha2.{j}"));
            let c1b = self.wt(&format!("{prefix}.convs1.{j}.bias"));
            let c2b = self.wt(&format!("{prefix}.convs2.{j}.bias"));
            let h1 = self.alloc(c * t);
            adain_chained(self.ctx, self.p, enc, &xacc, &g1, &b1, &h1, c, t, 1e-5);
            let h2 = self.alloc(c * t);
            snake_chained(self.ctx, self.p, enc, &h1, &a1, &h2, c, t);
            let h3 = self.alloc(c * t);
            self.conv1d_w(
                enc,
                &h2,
                &format!("{prefix}.convs1.{j}.weight"),
                Some(&c1b),
                &h3,
                c,
                t,
                c,
                k,
                1,
                (k * dil[j] - dil[j]) / 2,
                dil[j],
                1,
            );
            let h4 = self.alloc(c * t);
            adain_chained(self.ctx, self.p, enc, &h3, &g2, &b2, &h4, c, t, 1e-5);
            let h5 = self.alloc(c * t);
            snake_chained(self.ctx, self.p, enc, &h4, &a2, &h5, c, t);
            let rb = self.alloc(c * t);
            self.conv1d_w(
                enc,
                &h5,
                &format!("{prefix}.convs2.{j}.weight"),
                Some(&c2b),
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

    fn concat(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        parts: &[(&wgpu::Buffer, usize)],
        t: usize,
    ) -> wgpu::Buffer {
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
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st2.gpu"),
            })
    }
    fn submit(&self, e: wgpu::CommandEncoder) {
        self.ctx.queue.submit(Some(e.finish()));
        gpu_yield(self.ctx);
    }

    /// 0 ms JS event-loop yield (wasm32 only) between heavy GPU bursts. A full StyleTTS2 synth is
    /// one long chain of GPU submits; on iOS Safari that monopolizes the GPU (springboard TDR) and
    /// lets transient buffers pile up un-reclaimed (jetsam). Releasing the event loop for one tick
    /// lets the GPUProcess message pipe drain — completed work finishes and its buffers free before
    /// the next burst. **0 ms specifically**: `setTimeout(>0)` gets the Worker reaped by iOS jetsam
    /// (proven in the training path's `forward_chained::wasm_yield_zero`). No-op on native.
    async fn wasm_yield(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            if let Ok(scope) = js_sys::global().dyn_into::<web_sys::DedicatedWorkerGlobalScope>() {
                let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                    let resolve_fn: js_sys::Function = resolve.into();
                    let _ =
                        scope.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve_fn, 0);
                });
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
            }
        }
    }

    /// hifigan Generator on GPU. `x` buffer [512, xt], `har` buffer [1, har_len]. Returns the
    /// pre-tanh waveform buffer + length. **One submit per upsample stage** (project rule —
    /// keeps each command buffer small so large sequences don't trip a GPU timeout).
    async fn generator(
        &mut self,
        x: wgpu::Buffer,
        xt: usize,
        har: &wgpu::Buffer,
        har_len: usize,
        style: &[f32],
    ) -> Option<(wgpu::Buffer, usize)> {
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
            let ncb = self.wt(&format!("generator.noise_convs.{i}.bias"));
            let ncw_name = format!("generator.noise_convs.{i}.weight");
            let (xsrc, nres_k, ts) = if i + 1 < 4 {
                let sf: usize = RATES[i + 1..].iter().product();
                let ts = (har_len + 2 * sf.div_ceil(2) - sf * 2) / sf + 1;
                let o = self.alloc(cout * ts);
                self.conv1d_w(
                    &mut enc,
                    har,
                    &ncw_name,
                    Some(&ncb),
                    &o,
                    1,
                    har_len,
                    cout,
                    sf * 2,
                    sf,
                    sf.div_ceil(2),
                    1,
                    1,
                );
                (o, 7usize, ts)
            } else {
                let o = self.alloc(cout * har_len);
                self.conv1d_w(
                    &mut enc,
                    har,
                    &ncw_name,
                    Some(&ncb),
                    &o,
                    1,
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
                &mut enc,
                &xsrc,
                cout,
                ts,
                nres_k,
                [1, 3, 5],
                &format!("generator.noise_res.{i}"),
                style,
            );
            let ub = self.wt(&format!("generator.ups.{i}.bias"));
            let u = RATES[i];
            let kk = KERNELS[i];
            let pad = u / 2 + u % 2;
            let tup = (tcur - 1) * u + (kk - 1) + (u % 2) + 1 - 2 * pad;
            let up = self.alloc(cout * tup);
            self.conv_transpose1d_w(
                &mut enc,
                &sn,
                &format!("generator.ups.{i}.weight"),
                Some(&ub),
                &up,
                cin,
                tcur,
                cout,
                kk,
                u,
                pad,
                u % 2,
                1,
            );
            debug_assert_eq!(tup, ts, "stage {i}: up {tup} != src {ts}");
            residual_add_chained(self.ctx, self.p, &mut enc, &up, &xsrc, cout * tup);
            let acc = self.alloc(cout * tup);
            for (j, (&rk, rd)) in RK.iter().zip(rdil.iter()).enumerate() {
                let rb = self.adain_resblock1(
                    &mut enc,
                    &up,
                    cout,
                    tup,
                    rk,
                    [rd[0], rd[1], rd[2]],
                    &format!("generator.resblocks.{}", i * 3 + j),
                    style,
                );
                residual_add_chained(self.ctx, self.p, &mut enc, &acc, &rb, cout * tup);
            }
            scale_chained(self.ctx, self.p, &mut enc, &acc, cout * tup, 1.0 / 3.0);
            self.submit(enc);
            // Yield between upsample stages — this is the heaviest, longest GPU burst of the synth
            // (the end-of-gen vocoder) and the one that was tripping the iPhone springboard/jetsam.
            // The yield also lets a queued `cancel` land, so Stop works mid-vocoder.
            self.wasm_yield().await;
            if crate::cancel::requested() {
                return None;
            }
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
        let cpb = self.wt("generator.conv_post.bias");
        let post = self.alloc(tcur);
        self.conv1d_w(
            &mut enc,
            &sn,
            "generator.conv_post.weight",
            Some(&cpb),
            &post,
            cin,
            tcur,
            1,
            7,
            1,
            3,
            1,
            1,
        );
        self.submit(enc);
        self.wasm_yield().await;
        Some((post, tcur))
    }

    async fn read(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let rd = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rd"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rd") });
        e.copy_buffer_to_buffer(buf, 0, &rd, 0, (n * 4) as u64);
        self.ctx.queue.submit(Some(e.finish()));
        read_back_f32(&self.ctx.device, &rd)
            .await
            .expect("readback")
    }

    /// GPU StyleEncoder: reference mel `[n_mels, t]` → 128-d style vector. `prefix` =
    /// "acoustic" or "prosodic". The conv-heavy stack runs on GPU; the tiny adaptive-pool +
    /// Linear tail (512→128) runs on CPU after a single readback.
    async fn style_encoder(
        &mut self,
        mel_buf: &wgpu::Buffer,
        n_mels: usize,
        t: usize,
        prefix: &str,
    ) -> Vec<f32> {
        const BLK: [(usize, usize); 4] = [(64, 128), (128, 256), (256, 512), (512, 512)];
        let pn = |s: &str| format!("{prefix}.{s}");
        let mut enc = self.enc();
        // conv0: 1 → 64, k3 p1
        let c0b = self.wt(&pn("conv0.bias"));
        let mut x = self.alloc(64 * n_mels * t);
        self.conv2d_chf_w(
            &mut enc,
            &pn("conv0.weight"),
            mel_buf,
            Some(&c0b),
            &x,
            1,
            n_mels,
            t,
            64,
            n_mels,
            t,
            3,
            3,
            1,
            1,
            1,
            1,
            1,
        );
        let (mut h, mut w) = (n_mels, t);
        for (i, &(din, dout)) in BLK.iter().enumerate() {
            let (h2, w2) = (h / 2, w.div_ceil(2));
            // shortcut: optional 1×1 conv (when channels change) then avg-pool
            let sc = if din != dout {
                let o = self.alloc(dout * h * w);
                self.conv2d_chf_w(
                    &mut enc,
                    &pn(&format!("blk{i}.sc.weight")),
                    &x,
                    None,
                    &o,
                    din,
                    h,
                    w,
                    dout,
                    h,
                    w,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                );
                o
            } else {
                x.clone()
            };
            let sc_pool = self.alloc(dout * h2 * w2);
            avg_pool2d_half_chained(
                self.ctx, self.p, &mut enc, &sc, &sc_pool, dout, h, w, h2, w2,
            );
            // residual: leaky → conv1(k3p1) → strided depthwise down → leaky → conv2(k3p1)
            let r = self.alloc(din * h * w);
            enc.copy_buffer_to_buffer(&x, 0, &r, 0, (din * h * w * 4) as u64);
            leaky_relu_chained(self.ctx, self.p, &mut enc, &r, din * h * w, 0.2);
            let c1b = self.wt(&pn(&format!("blk{i}.conv1.bias")));
            let c1 = self.alloc(din * h * w);
            self.conv2d_chf_w(
                &mut enc,
                &pn(&format!("blk{i}.conv1.weight")),
                &r,
                Some(&c1b),
                &c1,
                din,
                h,
                w,
                din,
                h,
                w,
                3,
                3,
                1,
                1,
                1,
                1,
                1,
            );
            let db = self.wt(&pn(&format!("blk{i}.down.bias")));
            let dn = self.alloc(din * h2 * w2);
            self.conv2d_chf_w(
                &mut enc,
                &pn(&format!("blk{i}.down.weight")),
                &c1,
                Some(&db),
                &dn,
                din,
                h,
                w,
                din,
                h2,
                w2,
                3,
                3,
                2,
                2,
                1,
                1,
                din,
            );
            leaky_relu_chained(self.ctx, self.p, &mut enc, &dn, din * h2 * w2, 0.2);
            let c2b = self.wt(&pn(&format!("blk{i}.conv2.bias")));
            let c2 = self.alloc(dout * h2 * w2);
            self.conv2d_chf_w(
                &mut enc,
                &pn(&format!("blk{i}.conv2.weight")),
                &dn,
                Some(&c2b),
                &c2,
                din,
                h2,
                w2,
                dout,
                h2,
                w2,
                3,
                3,
                1,
                1,
                1,
                1,
                1,
            );
            residual_add_chained(self.ctx, self.p, &mut enc, &c2, &sc_pool, dout * h2 * w2);
            scale_chained(self.ctx, self.p, &mut enc, &c2, dout * h2 * w2, RSQRT2);
            x = c2;
            h = h2;
            w = w2;
        }
        // leaky → conv_out (512→512, k5, no pad) → [512, h-4, w-4]
        leaky_relu_chained(self.ctx, self.p, &mut enc, &x, 512 * h * w, 0.2);
        let (oh, ow) = (h - 4, w - 4);
        let cob = self.wt(&pn("conv_out.bias"));
        let co = self.alloc(512 * oh * ow);
        self.conv2d_chf_w(
            &mut enc,
            &pn("conv_out.weight"),
            &x,
            Some(&cob),
            &co,
            512,
            h,
            w,
            512,
            oh,
            ow,
            5,
            5,
            1,
            1,
            0,
            0,
            1,
        );
        self.submit(enc);
        // adaptive avg pool (mean over oh·ow) + leaky + Linear(512→128) on CPU (tiny)
        let feat = self.read(&co, 512 * oh * ow).await;
        let mut pooled: Vec<f32> = (0..512)
            .map(|c| feat[c * oh * ow..(c + 1) * oh * ow].iter().sum::<f32>() / (oh * ow) as f32)
            .collect();
        leaky_cpu(&mut pooled, 0.2);
        linear(
            &pooled,
            1,
            512,
            self.t(&pn("linear.weight")),
            Some(self.t(&pn("linear.bias"))),
            128,
        )
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
        let buf = write_storage_f16(
            &self.ctx.device,
            &self.ctx.queue,
            name,
            self.w
                .get(name)
                .unwrap_or_else(|| panic!("missing diff weight {name}")),
        );
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
    fn glin(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        rows: usize,
        kin: usize,
        nout: usize,
        w: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
    ) -> wgpu::Buffer {
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
    pub async fn diffusion_sample(
        &mut self,
        emb: &[f32],
        l: usize,
        ref_s: &[f32],
        noise_init: &[f32],
        noises: &[Vec<f32>],
        sigma_data: f32,
        sigma_min: f32,
        sigma_max: f32,
        rho: f32,
        steps: usize,
    ) -> Vec<f32> {
        // Karras schedule
        let inv = 1.0 / rho;
        let (a, b) = (sigma_max.powf(inv), sigma_min.powf(inv));
        let mut sig: Vec<f32> = (0..steps)
            .map(|i| (a + (i as f32 / (steps - 1) as f32) * (b - a)).powf(rho))
            .collect();
        sig.push(0.0);
        // ADPM2 (scalar math on CPU; the net eval on GPU)
        let mut x: Vec<f32> = noise_init.iter().map(|v| sig[0] * v).collect();
        for i in 0..steps - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            let sigma_up = (sn * sn * (s * s - sn * sn) / (s * s)).sqrt();
            let sigma_down = (sn * sn - sigma_up * sigma_up).sqrt();
            let sigma_mid = (s + sigma_down) * 0.5;
            let dn = self.diff_denoise(&x, s, sigma_data, emb, l, ref_s).await;
            self.wasm_yield().await;
            let d: Vec<f32> = (0..256).map(|k| (x[k] - dn[k]) / s).collect();
            let x_mid: Vec<f32> = (0..256).map(|k| x[k] + d[k] * (sigma_mid - s)).collect();
            let dn_mid = self
                .diff_denoise(&x_mid, sigma_mid, sigma_data, emb, l, ref_s)
                .await;
            self.wasm_yield().await;
            let d_mid: Vec<f32> = (0..256)
                .map(|k| (x_mid[k] - dn_mid[k]) / sigma_mid)
                .collect();
            let nz = &noises[i];
            for k in 0..256 {
                x[k] = x[k] + d_mid[k] * (sigma_down - s) + nz[k] * sigma_up;
            }
        }
        x
    }

    /// KDiffusion denoise_fn: c_skip·x + c_out·net(c_in·x, c_noise).
    async fn diff_denoise(
        &mut self,
        x: &[f32],
        sigma: f32,
        sd: f32,
        emb: &[f32],
        l: usize,
        ref_s: &[f32],
    ) -> Vec<f32> {
        let c_skip = sd * sd / (sigma * sigma + sd * sd);
        let c_out = sigma * sd / (sd * sd + sigma * sigma).sqrt();
        let c_in = 1.0 / (sigma * sigma + sd * sd).sqrt();
        let c_noise = sigma.ln() * 0.25;
        let xin: Vec<f32> = x.iter().map(|v| c_in * v).collect();
        let pred = self.diff_net(&xin, c_noise, emb, l, ref_s).await;
        (0..256).map(|k| c_skip * x[k] + c_out * pred[k]).collect()
    }

    /// Test hook: one isolated GPU denoiser eval (parity vs the CPU oracle's `net_eval`).
    pub async fn diff_net_eval(
        &mut self,
        x: &[f32],
        time: f32,
        emb: &[f32],
        l: usize,
        ref_s: &[f32],
    ) -> Vec<f32> {
        self.diff_net(x, time, emb, l, ref_s).await
    }

    /// One denoiser network eval on GPU. (x[256], time, emb[l,768], ref_s[256]) → [256].
    async fn diff_net(
        &mut self,
        x: &[f32],
        time: f32,
        emb: &[f32],
        l: usize,
        ref_s: &[f32],
    ) -> Vec<f32> {
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
            layernorm_affine_chained(
                self.ctx,
                self.p,
                &mut enc,
                &hb,
                Some(&gn),
                Some(&bn),
                &self.dummy,
                &xn,
                l,
                F,
                1e-5,
            );
            let cn = self.alloc(l * F);
            layernorm_affine_chained(
                self.ctx,
                self.p,
                &mut enc,
                &hb,
                Some(&gc),
                Some(&bc),
                &self.dummy,
                &cn,
                l,
                F,
                1e-5,
            );
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
        linear(
            &pooled,
            1,
            F,
            self.t("diffusion.to_out.1.weight"),
            Some(self.t("diffusion.to_out.1.bias")),
            256,
        )
    }

    /// AdaLayerNorm affine: returns uploaded ((1+γ_fc), β_fc) ∈ ℝ¹⁰²⁴, γ/β = fc(ref_s).
    fn diff_adaln_affine(
        &mut self,
        fc_prefix: &str,
        ref_s: &[f32],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
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
                let y = 1.0
                    - (((((1.061_405_4 * t - 1.453_152_) * t + 1.421_413_7) * t - 0.284_496_74)
                        * t
                        + 0.254_829_6)
                        * t)
                        * (-z * z).exp();
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
        let mut t_emb = linear(
            &tpos,
            1,
            257,
            self.t("diffusion.to_time.0.1.weight"),
            Some(self.t("diffusion.to_time.0.1.bias")),
            1024,
        );
        gelu(&mut t_emb);
        let mut f_emb = linear(
            ref_s,
            1,
            256,
            self.t("diffusion.to_features.0.weight"),
            Some(self.t("diffusion.to_features.0.bias")),
            1024,
        );
        gelu(&mut f_emb);
        let mut m: Vec<f32> = (0..1024).map(|k| t_emb[k] + f_emb[k]).collect();
        m = linear(
            &m,
            1,
            1024,
            self.t("diffusion.to_mapping.0.weight"),
            Some(self.t("diffusion.to_mapping.0.bias")),
            1024,
        );
        gelu(&mut m);
        m = linear(
            &m,
            1,
            1024,
            self.t("diffusion.to_mapping.2.weight"),
            Some(self.t("diffusion.to_mapping.2.bias")),
            1024,
        );
        gelu(&mut m);
        m
    }

    /// Full hifigan decoder on GPU: `asr [512,f]`, `f0`/`n [2f]` (CPU), `style [128]` → 24 kHz waveform.
    pub async fn decode(
        &mut self,
        asr: &[f32],
        f: usize,
        f0: &[f32],
        n: &[f32],
        style: &[f32],
    ) -> Vec<f32> {
        let dbg = std::env::var("ST2DBG").is_ok();
        let asr_buf = self.up(asr);
        let f0_buf = self.up(f0);
        let n_buf = self.up(n);
        // ---- decoder cat-stack (one submit; small tensors, t ≤ 2f) ----
        let mut enc = self.enc();
        let f0b = self.wt("F0_conv.bias");
        let f0d = self.alloc(f);
        self.conv1d_w(
            &mut enc,
            &f0_buf,
            "F0_conv.weight",
            Some(&f0b),
            &f0d,
            1,
            2 * f,
            1,
            3,
            2,
            1,
            1,
            1,
        );
        let nb = self.wt("N_conv.bias");
        let nd = self.alloc(f);
        self.conv1d_w(
            &mut enc,
            &n_buf,
            "N_conv.weight",
            Some(&nb),
            &nd,
            1,
            2 * f,
            1,
            3,
            2,
            1,
            1,
            1,
        );
        let cat0 = self.concat(&mut enc, &[(&asr_buf, 512), (&f0d, 1), (&nd, 1)], f);
        let (mut x, mut tcur) =
            self.adain_resblk1d(&mut enc, &cat0, 514, f, 1024, false, "encode", style);
        let arb = self.wt("asr_res.0.bias");
        let asr_res = self.alloc(64 * f);
        self.conv1d_w(
            &mut enc,
            &asr_buf,
            "asr_res.0.weight",
            Some(&arb),
            &asr_res,
            512,
            f,
            64,
            1,
            1,
            0,
            1,
            1,
        );
        // x is 1024 channels before every decode block (encode → 1024; blocks 0-2 stay
        // 1024; block 3 outputs 512 but its *input* is still 1024).
        for i in 0..4 {
            let xin = self.concat(
                &mut enc,
                &[(&x, 1024), (&asr_res, 64), (&f0d, 1), (&nd, 1)],
                tcur,
            );
            let (nx, nt) = self.adain_resblk1d(
                &mut enc,
                &xin,
                1090,
                tcur,
                if i < 3 { 1024 } else { 512 },
                i == 3,
                &format!("decode.{i}"),
                style,
            );
            x = nx;
            tcur = nt;
        }
        self.submit(enc);
        self.wasm_yield().await;
        if dbg {
            self.dbg("decode_x", &x, 512 * tcur).await;
        }
        // ---- har source (CPU) → generator (one submit per upsample stage) ----
        let lw = self.t("generator.m_source.l_linear.weight").to_vec();
        let lb = self.t("generator.m_source.l_linear.bias")[0];
        let har = source_signal(f0, 300, 9, &lw, lb);
        let har_buf = self.up(&har);
        // `None` ⇒ Stop was clicked mid-vocoder → resolve empty (treated as cancelled).
        let Some((post, tpost)) = self.generator(x, tcur, &har_buf, har.len(), style).await else {
            return Vec::new();
        };
        if dbg {
            self.dbg("post", &post, tpost).await;
        }
        // ---- readback + tanh on CPU ----
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rd"),
            size: (tpost * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e2 = self.enc();
        e2.copy_buffer_to_buffer(&post, 0, &read, 0, (tpost * 4) as u64);
        self.submit(e2);
        let raw = read_back_f32(&self.ctx.device, &read)
            .await
            .expect("readback");
        raw.iter().map(|v| v.tanh()).collect()
    }
}
