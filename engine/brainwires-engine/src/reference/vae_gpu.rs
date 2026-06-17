//! GPU VAE decoder (latent → RGB) — the first full GPU component forward, and
//! the first that streams its weights asynchronously so it runs in the browser.
//! Composes the parity-tested image kernels (conv2d_chw_f32, groupnorm, silu,
//! upsample2x_chw, residual_add); the single tiny mid-block self-attention runs
//! via a CPU readback (it's at latent resolution — a few dozen tokens, once).
//! Validated against the reference::vae CPU oracle.
//!
//! Channel-first `[C,H,W]` throughout, matching the CPU oracle and the kernels.
//! Generic over a [`BlobSource`] via [`StreamingShards`], so the identical code
//! runs natively (`FileBlobSource`, the parity harness) and in wasm
//! (`HttpRangeBlobSource`) — weights are range-fetched per tensor, never bulk.

use wgpu::util::DeviceExt;

use crate::backend::dispatch::{
    conv2d_chw_f32_chained, groupnorm_chained, read_back_f32, residual_add_chained, silu_chained,
    upsample2x_chw_chained,
};
use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::imagegen::config::VaeConfig;
use crate::imagegen::source::BlobSource;
use crate::imagegen::streaming::StreamingShards;

/// A GPU activation buffer with its channel-first dims.
struct Act {
    buf: wgpu::Buffer,
    c: usize,
    h: usize,
    w: usize,
}

pub struct VaeGpu<'a, S: BlobSource> {
    ctx: &'a WgpuCtx,
    pipes: &'a Pipelines,
    st: &'a StreamingShards<S>,
    cfg: &'a VaeConfig,
    dummy: wgpu::Buffer,
}

impl<'a, S: BlobSource> VaeGpu<'a, S> {
    pub fn new(
        ctx: &'a WgpuCtx,
        pipes: &'a Pipelines,
        st: &'a StreamingShards<S>,
        cfg: &'a VaeConfig,
    ) -> Self {
        let dummy = upload(&ctx.device, "vae.dummy", &[0.0f32]);
        Self {
            ctx,
            pipes,
            st,
            cfg,
            dummy,
        }
    }

    /// Decode latent `[latent_ch, lh, lw]` → RGB `[3, lh·8, lw·8]` in `[0,1]`.
    pub async fn decode(&self, latent: &[f32], lh: usize, lw: usize) -> Result<Vec<f32>> {
        let lc = self.cfg.latent_channels as usize;
        let groups = self.cfg.norm_num_groups as usize;
        let dev = &self.ctx.device;

        // pre-scale on CPU, upload as the initial activation
        let z: Vec<f32> = latent
            .iter()
            .map(|v| v / self.cfg.scaling_factor + self.cfg.shift_factor)
            .collect();
        let mut x = Act {
            buf: storage_rw_init(dev, "vae.z", &z),
            c: lc,
            h: lh,
            w: lw,
        };

        // conv_in (3×3 pad1)
        x = self.conv(&x, "decoder.conv_in", 1).await?;
        // mid block: resnet0 → attn → resnet1
        x = self
            .resnet(&x, "decoder.mid_block.resnets.0", groups)
            .await?;
        x = self
            .attn_cpu(&x, "decoder.mid_block.attentions.0", groups)
            .await?;
        x = self
            .resnet(&x, "decoder.mid_block.resnets.1", groups)
            .await?;

        // up blocks
        let n_up = self.cfg.block_out_channels.len();
        let resnets = self.cfg.layers_per_block as usize + 1;
        for bi in 0..n_up {
            let bp = format!("decoder.up_blocks.{bi}");
            for ri in 0..resnets {
                x = self
                    .resnet(&x, &format!("{bp}.resnets.{ri}"), groups)
                    .await?;
            }
            if self.st.has(&format!("{bp}.upsamplers.0.conv.weight")) {
                x = self.upsample(&x);
                x = self.conv(&x, &format!("{bp}.upsamplers.0.conv"), 1).await?;
            }
        }

        // conv_norm_out (GroupNorm) → silu → conv_out
        x = self.groupnorm(&x, "decoder.conv_norm_out", groups).await?;
        self.silu(&x);
        x = self.conv(&x, "decoder.conv_out", 1).await?;

        // readback, [-1,1]→[0,1] clip
        let mut out = self.read(&x).await?;
        for v in out.iter_mut() {
            *v = (*v * 0.5 + 0.5).clamp(0.0, 1.0);
        }
        Ok(out)
    }

    // ---- ops ----

    async fn conv(&self, x: &Act, p: &str, pad: usize) -> Result<Act> {
        let ws = self.st.shape_of(&format!("{p}.weight"))?;
        let (oc, _ic, k) = (ws[0], ws[1], ws[2]);
        let w = upload(
            &self.ctx.device,
            "w",
            &self.st.tensor_f32(&format!("{p}.weight")).await?,
        );
        let b = upload(
            &self.ctx.device,
            "b",
            &self.st.tensor_f32(&format!("{p}.bias")).await?,
        );
        let (oh, ow) = (x.h + 2 * pad - k + 1, x.w + 2 * pad - k + 1);
        let out = storage_rw(&self.ctx.device, "conv.out", oc * oh * ow);
        let mut enc = self.encoder("conv");
        conv2d_chw_f32_chained(
            self.ctx, self.pipes, &mut enc, &x.buf, &w, &b, &out, x.c, x.h, x.w, oc, k, pad,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        Ok(Act {
            buf: out,
            c: oc,
            h: oh,
            w: ow,
        })
    }

    async fn groupnorm(&self, x: &Act, p: &str, groups: usize) -> Result<Act> {
        let g = upload(
            &self.ctx.device,
            "gn.g",
            &self.st.tensor_f32(&format!("{p}.weight")).await?,
        );
        let b = upload(
            &self.ctx.device,
            "gn.b",
            &self.st.tensor_f32(&format!("{p}.bias")).await?,
        );
        let out = storage_rw(&self.ctx.device, "gn.out", x.c * x.h * x.w);
        let mut enc = self.encoder("gn");
        groupnorm_chained(
            self.ctx,
            self.pipes,
            &mut enc,
            &x.buf,
            Some(&g),
            Some(&b),
            &self.dummy,
            &out,
            groups,
            x.c / groups,
            x.h * x.w,
            1e-6,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        Ok(Act {
            buf: out,
            c: x.c,
            h: x.h,
            w: x.w,
        })
    }

    fn silu(&self, x: &Act) {
        let mut enc = self.encoder("silu");
        silu_chained(self.ctx, self.pipes, &mut enc, &x.buf, x.c * x.h * x.w);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    async fn resnet(&self, x: &Act, p: &str, groups: usize) -> Result<Act> {
        let mut h = self.groupnorm(x, &format!("{p}.norm1"), groups).await?;
        self.silu(&h);
        h = self.conv(&h, &format!("{p}.conv1"), 1).await?;
        h = self.groupnorm(&h, &format!("{p}.norm2"), groups).await?;
        self.silu(&h);
        h = self.conv(&h, &format!("{p}.conv2"), 1).await?;
        // residual (1×1 shortcut conv when channels changed): h += shortcut(x)
        let res = if self.st.has(&format!("{p}.conv_shortcut.weight")) {
            self.conv(x, &format!("{p}.conv_shortcut"), 0).await?
        } else {
            Act {
                buf: clone_buf(self.ctx, &x.buf, x.c * x.h * x.w),
                c: x.c,
                h: x.h,
                w: x.w,
            }
        };
        let mut enc = self.encoder("resadd");
        residual_add_chained(
            self.ctx,
            self.pipes,
            &mut enc,
            &h.buf,
            &res.buf,
            h.c * h.h * h.w,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        Ok(h)
    }

    fn upsample(&self, x: &Act) -> Act {
        let out = storage_rw(&self.ctx.device, "up.out", x.c * 4 * x.h * x.w);
        let mut enc = self.encoder("up");
        upsample2x_chw_chained(self.ctx, self.pipes, &mut enc, &x.buf, &out, x.c, x.h, x.w);
        self.ctx.queue.submit(Some(enc.finish()));
        Act {
            buf: out,
            c: x.c,
            h: x.h * 2,
            w: x.w * 2,
        }
    }

    /// Mid-block self-attention via CPU readback (latent res, few tokens, once).
    async fn attn_cpu(&self, x: &Act, p: &str, groups: usize) -> Result<Act> {
        let (c, n) = (x.c, x.h * x.w);
        // GroupNorm on GPU, read back to CPU for the small attention.
        let gn = self
            .groupnorm(x, &format!("{p}.group_norm"), groups)
            .await?;
        let normed = self.read(&gn).await?; // [c, n] channel-first
        // to [n, c]
        let mut tok = vec![0.0f32; n * c];
        for ch in 0..c {
            for t in 0..n {
                tok[t * c + ch] = normed[ch * n + t];
            }
        }
        // small linear over the readback (weights streamed per call).
        let lin = |inp: &[f32], w: &[f32], b: &[f32]| -> Vec<f32> {
            let mut y = vec![0.0f32; n * c];
            for r in 0..n {
                for o in 0..c {
                    let mut a = b[o];
                    for i in 0..c {
                        a += inp[r * c + i] * w[o * c + i];
                    }
                    y[r * c + o] = a;
                }
            }
            y
        };
        let (qw, qb) = self.load_wb(p, "to_q").await?;
        let (kw, kb) = self.load_wb(p, "to_k").await?;
        let (vw, vb) = self.load_wb(p, "to_v").await?;
        let q = lin(&tok, &qw, &qb);
        let k = lin(&tok, &kw, &kb);
        let v = lin(&tok, &vw, &vb);
        let scale = 1.0f32 / (c as f32).sqrt();
        let mut ctx_o = vec![0.0f32; n * c];
        for ti in 0..n {
            let mut sc = vec![0.0f32; n];
            let mut mx = f32::NEG_INFINITY;
            for tj in 0..n {
                let mut d = 0.0f32;
                for x in 0..c {
                    d += q[ti * c + x] * k[tj * c + x];
                }
                sc[tj] = d * scale;
                mx = mx.max(sc[tj]);
            }
            let mut s = 0.0f32;
            for v2 in sc.iter_mut() {
                *v2 = (*v2 - mx).exp();
                s += *v2;
            }
            for x in 0..c {
                let mut a = 0.0f32;
                for tj in 0..n {
                    a += sc[tj] * v[tj * c + x];
                }
                ctx_o[ti * c + x] = a / s;
            }
        }
        let (ow, ob) = self.load_wb(p, "to_out.0").await?;
        let out = lin(&ctx_o, &ow, &ob);
        // back to [c, n] + residual (read original x)
        let xv = self.read(x).await?;
        let mut y = xv;
        for ch in 0..c {
            for t in 0..n {
                y[ch * n + t] += out[t * c + ch];
            }
        }
        Ok(Act {
            buf: storage_rw_init(&self.ctx.device, "attn.out", &y),
            c,
            h: x.h,
            w: x.w,
        })
    }

    /// Stream a `<prefix>.<name>.{weight,bias}` pair to f32.
    async fn load_wb(&self, prefix: &str, name: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        let w = self
            .st
            .tensor_f32(&format!("{prefix}.{name}.weight"))
            .await?;
        let b = self.st.tensor_f32(&format!("{prefix}.{name}.bias")).await?;
        Ok((w, b))
    }

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    async fn read(&self, x: &Act) -> Result<Vec<f32>> {
        let n = x.c * x.h * x.w;
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vae.read"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.encoder("read");
        enc.copy_buffer_to_buffer(&x.buf, 0, &read, 0, (n * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));
        read_back_f32(&self.ctx.device, &read).await
    }
}

fn upload(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn storage_rw(device: &wgpu::Device, label: &str, n: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_rw_init(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn clone_buf(ctx: &WgpuCtx, src: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    let dst = storage_rw(&ctx.device, "clone", n);
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clone"),
        });
    enc.copy_buffer_to_buffer(src, 0, &dst, 0, (n * 4) as u64);
    ctx.queue.submit(Some(enc.finish()));
    dst
}
