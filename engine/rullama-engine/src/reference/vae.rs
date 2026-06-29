//! CPU f32 forward for the Z-Image VAE decoder (latent → RGB).
//!
//! Faithful port of Ollama `x/imagegen/models/zimage/vae.go` decoder path,
//! computed channel-first `[C, H, W]` internally (Ollama uses NHWC; the math is
//! identical). The oracle for the eventual GPU VAE.
//!
//! Decode(latent [16,h,w]):
//!   z = latent / scaling_factor + shift_factor
//!   h = conv_in(z)                      (3×3 pad1, 16→512)
//!   h = mid: resnet → self-attn → resnet      (all top channel)
//!   for up_block: resnets, then (nearest-2× + conv) except the last
//!   h = silu(group_norm(h)); h = conv_out(h)  (3×3 pad1, 128→3)
//!   rgb = clip(h*0.5 + 0.5, 0, 1)

use crate::error::{Result, RullamaError};
use crate::imagegen::config::VaeConfig;
use crate::imagegen::sharded::ShardedSafetensors;
use crate::reference::imagegen::group_norm;

/// A `[C, H, W]` f32 feature map.
struct Chw {
    c: usize,
    h: usize,
    w: usize,
    d: Vec<f32>,
}

/// VAE decoder over a single-file safetensors weight set.
pub struct VaeDecoder<'a> {
    st: &'a ShardedSafetensors,
    cfg: &'a VaeConfig,
}

impl<'a> VaeDecoder<'a> {
    pub fn new(st: &'a ShardedSafetensors, cfg: &'a VaeConfig) -> Self {
        Self { st, cfg }
    }

    /// Decode a latent `[latent_ch, lh, lw]` → RGB `[3, lh*8, lw*8]`, row-major
    /// per channel, values in `[0, 1]`.
    pub fn decode(&self, latent: &[f32], lh: usize, lw: usize) -> Result<Vec<f32>> {
        let lc = self.cfg.latent_channels as usize;
        if latent.len() != lc * lh * lw {
            return Err(RullamaError::Image(format!(
                "latent len {} != {lc}×{lh}×{lw}",
                latent.len()
            )));
        }
        let groups = self.cfg.norm_num_groups as usize;

        // pre-scale
        let z: Vec<f32> = latent
            .iter()
            .map(|v| v / self.cfg.scaling_factor + self.cfg.shift_factor)
            .collect();
        let mut x = Chw {
            c: lc,
            h: lh,
            w: lw,
            d: z,
        };

        // conv_in (3×3 pad1)
        x = self.conv(&x, "decoder.conv_in", 1)?;

        // mid block: resnet0 → attn → resnet1
        x = self.resnet(&x, "decoder.mid_block.resnets.0", groups)?;
        x = self.attn(&x, "decoder.mid_block.attentions.0", groups)?;
        x = self.resnet(&x, "decoder.mid_block.resnets.1", groups)?;

        // up blocks: reversed channels; layers_per_block+1 resnets each;
        // upsample (nearest-2× + conv) on all but the last block.
        let n_up = self.cfg.block_out_channels.len();
        let resnets = self.cfg.layers_per_block as usize + 1;
        for bi in 0..n_up {
            let bp = format!("decoder.up_blocks.{bi}");
            for ri in 0..resnets {
                x = self.resnet(&x, &format!("{bp}.resnets.{ri}"), groups)?;
            }
            if self.st.has(&format!("{bp}.upsamplers.0.conv.weight")) {
                x = upsample2x(&x);
                x = self.conv(&x, &format!("{bp}.upsamplers.0.conv"), 1)?;
            }
        }

        // conv_norm_out (GroupNorm) → silu → conv_out (3×3 pad1)
        let (gnw, gnb) = self.norm_pair("decoder.conv_norm_out")?;
        x.d = group_norm(
            &x.d,
            groups,
            x.c / groups,
            x.h * x.w,
            Some(&gnw),
            Some(&gnb),
            1e-6,
        );
        silu_(&mut x.d);
        x = self.conv(&x, "decoder.conv_out", 1)?;

        // [-1,1] → [0,1], clip
        for v in x.d.iter_mut() {
            *v = (*v * 0.5 + 0.5).clamp(0.0, 1.0);
        }
        Ok(x.d)
    }

    // ---- blocks ----

    fn resnet(&self, x: &Chw, p: &str, groups: usize) -> Result<Chw> {
        let (n1w, n1b) = self.norm_pair(&format!("{p}.norm1"))?;
        let mut h = Chw {
            c: x.c,
            h: x.h,
            w: x.w,
            d: group_norm(
                &x.d,
                groups,
                x.c / groups,
                x.h * x.w,
                Some(&n1w),
                Some(&n1b),
                1e-6,
            ),
        };
        silu_(&mut h.d);
        h = self.conv(&h, &format!("{p}.conv1"), 1)?;

        let (n2w, n2b) = self.norm_pair(&format!("{p}.norm2"))?;
        h.d = group_norm(
            &h.d,
            groups,
            h.c / groups,
            h.h * h.w,
            Some(&n2w),
            Some(&n2b),
            1e-6,
        );
        silu_(&mut h.d);
        h = self.conv(&h, &format!("{p}.conv2"), 1)?;

        // residual (1×1 conv shortcut if channels changed)
        let res = if self.st.has(&format!("{p}.conv_shortcut.weight")) {
            self.conv(x, &format!("{p}.conv_shortcut"), 0)?
        } else {
            Chw {
                c: x.c,
                h: x.h,
                w: x.w,
                d: x.d.clone(),
            }
        };
        for (hv, rv) in h.d.iter_mut().zip(&res.d) {
            *hv += rv;
        }
        Ok(h)
    }

    fn attn(&self, x: &Chw, p: &str, groups: usize) -> Result<Chw> {
        let c = x.c;
        let n = x.h * x.w; // tokens
        let (gw, gb) = self.norm_pair(&format!("{p}.group_norm"))?;
        let normed = group_norm(&x.d, groups, c / groups, n, Some(&gw), Some(&gb), 1e-6);
        // to [n, c] (tokens × channels)
        let mut h = vec![0.0f32; n * c];
        for ch in 0..c {
            for t in 0..n {
                h[t * c + ch] = normed[ch * n + t];
            }
        }
        let q = self.linear(&h, n, c, &format!("{p}.to_q"), c)?;
        let k = self.linear(&h, n, c, &format!("{p}.to_k"), c)?;
        let v = self.linear(&h, n, c, &format!("{p}.to_v"), c)?;
        // single-head softmax attention, scale 1/sqrt(c)
        let scale = 1.0f32 / (c as f32).sqrt();
        let mut ctx = vec![0.0f32; n * c];
        for ti in 0..n {
            let mut scores = vec![0.0f32; n];
            let mut maxs = f32::NEG_INFINITY;
            for tj in 0..n {
                let mut dot = 0.0f32;
                for d in 0..c {
                    dot += q[ti * c + d] * k[tj * c + d];
                }
                scores[tj] = dot * scale;
                if scores[tj] > maxs {
                    maxs = scores[tj];
                }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - maxs).exp();
                sum += *s;
            }
            for d in 0..c {
                let mut acc = 0.0f32;
                for tj in 0..n {
                    acc += scores[tj] * v[tj * c + d];
                }
                ctx[ti * c + d] = acc / sum;
            }
        }
        let out = self.linear(&ctx, n, c, &format!("{p}.to_out.0"), c)?;
        // back to [c, h, w] + residual
        let mut y = x.d.clone();
        for ch in 0..c {
            for t in 0..n {
                y[ch * n + t] += out[t * c + ch];
            }
        }
        Ok(Chw {
            c,
            h: x.h,
            w: x.w,
            d: y,
        })
    }

    // ---- ops ----

    /// Conv2D `[Cin,H,W] → [Cout,Hout,Wout]` with square kernel inferred from
    /// the weight shape `[Cout,Cin,kh,kw]`, stride 1, zero-pad `pad`.
    fn conv(&self, x: &Chw, p: &str, pad: usize) -> Result<Chw> {
        let wshape = self.st.shape(&format!("{p}.weight"))?;
        let (cout, cin, kh, kw) = (wshape[0], wshape[1], wshape[2], wshape[3]);
        if cin != x.c {
            return Err(RullamaError::Image(format!(
                "{p}: weight cin {cin} != input c {}",
                x.c
            )));
        }
        let weight = self.st.tensor_f32(&format!("{p}.weight"))?;
        let bias = self.st.tensor_f32(&format!("{p}.bias"))?;
        let (h, w) = (x.h, x.w);
        let hout = h + 2 * pad - kh + 1;
        let wout = w + 2 * pad - kw + 1;
        let mut y = vec![0.0f32; cout * hout * wout];
        for co in 0..cout {
            let wbase = co * cin * kh * kw;
            for oy in 0..hout {
                for ox in 0..wout {
                    let mut acc = bias[co];
                    for ci in 0..cin {
                        let xb = ci * h * w;
                        let wb = wbase + ci * kh * kw;
                        for ky in 0..kh {
                            let iy = oy + ky;
                            if iy < pad || iy >= h + pad {
                                continue;
                            }
                            let iy = iy - pad;
                            for kx in 0..kw {
                                let ix = ox + kx;
                                if ix < pad || ix >= w + pad {
                                    continue;
                                }
                                let ix = ix - pad;
                                acc += x.d[xb + iy * w + ix] * weight[wb + ky * kw + kx];
                            }
                        }
                    }
                    y[co * hout * wout + oy * wout + ox] = acc;
                }
            }
        }
        Ok(Chw {
            c: cout,
            h: hout,
            w: wout,
            d: y,
        })
    }

    /// Linear `y[t,o] = Σ_i x[t,i]·W[o,i] + b[o]`, weight `[out,in]`.
    fn linear(
        &self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        p: &str,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let w = self.st.tensor_f32(&format!("{p}.weight"))?;
        let b = self.st.tensor_f32(&format!("{p}.bias"))?;
        let mut y = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut acc = b[o];
                for i in 0..in_dim {
                    acc += x[r * in_dim + i] * w[o * in_dim + i];
                }
                y[r * out_dim + o] = acc;
            }
        }
        Ok(y)
    }

    fn norm_pair(&self, p: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((
            self.st.tensor_f32(&format!("{p}.weight"))?,
            self.st.tensor_f32(&format!("{p}.bias"))?,
        ))
    }
}

fn silu_(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// Nearest-neighbor 2× upsample of `[C,H,W]` → `[C,2H,2W]`.
fn upsample2x(x: &Chw) -> Chw {
    let (h2, w2) = (x.h * 2, x.w * 2);
    let mut d = vec![0.0f32; x.c * h2 * w2];
    for c in 0..x.c {
        for y in 0..h2 {
            for xx in 0..w2 {
                d[c * h2 * w2 + y * w2 + xx] = x.d[c * x.h * x.w + (y / 2) * x.w + (xx / 2)];
            }
        }
    }
    Chw {
        c: x.c,
        h: h2,
        w: w2,
        d,
    }
}
