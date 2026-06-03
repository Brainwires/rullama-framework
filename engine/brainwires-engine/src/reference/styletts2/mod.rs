//! StyleTTS2 zero-shot voice cloning — CPU f32 oracle.
//!
//! This is the capability Kokoro dropped: StyleTTS2's two `StyleEncoder`s turn a
//! reference clip's mel into a 256-d style vector (acoustic 128 ‖ prosodic 128) that
//! makes the decoder speak in that voice — no per-speaker training. 1:1 port of
//! `yl4579/StyleTTS2@main models.py` (StyleEncoder/ResBlk, lines 27-164). spectral_norm
//! is folded offline by the converter, so this oracle sees plain conv weights.
//!
//! Parity oracle = the PyTorch model (see `scripts/styletts2_dump_style_fixtures.py`),
//! NOT Kokoro — StyleTTS2 is a different checkpoint with a hifigan (not istftnet) decoder.
#![allow(dead_code)]

use crate::reference::kokoro::ops::leaky_relu;

pub mod style_encoder;
pub use style_encoder::StyleEncoder;

/// A 2D feature map, NCHW with N=1: `data[c*h*w + y*w + x]`.
#[derive(Clone)]
pub struct Map {
    pub data: Vec<f32>,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Map {
    pub fn new(data: Vec<f32>, c: usize, h: usize, w: usize) -> Self {
        debug_assert_eq!(data.len(), c * h * w);
        Self { data, c, h, w }
    }
}

/// General 2D convolution (NCHW, N=1) with stride/pad/groups. PyTorch weight layout
/// `[out_c, in_c/groups, kh, kw]`. Naive — this is a correctness oracle.
pub fn conv2d(
    x: &Map, w: &[f32], b: Option<&[f32]>, out_c: usize, kh: usize, kw: usize, stride: usize, pad: usize, groups: usize,
) -> Map {
    let (ic, h, win) = (x.c, x.h, x.w);
    let ho = (h + 2 * pad - kh) / stride + 1;
    let wo = (win + 2 * pad - kw) / stride + 1;
    let icpg = ic / groups;
    let ocpg = out_c / groups;
    let mut out = vec![0f32; out_c * ho * wo];
    for oc in 0..out_c {
        let g = oc / ocpg;
        let bias = b.map_or(0.0, |bb| bb[oc]);
        for oy in 0..ho {
            for ox in 0..wo {
                let mut acc = bias;
                for icj in 0..icpg {
                    let in_c = g * icpg + icj;
                    let wbase = (oc * icpg + icj) * kh * kw;
                    let xbase = in_c * h * win;
                    for ky in 0..kh {
                        let iy = oy * stride + ky;
                        if iy < pad || iy >= h + pad {
                            continue;
                        }
                        let iy = iy - pad;
                        for kx in 0..kw {
                            let ix = ox * stride + kx;
                            if ix < pad || ix >= win + pad {
                                continue;
                            }
                            let ix = ix - pad;
                            acc += x.data[xbase + iy * win + ix] * w[wbase + ky * kw + kx];
                        }
                    }
                }
                out[oc * ho * wo + oy * wo + ox] = acc;
            }
        }
    }
    Map::new(out, out_c, ho, wo)
}

/// `DownSample('half')`: if the width is odd, repeat the last column (StyleTTS2's
/// `torch.cat([x, x[...,-1:]])`), then 2×2 stride-2 average pool. Halves H and W.
pub fn avg_pool2d_half(x: &Map) -> Map {
    // pad last column when width is odd (height is always even in this pipeline)
    let (padded, w) = if x.w % 2 != 0 {
        let mut p = vec![0f32; x.c * x.h * (x.w + 1)];
        for c in 0..x.c {
            for y in 0..x.h {
                let src = c * x.h * x.w + y * x.w;
                let dst = c * x.h * (x.w + 1) + y * (x.w + 1);
                p[dst..dst + x.w].copy_from_slice(&x.data[src..src + x.w]);
                p[dst + x.w] = x.data[src + x.w - 1]; // repeat last col
            }
        }
        (p, x.w + 1)
    } else {
        (x.data.clone(), x.w)
    };
    let ho = x.h / 2;
    let wo = w / 2;
    let mut out = vec![0f32; x.c * ho * wo];
    for c in 0..x.c {
        for oy in 0..ho {
            for ox in 0..wo {
                let base = c * x.h * w;
                let s = padded[base + (2 * oy) * w + 2 * ox]
                    + padded[base + (2 * oy) * w + 2 * ox + 1]
                    + padded[base + (2 * oy + 1) * w + 2 * ox]
                    + padded[base + (2 * oy + 1) * w + 2 * ox + 1];
                out[c * ho * wo + oy * wo + ox] = s * 0.25;
            }
        }
    }
    Map::new(out, x.c, ho, wo)
}

/// `AdaptiveAvgPool2d(1)`: mean over H×W per channel → `[c]`.
pub fn adaptive_avg_pool2d_1(x: &Map) -> Vec<f32> {
    let hw = (x.h * x.w) as f32;
    (0..x.c).map(|c| x.data[c * x.h * x.w..(c + 1) * x.h * x.w].iter().sum::<f32>() / hw).collect()
}

const LRELU: f32 = 0.2;

/// One spectral-norm conv with bias, folded — just dims + weights.
pub struct Conv {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub oc: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
    pub groups: usize,
}

impl Conv {
    pub fn apply(&self, x: &Map) -> Map {
        conv2d(x, &self.w, self.b.as_deref(), self.oc, self.kh, self.kw, self.stride, self.pad, self.groups)
    }
}

/// StyleTTS2 `ResBlk(downsample='half', normalize=False)`. Shortcut: optional 1×1 conv
/// (when channels change) then avg-pool; residual: leaky→conv1(k3)→strided depthwise
/// down→leaky→conv2(k3); `(shortcut+residual)/√2`.
pub struct ResBlk {
    pub conv1: Conv,           // dim_in → dim_in, k3 p1
    pub down: Conv,            // depthwise dim_in → dim_in, k3 s2 p1
    pub conv2: Conv,           // dim_in → dim_out, k3 p1
    pub sc: Option<Conv>,      // 1×1 dim_in → dim_out (only when dim_in != dim_out)
}

impl ResBlk {
    pub fn forward(&self, x: &Map) -> Map {
        // shortcut
        let sc = match &self.sc {
            Some(c) => c.apply(x),
            None => x.clone(),
        };
        let shortcut = avg_pool2d_half(&sc);
        // residual
        let mut r = x.clone();
        leaky_relu(&mut r.data, LRELU);
        let r = self.conv1.apply(&r);
        let mut r = self.down.apply(&r);
        leaky_relu(&mut r.data, LRELU);
        let r = self.conv2.apply(&r);
        // (shortcut + residual) / sqrt(2)
        debug_assert_eq!(shortcut.data.len(), r.data.len());
        let inv = 1.0 / std::f32::consts::SQRT_2;
        let data: Vec<f32> = shortcut.data.iter().zip(&r.data).map(|(a, b)| (a + b) * inv).collect();
        Map::new(data, r.c, r.h, r.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_pool_odd_width_repeats_last_col() {
        // width 3 (odd) → pad to 4 → pool to 2; height 2 → 1
        let x = Map::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 2, 3);
        let p = avg_pool2d_half(&x);
        assert_eq!((p.c, p.h, p.w), (1, 1, 2));
        // col0 = mean(1,2,4,5)=3 ; col1 = mean(3,3,6,6)=4.5 (last col repeated)
        assert!((p.data[0] - 3.0).abs() < 1e-6);
        assert!((p.data[1] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn conv2d_identity_1x1() {
        let x = Map::new(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2);
        let w = vec![2.0]; // 1->1, k1, scale by 2
        let y = conv2d(&x, &w, None, 1, 1, 1, 1, 0, 1);
        assert_eq!(y.data, vec![2.0, 4.0, 6.0, 8.0]);
    }
}
