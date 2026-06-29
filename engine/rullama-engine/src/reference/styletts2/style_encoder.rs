//! `StyleEncoder` (models.py:139-164): Conv2d(1→64) → 4× ResBlk(half) → LeakyReLU →
//! Conv2d(512→512,k5,no-pad) → AdaptiveAvgPool2d(1) → LeakyReLU → Linear(512→128).
//! Two identical instances exist (acoustic `style_encoder`, prosodic `predictor_encoder`);
//! their concat is the 256-d reference style vector.

use std::collections::HashMap;

use super::{Conv, LRELU, Map, ResBlk, adaptive_avg_pool2d_1};
use crate::reference::kokoro::ops::{leaky_relu, linear};

/// Per-block (dim_in, dim_out), channel doubling capped at max_conv_dim=512.
const BLK_DIMS: [(usize, usize); 4] = [(64, 128), (128, 256), (256, 512), (512, 512)];

pub struct StyleEncoder {
    conv0: Conv,       // 1 → 64, k3 p1
    blks: Vec<ResBlk>, // 4× half-downsample
    conv_out: Conv,    // 512 → 512, k5 p0
    lin_w: Vec<f32>,   // [128, 512]
    lin_b: Vec<f32>,   // [128]
}

type W = HashMap<String, Vec<f32>>;

fn take(w: &W, name: &str) -> Vec<f32> {
    w.get(name)
        .unwrap_or_else(|| panic!("missing style-encoder weight: {name}"))
        .clone()
}

impl StyleEncoder {
    /// Build from a folded-weight map (see `styletts2_dump_style_fixtures.py` naming),
    /// e.g. `prefix = "acoustic"` reads `acoustic.conv0.weight`, `acoustic.blk0.conv1.*`, …
    pub fn from_weights(w: &W, prefix: &str) -> Self {
        let p = |s: &str| format!("{prefix}.{s}");
        let conv0 = Conv {
            w: take(w, &p("conv0.weight")),
            b: Some(take(w, &p("conv0.bias"))),
            oc: 64,
            kh: 3,
            kw: 3,
            stride: 1,
            pad: 1,
            groups: 1,
        };
        let blks = (0..4)
            .map(|i| {
                let (din, dout) = BLK_DIMS[i];
                let b = |s: &str| p(&format!("blk{i}.{s}"));
                ResBlk {
                    conv1: Conv {
                        w: take(w, &b("conv1.weight")),
                        b: Some(take(w, &b("conv1.bias"))),
                        oc: din,
                        kh: 3,
                        kw: 3,
                        stride: 1,
                        pad: 1,
                        groups: 1,
                    },
                    down: Conv {
                        w: take(w, &b("down.weight")),
                        b: Some(take(w, &b("down.bias"))),
                        oc: din,
                        kh: 3,
                        kw: 3,
                        stride: 2,
                        pad: 1,
                        groups: din,
                    },
                    conv2: Conv {
                        w: take(w, &b("conv2.weight")),
                        b: Some(take(w, &b("conv2.bias"))),
                        oc: dout,
                        kh: 3,
                        kw: 3,
                        stride: 1,
                        pad: 1,
                        groups: 1,
                    },
                    sc: (din != dout).then(|| Conv {
                        w: take(w, &b("sc.weight")),
                        b: None,
                        oc: dout,
                        kh: 1,
                        kw: 1,
                        stride: 1,
                        pad: 0,
                        groups: 1,
                    }),
                }
            })
            .collect();
        let conv_out = Conv {
            w: take(w, &p("conv_out.weight")),
            b: Some(take(w, &p("conv_out.bias"))),
            oc: 512,
            kh: 5,
            kw: 5,
            stride: 1,
            pad: 0,
            groups: 1,
        };
        Self {
            conv0,
            blks,
            conv_out,
            lin_w: take(w, &p("linear.weight")),
            lin_b: take(w, &p("linear.bias")),
        }
    }

    /// Reference mel `[1, n_mels=80, T]` (flattened) → 128-d style vector.
    pub fn forward(&self, mel: &[f32], n_mels: usize, t: usize) -> Vec<f32> {
        let mut x = self.conv0.apply(&Map::new(mel.to_vec(), 1, n_mels, t));
        for blk in &self.blks {
            x = blk.forward(&x);
        }
        leaky_relu(&mut x.data, LRELU);
        let x = self.conv_out.apply(&x); // 512 → 512, collapses H (5→1)
        let mut pooled = adaptive_avg_pool2d_1(&x); // [512]
        leaky_relu(&mut pooled, LRELU);
        linear(&pooled, 1, 512, &self.lin_w, Some(&self.lin_b), 128)
    }
}
