//! Conv / AdaIN building blocks (channel-major `[C, T]` activations, c*T + t).
//! Covers AdainResBlk1d used by the ProsodyPredictor F0/N stacks and the Decoder.
#![allow(dead_code)]

use super::ops::linear;
use super::KokoroModel;

/// General 1-D convolution, channel-major. `w [Cout, Cin/groups, K]`, `b [Cout]`.
/// Returns `(out [Cout, Tout], Tout)`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    inp: &[f32], cin: usize, t: usize, w: &[f32], b: Option<&[f32]>, cout: usize,
    k: usize, stride: usize, pad: usize, dilation: usize, groups: usize,
) -> (Vec<f32>, usize) {
    let tout = (t + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut out = vec![0.0f32; cout * tout];
    for g in 0..groups {
        for ocg in 0..cout_g {
            let co = g * cout_g + ocg;
            let wbase = co * cin_g * k;
            let bias = b.map_or(0.0, |bb| bb[co]);
            for to in 0..tout {
                let mut acc = bias;
                for icg in 0..cin_g {
                    let ci = g * cin_g + icg;
                    let wrow = &w[wbase + icg * k..wbase + (icg + 1) * k];
                    for kk in 0..k {
                        let ipos = (to * stride + kk * dilation) as isize - pad as isize;
                        if ipos >= 0 && (ipos as usize) < t {
                            acc += wrow[kk] * inp[ci * t + ipos as usize];
                        }
                    }
                }
                out[co * tout + to] = acc;
            }
        }
    }
    (out, tout)
}

/// Depthwise ConvTranspose1d (groups == channels), `w [C, 1, K]`, `b [C]`.
/// Used as StyleTTS2's upsampling "pool": stride=2, pad=1, output_padding=1, K=3 → 2×T.
pub fn conv_transpose1d_depthwise(
    inp: &[f32], c: usize, t: usize, w: &[f32], b: Option<&[f32]>,
    k: usize, stride: usize, pad: usize, output_padding: usize,
) -> (Vec<f32>, usize) {
    let tout = (t - 1) * stride + (k - 1) + output_padding + 1 - 2 * pad;
    let mut out = vec![0.0f32; c * tout];
    for ch in 0..c {
        let wrow = &w[ch * k..(ch + 1) * k];
        for i in 0..t {
            let v = inp[ch * t + i];
            for kk in 0..k {
                let opos = i as isize * stride as isize + kk as isize - pad as isize;
                if opos >= 0 && (opos as usize) < tout {
                    out[ch * tout + opos as usize] += v * wrow[kk];
                }
            }
        }
        if let Some(bb) = b {
            for to in 0..tout {
                out[ch * tout + to] += bb[ch];
            }
        }
    }
    (out, tout)
}

/// Nearest-neighbour ×2 upsample along time, channel-major.
pub fn upsample_nearest_2x(inp: &[f32], c: usize, t: usize) -> Vec<f32> {
    let tout = t * 2;
    let mut out = vec![0.0f32; c * tout];
    for ch in 0..c {
        for to in 0..tout {
            out[ch * tout + to] = inp[ch * t + to / 2];
        }
    }
    out
}

/// InstanceNorm1d (per-channel over time, affine) then AdaIN style modulation:
/// `(1+gamma_c) * (norm_affine(x)) + beta_c`, gamma/beta = chunk(fc(style)).
#[allow(clippy::too_many_arguments)]
pub fn adain1d(
    x: &[f32], c: usize, t: usize, norm_w: Option<&[f32]>, norm_b: Option<&[f32]>,
    fc_w: &[f32], fc_b: &[f32], style: &[f32], style_dim: usize,
) -> Vec<f32> {
    let gb = linear(style, 1, style_dim, fc_w, Some(fc_b), 2 * c); // [2C]
    let (gamma, beta) = gb.split_at(c);
    let mut out = vec![0.0f32; c * t];
    for ch in 0..c {
        let nw = norm_w.map_or(1.0, |w| w[ch]);
        let nb = norm_b.map_or(0.0, |b| b[ch]);
        let row = &x[ch * t..(ch + 1) * t];
        let mean = row.iter().sum::<f32>() / t as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / t as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ti in 0..t {
            let n = (row[ti] - mean) * inv * nw + nb;
            out[ch * t + ti] = (1.0 + gamma[ch]) * n + beta[ch];
        }
    }
    out
}

const RSQRT2: f32 = 0.707_106_77;

impl KokoroModel {
    /// AdainResBlk1d (modules.AdainResBlk1d / istftnet variant), LeakyReLU(0.2) activation.
    /// `x [dim_in, T]` → `(out [dim_out, Tout], Tout)`. `upsample` doubles T via the pool.
    pub fn adain_resblk1d(
        &self, prefix: &str, x: &[f32], dim_in: usize, t: usize, dim_out: usize,
        upsample: bool, style: &[f32],
    ) -> (Vec<f32>, usize) {
        let sd = self.cfg.style_dim;
        let learned_sc = dim_in != dim_out;

        // ---- residual ----
        let n1w = self.t_opt(&format!("{prefix}.norm1.norm.weight"));
        let n1b = self.t_opt(&format!("{prefix}.norm1.norm.bias"));
        let n1fw = self.t(&format!("{prefix}.norm1.fc.weight"));
        let n1fb = self.t(&format!("{prefix}.norm1.fc.bias"));
        let mut h = adain1d(x, dim_in, t, n1w.as_deref(), n1b.as_deref(), &n1fw, &n1fb, style, sd);
        super::ops::leaky_relu(&mut h, 0.2);

        // pool: identity, or depthwise ConvTranspose (2×T) on dim_in channels
        let (h, t_pool) = if upsample {
            let pw = self.t(&format!("{prefix}.pool.weight"));
            let pb = self.t(&format!("{prefix}.pool.bias"));
            conv_transpose1d_depthwise(&h, dim_in, t, &pw, Some(&pb), 3, 2, 1, 1)
        } else {
            (h, t)
        };

        let c1w = self.t(&format!("{prefix}.conv1.weight"));
        let c1b = self.t(&format!("{prefix}.conv1.bias"));
        let (h, t1) = conv1d(&h, dim_in, t_pool, &c1w, Some(&c1b), dim_out, 3, 1, 1, 1, 1);
        let n2w = self.t_opt(&format!("{prefix}.norm2.norm.weight"));
        let n2b = self.t_opt(&format!("{prefix}.norm2.norm.bias"));
        let n2fw = self.t(&format!("{prefix}.norm2.fc.weight"));
        let n2fb = self.t(&format!("{prefix}.norm2.fc.bias"));
        let mut h = adain1d(&h, dim_out, t1, n2w.as_deref(), n2b.as_deref(), &n2fw, &n2fb, style, sd);
        super::ops::leaky_relu(&mut h, 0.2);
        let c2w = self.t(&format!("{prefix}.conv2.weight"));
        let c2b = self.t(&format!("{prefix}.conv2.bias"));
        let (residual, tout) = conv1d(&h, dim_out, t1, &c2w, Some(&c2b), dim_out, 3, 1, 1, 1, 1);

        // ---- shortcut ----
        let sc = if upsample { upsample_nearest_2x(x, dim_in, t) } else { x.to_vec() };
        let sc = if learned_sc {
            let cw = self.t(&format!("{prefix}.conv1x1.weight"));
            conv1d(&sc, dim_in, tout, &cw, None, dim_out, 1, 1, 0, 1, 1).0
        } else {
            sc
        };

        let mut out = vec![0.0f32; dim_out * tout];
        for i in 0..dim_out * tout {
            out[i] = (residual[i] + sc[i]) * RSQRT2;
        }
        (out, tout)
    }
}
