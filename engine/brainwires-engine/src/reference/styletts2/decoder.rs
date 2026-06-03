//! StyleTTS2-LibriTTS hifigan Decoder — CPU f32 oracle.
//!
//! Shape-identical to Kokoro's istftnet Decoder for the cat-stack (encode + 4 decode
//! AdainResBlk1d), but the Generator is the net-new **hifigan** path: 4 upsamples
//! [10,5,3,2], a per-stage Snake on the trunk (`generator.alphas`), a single-channel
//! HnNSF source fed straight in (no STFT), and a direct-waveform `conv_post` + tanh.
//!
//! Ported piece-by-piece against the isolation fixtures from
//! `scripts/styletts2_dump_decoder_fixtures.py` (encode/decode0-3/har_source/conv_post/audio).
#![allow(dead_code)]

use std::collections::HashMap;
use std::f32::consts::PI;

use crate::reference::kokoro::convblocks::{adain1d, conv1d, conv_transpose1d, conv_transpose1d_depthwise, snake, upsample_nearest_2x};
use crate::reference::kokoro::ops::leaky_relu;

const SR: f32 = 24000.0;
const STYLE_DIM: usize = 128;
const RSQRT2: f32 = 0.707_106_77;
const SINE_AMP: f32 = 0.1;
const VOICED_THRESHOLD: f32 = 10.0;

/// `F.interpolate(mode="linear", align_corners=False)` for a 1-D signal.
pub(crate) fn interp_linear(x: &[f32], in_len: usize, out_len: usize) -> Vec<f32> {
    let scale = in_len as f32 / out_len as f32;
    let mut out = vec![0f32; out_len];
    for o in 0..out_len {
        let src = (o as f32 + 0.5) * scale - 0.5;
        if src <= 0.0 {
            out[o] = x[0];
        } else {
            let i0 = src.floor() as usize;
            let frac = src - i0 as f32;
            let i1 = (i0 + 1).min(in_len - 1);
            out[o] = x[i0.min(in_len - 1)] * (1.0 - frac) + x[i1] * frac;
        }
    }
    out
}

/// hifigan HnNSF excitation: `f0_curve[frames]` → `har_source[frames*up]`, deterministic
/// (random phase + noise zeroed, matching the fixtures). `up = prod(upsample_rates)`.
/// Algorithm mirrors StyleTTS2's SineGen: nearest-upsample f0 ×up, take `(f0·h/SR) mod 1`,
/// downsample to `frames`, cumulative phase ×2π·up, upsample back, sin, merge via
/// `l_linear` (+tanh). No STFT (unlike Kokoro's istftnet source).
pub fn source_signal(f0_curve: &[f32], up: usize, harmonics: usize, l_w: &[f32], l_b: f32) -> Vec<f32> {
    let frames_in = f0_curve.len();
    let l = frames_in * up;
    let f0_up: Vec<f32> = (0..l).map(|i| f0_curve[i / up]).collect(); // nearest upsample
    let uv: Vec<f32> = f0_up.iter().map(|&v| if v > VOICED_THRESHOLD { 1.0 } else { 0.0 }).collect();

    let mut har = vec![0f32; l];
    for h in 0..harmonics {
        let mult = (h + 1) as f32;
        let rad: Vec<f32> = f0_up.iter().map(|&f| (f * mult / SR).rem_euclid(1.0)).collect();
        let rad_ds = interp_linear(&rad, l, frames_in);
        let mut cum = vec![0f32; frames_in];
        let mut acc = 0.0f32;
        for i in 0..frames_in {
            acc += rad_ds[i];
            cum[i] = acc * 2.0 * PI * up as f32;
        }
        let phase = interp_linear(&cum, frames_in, l);
        let w = l_w[h];
        for i in 0..l {
            har[i] += phase[i].sin() * SINE_AMP * uv[i] * w;
        }
    }
    for v in har.iter_mut() {
        *v = (*v + l_b).tanh();
    }
    har
}

/// Concatenate channel-major `[C_i, T]` tensors along the channel axis → `[ΣC_i, T]`.
fn cat_channels(parts: &[(&[f32], usize)], t: usize) -> Vec<f32> {
    let ctot: usize = parts.iter().map(|(_, c)| *c).sum();
    let mut out = vec![0f32; ctot * t];
    let mut base = 0;
    for (data, c) in parts {
        out[base * t..(base + c) * t].copy_from_slice(&data[..c * t]);
        base += c;
    }
    out
}

/// The hifigan Decoder: holds the folded weights (PyTorch state-dict names) and runs
/// `(asr, F0_curve, N, style) → 24 kHz waveform`. AdaIN has `affine=False`, so the
/// InstanceNorm has no learned weight/bias (t_opt returns None → pure instance norm).
pub struct StyleTtsDecoder<'a> {
    w: &'a HashMap<String, Vec<f32>>,
}

impl<'a> StyleTtsDecoder<'a> {
    pub fn new(w: &'a HashMap<String, Vec<f32>>) -> Self {
        Self { w }
    }
    fn t(&self, n: &str) -> &[f32] {
        self.w.get(n).unwrap_or_else(|| panic!("missing decoder weight: {n}"))
    }
    fn t_opt(&self, n: &str) -> Option<&[f32]> {
        self.w.get(n).map(|v| v.as_slice())
    }

    /// AdainResBlk1d (LeakyReLU 0.2). `upsample` doubles T via the depthwise pool.
    /// `pub(crate)` so the predictor's F0/N stacks reuse it.
    pub(crate) fn adain_resblk1d(&self, p: &str, x: &[f32], dim_in: usize, t: usize, dim_out: usize, upsample: bool, s: &[f32]) -> (Vec<f32>, usize) {
        let learned_sc = dim_in != dim_out;
        let mut h = adain1d(x, dim_in, t, self.t_opt(&format!("{p}.norm1.norm.weight")), self.t_opt(&format!("{p}.norm1.norm.bias")),
            self.t(&format!("{p}.norm1.fc.weight")), self.t(&format!("{p}.norm1.fc.bias")), s, STYLE_DIM);
        leaky_relu(&mut h, 0.2);
        let (h, tp) = if upsample {
            conv_transpose1d_depthwise(&h, dim_in, t, self.t(&format!("{p}.pool.weight")), Some(self.t(&format!("{p}.pool.bias"))), 3, 2, 1, 1)
        } else {
            (h, t)
        };
        let (h, t1) = conv1d(&h, dim_in, tp, self.t(&format!("{p}.conv1.weight")), Some(self.t(&format!("{p}.conv1.bias"))), dim_out, 3, 1, 1, 1, 1);
        let mut h = adain1d(&h, dim_out, t1, self.t_opt(&format!("{p}.norm2.norm.weight")), self.t_opt(&format!("{p}.norm2.norm.bias")),
            self.t(&format!("{p}.norm2.fc.weight")), self.t(&format!("{p}.norm2.fc.bias")), s, STYLE_DIM);
        leaky_relu(&mut h, 0.2);
        let (residual, tout) = conv1d(&h, dim_out, t1, self.t(&format!("{p}.conv2.weight")), Some(self.t(&format!("{p}.conv2.bias"))), dim_out, 3, 1, 1, 1, 1);
        let sc = if upsample { upsample_nearest_2x(x, dim_in, t) } else { x.to_vec() };
        let sc = if learned_sc { conv1d(&sc, dim_in, tout, self.t(&format!("{p}.conv1x1.weight")), None, dim_out, 1, 1, 0, 1, 1).0 } else { sc };
        let out: Vec<f32> = residual.iter().zip(&sc).map(|(r, c)| (r + c) * RSQRT2).collect();
        (out, tout)
    }

    /// AdaINResBlock1 (MRF block): 3 dilated conv pairs, Snake activation, AdaIN before each.
    fn adain_resblock1(&self, p: &str, x: &[f32], c: usize, t: usize, k: usize, dil: &[usize], s: &[f32]) -> Vec<f32> {
        let mut x = x.to_vec();
        for j in 0..3 {
            let mut xt = adain1d(&x, c, t, self.t_opt(&format!("{p}.adain1.{j}.norm.weight")), self.t_opt(&format!("{p}.adain1.{j}.norm.bias")),
                self.t(&format!("{p}.adain1.{j}.fc.weight")), self.t(&format!("{p}.adain1.{j}.fc.bias")), s, STYLE_DIM);
            snake(&mut xt, c, t, self.t(&format!("{p}.alpha1.{j}")));
            let (xt, _) = conv1d(&xt, c, t, self.t(&format!("{p}.convs1.{j}.weight")), Some(self.t(&format!("{p}.convs1.{j}.bias"))), c, k, 1, (k * dil[j] - dil[j]) / 2, dil[j], 1);
            let mut xt = adain1d(&xt, c, t, self.t_opt(&format!("{p}.adain2.{j}.norm.weight")), self.t_opt(&format!("{p}.adain2.{j}.norm.bias")),
                self.t(&format!("{p}.adain2.{j}.fc.weight")), self.t(&format!("{p}.adain2.{j}.fc.bias")), s, STYLE_DIM);
            snake(&mut xt, c, t, self.t(&format!("{p}.alpha2.{j}")));
            let (xt, _) = conv1d(&xt, c, t, self.t(&format!("{p}.convs2.{j}.weight")), Some(self.t(&format!("{p}.convs2.{j}.bias"))), c, k, 1, (k - 1) / 2, 1, 1);
            for i in 0..c * t {
                x[i] += xt[i];
            }
        }
        x
    }

    /// hifigan Generator: per-stage Snake (generator.alphas), 4 upsamples + single-channel
    /// HnNSF source branch + MRF, then conv_post + tanh → waveform.
    fn generator(&self, x: &[f32], xt: usize, har: &[f32], s: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        const RATES: [usize; 4] = [10, 5, 3, 2];
        const KERNELS: [usize; 4] = [20, 10, 6, 4];
        const RK: [usize; 3] = [3, 7, 11];
        let rdil = [[1usize, 3, 5]; 3];
        let har_len = har.len();
        let mut cur = x.to_vec();
        let (mut cin, mut tcur) = (512usize, xt);
        for i in 0..4 {
            if let Some(p) = progress {
                p(0.45 + 0.45 * i as f32 / 4.0, "generating audio");
            }
            snake(&mut cur, cin, tcur, self.t(&format!("generator.alphas.{i}")));
            let cout = cin / 2;
            let ncw = self.t(&format!("generator.noise_convs.{i}.weight"));
            let ncb = self.t(&format!("generator.noise_convs.{i}.bias"));
            let (xsrc, xsrc_t, nres_k) = if i + 1 < 4 {
                let stride_f0: usize = RATES[i + 1..].iter().product();
                let (xs, ts) = conv1d(har, 1, har_len, ncw, Some(ncb), cout, stride_f0 * 2, stride_f0, (stride_f0 + 1) / 2, 1, 1);
                (xs, ts, 7usize)
            } else {
                let (xs, ts) = conv1d(har, 1, har_len, ncw, Some(ncb), cout, 1, 1, 0, 1, 1);
                (xs, ts, 11usize)
            };
            let xsrc = self.adain_resblock1(&format!("generator.noise_res.{i}"), &xsrc, cout, xsrc_t, nres_k, &[1, 3, 5], s);
            let u = RATES[i];
            let (mut up, tup) = conv_transpose1d(&cur, cin, tcur, self.t(&format!("generator.ups.{i}.weight")), Some(self.t(&format!("generator.ups.{i}.bias"))), cout, KERNELS[i], u, u / 2 + u % 2, u % 2);
            debug_assert_eq!(tup, xsrc_t, "stage {i}: up {tup} != src {xsrc_t}");
            for idx in 0..cout * tup {
                up[idx] += xsrc[idx];
            }
            let mut acc = vec![0f32; cout * tup];
            for (j, (&rk, rd)) in RK.iter().zip(rdil.iter()).enumerate() {
                let rb = self.adain_resblock1(&format!("generator.resblocks.{}", i * 3 + j), &up, cout, tup, rk, rd, s);
                for idx in 0..cout * tup {
                    acc[idx] += rb[idx];
                }
            }
            for v in acc.iter_mut() {
                *v /= 3.0;
            }
            cur = acc;
            cin = cout;
            tcur = tup;
        }
        snake(&mut cur, cin, tcur, self.t("generator.alphas.4"));
        let (post, _) = conv1d(&cur, cin, tcur, self.t("generator.conv_post.weight"), Some(self.t("generator.conv_post.bias")), 1, 7, 1, 3, 1, 1);
        post.iter().map(|v| v.tanh()).collect()
    }

    /// Full decoder: `asr [asr_c, asr_t]`, `f0_curve`/`n_curve [2·asr_t]`, `style [128]` → waveform.
    pub fn forward(&self, asr: &[f32], asr_c: usize, asr_t: usize, f0_curve: &[f32], n_curve: &[f32], s: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        if let Some(p) = progress {
            p(0.36, "building features");
        }
        let (f0d, _) = conv1d(f0_curve, 1, f0_curve.len(), self.t("F0_conv.weight"), Some(self.t("F0_conv.bias")), 1, 3, 2, 1, 1, 1);
        let (nd, _) = conv1d(n_curve, 1, n_curve.len(), self.t("N_conv.weight"), Some(self.t("N_conv.bias")), 1, 3, 2, 1, 1, 1);
        let t = f0d.len(); // == asr_t
        let cat0 = cat_channels(&[(asr, asr_c), (&f0d, 1), (&nd, 1)], t);
        let (mut x, mut tcur) = self.adain_resblk1d("encode", &cat0, asr_c + 2, t, 1024, false, s);
        let (asr_res, _) = conv1d(asr, asr_c, asr_t, self.t("asr_res.0.weight"), Some(self.t("asr_res.0.bias")), 64, 1, 1, 0, 1, 1);
        for i in 0..4 {
            let xin = cat_channels(&[(&x, x.len() / tcur), (&asr_res, 64), (&f0d, 1), (&nd, 1)], tcur);
            let (nx, nt) = self.adain_resblk1d(&format!("decode.{i}"), &xin, x.len() / tcur + 66, tcur, if i < 3 { 1024 } else { 512 }, i == 3, s);
            x = nx;
            tcur = nt;
        }
        let lw = self.t("generator.m_source.l_linear.weight");
        let lb = self.t("generator.m_source.l_linear.bias")[0];
        let har = source_signal(f0_curve, 300, 9, lw, lb);
        self.generator(&x, tcur, &har, s, progress)
    }
}
