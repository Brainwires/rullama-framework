//! ISTFTNet Generator (istftnet.Generator) + exact iSTFT.
//!
//! The harmonic source (`har`) is injected (non-deterministic upstream — random
//! phase + noise in SineGen). Everything downstream is deterministic. AdaINResBlock1
//! uses the Snake activation. Final iSTFT is the exact complex-equivalent path
//! (one-sided-bin conjugate completion + COLA window normalization), not Kokoro's
//! ONNX CustomSTFT approximation (see KOKORO_REFERENCE.md).
#![allow(dead_code)]

use std::f32::consts::PI;

use super::convblocks::{conv1d, conv_transpose1d, reflection_pad_left1, snake};
use super::ops::leaky_relu;
use super::KokoroModel;

impl KokoroModel {
    /// AdaINResBlock1 (3 dilated conv pairs, Snake activation, AdaIN before each).
    /// `x [C, T]` → `[C, T]`. `kernel`/`dilations` per the resblock config.
    fn adain_resblock1(&self, prefix: &str, x: &[f32], c: usize, t: usize, k: usize, dil: &[usize], style: &[f32]) -> Vec<f32> {
        let sd = self.cfg.style_dim;
        let mut x = x.to_vec();
        for j in 0..3 {
            let n1w = self.t_opt(&format!("{prefix}.adain1.{j}.norm.weight"));
            let n1b = self.t_opt(&format!("{prefix}.adain1.{j}.norm.bias"));
            let n1fw = self.t(&format!("{prefix}.adain1.{j}.fc.weight"));
            let n1fb = self.t(&format!("{prefix}.adain1.{j}.fc.bias"));
            let mut xt = super::convblocks::adain1d(&x, c, t, n1w.as_deref(), n1b.as_deref(), &n1fw, &n1fb, style, sd);
            let a1 = self.t(&format!("{prefix}.alpha1.{j}"));
            snake(&mut xt, c, t, &a1);
            // convs1[j]: dilation dil[j], pad = (k*dil - dil)/2
            let c1w = self.t(&format!("{prefix}.convs1.{j}.weight"));
            let c1b = self.t(&format!("{prefix}.convs1.{j}.bias"));
            let pad1 = (k * dil[j] - dil[j]) / 2;
            let (xt, _) = conv1d(&xt, c, t, &c1w, Some(&c1b), c, k, 1, pad1, dil[j], 1);

            let n2w = self.t_opt(&format!("{prefix}.adain2.{j}.norm.weight"));
            let n2b = self.t_opt(&format!("{prefix}.adain2.{j}.norm.bias"));
            let n2fw = self.t(&format!("{prefix}.adain2.{j}.fc.weight"));
            let n2fb = self.t(&format!("{prefix}.adain2.{j}.fc.bias"));
            let mut xt = super::convblocks::adain1d(&xt, c, t, n2w.as_deref(), n2b.as_deref(), &n2fw, &n2fb, style, sd);
            let a2 = self.t(&format!("{prefix}.alpha2.{j}"));
            snake(&mut xt, c, t, &a2);
            // convs2[j]: dilation 1, pad = (k-1)/2
            let c2w = self.t(&format!("{prefix}.convs2.{j}.weight"));
            let c2b = self.t(&format!("{prefix}.convs2.{j}.bias"));
            let (xt, _) = conv1d(&xt, c, t, &c2w, Some(&c2b), c, k, 1, (k - 1) / 2, 1, 1);

            for i in 0..c * t {
                x[i] += xt[i];
            }
        }
        x
    }

    /// ISTFTNet generator. `x [512, Tx]` (Tx=156), injected `har [22, Th]`,
    /// timbre `style [128]`. Returns the 24 kHz waveform.
    pub fn generator(&self, x: &[f32], xt_len: usize, har: &[f32], har_len: usize, style: &[f32]) -> Vec<f32> {
        let rates = &self.cfg.upsample_rates; // [10, 6]
        let rkernels = self.cfg.resblock_kernel_sizes.clone(); // [3,7,11]
        let rdil = self.cfg.resblock_dilation_sizes.clone(); // [[1,3,5]x3]
        let nfft = self.cfg.gen_istft_n_fft; // 20
        let nbins = nfft / 2 + 1; // 11

        let mut cur = x.to_vec();
        let mut cin = self.cfg.upsample_initial_channel; // 512
        let mut tcur = xt_len;

        for i in 0..rates.len() {
            leaky_relu(&mut cur, 0.1);

            // noise branch: noise_convs[i](har) then noise_res[i]
            let ncw = self.t(&format!("k.decoder.generator.noise_convs.{i}.weight"));
            let ncb = self.t(&format!("k.decoder.generator.noise_convs.{i}.bias"));
            let cout = cin / 2;
            let (xsrc, _, nres_k) = if i + 1 < rates.len() {
                let stride_f0: usize = rates[i + 1..].iter().product();
                let (xs, ts) = conv1d(har, nfft + 2, har_len, &ncw, Some(&ncb), cout, stride_f0 * 2, stride_f0, (stride_f0 + 1) / 2, 1, 1);
                (xs, ts, 7usize)
            } else {
                let (xs, ts) = conv1d(har, nfft + 2, har_len, &ncw, Some(&ncb), cout, 1, 1, 0, 1, 1);
                (xs, ts, 11usize)
            };
            let xsrc_t = xsrc.len() / cout;
            let xsrc = self.adain_resblock1(&format!("k.decoder.generator.noise_res.{i}"), &xsrc, cout, xsrc_t, nres_k, &[1, 3, 5], style);

            // upsample x
            let uw = self.t(&format!("k.decoder.generator.ups.{i}.weight"));
            let ub = self.t(&format!("k.decoder.generator.ups.{i}.bias"));
            let k = self.cfg.upsample_kernel_sizes[i];
            let (mut up, mut tup) = conv_transpose1d(&cur, cin, tcur, &uw, Some(&ub), cout, k, rates[i], (k - rates[i]) / 2, 0);
            if i == rates.len() - 1 {
                up = reflection_pad_left1(&up, cout, tup);
                tup += 1;
            }
            debug_assert_eq!(tup, xsrc_t, "source/upsample length mismatch at stage {i}");
            for idx in 0..cout * tup {
                up[idx] += xsrc[idx];
            }

            // 3 resblocks summed / num_kernels
            let mut acc = vec![0.0f32; cout * tup];
            for (j, (&rk, rd)) in rkernels.iter().zip(rdil.iter()).enumerate() {
                let rb = self.adain_resblock1(&format!("k.decoder.generator.resblocks.{}", i * rkernels.len() + j), &up, cout, tup, rk, rd, style);
                for idx in 0..cout * tup {
                    acc[idx] += rb[idx];
                }
            }
            for v in acc.iter_mut() {
                *v /= rkernels.len() as f32;
            }
            cur = acc;
            cin = cout;
            tcur = tup;
        }

        leaky_relu(&mut cur, 0.01);
        let cpw = self.t("k.decoder.generator.conv_post.weight");
        let cpb = self.t("k.decoder.generator.conv_post.bias");
        let (post, tpost) = conv1d(&cur, cin, tcur, &cpw, Some(&cpb), nfft + 2, 7, 1, 3, 1, 1); // [22, T]

        // spec = exp(post[:11]), phase = sin(post[11:22])
        let mut spec = vec![0.0f32; nbins * tpost];
        let mut phase = vec![0.0f32; nbins * tpost];
        for b in 0..nbins {
            for ti in 0..tpost {
                spec[b * tpost + ti] = post[b * tpost + ti].exp();
                phase[b * tpost + ti] = post[(b + nbins) * tpost + ti].sin();
            }
        }
        istft(&spec, &phase, nbins, tpost, nfft, self.cfg.gen_istft_hop)
    }
}

/// Exact iSTFT (onesided, center, COLA-normalized) — matches torch.istft.
/// `spec`/`phase` are `[nbins, F]` channel-major (magnitude, angle).
fn istft(spec: &[f32], phase: &[f32], nbins: usize, frames: usize, nfft: usize, hop: usize) -> Vec<f32> {
    // Hann window, periodic
    let win: Vec<f32> = (0..nfft).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / nfft as f32).cos()).collect();
    // DFT tables
    let mut cos_t = vec![0.0f32; nfft * nfft];
    let mut sin_t = vec![0.0f32; nfft * nfft];
    for k in 0..nfft {
        for n in 0..nfft {
            let ang = 2.0 * PI * (k * n) as f32 / nfft as f32;
            cos_t[k * nfft + n] = ang.cos();
            sin_t[k * nfft + n] = ang.sin();
        }
    }
    let ola_len = (frames - 1) * hop + nfft;
    let mut y = vec![0.0f32; ola_len];
    let mut env = vec![0.0f32; ola_len];
    let mut re = vec![0.0f32; nfft];
    let mut im = vec![0.0f32; nfft];
    for f in 0..frames {
        // full complex spectrum via conjugate symmetry
        for k in 0..nbins {
            let m = spec[k * frames + f];
            let p = phase[k * frames + f];
            re[k] = m * p.cos();
            im[k] = m * p.sin();
        }
        for k in nbins..nfft {
            re[k] = re[nfft - k];
            im[k] = -im[nfft - k];
        }
        // ifft → real time frame, window, overlap-add
        for n in 0..nfft {
            let mut acc = 0.0f32;
            for k in 0..nfft {
                acc += re[k] * cos_t[k * nfft + n] - im[k] * sin_t[k * nfft + n];
            }
            let tn = acc / nfft as f32;
            let pos = f * hop + n;
            y[pos] += tn * win[n];
            env[pos] += win[n] * win[n];
        }
    }
    for i in 0..ola_len {
        if env[i] > 1e-11 {
            y[i] /= env[i];
        }
    }
    // remove center padding (nfft/2 each side)
    let pad = nfft / 2;
    y[pad..ola_len - pad].to_vec()
}
