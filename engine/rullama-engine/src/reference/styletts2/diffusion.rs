//! StyleTTS2 style-diffusion sampler (CPU f32 oracle).
//!
//! Restores natural prosody (the `alpha=0.3/beta=0.7` path) by sampling a style vector
//! `s_pred [256]` from the StyleTransformer1d denoiser, conditioned on the PLBERT output
//! `bert_dur [L,768]` and the reference style `ref_s [256]`. Mirrors, 1:1:
//!   Modules/diffusion/modules.py  (StyleTransformer1d denoiser)
//!   Modules/diffusion/sampler.py  (KDiffusion denoise_fn + ADPM2Sampler + KarrasSchedule)
//!
//! Network (channels=256, ctx_emb=768 ⇒ block features F=1024, style_dim S=256, 8 heads × 64):
//!   run(x[256], t, emb[L,768], feat[256]):
//!     mapping = to_mapping( GELU(Lin(time_pos(t))) + GELU(Lin(feat)) )         [1024]
//!     h = [x.broadcast(L) ‖ emb]                                               [L,1024]
//!     for blk in 3: h += mapping; h = blk(h, feat)
//!     return Conv1x1_1024→256( mean_t h )                                      [256]
//!
//! The denoiser uses exact (erf) GELU — NOT the tanh `gelu_new` the ALBERT path uses.

use crate::reference::kokoro::ops::{layer_norm_plain, linear, softmax};
use std::collections::HashMap;

const C: usize = 256; // style channels (style_dim*2)
const E: usize = 768; // PLBERT embedding features
const F: usize = C + E; // block feature width = 1024
const S: usize = 256; // AdaLayerNorm style_dim (context_features)
const HEADS: usize = 8;
const HD: usize = 64; // head_features
const MID: usize = HEADS * HD; // 512
const NBLK: usize = 3;
const EPS: f32 = 1e-5;

/// Exact GELU: x·½(1+erf(x/√2)). erf via Abramowitz-Stegun 7.1.26 (~1e-7, ample for 2e-3 tol).
fn gelu_exact(v: &mut [f32]) {
    for x in v.iter_mut() {
        let z = *x / std::f32::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
        let y = 1.0
            - (((((1.061_405_4 * t - 1.453_152_) * t) + 1.421_413_7) * t - 0.284_496_74) * t
                + 0.254_829_6)
                * t
                * (-z * z).exp();
        let erf = if z >= 0.0 { y } else { -y };
        *x *= 0.5 * (1.0 + erf);
    }
}

pub struct StyleDiffusion<'a> {
    w: &'a HashMap<String, Vec<f32>>,
    sigma_data: f32,
    sigma_min: f32,
    sigma_max: f32,
    rho: f32,
    steps: usize,
}

impl<'a> StyleDiffusion<'a> {
    pub fn new(w: &'a HashMap<String, Vec<f32>>) -> Self {
        // params come from the GGUF KV (baked by the converter); fall back to the LibriTTS demo.
        let f = |k: &str, d: f32| w.get(k).and_then(|v| v.first().copied()).unwrap_or(d);
        Self {
            w,
            sigma_data: f("diff_sigma_data", 0.2),
            sigma_min: f("diff_sigma_min", 1e-4),
            sigma_max: f("diff_sigma_max", 3.0),
            rho: f("diff_rho", 9.0),
            steps: w
                .get("diff_steps")
                .and_then(|v| v.first().copied())
                .unwrap_or(5.0) as usize,
        }
    }

    fn g(&self, name: &str) -> &[f32] {
        self.w
            .get(&format!("diffusion.{name}"))
            .unwrap_or_else(|| panic!("missing diffusion.{name}"))
    }

    /// AdaLayerNorm: layernorm over F then per-channel affine (1+γ)·ln+β, γ/β = fc(s)∈ℝ²⁴⁸.
    fn ada_ln(&self, h: &[f32], l: usize, s: &[f32], fc: &str) -> Vec<f32> {
        let gb = linear(
            s,
            1,
            S,
            self.g(&format!("{fc}.weight")),
            Some(self.g(&format!("{fc}.bias"))),
            2 * F,
        );
        let (gamma, beta) = (&gb[..F], &gb[F..]);
        let ln = layer_norm_plain(h, l, F, EPS);
        let mut out = vec![0f32; l * F];
        for t in 0..l {
            for c in 0..F {
                out[t * F + c] = (1.0 + gamma[c]) * ln[t * F + c] + beta[c];
            }
        }
        out
    }

    /// One StyleTransformerBlock: x += StyleAttention(x,s); x += FFN(x).
    fn block(&self, h: &mut [f32], l: usize, s: &[f32], i: usize) {
        let p = |s: &str| format!("blocks.{i}.{s}");
        // --- self-attention ---
        let xn = self.ada_ln(h, l, s, &p("attention.norm.fc"));
        let cn = self.ada_ln(h, l, s, &p("attention.norm_context.fc"));
        let q = linear(&xn, l, F, self.g(&p("attention.to_q.weight")), None, MID);
        let kv = linear(
            &cn,
            l,
            F,
            self.g(&p("attention.to_kv.weight")),
            None,
            2 * MID,
        );
        let scale = (HD as f32).powf(-0.5);
        let mut ctx = vec![0f32; l * MID]; // attention output [L, MID]
        for hd in 0..HEADS {
            let off = hd * HD;
            for i_q in 0..l {
                let mut sim = vec![0f32; l];
                for j in 0..l {
                    let mut dot = 0.0;
                    for d in 0..HD {
                        dot += q[i_q * MID + off + d] * kv[j * 2 * MID + off + d];
                    }
                    sim[j] = dot * scale;
                }
                softmax(&mut sim);
                for d in 0..HD {
                    let mut acc = 0.0;
                    for j in 0..l {
                        acc += sim[j] * kv[j * 2 * MID + MID + off + d]; // v = kv[:, MID:]
                    }
                    ctx[i_q * MID + off + d] = acc;
                }
            }
        }
        let attn = linear(
            &ctx,
            l,
            MID,
            self.g(&p("attention.attention.to_out.weight")),
            Some(self.g(&p("attention.attention.to_out.bias"))),
            F,
        );
        for k in 0..l * F {
            h[k] += attn[k];
        }
        // --- feed-forward ---
        let mut ff = linear(
            h,
            l,
            F,
            self.g(&p("feed_forward.0.weight")),
            Some(self.g(&p("feed_forward.0.bias"))),
            2 * F,
        );
        gelu_exact(&mut ff);
        let ff = linear(
            &ff,
            l,
            2 * F,
            self.g(&p("feed_forward.2.weight")),
            Some(self.g(&p("feed_forward.2.bias"))),
            F,
        );
        for k in 0..l * F {
            h[k] += ff[k];
        }
    }

    /// The denoiser net: (x[256], time, emb[L,768], feat[256]) → [256].
    fn net(&self, x: &[f32], time: f32, emb: &[f32], l: usize, s: &[f32]) -> Vec<f32> {
        // mapping = to_mapping( GELU(Lin_257→1024(time_pos)) + GELU(Lin_256→1024(feat)) )
        let mut tpos = vec![0f32; 257];
        tpos[0] = time;
        let tw = self.g("to_time.0.0.weights"); // [128]
        for j in 0..128 {
            let f = time * tw[j] * 2.0 * std::f32::consts::PI;
            tpos[1 + j] = f.sin();
            tpos[1 + 128 + j] = f.cos();
        }
        let mut t_emb = linear(
            &tpos,
            1,
            257,
            self.g("to_time.0.1.weight"),
            Some(self.g("to_time.0.1.bias")),
            F,
        );
        gelu_exact(&mut t_emb);
        let mut f_emb = linear(
            s,
            1,
            S,
            self.g("to_features.0.weight"),
            Some(self.g("to_features.0.bias")),
            F,
        );
        gelu_exact(&mut f_emb);
        let mut mapping: Vec<f32> = (0..F).map(|k| t_emb[k] + f_emb[k]).collect();
        mapping = linear(
            &mapping,
            1,
            F,
            self.g("to_mapping.0.weight"),
            Some(self.g("to_mapping.0.bias")),
            F,
        );
        gelu_exact(&mut mapping);
        mapping = linear(
            &mapping,
            1,
            F,
            self.g("to_mapping.2.weight"),
            Some(self.g("to_mapping.2.bias")),
            F,
        );
        gelu_exact(&mut mapping);

        // h[t] = [ x(256, broadcast) ‖ emb[t](768) ]
        let mut h = vec![0f32; l * F];
        for t in 0..l {
            h[t * F..t * F + C].copy_from_slice(&x[..C]);
            h[t * F + C..t * F + F].copy_from_slice(&emb[t * E..t * E + E]);
        }
        for i in 0..NBLK {
            for t in 0..l {
                for k in 0..F {
                    h[t * F + k] += mapping[k];
                }
            }
            self.block(&mut h, l, s, i);
        }
        // mean over time → [1024]; Conv1x1(1024→256) == Linear
        let mut pooled = vec![0f32; F];
        for t in 0..l {
            for k in 0..F {
                pooled[k] += h[t * F + k];
            }
        }
        for v in pooled.iter_mut() {
            *v /= l as f32;
        }
        linear(
            &pooled,
            1,
            F,
            self.g("to_out.1.weight"),
            Some(self.g("to_out.1.bias")),
            C,
        )
    }

    /// KDiffusion denoise_fn: c_skip·x + c_out·net(c_in·x, c_noise).
    fn denoise(&self, x: &[f32], sigma: f32, emb: &[f32], l: usize, s: &[f32]) -> Vec<f32> {
        let sd = self.sigma_data;
        let c_skip = sd * sd / (sigma * sigma + sd * sd);
        let c_out = sigma * sd / (sd * sd + sigma * sigma).sqrt();
        let c_in = 1.0 / (sigma * sigma + sd * sd).sqrt();
        let c_noise = sigma.ln() * 0.25;
        let xin: Vec<f32> = x.iter().map(|v| c_in * v).collect();
        let pred = self.net(&xin, c_noise, emb, l, s);
        (0..C).map(|k| c_skip * x[k] + c_out * pred[k]).collect()
    }

    fn karras_sigmas(&self) -> Vec<f32> {
        let inv = 1.0 / self.rho;
        let (a, b) = (self.sigma_max.powf(inv), self.sigma_min.powf(inv));
        let mut s: Vec<f32> = (0..self.steps)
            .map(|i| (a + (i as f32 / (self.steps - 1) as f32) * (b - a)).powf(self.rho))
            .collect();
        s.push(0.0);
        s
    }

    /// ADPM2 sample → `s_pred [256]`. `noise_init`/`noises` are the replayed RNG draws
    /// (one initial + steps-1 per-step), so output is deterministic given them.
    pub fn sample(
        &self,
        noise_init: &[f32],
        noises: &[Vec<f32>],
        emb: &[f32],
        l: usize,
        ref_s: &[f32],
    ) -> Vec<f32> {
        let sig = self.karras_sigmas();
        let mut x: Vec<f32> = noise_init.iter().map(|v| sig[0] * v).collect();
        for i in 0..self.steps - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            let sigma_up = (sn * sn * (s * s - sn * sn) / (s * s)).sqrt();
            let sigma_down = (sn * sn - sigma_up * sigma_up).sqrt();
            let sigma_mid = (s + sigma_down) * 0.5; // rho=1
            let dn = self.denoise(&x, s, emb, l, ref_s);
            let d: Vec<f32> = (0..C).map(|k| (x[k] - dn[k]) / s).collect();
            let x_mid: Vec<f32> = (0..C).map(|k| x[k] + d[k] * (sigma_mid - s)).collect();
            let dn_mid = self.denoise(&x_mid, sigma_mid, emb, l, ref_s);
            let d_mid: Vec<f32> = (0..C).map(|k| (x_mid[k] - dn_mid[k]) / sigma_mid).collect();
            let nz = &noises[i];
            for k in 0..C {
                x[k] = x[k] + d_mid[k] * (sigma_down - s) + nz[k] * sigma_up;
            }
        }
        x // clamp=False
    }

    /// Exposed for the parity fixture: one isolated denoiser eval at the first sigma.
    pub fn net_eval(&self, x: &[f32], time: f32, emb: &[f32], l: usize, ref_s: &[f32]) -> Vec<f32> {
        self.net(x, time, emb, l, ref_s)
    }
}
