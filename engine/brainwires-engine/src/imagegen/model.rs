//! `ImageModel` — the end-to-end Z-Image-Turbo engine: the 4th wasm-bindgen
//! class (alongside `Model`, `EmbeddingModel`, `DiffusionGemma`), composing the
//! three async-streaming GPU component forwards into one text → image path.
//!
//!   cap   = Qwen3Gpu(tokens)                              [cap_len, 2560]
//!   latent= seeded N(0,1)                                 [16, lh, lw]
//!   sched = FlowMatch(steps, dyn, calculate_shift(img_tokens))
//!   for s: v = DiT(latent, σ[s], cap)  [+ CFG vs neg];  latent += (σ'-σ)·v
//!   rgb   = VAE.decode(latent)                            [3, lh·8, lw·8]
//!
//! The generic [`ImageBundle`] holds the three [`StreamingShards`] + configs +
//! GPU context and is reusable native (`FileBlobSource`, the parity harness) and
//! in wasm (`HttpRangeBlobSource`). [`ImageModel`] is the concrete wasm wrapper.
//! Weights are range-fetched per tensor — never bulk-resident — so the 31 GB
//! model streams from the CDN without ever sitting in wasm memory.

use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::imagegen::config::{Qwen3Config, TransformerConfig, VaeConfig};
use crate::imagegen::dit_forward::DitGpu;
use crate::imagegen::qwen3_forward::Qwen3Gpu;
use crate::imagegen::scheduler::{FlowMatchScheduler, calculate_shift};
use crate::imagegen::sharded::ShardIndex;
use crate::imagegen::source::BlobSource;
use crate::imagegen::streaming::StreamingShards;
use crate::reference::vae_gpu::VaeGpu;

/// Default classifier-free-guidance scale (Z-Image-Turbo default).
pub const DEFAULT_CFG_SCALE: f32 = 4.0;
/// Default sampling steps (Z-Image-Turbo is a few-step turbo model).
pub const DEFAULT_STEPS: usize = 9;

/// Generic end-to-end image engine over a single `BlobSource` type.
pub struct ImageBundle<S: BlobSource> {
    ctx: WgpuCtx,
    pipes: Pipelines,
    enc_ss: StreamingShards<S>,
    enc_cfg: Qwen3Config,
    dit_ss: StreamingShards<S>,
    dit_cfg: TransformerConfig,
    vae_ss: StreamingShards<S>,
    vae_cfg: VaeConfig,
}

impl<S: BlobSource> ImageBundle<S> {
    /// Open all three components from their own blob sources. Each source roots a
    /// component directory (`text_encoder/`, `transformer/`, `vae/`): we read the
    /// component `config.json` + shard index (or single file) through it, then
    /// build a per-tensor streaming view.
    pub async fn open(enc_src: S, dit_src: S, vae_src: S) -> Result<Self> {
        // text encoder (sharded)
        let enc_cfg = Qwen3Config::parse(&enc_src.read_blob("config.json").await?)?;
        let enc_idx = ShardIndex::parse(&enc_src.read_blob("model.safetensors.index.json").await?)?;
        let enc_ss = StreamingShards::open_index(enc_src, &enc_idx).await?;

        // transformer / DiT (sharded)
        let dit_cfg = TransformerConfig::parse(&dit_src.read_blob("config.json").await?)?;
        let dit_idx = ShardIndex::parse(
            &dit_src
                .read_blob("diffusion_pytorch_model.safetensors.index.json")
                .await?,
        )?;
        let dit_ss = StreamingShards::open_index(dit_src, &dit_idx).await?;

        // VAE (single file)
        let vae_cfg = VaeConfig::parse(&vae_src.read_blob("config.json").await?)?;
        let vae_ss =
            StreamingShards::open_single(vae_src, "diffusion_pytorch_model.safetensors").await?;

        let ctx = WgpuCtx::new().await?;
        let pipes = Pipelines::new(&ctx.device);
        Ok(Self {
            ctx,
            pipes,
            enc_ss,
            enc_cfg,
            dit_ss,
            dit_cfg,
            vae_ss,
            vae_cfg,
        })
    }

    /// Latent channel count (DiT in_channels == VAE latent).
    pub fn latent_channels(&self) -> usize {
        self.dit_cfg.in_channels as usize
    }

    /// Generate an RGB image `[3, lh·8, lw·8]` (values in [0,1]) from caption +
    /// negative token ids. `cfg_scale == 1.0` skips the negative pass. `lh`/`lw`
    /// are the latent dims (image px / VAE 8×).
    ///
    /// `report(label, done, total)` (optional) is called frequently — per
    /// encoder/DiT layer and per VAE stage — with a human-readable phase label
    /// and an in-phase progress fraction, so the UI shows constant movement
    /// during the (network-bound, multi-minute) run rather than a silent hang.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        tokens: &[u32],
        neg_tokens: &[u32],
        cfg_scale: f32,
        lh: usize,
        lw: usize,
        steps: usize,
        seed: u64,
        report: Option<&dyn Fn(&str, usize, usize)>,
    ) -> Result<Vec<f32>> {
        let noop = |_: &str, _: usize, _: usize| {};
        let report: &dyn Fn(&str, usize, usize) = report.unwrap_or(&noop);

        // 1. encode caption (+ negative for CFG)
        let enc = Qwen3Gpu::new(&self.ctx, &self.pipes, &self.enc_ss, &self.enc_cfg);
        let cap = enc
            .forward(tokens, Some(&|i, n| report("Encoding prompt", i, n)))
            .await?;
        let use_cfg = (cfg_scale - 1.0).abs() > 1e-3 && !neg_tokens.is_empty();
        let ncap = if use_cfg {
            Some(
                enc.forward(
                    neg_tokens,
                    Some(&|i, n| report("Encoding negative prompt", i, n)),
                )
                .await?,
            )
        } else {
            None
        };

        // 2. seeded latent noise
        let cin = self.latent_channels();
        let mut latent = gaussian_noise(cin * lh * lw, seed);

        // 3. dynamic-shift flow-match schedule (image-token count → mu)
        let p = self.dit_cfg.patch_size() as usize;
        let img_tokens = (lh / p) * (lw / p);
        let sched = FlowMatchScheduler::new(steps, true, calculate_shift(img_tokens));

        // 4. denoise loop
        let dit = DitGpu::new(&self.ctx, &self.pipes, &self.dit_ss, &self.dit_cfg);
        for s in 0..steps {
            let sigma = sched.sigma(s);
            let lbl = format!("Denoising · step {}/{}", s + 1, steps);
            let v_pos = dit
                .forward(
                    &latent,
                    lh,
                    lw,
                    sigma,
                    &cap,
                    tokens.len(),
                    Some(&|i, n| report(&lbl, i, n)),
                )
                .await?;
            let v = if let Some(ncap) = &ncap {
                let nlbl = format!("Denoising · step {}/{} (guidance)", s + 1, steps);
                let v_neg = dit
                    .forward(
                        &latent,
                        lh,
                        lw,
                        sigma,
                        ncap,
                        neg_tokens.len(),
                        Some(&|i, n| report(&nlbl, i, n)),
                    )
                    .await?;
                cfg_combine(&v_pos, &v_neg, cfg_scale)
            } else {
                v_pos
            };
            sched.step_in_place(&mut latent, &v, s);
        }

        // 5. decode
        VaeGpu::new(&self.ctx, &self.pipes, &self.vae_ss, &self.vae_cfg)
            .decode_reporting(&latent, lh, lw, Some(&|i, n| report("Decoding image", i, n)))
            .await
    }
}

/// CFG combine: `v_neg + scale·(v_pos − v_neg)`.
pub fn cfg_combine(v_pos: &[f32], v_neg: &[f32], scale: f32) -> Vec<f32> {
    v_pos
        .iter()
        .zip(v_neg)
        .map(|(&p, &n)| n + scale * (p - n))
        .collect()
}

/// Deterministic `N(0,1)` via splitmix64 + Box–Muller (no rng dep, no
/// `Math.random`, so it ports to wasm verbatim — matches reference::pipeline).
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let unit = |u: u64| ((u >> 11) as f64) / ((1u64 << 53) as f64);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = unit(next_u64()).max(1e-12);
        let u2 = unit(next_u64());
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = std::f64::consts::TAU * u2;
        out.push((r * ang.cos()) as f32);
        if out.len() < n {
            out.push((r * ang.sin()) as f32);
        }
    }
    out
}

/// Channel-first RGB `[3,H,W]` in [0,1] → row-major RGBA8 `[H*W*4]` (alpha 255),
/// the layout a browser `ImageData` / canvas `putImageData` consumes directly.
pub fn rgb_chw_to_rgba8(rgb: &[f32], h: usize, w: usize) -> Vec<u8> {
    let plane = h * w;
    let mut out = vec![255u8; plane * 4];
    for i in 0..plane {
        for c in 0..3 {
            out[i * 4 + c] = (rgb[c * plane + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_combine_scales() {
        // scale 1 ⇒ positive; scale 4 pushes away from negative
        assert_eq!(cfg_combine(&[1.0, -2.0], &[0.5, 0.5], 1.0), vec![1.0, -2.0]);
        assert_eq!(cfg_combine(&[1.0], &[0.0], 4.0), vec![4.0]);
    }

    #[test]
    fn gaussian_noise_is_deterministic_and_finite() {
        let a = gaussian_noise(1000, 42);
        let b = gaussian_noise(1000, 42);
        assert_eq!(a, b, "same seed ⇒ same noise (no Math.random)");
        assert_ne!(a, gaussian_noise(1000, 43), "different seed ⇒ different");
        assert!(a.iter().all(|v| v.is_finite()));
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.15, "≈zero-mean N(0,1), got {mean}");
    }

    #[test]
    fn rgba8_layout_is_interleaved_with_opaque_alpha() {
        // 1×2 image, channel-first [R0 R1 | G0 G1 | B0 B1]
        let rgb = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // px0 red, px1 green
        let out = rgb_chw_to_rgba8(&rgb, 1, 2);
        assert_eq!(out, vec![255, 0, 0, 255, 0, 255, 0, 255]);
    }
}

// ---------- wasm-bindgen surface ----------

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use crate::imagegen::source::HttpRangeBlobSource;
    use wasm_bindgen::prelude::*;

    /// JS-facing Z-Image-Turbo engine. Streams its weights from a CDN base URL
    /// via HTTP `Range` — never holding a shard in memory — and renders an image
    /// on the GPU. Tokenization happens JS-side (pass token id arrays); see the
    /// PWA's image worker.
    #[wasm_bindgen]
    pub struct ImageModel {
        bundle: ImageBundle<HttpRangeBlobSource>,
        last_step: std::cell::Cell<u32>,
        total_steps: std::cell::Cell<u32>,
    }

    #[wasm_bindgen]
    impl ImageModel {
        /// Load all three components under `base_url` (expects `text_encoder/`,
        /// `transformer/`, `vae/` subpaths each with their config + safetensors).
        /// e.g. `https://models.brainwires.dev/z-image-turbo`.
        #[wasm_bindgen(js_name = loadFromUrl)]
        pub async fn load_from_url(base_url: String) -> std::result::Result<ImageModel, JsError> {
            let base = base_url.trim_end_matches('/').to_string();
            let enc = HttpRangeBlobSource::new(format!("{base}/text_encoder"));
            let dit = HttpRangeBlobSource::new(format!("{base}/transformer"));
            let vae = HttpRangeBlobSource::new(format!("{base}/vae"));
            let bundle = ImageBundle::open(enc, dit, vae)
                .await
                .map_err(|e| JsError::new(&format!("{e:?}")))?;
            Ok(ImageModel {
                bundle,
                last_step: std::cell::Cell::new(0),
                total_steps: std::cell::Cell::new(0),
            })
        }

        /// Default sampling steps.
        #[wasm_bindgen(js_name = defaultSteps, getter)]
        pub fn default_steps(&self) -> u32 {
            DEFAULT_STEPS as u32
        }

        /// 0-based index of the last completed denoise step (for progress UI).
        #[wasm_bindgen(js_name = stepIndex, getter)]
        pub fn step_index(&self) -> u32 {
            self.last_step.get()
        }

        /// Total denoise steps for the active generation.
        #[wasm_bindgen(js_name = totalSteps, getter)]
        pub fn total_steps_getter(&self) -> u32 {
            self.total_steps.get()
        }

        /// Generate an image. `tokens`/`neg_tokens` are caption / negative token
        /// ids (JS-tokenized). `cfg_scale <= 0` ⇒ the model default; `steps == 0`
        /// ⇒ the default. `lh`/`lw` are latent dims (image px ÷ 8).
        ///
        /// `on_progress` (optional) is invoked frequently — per encoder/DiT
        /// layer and per VAE stage — as `on_progress(label, done, total)` so the
        /// worker can post live progress to the UI during the (network-bound,
        /// multi-minute) run. Returns RGBA8 bytes `[lh·8 · lw·8 · 4]` for
        /// `putImageData`.
        #[wasm_bindgen(js_name = generate)]
        #[allow(clippy::too_many_arguments)]
        pub async fn generate(
            &self,
            tokens: Vec<u32>,
            neg_tokens: Vec<u32>,
            cfg_scale: f32,
            lh: u32,
            lw: u32,
            steps: u32,
            seed: f64,
            on_progress: Option<js_sys::Function>,
        ) -> std::result::Result<Vec<u8>, JsError> {
            let steps = if steps == 0 {
                DEFAULT_STEPS
            } else {
                steps as usize
            };
            let scale = if cfg_scale <= 0.0 {
                DEFAULT_CFG_SCALE
            } else {
                cfg_scale
            };
            let (lh, lw) = (lh as usize, lw as usize);
            self.total_steps.set(steps as u32);
            self.last_step.set(0);

            // Bridge the Rust reporter → the JS callback. Called synchronously
            // from deep in the async forward; the worker callback just
            // `postMessage`s, and delivery happens at the next `await` yield
            // (every layer does a network fetch), so the UI updates live.
            let reporter = move |label: &str, done: usize, total: usize| {
                if let Some(f) = &on_progress {
                    let _ = f.call3(
                        &JsValue::NULL,
                        &JsValue::from_str(label),
                        &JsValue::from_f64(done as f64),
                        &JsValue::from_f64(total as f64),
                    );
                }
            };

            let rgb = self
                .bundle
                .generate(
                    &tokens,
                    &neg_tokens,
                    scale,
                    lh,
                    lw,
                    steps,
                    seed as u64,
                    Some(&reporter),
                )
                .await
                .map_err(|e| JsError::new(&format!("{e:?}")))?;
            Ok(rgb_chw_to_rgba8(&rgb, lh * 8, lw * 8))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::ImageModel;
