//! JS-facing DiffusionGemma surface: `DiffusionGemma` — load the
//! `diffusion-gemma` GGUF (the 26B-A4B sparse-MoE block-diffusion model),
//! denoise a masked canvas into text.
//!
//! Mirrors [`crate::embed::EmbeddingModel`] for the GPU context + streaming
//! loader. The forward is the full GPU path
//! ([`crate::reference::diffusion::gpu::diffusion_forward_gpu`]) — dense + MoE
//! matmuls on the GPU, the bidirectional masked attention / norms / sampler in
//! CPU f32 — validated argmax-exact vs the CPU oracle (which is itself the 1:1
//! mirror of llama.cpp PR 24423, the only runner for this architecture).
//!
//! Two entry points:
//!   - native [`DiffusionGemma::generate_native`] runs the whole entropy-bound
//!     denoise loop in-process (blocks on each GPU forward) and returns text;
//!   - the wasm surface (C5b) exposes a `denoiseStep` so the JS worker drives
//!     the loop and can render the canvas condensing out of noise each step.
//!
//! **Streaming.** Weights flow through a persistent `WeightCache`; each MoE
//! layer's ~0.5 GB of stacked experts is made resident then destroyed before
//! the next layer (a 256-token canvas routes its top-8 across ~all 128 experts,
//! so per-layer is the right grain). wasm peak stays bounded to one tensor.

use std::sync::Arc;

use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::{GgufReader, TensorFetcher};
use crate::reference::diffusion::DiffusionConfig;
use crate::reference::diffusion::gpu::diffusion_forward_gpu;
use crate::reference::diffusion::sampler::{
    CanvasForward, EbParams, StepInfo, XorShiftRng, generate_entropy_bound,
};
use crate::reference::weights::Weights;
use crate::tokenizer::BpeTokenizer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Default canvas length (block size) — matches the released checkpoints'
/// `canvas_length` and the llama.cpp runner's default.
pub const DEFAULT_CANVAS_LEN: usize = 256;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct DiffusionGemma {
    cfg: DiffusionConfig,
    weights: Weights,
    tok: BpeTokenizer,
    ctx: WgpuCtx,
    pipes: Pipelines,
    wcache: WeightCache,
    bos: u32,
}

impl DiffusionGemma {
    async fn from_reader(reader: GgufReader) -> Result<Self> {
        let r_arc = Arc::new(reader);
        let cfg = DiffusionConfig::from_gguf(&r_arc)?;
        let tok = BpeTokenizer::from_gguf(&r_arc)?;
        let bos = r_arc
            .get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(2);
        let weights = Weights::new(r_arc.clone());
        let ctx = WgpuCtx::new().await?;
        let pipes = Pipelines::new(&ctx.device);
        let wcache = WeightCache::new(
            r_arc,
            ctx.device.clone(),
            ctx.queue.clone(),
            Arc::clone(&ctx.bind_cache),
        );
        Ok(Self {
            cfg,
            weights,
            tok,
            ctx,
            pipes,
            wcache,
            bos,
        })
    }

    /// Load from in-memory GGUF bytes (desktop convenience). For the PWA use the
    /// streaming loader — this 16.8 GB model would never fit wasm memory.
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        Self::from_reader(GgufReader::new(bytes)?).await
    }

    /// Load from a streaming `TensorFetcher` (OPFS / HTTP-range / file). Weights
    /// are fetched on demand; the file is never fully resident.
    pub async fn load_streaming_native(fetcher: Arc<dyn TensorFetcher>) -> Result<Self> {
        Self::from_reader(GgufReader::new_streaming(fetcher).await?).await
    }

    /// Default canvas/block length for this model.
    pub fn canvas_len(&self) -> usize {
        DEFAULT_CANVAS_LEN
    }

    /// Run the entropy-bound denoise loop in-process and return the decoded
    /// text. `on_step` (optional) is invoked once per denoise step with the
    /// current argmax canvas + stats — return `false` to abort early.
    pub fn generate_native(
        &self,
        prompt: &str,
        canvas_len: usize,
        params: &EbParams,
        seed: u64,
        mut on_step: Option<&mut dyn FnMut(&StepInfo) -> bool>,
    ) -> Result<String> {
        let mut prompt_ids = vec![self.bos];
        prompt_ids.extend(self.tok.encode(prompt));

        // Adapter: the sampler drives a sync `CanvasForward`; on native we block
        // on each async GPU forward.
        struct Fwd<'a> {
            m: &'a DiffusionGemma,
            prompt_ids: Vec<u32>,
        }
        impl CanvasForward for Fwd<'_> {
            fn forward(&mut self, canvas: &[u32], prev: Option<(&[f32], f32)>) -> Result<Vec<f32>> {
                let (pl, ti) = match prev {
                    Some((l, t)) => (Some(l), t),
                    None => (None, 1.0),
                };
                pollster::block_on(diffusion_forward_gpu(
                    &self.m.cfg,
                    &self.m.ctx,
                    &self.m.pipes,
                    &self.m.wcache,
                    &self.m.weights,
                    &self.prompt_ids,
                    canvas,
                    pl,
                    ti,
                ))
            }
            fn n_vocab(&self) -> usize {
                self.m.cfg.base.vocab_size as usize
            }
        }

        let mut fwd = Fwd {
            m: self,
            prompt_ids,
        };
        let mut rng = XorShiftRng(seed);
        let ids = generate_entropy_bound(&mut fwd, canvas_len, params, &mut rng, on_step.take())?;
        Ok(self.detokenize(&ids))
    }

    /// Join the SentencePiece pieces for `ids`, rendering the ▁ word-boundary
    /// marker as a space. Unknown ids are skipped.
    pub fn detokenize(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(piece) = self.tok.id_to_str(id) {
                s.push_str(piece);
            }
        }
        s.replace('\u{2581}', " ")
    }
}
