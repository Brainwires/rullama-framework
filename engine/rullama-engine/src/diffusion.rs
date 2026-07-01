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

use std::cell::RefCell;
use std::sync::Arc;

use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::{GgufReader, TensorFetcher};
use crate::reference::diffusion::DiffusionConfig;
use crate::reference::diffusion::gpu::diffusion_forward_gpu;
#[cfg(not(target_arch = "wasm32"))]
use crate::reference::diffusion::sampler::{CanvasForward, StepInfo, generate_entropy_bound};
use crate::reference::diffusion::sampler::{DenoiseState, EbParams, XorShiftRng};
use crate::reference::weights::Weights;
use crate::tokenizer::BpeTokenizer;

/// JS-driven denoise generation state: the resumable sampler state plus the
/// prompt + rng it runs against, and the last step's stats for the getters.
struct GenState {
    state: DenoiseState,
    prompt_ids: Vec<u32>,
    rng: XorShiftRng,
    last_step: u32,
    total_steps: u32,
    last_accepted: usize,
    last_mean_entropy: f32,
}

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
    /// JS-driven generation state (the wasm `denoiseStep` loop). `None` until
    /// `startGenerate`; unused by the native `generate_native` (which keeps its
    /// own local state).
    gen_state: RefCell<Option<GenState>>,
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
            gen_state: RefCell::new(None),
        })
    }

    /// Encode a prompt into ids with the leading BOS (shared by native + wasm).
    fn prompt_ids(&self, prompt: &str) -> Vec<u32> {
        let mut ids = vec![self.bos];
        ids.extend(self.tok.encode(prompt));
        ids
    }

    /// One async denoise step against the GPU forward. Returns the step outcome
    /// (and leaves the new argmax canvas in `gen_state`). Drives the wasm
    /// `denoiseStep`; native callers can use it for step-by-step streaming.
    /// Errors if `start_generate` wasn't called.
    pub async fn denoise_step(&self) -> Result<crate::reference::diffusion::sampler::StepOutcome> {
        // Pull the forward inputs out under a short borrow (no borrow is held
        // across the await).
        let (canvas, prev, prompt_ids) = {
            let mut slot = self.gen_state.borrow_mut();
            let g = slot.as_mut().ok_or_else(|| {
                crate::error::RullamaError::Inference("startGenerate not called".into())
            })?;
            (
                g.state.input_canvas(),
                g.state.take_prev(),
                g.prompt_ids.clone(),
            )
        };
        let logits = diffusion_forward_gpu(
            &self.cfg,
            &self.ctx,
            &self.pipes,
            &self.wcache,
            &self.weights,
            &prompt_ids,
            &canvas,
            prev.as_ref().map(|(l, _)| l.as_slice()),
            prev.as_ref().map(|(_, t)| *t).unwrap_or(1.0),
        )
        .await?;
        let mut slot = self.gen_state.borrow_mut();
        let g = slot.as_mut().unwrap();
        let outcome = g.state.ingest(logits, &mut g.rng);
        g.last_step = outcome.step_idx;
        g.total_steps = outcome.total_steps;
        g.last_accepted = outcome.n_accepted;
        g.last_mean_entropy = outcome.mean_entropy;
        Ok(outcome)
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
    /// current argmax canvas + stats — return `false` to abort early. Native
    /// only (blocks on each GPU forward); the wasm path drives `denoiseStep`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate_native(
        &self,
        prompt: &str,
        canvas_len: usize,
        params: &EbParams,
        seed: u64,
        mut on_step: Option<&mut dyn FnMut(&StepInfo) -> bool>,
    ) -> Result<String> {
        let prompt_ids = self.prompt_ids(prompt);

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

    /// Arm a step-driven generation: encode the prompt, random-init the canvas,
    /// stash the resumable state in `self.gen_state`. Drive it with [`Self::denoise_step`]
    /// until the outcome reports `done`. (The wasm `denoiseStep` loop uses this;
    /// native callers usually prefer the one-shot `generate_native`.)
    pub fn start_generate(&self, prompt: &str, canvas_len: usize, params: EbParams, seed: u64) {
        let prompt_ids = self.prompt_ids(prompt);
        let mut rng = XorShiftRng(seed);
        let n_vocab = self.cfg.base.vocab_size as usize;
        let state = DenoiseState::new(canvas_len, n_vocab, params, &mut rng);
        *self.gen_state.borrow_mut() = Some(GenState {
            state,
            prompt_ids,
            rng,
            last_step: 0,
            total_steps: 0,
            last_accepted: 0,
            last_mean_entropy: 0.0,
        });
    }

    /// Decode the current best-guess canvas to text (`""` if not generating).
    pub fn canvas_text(&self) -> String {
        match self.gen_state.borrow().as_ref() {
            Some(g) => self.detokenize(g.state.argmax_canvas()),
            None => String::new(),
        }
    }

    /// Whether the current generation has converged / spent its step budget.
    pub fn is_done(&self) -> bool {
        self.gen_state
            .borrow()
            .as_ref()
            .map(|g| g.state.is_done())
            .unwrap_or(true)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl DiffusionGemma {
    /// Streaming load from OPFS. `read_fn(offset, length) -> Uint8Array` is the
    /// worker's sync OPFS reader; weights are fetched on demand (the 16.8 GB
    /// file never fully enters wasm memory — per-layer MoE experts stream in and
    /// are destroyed each layer).
    #[wasm_bindgen(js_name = loadFromOpfs)]
    pub async fn load_from_opfs_js(
        read_fn: js_sys::Function,
        total_bytes: f64,
    ) -> std::result::Result<DiffusionGemma, JsError> {
        if !total_bytes.is_finite() || total_bytes < 0.0 {
            return Err(JsError::new(
                "loadFromOpfs: total_bytes must be a non-negative finite number",
            ));
        }
        let fetcher = crate::gguf::OpfsFetcher::new(read_fn, total_bytes as u64);
        let arc: Arc<dyn TensorFetcher> = Arc::new(fetcher);
        Self::load_streaming_native(arc)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Default canvas/block length.
    #[wasm_bindgen(js_name = canvasLen, getter)]
    pub fn canvas_len_js(&self) -> u32 {
        DEFAULT_CANVAS_LEN as u32
    }

    /// Arm a generation. `canvasLen = 0` ⇒ the model default; `maxSteps = 0` ⇒
    /// the entropy-bound default (48). The JS worker then calls `denoiseStep`
    /// repeatedly, rendering `canvasText` (or the returned text) each step,
    /// until `done` is true.
    #[wasm_bindgen(js_name = startGenerate)]
    pub fn start_generate_js(&self, prompt: String, canvas_len: u32, max_steps: u32, seed: f64) {
        let cl = if canvas_len == 0 {
            DEFAULT_CANVAS_LEN
        } else {
            canvas_len as usize
        };
        let params = EbParams {
            max_denoising_steps: if max_steps == 0 { 48 } else { max_steps },
            ..Default::default()
        };
        self.start_generate(&prompt, cl, params, seed as u64);
    }

    /// Run ONE denoise step (a full canvas forward + sample/accept/renoise) and
    /// return the current best-guess canvas decoded to text. Read `done` /
    /// `stepIndex` / `accepted` / `meanEntropy` getters for the loop control +
    /// progress. Render the returned text in place each step to show the canvas
    /// condensing out of noise.
    #[wasm_bindgen(js_name = denoiseStep)]
    pub async fn denoise_step_js(&self) -> std::result::Result<String, JsError> {
        if self.is_done() {
            return Ok(self.canvas_text());
        }
        self.denoise_step()
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))?;
        Ok(self.canvas_text())
    }

    /// Generation converged / budget spent.
    #[wasm_bindgen(js_name = done, getter)]
    pub fn done_js(&self) -> bool {
        self.is_done()
    }

    /// 0-based index of the last completed denoise step.
    #[wasm_bindgen(js_name = stepIndex, getter)]
    pub fn step_index_js(&self) -> u32 {
        self.gen_state
            .borrow()
            .as_ref()
            .map(|g| g.last_step)
            .unwrap_or(0)
    }

    /// Total step budget for the active generation.
    #[wasm_bindgen(js_name = totalSteps, getter)]
    pub fn total_steps_js(&self) -> u32 {
        self.gen_state
            .borrow()
            .as_ref()
            .map(|g| g.total_steps)
            .unwrap_or(0)
    }

    /// Positions accepted (unmasked) on the last step.
    #[wasm_bindgen(js_name = accepted, getter)]
    pub fn accepted_js(&self) -> u32 {
        self.gen_state
            .borrow()
            .as_ref()
            .map(|g| g.last_accepted as u32)
            .unwrap_or(0)
    }

    /// Mean per-position entropy on the last step (a confidence signal).
    #[wasm_bindgen(js_name = meanEntropy, getter)]
    pub fn mean_entropy_js(&self) -> f32 {
        self.gen_state
            .borrow()
            .as_ref()
            .map(|g| g.last_mean_entropy)
            .unwrap_or(0.0)
    }
}
