//! JS-facing embedding surface: `EmbeddingModel` — load an EmbeddingGemma
//! GGUF, embed text → an L2-normalized float vector (Matryoshka-truncatable).
//!
//! Mirrors `tts.rs`'s `KokoroTts` for the GPU context + `api::Model` for the
//! streaming loader. The forward is the hybrid GPU path
//! (`reference::embed::gpu`) — matmuls on the GPU, norms/attention/pool in CPU
//! f32 — bit-identical to the CPU oracle (`reference::embed::forward`), itself
//! cos 0.9997 vs Ollama. `load*`/`embed*` are async (GPU readback).
//!
//! **Streaming.** `loadFromOpfs` builds a streaming GGUF reader: matmul
//! weights flow through a persistent `WeightCache` (GPU-resident, fetched
//! once each), and `token_embd` (~400 MB of the 621 MB file) is never made
//! resident — only per-token rows are range-fetched. wasm linear-memory peak
//! is one tensor, not the whole file. iPhone-critical.
//!
//! Tokenization uses the SentencePiece unigram tokenizer (`tokenizer::spm`)
//! because EmbeddingGemma's GGUF ships scores, not BPE merges.

use std::sync::Arc;

use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::{GgufReader, TensorFetcher};
use crate::reference::embed::EmbedModel;
use crate::tokenizer::SpmTokenizer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct EmbeddingModel {
    model: EmbedModel,
    tok: SpmTokenizer,
    ctx: WgpuCtx,
    pipes: Arc<Pipelines>,
    wcache: WeightCache,
    bos: u32,
    eos: u32,
    add_bos: bool,
    add_eos: bool,
}

impl EmbeddingModel {
    /// Build everything from a (possibly streaming) reader + GPU context.
    async fn from_reader(reader: GgufReader) -> Result<Self> {
        let r_arc = Arc::new(reader);
        let tok = SpmTokenizer::from_gguf(&r_arc)?;
        let bos = meta_u32(&r_arc, "tokenizer.ggml.bos_token_id", 2);
        let eos = meta_u32(&r_arc, "tokenizer.ggml.eos_token_id", 1);
        let add_bos = meta_bool(&r_arc, "tokenizer.ggml.add_bos_token", true);
        let add_eos = meta_bool(&r_arc, "tokenizer.ggml.add_eos_token", true);
        let model = EmbedModel::new(r_arc.clone())?;
        let ctx = WgpuCtx::new().await?;
        let pipes = Arc::new(Pipelines::new(&ctx.device));
        let wcache = WeightCache::new(
            r_arc,
            ctx.device.clone(),
            ctx.queue.clone(),
            Arc::clone(&ctx.bind_cache),
        );
        Ok(Self {
            model,
            tok,
            ctx,
            pipes,
            wcache,
            bos,
            eos,
            add_bos,
            add_eos,
        })
    }

    /// Load from in-memory GGUF bytes (native / desktop convenience). For the
    /// PWA / iPhone use [`EmbeddingModel::load_streaming_native`] instead.
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        Self::from_reader(GgufReader::new(bytes)?).await
    }

    /// Load from a streaming `TensorFetcher` (OPFS / HTTP-range). Weights are
    /// fetched on demand; the file is never fully resident in wasm memory.
    pub async fn load_streaming_native(fetcher: Arc<dyn TensorFetcher>) -> Result<Self> {
        Self::from_reader(GgufReader::new_streaming(fetcher).await?).await
    }

    /// Output embedding dimension (before Matryoshka truncation).
    pub fn dim_native(&self) -> u32 {
        self.model.cfg.embed_dim
    }

    /// Tokenize + wrap with BOS/EOS per the GGUF's tokenizer flags.
    fn ids_for(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos {
            ids.push(self.bos);
        }
        ids.extend(self.tok.encode(text));
        if self.add_eos {
            ids.push(self.eos);
        }
        ids
    }

    /// Embed one string → L2-normalized vector of length
    /// `min(target_dim, dim)` (`target_dim = 0` ⇒ full dim).
    pub async fn embed_native(&self, text: &str, target_dim: usize) -> Result<Vec<f32>> {
        let ids = self.ids_for(text);
        self.model
            .embed_ids_gpu(&self.ctx, &self.pipes, &self.wcache, &ids, target_dim)
            .await
    }

    /// Embed many strings (sequentially). Returns a flat `[n * out_dim]` buffer
    /// plus the per-vector dimension so JS can reshape.
    pub async fn embed_batch_native(
        &self,
        texts: &[String],
        target_dim: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let mut out = Vec::new();
        let mut dim = 0usize;
        for t in texts {
            let v = self.embed_native(t, target_dim).await?;
            dim = v.len();
            out.extend_from_slice(&v);
        }
        Ok((out, dim))
    }
}

fn meta_u32(r: &GgufReader, key: &str, default: u32) -> u32 {
    r.get(key)
        .ok()
        .and_then(|v| v.as_u32().ok())
        .unwrap_or(default)
}
fn meta_bool(r: &GgufReader, key: &str, default: bool) -> bool {
    r.get(key)
        .ok()
        .and_then(|v| v.as_bool().ok())
        .unwrap_or(default)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl EmbeddingModel {
    /// Load from GGUF bytes (whole-in-memory). Prefer `loadFromOpfs` on the
    /// PWA — this holds the full file in wasm linear memory.
    #[wasm_bindgen(js_name = load)]
    pub async fn load_js(bytes: Vec<u8>) -> std::result::Result<EmbeddingModel, JsError> {
        Self::load_native(bytes)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Streaming load from OPFS. `read_fn(offset, length) -> Uint8Array` is a
    /// JS callback (the worker's sync OPFS reader). Weights are fetched on
    /// demand; the 621 MB file never fully enters wasm memory.
    #[wasm_bindgen(js_name = loadFromOpfs)]
    pub async fn load_from_opfs_js(
        read_fn: js_sys::Function,
        total_bytes: f64,
    ) -> std::result::Result<EmbeddingModel, JsError> {
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

    /// Output embedding dimension.
    #[wasm_bindgen(js_name = dim, getter)]
    pub fn dim_js(&self) -> u32 {
        self.dim_native()
    }

    /// Embed one string → Float32Array. `targetDim = 0` ⇒ full dimension.
    #[wasm_bindgen(js_name = embed)]
    pub async fn embed_js(
        &self,
        text: String,
        target_dim: u32,
    ) -> std::result::Result<Vec<f32>, JsError> {
        self.embed_native(&text, target_dim as usize)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Embed many strings → flat Float32Array of `n * dim`. The caller knows
    /// `dim` from the `dim` getter (or `targetDim` when non-zero) and reshapes.
    #[wasm_bindgen(js_name = embedBatch)]
    pub async fn embed_batch_js(
        &self,
        texts: Vec<String>,
        target_dim: u32,
    ) -> std::result::Result<Vec<f32>, JsError> {
        self.embed_batch_native(&texts, target_dim as usize)
            .await
            .map(|(v, _dim)| v)
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }
}
