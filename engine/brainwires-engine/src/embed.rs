//! JS-facing embedding surface: `EmbeddingModel` — load an EmbeddingGemma
//! GGUF, embed text → an L2-normalized float vector (Matryoshka-truncatable).
//!
//! Mirrors `tts.rs`'s `KokoroTts` loading pattern: the PWA downloads +
//! OPFS-caches the GGUF and passes the bytes; the model loads whole
//! in-memory. The forward currently runs on the CPU oracle
//! (`reference::embed`) — correct and wasm-compatible; a GPU forward is a
//! follow-up perf optimization (the CPU path is fine for queries + small
//! documents, slower for large batches).
//!
//! Tokenization uses the SentencePiece unigram tokenizer (`tokenizer::spm`)
//! because EmbeddingGemma's GGUF ships scores, not BPE merges.

use std::sync::Arc;

use crate::error::Result;
use crate::gguf::GgufReader;
use crate::reference::embed::EmbedModel;
use crate::tokenizer::SpmTokenizer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct EmbeddingModel {
    model: EmbedModel,
    tok: SpmTokenizer,
    bos: u32,
    eos: u32,
    add_bos: bool,
    add_eos: bool,
}

impl EmbeddingModel {
    /// Load from in-memory GGUF bytes.
    pub fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = Arc::new(GgufReader::new(bytes)?);
        let tok = SpmTokenizer::from_gguf(&reader)?;
        let bos = reader
            .get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(2);
        let eos = reader
            .get("tokenizer.ggml.eos_token_id")
            .ok()
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(1);
        let add_bos = reader
            .get("tokenizer.ggml.add_bos_token")
            .ok()
            .and_then(|v| v.as_bool().ok())
            .unwrap_or(true);
        let add_eos = reader
            .get("tokenizer.ggml.add_eos_token")
            .ok()
            .and_then(|v| v.as_bool().ok())
            .unwrap_or(true);
        let model = EmbedModel::new(reader)?;
        Ok(Self { model, tok, bos, eos, add_bos, add_eos })
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
    pub fn embed_native(&self, text: &str, target_dim: usize) -> Result<Vec<f32>> {
        let ids = self.ids_for(text);
        self.model.embed_ids(&ids, target_dim)
    }

    /// Embed many strings (sequentially). Returns a flat `[n * out_dim]` buffer
    /// plus the per-vector dimension so JS can reshape.
    pub fn embed_batch_native(
        &self,
        texts: &[String],
        target_dim: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let mut out = Vec::new();
        let mut dim = 0usize;
        for t in texts {
            let v = self.embed_native(t, target_dim)?;
            dim = v.len();
            out.extend_from_slice(&v);
        }
        Ok((out, dim))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl EmbeddingModel {
    /// Load from GGUF bytes (the PWA reads the OPFS-cached file and passes them).
    #[wasm_bindgen(js_name = load)]
    pub fn load_js(bytes: Vec<u8>) -> std::result::Result<EmbeddingModel, JsError> {
        Self::load_native(bytes).map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Output embedding dimension.
    #[wasm_bindgen(js_name = dim, getter)]
    pub fn dim_js(&self) -> u32 {
        self.dim_native()
    }

    /// Embed one string → Float32Array. `targetDim = 0` ⇒ full dimension.
    #[wasm_bindgen(js_name = embed)]
    pub fn embed_js(&self, text: String, target_dim: u32) -> std::result::Result<Vec<f32>, JsError> {
        self.embed_native(&text, target_dim as usize)
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Embed many strings → flat Float32Array of `n * dim`. The caller knows
    /// `dim` from the `dim` getter (or `targetDim` when non-zero) and reshapes.
    #[wasm_bindgen(js_name = embedBatch)]
    pub fn embed_batch_js(
        &self,
        texts: Vec<String>,
        target_dim: u32,
    ) -> std::result::Result<Vec<f32>, JsError> {
        self.embed_batch_native(&texts, target_dim as usize)
            .map(|(v, _dim)| v)
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }
}
