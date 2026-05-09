//! JS-facing types and entry points.
//!
//! On wasm32 these are exposed via wasm-bindgen. On native they remain Rust-only
//! and are used by integration tests / examples.
//!
//! Minimal API surface (M5 v0):
//!   - `Model::load(bytes)` — parse GGUF, init wgpu, upload pipelines (no weights yet).
//!   - `Model::encode(text)` / `Model::token_str(id)` — tokenizer access.
//!   - `Model::step(token_id)` — feed a single token at the current position; returns
//!     the argmax of the resulting next-token logits. Mutates internal KV cache.
//!   - `Model::reset()` — clear KV state to start a fresh conversation.
//!   - `Model::is_eos(id)` — checks against the GGUF's eos token id list.
//!
//! Streaming is JS's responsibility: loop `step` and call `token_str(id)` per step.
//! A `ReadableStream<string>` wrapper lands in v0.2.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgufReader;
use crate::model::config::Gemma4Config;
use crate::reference::Weights;
use crate::reference::forward_chained::Forward;
use crate::sampling::{Sampler, SamplingOptions};
use crate::template::gemma4_small;
use crate::tokenizer::BpeTokenizer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// M0 smoke export: doubles every f32 on the GPU. Useful from JS to confirm WebGPU
/// is wired up before loading the full model.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = computeSpike)]
pub async fn compute_spike_js(input: Vec<f32>) -> std::result::Result<Vec<f32>, JsError> {
    crate::backend::compute_spike(&input)
        .await
        .map_err(|e| JsError::new(&format!("{e}")))
}

// ---------- public Model surface ----------

/// A loaded Gemma 4 model with all GPU resources allocated. One `Model` corresponds to
/// one conversation: it owns the KV cache and tracks the current position.
///
/// Internally a `Model` is a tokenizer + a [`Forward`] + a [`Sampler`]. `Forward` runs
/// one wgpu CommandEncoder per token (M7 work) — significantly faster than the original
/// per-kernel-readback path, which is now retained only as a parity oracle.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Model {
    tokenizer: BpeTokenizer,
    forward: Forward,
    sampler: Sampler,
}

impl Model {
    /// Build a Model from an already-constructed GGUF reader. Shared by both
    /// the in-memory and streaming entry points so they can't drift.
    async fn from_reader(reader: GgufReader) -> Result<Self> {
        let cfg = Gemma4Config::from_gguf(&reader)?;
        let tokenizer = BpeTokenizer::from_gguf(&reader)?;
        let r_arc = Arc::new(reader);
        let weights = Weights::new(r_arc.clone());
        let ctx = WgpuCtx::new().await?;
        let pipes = Arc::new(Pipelines::new(&ctx.device));
        let wcache = Arc::new(WeightCache::new(r_arc, ctx.device.clone(), ctx.queue.clone()));
        let forward = Forward::new(cfg, ctx, pipes, weights, wcache).await?;
        Ok(Self {
            tokenizer,
            forward,
            sampler: Sampler::new(SamplingOptions::default()),
        })
    }

    /// Native-friendly constructor: takes ownership of GGUF bytes, initializes WebGPU,
    /// and prepares all the on-GPU resources (compute pipelines, weight cache).
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = GgufReader::new(bytes)?;
        Self::from_reader(reader).await
    }

    /// Streaming constructor: takes any [`crate::gguf::TensorFetcher`] (in-memory or
    /// HTTP) and reads only the header up front. Tensor bytes are pulled lazily
    /// through the fetcher and dropped after each GPU upload — this is what keeps
    /// peak CPU memory bounded for the wasm32 4 GB linear-memory cap.
    pub async fn load_streaming(
        fetcher: std::sync::Arc<dyn crate::gguf::TensorFetcher>,
    ) -> Result<Self> {
        let reader = GgufReader::new_streaming(fetcher).await?;
        Self::from_reader(reader).await
    }

    /// Encode text → token IDs (Ollama-matching BPE).
    pub fn encode_tokens(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Look up a token ID's string form (raw vocab entry; SentencePiece `▁` markers
    /// are not stripped — the caller does that in JS if it wants display text).
    pub fn token_str_native(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_str(id).map(|s| s.to_string())
    }

    /// Number of tokens in the vocab.
    pub fn vocab_size_native(&self) -> u32 { self.forward.cfg().vocab_size }

    /// Current sequence position (number of tokens fed so far).
    pub fn position_native(&self) -> u32 { self.forward.pos() }

    /// True iff `id` is one of the GGUF's EOS / EOT / end-of-turn tokens.
    pub fn is_eos_native(&self, id: u32) -> bool {
        self.forward.cfg().eos_ids.iter().any(|&e| e == id)
    }

    /// Reset KV state so the next call starts from an empty conversation.
    pub fn reset_native(&mut self) {
        self.forward.reset();
        self.sampler.clear_history();
    }

    /// Configure sampling. Defaults: temperature=0.7, top_k=40, top_p=0.95, no rep penalty.
    pub fn set_sampling_native(&mut self, opts: SamplingOptions) {
        self.sampler.set_options(opts);
    }

    /// Feed one token at the current position. Returns the *sampled* next token id
    /// (using current SamplingOptions). With `temperature=0`, this is the argmax.
    pub async fn step_native(&mut self, token_id: u32) -> Result<u32> {
        self.sampler.observe(token_id);
        let logits = self.forward.step(token_id).await?;
        let next = self.sampler.sample(&logits);
        Ok(next)
    }

    /// Render a list of chat messages into the Gemma 4 prompt format, ready to feed
    /// to `encode_tokens` + `step`. Includes the trailing `<|turn>model\n` so the
    /// next sampled token starts the assistant reply.
    pub fn render_chat_native(&self, messages: &[ChatMessage], with_bos: bool) -> String {
        gemma4_small::render_for_completion(messages, with_bos)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Model {
    /// JS entry point: build a Model from raw GGUF bytes (e.g. a `Uint8Array` from
    /// `fetch().then(r => r.arrayBuffer())`). Holds the entire GGUF in wasm linear
    /// memory; only suitable for files that fit under the 4 GB wasm32 cap.
    #[wasm_bindgen(js_name = load)]
    pub async fn load_js(bytes: Vec<u8>) -> std::result::Result<Model, JsError> {
        Self::load_native(bytes).await.map_err(|e| JsError::new(&format!("{e}")))
    }

    /// JS entry point: stream the GGUF over HTTP via byte-range requests. The full
    /// file never lands in wasm memory — tensors are fetched on demand and dropped
    /// after each GPU upload. This is the path that lets `gemma4:e2b` (~7 GB) load
    /// in the browser despite wasm32's 4 GB linear-memory cap.
    ///
    /// Requires the server to support `Range: bytes=N-M` and to expose either
    /// `Content-Range` or `X-Total-Size` so the client can discover the file length.
    #[wasm_bindgen(js_name = loadFromUrl)]
    pub async fn load_from_url_js(url: String) -> std::result::Result<Model, JsError> {
        let fetcher = crate::gguf::HttpRangeFetcher::new(url)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))?;
        let arc: std::sync::Arc<dyn crate::gguf::TensorFetcher> = std::sync::Arc::new(fetcher);
        Self::load_streaming(arc).await.map_err(|e| JsError::new(&format!("{e}")))
    }

    #[wasm_bindgen(js_name = encode)]
    pub fn encode_js(&self, text: &str) -> Vec<u32> { self.encode_tokens(text) }

    #[wasm_bindgen(js_name = tokenStr)]
    pub fn token_str_js(&self, id: u32) -> Option<String> { self.token_str_native(id) }

    #[wasm_bindgen(js_name = vocabSize, getter)]
    pub fn vocab_size_js(&self) -> u32 { self.vocab_size_native() }

    #[wasm_bindgen(js_name = position, getter)]
    pub fn position_js(&self) -> u32 { self.position_native() }

    #[wasm_bindgen(js_name = isEos)]
    pub fn is_eos_js(&self, id: u32) -> bool { self.is_eos_native(id) }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset_js(&mut self) { self.reset_native() }

    /// Feed one token, advance pos, return sampled next token id.
    #[wasm_bindgen(js_name = step)]
    pub async fn step_js(&mut self, token_id: u32) -> std::result::Result<u32, JsError> {
        self.step_native(token_id).await.map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Configure sampling from a JSON-shape `{temperature, top_k, top_p, repetition_penalty, seed}`.
    /// JS callers pass an object; serde decodes it.
    #[wasm_bindgen(js_name = setSampling)]
    pub fn set_sampling_js(&mut self, opts_json: JsValue) -> std::result::Result<(), JsError> {
        let opts: SamplingOptions = serde_wasm_bindgen::from_value(opts_json)
            .map_err(|e| JsError::new(&format!("invalid sampling options: {e}")))?;
        self.sampler.set_options(opts);
        Ok(())
    }

    /// Render a single user message (and optional system message) into the Gemma 4
    /// chat-template prompt. JS callers pass `[{role, content}, ...]` as JSON.
    #[wasm_bindgen(js_name = renderChat)]
    pub fn render_chat_js(&self, messages_json: JsValue, with_bos: bool) -> std::result::Result<String, JsError> {
        let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages_json)
            .map_err(|e| JsError::new(&format!("invalid messages: {e}")))?;
        Ok(self.render_chat_native(&msgs, with_bos))
    }
}

// ---------- (legacy) options shapes — retained from M0 stub for future use ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_max_tokens() -> u32 { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.95 }
fn default_top_k() -> u32 { 40 }
fn default_repetition_penalty() -> f32 { 1.0 }
