//! Text-only generation pipeline around `quantized_gemma4::ModelWeights`.
//!
//! Parallel to the BF16 `Gemma4MultiModal` (which carries vision +
//! audio + multimodal preprocessing), this is a text-only wrapper for
//! the QMatMul/GGUF model. Loads weights via
//! `gguf_loader::load_quantized_gemma4_from_reader`, encodes prompts
//! through a `tokenizers::Tokenizer`, and exposes a `generate_greedy_streaming`
//! that drives `ModelWeights::forward` token-by-token.
//!
//! Why a separate type instead of reusing Gemma4MultiModal:
//! - Different model — `quantized_gemma4::ModelWeights` keeps weights
//!   as `QMatMul`, and the `forward` signature is simpler (one input
//!   tensor of token ids → logits).
//! - No vision/audio plumbing — Ollama's gemma4:e2b GGUF is text-only,
//!   so we skip the SigLIP + audio tower entirely.
//! - No PLE pre-computation on the host — the quantized model handles
//!   PLE internally inside `forward`.

#![cfg(feature = "local-llm-candle")]

use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::gemma4::config::Gemma4Config;
use candle_transformers::models::quantized_gemma4::ModelWeights;
use tokenizers::Tokenizer;

/// Errors surfaced by the quantized Gemma 4 text-only pipeline.
#[derive(Debug, thiserror::Error)]
pub enum QuantizedPipelineError {
    /// Failed to encode / decode through `tokenizers::Tokenizer`.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// `ModelWeights::forward` returned an error or the surrounding
    /// candle ops failed.
    #[error("model forward: {0}")]
    Model(String),
    /// Cumulative-decode call after a forward step failed.
    #[error("decode: {0}")]
    Decode(String),
    /// Internal mutex around the model state was poisoned by a panic
    /// on a previous generate call.
    #[error("mutex poisoned")]
    MutexPoisoned,
}

impl From<candle_core::Error> for QuantizedPipelineError {
    fn from(e: candle_core::Error) -> Self {
        QuantizedPipelineError::Model(format!("{e}"))
    }
}

/// Text-only generation pipeline around `quantized_gemma4::ModelWeights`.
pub struct Gemma4QuantizedTextOnly {
    model: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    cfg: Gemma4Config,
}

impl Gemma4QuantizedTextOnly {
    /// Build a pipeline from already-loaded components — typically the
    /// outputs of `gguf_loader::load_quantized_gemma4_from_reader` plus
    /// a `tokenizers::Tokenizer` parsed from the companion `tokenizer.json`.
    pub fn from_components(
        model: ModelWeights,
        tokenizer: Tokenizer,
        device: Device,
        cfg: Gemma4Config,
    ) -> Self {
        Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            cfg,
        }
    }

    /// Reference to the model's config (sliding window, layer types,
    /// vocab size, etc.). Used by upstream callers to size sampling
    /// state before invoking generate.
    pub fn config(&self) -> &Gemma4Config {
        &self.cfg
    }

    /// Reference to the device the model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Greedy generation. `eos_token_ids`: stop as soon as the model
    /// emits any of these. Gemma 4 IT publishes three: `<eos>` (1),
    /// `<turn|>` (106), and id 50; pass all three so generation stops
    /// at the natural turn boundary instead of running to the cap.
    /// `max_new_tokens`: hard cap on emit count.
    pub async fn generate_greedy(
        &self,
        prompt_text: &str,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<String, QuantizedPipelineError> {
        self.generate_greedy_streaming(prompt_text, max_new_tokens, eos_token_ids, |_, _| {})
            .await
    }

    /// Streaming generate — `on_delta(token_id, &decoded_delta)` runs
    /// after each new token. Same shape as `Gemma4MultiModal::generate_greedy_streaming`
    /// so wasm streaming bridges can plug in either pipeline behind
    /// a single trait if a future caller wants polymorphism.
    pub async fn generate_greedy_streaming(
        &self,
        prompt_text: &str,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
        mut on_delta: impl FnMut(u32, &str),
    ) -> Result<String, QuantizedPipelineError> {
        let enc = self
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| QuantizedPipelineError::Tokenizer(e.to_string()))?;
        let mut input_ids: Vec<u32> = enc.get_ids().to_vec();
        let prompt_len = input_ids.len();

        let mut model = self
            .model
            .lock()
            .map_err(|_| QuantizedPipelineError::MutexPoisoned)?;
        // KvCache lives across the generation loop; reset so we don't
        // mix runs.
        model.clear_kv_cache();

        // Prefill: feed the entire prompt in one forward, take the
        // last-token logits.
        let prompt_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = model.forward(&prompt_tensor, 0)?;
        let mut emitted_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut decoded = String::new();
        let mut prev_decoded_len = 0usize;

        for step in 0..max_new_tokens {
            let next_id = argmax_last(&logits)?;
            emitted_ids.push(next_id);

            // Decode the cumulative emitted tokens to compute the
            // delta. Tokenizers' decode handles BPE merges across
            // tokens so we can't just decode one id at a time.
            let cumulative = self
                .tokenizer
                .decode(&emitted_ids, true)
                .map_err(|e| QuantizedPipelineError::Decode(e.to_string()))?;
            let delta = &cumulative[prev_decoded_len..];
            on_delta(next_id, delta);
            decoded = cumulative.clone();
            prev_decoded_len = decoded.len();

            if eos_token_ids.contains(&next_id) {
                break;
            }

            // Decode step: feed only the new token, with seqlen_offset
            // pointing past the prompt + previously-emitted tokens.
            let seqlen_offset = prompt_len + step;
            input_ids.push(next_id);
            let one = Tensor::new(&[next_id], &self.device)?.unsqueeze(0)?;
            logits = model.forward(&one, seqlen_offset)?;
        }
        Ok(decoded)
    }
}

/// Argmax over the last token's logits. `logits` is `[B, T, vocab_size]`;
/// result is the highest-scoring vocab id at position `T-1`.
fn argmax_last(logits: &Tensor) -> Result<u32, QuantizedPipelineError> {
    let last = logits.i((.., logits.dim(1)? - 1, ..))?.squeeze(0)?;
    let vec = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let (id, _) = vec
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
            if v > acc.1 { (i, v) } else { acc }
        });
    Ok(id as u32)
}
