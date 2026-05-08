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

/// Best-effort diagnostic log. On wasm32 forwards to `console.log`; on
/// native, writes to stderr. Mirrors the BF16 `gemma4_mm` pipeline so
/// the chat-pwa shows the same `[gemma4/diag]` and `[gemma4/perf]`
/// signal regardless of which weight path the user picked.
fn diag_log(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{msg}");
}

/// Cross-platform millisecond timestamp for perf diags.
fn perf_now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

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

        let preview_n = prompt_len.min(40);
        diag_log(&format!(
            "[gemma4/diag] prompt encoded: len={prompt_len} \
             ids[..{preview_n}]={:?} \
             tokens[..{preview_n}]={:?}",
            &input_ids[..preview_n],
            enc.get_tokens()
                .iter()
                .take(preview_n)
                .collect::<Vec<_>>(),
        ));

        let mut model = self
            .model
            .lock()
            .map_err(|_| QuantizedPipelineError::MutexPoisoned)?;
        // KvCache lives across the generation loop; reset so we don't
        // mix runs.
        model.clear_kv_cache();

        // Prefill: feed the entire prompt in one forward, take the
        // last-token logits.
        //
        // input_ids is only used for index lookups inside the model —
        // never as a numeric input to a kernel — so we keep it on CPU
        // regardless of `self.device`. Required on wasm32 + WebGPU,
        // where sync GPU→CPU readback is forbidden (the embed table
        // and PLE both live on CPU and consume the ids there).
        let t_prefill_start = perf_now_ms();
        let prompt_tensor = Tensor::new(input_ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
        let mut logits = model.forward(&prompt_tensor, 0)?;
        let t_prefill_done = perf_now_ms();
        diag_log(&format!(
            "[gemma4/perf] prefill forward: {:.1}ms (prompt_len={prompt_len})",
            t_prefill_done - t_prefill_start,
        ));

        let mut emitted_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut decoded = String::new();
        let mut prev_decoded_len = 0usize;

        for step in 0..max_new_tokens {
            let t_step_start = perf_now_ms();
            // Top-5 logit dump for the first 4 steps so we can side-by-side
            // diff browser vs native gemma4_diag --device wgpu output and
            // pin down the kernel that diverges between Naga (native) and
            // Tint/Dawn (browser) on the same Metal hardware.
            if step < 4 {
                let last = logits
                    .i((.., logits.dim(1)? - 1, ..))?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?;
                if let Ok(vals) = last.to_vec1_async::<f32>().await {
                    let mut indexed: Vec<(usize, f32)> = vals.into_iter().enumerate().collect();
                    indexed.sort_unstable_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
                    diag_log(&format!(
                        "[gemma4/logits] step {step}: top5={top5:?}",
                    ));
                }
            }
            let next_id = argmax_last(&logits).await?;
            let t_argmax = perf_now_ms();
            emitted_ids.push(next_id);

            // Decode the cumulative emitted tokens to compute the
            // delta. Tokenizers' decode handles BPE merges across
            // tokens so we can't just decode one id at a time.
            let cumulative = self
                .tokenizer
                .decode(&emitted_ids, true)
                .map_err(|e| QuantizedPipelineError::Decode(e.to_string()))?;
            let delta = &cumulative[prev_decoded_len..];
            // Surface the actual decoded text for the first few steps
            // — the chat UI may swallow whitespace-only deltas.
            if step < 16 {
                diag_log(&format!(
                    "[gemma4/text] step {step}: id={next_id} delta={:?} cum={:?}",
                    delta,
                    if cumulative.len() > 80 {
                        format!("{}…", &cumulative[..80])
                    } else {
                        cumulative.clone()
                    },
                ));
            }
            on_delta(next_id, delta);
            decoded = cumulative.clone();
            prev_decoded_len = decoded.len();

            if eos_token_ids.contains(&next_id) {
                diag_log(&format!(
                    "[gemma4/diag] step {step}: EOS {next_id} — generation terminated",
                ));
                break;
            }

            // Soft-EOS on repetition. Greedy decoding can wander into
            // an emit loop, particularly when upstream logits drift
            // and the natural EOS no longer wins the argmax. We
            // detect three patterns and treat any as EOS so the
            // pipeline stops cleanly instead of burning through
            // max_new_tokens emitting garbage.
            //
            // Pattern A — same token N times consecutively. Catches
            // the emoji-spam attractor ("😊😊😊…").
            //
            // Pattern B — exact-cycle repetition: same k-gram repeats
            // back-to-back. Catches " a b a b a b" and
            // " a b c a b c a b c". 3 full cycles is the trigger.
            //
            // Pattern C — long-window n-gram repeat: any 4-gram
            // appearing 2+ times in the last 24 emitted tokens.
            // Catches near-cycles like "How do you feel? How do you
            // feel today?" where the second occurrence is preceded
            // by a separator. Single recurrence is permitted (a
            // model that emits "I I" or "the the" in passing isn't
            // looping).
            const REPEAT_RUN: usize = 4;
            const REPEAT_NGRAM_CYCLES: usize = 3;
            const NGRAM_K: usize = 4;
            const NGRAM_WINDOW: usize = 24;
            let n = emitted_ids.len();
            let same_run = n >= REPEAT_RUN
                && emitted_ids[n - REPEAT_RUN..].iter().all(|&id| id == next_id);
            let exact_cycle = |k: usize| -> bool {
                let need = k * REPEAT_NGRAM_CYCLES;
                if n < need {
                    return false;
                }
                let tail = &emitted_ids[n - need..];
                let first = &tail[..k];
                tail.chunks_exact(k).all(|chunk| chunk == first)
            };
            let window_repeat = || -> bool {
                if n < NGRAM_K * 2 {
                    return false;
                }
                let win_start = n.saturating_sub(NGRAM_WINDOW);
                let tail = &emitted_ids[win_start..];
                if tail.len() < NGRAM_K * 2 {
                    return false;
                }
                let needle = &tail[tail.len() - NGRAM_K..];
                tail[..tail.len() - NGRAM_K]
                    .windows(NGRAM_K)
                    .any(|w| w == needle)
            };
            if same_run || exact_cycle(2) || exact_cycle(3) || window_repeat() {
                let kind = if same_run {
                    format!("{REPEAT_RUN}× same token id={next_id}")
                } else if exact_cycle(2) {
                    format!("2-gram cycled {REPEAT_NGRAM_CYCLES}×")
                } else if exact_cycle(3) {
                    format!("3-gram cycled {REPEAT_NGRAM_CYCLES}×")
                } else {
                    format!("{NGRAM_K}-gram repeated within last {NGRAM_WINDOW}")
                };
                diag_log(&format!(
                    "[gemma4/diag] step {step}: repetition detected ({kind}) — soft-EOS",
                ));
                break;
            }

            // Decode step: feed only the new token, with seqlen_offset
            // pointing past the prompt + previously-emitted tokens.
            let seqlen_offset = prompt_len + step;
            input_ids.push(next_id);
            let one = Tensor::new(&[next_id], &Device::Cpu)?.unsqueeze(0)?;
            let t_forward_start = perf_now_ms();
            logits = model.forward(&one, seqlen_offset)?;
            let t_forward_done = perf_now_ms();

            // Per-token timing for the first 5 steps so we can profile
            // what's slow without spamming logs forever.
            if step < 5 {
                diag_log(&format!(
                    "[gemma4/perf] step {step}: total={:.1}ms \
                     [argmax={:.1} forward={:.1}] next_id={next_id}",
                    t_forward_done - t_step_start,
                    t_argmax - t_step_start,
                    t_forward_done - t_forward_start,
                ));
            }
        }
        Ok(decoded)
    }
}

/// Argmax over the last token's logits. `logits` is `[B, T, vocab_size]`;
/// result is the highest-scoring vocab id at position `T-1`.
///
/// Runs argmax on-device (so we only read back a single u32) and uses
/// `to_vec0_async` for the readback. Required on wasm32 + WebGPU,
/// where sync GPU→CPU copy panics — `mapAsync` is the only readback
/// path. On native CPU the async path is a thin wrapper over the sync
/// one, no observable cost.
async fn argmax_last(logits: &Tensor) -> Result<u32, QuantizedPipelineError> {
    let last = logits.i((.., logits.dim(1)? - 1, ..))?.squeeze(0)?;
    // Promote to F32 before argmax — keeps numerical ordering stable
    // when logits arrive as BF16 / F16 (some tied scores otherwise
    // collapse to the lower-id token).
    let last = last.to_dtype(DType::F32)?;
    let arg = last.argmax(0)?;
    let id: u32 = arg.to_vec0_async::<u32>().await?;
    Ok(id)
}
