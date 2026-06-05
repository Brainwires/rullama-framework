//! JS-facing StyleTTS2 voice-cloning surface — `StyleTtsClone`. **Desktop only.**
//!
//! Loads the StyleTTS2-LibriTTS GGUF whole in-memory (like `KokoroTts`; cloning is not
//! shipped to the iPhone text-only path), then:
//!   encodeVoice(refPcm24k) → 256-d voice vector
//!   synthesize(text, voice) → 24 kHz PCM in that voice
//! Runs the CPU f32 oracle (validated corr 0.99997 vs the reference). A WGSL/GPU port is
//! a follow-up perf task; synthesis is one-shot so CPU is acceptable for v1 desktop.

use std::collections::HashMap;

use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgufReader;
use crate::reference::kokoro::g2p::{Lexicon, g2p};
use crate::reference::styletts2::StyleTtsModel;
use crate::reference::styletts2::acoustic::DiffusionConfig;
use crate::reference::styletts2::gpu::GpuWeightCache;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub const SAMPLE_RATE: u32 = 24000;

/// 178-symbol StyleTTS2/Kokoro phoneme vocab, index == position ('~' at the unused gap 174).
const VOCAB: &str = include_str!("reference/styletts2/vocab.txt");

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct StyleTtsClone {
    model: StyleTtsModel,
    lex: Option<Lexicon>,
    vocab: HashMap<char, i64>,
    ctx: WgpuCtx,
    pipes: Pipelines,
    wc: GpuWeightCache,
}

impl StyleTtsClone {
    /// Init the GPU context + lexicon scaffolding around an already-loaded model. Shared by the
    /// in-memory (`load_native`) and streaming (`load_streaming`) entry points.
    async fn from_model(model: StyleTtsModel) -> Result<Self> {
        let vocab = VOCAB
            .chars()
            .enumerate()
            .map(|(i, c)| (c, i as i64))
            .collect();
        let ctx = WgpuCtx::new().await?;
        let pipes = Pipelines::new(&ctx.device);
        Ok(Self {
            model,
            lex: None,
            vocab,
            ctx,
            pipes,
            wc: GpuWeightCache::new(),
        })
    }

    /// Load from in-memory GGUF bytes + init a GPU context (synthesis runs on the GPU).
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = GgufReader::new(bytes)?;
        let model = StyleTtsModel::load(&reader)?; // dequant into the weight map; reader dropped after
        Self::from_model(model).await
    }

    /// Streaming load from any [`crate::gguf::TensorFetcher`] — the wasm/iPhone path. Fetches the
    /// model one tensor at a time (range reads) so the whole GGUF never lands in linear memory.
    /// See [`StyleTtsModel::load_streaming`] for the jetsam rationale.
    pub async fn load_streaming(reader: &GgufReader) -> Result<Self> {
        let model = StyleTtsModel::load_streaming(reader).await?;
        Self::from_model(model).await
    }

    pub fn set_lexicon_native(&mut self, gold: &[u8], silver: &[u8]) {
        self.lex = Some(Lexicon::load(gold, silver));
    }

    /// Phoneme string → token ids, BOS(0)-prefixed, dropping OOV symbols (matches TextCleaner).
    fn phonemes_to_ids(&self, ps: &str) -> Vec<i64> {
        let mut ids = vec![0i64];
        for ch in ps.chars() {
            if let Some(&id) = self.vocab.get(&ch) {
                ids.push(id);
            }
        }
        ids
    }

    /// Reference 24 kHz mono PCM → 256-d voice vector (GPU encoder).
    pub async fn encode_voice_native(
        &mut self,
        pcm24k: &[f32],
        progress: Option<&dyn Fn(f32, &str)>,
    ) -> Vec<f32> {
        self.model
            .encode_voice_gpu(&self.ctx, &self.pipes, &mut self.wc, pcm24k, progress)
            .await
    }

    /// Text + voice vector → 24 kHz PCM (GPU decoder). Requires the lexicon to be set.
    /// Uses the style-diffusion prosody path (alpha=0.3/beta=0.7) by default.
    pub async fn synthesize_native(
        &mut self,
        text: &str,
        voice: &[f32],
        progress: Option<&dyn Fn(f32, &str)>,
    ) -> Vec<f32> {
        let ids = {
            let lex = self.lex.as_ref().expect("lexicon not set");
            let (ps, _oov) = g2p(text, lex);
            self.phonemes_to_ids(&ps)
        };
        self.model
            .synthesize_gpu(
                &self.ctx,
                &self.pipes,
                &mut self.wc,
                &ids,
                voice,
                Some(DiffusionConfig::default()),
                progress,
            )
            .await
    }

    pub async fn synthesize_phonemes_native(
        &mut self,
        phonemes: &str,
        voice: &[f32],
        progress: Option<&dyn Fn(f32, &str)>,
    ) -> Vec<f32> {
        let ids = self.phonemes_to_ids(phonemes);
        self.model
            .synthesize_gpu(
                &self.ctx,
                &self.pipes,
                &mut self.wc,
                &ids,
                voice,
                Some(DiffusionConfig::default()),
                progress,
            )
            .await
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl StyleTtsClone {
    /// Load from GGUF bytes (StyleTTS2-LibriTTS f32) + init the GPU.
    #[wasm_bindgen(js_name = load)]
    pub async fn load_js(bytes: Vec<u8>) -> std::result::Result<StyleTtsClone, JsError> {
        Self::load_native(bytes)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Streaming load over an OPFS file (the iPhone-safe path). `read_fn(offset, len) ->
    /// Uint8Array` is a JS callback backed by a `FileSystemSyncAccessHandle`; weights are pulled
    /// one tensor at a time so the 543 MB GGUF never enters wasm linear memory in bulk. Mirrors
    /// `Model.loadFromOpfs`. See [`StyleTtsModel::load_streaming`] for the jetsam rationale.
    #[wasm_bindgen(js_name = loadStreaming)]
    pub async fn load_streaming_js(
        read_fn: js_sys::Function,
        total_bytes: f64,
    ) -> std::result::Result<StyleTtsClone, JsError> {
        use crate::gguf::{OpfsFetcher, TensorFetcher};
        use std::sync::Arc;
        if !(total_bytes.is_finite() && total_bytes >= 0.0) {
            return Err(JsError::new(
                "loadStreaming: total_bytes must be a non-negative finite number",
            ));
        }
        let fetcher: Arc<dyn TensorFetcher> =
            Arc::new(OpfsFetcher::new(read_fn, total_bytes as u64));
        let reader = GgufReader::new_streaming(fetcher)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))?;
        Self::load_streaming(&reader)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Set the G2P lexicon (misaki us_gold + optional us_silver JSON bytes).
    #[wasm_bindgen(js_name = setLexicon)]
    pub fn set_lexicon_js(&mut self, gold: Vec<u8>, silver: Vec<u8>) {
        self.set_lexicon_native(&gold, &silver);
    }

    /// Reference clip (24 kHz mono Float32) → 256-d voice vector (Float32Array).
    /// `onProgress(fraction, stage)` is called synchronously at each stage — the worker
    /// posts these out mid-computation so the UI gets a live progress bar + log.
    #[wasm_bindgen(js_name = encodeVoice)]
    pub async fn encode_voice_js(
        &mut self,
        pcm24k: Vec<f32>,
        on_progress: js_sys::Function,
    ) -> Vec<f32> {
        let cb = |frac: f32, stage: &str| {
            let _ = on_progress.call2(
                &JsValue::NULL,
                &JsValue::from_f64(frac as f64),
                &JsValue::from_str(stage),
            );
        };
        self.encode_voice_native(&pcm24k, Some(&cb)).await
    }

    /// Synthesize text in a voice → 24 kHz PCM (Float32Array). `onProgress(fraction, stage)`.
    #[wasm_bindgen(js_name = synthesize)]
    pub async fn synthesize_js(
        &mut self,
        text: String,
        voice: Vec<f32>,
        on_progress: js_sys::Function,
    ) -> Vec<f32> {
        let cb = |frac: f32, stage: &str| {
            let _ = on_progress.call2(
                &JsValue::NULL,
                &JsValue::from_f64(frac as f64),
                &JsValue::from_str(stage),
            );
        };
        self.synthesize_native(&text, &voice, Some(&cb)).await
    }

    /// Synthesize a phoneme string in a voice → 24 kHz PCM (skips G2P).
    #[wasm_bindgen(js_name = synthesizePhonemes)]
    pub async fn synthesize_phonemes_js(&mut self, phonemes: String, voice: Vec<f32>) -> Vec<f32> {
        self.synthesize_phonemes_native(&phonemes, &voice, None)
            .await
    }

    #[wasm_bindgen(js_name = sampleRate, getter)]
    pub fn sample_rate_js(&self) -> u32 {
        SAMPLE_RATE
    }
}
