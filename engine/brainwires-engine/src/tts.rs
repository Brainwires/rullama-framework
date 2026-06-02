//! JS-facing TTS surface: `KokoroTts` — load the Kokoro GGUF, set the G2P lexicon,
//! synthesize text → 24 kHz PCM on the GPU. Kokoro is small (~164 MB f16) so it loads
//! whole in-memory (no streaming); the PWA downloads + OPFS-caches the GGUF and passes
//! the bytes. Mirrors `api::Model`'s ctx/pipes pattern. The Voice tab consumes this.

use std::sync::Arc;

use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgufReader;
use crate::reference::kokoro::g2p::{g2p, Lexicon};
use crate::reference::kokoro::gpu_fast::WeightCache;
use crate::reference::kokoro::KokoroModel;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub const SAMPLE_RATE: u32 = 24000;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct KokoroTts {
    model: KokoroModel,
    ctx: WgpuCtx,
    pipes: Arc<Pipelines>,
    lex: Option<Lexicon>,
    /// Persistent GPU weight cache — warm synths skip re-dequant + re-upload.
    wc: WeightCache,
}

impl KokoroTts {
    /// Load from in-memory GGUF bytes + init a GPU context.
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = Arc::new(GgufReader::new(bytes)?);
        let model = KokoroModel::new(reader)?;
        let ctx = WgpuCtx::new().await?;
        let pipes = Arc::new(Pipelines::new(&ctx.device));
        Ok(Self { model, ctx, pipes, lex: None, wc: WeightCache::new() })
    }

    /// Provide the G2P lexicon (misaki us_gold + optional us_silver JSON bytes).
    pub fn set_lexicon_native(&mut self, gold: &[u8], silver: &[u8]) {
        self.lex = Some(Lexicon::load(gold, silver));
    }

    /// Text → PCM. Returns (pcm, oov words). Requires the lexicon to be set.
    /// Uses the buffer-chained, weight-cached fast path (warm after the first synth).
    pub async fn synthesize_native(&mut self, text: &str, voice: &str) -> (Vec<f32>, Vec<String>) {
        let (phonemes, oov) = {
            let lex = self.lex.as_ref().expect("lexicon not set");
            g2p(text, lex)
        };
        let audio = self.synthesize_phonemes_native(&phonemes, voice).await;
        (audio, oov)
    }

    /// Phoneme string → PCM (skips G2P).
    pub async fn synthesize_phonemes_native(&mut self, phonemes: &str, voice: &str) -> Vec<f32> {
        let ids = self.model.phonemes_to_ids(phonemes);
        let ref_s = self.model.load_voice(voice, ids.len());
        self.model.synthesize_gpu_fast(&self.ctx, &self.pipes, &mut self.wc, &ids, &ref_s).await
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl KokoroTts {
    /// Load from GGUF bytes (the PWA reads the OPFS-cached file and passes them here).
    #[wasm_bindgen(js_name = load)]
    pub async fn load_js(bytes: Vec<u8>) -> std::result::Result<KokoroTts, JsError> {
        Self::load_native(bytes).await.map_err(|e| JsError::new(&format!("{e:?}")))
    }

    /// Set the G2P lexicon from us_gold (+ optional us_silver) JSON bytes.
    #[wasm_bindgen(js_name = setLexicon)]
    pub fn set_lexicon_js(&mut self, gold: Vec<u8>, silver: Vec<u8>) {
        self.set_lexicon_native(&gold, &silver);
    }

    /// Synthesize text → Float32Array PCM (24 kHz mono).
    #[wasm_bindgen(js_name = synthesize)]
    pub async fn synthesize_js(&mut self, text: String, voice: String) -> Vec<f32> {
        self.synthesize_native(&text, &voice).await.0
    }

    /// Synthesize a phoneme string → Float32Array PCM (skips G2P).
    #[wasm_bindgen(js_name = synthesizePhonemes)]
    pub async fn synthesize_phonemes_js(&mut self, phonemes: String, voice: String) -> Vec<f32> {
        self.synthesize_phonemes_native(&phonemes, &voice).await
    }

    #[wasm_bindgen(js_name = sampleRate, getter)]
    pub fn sample_rate_js(&self) -> u32 {
        SAMPLE_RATE
    }
}
