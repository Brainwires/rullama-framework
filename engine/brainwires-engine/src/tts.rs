//! JS-facing TTS surface: `KokoroTts` — load the Kokoro GGUF, set the G2P lexicon,
//! synthesize text → 24 kHz PCM on the GPU. Kokoro is small (~164 MB f16) so it loads
//! whole in-memory (no streaming); the PWA downloads + OPFS-caches the GGUF and passes
//! the bytes. Mirrors `api::Model`'s ctx/pipes pattern. The Voice tab consumes this.

use std::sync::Arc;

use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgufReader;
use crate::reference::kokoro::KokoroModel;
use crate::reference::kokoro::g2p::{Lexicon, g2p};
use crate::reference::kokoro::gpu_fast::WeightCache;
use crate::reference::kokoro::voice_train::voice_signature;

fn sig_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len().max(1) as f32
}

/// In-progress gradient-free voice-training run.
struct VoiceTrainState {
    target_sig: Vec<f32>,
    ids: Vec<i64>,
    style: Vec<f32>,
    loss: f32,
    step: f32,
    rng: u64,
}

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
    /// Active voice-training run (gradient-free style optimization).
    train: Option<VoiceTrainState>,
}

impl KokoroTts {
    /// Load from in-memory GGUF bytes + init a GPU context.
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = Arc::new(GgufReader::new(bytes)?);
        let model = KokoroModel::new(reader)?;
        let ctx = WgpuCtx::new().await?;
        let pipes = Arc::new(Pipelines::new(&ctx.device));
        Ok(Self {
            model,
            ctx,
            pipes,
            lex: None,
            wc: WeightCache::new(),
            train: None,
        })
    }

    /// Begin a voice-training run: target speaker PCM (resampled to 24 kHz by the
    /// caller) + a reference text the model re-synthesizes each step (G2P'd internally).
    pub async fn train_begin_native(
        &mut self,
        target_pcm: &[f32],
        ref_text: &str,
        init_voice: &str,
        step0: f32,
        seed: u64,
    ) {
        let target_sig = voice_signature(target_pcm);
        let phonemes = {
            let lex = self.lex.as_ref().expect("lexicon not set");
            g2p(ref_text, lex).0
        };
        let ids = self.model.phonemes_to_ids(&phonemes);
        let style = self.model.load_voice(init_voice, ids.len());
        let audio = self
            .model
            .synthesize_gpu_fast(&self.ctx, &self.pipes, &mut self.wc, &ids, &style)
            .await;
        let loss = sig_l2(&voice_signature(&audio), &target_sig);
        self.train = Some(VoiceTrainState {
            target_sig,
            ids,
            style,
            loss,
            step: step0,
            rng: seed | 1,
        });
    }

    /// One training step. Returns the current best loss.
    pub async fn train_step_native(&mut self) -> f32 {
        let st = self.train.as_mut().expect("train_begin first");
        self.model
            .voice_train_step(
                &self.ctx,
                &self.pipes,
                &mut self.wc,
                &st.ids,
                &st.target_sig,
                &mut st.style,
                &mut st.loss,
                &mut st.step,
                &mut st.rng,
            )
            .await
    }

    /// The current best voice vector (256-d), to save/reuse as a custom voicepack.
    pub fn trained_voice_native(&self) -> Vec<f32> {
        self.train
            .as_ref()
            .map(|s| s.style.clone())
            .unwrap_or_default()
    }

    /// Synthesize text with an explicit voice vector (a trained/custom voicepack).
    pub async fn synthesize_text_style_native(&mut self, text: &str, style: &[f32]) -> Vec<f32> {
        let phonemes = {
            let lex = self.lex.as_ref().expect("lexicon not set");
            g2p(text, lex).0
        };
        let ids = self.model.phonemes_to_ids(&phonemes);
        self.model
            .synthesize_gpu_fast(&self.ctx, &self.pipes, &mut self.wc, &ids, style)
            .await
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
        self.model
            .synthesize_gpu_fast(&self.ctx, &self.pipes, &mut self.wc, &ids, &ref_s)
            .await
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl KokoroTts {
    /// Load from GGUF bytes (the PWA reads the OPFS-cached file and passes them here).
    #[wasm_bindgen(js_name = load)]
    pub async fn load_js(bytes: Vec<u8>) -> std::result::Result<KokoroTts, JsError> {
        Self::load_native(bytes)
            .await
            .map_err(|e| JsError::new(&format!("{e:?}")))
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

    // ---- gradient-free voice training ----

    /// Begin training toward a target speaker clip (24 kHz mono PCM). `refPhonemes`
    /// is the phoneme string the model re-synthesizes each step; `initVoice` seeds it.
    #[wasm_bindgen(js_name = trainBegin)]
    pub async fn train_begin_js(
        &mut self,
        target_pcm: Vec<f32>,
        ref_text: String,
        init_voice: String,
    ) {
        self.train_begin_native(&target_pcm, &ref_text, &init_voice, 0.06, 1)
            .await;
    }

    /// Run one training step; returns the current best loss (call in a JS loop with a stop flag).
    #[wasm_bindgen(js_name = trainStep)]
    pub async fn train_step_js(&mut self) -> f32 {
        self.train_step_native().await
    }

    /// The current trained voice vector (Float32Array, 256-d).
    #[wasm_bindgen(js_name = trainedVoice)]
    pub fn trained_voice_js(&self) -> Vec<f32> {
        self.trained_voice_native()
    }

    /// Synthesize text with a trained/custom voice vector → Float32Array PCM.
    #[wasm_bindgen(js_name = synthesizeWithVoice)]
    pub async fn synthesize_with_voice_js(&mut self, text: String, voice: Vec<f32>) -> Vec<f32> {
        self.synthesize_text_style_native(&text, &voice).await
    }
}
