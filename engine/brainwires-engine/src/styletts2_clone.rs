//! JS-facing StyleTTS2 voice-cloning surface — `StyleTtsClone`. **Desktop only.**
//!
//! Loads the StyleTTS2-LibriTTS GGUF whole in-memory (like `KokoroTts`; cloning is not
//! shipped to the iPhone text-only path), then:
//!   encodeVoice(refPcm24k) → 256-d voice vector
//!   synthesize(text, voice) → 24 kHz PCM in that voice
//! Runs the CPU f32 oracle (validated corr 0.99997 vs the reference). A WGSL/GPU port is
//! a follow-up perf task; synthesis is one-shot so CPU is acceptable for v1 desktop.

use std::collections::HashMap;

use crate::error::Result;
use crate::gguf::GgufReader;
use crate::reference::kokoro::g2p::{g2p, Lexicon};
use crate::reference::styletts2::StyleTtsModel;

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
}

impl StyleTtsClone {
    /// Load from in-memory GGUF bytes (the PWA reads the OPFS-cached file and passes them).
    pub fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = GgufReader::new(bytes)?;
        let model = StyleTtsModel::load(&reader)?; // dequant into the weight map; reader dropped after
        let vocab = VOCAB.chars().enumerate().map(|(i, c)| (c, i as i64)).collect();
        Ok(Self { model, lex: None, vocab })
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

    /// Reference 24 kHz mono PCM → 256-d voice vector.
    pub fn encode_voice_native(&self, pcm24k: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        self.model.encode_voice(pcm24k, progress)
    }

    /// Text + voice vector → 24 kHz PCM. Requires the lexicon to be set.
    pub fn synthesize_native(&self, text: &str, voice: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        let (ps, _oov) = {
            let lex = self.lex.as_ref().expect("lexicon not set");
            g2p(text, lex)
        };
        self.model.synthesize(&self.phonemes_to_ids(&ps), voice, progress)
    }

    pub fn synthesize_phonemes_native(&self, phonemes: &str, voice: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        self.model.synthesize(&self.phonemes_to_ids(phonemes), voice, progress)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl StyleTtsClone {
    /// Load from GGUF bytes (StyleTTS2-LibriTTS f32).
    #[wasm_bindgen(js_name = load)]
    pub fn load_js(bytes: Vec<u8>) -> std::result::Result<StyleTtsClone, JsError> {
        Self::load_native(bytes).map_err(|e| JsError::new(&format!("{e:?}")))
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
    pub fn encode_voice_js(&self, pcm24k: Vec<f32>, on_progress: &js_sys::Function) -> Vec<f32> {
        let cb = |frac: f32, stage: &str| {
            let _ = on_progress.call2(&JsValue::NULL, &JsValue::from_f64(frac as f64), &JsValue::from_str(stage));
        };
        self.encode_voice_native(&pcm24k, Some(&cb))
    }

    /// Synthesize text in a voice → 24 kHz PCM (Float32Array). `onProgress(fraction, stage)`.
    #[wasm_bindgen(js_name = synthesize)]
    pub fn synthesize_js(&self, text: String, voice: Vec<f32>, on_progress: &js_sys::Function) -> Vec<f32> {
        let cb = |frac: f32, stage: &str| {
            let _ = on_progress.call2(&JsValue::NULL, &JsValue::from_f64(frac as f64), &JsValue::from_str(stage));
        };
        self.synthesize_native(&text, &voice, Some(&cb))
    }

    /// Synthesize a phoneme string in a voice → 24 kHz PCM (skips G2P).
    #[wasm_bindgen(js_name = synthesizePhonemes)]
    pub fn synthesize_phonemes_js(&self, phonemes: String, voice: Vec<f32>) -> Vec<f32> {
        self.synthesize_phonemes_native(&phonemes, &voice, None)
    }

    #[wasm_bindgen(js_name = sampleRate, getter)]
    pub fn sample_rate_js(&self) -> u32 {
        SAMPLE_RATE
    }
}
