//! Pure-Rust f32 oracle for Kokoro-82M TTS (StyleTTS2 acoustic model + ISTFTNet
//! vocoder). The parity reference for the eventual WGSL kernels — diffed against
//! the upstream `hexgrad/kokoro` PyTorch model (see `KOKORO_REFERENCE.md`).
//!
//! Reads weights from a converted GGUF (`scripts/convert-kokoro-gguf.py`) via the
//! existing [`Weights`] / [`GgufReader`] path, identical to the Gemma oracle.
#![allow(dead_code)]

pub mod bert;
pub mod convblocks;
pub mod decoder;
pub mod g2p;
pub mod generator;
pub mod gpu;
pub mod gpu_fast;
pub mod ops;
pub mod prosody;
pub mod source;
pub mod text_encoder;
pub mod voice_train;

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::gguf::GgufReader;
use crate::reference::weights::Weights;

/// Parsed `kokoro.*` GGUF metadata.
#[derive(Clone, Debug)]
pub struct KokoroConfig {
    pub n_token: usize,
    pub hidden_dim: usize,
    pub style_dim: usize,
    pub dim_in: usize,
    pub n_mels: usize,
    pub n_layer: usize,
    pub max_dur: usize,
    pub text_encoder_kernel_size: usize,
    pub context_length: usize,
    // PL-BERT (ALBERT)
    pub plbert_hidden: usize,
    pub plbert_heads: usize,
    pub plbert_layers: usize,
    pub plbert_inter: usize,
    // ISTFTNet
    pub gen_istft_n_fft: usize,
    pub gen_istft_hop: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_initial_channel: usize,
    /// phoneme -> id
    pub vocab: HashMap<String, i64>,
}

fn json_usize_vec(s: &str) -> Vec<usize> {
    let v: Vec<i64> = serde_json::from_str(s).unwrap_or_default();
    v.into_iter().map(|x| x as usize).collect()
}

impl KokoroConfig {
    pub fn from_gguf(r: &GgufReader) -> Result<Self> {
        let u = |k: &str| -> usize {
            r.get(k)
                .and_then(|v| v.as_u32())
                .map(|x| x as usize)
                .unwrap_or(0)
        };
        let s = |k: &str| -> String {
            r.get(k)
                .and_then(|v| v.as_str())
                .map(|x| x.to_string())
                .unwrap_or_default()
        };
        let dil: Vec<Vec<usize>> =
            serde_json::from_str::<Vec<Vec<i64>>>(&s("kokoro.resblock_dilation_sizes_json"))
                .unwrap_or_default()
                .into_iter()
                .map(|row| row.into_iter().map(|x| x as usize).collect())
                .collect();
        let vocab: HashMap<String, i64> =
            serde_json::from_str(&s("kokoro.vocab_json")).unwrap_or_default();
        Ok(Self {
            n_token: u("kokoro.n_token"),
            hidden_dim: u("kokoro.hidden_dim"),
            style_dim: u("kokoro.style_dim"),
            dim_in: u("kokoro.dim_in"),
            n_mels: u("kokoro.n_mels"),
            n_layer: u("kokoro.n_layer"),
            max_dur: u("kokoro.max_dur"),
            text_encoder_kernel_size: u("kokoro.text_encoder_kernel_size"),
            context_length: u("kokoro.context_length"),
            plbert_hidden: u("kokoro.plbert.hidden_size"),
            plbert_heads: u("kokoro.plbert.num_attention_heads"),
            plbert_layers: u("kokoro.plbert.num_hidden_layers"),
            plbert_inter: u("kokoro.plbert.intermediate_size"),
            gen_istft_n_fft: u("kokoro.gen_istft_n_fft"),
            gen_istft_hop: u("kokoro.gen_istft_hop_size"),
            upsample_rates: json_usize_vec(&s("kokoro.upsample_rates_json")),
            upsample_kernel_sizes: json_usize_vec(&s("kokoro.upsample_kernel_sizes_json")),
            resblock_kernel_sizes: json_usize_vec(&s("kokoro.resblock_kernel_sizes_json")),
            resblock_dilation_sizes: dil,
            upsample_initial_channel: u("kokoro.upsample_initial_channel"),
            vocab,
        })
    }
}

/// The Kokoro oracle: config + lazy GGUF-backed weights.
pub struct KokoroModel {
    pub cfg: KokoroConfig,
    pub w: Weights,
}

impl KokoroModel {
    pub fn new(reader: Arc<GgufReader>) -> Result<Self> {
        let cfg = KokoroConfig::from_gguf(&reader)?;
        Ok(Self {
            cfg,
            w: Weights::new(reader),
        })
    }

    /// Load+dequant a tensor to f32, panicking with the name on error (oracle convenience).
    pub(crate) fn t(&self, name: &str) -> Vec<f32> {
        self.w
            .load(name)
            .unwrap_or_else(|e| panic!("kokoro tensor {name}: {e:?}"))
    }

    /// Optional tensor — `None` if absent. Used for InstanceNorm affine params, which
    /// the checkpoint omits (StyleTTS2 trained AdaIN with affine=False → identity 1/0).
    pub(crate) fn t_opt(&self, name: &str) -> Option<Vec<f32>> {
        self.w.load_opt(name).ok().flatten()
    }

    /// Map a phoneme string to input_ids, wrapped with BOS/EOS (id 0), dropping OOV.
    pub fn phonemes_to_ids(&self, phonemes: &str) -> Vec<i64> {
        let mut ids = vec![0i64];
        for ch in phonemes.chars() {
            if let Some(&id) = self.cfg.vocab.get(&ch.to_string()) {
                ids.push(id);
            }
        }
        ids.push(0);
        ids
    }

    /// Voice/style vector `ref_s [256]` for `n_tokens`, selected from the voicepack
    /// `[510, 1, 256]` by token length (row = n_tokens - 1).
    pub fn load_voice(&self, voice_id: &str, n_tokens: usize) -> Vec<f32> {
        let row = 2 * self.cfg.style_dim; // 256
        let vp = self.t(&format!("k.voice.{voice_id}"));
        let r = (n_tokens - 1).min(vp.len() / row - 1);
        vp[r * row..(r + 1) * row].to_vec()
    }

    /// Full pipeline: phoneme string + voice id → 24 kHz waveform. Deterministic
    /// (zeroed source randomness). The single composed reference for the WGSL port.
    pub fn synthesize(&self, phonemes: &str, voice_id: &str) -> Vec<f32> {
        let ids = self.phonemes_to_ids(phonemes);
        let ref_s = self.load_voice(voice_id, ids.len());
        self.synthesize_ids(&ids, &ref_s)
    }

    /// Pipeline driven by explicit input_ids + voice vector (`ref_s[:128]`=timbre,
    /// `ref_s[128:]`=prosodic).
    pub fn synthesize_ids(&self, ids: &[i64], ref_s: &[f32]) -> Vec<f32> {
        let t = ids.len();
        let sd = self.cfg.style_dim;
        let (timbre, prosodic) = (&ref_s[..sd], &ref_s[sd..2 * sd]);
        let cat = self.cfg.hidden_dim + sd;

        let bert = self.bert(ids);
        let be = self.bert_encoder(&bert, t);
        let d = self.duration_encode(&be, t, prosodic);
        let (_logits, dur) = self.predict_duration(&d, t);
        let (en, f) = self.expand_by_dur_cm(&d, t, cat, &dur);
        let (f0, n) = self.f0_n(&en, f, prosodic);
        let t_en = self.text_encoder(ids);
        let (_de, x_dec, _f0d, _nd) = self.decoder_features(&t_en, &f0, &n, &dur, timbre);
        let (har, frames) = self.generator_source(&f0);
        self.generator(
            &x_dec,
            x_dec.len() / self.cfg.hidden_dim,
            &har,
            frames,
            timbre,
        )
    }
}
