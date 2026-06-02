//! Pure-Rust f32 oracle for Kokoro-82M TTS (StyleTTS2 acoustic model + ISTFTNet
//! vocoder). The parity reference for the eventual WGSL kernels — diffed against
//! the upstream `hexgrad/kokoro` PyTorch model (see `KOKORO_REFERENCE.md`).
//!
//! Reads weights from a converted GGUF (`scripts/convert-kokoro-gguf.py`) via the
//! existing [`Weights`] / [`GgufReader`] path, identical to the Gemma oracle.
#![allow(dead_code)]

pub mod bert;
pub mod ops;
pub mod prosody;

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
        let u = |k: &str| -> usize { r.get(k).and_then(|v| v.as_u32()).map(|x| x as usize).unwrap_or(0) };
        let s = |k: &str| -> String { r.get(k).and_then(|v| v.as_str()).map(|x| x.to_string()).unwrap_or_default() };
        let dil: Vec<Vec<usize>> = serde_json::from_str::<Vec<Vec<i64>>>(&s("kokoro.resblock_dilation_sizes_json"))
            .unwrap_or_default()
            .into_iter()
            .map(|row| row.into_iter().map(|x| x as usize).collect())
            .collect();
        let vocab: HashMap<String, i64> = serde_json::from_str(&s("kokoro.vocab_json")).unwrap_or_default();
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
        Ok(Self { cfg, w: Weights::new(reader) })
    }

    /// Load+dequant a tensor to f32, panicking with the name on error (oracle convenience).
    pub(crate) fn t(&self, name: &str) -> Vec<f32> {
        self.w.load(name).unwrap_or_else(|e| panic!("kokoro tensor {name}: {e:?}"))
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
}
