//! GGUF-backed StyleTTS2 cloning model — the path the wasm/browser engine uses.
//!
//! Loads every tensor from the GGUF (f16→f32 dequant) into a name→weights map, remapping
//! the PyTorch state-dict names to what the validated oracle structs expect (style
//! encoders → "acoustic"/"prosodic", decoder prefix stripped). Then:
//!   encode_voice(ref_pcm) → 256-d voice vector   (MelFrontend + StyleEncoder ×2)
//!   synthesize(ids, voice) → 24 kHz waveform      (StyleTtsAcoustic + hifigan decoder)
#![allow(dead_code)]

use std::collections::HashMap;

use super::{MelFrontend, StyleEncoder, StyleTtsAcoustic};
use crate::error::Result;
use crate::gguf::tensor::dequant_tensor_to_f32;
use crate::gguf::GgufReader;

pub struct StyleTtsModel {
    w: HashMap<String, Vec<f32>>,
}

/// Map a checkpoint tensor name to the oracle-struct name.
fn remap(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("style_encoder.") {
        return format!("acoustic.{}", remap_style(rest));
    }
    if let Some(rest) = name.strip_prefix("predictor_encoder.") {
        return format!("prosodic.{}", remap_style(rest));
    }
    if let Some(rest) = name.strip_prefix("decoder.") {
        return rest.to_string(); // StyleTtsDecoder reads decoder weights unprefixed
    }
    name.to_string() // text_encoder.* / bert.* / bert_encoder.* / predictor.* / mel.* as-is
}

/// StyleEncoder: `shared.0`→conv0, `shared.{1+i}`→blk{i} (downsample_res.conv→down,
/// conv1x1→sc), `shared.6`→conv_out, `unshared`→linear.
fn remap_style(rest: &str) -> String {
    if let Some(r) = rest.strip_prefix("unshared.") {
        return format!("linear.{r}");
    }
    if let Some(r) = rest.strip_prefix("shared.") {
        let (idx, tail) = r.split_once('.').unwrap_or((r, ""));
        match idx.parse::<usize>().unwrap_or(99) {
            0 => format!("conv0.{tail}"),
            6 => format!("conv_out.{tail}"),
            i @ 1..=4 => {
                let t2 = if let Some(x) = tail.strip_prefix("downsample_res.conv.") {
                    format!("down.{x}")
                } else if let Some(x) = tail.strip_prefix("conv1x1.") {
                    format!("sc.{x}")
                } else {
                    tail.to_string() // conv1.* / conv2.*
                };
                format!("blk{}.{t2}", i - 1)
            }
            _ => rest.to_string(),
        }
    } else {
        rest.to_string()
    }
}

impl StyleTtsModel {
    /// Load + dequant every tensor from the GGUF into the remapped weight map.
    pub fn load(reader: &GgufReader) -> Result<Self> {
        let mut w = HashMap::new();
        for td in reader.tensors() {
            let data = dequant_tensor_to_f32(reader, &td.name)?;
            w.insert(remap(&td.name), data);
        }
        Ok(Self { w })
    }

    /// Reference 24 kHz mono PCM → 256-d voice vector (acoustic ‖ prosodic).
    pub fn encode_voice(&self, pcm24k: &[f32]) -> Vec<f32> {
        let front = MelFrontend::new(self.w.get("mel.window").expect("mel.window"), self.w.get("mel.filterbank").expect("mel.filterbank"));
        let (mel, t) = front.compute(pcm24k);
        let a = StyleEncoder::from_weights(&self.w, "acoustic").forward(&mel, 80, t);
        let p = StyleEncoder::from_weights(&self.w, "prosodic").forward(&mel, 80, t);
        a.into_iter().chain(p).collect()
    }

    /// Token ids + voice vector → 24 kHz waveform.
    pub fn synthesize(&self, ids: &[i64], voice: &[f32]) -> Vec<f32> {
        StyleTtsAcoustic::new(&self.w).synthesize(ids, voice)
    }
}
