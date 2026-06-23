//! GGUF-backed StyleTTS2 cloning model — the path the wasm/browser engine uses.
//!
//! Loads every tensor from the GGUF (f16→f32 dequant) into a name→weights map, remapping
//! the PyTorch state-dict names to what the validated oracle structs expect (style
//! encoders → "acoustic"/"prosodic", decoder prefix stripped). Then:
//!   encode_voice(ref_pcm) → 256-d voice vector   (MelFrontend + StyleEncoder ×2)
//!   synthesize(ids, voice) → 24 kHz waveform      (StyleTtsAcoustic + hifigan decoder)
#![allow(dead_code)]

use std::collections::HashMap;

use super::acoustic::DiffusionConfig;
use super::gpu::{GpuWeightCache, StyleTtsGpu};
use super::{MelFrontend, StyleEncoder, StyleTtsAcoustic};
use crate::backend::{Pipelines, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgmlDtype;
use crate::gguf::GgufReader;
use crate::gguf::tensor::{
    dequant_tensor_to_f16_async, dequant_tensor_to_f32, dequant_tensor_to_f32_async,
};

pub struct StyleTtsModel {
    /// f32 weights: everything in the f32 variant; small weights + linears in
    /// the f16 variant (linears are consumed on the CPU via `t()`).
    w: HashMap<String, Vec<f32>>,
    /// f16 weights (raw bits) for the memory-tight variant: the 3-D/4-D conv
    /// tensors — the bulk of the model — routed to the f16 conv GPU kernels.
    /// Empty for the f32 variant. See [`StyleTtsModel::load_streaming_f16`].
    w16: HashMap<String, Vec<u16>>,
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
        Ok(Self {
            w,
            w16: HashMap::new(),
        })
    }

    /// Streaming load: fetch + dequant **one tensor at a time** via range reads, so nothing
    /// bigger than a single tensor is ever materialized on top of the growing weight map. The
    /// bulk-load path (`load`) holds the whole GGUF `Vec` *and* the map simultaneously (~2× the
    /// model) plus the JS-side bytes — ~1.6 GB transient for the 543 MB f32 cloning model, which
    /// trips iOS jetsam on load. This keeps the peak at `map + one tensor`. Bit-identical to
    /// [`load`]; works for any fetcher (in-memory native or OPFS browser).
    pub async fn load_streaming(reader: &GgufReader) -> Result<Self> {
        let mut w = HashMap::new();
        let names: Vec<String> = reader.tensors().iter().map(|td| td.name.clone()).collect();
        for name in names {
            let data = dequant_tensor_to_f32_async(reader, &name).await?;
            w.insert(remap(&name), data);
        }
        Ok(Self {
            w,
            w16: HashMap::new(),
        })
    }

    /// Memory-tight streaming load: the **3-D/4-D conv weights** (the bulk of
    /// the model) are kept f16 in `w16` and run through the f16 conv GPU
    /// kernels; everything else (biases, AdaIN/snake affines, 2-D linears that
    /// are consumed on the CPU) is dequantized to f32 in `w`. Roughly halves
    /// the resident weight footprint (host *and* GPU) versus [`load_streaming`].
    /// Intended for the f16 GGUF variant on memory-tight devices; the CPU
    /// reference synth (`encode_voice`/`synthesize`) is NOT available on a model
    /// loaded this way (the conv weights aren't in `w` as f32) — use the GPU
    /// path. Same per-tensor streaming peak as [`load_streaming`].
    pub async fn load_streaming_f16(reader: &GgufReader) -> Result<Self> {
        let mut w = HashMap::new();
        let mut w16 = HashMap::new();
        let descs: Vec<(String, GgmlDtype, usize)> = reader
            .tensors()
            .iter()
            .map(|td| (td.name.clone(), td.dtype, td.dims.len()))
            .collect();
        for (name, dtype, rank) in descs {
            let key = remap(&name);
            // 3-D/4-D conv weights go f16 — EXCEPT the `text_encoder.*` and
            // `predictor.*` convs, which run on the CPU acoustic graph
            // (StyleTtsAcoustic) and must stay f32 in `w`. The GPU decoder /
            // generator / style-encoder convs (the bulk) are routed to the f16
            // GPU kernels.
            let cpu_conv = key.starts_with("text_encoder.") || key.starts_with("predictor.");
            if dtype == GgmlDtype::F16 && (rank == 3 || rank == 4) && !cpu_conv {
                w16.insert(key, dequant_tensor_to_f16_async(reader, &name).await?);
            } else {
                w.insert(key, dequant_tensor_to_f32_async(reader, &name).await?);
            }
        }
        Ok(Self { w, w16 })
    }

    /// Reference 24 kHz mono PCM → 256-d voice vector (acoustic ‖ prosodic).
    /// `progress(fraction, stage)` is invoked at stage boundaries.
    pub fn encode_voice(&self, pcm24k: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        if let Some(p) = progress {
            p(0.10, "computing spectrogram");
        }
        let front = MelFrontend::new(
            self.w.get("mel.window").expect("mel.window"),
            self.w.get("mel.filterbank").expect("mel.filterbank"),
        );
        let (mel, t) = front.compute(pcm24k);
        if let Some(p) = progress {
            p(0.35, "analyzing timbre");
        }
        let a = StyleEncoder::from_weights(&self.w, "acoustic").forward(&mel, 80, t);
        if let Some(p) = progress {
            p(0.70, "analyzing prosody");
        }
        let pros = StyleEncoder::from_weights(&self.w, "prosodic").forward(&mel, 80, t);
        if let Some(p) = progress {
            p(1.0, "voice ready");
        }
        a.into_iter().chain(pros).collect()
    }

    /// Token ids + voice vector → 24 kHz waveform (CPU decoder). `diffuse` enables the
    /// natural-prosody style-diffusion path (alpha/beta blend); `None` = flat zero-shot.
    pub fn synthesize(
        &self,
        ids: &[i64],
        voice: &[f32],
        diffuse: Option<DiffusionConfig>,
        progress: Option<&dyn Fn(f32, &str)>,
    ) -> Vec<f32> {
        let out = StyleTtsAcoustic::new(&self.w).synthesize(ids, voice, diffuse, progress);
        if let Some(p) = progress {
            p(1.0, "done");
        }
        out
    }

    /// GPU voice encode: CPU mel frontend → the StyleEncoder conv stack on the GPU.
    pub async fn encode_voice_gpu(
        &self,
        ctx: &WgpuCtx,
        p: &Pipelines,
        wc: &mut GpuWeightCache,
        pcm24k: &[f32],
        progress: Option<&dyn Fn(f32, &str)>,
    ) -> Vec<f32> {
        if let Some(pp) = progress {
            pp(0.10, "computing spectrogram");
        }
        let front = MelFrontend::new(
            self.w.get("mel.window").expect("mel.window"),
            self.w.get("mel.filterbank").expect("mel.filterbank"),
        );
        let (mel, t) = front.compute(pcm24k);
        if let Some(pp) = progress {
            pp(0.30, "analyzing voice (GPU)");
        }
        let out = StyleTtsGpu::new(&self.w, &self.w16, ctx, p, wc)
            .encode(&mel, 80, t)
            .await;
        if let Some(pp) = progress {
            pp(1.0, "voice ready");
        }
        out
    }

    /// GPU synthesis: CPU acoustic graph (text_encoder/bert/predictor — small) then the
    /// hifigan decoder + generator on the GPU (the dominant cost). `wc` caches uploaded
    /// weights across calls.
    pub async fn synthesize_gpu(
        &self,
        ctx: &WgpuCtx,
        p: &Pipelines,
        wc: &mut GpuWeightCache,
        ids: &[i64],
        voice: &[f32],
        diffuse: Option<DiffusionConfig>,
        progress: Option<&dyn Fn(f32, &str)>,
    ) -> Vec<f32> {
        crate::cancel::clear();
        let ac = StyleTtsAcoustic::new(&self.w);
        let (t_en, bert_out, t) = ac.acoustic_prep(ids, progress);
        if crate::cancel::requested() {
            return Vec::new();
        }
        // style diffusion (natural prosody) on the GPU, between PLBERT and the predictor
        let eff_s = match diffuse {
            Some(cfg) => {
                if let Some(pp) = progress {
                    pp(0.16, "imagining delivery (style diffusion)");
                }
                // sampler params match the converter/oracle (ADPM2 + Karras, sigma_data=0.2)
                let (noise_init, noises) =
                    crate::reference::styletts2::acoustic::diffusion_noise(&cfg);
                let s_pred = StyleTtsGpu::new(&self.w, &self.w16, ctx, p, wc)
                    .diffusion_sample(
                        &bert_out,
                        t,
                        voice,
                        &noise_init,
                        &noises,
                        0.2,
                        1e-4,
                        3.0,
                        9.0,
                        crate::reference::styletts2::acoustic::DIFFUSION_STEPS,
                    )
                    .await;
                crate::reference::styletts2::acoustic::blend_style(&s_pred, voice, &cfg)
            }
            None => voice.to_vec(),
        };
        if crate::cancel::requested() {
            return Vec::new();
        }
        let (asr, f0, n, r, f) = ac.acoustic_rest(&t_en, &bert_out, t, &eff_s, progress);
        if let Some(pp) = progress {
            pp(0.36, "vocoding waveform (GPU decoder)");
        }
        // Last chance to bail before the expensive GPU decoder pass.
        if crate::cancel::requested() {
            return Vec::new();
        }
        let out = StyleTtsGpu::new(&self.w, &self.w16, ctx, p, wc)
            .decode(&asr, f, &f0, &n, &r)
            .await;
        if let Some(pp) = progress {
            pp(1.0, "done");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streaming load must be byte-for-byte identical to the bulk load — it only changes *how*
    /// the bytes reach the dequantizer (per-tensor range read vs one resident `Vec`), not the
    /// math. Gated on `ST2_GGUF` so it's a no-op without the 518 MB local fixture.
    ///   ST2_GGUF=~/.cache/styletts2/styletts2-libritts-f32.gguf cargo test -p rullama st2_streaming
    #[test]
    fn st2_streaming_load_is_bit_identical() {
        let Ok(path) = std::env::var("ST2_GGUF") else {
            eprintln!("skip: set ST2_GGUF to the styletts2 f32 gguf to run");
            return;
        };
        let reader = GgufReader::new(std::fs::read(&path).unwrap()).unwrap();
        let bulk = StyleTtsModel::load(&reader).unwrap();
        let streamed = pollster::block_on(StyleTtsModel::load_streaming(&reader)).unwrap();
        assert_eq!(bulk.w.len(), streamed.w.len(), "tensor count differs");
        for (k, v) in &bulk.w {
            let s = streamed
                .w
                .get(k)
                .unwrap_or_else(|| panic!("streamed missing {k}"));
            assert_eq!(v.as_slice(), s.as_slice(), "weights differ for {k}");
        }
    }
}
