// Conv2D + LayerNorm CPU forward fns take many dims (C_in, C_out, kH, kW,
// stride_h, stride_w, pad_h, pad_w, …) to match the Go reference. Loop
// indices walk parallel input/output planes per iteration.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

//! Gemma 4 audio SSCP prefix (CPU): waveform → `[seq, hidden]` f32.
//!
//! What's here:
//!   * `AudioConfig`: shape constants pulled from the GGUF, plus the
//!     Ollama-Go-hardcoded scalars (chunk/past/future, softcap, residual
//!     weight) that don't live in the file.
//!   * `AudioPrefix`: the small CPU front-end that produces the input to
//!     the GPU Conformer block loop — mel-spec → 2× (Conv2D 3×3 stride-2 +
//!     LayerNorm + ReLU) → linear projection to `hidden`. A few MB of
//!     resident weight.
//!
//! What used to be here:
//!   * A full 12-block CPU Conformer mirror + projector chain (`AudioForward`)
//!     that served as the M13 parity oracle. M16 dropped it once the GPU
//!     encoder was bit-identical to Ollama (verified per the project memory)
//!     — ~360 MB of resident f32 weight saved on every multimodal load.
//!
//! The downstream GPU encoder lives in `audio_gpu::GpuAudioForward`.

use std::sync::Arc;

use crate::backend::WeightCache;
use crate::error::{Result, RullamaError};
use crate::gguf::{GgufReader, dequant_tensor_to_f32_async};

use super::audio_features::{MEL_BINS, MelEngine};

#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub n_layers: u32,     // gemma4.audio.block_count                (12)
    pub hidden: u32,       // gemma4.audio.embedding_length           (1024)
    pub ffn_inter: u32,    // gemma4.audio.feed_forward_length        (4096)
    pub n_heads: u32,      // gemma4.audio.attention.head_count       (8)
    pub conv_kernel: u32,  // gemma4.audio.conv_kernel_size           (5)
    pub mel_bins: u32,     // 128 (also our MEL_BINS)
    pub eps: f32,          // 1e-6 default
    pub chunk_size: u32,   // 12  (Ollama-hardcoded)
    pub max_past: u32,     // 12
    pub max_future: u32,   // 0
    pub context_size: u32, // chunk_size + max_past + max_future = 24
    pub logit_cap: f32,    // 50.0
    pub residual_w: f32,   // 0.5
    pub grad_clip: f32,    // 1e10
    pub d_text: u32,       // text d_model — projector output
}

impl AudioConfig {
    pub fn from_gguf(r: &GgufReader, d_text: u32) -> Result<Self> {
        let n_layers = r
            .get_opt("gemma4.audio.block_count")
            .and_then(|v| v.as_u32().ok())
            .ok_or_else(|| {
                RullamaError::Inference(
                    "gemma4.audio.block_count missing — not a multimodal-audio GGUF?".into(),
                )
            })?;
        let hidden = r.get("gemma4.audio.embedding_length")?.as_u32()?;
        let ffn_inter = r.get("gemma4.audio.feed_forward_length")?.as_u32()?;
        let n_heads = r.get("gemma4.audio.attention.head_count")?.as_u32()?;
        let conv_kernel = r
            .get_opt("gemma4.audio.conv_kernel_size")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(5);
        let mel_bins = r
            .get_opt("gemma4.audio.num_mel_bins")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(MEL_BINS as u32);
        let eps = r
            .get_opt("gemma4.audio.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f32().ok())
            .unwrap_or(1e-6);
        // The chunk/past/future/cap/residual constants live in Ollama's Go (not the GGUF).
        let chunk_size = 12;
        let max_past = 12;
        let max_future = 0;
        Ok(Self {
            n_layers,
            hidden,
            ffn_inter,
            n_heads,
            conv_kernel,
            mel_bins,
            eps,
            chunk_size,
            max_past,
            max_future,
            context_size: chunk_size + max_past + max_future,
            logit_cap: 50.0,
            residual_w: 0.5,
            grad_clip: 1e10,
            d_text,
        })
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden / self.n_heads
    }
}

/// CPU-side audio SSCP prefix: mel-spec → 2× (Conv2D 3×3 stride-2 + LayerNorm + ReLU)
/// → linear projection to `hidden`. Produces the `[seq, hidden]` f32 input that
/// `GpuAudioForward` feeds into the GPU Conformer block loop.
///
/// Pre-M16 this struct was `AudioForward` and held the full 12-block CPU
/// Conformer mirror (~360 MB f32 dequant) as a parity oracle. The GPU
/// encoder is bit-identical to Ollama now (M13 validation), so the oracle
/// is gone and only the SSCP prefix remains. Total resident weight here:
/// the two 3×3 convs + their norms + one linear ≈ a few MB.
pub struct AudioPrefix {
    cfg: AudioConfig,
    mel: MelEngine,

    // SSCP weights (2 × Conv2D + LayerNorm + linear projection).
    sscp0_w: Vec<f32>, // [out_C0, in_C=1, kH=3, kW=3]
    sscp0_norm_w: Vec<f32>,
    sscp0_norm_b: Option<Vec<f32>>,
    sscp1_w: Vec<f32>, // [out_C1, in_C=out_C0, kH=3, kW=3]
    sscp1_norm_w: Vec<f32>,
    sscp1_norm_b: Option<Vec<f32>>,
    pre_encode_out_w: Vec<f32>, // linear: out_C1 * F'' → hidden
    pre_encode_out_b: Option<Vec<f32>>,
    sscp0_out_c: usize,
    sscp1_out_c: usize,
    sscp_proj_in: usize,
}

impl AudioPrefix {
    /// Load just the SSCP prefix weights (two 3×3 Conv2D + their LayerNorms
    /// + one linear projection to `hidden`). Total fetched: a few MB.
    pub async fn new(cfg: AudioConfig, wcache: Arc<WeightCache>) -> Result<Self> {
        let r = wcache.reader();

        // SSCP. The Conv2D weight shapes are stored in GGUF as [kW, kH, in_C, out_C]
        // with dim[0] fastest. Ollama's converter doesn't reshape these; we read
        // the raw bytes and reinterpret as f32.
        let sscp0_desc = r.tensor("a.conv1d.0.weight")?;
        let sscp0_w = dequant_tensor_to_f32_async(r, "a.conv1d.0.weight").await?;
        let sscp0_out_c = *sscp0_desc.dims.last().unwrap_or(&1) as usize;

        let sscp1_desc = r.tensor("a.conv1d.1.weight")?;
        let sscp1_w = dequant_tensor_to_f32_async(r, "a.conv1d.1.weight").await?;
        let sscp1_out_c = *sscp1_desc.dims.last().unwrap_or(&1) as usize;

        let sscp0_norm_w = dequant_tensor_to_f32_async(r, "a.conv1d.0.norm.weight").await?;
        let sscp0_norm_b = load_opt_f32(r, "a.conv1d.0.norm.bias").await?;
        let sscp1_norm_w = dequant_tensor_to_f32_async(r, "a.conv1d.1.norm.weight").await?;
        let sscp1_norm_b = load_opt_f32(r, "a.conv1d.1.norm.bias").await?;

        let pre_encode_out_w = dequant_tensor_to_f32_async(r, "a.pre_encode.out.weight").await?;
        let pre_encode_out_b = load_opt_f32(r, "a.pre_encode.out.bias").await?;

        // The pre_encode linear's input dim = (out_C1 * F'') where F'' is the
        // post-SSCP frequency dimension. Read it from the linear's k-axis (the
        // input axis is dims[0] in GGUF storage, i.e. the fast axis of the [k, n]
        // weight). Pre-encode weight shape: dim[0] = sscp_proj_in, dim[1] = hidden.
        let pre_desc = r.tensor("a.pre_encode.out.weight")?;
        let sscp_proj_in = *pre_desc.dims.first().unwrap_or(&1) as usize;

        Ok(Self {
            cfg,
            mel: MelEngine::new(),
            sscp0_w,
            sscp0_norm_w,
            sscp0_norm_b,
            sscp1_w,
            sscp1_norm_w,
            sscp1_norm_b,
            pre_encode_out_w,
            pre_encode_out_b,
            sscp0_out_c,
            sscp1_out_c,
            sscp_proj_in,
        })
    }

    pub fn cfg(&self) -> &AudioConfig {
        &self.cfg
    }

    /// Compute a log-mel spectrogram from `samples` (16 kHz mono f32, [-1, 1]).
    /// Returns the flat `[n_frames * mel_bins]` tensor and the frame count.
    pub fn mel_spectrogram(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.mel.log_mel(samples)
    }

    /// Run sections 1-3 of the encoder (mel features + SSCP convs + pre_encode
    /// linear projection) and return the post-pre-encode hidden state plus the
    /// post-SSCP frame count. Used by `GpuAudioForward` to handle the
    /// CPU-friendly prefix while the heavy block loop runs on GPU.
    pub fn prefix_to_hidden(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden as usize;
        let mel_bins = cfg.mel_bins as usize;

        let (mel, n_frames) = self.mel.log_mel(samples);
        if n_frames == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut x = self.sscp_conv_block(
            &mel,
            1,
            n_frames,
            mel_bins,
            self.sscp0_out_c,
            &self.sscp0_w,
            &self.sscp0_norm_w,
            self.sscp0_norm_b.as_deref(),
        );
        let t1 = n_frames.div_ceil(2);
        let f1 = mel_bins.div_ceil(2);
        x = self.sscp_conv_block(
            &x,
            self.sscp0_out_c,
            t1,
            f1,
            self.sscp1_out_c,
            &self.sscp1_w,
            &self.sscp1_norm_w,
            self.sscp1_norm_b.as_deref(),
        );
        let t_out = t1.div_ceil(2);
        let f_out = f1.div_ceil(2);
        let flat_per_frame = f_out * self.sscp1_out_c;
        if flat_per_frame != self.sscp_proj_in {
            return Err(RullamaError::Inference(format!(
                "audio SSCP: flat per-frame dim {flat_per_frame} != pre_encode k {}",
                self.sscp_proj_in
            )));
        }
        let h = Self::linear_rows(
            &x,
            &self.pre_encode_out_w,
            self.pre_encode_out_b.as_deref(),
            t_out,
            self.sscp_proj_in,
            hidden,
        );
        Ok((h, t_out))
    }

    // The pre-M16 `encode()` (full 12-block CPU Conformer + projector) was
    // deleted alongside the CpuAudioForward oracle. The GPU encoder in
    // `audio_gpu::GpuAudioForward` is the only `encode` path now; parity is
    // gated by `examples/audio_parity.rs` against Ollama.

    /// One SSCP block: Conv2D (kernel=3, stride=2, padding=1) → LayerNorm → ReLU.
    /// Input layout: `[T, F, C_in]` channel-LAST flat.
    /// Output layout: `[T_out, F_out, C_out]` channel-LAST flat.
    fn sscp_conv_block(
        &self,
        x: &[f32],
        in_c: usize,
        in_t: usize,
        in_f: usize,
        out_c: usize,
        weight: &[f32],
        norm_w: &[f32],
        norm_b: Option<&[f32]>,
    ) -> Vec<f32> {
        // Conv2D kernel layout from GGUF: dims = [kW, kH, in_C, out_C]
        // (kW fastest); element at (oC, iC, kH, kW) = weight[((oC*in_c + iC)*3 + kH)*3 + kW].
        // Spatial: stride=(2,2), padding=(1,1), dilation=1.
        let k_h = 3usize;
        let k_w = 3usize;
        let s = 2usize;
        let pad = 1usize;
        let out_t = (in_t + 2 * pad).saturating_sub(k_h) / s + 1;
        let out_f = (in_f + 2 * pad).saturating_sub(k_w) / s + 1;
        let mut y = vec![0f32; out_t * out_f * out_c];

        for ot in 0..out_t {
            for of in 0..out_f {
                let in_t_base = (ot * s) as i64 - pad as i64;
                let in_f_base = (of * s) as i64 - pad as i64;
                for oc in 0..out_c {
                    let mut acc = 0f32;
                    for ic in 0..in_c {
                        for kh in 0..k_h {
                            let it = in_t_base + kh as i64;
                            if it < 0 || it >= in_t as i64 {
                                continue;
                            }
                            for kw in 0..k_w {
                                let if_ = in_f_base + kw as i64;
                                if if_ < 0 || if_ >= in_f as i64 {
                                    continue;
                                }
                                let xi = ((it as usize) * in_f + if_ as usize) * in_c + ic;
                                let wi = ((oc * in_c + ic) * k_h + kh) * k_w + kw;
                                acc += x[xi] * weight[wi];
                            }
                        }
                    }
                    y[(ot * out_f + of) * out_c + oc] = acc;
                }
            }
        }

        // LayerNorm across the channel axis (per spatial position), then ReLU.
        for ot in 0..out_t {
            for of in 0..out_f {
                let off = (ot * out_f + of) * out_c;
                let row = &mut y[off..off + out_c];
                let mean: f32 = row.iter().sum::<f32>() / out_c as f32;
                let var: f32 =
                    row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out_c as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for c in 0..out_c {
                    let normed =
                        (row[c] - mean) * inv * norm_w[c] + norm_b.map(|b| b[c]).unwrap_or(0.0);
                    row[c] = normed.max(0.0); // ReLU
                }
            }
        }
        y
    }

    // ---- CPU primitives (per-frame ops; channel-LAST [seq, channels] layout) ----

    /// In-place RMSNorm with optional learned weight.
    /// `x` is `[seq, dim]`; norms each row of `dim` independently.
    pub fn rmsnorm_rows(x: &mut [f32], seq: usize, dim: usize, weight: Option<&[f32]>, eps: f32) {
        for r in 0..seq {
            let row = &mut x[r * dim..(r + 1) * dim];
            let mut sum_sq = 0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            if let Some(w) = weight {
                for i in 0..dim {
                    row[i] = row[i] * inv_rms * w[i];
                }
            } else {
                for v in row.iter_mut() {
                    *v *= inv_rms;
                }
            }
        }
    }

    /// `y[s, n] = Σ_k x[s, k] * w[n, k]` (+ optional `b[n]`).
    /// `w` shape `[k_dim, n_dim]` (GGUF dim[0] = k = fast axis).
    pub fn linear_rows(
        x: &[f32],
        w: &[f32],
        b: Option<&[f32]>,
        seq: usize,
        k_dim: usize,
        n_dim: usize,
    ) -> Vec<f32> {
        let mut y = vec![0f32; seq * n_dim];
        for s in 0..seq {
            for n in 0..n_dim {
                let mut acc = 0f32;
                for k in 0..k_dim {
                    acc += x[s * k_dim + k] * w[n * k_dim + k];
                }
                if let Some(bias) = b {
                    acc += bias[n];
                }
                y[s * n_dim + n] = acc;
            }
        }
        y
    }
}

async fn load_opt_f32(r: &GgufReader, name: &str) -> Result<Option<Vec<f32>>> {
    match r.tensor(name) {
        Ok(_) => Ok(Some(dequant_tensor_to_f32_async(r, name).await?)),
        Err(_) => Ok(None),
    }
}
