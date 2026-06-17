//! Image-generation engine (in progress).
//!
//! A sibling engine to the Gemma 4 text path and the TTS/embedding engines:
//! it shares the `backend::WgpuCtx` + `Pipelines` + `WeightCache` + bind-cache
//! foundation but owns its own weight format, forward path, and WGSL kernels.
//! The Gemma 4 path is untouched.
//!
//! First target is **Z-Image-Turbo** (single-stream S3-DiT + FLUX VAE +
//! Qwen3 text encoder), then **FLUX.2 Klein**. Unlike the LLM path, image
//! models are NOT GGUF: Ollama packages them as one content-addressed
//! safetensors blob per tensor (component-namespaced `text_encoder/` /
//! `transformer/` / `vae/`), with float (`BF16`/`F16`/`F32`/`F8`) or grouped
//! (`int4`/`int8`/`nvfp4`/`mxfp8`) quantization.
//!
//! IM0 (this slice): the ingestion layer —
//! - [`manifest::ImageManifest`] parses the OCI manifest → tensor table.
//! - [`safetensors::SafetensorsBlob`] parses one per-tensor blob.
//! - [`dtype::StDtype`] decodes a tensor's raw bytes to f32.
//!
//! Still to come: grouped-quant reconstruction, the blob source over OPFS/HTTP
//! `Range`, the Qwen3 encoder (IM1), the DiT denoiser (IM2), the VAE decoder
//! (IM3), the sampling loop (IM4), and the `ImageModel` wasm surface (IM5).

pub mod config;
pub mod dit_forward;
pub mod dtype;
pub mod manifest;
pub mod model;
pub mod qwen3_forward;
pub mod safetensors;
pub mod scheduler;
pub mod sharded;
pub mod source;
pub mod streaming;
pub mod timestep;

pub use config::{Qwen3Config, SchedulerConfig, TransformerConfig, VaeConfig};
pub use dit_forward::DitGpu;
pub use dtype::StDtype;
pub use manifest::{BlobRef, ImageManifest, MEDIA_JSON, MEDIA_TENSOR};
#[cfg(target_arch = "wasm32")]
pub use model::ImageModel;
pub use model::{ImageBundle, rgb_chw_to_rgba8};
pub use qwen3_forward::Qwen3Gpu;
pub use safetensors::{SafetensorsBlob, SafetensorsHeader, TensorEntry, read_header};
pub use scheduler::{FlowMatchScheduler, calculate_shift, latent_hw, time_shift};
pub use sharded::ShardIndex;
#[cfg(not(target_arch = "wasm32"))]
pub use sharded::ShardedSafetensors;
pub use source::BlobSource;
#[cfg(target_arch = "wasm32")]
pub use source::HttpRangeBlobSource;
#[cfg(not(target_arch = "wasm32"))]
pub use source::{FileBlobSource, find_manifest, ollama_models_root};
pub use streaming::StreamingShards;
pub use timestep::sinusoidal_timestep_embedding;
