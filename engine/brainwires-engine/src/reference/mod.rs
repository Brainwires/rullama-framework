//! Pure-Rust f32 forward pass for Gemma 4. The parity oracle for our WGSL kernels.
//!
//! Performance is irrelevant here — correctness against the Ollama Go reference
//! implementation (`/Users/nightness/Source/ollama/model/models/gemma4/model_text.go`)
//! is the only thing that matters.
//!
//! Built only when the `cpu-reference` cargo feature is enabled, to keep WASM bundle
//! size small.

pub mod diffusion;
pub mod embed;
pub mod forward;
pub mod forward_chained;
pub mod forward_gpu;
pub mod imagegen;
pub mod kokoro;
pub mod moe;
pub mod ops;
/// Qwen3 text-encoder CPU oracle. Reads sharded safetensors from disk, so
/// native-only (the GPU encoder forward is the wasm-facing path).
#[cfg(not(target_arch = "wasm32"))]
pub mod qwen3;
pub mod rome;
pub mod styletts2;
pub mod weights;

pub use forward::{KvState, LayerKv, forward_token};
pub use forward_gpu::forward_token_gpu;
pub use weights::Weights;
