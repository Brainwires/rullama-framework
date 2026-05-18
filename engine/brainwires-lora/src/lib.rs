//! Local LoRA fine-tuning for the rullama Rust runtime.
//!
//! Same trainer on native and `wasm32-unknown-unknown`. The forward,
//! backward, LoRA state, optimizer, and dataset parsing all compile on
//! both targets — the only native-only bits are filesystem helpers
//! that wrap the bytes-based core API (see `load_jsonl_from_bytes` /
//! `save_adapter_to_bytes` / `load_adapter_into_state_from_bytes`).
//!
//! Module map:
//!
//! - [`shared`] — config / error / progress types.
//! - [`dataset_loader`] — JSONL parser (bytes-in core + native path
//!   wrapper) + `Tokenizer` trait + byte-level and HF-`tokenizers`-backed
//!   implementations.
//! - [`lr_schedule`] — warmup + linear / cosine / cosine-warm-restarts
//!   schedules. Cosine clamps `progress` at 1.0.
//! - [`lora`] — LoRA A/B state, forward correction, A/B grad accumulation.
//! - [`scratch`] — per-step GPU scratch buffers for the backward pass.
//! - [`session`] — `TrainingSession` driving one training step
//!   end-to-end (forward → loss → backward → Adam).

/// JSONL dataset loader + tokenizer trait.
pub mod dataset_loader;
/// Per-LoRA GPU state: A and B matrices for each wrapped projection.
pub mod lora;
/// Learning rate schedules.
pub mod lr_schedule;
/// Per-step GPU scratch buffers for the backward pass.
pub mod scratch;
/// `TrainingSession` — drives one training step end-to-end.
pub mod session;
/// Shared configuration, error, and progress types.
pub mod shared;

/// JS-facing wasm-bindgen surface — only compiled for wasm32.
#[cfg(target_arch = "wasm32")]
pub mod wasm_bindgen_api;

#[cfg(not(target_arch = "wasm32"))]
pub use session::load_adapter_into_state;
pub use session::{TrainingSession, load_adapter_into_state_from_bytes};
