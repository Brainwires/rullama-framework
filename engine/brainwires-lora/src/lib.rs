#![cfg(not(target_arch = "wasm32"))]
//! Local LoRA fine-tuning for the rullama Rust runtime.
//!
//! Native-only — the crate is empty on `wasm32-unknown-unknown`.
//!
//! Status: **skeleton only**. The previously vendored Burn-based trainer
//! was found to be broken end-to-end (see `MIGRATION-REPORT.md`) and was
//! gutted in the teardown commit that immediately precedes the M0
//! rewrite. The next milestones replace it with a hand-written reverse
//! pass over rullama's existing wgpu kernels.
//!
//! Module map after teardown:
//!
//! - [`shared`] — config / error / progress types.
//! - [`dataset_loader`] — JSONL parser + `Tokenizer` trait + byte-level
//!   and HF-`tokenizers`-backed implementations.
//! - [`lr_schedule`] — warmup + linear / cosine / cosine-warm-restarts
//!   schedules. Cosine clamps `progress` at 1.0.
//!
//! Modules added in M0+:
//!
//! - `backward` — reverse pass mirroring `encode_layer`.
//! - `lora` — LoRA A/B state, forward correction, A/B grad accumulation.
//! - `optim` — Adam over GPU buffers.
//! - `loss` — cross-entropy forward + backward.

/// Shared configuration, error, and progress types.
pub mod shared;
/// JSONL dataset loader + tokenizer trait.
pub mod dataset_loader;
/// Learning rate schedules.
pub mod lr_schedule;
/// Per-LoRA GPU state: A and B matrices for each wrapped projection.
pub mod lora;
/// Per-step GPU scratch buffers for the backward pass.
pub mod scratch;
/// `TrainingSession` — drives one training step end-to-end.
pub mod session;
pub use session::{load_adapter_into_state, TrainingSession};
