//! rullama — Gemma 4 inference in the browser via wgpu + WASM.
//!
//! # Stability
//!
//! Three modules are the **stable public API** and follow semver across
//! 0.x patch releases:
//!
//! - [`api`] — the high-level [`api::Model`] handle, [`api::ChatMessage`],
//!   [`api::ChatRole`], [`api::GenerateOptions`], and the
//!   `loadFrom*` / `generate` / `stop` entry points. This is what
//!   `#[wasm_bindgen]` exposes to JS, and what native Rust consumers
//!   should program against.
//! - [`error`] — [`error::RullamaError`] and [`error::Result`].
//! - [`sampling`] — [`sampling::SamplingOptions`] and [`sampling::Sampler`].
//!
//! Every other module (`backend`, `gguf`, `kernels`, `model`, `multimodal`,
//! `reference`, `template`, `tokenizer`) is `#[doc(hidden)]` and is
//! considered **implementation detail**. They are reachable so that the
//! sibling workspace crates (`rullama-finetune`, `rullama-ios-bench`) can
//! link against the wgpu kernel set, the GGUF parser, and the parity
//! oracles — but their layout, names, and signatures may change in any
//! patch release without notice. External callers that pin against them
//! do so at their own risk.

pub mod api;
pub mod error;
pub mod sampling;
/// Inference-time LoRA adapter — parsed from safetensors bytes,
/// attaches to a `Model` via `loadAdapter` / `clearAdapter`.
pub mod lora;

#[doc(hidden)]
pub mod backend;
#[doc(hidden)]
pub mod gguf;
#[doc(hidden)]
pub mod kernels;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod multimodal;
#[doc(hidden)]
pub mod reference;
#[doc(hidden)]
pub mod template;
#[doc(hidden)]
pub mod tokenizer;

pub use error::RullamaError;

// Re-export rsqlite-wasm's `WasmDatabase` so wasm-bindgen sees it as a
// reachable public symbol and emits its JS bindings into pkg/rullama.js.
// Without the re-export, the rlib's #[wasm_bindgen] exports get
// dead-code-stripped because nothing in rullama itself calls them — the
// PWA uses them only from JS.
#[cfg(target_arch = "wasm32")]
pub use rsqlite_wasm::WasmDatabase;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn __wasm_start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}
