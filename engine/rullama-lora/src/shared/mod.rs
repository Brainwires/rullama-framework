//! Shared config / error / types — vendored from `rullama-finetune` so
//! `rullama-lora` is fully standalone (no cross-repo dep on
//! rullama-framework). Kept structurally identical to the upstream
//! modules so types map 1:1; tweaks can diverge over time.

pub mod config;
pub mod error;
pub mod types;
