//! Shared config / error / types — vendored from `brainwires-finetune` so
//! `brainwires-lora` is fully standalone (no cross-repo dep on
//! brainwires-framework). Kept structurally identical to the upstream
//! modules so types map 1:1; tweaks can diverge over time.

pub mod config;
pub mod error;
pub mod types;
