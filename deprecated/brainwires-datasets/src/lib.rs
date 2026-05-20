//! Tombstone — see README.md.
//!
//! `brainwires-training` itself moved to `rullama-finetune` in 0.11. The
//! re-export chain via `brainwires_training::datasets::*` no longer compiles
//! because the upstream surface has shifted. Consumers should migrate to
//! `brainwires-finetune::datasets` (cloud-side) or the `rullama-finetune`
//! sibling workspace (local-side). This crate intentionally exports nothing.
#![deprecated(
    since = "0.8.0",
    note = "datasets merged into `brainwires-finetune::datasets` (cloud) or the rullama-finetune sibling workspace (local)"
)]
