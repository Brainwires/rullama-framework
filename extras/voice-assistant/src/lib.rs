//! Library surface for the `voice-assistant` binary.
//!
//! The crate is primarily a CLI (`src/main.rs`), but exposing the pure
//! modules here lets integration tests and downstream embedders reuse the
//! configuration and handler logic without shelling out to the binary.

pub mod config;
pub mod handler;
