//! Internal test fixtures for the Brainwires framework.
//!
//! Consolidates mock implementations that were previously duplicated inline
//! across multiple `crates/brainwires-*` test modules. Not published; intended
//! for use in `#[cfg(test)]` and `tests/` directories within the workspace.

pub mod provider;

pub use provider::{
    FailingProvider, RecordedCall, RecordingProvider, ScriptedProvider, ScriptedResponse,
};
