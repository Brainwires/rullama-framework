//! Internal test fixtures for the rullama framework.
//!
//! Consolidates mock implementations that were previously duplicated inline
//! across multiple `crates/rullama-*` test modules. Not published; intended
//! for use in `#[cfg(test)]` and `tests/` directories within the workspace.

pub mod provider;

pub use provider::{
    FailingProvider, RecordedCall, RecordingProvider, ScriptedProvider, ScriptedResponse,
};
