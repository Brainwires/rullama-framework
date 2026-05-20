//! # Reload Daemon
//!
//! Library surface for the `reload-daemon` binary. Exposes `config` and
//! `reload` so integration tests (and in principle other tools) can drive the
//! signal-escalation and argument-transform logic directly.

pub mod config;
pub mod reload;
pub mod server;
