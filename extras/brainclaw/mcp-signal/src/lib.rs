//! # Brainwires Signal Channel
//!
//! Signal messenger channel adapter for the Brainwires Agent Framework.
//!
//! This crate implements the `Channel` trait from `brainwires-channels` for Signal,
//! connecting via the `signal-cli-rest-api` daemon (REST + WebSocket).
//! It connects to the brainwires-gateway over WebSocket and can also serve as a
//! standalone MCP tool server.
//!
//! ## Prerequisites
//!
//! You need `signal-cli` running in daemon mode:
//!
//! ```text
//! signal-cli -a +14155552671 daemon --http 127.0.0.1:8080
//! ```
//!
//! Or the Docker image `bbernhard/signal-cli-rest-api`.

/// Configuration types for the Signal adapter.
pub mod config;
/// Event handler connecting to signal-cli REST API (WebSocket + polling).
pub mod event_handler;
/// WebSocket client for connecting to the brainwires-gateway.
pub mod gateway_client;
/// MCP server exposing Signal operations as tools.
pub mod mcp_server;
/// Signal channel implementation of the `Channel` trait.
pub mod signal;
