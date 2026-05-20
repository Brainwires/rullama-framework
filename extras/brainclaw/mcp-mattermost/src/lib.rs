//! # Brainwires Mattermost Channel
//!
//! Mattermost channel adapter for the Brainwires Agent Framework.
//!
//! This crate implements the `Channel` trait from `brainwires-channels` for Mattermost,
//! using the Mattermost WebSocket API and REST API v4 via `reqwest` + `tokio-tungstenite`.
//! It connects to the brainwires-gateway over WebSocket and can also serve as a
//! standalone MCP tool server.

/// Configuration types for the Mattermost adapter.
pub mod config;
/// WebSocket event handler that converts Mattermost events to `ChannelEvent`.
pub mod event_handler;
/// WebSocket client for connecting to the brainwires-gateway.
pub mod gateway_client;
/// Mattermost bot implementation of the `Channel` trait.
pub mod mattermost;
/// MCP server exposing Mattermost operations as tools.
pub mod mcp_server;
