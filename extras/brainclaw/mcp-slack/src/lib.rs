//! # Brainwires Slack Channel
//!
//! Slack channel adapter for the Brainwires Agent Framework.
//!
//! This crate implements the `Channel` trait from `brainwires-channels` for Slack,
//! using Slack's Socket Mode (WebSocket) and Web API via `reqwest` + `tokio-tungstenite`.
//! It connects to the brainwires-gateway over WebSocket and can also serve as a
//! standalone MCP tool server.

/// Configuration types for the Slack adapter.
pub mod config;
/// Socket Mode event handler that converts Slack events to `ChannelEvent`.
pub mod event_handler;
/// WebSocket client for connecting to the brainwires-gateway.
pub mod gateway_client;
/// MCP server exposing Slack operations as tools.
pub mod mcp_server;
/// Slack bot implementation of the `Channel` trait.
pub mod slack;
