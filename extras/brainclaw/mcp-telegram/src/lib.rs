//! # Brainwires Telegram Channel
//!
//! Telegram channel adapter for the Brainwires Agent Framework.
//!
//! This crate implements the `Channel` trait from `brainwires-channels` for Telegram,
//! using the teloxide library. It connects to the brainwires-gateway over WebSocket
//! and can also serve as a standalone MCP tool server.

/// Configuration types for the Telegram adapter.
pub mod config;
/// Teloxide dispatcher that converts Telegram updates to `ChannelEvent`.
pub mod event_handler;
/// WebSocket client for connecting to the brainwires-gateway.
pub mod gateway_client;
/// MCP server exposing Telegram operations as tools.
pub mod mcp_server;
/// Telegram bot implementation of the `Channel` trait.
pub mod telegram;
