//! # Brainwires GitHub Channel
//!
//! GitHub channel adapter for the Brainwires Agent Framework.
//!
//! This crate implements the `Channel` trait from `brainwires-channels` for GitHub.
//! It receives GitHub webhook events via an Axum HTTP server, normalises them to
//! `ChannelMessage`, forwards them to the brainwires-gateway over WebSocket, and
//! can also serve as a standalone MCP tool server exposing GitHub operations.
//!
//! ## Architecture
//!
//! ```text
//! GitHub webhook ──► axum /webhook ──► mpsc channel
//!                                             │
//!                                   gateway_client (WebSocket)
//!                                             │
//!                               brainwires-gateway ──► agent session
//!
//! AI tool call ──► MCP stdio/HTTP ──► GitHubMcpServer ──► GitHub REST API
//! ```

/// Configuration types for the GitHub adapter.
pub mod config;
/// WebSocket client for connecting to the brainwires-gateway.
pub mod gateway_client;
/// GitHub REST API client implementing the `Channel` trait.
pub mod github;
/// MCP server exposing GitHub operations as tools.
pub mod mcp_server;
/// Webhook receiver (Axum) for inbound GitHub events.
pub mod webhook;
