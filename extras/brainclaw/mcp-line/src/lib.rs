//! # Brainwires LINE Channel
//!
//! LINE Messaging API adapter for the Brainwires Agent Framework.
//! Inbound events arrive on a webhook signed with HMAC-SHA256 using the
//! channel secret; outbound messages go through the LINE REST API with
//! a long-lived channel access token.

/// Adapter configuration.
pub mod config;
/// Gateway WebSocket client.
pub mod gateway_client;
/// LINE REST client + Channel trait implementation.
pub mod line;
/// MCP stdio tool server.
pub mod mcp_server;
/// HTTPS webhook ingress + signature verification.
pub mod webhook;
