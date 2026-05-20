//! # Brainwires Nextcloud Talk Channel
//!
//! Nextcloud Talk (Spreed) channel adapter for the Brainwires Agent
//! Framework. Polls the Talk REST API for new chat messages in
//! configured rooms and POSTs replies back via the same API.
//!
//! All calls use HTTP Basic authentication with a Nextcloud username +
//! app password, and require the `OCS-APIRequest: true` header.

/// Adapter configuration.
pub mod config;
/// Gateway WebSocket client.
pub mod gateway_client;
/// Polling ingress loop per configured room.
pub mod ingress;
/// MCP stdio tool server.
pub mod mcp_server;
/// Nextcloud Talk REST client + Channel implementation.
pub mod nextcloud_talk;
/// Persistent cursor (last-seen message id per room).
pub mod state;
