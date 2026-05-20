//! # Brainwires Microsoft Teams Channel
//!
//! Microsoft Teams channel adapter. Teams bots use the Bot Framework
//! protocol: Microsoft posts `Activity` JSON to an HTTPS endpoint,
//! authenticated via a Microsoft-signed JWT. Replies are POSTed back to
//! the `serviceUrl` embedded in the inbound activity, authenticated with
//! an OAuth client-credentials bearer.

/// Adapter configuration.
pub mod config;
/// Gateway WebSocket client (copy of shared pattern).
pub mod gateway_client;
/// Bot Framework JWT verification + JWKs cache.
pub mod jwt;
/// MCP stdio server.
pub mod mcp_server;
/// OAuth client-credentials bearer minter.
pub mod oauth;
/// `Channel` trait implementation — egress.
pub mod teams;
/// HTTPS webhook (`/api/messages`) — ingress.
pub mod webhook;
