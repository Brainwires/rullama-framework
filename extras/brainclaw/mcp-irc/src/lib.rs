//! # Brainwires IRC Channel
//!
//! IRC channel adapter. Connects to a configured IRC network (optionally
//! TLS + SASL), joins configured channels, and forwards PRIVMSG traffic
//! both ways between IRC and the brainwires-gateway.

/// Adapter configuration.
pub mod config;
/// Gateway WebSocket client.
pub mod gateway_client;
/// IRC client handle + `Channel` trait implementation.
pub mod irc_client;
/// MCP stdio tool server.
pub mod mcp_server;
/// IRC protocol helpers — frame parsing, PRIVMSG splitting, etc.
pub mod protocol;
