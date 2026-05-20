//! # Brainwires iMessage Channel (BlueBubbles)
//!
//! iMessage adapter for the Brainwires Agent Framework. Talks to a
//! user-operated [BlueBubbles] server running on macOS; that server
//! exposes iMessage as a REST API and our adapter polls it for new
//! messages and POSTs outbound text back to it.
//!
//! [BlueBubbles]: https://bluebubbles.app

/// Adapter configuration.
pub mod config;
/// Gateway WebSocket client.
pub mod gateway_client;
/// BlueBubbles REST client + Channel implementation.
pub mod imessage;
/// Polling ingress loop.
pub mod ingress;
/// MCP stdio tool server.
pub mod mcp_server;
/// Persistent cursor (last-seen message guid per chat).
pub mod state;
