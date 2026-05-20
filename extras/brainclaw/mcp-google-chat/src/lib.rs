//! # Brainwires Google Chat Channel
//!
//! Google Chat channel adapter for the Brainwires Agent Framework.
//!
//! Google Chat bots ingest events over HTTPS webhooks signed by Google
//! (RS256 JWT with `aud` == this bot's audience). Outbound messages are
//! posted to the Chat REST API using an OAuth bearer minted from a
//! configured service-account JSON key.
//!
//! This crate implements the `Channel` trait from `brainwires-network` for
//! Google Chat and also doubles as a standalone MCP tool server.

/// Adapter configuration.
pub mod config;
/// Gateway WebSocket client (copy of the shared pattern — see crate README).
pub mod gateway_client;
/// Google Chat REST API client + `Channel` trait implementation.
pub mod google_chat;
/// MCP stdio tool server exposing send_message / get_history.
pub mod mcp_server;
/// OAuth bearer minting from a service account JSON key (JWT self-signed
/// flow — no external SDK).
pub mod oauth;
/// Webhook ingress — Axum server + Google JWT verification.
pub mod webhook;
