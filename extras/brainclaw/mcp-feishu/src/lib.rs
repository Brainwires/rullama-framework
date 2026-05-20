//! # Brainwires Feishu / Lark Channel
//!
//! Feishu / Lark channel adapter for the Brainwires Agent Framework.
//!
//! Inbound events arrive on a webhook signed with a custom HMAC-SHA256
//! scheme over `timestamp + nonce + body`. Outbound messages are
//! posted to the Feishu Open Platform after minting a short-lived
//! tenant access token from the configured app id + secret.

/// Adapter configuration.
pub mod config;
/// Feishu REST client + Channel trait implementation.
pub mod feishu;
/// Gateway WebSocket client.
pub mod gateway_client;
/// MCP stdio tool server.
pub mod mcp_server;
/// Tenant-access-token minter.
pub mod oauth;
/// HTTPS webhook ingress + signature verification.
pub mod webhook;
