//! # Brainwires Gateway
//!
//! Always-on WebSocket server that routes messages between channel MCP servers
//! and agent sessions. This is the hub of the personal AI assistant architecture.
//!
//! Channel adapters (Discord, Telegram, Slack, etc.) connect to the gateway via
//! WebSocket, perform a handshake, and then exchange `ChannelEvent` messages.
//! The gateway manages session mapping and routes messages to/from agent processes.

/// Admin API handlers (health check, channel listing, session listing, broadcast).
pub mod admin;
/// Agent-backed inbound handler that bridges gateway events to ChatAgent.
pub mod agent_handler;
/// Interactive tool approval via chat (ask user yes/no before executing tools).
pub mod approval;
/// Audit logging for security-relevant events.
pub mod audit;
/// Channel registry for tracking connected channel adapters.
pub mod channel_registry;
/// Gateway configuration.
pub mod config;
/// Cron job data types and persistent store.
pub mod cron;
/// Cross-channel user identity mapping.
pub mod identity;
/// Security middleware (sanitizer, origin validation, rate limiting).
pub mod middleware;
/// OpenAI-compatible API endpoint (/v1/chat/completions, /v1/models, /v1/embeddings).
pub mod openai_compat;
/// DM pairing policy — gate unknown peers behind an operator-approval flow.
pub mod pairing;
/// Message routing logic.
pub mod router;

// Re-export key types for external consumers.
pub use agent_handler::AgentInboundHandler;
pub use router::InboundHandler;
/// Gmail push ingestion via Google Pub/Sub (OpenClaw parity P3.1).
#[cfg(feature = "email-push")]
pub mod gmail_push;
/// Media processing pipeline for attachments.
pub mod media;
/// In-memory metrics collection.
pub mod metrics;
/// Axum server setup and route definitions.
pub mod server;
/// Session management (user-to-agent session mapping).
pub mod session;
/// Session persistence — save/restore conversation history across restarts.
pub mod session_persistence;
/// Gateway-side [`brainwires_tools::SessionBroker`] implementation for the
/// agent's session-control tools (sessions_list / history / send / spawn).
pub mod sessions_broker;
/// In-chat slash commands (/new, /compact, /think, /usage, /trace, /status, /restart, /help).
pub mod slash;
/// Shared application state.
pub mod state;
/// TTS response processor (requires `voice` feature).
pub mod tts;
/// Built-in WebChat channel (browser-based chat UI).
pub mod webchat;
/// Webhook handler for HTTP-based channel integrations.
pub mod webhook;
/// WebSocket connection handler for channel adapters.
pub mod ws_handler;
