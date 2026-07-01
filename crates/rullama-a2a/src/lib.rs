#![deny(missing_docs)]
//! # rullama-a2a
//!
//! Agent-to-Agent (A2A) protocol implementation with JSON-RPC, REST, and gRPC bindings.
//!
//! ## Features
//!
//! - `client` — HTTP client for JSON-RPC and REST (reqwest)
//! - `server` — HTTP server for JSON-RPC and REST (hyper)
//! - `native` — Both client and server (default)
//! - `grpc` — gRPC types (prost + tonic)
//! - `grpc-client` — gRPC client transport
//! - `grpc-server` — gRPC server service
//! - `full` — Everything

/// The A2A protocol version this crate targets (merged with ACP under AAIF, Dec 2025).
pub const A2A_PROTOCOL_VERSION: &str = "0.3";

// Core types (always available)
/// Agent card and capability types.
pub mod agent_card;
/// Type conversions between serde and proto types.
pub mod convert;
/// Error types and JSON-RPC error codes.
pub mod error;
/// JSON-RPC 2.0 envelopes and method constants.
pub mod jsonrpc;
/// Typed request parameter structs.
pub mod params;
/// Generated proto types (gRPC feature).
pub mod proto;
/// Push notification configuration types.
pub mod push_notification;
/// Streaming event types.
pub mod streaming;
/// Task lifecycle types: Task, TaskStatus, TaskState.
pub mod task;
/// Core message types: Message, Part, Artifact, Role.
pub mod types;

// Client (feature-gated)
/// A2A client with transport selection.
#[cfg(feature = "client")]
pub mod client;

// Server (feature-gated)
/// A2A server serving all protocol bindings.
#[cfg(feature = "server")]
pub mod server;

// Re-exports for convenience
pub use agent_card::{
    AgentCapabilities, AgentCard, AgentCardSignature, AgentExtension, AgentInterface,
    AgentProvider, AgentSkill, ApiKeySecurityScheme, AuthorizationCodeOAuthFlow,
    ClientCredentialsOAuthFlow, DeviceCodeOAuthFlow, HttpAuthSecurityScheme, ImplicitOAuthFlow,
    MutualTlsSecurityScheme, OAuth2SecurityScheme, OAuthFlows, OpenIdConnectSecurityScheme,
    PasswordOAuthFlow, SecurityRequirement, SecurityScheme,
};
pub use error::A2aError;
pub use jsonrpc::{JsonRpcRequest, JsonRpcResponse, RequestId};
pub use params::*;
pub use push_notification::{AuthenticationInfo, TaskPushNotificationConfig};
pub use streaming::{
    SendMessageResponse, StreamResponse, TaskArtifactUpdateEvent, TaskStatusUpdateEvent,
};
pub use task::{Task, TaskState, TaskStatus};
pub use types::{Artifact, Message, Part, Role};

#[cfg(feature = "client")]
pub use client::{A2aClient, Transport};

#[cfg(feature = "server")]
pub use server::{A2aHandler, A2aServer};
