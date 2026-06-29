/// Agent operations (spawn, list, status, stop, await).
pub mod agent_ops;
/// Relay client implementation.
#[allow(clippy::module_inception)]
pub mod client;
/// Client error types.
pub mod error;
/// Wire protocol types for relay communication.
pub mod protocol;

pub use agent_ops::{AgentConfig, AgentInfo, AgentResult};
pub use client::AgentNetworkClient;
pub use error::AgentNetworkClientError;
