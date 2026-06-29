//! Trait abstractions for decoupling bridge crate from CLI-specific types
//!
//! These traits allow the bridge crate to be used as a standalone framework
//! library, with CLI-specific implementations injected at runtime.

use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;
use zeroize::Zeroizing;

use crate::ipc::protocol::AgentMetadata;

// ============================================================================
// 1. Path Resolution (replaces PlatformPaths static methods)
// ============================================================================

/// Provides platform-specific directory paths for IPC sessions and data storage.
///
/// CLI implements this wrapping `PlatformPaths` static methods.
pub trait SessionDir: Send + Sync {
    /// Directory where agent session files (.sock, .token, .meta.json) are stored
    fn sessions_dir(&self) -> Result<PathBuf>;

    /// Path to the authentication session file (session.json)
    fn session_file(&self) -> Result<PathBuf>;

    /// Base data directory for application data
    fn data_dir(&self) -> Result<PathBuf>;
}

// ============================================================================
// 2. Secure Key Storage (replaces keyring module coupling)
// ============================================================================

/// Trait for secure credential storage (system keyring, encrypted file, etc).
///
/// CLI implements this with the `keyring` crate (GNOME Keyring, macOS Keychain,
/// Windows Credential Manager).
pub trait KeyStore: Send + Sync {
    /// Store a key for a given user ID
    fn store_key(&self, user_id: &str, key: &str) -> Result<()>;

    /// Retrieve a key for a given user ID (returns Zeroizing for secure memory clearing)
    fn get_key(&self, user_id: &str) -> Result<Option<Zeroizing<String>>>;

    /// Delete the key for a given user ID
    fn delete_key(&self, user_id: &str) -> Result<()>;

    /// Check if the key store backend is available on this system
    fn is_available(&self) -> bool;
}

// ============================================================================
// 3. Backend URL Configuration (replaces config::constants)
// ============================================================================

/// Provides authentication endpoint configuration.
///
/// CLI implements this reading from `config::constants`.
pub trait AuthEndpoints: Send + Sync {
    /// Full URL for the CLI authentication endpoint (e.g., `https://brainwires.studio/api/cli/auth`)
    fn auth_endpoint(&self) -> String;

    /// Regex pattern for validating API key format
    fn api_key_pattern(&self) -> &str;

    /// Base backend URL (e.g., `https://brainwires.studio`)
    fn backend_url(&self) -> &str;
}

// ============================================================================
// 4. Agent Process Creation (replaces spawn_agent_process_with_options)
// ============================================================================

/// Trait for spawning new agent processes.
///
/// CLI implements this wrapping `spawn_agent_process_with_options`.
#[async_trait]
pub trait AgentSpawner: Send + Sync {
    /// Spawn a new agent process and return the socket path.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier for the new agent
    /// * `model` - Optional AI model to use
    /// * `working_directory` - Optional working directory for the agent
    async fn spawn_agent(
        &self,
        session_id: &str,
        model: Option<String>,
        working_directory: Option<PathBuf>,
    ) -> Result<PathBuf>;
}

// ============================================================================
// 5. Agent Discovery (replaces IPC socket statics bound to PlatformPaths)
// ============================================================================

/// Trait for discovering and managing running agents.
///
/// CLI implements this by delegating to the bridge's discovery module
/// with `PlatformPaths::sessions_dir()` injected.
#[async_trait]
pub trait AgentDiscovery: Send + Sync {
    /// List all agents with their metadata
    fn list_agents_with_metadata(&self) -> Result<Vec<AgentMetadata>>;

    /// Clean up stale agent sockets (dead processes)
    async fn cleanup_stale(&self) -> Result<()>;

    /// Check if an agent is alive and accepting connections
    async fn is_agent_alive(&self, session_id: &str) -> bool;
}

// ============================================================================
// 6. Remote Bridge Configuration (replaces ConfigManager + SessionManager coupling)
// ============================================================================

/// Configuration for the remote bridge (extracted from CLI's RemoteSettings + BridgeConfig)
#[derive(Debug, Clone)]
pub struct RemoteBridgeConfig {
    /// Backend base URL
    pub backend_url: String,
    /// API key for authentication
    pub api_key: String,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u32,
    /// Reconnect delay in seconds
    pub reconnect_delay_secs: u32,
    /// Maximum reconnect attempts (0 = unlimited)
    pub max_reconnect_attempts: u32,
}

/// Provides remote bridge configuration and API key access.
///
/// CLI implements this using `ConfigManager` + `SessionManager`.
pub trait BridgeConfigProvider: Send + Sync {
    /// Get the remote bridge configuration (returns None if not enabled)
    fn get_remote_config(&self) -> Result<Option<RemoteBridgeConfig>>;

    /// Get the API key (from keyring or session, Zeroizing for secure memory)
    fn get_api_key(&self) -> Result<Option<Zeroizing<String>>>;
}
