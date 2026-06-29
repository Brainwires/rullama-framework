//! Bridge types — configuration, state enums, and internal structs.

use std::path::PathBuf;

/// Remote bridge configuration
///
/// All platform-specific values (version, paths) are injected via this config
/// instead of being read from CLI globals.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Backend base URL (https://...)
    pub backend_url: String,
    /// API key for authentication
    pub api_key: String,
    /// Heartbeat/poll interval in seconds
    pub heartbeat_interval_secs: u32,
    /// Reconnect delay on disconnect
    pub reconnect_delay_secs: u32,
    /// Maximum reconnect attempts (0 = unlimited)
    pub max_reconnect_attempts: u32,
    /// CLI version string (injected, replaces build_info::VERSION)
    pub version: String,
    /// Sessions directory for IPC discovery (injected, replaces PlatformPaths)
    pub sessions_dir: PathBuf,
    /// Attachment storage directory (injected, replaces PlatformPaths::data_dir())
    pub attachment_dir: PathBuf,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            backend_url: "https://brainwires.studio".to_string(),
            api_key: String::new(),
            heartbeat_interval_secs: 5,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 0,
            version: "unknown".to_string(),
            sessions_dir: PathBuf::from("/tmp/rullama-sessions"),
            attachment_dir: PathBuf::from("/tmp/rullama-attachments"),
        }
    }
}

/// Bridge state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeState {
    /// Not connected to the backend.
    Disconnected,
    /// Connection in progress.
    Connecting,
    /// Connected but not yet authenticated.
    Connected,
    /// Successfully authenticated with the backend.
    Authenticated,
    /// Gracefully shutting down.
    ShuttingDown,
}

/// Handle for an active agent subscription reader task
pub(super) struct AgentSubscription {
    /// Cancel token to stop the reader task
    pub cancel_tx: tokio::sync::oneshot::Sender<()>,
    /// Task handle
    pub task_handle: tokio::task::JoinHandle<()>,
    /// Writer for sending messages to this agent
    pub writer_tx: tokio::sync::mpsc::Sender<crate::ipc::ViewerMessage>,
}

/// Connection mode (Realtime or Polling)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionMode {
    /// Using Supabase Realtime WebSocket (preferred)
    Realtime,
    /// Using HTTP polling (fallback)
    Polling,
}

/// Realtime credentials returned by backend
#[derive(Debug, Clone)]
pub struct RealtimeCredentials {
    /// JWT token for Supabase Realtime authentication.
    pub realtime_token: String,
    /// WebSocket URL for Supabase Realtime.
    pub realtime_url: String,
    /// Channel name to subscribe to.
    pub channel_name: String,
    /// Supabase anonymous key for Kong auth.
    pub supabase_anon_key: String,
}
