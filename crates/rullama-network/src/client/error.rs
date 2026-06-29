/// Errors from the relay client.
#[derive(Debug, thiserror::Error)]
pub enum AgentNetworkClientError {
    /// Failed to spawn the relay subprocess.
    #[error("Failed to spawn relay process: {0}")]
    SpawnFailed(#[source] std::io::Error),
    /// The relay process exited unexpectedly.
    #[error("Relay process exited unexpectedly")]
    ProcessExited,
    /// Protocol-level error.
    #[error("Protocol error: {0}")]
    Protocol(String),
    /// JSON-RPC error returned by the server.
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc {
        /// JSON-RPC error code.
        code: i32,
        /// Error message.
        message: String,
    },
    /// Request timed out.
    #[error("Timeout after {0} seconds")]
    Timeout(u64),
    /// Client was not initialized before use.
    #[error("Not initialized - call initialize() first")]
    NotInitialized,
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON serialization/deserialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
