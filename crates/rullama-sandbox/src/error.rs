//! Error types for the sandbox crate.

use thiserror::Error;

/// Errors returned by sandbox operations.
#[derive(Debug, Error)]
pub enum SandboxError {
    /// Underlying I/O failure (process spawn, socket, filesystem).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Docker / Podman daemon returned an error.
    #[cfg(feature = "docker")]
    #[error("docker error: {0}")]
    Docker(String),

    /// The sandboxed process exceeded its wall-clock budget.
    #[error("sandbox execution timed out")]
    Timeout,

    /// A `SandboxPolicy` rule rejected the requested operation.
    #[error("policy violation: {0}")]
    PolicyViolation(String),

    /// The sandboxed process exited with a non-zero status.
    #[error("sandbox process exited with code {code}: {stderr}")]
    ExitFailure {
        /// Exit code reported by the container/process.
        code: i32,
        /// Captured stderr (UTF-8 lossy) at the time of failure.
        stderr: String,
    },

    /// A requested runtime or feature is not available in this build/environment.
    #[error("not available: {0}")]
    NotAvailable(String),
}

#[cfg(feature = "docker")]
impl From<bollard::errors::Error> for SandboxError {
    fn from(e: bollard::errors::Error) -> Self {
        SandboxError::Docker(e.to_string())
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SandboxError>;
