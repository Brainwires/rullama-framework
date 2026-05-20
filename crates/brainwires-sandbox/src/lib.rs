#![deny(missing_docs)]
//! Container-based sandboxing for tool execution in the Brainwires framework.
//!
//! This crate provides a [`Sandbox`] trait with Docker/Podman implementations
//! backed by [bollard](https://crates.io/crates/bollard). Policy is expressed
//! via [`SandboxPolicy`]: resource limits (CPU, memory, PIDs), a read-only
//! root filesystem, network egress controls, and a whitelist of allowed
//! bind-mount source paths.
//!
//! The `HostSandbox` implementation (behind the `unsafe-host` feature) is
//! intentionally dangerous — it is a no-op pass-through that runs the
//! requested command directly on the host with no isolation, intended for
//! development only. Production callers should use [`DockerSandbox`].
//!
//! # Example
//!
//! ```ignore
//! use brainwires_sandbox::{DockerSandbox, ExecSpec, SandboxPolicy, Sandbox};
//! use std::collections::BTreeMap;
//! use std::path::PathBuf;
//! use std::time::Duration;
//!
//! # async fn demo() -> brainwires_sandbox::Result<()> {
//! let sandbox = DockerSandbox::connect(SandboxPolicy::default())?;
//! let spec = ExecSpec {
//!     cmd: vec!["echo".into(), "hello".into()],
//!     env: BTreeMap::new(),
//!     workdir: PathBuf::from("/"),
//!     stdin: None,
//!     mounts: vec![],
//!     timeout: Duration::from_secs(10),
//! };
//! let handle = sandbox.spawn(spec).await?;
//! let out = sandbox.wait(handle).await?;
//! assert_eq!(out.exit_code, 0);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod error;
mod policy;

#[cfg(feature = "docker")]
mod docker;
#[cfg(feature = "unsafe-host")]
mod host;

pub use error::{Result, SandboxError};
pub use policy::{NetworkPolicy, SandboxPolicy, SandboxRuntime};

#[cfg(feature = "docker")]
pub use docker::DockerSandbox;
#[cfg(feature = "unsafe-host")]
pub use host::HostSandbox;

/// A bind-mount request from the host filesystem into the sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    /// Source path on the host.
    pub source: PathBuf,
    /// Mount target inside the sandbox.
    pub target: PathBuf,
    /// Whether the mount is read-only inside the sandbox.
    pub read_only: bool,
}

/// A single sandboxed process to launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecSpec {
    /// Command and arguments. `cmd[0]` is the program.
    pub cmd: Vec<String>,
    /// Environment variables passed into the sandbox. The host environment is
    /// never inherited.
    pub env: BTreeMap<String, String>,
    /// Working directory inside the sandbox.
    pub workdir: PathBuf,
    /// Optional stdin bytes to feed to the process.
    pub stdin: Option<Vec<u8>>,
    /// Bind mounts; each is validated by the active [`SandboxPolicy`].
    pub mounts: Vec<Mount>,
    /// Wall-clock timeout for the execution.
    pub timeout: Duration,
}

/// Opaque identifier for a running sandboxed process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecHandle(Uuid);

impl ExecHandle {
    /// Mint a fresh handle.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Inner UUID, for logging/debugging.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for ExecHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Captured output of a completed sandbox execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecOutput {
    /// Exit code returned by the process.
    pub exit_code: i32,
    /// Captured stdout bytes.
    pub stdout: Vec<u8>,
    /// Captured stderr bytes.
    pub stderr: Vec<u8>,
    /// Measured wall-clock execution time.
    pub wall_time: Duration,
}

/// Abstraction over a sandbox runtime.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Launch a process described by `spec`.
    async fn spawn(&self, spec: ExecSpec) -> Result<ExecHandle>;

    /// Wait for `handle` to terminate and collect its output.
    async fn wait(&self, handle: ExecHandle) -> Result<ExecOutput>;

    /// Stop and clean up any resources owned by this sandbox.
    async fn shutdown(&self) -> Result<()>;

    /// Which runtime this implementation targets.
    fn runtime(&self) -> SandboxRuntime;
}
