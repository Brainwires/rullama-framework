//! System service management — systemd, docker, and process control.
//!
//! Provides controlled access to system services with strict safety
//! policies: default read-only, explicit allow-lists, and hardcoded
//! deny-lists for critical system services.

pub mod docker;
pub mod process;
pub mod safety;
pub mod systemd;

pub use docker::DockerManager;
pub use process::ProcessManager;
pub use safety::{CRITICAL_SERVICES, ServiceSafety};
pub use systemd::SystemdManager;

use serde::{Deserialize, Serialize};

/// Type of service being managed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceType {
    /// systemd unit.
    Systemd,
    /// Docker container.
    DockerContainer,
    /// Docker Compose stack.
    DockerCompose,
    /// System process.
    Process,
}

impl std::fmt::Display for ServiceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Systemd => write!(f, "systemd"),
            Self::DockerContainer => write!(f, "docker"),
            Self::DockerCompose => write!(f, "docker-compose"),
            Self::Process => write!(f, "process"),
        }
    }
}

/// Status of a service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceStatus {
    /// Service is running.
    Running,
    /// Service is stopped.
    Stopped,
    /// Service is in a failed state.
    Failed,
    /// Status could not be determined.
    Unknown,
}

/// Information about a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    /// Service name.
    pub name: String,
    /// Type of service.
    pub service_type: ServiceType,
    /// Current status.
    pub status: ServiceStatus,
    /// Process ID, if running.
    pub pid: Option<u32>,
}

/// Operations that can be performed on a service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceOperation {
    /// List all services.
    List,
    /// Get status of a specific service.
    Status(String),
    /// Get logs for a service.
    Logs {
        /// Service name.
        name: String,
        /// Number of lines to retrieve.
        lines: u32,
    },
    /// Start a service.
    Start(String),
    /// Stop a service.
    Stop(String),
    /// Restart a service.
    Restart(String),
    /// Start a docker-compose stack.
    DockerComposeUp {
        /// Path to docker-compose file.
        path: String,
    },
    /// Stop a docker-compose stack.
    DockerComposeDown {
        /// Path to docker-compose file.
        path: String,
    },
}

impl ServiceOperation {
    /// Returns true if this operation only reads state (no mutations).
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::List | Self::Status(_) | Self::Logs { .. })
    }

    /// Get the service name for this operation, if applicable.
    pub fn service_name(&self) -> Option<&str> {
        match self {
            Self::Status(n) | Self::Start(n) | Self::Stop(n) | Self::Restart(n) => Some(n),
            Self::Logs { name, .. } => Some(name),
            Self::DockerComposeUp { path } | Self::DockerComposeDown { path } => Some(path),
            Self::List => None,
        }
    }
}
