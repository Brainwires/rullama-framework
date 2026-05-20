//! Sandbox policies: runtime selection, resource limits, and mount whitelisting.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Mount;
use crate::error::{Result, SandboxError};

/// Which sandbox runtime to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SandboxRuntime {
    /// No isolation — runs directly on the host. Dev/testing only.
    Host,
    /// Docker (unix socket).
    #[default]
    Docker,
    /// Podman (rootless), reached over a bollard-compatible socket.
    Podman,
}

/// Network egress policy applied to the sandboxed process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NetworkPolicy {
    /// No network access at all.
    #[default]
    None,
    /// Egress restricted to the listed host names. Not fully implemented yet
    /// (see `DockerSandbox` docs); callers should treat `Limited` as a
    /// forward-compatible intent, not a guarantee.
    Limited(Vec<String>),
    /// Full network access. Dangerous — use only for trusted images.
    Full,
}

/// Runtime + resource-limit + mount-whitelist policy for a sandbox instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Runtime backend.
    pub runtime: SandboxRuntime,
    /// Container image to launch (ignored by `HostSandbox`).
    pub image: String,
    /// Egress network policy.
    pub network: NetworkPolicy,
    /// CPU cores limit (e.g. `2.0` = two cores). `None` disables the limit.
    pub cpu_limit: Option<f64>,
    /// Memory cap in megabytes. `None` disables the limit.
    pub memory_limit_mb: Option<u64>,
    /// Maximum process count inside the sandbox. `None` disables the limit.
    pub pid_limit: Option<u64>,
    /// Whether to mount the container root filesystem read-only.
    pub read_only_rootfs: bool,
    /// Optional workspace directory bind-mounted into the container.
    pub workspace_mount: Option<PathBuf>,
    /// Whitelist of host paths permitted as bind-mount sources.
    pub allowed_mount_sources: Vec<PathBuf>,
    /// Image used for the egress-allowlist proxy when
    /// [`NetworkPolicy::Limited`] is active.
    pub proxy_image: String,
    /// TCP port the proxy listens on inside its container.
    pub proxy_listen_port: u16,
    /// Optional stable name for a long-lived shared proxy container. When
    /// set, the sandbox will attach to an existing container with this name
    /// instead of spawning an ephemeral one per exec. Leave `None` for
    /// per-spawn ephemeral proxies (safer default).
    pub proxy_container_name: Option<String>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            runtime: SandboxRuntime::default(),
            image: "ghcr.io/brainwires/brainclaw-sandbox:latest".to_string(),
            network: NetworkPolicy::default(),
            cpu_limit: Some(2.0),
            memory_limit_mb: Some(1024),
            pid_limit: Some(256),
            read_only_rootfs: true,
            workspace_mount: None,
            allowed_mount_sources: Vec::new(),
            proxy_image: "ghcr.io/brainwires/brainwires-sandbox-proxy:latest".to_string(),
            proxy_listen_port: 3128,
            proxy_container_name: None,
        }
    }
}

impl SandboxPolicy {
    /// Reject `mount` unless its source is inside `workspace_mount` or one of
    /// the `allowed_mount_sources`. Also rejects any path containing `..`
    /// traversal components to prevent tool-arg host-escape.
    pub fn validate_mount(&self, mount: &Mount) -> Result<()> {
        if mount
            .source
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(SandboxError::PolicyViolation(format!(
                "mount source contains parent-dir traversal: {}",
                mount.source.display()
            )));
        }

        let allowed = self
            .allowed_mount_sources
            .iter()
            .chain(self.workspace_mount.iter())
            .any(|root| is_within(&mount.source, root));

        if !allowed {
            return Err(SandboxError::PolicyViolation(format!(
                "mount source {} is not in any allowed root",
                mount.source.display()
            )));
        }
        Ok(())
    }
}

fn is_within(candidate: &Path, root: &Path) -> bool {
    // Pure lexical prefix check — we deliberately do not canonicalize here
    // because policy is a contract over the requested paths, not the
    // filesystem state at a given moment.
    match (candidate.is_absolute(), root.is_absolute()) {
        (true, true) | (false, false) => candidate.starts_with(root),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn policy_with(allowed: Vec<PathBuf>) -> SandboxPolicy {
        SandboxPolicy {
            allowed_mount_sources: allowed,
            ..SandboxPolicy::default()
        }
    }

    #[test]
    fn allows_paths_inside_whitelist() {
        let p = policy_with(vec![PathBuf::from("/workspace")]);
        let m = Mount {
            source: PathBuf::from("/workspace/project"),
            target: PathBuf::from("/mnt/project"),
            read_only: true,
        };
        assert!(p.validate_mount(&m).is_ok());
    }

    #[test]
    fn rejects_etc_passwd() {
        let p = policy_with(vec![PathBuf::from("/workspace")]);
        let m = Mount {
            source: PathBuf::from("/etc/passwd"),
            target: PathBuf::from("/mnt/passwd"),
            read_only: true,
        };
        let err = p.validate_mount(&m).unwrap_err();
        assert!(matches!(err, SandboxError::PolicyViolation(_)));
    }

    #[test]
    fn rejects_parent_dir_traversal() {
        let p = policy_with(vec![PathBuf::from("/workspace")]);
        let m = Mount {
            source: PathBuf::from("/workspace/../etc"),
            target: PathBuf::from("/mnt/etc"),
            read_only: true,
        };
        let err = p.validate_mount(&m).unwrap_err();
        assert!(matches!(err, SandboxError::PolicyViolation(_)));
    }

    #[test]
    fn workspace_mount_is_implicitly_allowed() {
        let p = SandboxPolicy {
            workspace_mount: Some(PathBuf::from("/ws")),
            ..SandboxPolicy::default()
        };
        let m = Mount {
            source: PathBuf::from("/ws/sub"),
            target: PathBuf::from("/mnt/sub"),
            read_only: false,
        };
        assert!(p.validate_mount(&m).is_ok());
    }
}
