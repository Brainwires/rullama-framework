//! systemd service management via `systemctl` CLI.

use anyhow::Result;

use super::safety::ServiceSafety;
use super::{ServiceInfo, ServiceOperation, ServiceStatus, ServiceType};

/// Manager for systemd services via the `systemctl` and `journalctl` CLIs.
///
/// All operations are gated by a [`ServiceSafety`] policy before execution.
pub struct SystemdManager {
    safety: ServiceSafety,
}

impl SystemdManager {
    /// Create a new systemd manager with the given safety policy.
    pub fn new(safety: ServiceSafety) -> Self {
        Self { safety }
    }

    /// Execute a service operation.
    pub async fn execute(&self, operation: &ServiceOperation) -> Result<String> {
        self.safety
            .check(operation)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        match operation {
            ServiceOperation::List => self.list_services().await,
            ServiceOperation::Status(name) => self.get_status(name).await,
            ServiceOperation::Logs { name, lines } => self.get_logs(name, *lines).await,
            ServiceOperation::Start(name) => self.start(name).await,
            ServiceOperation::Stop(name) => self.stop(name).await,
            ServiceOperation::Restart(name) => self.restart(name).await,
            _ => anyhow::bail!("Operation not supported for systemd"),
        }
    }

    async fn list_services(&self) -> Result<String> {
        let output = tokio::process::Command::new("systemctl")
            .args(["list-units", "--type=service", "--no-pager", "--plain"])
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn get_status(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("systemctl")
            .args(["status", name, "--no-pager"])
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn get_logs(&self, name: &str, lines: u32) -> Result<String> {
        let output = tokio::process::Command::new("journalctl")
            .args(["-u", name, "-n", &lines.to_string(), "--no-pager"])
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn start(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("systemctl")
            .args(["start", name])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Service {name} started"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to start {name}: {stderr}")
        }
    }

    async fn stop(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("systemctl")
            .args(["stop", name])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Service {name} stopped"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to stop {name}: {stderr}")
        }
    }

    async fn restart(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("systemctl")
            .args(["restart", name])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Service {name} restarted"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to restart {name}: {stderr}")
        }
    }

    /// Parse service status and PID from `systemctl show` output.
    pub async fn parse_status(&self, name: &str) -> Result<ServiceInfo> {
        let output = tokio::process::Command::new("systemctl")
            .args(["show", name, "--property=ActiveState,MainPID", "--no-pager"])
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut status = ServiceStatus::Unknown;
        let mut pid = None;

        for line in stdout.lines() {
            if let Some(state) = line.strip_prefix("ActiveState=") {
                status = match state {
                    "active" => ServiceStatus::Running,
                    "inactive" => ServiceStatus::Stopped,
                    "failed" => ServiceStatus::Failed,
                    _ => ServiceStatus::Unknown,
                };
            }
            if let Some(pid_str) = line.strip_prefix("MainPID=")
                && let Ok(p) = pid_str.parse::<u32>()
                && p > 0
            {
                pid = Some(p);
            }
        }

        Ok(ServiceInfo {
            name: name.to_string(),
            service_type: ServiceType::Systemd,
            status,
            pid,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn systemd_manager_module_compiles() {
        // Compilation test — actual tests require systemd
    }
}
