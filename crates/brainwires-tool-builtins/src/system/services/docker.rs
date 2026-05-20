//! Docker container and docker-compose management via CLI.

use anyhow::Result;

use super::safety::ServiceSafety;
use super::{ServiceInfo, ServiceOperation, ServiceStatus, ServiceType};

/// Manager for Docker containers and compose stacks via the `docker` CLI.
///
/// All operations are gated by a [`ServiceSafety`] policy before execution.
pub struct DockerManager {
    safety: ServiceSafety,
}

impl DockerManager {
    /// Create a new Docker manager with the given safety policy.
    pub fn new(safety: ServiceSafety) -> Self {
        Self { safety }
    }

    /// Execute a service operation.
    pub async fn execute(&self, operation: &ServiceOperation) -> Result<String> {
        self.safety
            .check(operation)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        match operation {
            ServiceOperation::List => self.list_containers().await,
            ServiceOperation::Status(name) => self.get_status(name).await,
            ServiceOperation::Logs { name, lines } => self.get_logs(name, *lines).await,
            ServiceOperation::Start(name) => self.start(name).await,
            ServiceOperation::Stop(name) => self.stop(name).await,
            ServiceOperation::Restart(name) => self.restart(name).await,
            ServiceOperation::DockerComposeUp { path } => self.compose_up(path).await,
            ServiceOperation::DockerComposeDown { path } => self.compose_down(path).await,
        }
    }

    async fn list_containers(&self) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args([
                "ps",
                "--format",
                "table {{.Names}}\t{{.Status}}\t{{.Image}}",
            ])
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn get_status(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["inspect", "--format", "{{.State.Status}}", name])
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    async fn get_logs(&self, name: &str, lines: u32) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["logs", "--tail", &lines.to_string(), name])
            .output()
            .await?;
        // Docker logs go to both stdout and stderr
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!("{stdout}{stderr}"))
    }

    async fn start(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["start", name])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Container {name} started"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to start {name}: {stderr}")
        }
    }

    async fn stop(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["stop", name])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Container {name} stopped"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to stop {name}: {stderr}")
        }
    }

    async fn restart(&self, name: &str) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["restart", name])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Container {name} restarted"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to restart {name}: {stderr}")
        }
    }

    async fn compose_up(&self, path: &str) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["compose", "-f", path, "up", "-d"])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Compose stack started: {path}"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to start compose stack: {stderr}")
        }
    }

    async fn compose_down(&self, path: &str) -> Result<String> {
        let output = tokio::process::Command::new("docker")
            .args(["compose", "-f", path, "down"])
            .output()
            .await?;
        if output.status.success() {
            Ok(format!("Compose stack stopped: {path}"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to stop compose stack: {stderr}")
        }
    }

    /// Parse container status and PID from `docker inspect` output.
    pub async fn parse_container_info(&self, name: &str) -> Result<ServiceInfo> {
        let output = tokio::process::Command::new("docker")
            .args([
                "inspect",
                "--format",
                "{{.State.Status}} {{.State.Pid}}",
                name,
            ])
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.split_whitespace().collect();

        let status = match parts.first().copied() {
            Some("running") => ServiceStatus::Running,
            Some("exited") => ServiceStatus::Stopped,
            Some("dead") | Some("removing") => ServiceStatus::Failed,
            _ => ServiceStatus::Unknown,
        };

        let pid = parts
            .get(1)
            .and_then(|s| s.parse().ok())
            .filter(|&p: &u32| p > 0);

        Ok(ServiceInfo {
            name: name.to_string(),
            service_type: ServiceType::DockerContainer,
            status,
            pid,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn docker_manager_module_compiles() {
        // Compilation test — actual tests require docker
    }
}
