//! Process management — list, signal, and spawn system processes.

use anyhow::Result;

use super::{ServiceInfo, ServiceStatus, ServiceType};

/// Manager for system processes using `ps`, `pgrep`, and `kill -0`.
pub struct ProcessManager;

impl ProcessManager {
    /// List running processes.
    pub async fn list() -> Result<String> {
        let output = tokio::process::Command::new("ps")
            .args(["aux", "--sort=-%mem"])
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get info about a specific process by name.
    pub async fn find_by_name(name: &str) -> Result<Vec<ServiceInfo>> {
        let output = tokio::process::Command::new("pgrep")
            .args(["-a", name])
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut results = Vec::new();

        for line in stdout.lines() {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if let Some(pid_str) = parts.first()
                && let Ok(pid) = pid_str.parse::<u32>()
            {
                results.push(ServiceInfo {
                    name: parts.get(1).unwrap_or(&name).to_string(),
                    service_type: ServiceType::Process,
                    status: ServiceStatus::Running,
                    pid: Some(pid),
                });
            }
        }

        Ok(results)
    }

    /// Check if a process with the given PID is running.
    pub async fn is_running(pid: u32) -> bool {
        tokio::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn process_manager_module_compiles() {
        // Compilation test
    }
}
