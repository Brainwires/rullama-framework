//! Service-specific safety policies.

use std::collections::HashSet;

use crate::system::config::ServiceConfig;

use super::ServiceOperation;

/// Hardcoded list of critical system services that can never be managed.
pub const CRITICAL_SERVICES: &[&str] = &[
    "sshd",
    "ssh",
    "systemd-journald",
    "systemd-logind",
    "systemd-networkd",
    "systemd-resolved",
    "systemd-timesyncd",
    "systemd-udevd",
    "dbus",
    "dbus-broker",
    "NetworkManager",
    "polkit",
    "accounts-daemon",
    "login",
    "getty",
    "init",
    "kernel",
];

/// Service safety enforcer that gates operations against allow-lists, deny-lists,
/// and read-only mode.
pub struct ServiceSafety {
    allowed: HashSet<String>,
    forbidden: HashSet<String>,
    read_only: bool,
}

impl ServiceSafety {
    /// Create from configuration.
    pub fn from_config(config: &ServiceConfig) -> Self {
        let mut forbidden: HashSet<String> = config.forbidden_services.iter().cloned().collect();

        // Always include critical services in the deny-list
        for &svc in CRITICAL_SERVICES {
            forbidden.insert(svc.to_string());
        }

        Self {
            allowed: config.allowed_services.iter().cloned().collect(),
            forbidden,
            read_only: config.read_only,
        }
    }

    /// Check if an operation is allowed by the safety policy.
    ///
    /// Read-only operations are always permitted (unless the service is forbidden).
    /// Write operations require `read_only=false` and the service to be in the allow-list.
    pub fn check(&self, operation: &ServiceOperation) -> Result<(), String> {
        // Read-only operations are always allowed (unless service is forbidden)
        if operation.is_read_only() {
            if let Some(name) = operation.service_name()
                && self.is_forbidden(name)
            {
                return Err(format!(
                    "Service '{name}' is in the deny-list and cannot be accessed"
                ));
            }
            return Ok(());
        }

        // Write operations require explicit opt-in
        if self.read_only {
            return Err(
                "Service management is read-only. Set read_only=false in ServiceConfig to enable write operations".to_string()
            );
        }

        // Check service name against allow/deny lists
        if let Some(name) = operation.service_name() {
            if self.is_forbidden(name) {
                return Err(format!(
                    "Service '{name}' is a critical system service and cannot be managed"
                ));
            }

            if !self.allowed.is_empty() && !self.allowed.contains(name) {
                return Err(format!("Service '{name}' is not in the allow-list"));
            }
        }

        Ok(())
    }

    fn is_forbidden(&self, name: &str) -> bool {
        self.forbidden.contains(name)
            || CRITICAL_SERVICES
                .iter()
                .any(|&critical| name == critical || name.starts_with(&format!("{critical}@")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_safety() -> ServiceSafety {
        ServiceSafety::from_config(&ServiceConfig::default())
    }

    #[test]
    fn read_only_operations_allowed_by_default() {
        let safety = default_safety();
        assert!(safety.check(&ServiceOperation::List).is_ok());
        assert!(
            safety
                .check(&ServiceOperation::Status("myapp".to_string()))
                .is_ok()
        );
    }

    #[test]
    fn write_operations_blocked_by_default() {
        let safety = default_safety();
        assert!(
            safety
                .check(&ServiceOperation::Start("myapp".to_string()))
                .is_err()
        );
        assert!(
            safety
                .check(&ServiceOperation::Stop("myapp".to_string()))
                .is_err()
        );
    }

    #[test]
    fn critical_services_always_blocked() {
        let config = ServiceConfig {
            read_only: false,
            allowed_services: vec!["sshd".to_string()], // even if allowed
            ..Default::default()
        };
        let safety = ServiceSafety::from_config(&config);
        assert!(
            safety
                .check(&ServiceOperation::Restart("sshd".to_string()))
                .is_err()
        );
    }

    #[test]
    fn allowed_services_can_be_managed() {
        let config = ServiceConfig {
            read_only: false,
            allowed_services: vec!["myapp".to_string()],
            ..Default::default()
        };
        let safety = ServiceSafety::from_config(&config);
        assert!(
            safety
                .check(&ServiceOperation::Restart("myapp".to_string()))
                .is_ok()
        );
        assert!(
            safety
                .check(&ServiceOperation::Restart("other".to_string()))
                .is_err()
        );
    }

    #[test]
    fn empty_allow_list_allows_all_non_critical() {
        let config = ServiceConfig {
            read_only: false,
            allowed_services: vec![],
            ..Default::default()
        };
        let safety = ServiceSafety::from_config(&config);
        assert!(
            safety
                .check(&ServiceOperation::Restart("myapp".to_string()))
                .is_ok()
        );
    }
}
