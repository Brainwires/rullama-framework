//! Push notification configuration types.

use serde::{Deserialize, Serialize};

/// Authentication details for push notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationInfo {
    /// HTTP authentication scheme (e.g. `Bearer`, `Basic`).
    pub scheme: String,
    /// Credentials (format depends on scheme).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<String>,
}

/// Push notification configuration for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPushNotificationConfig {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Configuration identifier.
    #[serde(rename = "configId", skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    /// Associated task identifier.
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// URL where the notification should be sent.
    pub url: String,
    /// Session/task-specific token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Authentication information for sending the notification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication: Option<AuthenticationInfo>,
    /// ISO 8601 timestamp when this configuration was created.
    #[serde(rename = "createdAt", skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> TaskPushNotificationConfig {
        TaskPushNotificationConfig {
            tenant: None,
            config_id: None,
            task_id: "task-1".to_string(),
            url: "https://example.com/notify".to_string(),
            token: None,
            authentication: None,
            created_at: None,
        }
    }

    #[test]
    fn minimal_config_roundtrip() {
        let cfg = minimal_config();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: TaskPushNotificationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, cfg.task_id);
        assert_eq!(back.url, cfg.url);
    }

    #[test]
    fn optional_fields_omitted_when_none() {
        let cfg = minimal_config();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(!json.contains("tenant"));
        assert!(!json.contains("configId"));
        assert!(!json.contains("token"));
        assert!(!json.contains("authentication"));
        assert!(!json.contains("createdAt"));
    }

    #[test]
    fn json_uses_camel_case_task_id() {
        let cfg = minimal_config();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("taskId"));
        assert!(!json.contains("task_id"));
    }

    #[test]
    fn with_authentication_roundtrip() {
        let mut cfg = minimal_config();
        cfg.authentication = Some(AuthenticationInfo {
            scheme: "Bearer".to_string(),
            credentials: Some("token-abc".to_string()),
        });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: TaskPushNotificationConfig = serde_json::from_str(&json).unwrap();
        let auth = back.authentication.unwrap();
        assert_eq!(auth.scheme, "Bearer");
        assert_eq!(auth.credentials.as_deref(), Some("token-abc"));
    }

    #[test]
    fn authentication_info_credentials_omitted_when_none() {
        let auth = AuthenticationInfo {
            scheme: "Basic".to_string(),
            credentials: None,
        };
        let json = serde_json::to_string(&auth).unwrap();
        assert!(!json.contains("credentials"));
    }
}
