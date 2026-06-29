//! Typed request parameter structs for all A2A methods.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::agent_card::AgentCard;
use crate::push_notification::TaskPushNotificationConfig;
use crate::task::{Task, TaskState};
use crate::types::Message;

/// Configuration for a send-message request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SendMessageConfiguration {
    /// Accepted output media types.
    #[serde(
        rename = "acceptedOutputModes",
        skip_serializing_if = "Option::is_none"
    )]
    pub accepted_output_modes: Option<Vec<String>>,
    /// Push notification configuration.
    #[serde(
        rename = "taskPushNotificationConfig",
        skip_serializing_if = "Option::is_none"
    )]
    pub task_push_notification_config: Option<TaskPushNotificationConfig>,
    /// Max number of history messages to return.
    #[serde(rename = "historyLength", skip_serializing_if = "Option::is_none")]
    pub history_length: Option<i32>,
    /// If true, return immediately without waiting for terminal/interrupted state.
    #[serde(rename = "returnImmediately", skip_serializing_if = "Option::is_none")]
    pub return_immediately: Option<bool>,
}

/// Request parameters for `message/send` and `message/stream`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendMessageRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// The message to send.
    pub message: Message,
    /// Request configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<SendMessageConfiguration>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Request parameters for `tasks/get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTaskRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Task identifier.
    pub id: String,
    /// Max number of history messages to return.
    #[serde(rename = "historyLength", skip_serializing_if = "Option::is_none")]
    pub history_length: Option<i32>,
}

/// Request parameters for `tasks/list`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListTasksRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Filter by context ID.
    #[serde(rename = "contextId", skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// Filter by task state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<TaskState>,
    /// Maximum number of tasks to return.
    #[serde(rename = "pageSize", skip_serializing_if = "Option::is_none")]
    pub page_size: Option<i32>,
    /// Pagination token.
    #[serde(rename = "pageToken", skip_serializing_if = "Option::is_none")]
    pub page_token: Option<String>,
    /// Max history messages per task.
    #[serde(rename = "historyLength", skip_serializing_if = "Option::is_none")]
    pub history_length: Option<i32>,
    /// Filter tasks with status updated after this ISO 8601 timestamp.
    #[serde(
        rename = "statusTimestampAfter",
        skip_serializing_if = "Option::is_none"
    )]
    pub status_timestamp_after: Option<String>,
    /// Whether to include artifacts.
    #[serde(rename = "includeArtifacts", skip_serializing_if = "Option::is_none")]
    pub include_artifacts: Option<bool>,
}

/// Response for `tasks/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTasksResponse {
    /// Matching tasks.
    pub tasks: Vec<Task>,
    /// Pagination token for next page.
    #[serde(rename = "nextPageToken")]
    pub next_page_token: String,
    /// Page size used.
    #[serde(rename = "pageSize")]
    pub page_size: i32,
    /// Total number of matching tasks.
    #[serde(rename = "totalSize")]
    pub total_size: i32,
}

/// Request parameters for `tasks/cancel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelTaskRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Task identifier.
    pub id: String,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Request parameters for `tasks/resubscribe`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeToTaskRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Task identifier.
    pub id: String,
}

/// Request for `tasks/pushNotificationConfig/get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTaskPushNotificationConfigRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Parent task identifier.
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// Configuration identifier.
    #[serde(rename = "configId")]
    pub config_id: String,
}

/// Request for `tasks/pushNotificationConfig/delete`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteTaskPushNotificationConfigRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Parent task identifier.
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// Configuration identifier.
    #[serde(rename = "configId")]
    pub config_id: String,
}

/// Request for `tasks/pushNotificationConfig/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTaskPushNotificationConfigsRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Parent task identifier.
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// Maximum configs to return.
    #[serde(rename = "pageSize", skip_serializing_if = "Option::is_none")]
    pub page_size: Option<i32>,
    /// Pagination token.
    #[serde(rename = "pageToken", skip_serializing_if = "Option::is_none")]
    pub page_token: Option<String>,
}

/// Response for listing push notification configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTaskPushNotificationConfigsResponse {
    /// The configs.
    pub configs: Vec<TaskPushNotificationConfig>,
    /// Pagination token for next page.
    #[serde(rename = "nextPageToken", skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// Request for `agent/authenticatedExtendedCard`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GetExtendedAgentCardRequest {
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

/// Response for the extended agent card.
pub type GetExtendedAgentCardResponse = AgentCard;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskState;
    use crate::types::Message;

    // --- SendMessageRequest ---

    #[test]
    fn send_message_request_roundtrip() {
        let req = SendMessageRequest {
            tenant: None,
            message: Message::user_text("hello"),
            configuration: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: SendMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message.parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn send_message_request_optional_fields_omitted() {
        let req = SendMessageRequest {
            tenant: None,
            message: Message::user_text("test"),
            configuration: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tenant"));
        assert!(!json.contains("configuration"));
        assert!(!json.contains("metadata"));
    }

    #[test]
    fn send_message_configuration_defaults_empty() {
        let cfg = SendMessageConfiguration::default();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(!json.contains("acceptedOutputModes"));
        assert!(!json.contains("historyLength"));
        assert!(!json.contains("returnImmediately"));
    }

    // --- GetTaskRequest ---

    #[test]
    fn get_task_request_roundtrip() {
        let req = GetTaskRequest {
            tenant: None,
            id: "task-xyz".to_string(),
            history_length: Some(10),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GetTaskRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "task-xyz");
        assert_eq!(back.history_length, Some(10));
    }

    #[test]
    fn get_task_request_json_uses_camel_case() {
        let req = GetTaskRequest {
            tenant: None,
            id: "t1".to_string(),
            history_length: Some(5),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("historyLength"));
        assert!(!json.contains("history_length"));
    }

    // --- ListTasksRequest ---

    #[test]
    fn list_tasks_request_defaults_all_none() {
        let req = ListTasksRequest::default();
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("contextId"));
        assert!(!json.contains("status"));
        assert!(!json.contains("pageSize"));
    }

    #[test]
    fn list_tasks_request_with_filter_roundtrip() {
        let req = ListTasksRequest {
            tenant: None,
            context_id: Some("ctx-1".to_string()),
            status: Some(TaskState::Completed),
            page_size: Some(20),
            page_token: None,
            history_length: None,
            status_timestamp_after: None,
            include_artifacts: Some(true),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ListTasksRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.context_id.as_deref(), Some("ctx-1"));
        assert_eq!(back.status, Some(TaskState::Completed));
        assert_eq!(back.page_size, Some(20));
    }

    // --- CancelTaskRequest ---

    #[test]
    fn cancel_task_request_roundtrip() {
        let req = CancelTaskRequest {
            tenant: None,
            id: "task-cancel-me".to_string(),
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CancelTaskRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "task-cancel-me");
    }

    // --- SubscribeToTaskRequest ---

    #[test]
    fn subscribe_to_task_request_roundtrip() {
        let req = SubscribeToTaskRequest {
            tenant: None,
            id: "task-sub".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: SubscribeToTaskRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "task-sub");
    }

    // --- Push notification config requests ---

    #[test]
    fn get_push_config_request_roundtrip() {
        let req = GetTaskPushNotificationConfigRequest {
            tenant: None,
            task_id: "t1".to_string(),
            config_id: "cfg-1".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GetTaskPushNotificationConfigRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "t1");
        assert_eq!(back.config_id, "cfg-1");
    }

    #[test]
    fn delete_push_config_request_json_camel_case() {
        let req = DeleteTaskPushNotificationConfigRequest {
            tenant: None,
            task_id: "t1".to_string(),
            config_id: "cfg-1".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("taskId"));
        assert!(json.contains("configId"));
    }

    #[test]
    fn list_push_configs_request_roundtrip() {
        let req = ListTaskPushNotificationConfigsRequest {
            tenant: None,
            task_id: "t1".to_string(),
            page_size: Some(5),
            page_token: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ListTaskPushNotificationConfigsRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "t1");
        assert_eq!(back.page_size, Some(5));
    }

    // --- ListTasksResponse ---

    #[test]
    fn list_tasks_response_roundtrip() {
        use crate::task::{Task, TaskStatus};
        let resp = ListTasksResponse {
            tasks: vec![Task {
                id: "t1".to_string(),
                context_id: None,
                status: TaskStatus {
                    state: TaskState::Completed,
                    message: None,
                    timestamp: None,
                },
                artifacts: None,
                history: None,
                metadata: None,
            }],
            next_page_token: "tok".to_string(),
            page_size: 10,
            total_size: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: ListTasksResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tasks.len(), 1);
        assert_eq!(back.total_size, 1);
    }
}
