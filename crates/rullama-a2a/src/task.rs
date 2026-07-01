//! Task lifecycle types: Task, TaskStatus, TaskState.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{Artifact, Message};

/// Possible lifecycle states of a Task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// Unspecified or indeterminate state.
    #[serde(rename = "TASK_STATE_UNSPECIFIED")]
    Unspecified,
    /// Task has been submitted and acknowledged.
    #[serde(rename = "TASK_STATE_SUBMITTED")]
    Submitted,
    /// Task is actively being processed.
    #[serde(rename = "TASK_STATE_WORKING")]
    Working,
    /// Task finished successfully (terminal).
    #[serde(rename = "TASK_STATE_COMPLETED")]
    Completed,
    /// Task finished with an error (terminal).
    #[serde(rename = "TASK_STATE_FAILED")]
    Failed,
    /// Task was canceled (terminal).
    #[serde(rename = "TASK_STATE_CANCELED")]
    Canceled,
    /// Task was rejected by the agent (terminal).
    #[serde(rename = "TASK_STATE_REJECTED")]
    Rejected,
    /// Agent requires additional user input (interrupted).
    #[serde(rename = "TASK_STATE_INPUT_REQUIRED")]
    InputRequired,
    /// Authentication is required to proceed (interrupted).
    #[serde(rename = "TASK_STATE_AUTH_REQUIRED")]
    AuthRequired,
}

/// Current status of a task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskStatus {
    /// Current state.
    pub state: TaskState,
    /// Optional message associated with the status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    /// ISO 8601 timestamp when the status was recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// The core unit of action in A2A.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Task {
    /// Unique task identifier (UUID).
    pub id: String,
    /// Context identifier for the conversation/session.
    #[serde(rename = "contextId", skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// Current task status.
    pub status: TaskStatus,
    /// Output artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Vec<Artifact>>,
    /// History of interactions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<Message>>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submitted_task(id: &str) -> Task {
        Task {
            id: id.to_string(),
            context_id: None,
            status: TaskStatus {
                state: TaskState::Submitted,
                message: None,
                timestamp: None,
            },
            artifacts: None,
            history: None,
            metadata: None,
        }
    }

    // --- TaskState ---

    #[test]
    fn task_state_serializes_to_uppercase_snake() {
        assert_eq!(
            serde_json::to_string(&TaskState::Submitted).unwrap(),
            r#""TASK_STATE_SUBMITTED""#
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Completed).unwrap(),
            r#""TASK_STATE_COMPLETED""#
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Failed).unwrap(),
            r#""TASK_STATE_FAILED""#
        );
    }

    #[test]
    fn all_task_states_roundtrip() {
        let states = [
            TaskState::Unspecified,
            TaskState::Submitted,
            TaskState::Working,
            TaskState::Completed,
            TaskState::Failed,
            TaskState::Canceled,
            TaskState::Rejected,
            TaskState::InputRequired,
            TaskState::AuthRequired,
        ];
        for state in states {
            let json = serde_json::to_string(&state).unwrap();
            let back: TaskState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }
    }

    // --- TaskStatus ---

    #[test]
    fn task_status_roundtrip() {
        let status = TaskStatus {
            state: TaskState::Working,
            message: None,
            timestamp: Some("2024-01-01T00:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&status).unwrap();
        let back: TaskStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, status);
    }

    #[test]
    fn task_status_omits_optional_fields_when_none() {
        let status = TaskStatus {
            state: TaskState::Submitted,
            message: None,
            timestamp: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(!json.contains("message"));
        assert!(!json.contains("timestamp"));
    }

    // --- Task ---

    #[test]
    fn task_roundtrip_minimal() {
        let task = submitted_task("task-001");
        let json = serde_json::to_string(&task).unwrap();
        let back: Task = serde_json::from_str(&json).unwrap();
        assert_eq!(back, task);
    }

    #[test]
    fn task_optional_fields_omitted_when_none() {
        let task = submitted_task("t1");
        let json = serde_json::to_string(&task).unwrap();
        assert!(!json.contains("contextId"));
        assert!(!json.contains("artifacts"));
        assert!(!json.contains("history"));
        assert!(!json.contains("metadata"));
    }

    #[test]
    fn task_json_uses_camel_case_context_id() {
        let mut task = submitted_task("t2");
        task.context_id = Some("ctx-1".to_string());
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("contextId"));
        assert!(!json.contains("context_id"));
    }

    #[test]
    fn task_with_history_roundtrip() {
        use crate::types::Message;
        let mut task = submitted_task("t3");
        task.history = Some(vec![Message::user_text("hello")]);
        let json = serde_json::to_string(&task).unwrap();
        let back: Task = serde_json::from_str(&json).unwrap();
        assert_eq!(back.history.as_ref().unwrap().len(), 1);
    }
}
