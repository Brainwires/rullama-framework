//! Streaming event types for A2A.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::task::{Task, TaskStatus};
use crate::types::{Artifact, Message};

/// Event notifying a change in task status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskStatusUpdateEvent {
    /// Task identifier.
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// Context identifier.
    #[serde(rename = "contextId")]
    pub context_id: String,
    /// New task status.
    pub status: TaskStatus,
    /// Trace ID for cross-system correlation. Matches the `trace_id` stamped by
    /// the originating `TaskAgent` and carried in `AuditEvent.metadata["trace_id"]`.
    #[serde(rename = "traceId", skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<Uuid>,
    /// Monotonically increasing sequence number within the trace.
    /// Allows receivers to detect and reorder out-of-order events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Event notifying an artifact update.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskArtifactUpdateEvent {
    /// Task identifier.
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// Context identifier.
    #[serde(rename = "contextId")]
    pub context_id: String,
    /// The artifact.
    pub artifact: Artifact,
    /// Index of this artifact within the task's artifact list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// If true, append to previously sent artifact with same ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub append: Option<bool>,
    /// If true, this is the final chunk.
    #[serde(rename = "lastChunk", skip_serializing_if = "Option::is_none")]
    pub last_chunk: Option<bool>,
    /// Trace ID for cross-system correlation.
    #[serde(rename = "traceId", skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<Uuid>,
    /// Monotonically increasing sequence number within the trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Wrapper-based stream response (exactly one field should be set).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamResponse {
    /// Full task snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
    /// Agent message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    /// Task status change.
    #[serde(skip_serializing_if = "Option::is_none", rename = "statusUpdate")]
    pub status_update: Option<TaskStatusUpdateEvent>,
    /// Artifact update.
    #[serde(skip_serializing_if = "Option::is_none", rename = "artifactUpdate")]
    pub artifact_update: Option<TaskArtifactUpdateEvent>,
}

/// Response for `message/send` — wrapper-based (exactly one field should be set).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SendMessageResponse {
    /// A task was created or updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
    /// A direct message response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
}
