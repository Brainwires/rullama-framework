//! Core A2A message types: Message, Part, Artifact, Role.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sender role in A2A communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Client-to-server message.
    #[serde(rename = "ROLE_USER")]
    User,
    /// Server-to-client message.
    #[serde(rename = "ROLE_AGENT")]
    Agent,
    /// Unspecified role.
    #[serde(rename = "ROLE_UNSPECIFIED")]
    Unspecified,
}

/// A single unit of communication content.
///
/// Exactly one of `text`, `raw`, `url`, or `data` must be set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Part {
    /// Text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded raw bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// URL reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Structured JSON data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    /// MIME type of the content.
    #[serde(skip_serializing_if = "Option::is_none", rename = "mediaType")]
    pub media_type: Option<String>,
    /// File name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// A single communication message between client and server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// Unique message identifier.
    #[serde(rename = "messageId")]
    pub message_id: String,
    /// Sender role.
    pub role: Role,
    /// Content parts.
    pub parts: Vec<Part>,
    /// Context identifier (conversation/session).
    #[serde(rename = "contextId", skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// Associated task identifier.
    #[serde(rename = "taskId", skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Referenced task identifiers for additional context.
    #[serde(rename = "referenceTaskIds", skip_serializing_if = "Option::is_none")]
    pub reference_task_ids: Option<Vec<String>>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Extension URIs present in this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
}

impl Message {
    /// Create a new user message with text content.
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            message_id: uuid::Uuid::new_v4().to_string(),
            role: Role::User,
            parts: vec![Part {
                text: Some(text.into()),
                raw: None,
                url: None,
                data: None,
                media_type: None,
                filename: None,
                metadata: None,
            }],
            context_id: None,
            task_id: None,
            reference_task_ids: None,
            metadata: None,
            extensions: None,
        }
    }

    /// Create a new agent message with text content.
    pub fn agent_text(text: impl Into<String>) -> Self {
        let mut msg = Self::user_text(text);
        msg.role = Role::Agent;
        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Role ---

    #[test]
    fn role_serializes_to_expected_strings() {
        assert_eq!(
            serde_json::to_string(&Role::User).unwrap(),
            r#""ROLE_USER""#
        );
        assert_eq!(
            serde_json::to_string(&Role::Agent).unwrap(),
            r#""ROLE_AGENT""#
        );
        assert_eq!(
            serde_json::to_string(&Role::Unspecified).unwrap(),
            r#""ROLE_UNSPECIFIED""#
        );
    }

    #[test]
    fn role_roundtrip() {
        for role in [Role::User, Role::Agent, Role::Unspecified] {
            let json = serde_json::to_string(&role).unwrap();
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(back, role);
        }
    }

    // --- Part ---

    #[test]
    fn part_text_only_omits_other_fields() {
        let p = Part {
            text: Some("hello".to_string()),
            raw: None,
            url: None,
            data: None,
            media_type: None,
            filename: None,
            metadata: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("hello"));
        assert!(!json.contains("raw"));
        assert!(!json.contains("url"));
    }

    #[test]
    fn part_roundtrip() {
        let p = Part {
            text: Some("content".to_string()),
            raw: None,
            url: Some("https://example.com".to_string()),
            data: Some(serde_json::json!({"key": 1})),
            media_type: Some("text/plain".to_string()),
            filename: Some("file.txt".to_string()),
            metadata: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: Part = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    // --- Message ---

    #[test]
    fn user_text_creates_user_role_message() {
        let msg = Message::user_text("hi");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].text.as_deref(), Some("hi"));
        assert!(!msg.message_id.is_empty());
    }

    #[test]
    fn agent_text_creates_agent_role_message() {
        let msg = Message::agent_text("response");
        assert_eq!(msg.role, Role::Agent);
        assert_eq!(msg.parts[0].text.as_deref(), Some("response"));
    }

    #[test]
    fn message_ids_are_unique() {
        let a = Message::user_text("a");
        let b = Message::user_text("b");
        assert_ne!(a.message_id, b.message_id);
    }

    #[test]
    fn message_roundtrip() {
        let msg = Message::user_text("roundtrip test");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn message_json_uses_camel_case_message_id() {
        let msg = Message::user_text("x");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("messageId"));
        assert!(!json.contains("message_id"));
    }

    #[test]
    fn message_optional_fields_omitted_when_none() {
        let msg = Message::user_text("x");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("contextId"));
        assert!(!json.contains("taskId"));
        assert!(!json.contains("extensions"));
    }
}

/// Task output artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    /// Unique artifact identifier (unique within a task).
    #[serde(rename = "artifactId")]
    pub artifact_id: String,
    /// Human-readable name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Artifact content parts.
    pub parts: Vec<Part>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Extension URIs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
}
