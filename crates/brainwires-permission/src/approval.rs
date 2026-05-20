//! Runtime approval request/response types
//!
//! Provides types for interactive approval workflows where tool execution
//! requires user consent before proceeding. Used by the ToolExecutor, MCP
//! server handler, and CLI approval systems.

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

/// A request for user approval before executing a tool.
pub struct ApprovalRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// Name of the tool requesting approval.
    pub tool_name: String,
    /// The action being performed.
    pub action: ApprovalAction,
    /// Additional details about the action.
    pub details: ApprovalDetails,
    /// Channel to send the response back.
    pub response_tx: oneshot::Sender<ApprovalResponse>,
}

/// The type of action being performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalAction {
    /// Writing to a file (create or overwrite).
    WriteFile {
        /// File path being written.
        path: String,
    },
    /// Editing a file (find/replace).
    EditFile {
        /// File path being edited.
        path: String,
    },
    /// Deleting a file or directory.
    DeleteFile {
        /// File path being deleted.
        path: String,
    },
    /// Creating a directory.
    CreateDirectory {
        /// Directory path being created.
        path: String,
    },
    /// Executing a shell command.
    ExecuteCommand {
        /// The command to execute.
        command: String,
    },
    /// Git operations that modify state.
    GitModify {
        /// The git operation name.
        operation: String,
    },
    /// Network access (non-read operations).
    NetworkAccess {
        /// The target domain.
        domain: String,
    },
    /// Other actions requiring approval.
    Other {
        /// Description of the action.
        description: String,
    },
}

impl ApprovalAction {
    /// Get a human-readable description of the action.
    pub fn description(&self) -> String {
        match self {
            ApprovalAction::WriteFile { path } => format!("Write file: {}", path),
            ApprovalAction::EditFile { path } => format!("Edit file: {}", path),
            ApprovalAction::DeleteFile { path } => format!("Delete: {}", path),
            ApprovalAction::CreateDirectory { path } => format!("Create directory: {}", path),
            ApprovalAction::ExecuteCommand { command } => {
                let truncated = if command.len() > 50 {
                    format!("{}...", &command[..50])
                } else {
                    command.clone()
                };
                format!("Execute: {}", truncated)
            }
            ApprovalAction::GitModify { operation } => format!("Git: {}", operation),
            ApprovalAction::NetworkAccess { domain } => format!("Network: {}", domain),
            ApprovalAction::Other { description } => description.clone(),
        }
    }

    /// Get the category name for display.
    pub fn category(&self) -> &'static str {
        match self {
            ApprovalAction::WriteFile { .. } => "File Write",
            ApprovalAction::EditFile { .. } => "File Edit",
            ApprovalAction::DeleteFile { .. } => "Delete",
            ApprovalAction::CreateDirectory { .. } => "Create Directory",
            ApprovalAction::ExecuteCommand { .. } => "Shell Command",
            ApprovalAction::GitModify { .. } => "Git Operation",
            ApprovalAction::NetworkAccess { .. } => "Network Access",
            ApprovalAction::Other { .. } => "Other",
        }
    }

    /// Get the severity level for the action.
    pub fn severity(&self) -> ApprovalSeverity {
        match self {
            ApprovalAction::DeleteFile { .. } => ApprovalSeverity::High,
            ApprovalAction::ExecuteCommand { .. } => ApprovalSeverity::High,
            ApprovalAction::GitModify { .. } => ApprovalSeverity::Medium,
            ApprovalAction::WriteFile { .. } => ApprovalSeverity::Medium,
            ApprovalAction::EditFile { .. } => ApprovalSeverity::Medium,
            ApprovalAction::CreateDirectory { .. } => ApprovalSeverity::Low,
            ApprovalAction::NetworkAccess { .. } => ApprovalSeverity::Low,
            ApprovalAction::Other { .. } => ApprovalSeverity::Medium,
        }
    }
}

/// Severity level for approval actions (affects UI presentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalSeverity {
    /// Low risk - informational.
    Low,
    /// Medium risk - caution.
    Medium,
    /// High risk - dangerous.
    High,
}

/// Additional details about an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDetails {
    /// Description of the tool.
    pub tool_description: String,
    /// The parameters being passed to the tool.
    pub parameters: serde_json::Value,
}

/// User's response to an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalResponse {
    /// Approve this single request.
    Approve,
    /// Deny this single request.
    Deny,
    /// Approve and remember for this tool for the session.
    ApproveForSession,
    /// Deny and remember for this tool for the session.
    DenyForSession,
}

impl ApprovalResponse {
    /// Check if this is an approval (yes or always).
    pub fn is_approved(&self) -> bool {
        matches!(
            self,
            ApprovalResponse::Approve | ApprovalResponse::ApproveForSession
        )
    }

    /// Check if this should be remembered for the session.
    pub fn is_session_persistent(&self) -> bool {
        matches!(
            self,
            ApprovalResponse::ApproveForSession | ApprovalResponse::DenyForSession
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approval_action_description() {
        let action = ApprovalAction::WriteFile {
            path: "/tmp/test.txt".to_string(),
        };
        assert!(action.description().contains("/tmp/test.txt"));
        assert_eq!(action.category(), "File Write");
    }

    #[test]
    fn test_approval_response_is_approved() {
        assert!(ApprovalResponse::Approve.is_approved());
        assert!(ApprovalResponse::ApproveForSession.is_approved());
        assert!(!ApprovalResponse::Deny.is_approved());
        assert!(!ApprovalResponse::DenyForSession.is_approved());
    }

    #[test]
    fn test_approval_response_is_session_persistent() {
        assert!(!ApprovalResponse::Approve.is_session_persistent());
        assert!(ApprovalResponse::ApproveForSession.is_session_persistent());
        assert!(!ApprovalResponse::Deny.is_session_persistent());
        assert!(ApprovalResponse::DenyForSession.is_session_persistent());
    }

    #[test]
    fn test_command_truncation() {
        let long_command = "a".repeat(100);
        let action = ApprovalAction::ExecuteCommand {
            command: long_command,
        };
        let desc = action.description();
        assert!(desc.len() < 70);
        assert!(desc.ends_with("..."));
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(
            ApprovalAction::DeleteFile { path: "x".into() }.severity(),
            ApprovalSeverity::High
        );
        assert_eq!(
            ApprovalAction::WriteFile { path: "x".into() }.severity(),
            ApprovalSeverity::Medium
        );
        assert_eq!(
            ApprovalAction::CreateDirectory { path: "x".into() }.severity(),
            ApprovalSeverity::Low
        );
    }
}
