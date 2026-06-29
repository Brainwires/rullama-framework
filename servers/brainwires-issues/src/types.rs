//! Core data types for the issue tracking system.

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── IssueStatus ──────────────────────────────────────────────────────────

/// Workflow status of an issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum IssueStatus {
    /// Not yet scheduled.
    #[default]
    Backlog,
    /// Scheduled, not started.
    Todo,
    /// Actively being worked on.
    InProgress,
    /// In code/design review.
    InReview,
    /// Completed successfully.
    Done,
    /// Cancelled (won't fix).
    Cancelled,
}

impl IssueStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            IssueStatus::Backlog => "backlog",
            IssueStatus::Todo => "todo",
            IssueStatus::InProgress => "in_progress",
            IssueStatus::InReview => "in_review",
            IssueStatus::Done => "done",
            IssueStatus::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "backlog" => IssueStatus::Backlog,
            "todo" => IssueStatus::Todo,
            "in_progress" => IssueStatus::InProgress,
            "in_review" => IssueStatus::InReview,
            "done" => IssueStatus::Done,
            "cancelled" => IssueStatus::Cancelled,
            _ => IssueStatus::Backlog,
        }
    }

    /// Returns true if this status represents a closed issue.
    pub fn is_closed(&self) -> bool {
        matches!(self, IssueStatus::Done | IssueStatus::Cancelled)
    }
}

// ── IssuePriority ────────────────────────────────────────────────────────

/// Priority level of an issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum IssuePriority {
    /// No priority assigned.
    #[default]
    NoPriority,
    /// Low priority.
    Low,
    /// Medium priority.
    Medium,
    /// High priority.
    High,
    /// Urgent — needs immediate attention.
    Urgent,
}

impl IssuePriority {
    pub fn as_str(&self) -> &'static str {
        match self {
            IssuePriority::NoPriority => "no_priority",
            IssuePriority::Low => "low",
            IssuePriority::Medium => "medium",
            IssuePriority::High => "high",
            IssuePriority::Urgent => "urgent",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "no_priority" => IssuePriority::NoPriority,
            "low" => IssuePriority::Low,
            "medium" => IssuePriority::Medium,
            "high" => IssuePriority::High,
            "urgent" => IssuePriority::Urgent,
            _ => IssuePriority::NoPriority,
        }
    }
}

// ── Issue ────────────────────────────────────────────────────────────────

/// A tracked issue or bug report.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Issue {
    /// Unique identifier (UUID v4).
    pub id: String,
    /// Auto-incrementing display number (e.g. #42).
    pub number: u64,
    /// Short title of the issue.
    pub title: String,
    /// Full description in Markdown.
    pub description: String,
    /// Current workflow status.
    pub status: IssueStatus,
    /// Priority level.
    pub priority: IssuePriority,
    /// Comma-separated label tags stored as a JSON array string internally.
    pub labels: Vec<String>,
    /// Person or agent assigned to this issue.
    pub assignee: Option<String>,
    /// Project or milestone this issue belongs to.
    pub project: Option<String>,
    /// Parent issue ID for sub-issues.
    pub parent_id: Option<String>,
    /// Creation time (Unix seconds).
    pub created_at: i64,
    /// Last update time (Unix seconds).
    pub updated_at: i64,
    /// Time when the issue was closed (Unix seconds).
    pub closed_at: Option<i64>,
}

impl Issue {
    /// Create a new issue with defaults.
    pub fn new(number: u64, title: impl Into<String>) -> Self {
        let now = Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            number,
            title: title.into(),
            description: String::new(),
            status: IssueStatus::Backlog,
            priority: IssuePriority::NoPriority,
            labels: Vec::new(),
            assignee: None,
            project: None,
            parent_id: None,
            created_at: now,
            updated_at: now,
            closed_at: None,
        }
    }
}

// ── IssuePatch ───────────────────────────────────────────────────────────

/// Partial update for an issue — all fields are optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct IssuePatch {
    pub title: Option<String>,
    pub description: Option<String>,
    pub status: Option<IssueStatus>,
    pub priority: Option<IssuePriority>,
    pub labels: Option<Vec<String>>,
    pub assignee: Option<String>,
    /// Pass `""` to clear the assignee.
    pub clear_assignee: Option<bool>,
    pub project: Option<String>,
    /// Pass `""` to clear the project.
    pub clear_project: Option<bool>,
    pub parent_id: Option<String>,
    /// Pass `true` to clear the parent.
    pub clear_parent: Option<bool>,
}

// ── Comment ──────────────────────────────────────────────────────────────

/// A comment on an issue.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Comment {
    /// Unique identifier (UUID v4).
    pub id: String,
    /// The issue this comment belongs to.
    pub issue_id: String,
    /// Author name or identifier.
    pub author: Option<String>,
    /// Comment body in Markdown.
    pub body: String,
    /// Creation time (Unix seconds).
    pub created_at: i64,
    /// Last update time (Unix seconds).
    pub updated_at: i64,
}

impl Comment {
    pub fn new(issue_id: impl Into<String>, body: impl Into<String>) -> Self {
        let now = Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            issue_id: issue_id.into(),
            author: None,
            body: body.into(),
            created_at: now,
            updated_at: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- IssueStatus ---

    #[test]
    fn issue_status_as_str_roundtrip() {
        let cases = [
            (IssueStatus::Backlog, "backlog"),
            (IssueStatus::Todo, "todo"),
            (IssueStatus::InProgress, "in_progress"),
            (IssueStatus::InReview, "in_review"),
            (IssueStatus::Done, "done"),
            (IssueStatus::Cancelled, "cancelled"),
        ];
        for (status, expected) in &cases {
            assert_eq!(status.as_str(), *expected);
            assert_eq!(IssueStatus::parse(expected), *status);
        }
    }

    #[test]
    fn issue_status_unknown_falls_back_to_backlog() {
        assert_eq!(IssueStatus::parse("unknown_status"), IssueStatus::Backlog);
    }

    #[test]
    fn issue_status_is_closed() {
        assert!(IssueStatus::Done.is_closed());
        assert!(IssueStatus::Cancelled.is_closed());
        assert!(!IssueStatus::Backlog.is_closed());
        assert!(!IssueStatus::Todo.is_closed());
        assert!(!IssueStatus::InProgress.is_closed());
        assert!(!IssueStatus::InReview.is_closed());
    }

    #[test]
    fn issue_status_serde_roundtrip() {
        let statuses = [
            IssueStatus::Backlog,
            IssueStatus::Todo,
            IssueStatus::InProgress,
            IssueStatus::Done,
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let back: IssueStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, s);
        }
    }

    #[test]
    fn issue_status_default_is_backlog() {
        assert_eq!(IssueStatus::default(), IssueStatus::Backlog);
    }

    // --- IssuePriority ---

    #[test]
    fn issue_priority_as_str_roundtrip() {
        let cases = [
            (IssuePriority::NoPriority, "no_priority"),
            (IssuePriority::Low, "low"),
            (IssuePriority::Medium, "medium"),
            (IssuePriority::High, "high"),
            (IssuePriority::Urgent, "urgent"),
        ];
        for (priority, expected) in &cases {
            assert_eq!(priority.as_str(), *expected);
            assert_eq!(IssuePriority::parse(expected), *priority);
        }
    }

    #[test]
    fn issue_priority_unknown_falls_back_to_no_priority() {
        assert_eq!(IssuePriority::parse("unknown"), IssuePriority::NoPriority);
    }

    #[test]
    fn issue_priority_default_is_no_priority() {
        assert_eq!(IssuePriority::default(), IssuePriority::NoPriority);
    }

    // --- Issue::new ---

    #[test]
    fn issue_new_sets_defaults() {
        let issue = Issue::new(1, "My Issue");
        assert_eq!(issue.number, 1);
        assert_eq!(issue.title, "My Issue");
        assert_eq!(issue.description, "");
        assert_eq!(issue.status, IssueStatus::Backlog);
        assert_eq!(issue.priority, IssuePriority::NoPriority);
        assert!(issue.labels.is_empty());
        assert!(issue.assignee.is_none());
        assert!(issue.project.is_none());
        assert!(issue.parent_id.is_none());
        assert!(issue.closed_at.is_none());
        assert!(!issue.id.is_empty());
        assert!(issue.created_at > 0);
        assert_eq!(issue.created_at, issue.updated_at);
    }

    #[test]
    fn issue_ids_are_unique() {
        let a = Issue::new(1, "A");
        let b = Issue::new(2, "B");
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn issue_serde_roundtrip() {
        let issue = Issue::new(42, "Test Issue");
        let json = serde_json::to_string(&issue).unwrap();
        let back: Issue = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, issue.id);
        assert_eq!(back.number, 42);
        assert_eq!(back.title, "Test Issue");
    }

    // --- IssuePatch ---

    #[test]
    fn issue_patch_default_all_none() {
        let patch = IssuePatch::default();
        assert!(patch.title.is_none());
        assert!(patch.status.is_none());
        assert!(patch.assignee.is_none());
        assert!(patch.clear_assignee.is_none());
    }

    // --- Comment::new ---

    #[test]
    fn comment_new_sets_fields() {
        let c = Comment::new("issue-1", "This is a comment");
        assert_eq!(c.issue_id, "issue-1");
        assert_eq!(c.body, "This is a comment");
        assert!(c.author.is_none());
        assert!(!c.id.is_empty());
        assert!(c.created_at > 0);
    }

    #[test]
    fn comment_ids_are_unique() {
        let a = Comment::new("i1", "a");
        let b = Comment::new("i1", "b");
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn comment_serde_roundtrip() {
        let c = Comment::new("issue-42", "body text");
        let json = serde_json::to_string(&c).unwrap();
        let back: Comment = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, c.id);
        assert_eq!(back.body, "body text");
    }
}
