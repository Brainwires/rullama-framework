//! Workflow trigger sources — webhook events, programmatic triggers, etc.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::forge::{Comment, CommitRef, Issue, PullRequest, RepoRef};

/// Events that can trigger a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowEvent {
    /// A new issue was opened.
    IssueOpened {
        /// The opened issue.
        issue: Issue,
        /// Repository where the issue was opened.
        repo: RepoRef,
    },
    /// A comment was added to an issue.
    IssueCommented {
        /// The issue that was commented on.
        issue: Issue,
        /// The new comment.
        comment: Comment,
        /// Repository of the issue.
        repo: RepoRef,
    },
    /// Commits were pushed to a branch.
    PushReceived {
        /// Branch that received the push.
        branch: String,
        /// Commits that were pushed.
        commits: Vec<CommitRef>,
        /// Repository of the push.
        repo: RepoRef,
    },
    /// A PR review was approved.
    PrReviewApproved {
        /// The approved pull request.
        pr: PullRequest,
        /// Repository of the PR.
        repo: RepoRef,
    },
    /// A manually triggered event.
    Manual {
        /// Description of the manual trigger.
        description: String,
        /// Target repository.
        repo: RepoRef,
    },
}

/// Trait for event sources that emit workflow events.
#[async_trait]
pub trait WorkflowTrigger: Send + Sync {
    /// Start listening for events and send them to the given channel.
    async fn start(&self, tx: mpsc::Sender<WorkflowEvent>) -> anyhow::Result<()>;
}

/// Programmatic trigger — allows sending events directly from code.
pub struct ProgrammaticTrigger {
    tx: Option<mpsc::Sender<WorkflowEvent>>,
}

impl ProgrammaticTrigger {
    /// Create a new programmatic trigger (not yet connected to a channel).
    pub fn new() -> Self {
        Self { tx: None }
    }

    /// Send an event manually.
    pub async fn emit(&self, event: WorkflowEvent) -> anyhow::Result<()> {
        if let Some(tx) = &self.tx {
            tx.send(event)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send event: {e}"))?;
        }
        Ok(())
    }
}

impl Default for ProgrammaticTrigger {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorkflowTrigger for ProgrammaticTrigger {
    async fn start(&self, _tx: mpsc::Sender<WorkflowEvent>) -> anyhow::Result<()> {
        // Programmatic trigger doesn't listen — events are emitted via emit()
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::forge::{CommitRef, Issue, PrState, PullRequest, RepoRef};
    use super::*;

    fn sample_repo() -> RepoRef {
        RepoRef {
            owner: "org".to_string(),
            name: "repo".to_string(),
        }
    }

    fn sample_issue() -> Issue {
        Issue {
            id: "1".to_string(),
            number: 42,
            title: "Fix login bug".to_string(),
            body: "Users can't log in".to_string(),
            labels: vec!["bug".to_string()],
            author: "alice".to_string(),
            url: "https://github.com/org/repo/issues/42".to_string(),
        }
    }

    // --- WorkflowEvent serde roundtrips ---

    #[test]
    fn issue_opened_serde_roundtrip() {
        let event = WorkflowEvent::IssueOpened {
            issue: sample_issue(),
            repo: sample_repo(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: WorkflowEvent = serde_json::from_str(&json).unwrap();
        match back {
            WorkflowEvent::IssueOpened { issue, repo } => {
                assert_eq!(issue.number, 42);
                assert_eq!(repo.name, "repo");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn push_received_serde_roundtrip() {
        let event = WorkflowEvent::PushReceived {
            branch: "main".to_string(),
            commits: vec![CommitRef {
                sha: "abc123".to_string(),
                message: "fix: login".to_string(),
            }],
            repo: sample_repo(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: WorkflowEvent = serde_json::from_str(&json).unwrap();
        match back {
            WorkflowEvent::PushReceived {
                branch, commits, ..
            } => {
                assert_eq!(branch, "main");
                assert_eq!(commits.len(), 1);
                assert_eq!(commits[0].sha, "abc123");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn manual_event_serde_roundtrip() {
        let event = WorkflowEvent::Manual {
            description: "Run analysis".to_string(),
            repo: sample_repo(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: WorkflowEvent = serde_json::from_str(&json).unwrap();
        match back {
            WorkflowEvent::Manual { description, .. } => {
                assert_eq!(description, "Run analysis");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn pr_review_approved_serde_roundtrip() {
        let event = WorkflowEvent::PrReviewApproved {
            pr: PullRequest {
                id: "pr-1".to_string(),
                number: 5,
                title: "Fix bug".to_string(),
                body: "body".to_string(),
                head_branch: "fix/bug".to_string(),
                base_branch: "main".to_string(),
                url: "https://github.com/org/repo/pull/5".to_string(),
                state: PrState::Open,
            },
            repo: sample_repo(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: WorkflowEvent = serde_json::from_str(&json).unwrap();
        match back {
            WorkflowEvent::PrReviewApproved { pr, .. } => {
                assert_eq!(pr.number, 5);
                assert_eq!(pr.state, PrState::Open);
            }
            _ => panic!("wrong variant"),
        }
    }

    // --- ProgrammaticTrigger ---

    #[tokio::test]
    async fn programmatic_trigger_start_is_noop() {
        let trigger = ProgrammaticTrigger::new();
        let (tx, _rx) = mpsc::channel(1);
        trigger.start(tx).await.unwrap();
    }

    #[tokio::test]
    async fn programmatic_trigger_emit_without_channel_succeeds() {
        let trigger = ProgrammaticTrigger::new();
        let event = WorkflowEvent::Manual {
            description: "test".to_string(),
            repo: sample_repo(),
        };
        // Without a connected channel, emit is a no-op
        trigger.emit(event).await.unwrap();
    }

    #[test]
    fn programmatic_trigger_default_is_new() {
        let trigger = ProgrammaticTrigger::default();
        assert!(trigger.tx.is_none());
    }
}
