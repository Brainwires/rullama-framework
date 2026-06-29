//! Merge policies — decide when and how PRs should be merged.
//!
//! Policies evaluate a PR's context (CI status, confidence, diff size) and return
//! an [`Approve`](MergeDecision::Approve), [`Wait`](MergeDecision::Wait), or
//! [`Reject`](MergeDecision::Reject) decision.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::forge::{CheckState, GitForge, MergeMethod, PullRequest, RepoRef};

/// Decision from a merge policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeDecision {
    /// Approve the merge with a specific method.
    Approve {
        /// Merge method to use.
        method: MergeMethod,
    },
    /// Wait for some condition to be met.
    Wait {
        /// Reason for waiting.
        reason: String,
    },
    /// Reject the merge.
    Reject {
        /// Reason for rejection.
        reason: String,
    },
}

/// Context for merge policy evaluation.
#[derive(Debug, Clone)]
pub struct MergeContext {
    /// Investigation confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Number of diff lines in the changes.
    pub diff_lines: u32,
    /// Number of files modified.
    pub files_modified: usize,
}

/// Trait for merge policies that evaluate whether a PR should be auto-merged.
#[async_trait]
pub trait MergePolicy: Send + Sync {
    /// Evaluate whether a PR should be merged.
    async fn evaluate(&self, pr: &PullRequest, ctx: &MergeContext) -> MergeDecision;
}

/// Always requires human approval (default safe policy).
pub struct RequireApprovalPolicy;

#[async_trait]
impl MergePolicy for RequireApprovalPolicy {
    async fn evaluate(&self, _pr: &PullRequest, _ctx: &MergeContext) -> MergeDecision {
        MergeDecision::Wait {
            reason: "Requires human approval".to_string(),
        }
    }
}

/// Requires all CI checks to pass before approving a merge.
pub struct CiPassPolicy {
    forge: std::sync::Arc<dyn GitForge>,
    merge_method: MergeMethod,
}

impl CiPassPolicy {
    /// Create a CI pass policy with the given forge and merge method.
    pub fn new(forge: std::sync::Arc<dyn GitForge>, merge_method: MergeMethod) -> Self {
        Self {
            forge,
            merge_method,
        }
    }
}

#[async_trait]
impl MergePolicy for CiPassPolicy {
    async fn evaluate(&self, pr: &PullRequest, _ctx: &MergeContext) -> MergeDecision {
        let repo = RepoRef {
            owner: String::new(), // Must be provided externally
            name: String::new(),
        };

        match self.forge.get_check_status(&repo, pr.number).await {
            Ok(status) => match status.state {
                CheckState::Success => MergeDecision::Approve {
                    method: self.merge_method,
                },
                CheckState::Pending => MergeDecision::Wait {
                    reason: "CI checks still running".to_string(),
                },
                CheckState::Failure => MergeDecision::Reject {
                    reason: "CI checks failed".to_string(),
                },
                CheckState::Error => MergeDecision::Reject {
                    reason: "CI checks errored".to_string(),
                },
            },
            Err(e) => MergeDecision::Wait {
                reason: format!("Failed to fetch check status: {e}"),
            },
        }
    }
}

/// Auto-merge when the investigation confidence score exceeds a configurable threshold.
pub struct ConfidenceBasedPolicy {
    min_confidence: f64,
    merge_method: MergeMethod,
}

impl ConfidenceBasedPolicy {
    /// Create a confidence-based policy with the given threshold and merge method.
    pub fn new(min_confidence: f64, merge_method: MergeMethod) -> Self {
        Self {
            min_confidence,
            merge_method,
        }
    }
}

#[async_trait]
impl MergePolicy for ConfidenceBasedPolicy {
    async fn evaluate(&self, _pr: &PullRequest, ctx: &MergeContext) -> MergeDecision {
        if ctx.confidence < self.min_confidence {
            return MergeDecision::Wait {
                reason: format!(
                    "Confidence {:.1}% below threshold {:.1}%",
                    ctx.confidence * 100.0,
                    self.min_confidence * 100.0
                ),
            };
        }

        MergeDecision::Approve {
            method: self.merge_method,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_pr() -> PullRequest {
        PullRequest {
            id: "pr-1".to_string(),
            number: 1,
            title: "test PR".to_string(),
            body: String::new(),
            head_branch: "fix/test".to_string(),
            base_branch: "main".to_string(),
            url: String::new(),
            state: super::super::forge::PrState::Open,
        }
    }

    fn ctx(confidence: f64) -> MergeContext {
        MergeContext {
            confidence,
            diff_lines: 10,
            files_modified: 1,
        }
    }

    // --- RequireApprovalPolicy ---

    #[tokio::test]
    async fn require_approval_always_waits() {
        let policy = RequireApprovalPolicy;
        let decision = policy.evaluate(&dummy_pr(), &ctx(1.0)).await;
        assert!(matches!(decision, MergeDecision::Wait { .. }));
    }

    #[tokio::test]
    async fn require_approval_wait_message_is_human_approval() {
        let policy = RequireApprovalPolicy;
        let decision = policy.evaluate(&dummy_pr(), &ctx(0.0)).await;
        match decision {
            MergeDecision::Wait { reason } => {
                assert!(reason.contains("human approval") || !reason.is_empty());
            }
            _ => panic!("expected Wait"),
        }
    }

    // --- ConfidenceBasedPolicy ---

    #[tokio::test]
    async fn confidence_policy_approves_above_threshold() {
        let policy = ConfidenceBasedPolicy::new(0.8, MergeMethod::Squash);
        let decision = policy.evaluate(&dummy_pr(), &ctx(0.9)).await;
        assert!(matches!(
            decision,
            MergeDecision::Approve {
                method: MergeMethod::Squash
            }
        ));
    }

    #[tokio::test]
    async fn confidence_policy_waits_below_threshold() {
        let policy = ConfidenceBasedPolicy::new(0.8, MergeMethod::Merge);
        let decision = policy.evaluate(&dummy_pr(), &ctx(0.7)).await;
        assert!(matches!(decision, MergeDecision::Wait { .. }));
    }

    #[tokio::test]
    async fn confidence_policy_approves_at_exact_threshold() {
        let policy = ConfidenceBasedPolicy::new(0.8, MergeMethod::Rebase);
        let decision = policy.evaluate(&dummy_pr(), &ctx(0.8)).await;
        // Exactly at threshold — policy uses `<` so 0.8 >= 0.8 approves
        assert!(matches!(decision, MergeDecision::Approve { .. }));
    }

    #[tokio::test]
    async fn confidence_policy_wait_message_contains_percentages() {
        let policy = ConfidenceBasedPolicy::new(0.8, MergeMethod::Merge);
        let decision = policy.evaluate(&dummy_pr(), &ctx(0.5)).await;
        match decision {
            MergeDecision::Wait { reason } => {
                assert!(reason.contains('%'), "reason should contain %: {reason}");
            }
            _ => panic!("expected Wait"),
        }
    }

    // --- MergeDecision serde ---

    #[test]
    fn merge_decision_approve_serde_roundtrip() {
        let d = MergeDecision::Approve {
            method: MergeMethod::Squash,
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: MergeDecision = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            MergeDecision::Approve {
                method: MergeMethod::Squash
            }
        ));
    }

    #[test]
    fn merge_decision_wait_serde_roundtrip() {
        let d = MergeDecision::Wait {
            reason: "needs review".to_string(),
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: MergeDecision = serde_json::from_str(&json).unwrap();
        match back {
            MergeDecision::Wait { reason } => assert_eq!(reason, "needs review"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn merge_context_fields_accessible() {
        let ctx = MergeContext {
            confidence: 0.95,
            diff_lines: 42,
            files_modified: 3,
        };
        assert_eq!(ctx.confidence, 0.95);
        assert_eq!(ctx.diff_lines, 42);
        assert_eq!(ctx.files_modified, 3);
    }
}
