//! End-to-end Git workflow pipeline orchestrator.
//!
//! Coordinates the full autonomous issue-to-merge lifecycle: receiving workflow
//! events, investigating issues with AI, creating fix branches, applying changes,
//! opening PRs, and evaluating merge policies.

use std::sync::Arc;

use brainwires_core::Provider;
use tokio::sync::mpsc;

use super::branch_manager::BranchManager;
use super::change_maker::ChangeMaker;
use super::forge::{GitForge, Issue, RepoRef};
use super::investigator::IssueInvestigator;
use super::merge_policy::{MergeContext, MergeDecision, MergePolicy};
use super::pr_manager::PullRequestManager;
use super::trigger::WorkflowEvent;
use crate::config::GitWorkflowConfig;
use crate::safety::{ApprovalPolicy, AutonomousOperation};

/// The end-to-end pipeline: trigger -> investigate -> branch -> fix -> PR -> merge.
///
/// Combines an AI provider, Git forge, merge policy, and approval policy to
/// autonomously process issues and produce pull requests.
pub struct GitWorkflowPipeline {
    config: GitWorkflowConfig,
    forge: Arc<dyn GitForge>,
    provider: Arc<dyn Provider>,
    merge_policy: Arc<dyn MergePolicy>,
    approval: Arc<dyn ApprovalPolicy>,
    repo: RepoRef,
    base_branch: String,
}

impl GitWorkflowPipeline {
    /// Create a new Git workflow pipeline.
    pub fn new(
        config: GitWorkflowConfig,
        forge: Arc<dyn GitForge>,
        provider: Arc<dyn Provider>,
        merge_policy: Arc<dyn MergePolicy>,
        approval: Arc<dyn ApprovalPolicy>,
        repo: RepoRef,
        base_branch: String,
    ) -> Self {
        Self {
            config,
            forge,
            provider,
            merge_policy,
            approval,
            repo,
            base_branch,
        }
    }

    /// Run the pipeline, listening for events from the given channel.
    pub async fn run(&self, mut rx: mpsc::Receiver<WorkflowEvent>) {
        while let Some(event) = rx.recv().await {
            match event {
                WorkflowEvent::IssueOpened { issue, repo } => {
                    if let Err(e) = self.handle_issue(&issue, &repo).await {
                        tracing::error!("Pipeline failed for issue #{}: {e}", issue.number);
                    }
                }
                WorkflowEvent::Manual { description, repo } => {
                    tracing::info!("Manual trigger: {description} for {}", repo.full_name());
                }
                _ => {
                    tracing::debug!("Ignoring event type");
                }
            }
        }
    }

    /// Handle a single issue through the full pipeline.
    pub async fn handle_issue(&self, issue: &Issue, _repo: &RepoRef) -> anyhow::Result<()> {
        tracing::info!(
            "Pipeline: processing issue #{} - {}",
            issue.number,
            issue.title
        );

        // Stage 1: Investigate
        let investigator = IssueInvestigator::new(self.provider.clone());
        let investigation = investigator.investigate(issue, &self.repo).await?;

        if investigation.confidence < self.config.min_confidence {
            tracing::warn!(
                "Investigation confidence {:.1}% below threshold {:.1}%, skipping",
                investigation.confidence * 100.0,
                self.config.min_confidence * 100.0
            );
            return Ok(());
        }

        // Stage 2: Create branch
        let branch_mgr = BranchManager::new(self.config.branch_prefix.clone());
        let branch_name = branch_mgr.branch_name(issue.number, &issue.title);

        let repo_path = std::env::current_dir()?.to_string_lossy().to_string();
        let _branch_info = branch_mgr
            .create_branch(&repo_path, &branch_name, &self.base_branch)
            .await?;

        // Stage 3: Make changes
        let change_maker = ChangeMaker::new(self.provider.clone(), 20);
        let changes = change_maker
            .make_changes(&investigation, &repo_path)
            .await?;

        if !changes.success {
            tracing::warn!("Change maker failed for issue #{}", issue.number);
            return Ok(());
        }

        // Commit changes
        change_maker
            .commit_changes(&repo_path, issue.number, &changes.summary)
            .await?;

        // Stage 4: Push and create PR
        branch_mgr.push_branch(&repo_path, &branch_name).await?;

        let pr_mgr = PullRequestManager::new(self.forge.clone());

        // Check approval for PR creation
        let pr_op = AutonomousOperation::CreatePullRequest {
            branch: branch_name.clone(),
            title: issue.title.clone(),
        };
        self.approval
            .check(&pr_op)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let pr = pr_mgr
            .create_pr(&self.repo, issue, &branch_name, &self.base_branch, &changes)
            .await?;

        tracing::info!("Created PR #{} for issue #{}", pr.number, issue.number);

        // Stage 5: Evaluate merge policy
        let merge_ctx = MergeContext {
            confidence: investigation.confidence,
            diff_lines: changes.diff_lines,
            files_modified: changes.files_modified.len(),
        };

        match self.merge_policy.evaluate(&pr, &merge_ctx).await {
            MergeDecision::Approve { method } => {
                let merge_op = AutonomousOperation::MergePullRequest {
                    pr_id: pr.number.to_string(),
                    confidence: investigation.confidence,
                };

                if self.approval.check(&merge_op).await.is_ok() {
                    self.forge
                        .merge_pull_request(&self.repo, pr.number, method)
                        .await?;
                    tracing::info!("Auto-merged PR #{}", pr.number);
                } else {
                    tracing::info!("Merge of PR #{} requires manual approval", pr.number);
                }
            }
            MergeDecision::Wait { reason } => {
                tracing::info!("PR #{} waiting: {reason}", pr.number);
            }
            MergeDecision::Reject { reason } => {
                tracing::warn!("PR #{} merge rejected: {reason}", pr.number);
            }
        }

        Ok(())
    }
}
