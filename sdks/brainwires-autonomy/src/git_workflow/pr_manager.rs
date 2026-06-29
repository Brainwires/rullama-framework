//! Pull request creation and management.

use std::sync::Arc;

use super::change_maker::ChangeResult;
use super::forge::{CreatePrParams, GitForge, Issue, PullRequest, RepoRef};

/// Manages PR creation and linking to issues.
pub struct PullRequestManager {
    forge: Arc<dyn GitForge>,
}

impl PullRequestManager {
    /// Create a new pull request manager using the given forge.
    pub fn new(forge: Arc<dyn GitForge>) -> Self {
        Self { forge }
    }

    /// Create a PR for the changes made to fix an issue.
    pub async fn create_pr(
        &self,
        repo: &RepoRef,
        issue: &Issue,
        branch_name: &str,
        base_branch: &str,
        changes: &ChangeResult,
    ) -> anyhow::Result<PullRequest> {
        let title = format!("fix: {} (#{}) ", issue.title, issue.number);

        let body = format!(
            "## Summary\n\n\
             Automated fix for #{issue_number}.\n\n\
             {summary}\n\n\
             ## Changes\n\n\
             {files}\n\n\
             ## Issue\n\n\
             Closes #{issue_number}\n\n\
             ---\n\
             *This PR was created automatically by brainwires-autonomy.*",
            issue_number = issue.number,
            summary = changes.summary,
            files = changes
                .files_modified
                .iter()
                .map(|f| format!("- `{f}`"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let params = CreatePrParams {
            title,
            body,
            head_branch: branch_name.to_string(),
            base_branch: base_branch.to_string(),
            labels: vec!["automated".to_string()],
            draft: false,
        };

        let pr = self.forge.create_pull_request(repo, params).await?;
        tracing::info!("Created PR #{} for issue #{}", pr.number, issue.number);
        Ok(pr)
    }

    /// Request reviewers for a PR.
    pub async fn request_reviewers(
        &self,
        repo: &RepoRef,
        pr_number: u64,
        reviewers: &[String],
    ) -> anyhow::Result<()> {
        self.forge.request_review(repo, pr_number, reviewers).await
    }

    /// Add a comment to a PR.
    pub async fn add_comment(
        &self,
        repo: &RepoRef,
        pr_number: u64,
        body: &str,
    ) -> anyhow::Result<()> {
        self.forge.add_comment(repo, pr_number, body).await
    }
}
