//! Git forge abstraction — GitHub, GitLab, Gitea, etc.
//!
//! Defines the [`GitForge`] trait for interacting with Git hosting platforms and
//! provides a [`GitHubForge`] implementation using the GitHub REST API.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Reference to a repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoRef {
    /// Repository owner (user or organization).
    pub owner: String,
    /// Repository name.
    pub name: String,
}

impl RepoRef {
    /// Return the full `owner/name` string.
    pub fn full_name(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

/// An issue from the forge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    /// Issue identifier.
    pub id: String,
    /// Issue number.
    pub number: u64,
    /// Issue title.
    pub title: String,
    /// Issue body text.
    pub body: String,
    /// Labels applied to the issue.
    pub labels: Vec<String>,
    /// Author username.
    pub author: String,
    /// URL of the issue.
    pub url: String,
}

/// A comment on an issue or PR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    /// Comment identifier.
    pub id: String,
    /// Author username.
    pub author: String,
    /// Comment body text.
    pub body: String,
}

/// A commit reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRef {
    /// Commit SHA hash.
    pub sha: String,
    /// Commit message.
    pub message: String,
}

/// A pull request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    /// Pull request identifier.
    pub id: String,
    /// PR number.
    pub number: u64,
    /// PR title.
    pub title: String,
    /// PR body text.
    pub body: String,
    /// Source branch name.
    pub head_branch: String,
    /// Target branch name.
    pub base_branch: String,
    /// URL of the pull request.
    pub url: String,
    /// Current PR state.
    pub state: PrState,
}

/// Pull request state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrState {
    /// PR is open and accepting changes.
    Open,
    /// PR has been closed without merging.
    Closed,
    /// PR has been merged.
    Merged,
}

/// Parameters for creating a pull request.
#[derive(Debug, Clone)]
pub struct CreatePrParams {
    /// Title for the new PR.
    pub title: String,
    /// Body text for the new PR.
    pub body: String,
    /// Source branch name.
    pub head_branch: String,
    /// Target branch name.
    pub base_branch: String,
    /// Labels to apply to the PR.
    pub labels: Vec<String>,
    /// Whether to create as a draft PR.
    pub draft: bool,
}

/// Merge method for pull requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeMethod {
    /// Standard merge commit.
    Merge,
    /// Squash and merge.
    Squash,
    /// Rebase and merge.
    Rebase,
}

/// CI/CD check status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckStatus {
    /// Aggregate check state.
    pub state: CheckState,
    /// Individual check runs.
    pub checks: Vec<CheckRun>,
}

/// Overall state of CI checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckState {
    /// Checks are still running.
    Pending,
    /// All checks passed.
    Success,
    /// One or more checks failed.
    Failure,
    /// An error occurred evaluating checks.
    Error,
}

/// A single CI check run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckRun {
    /// Name of the check run.
    pub name: String,
    /// Current status (e.g. "completed", "in_progress").
    pub status: String,
    /// Conclusion if the check has completed (e.g. "success", "failure").
    pub conclusion: Option<String>,
}

/// Abstract interface over Git forges (GitHub, GitLab, Gitea, etc.).
///
/// Implementations handle issue retrieval, PR creation/merging, CI status checks,
/// and review requests for a specific forge platform.
#[async_trait]
pub trait GitForge: Send + Sync {
    /// Forge name (e.g. "github", "gitlab").
    fn name(&self) -> &str;

    /// Fetch an issue by reference (e.g. "owner/repo#123" or just "123").
    async fn get_issue(&self, repo: &RepoRef, issue_ref: &str) -> anyhow::Result<Issue>;

    /// Create a pull request.
    async fn create_pull_request(
        &self,
        repo: &RepoRef,
        params: CreatePrParams,
    ) -> anyhow::Result<PullRequest>;

    /// Add a comment to an issue or PR.
    async fn add_comment(
        &self,
        repo: &RepoRef,
        target_number: u64,
        body: &str,
    ) -> anyhow::Result<()>;

    /// Merge a pull request.
    async fn merge_pull_request(
        &self,
        repo: &RepoRef,
        pr_number: u64,
        method: MergeMethod,
    ) -> anyhow::Result<()>;

    /// Get CI check status for a PR.
    async fn get_check_status(&self, repo: &RepoRef, pr_number: u64)
    -> anyhow::Result<CheckStatus>;

    /// Request reviewers for a PR.
    async fn request_review(
        &self,
        repo: &RepoRef,
        pr_number: u64,
        reviewers: &[String],
    ) -> anyhow::Result<()>;
}

/// GitHub forge implementation using the REST API via reqwest.
pub struct GitHubForge {
    token: String,
    client: reqwest::Client,
    api_base: String,
}

impl GitHubForge {
    /// Create a new GitHub forge client with the given API token.
    pub fn new(token: String) -> Self {
        Self {
            token,
            client: reqwest::Client::new(),
            api_base: "https://api.github.com".to_string(),
        }
    }

    /// Override the API base URL (for GitHub Enterprise).
    pub fn with_api_base(mut self, base: String) -> Self {
        self.api_base = base;
        self
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.token)
    }
}

#[async_trait]
impl GitForge for GitHubForge {
    fn name(&self) -> &str {
        "github"
    }

    async fn get_issue(&self, repo: &RepoRef, issue_ref: &str) -> anyhow::Result<Issue> {
        let number: u64 = issue_ref
            .trim_start_matches('#')
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid issue reference: {issue_ref}"))?;

        let url = format!(
            "{}/repos/{}/{}/issues/{number}",
            self.api_base, repo.owner, repo.name
        );

        let resp: serde_json::Value = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(Issue {
            id: resp["id"].to_string(),
            number,
            title: resp["title"].as_str().unwrap_or("").to_string(),
            body: resp["body"].as_str().unwrap_or("").to_string(),
            labels: resp["labels"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|l| l["name"].as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            author: resp["user"]["login"].as_str().unwrap_or("").to_string(),
            url: resp["html_url"].as_str().unwrap_or("").to_string(),
        })
    }

    async fn create_pull_request(
        &self,
        repo: &RepoRef,
        params: CreatePrParams,
    ) -> anyhow::Result<PullRequest> {
        let url = format!("{}/repos/{}/{}/pulls", self.api_base, repo.owner, repo.name);

        let body = serde_json::json!({
            "title": params.title,
            "body": params.body,
            "head": params.head_branch,
            "base": params.base_branch,
            "draft": params.draft,
        });

        let resp: serde_json::Value = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(PullRequest {
            id: resp["id"].to_string(),
            number: resp["number"].as_u64().unwrap_or(0),
            title: resp["title"].as_str().unwrap_or("").to_string(),
            body: resp["body"].as_str().unwrap_or("").to_string(),
            head_branch: params.head_branch,
            base_branch: params.base_branch,
            url: resp["html_url"].as_str().unwrap_or("").to_string(),
            state: PrState::Open,
        })
    }

    async fn add_comment(
        &self,
        repo: &RepoRef,
        target_number: u64,
        body: &str,
    ) -> anyhow::Result<()> {
        let url = format!(
            "{}/repos/{}/{}/issues/{target_number}/comments",
            self.api_base, repo.owner, repo.name
        );

        self.client
            .post(&url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .json(&serde_json::json!({ "body": body }))
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }

    async fn merge_pull_request(
        &self,
        repo: &RepoRef,
        pr_number: u64,
        method: MergeMethod,
    ) -> anyhow::Result<()> {
        let url = format!(
            "{}/repos/{}/{}/pulls/{pr_number}/merge",
            self.api_base, repo.owner, repo.name
        );

        let merge_method = match method {
            MergeMethod::Merge => "merge",
            MergeMethod::Squash => "squash",
            MergeMethod::Rebase => "rebase",
        };

        self.client
            .put(&url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .json(&serde_json::json!({ "merge_method": merge_method }))
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }

    async fn get_check_status(
        &self,
        repo: &RepoRef,
        pr_number: u64,
    ) -> anyhow::Result<CheckStatus> {
        // First get the PR to find the head SHA
        let pr_url = format!(
            "{}/repos/{}/{}/pulls/{pr_number}",
            self.api_base, repo.owner, repo.name
        );

        let pr_resp: serde_json::Value = self
            .client
            .get(&pr_url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let sha = pr_resp["head"]["sha"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No head SHA found"))?;

        let status_url = format!(
            "{}/repos/{}/{}/commits/{sha}/check-runs",
            self.api_base, repo.owner, repo.name
        );

        let resp: serde_json::Value = self
            .client
            .get(&status_url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let checks: Vec<CheckRun> = resp["check_runs"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|c| CheckRun {
                        name: c["name"].as_str().unwrap_or("").to_string(),
                        status: c["status"].as_str().unwrap_or("").to_string(),
                        conclusion: c["conclusion"].as_str().map(|s| s.to_string()),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let state = if checks
            .iter()
            .all(|c| c.conclusion.as_deref() == Some("success"))
        {
            CheckState::Success
        } else if checks
            .iter()
            .any(|c| c.conclusion.as_deref() == Some("failure"))
        {
            CheckState::Failure
        } else if checks.iter().any(|c| c.status != "completed") {
            CheckState::Pending
        } else {
            CheckState::Error
        };

        Ok(CheckStatus { state, checks })
    }

    async fn request_review(
        &self,
        repo: &RepoRef,
        pr_number: u64,
        reviewers: &[String],
    ) -> anyhow::Result<()> {
        let url = format!(
            "{}/repos/{}/{}/pulls/{pr_number}/requested_reviewers",
            self.api_base, repo.owner, repo.name
        );

        self.client
            .post(&url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "brainwires-autonomy")
            .json(&serde_json::json!({ "reviewers": reviewers }))
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ref_full_name() {
        let r = RepoRef {
            owner: "nightness".to_string(),
            name: "brainwires".to_string(),
        };
        assert_eq!(r.full_name(), "nightness/brainwires");
    }

    #[test]
    fn repo_ref_full_name_with_org() {
        let r = RepoRef {
            owner: "my-org".to_string(),
            name: "my-repo".to_string(),
        };
        assert_eq!(r.full_name(), "my-org/my-repo");
    }
}
