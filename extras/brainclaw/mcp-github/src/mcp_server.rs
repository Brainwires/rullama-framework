//! MCP server exposing GitHub operations as tools.
//!
//! Tools:
//! - `post_comment`          — Post a comment on an issue or PR
//! - `edit_comment`          — Edit an existing comment
//! - `delete_comment`        — Delete a comment
//! - `get_comments`          — Fetch comment history for an issue/PR
//! - `create_issue`          — Open a new issue
//! - `close_issue`           — Close an existing issue
//! - `add_labels`            — Add labels to an issue or PR
//! - `create_pull_request`   — Open a new pull request
//! - `merge_pull_request`    — Merge a pull request
//! - `add_reaction`          — React to a comment with an emoji

use std::sync::Arc;

use anyhow::Result;
use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};
use chrono::Utc;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

use crate::github::GitHubChannel;

/// MCP server wrapping a `GitHubChannel` to expose GitHub operations as tools.
#[derive(Clone)]
pub struct GitHubMcpServer {
    channel: Arc<GitHubChannel>,
    /// Direct access to reqwest client for operations beyond the Channel trait
    api_url: String,
    token: String,
    tool_router: ToolRouter<Self>,
}

impl GitHubMcpServer {
    /// Create a new MCP server.
    pub fn new(channel: Arc<GitHubChannel>, api_url: String, token: String) -> Self {
        Self {
            channel,
            api_url,
            token,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdin/stdout.
    pub async fn serve_stdio(
        channel: Arc<GitHubChannel>,
        api_url: String,
        token: String,
    ) -> Result<()> {
        tracing::info!("Starting GitHub MCP server on stdio");
        let server = Self::new(channel, api_url, token);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }

    /// Build an HTTP client with GitHub auth headers for extended operations.
    fn http_client(&self) -> reqwest::Client {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.token))
                .expect("valid token"),
        );
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("application/vnd.github+json"),
        );
        headers.insert(
            "X-GitHub-Api-Version",
            reqwest::header::HeaderValue::from_static("2022-11-28"),
        );
        headers.insert(
            reqwest::header::USER_AGENT,
            reqwest::header::HeaderValue::from_static("brainclaw-mcp-github/0.8.0"),
        );
        reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .expect("HTTP client")
    }
}

// ── Tool request types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PostCommentRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Issue or PR number.
    pub issue_number: u64,
    /// Markdown body of the comment.
    pub body: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditCommentRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Comment ID (as returned by `post_comment`).
    pub comment_id: u64,
    /// New body text.
    pub body: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteCommentRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Comment ID to delete.
    pub comment_id: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetCommentsRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Issue or PR number.
    pub issue_number: u64,
    /// Maximum number of comments to return (1-100, default 25).
    pub limit: Option<u8>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreateIssueRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Issue title.
    pub title: String,
    /// Issue body (Markdown). Optional.
    pub body: Option<String>,
    /// Labels to apply. Optional.
    pub labels: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CloseIssueRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Issue number to close.
    pub issue_number: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddLabelsRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Issue or PR number.
    pub issue_number: u64,
    /// Labels to add.
    pub labels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreatePullRequestRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// PR title.
    pub title: String,
    /// Head branch (e.g. `feature/my-branch`).
    pub head: String,
    /// Base branch (e.g. `main`).
    pub base: String,
    /// PR body (Markdown). Optional.
    pub body: Option<String>,
    /// Open as a draft PR.
    #[serde(default)]
    pub draft: bool,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MergePullRequestRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// PR number.
    pub pull_number: u64,
    /// Merge commit message. Optional.
    pub commit_message: Option<String>,
    /// Merge strategy: `merge`, `squash`, or `rebase`. Default: `merge`.
    pub merge_method: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// Repository in `owner/repo` format.
    pub repo: String,
    /// Comment ID.
    pub comment_id: u64,
    /// Emoji to react with (Unicode or GitHub name like `+1`, `rocket`, `eyes`).
    pub emoji: String,
}

// ── Tool implementations ──────────────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl GitHubMcpServer {
    #[tool(description = "Post a comment on a GitHub issue or pull request.")]
    async fn post_comment(
        &self,
        Parameters(req): Parameters<PostCommentRequest>,
    ) -> Result<String, String> {
        let conv = ConversationId {
            platform: "github".to_string(),
            channel_id: format!("{}#{}", req.repo, req.issue_number),
            server_id: None,
        };
        let msg = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conv.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.body),
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        };
        let id = self
            .channel
            .send_message(&conv, &msg)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(json!({ "comment_id": id.0 }).to_string())
    }

    #[tool(description = "Edit an existing GitHub issue or PR comment.")]
    async fn edit_comment(
        &self,
        Parameters(req): Parameters<EditCommentRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(format!("{}/{}", req.repo, req.comment_id));
        let msg = ChannelMessage {
            id: id.clone(),
            conversation: ConversationId {
                platform: "github".to_string(),
                channel_id: req.repo.clone(),
                server_id: None,
            },
            author: "bot".to_string(),
            content: MessageContent::Text(req.body),
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        };
        self.channel
            .edit_message(&id, &msg)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(r#"{"status":"edited"}"#.to_string())
    }

    #[tool(description = "Delete a GitHub issue or PR comment.")]
    async fn delete_comment(
        &self,
        Parameters(req): Parameters<DeleteCommentRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(format!("{}/{}", req.repo, req.comment_id));
        self.channel
            .delete_message(&id)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(r#"{"status":"deleted"}"#.to_string())
    }

    #[tool(description = "Fetch comments on a GitHub issue or PR (up to 100).")]
    async fn get_comments(
        &self,
        Parameters(req): Parameters<GetCommentsRequest>,
    ) -> Result<String, String> {
        let conv = ConversationId {
            platform: "github".to_string(),
            channel_id: format!("{}#{}", req.repo, req.issue_number),
            server_id: None,
        };
        let limit = req.limit.unwrap_or(25) as usize;
        let msgs = self
            .channel
            .get_history(&conv, limit)
            .await
            .map_err(|e| format!("{e:#}"))?;
        serde_json::to_string_pretty(&msgs).map_err(|e| format!("serialization error: {e}"))
    }

    #[tool(description = "Create a new GitHub issue.")]
    async fn create_issue(
        &self,
        Parameters(req): Parameters<CreateIssueRequest>,
    ) -> Result<String, String> {
        let parts: Vec<&str> = req.repo.splitn(2, '/').collect();
        if parts.len() != 2 {
            return Err(format!("invalid repo format: {}", req.repo));
        }
        let url = format!("{}/repos/{}/issues", self.api_url, req.repo);
        let mut body_json = json!({ "title": req.title });
        if let Some(b) = req.body {
            body_json["body"] = json!(b);
        }
        if let Some(labels) = req.labels {
            body_json["labels"] = json!(labels);
        }
        let resp: serde_json::Value = self
            .http_client()
            .post(&url)
            .json(&body_json)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GitHub error: {e}"))?
            .json()
            .await
            .map_err(|e| format!("parse error: {e}"))?;
        Ok(json!({ "issue_number": resp["number"], "url": resp["html_url"] }).to_string())
    }

    #[tool(description = "Close a GitHub issue.")]
    async fn close_issue(
        &self,
        Parameters(req): Parameters<CloseIssueRequest>,
    ) -> Result<String, String> {
        let url = format!(
            "{}/repos/{}/issues/{}",
            self.api_url, req.repo, req.issue_number
        );
        self.http_client()
            .patch(&url)
            .json(&json!({ "state": "closed" }))
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GitHub error: {e}"))?;
        Ok(r#"{"status":"closed"}"#.to_string())
    }

    #[tool(description = "Add labels to a GitHub issue or pull request.")]
    async fn add_labels(
        &self,
        Parameters(req): Parameters<AddLabelsRequest>,
    ) -> Result<String, String> {
        let url = format!(
            "{}/repos/{}/issues/{}/labels",
            self.api_url, req.repo, req.issue_number
        );
        self.http_client()
            .post(&url)
            .json(&json!({ "labels": req.labels }))
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GitHub error: {e}"))?;
        Ok(r#"{"status":"labels_added"}"#.to_string())
    }

    #[tool(description = "Open a new GitHub pull request.")]
    async fn create_pull_request(
        &self,
        Parameters(req): Parameters<CreatePullRequestRequest>,
    ) -> Result<String, String> {
        let url = format!("{}/repos/{}/pulls", self.api_url, req.repo);
        let mut body_json = json!({
            "title": req.title,
            "head": req.head,
            "base": req.base,
            "draft": req.draft,
        });
        if let Some(b) = req.body {
            body_json["body"] = json!(b);
        }
        let resp: serde_json::Value = self
            .http_client()
            .post(&url)
            .json(&body_json)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GitHub error: {e}"))?
            .json()
            .await
            .map_err(|e| format!("parse error: {e}"))?;
        Ok(json!({ "pr_number": resp["number"], "url": resp["html_url"] }).to_string())
    }

    #[tool(description = "Merge a GitHub pull request.")]
    async fn merge_pull_request(
        &self,
        Parameters(req): Parameters<MergePullRequestRequest>,
    ) -> Result<String, String> {
        let url = format!(
            "{}/repos/{}/pulls/{}/merge",
            self.api_url, req.repo, req.pull_number
        );
        let mut body_json = json!({});
        if let Some(msg) = req.commit_message {
            body_json["commit_message"] = json!(msg);
        }
        body_json["merge_method"] = json!(req.merge_method.as_deref().unwrap_or("merge"));
        self.http_client()
            .put(&url)
            .json(&body_json)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GitHub error: {e}"))?;
        Ok(r#"{"status":"merged"}"#.to_string())
    }

    #[tool(description = "Add an emoji reaction to a GitHub issue comment.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(format!("{}/{}", req.repo, req.comment_id));
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(r#"{"status":"reacted"}"#.to_string())
    }
}

// ── ServerHandler ─────────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
impl ServerHandler for GitHubMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainclaw-mcp-github", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires GitHub Channel — MCP Tool Server");
        info.instructions = Some(
            "GitHub channel adapter. Use post_comment to comment on issues/PRs, \
             create_issue to open issues, create_pull_request to open PRs, \
             merge_pull_request to merge, add_labels to label, and get_comments \
             to fetch comment history."
                .into(),
        );
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_comment_request_serialization() {
        let req = PostCommentRequest {
            repo: "octocat/hello-world".to_string(),
            issue_number: 42,
            body: "LGTM!".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("octocat/hello-world"));
        assert!(json.contains("42"));
    }

    #[test]
    fn create_issue_request_optional_fields() {
        let req = CreateIssueRequest {
            repo: "owner/repo".to_string(),
            title: "Bug report".to_string(),
            body: None,
            labels: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: CreateIssueRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.title, "Bug report");
        assert!(parsed.body.is_none());
    }

    #[test]
    fn merge_method_defaults() {
        let req = MergePullRequestRequest {
            repo: "owner/repo".to_string(),
            pull_number: 5,
            commit_message: None,
            merge_method: None,
        };
        assert!(req.merge_method.is_none());
    }
}
