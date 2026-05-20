//! GitHub REST API client implementing the `Channel` trait.
//!
//! `ConversationId` mapping:
//! - `platform`   : `"github"`
//! - `channel_id` : `"owner/repo#<issue-or-pr-number>"`  (e.g. `"octocat/hello-world#42"`)
//! - `server_id`  : `None`

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
};
use chrono::Utc;
use reqwest::{Client, header};
use serde_json::{Value, json};
use std::collections::HashMap;

/// GitHub channel adapter backed by the REST API.
pub struct GitHubChannel {
    client: Client,
    api_url: String,
}

impl GitHubChannel {
    /// Create a new channel with the given PAT/App token.
    pub fn new(token: &str, api_url: &str) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {token}"))
                .context("invalid token characters")?,
        );
        headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static("application/vnd.github+json"),
        );
        headers.insert(
            "X-GitHub-Api-Version",
            header::HeaderValue::from_static("2022-11-28"),
        );
        // GitHub requires a User-Agent header
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_static("brainclaw-mcp-github/0.8.0"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            client,
            api_url: api_url.trim_end_matches('/').to_string(),
        })
    }

    /// Parse `"owner/repo#123"` → `(owner, repo, issue_number)`.
    fn parse_conversation(conv: &ConversationId) -> Result<(String, String, u64)> {
        let parts: Vec<&str> = conv.channel_id.splitn(2, '#').collect();
        if parts.len() != 2 {
            return Err(anyhow!(
                "expected channel_id = 'owner/repo#<number>', got '{}'",
                conv.channel_id
            ));
        }
        let repo_parts: Vec<&str> = parts[0].splitn(2, '/').collect();
        if repo_parts.len() != 2 {
            return Err(anyhow!("expected repo = 'owner/repo', got '{}'", parts[0]));
        }
        let num: u64 = parts[1]
            .parse()
            .with_context(|| format!("non-numeric issue number: '{}'", parts[1]))?;
        Ok((repo_parts[0].to_string(), repo_parts[1].to_string(), num))
    }

    /// Extract comment ID from `MessageId` (format: `"<owner>/<repo>#<issue>/<comment_id>"`).
    fn parse_message_id(id: &MessageId) -> Result<(String, String, u64)> {
        // Format: "owner/repo/comments/<comment_id>"
        let s = &id.0;
        let parts: Vec<&str> = s.rsplitn(2, '/').collect();
        if parts.len() != 2 {
            return Err(anyhow!("unexpected MessageId format: {s}"));
        }
        let comment_id: u64 = parts[0]
            .parse()
            .with_context(|| format!("non-numeric comment id: '{}'", parts[0]))?;
        // parts[1] = "owner/repo/issues" or "owner/repo"
        let repo_parts: Vec<&str> = parts[1].splitn(3, '/').collect();
        if repo_parts.len() < 2 {
            return Err(anyhow!("cannot extract owner/repo from MessageId: {s}"));
        }
        Ok((
            repo_parts[0].to_string(),
            repo_parts[1].to_string(),
            comment_id,
        ))
    }
}

#[async_trait]
impl Channel for GitHubChannel {
    fn channel_type(&self) -> &str {
        "github"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let (owner, repo, issue_number) = Self::parse_conversation(target)?;
        let body = match &message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            _ => return Err(anyhow!("GitHub only supports text/markdown content")),
        };

        let url = format!(
            "{}/repos/{owner}/{repo}/issues/{issue_number}/comments",
            self.api_url
        );
        let resp: Value = self
            .client
            .post(&url)
            .json(&json!({ "body": body }))
            .send()
            .await
            .context("GitHub API request failed")?
            .error_for_status()
            .context("GitHub API returned an error")?
            .json()
            .await
            .context("failed to parse GitHub response")?;

        let comment_id = resp["id"]
            .as_u64()
            .ok_or_else(|| anyhow!("missing 'id' in GitHub response"))?;

        Ok(MessageId::new(format!("{owner}/{repo}/{comment_id}")))
    }

    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()> {
        let (owner, repo, comment_id) = Self::parse_message_id(id)?;
        let body = match &message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            _ => return Err(anyhow!("GitHub only supports text/markdown content")),
        };

        let url = format!(
            "{}/repos/{owner}/{repo}/issues/comments/{comment_id}",
            self.api_url
        );
        self.client
            .patch(&url)
            .json(&json!({ "body": body }))
            .send()
            .await
            .context("GitHub API PATCH failed")?
            .error_for_status()
            .context("GitHub API returned an error")?;
        Ok(())
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        let (owner, repo, comment_id) = Self::parse_message_id(id)?;
        let url = format!(
            "{}/repos/{owner}/{repo}/issues/comments/{comment_id}",
            self.api_url
        );
        self.client
            .delete(&url)
            .send()
            .await
            .context("GitHub API DELETE failed")?
            .error_for_status()
            .context("GitHub API returned an error")?;
        Ok(())
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        // GitHub has no typing indicator; silently succeed
        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        // Map from Unicode emoji to GitHub reaction content string
        let content = unicode_to_github_reaction(emoji);
        let (owner, repo, comment_id) = Self::parse_message_id(id)?;
        let url = format!(
            "{}/repos/{owner}/{repo}/issues/comments/{comment_id}/reactions",
            self.api_url
        );
        self.client
            .post(&url)
            .json(&json!({ "content": content }))
            .send()
            .await
            .context("GitHub API reactions POST failed")?
            .error_for_status()
            .context("GitHub API returned an error")?;
        Ok(())
    }

    async fn get_history(
        &self,
        target: &ConversationId,
        limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        let (owner, repo, issue_number) = Self::parse_conversation(target)?;
        let per_page = limit.min(100);
        let url = format!(
            "{}/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page={per_page}&direction=asc",
            self.api_url
        );
        let items: Vec<Value> = self
            .client
            .get(&url)
            .send()
            .await
            .context("GitHub API GET failed")?
            .error_for_status()
            .context("GitHub API returned an error")?
            .json()
            .await
            .context("failed to parse GitHub response")?;

        let messages = items
            .into_iter()
            .take(limit)
            .map(|item| {
                let comment_id = item["id"].as_u64().unwrap_or(0);
                let author = item["user"]["login"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                let body = item["body"].as_str().unwrap_or("").to_string();
                let created_at = item["created_at"]
                    .as_str()
                    .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok())
                    .unwrap_or_else(Utc::now);

                ChannelMessage {
                    id: MessageId::new(format!("{owner}/{repo}/{comment_id}")),
                    conversation: target.clone(),
                    author,
                    content: MessageContent::Text(body),
                    thread_id: None,
                    reply_to: None,
                    timestamp: created_at,
                    attachments: vec![],
                    metadata: HashMap::new(),
                }
            })
            .collect();

        Ok(messages)
    }
}

/// Map a Unicode emoji to the nearest GitHub reaction identifier.
fn unicode_to_github_reaction(emoji: &str) -> &str {
    match emoji {
        "👍" | "+1" => "+1",
        "👎" | "-1" => "-1",
        "😄" | "😀" => "laugh",
        "🎉" | "party" => "hooray",
        "😕" | "confused" => "confused",
        "❤️" | "heart" | "♥" => "heart",
        "🚀" | "rocket" => "rocket",
        "👀" | "eyes" => "eyes",
        _ => "+1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_conversation_valid() {
        let conv = ConversationId {
            platform: "github".to_string(),
            channel_id: "octocat/hello-world#42".to_string(),
            server_id: None,
        };
        let (owner, repo, num) = GitHubChannel::parse_conversation(&conv).unwrap();
        assert_eq!(owner, "octocat");
        assert_eq!(repo, "hello-world");
        assert_eq!(num, 42);
    }

    #[test]
    fn parse_conversation_invalid_missing_hash() {
        let conv = ConversationId {
            platform: "github".to_string(),
            channel_id: "octocat/hello-world".to_string(),
            server_id: None,
        };
        assert!(GitHubChannel::parse_conversation(&conv).is_err());
    }

    #[test]
    fn parse_message_id_valid() {
        let id = MessageId::new("octocat/hello-world/12345");
        let (owner, repo, cid) = GitHubChannel::parse_message_id(&id).unwrap();
        assert_eq!(owner, "octocat");
        assert_eq!(repo, "hello-world");
        assert_eq!(cid, 12345);
    }

    #[test]
    fn unicode_to_github_reaction_known() {
        assert_eq!(unicode_to_github_reaction("👍"), "+1");
        assert_eq!(unicode_to_github_reaction("🎉"), "hooray");
        assert_eq!(unicode_to_github_reaction("🚀"), "rocket");
    }

    #[test]
    fn unicode_to_github_reaction_unknown_falls_back() {
        assert_eq!(unicode_to_github_reaction("🦄"), "+1");
    }
}
