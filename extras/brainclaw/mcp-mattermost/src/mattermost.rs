//! Mattermost channel adapter implementing the `Channel` trait.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde_json::{Value, json};

use brainwires_network::channels::{
    ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId, ThreadId,
};

/// Mattermost REST API v4 base path.
const MM_API_BASE: &str = "/api/v4";

/// Mattermost channel adapter.
pub struct MattermostChannel {
    /// HTTP client.
    http: Client,
    /// Personal access token.
    token: String,
    /// Server base URL (e.g. "https://mattermost.example.com").
    server_url: String,
    /// Bot user ID — used for reactions and self-filtering.
    pub bot_user_id: String,
}

impl MattermostChannel {
    /// Create a new `MattermostChannel`.
    pub fn new(server_url: String, token: String, bot_user_id: String) -> Self {
        Self {
            http: Client::new(),
            token,
            server_url: server_url.trim_end_matches('/').to_string(),
            bot_user_id,
        }
    }

    /// Make an authenticated GET request to the Mattermost API.
    async fn api_get(&self, path: &str) -> Result<Value> {
        let url = format!("{}{}{}", self.server_url, MM_API_BASE, path);
        let response = self
            .http
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .with_context(|| format!("GET {path} failed"))?;

        let status = response.status();
        let json: Value = response
            .json()
            .await
            .with_context(|| format!("Failed to parse response for GET {path}"))?;

        if !status.is_success() {
            let msg = json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("Mattermost API error {status} at {path}: {msg}");
        }

        Ok(json)
    }

    /// Make an authenticated POST request to the Mattermost API.
    async fn api_post(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}{}", self.server_url, MM_API_BASE, path);
        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {path} failed"))?;

        let status = response.status();
        let json: Value = response
            .json()
            .await
            .with_context(|| format!("Failed to parse response for POST {path}"))?;

        if !status.is_success() {
            let msg = json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("Mattermost API error {status} at {path}: {msg}");
        }

        Ok(json)
    }

    /// Make an authenticated PUT request to the Mattermost API.
    async fn api_put(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}{}", self.server_url, MM_API_BASE, path);
        let response = self
            .http
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(body)
            .send()
            .await
            .with_context(|| format!("PUT {path} failed"))?;

        let status = response.status();
        let json: Value = response
            .json()
            .await
            .with_context(|| format!("Failed to parse response for PUT {path}"))?;

        if !status.is_success() {
            let msg = json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("Mattermost API error {status} at {path}: {msg}");
        }

        Ok(json)
    }

    /// Make an authenticated DELETE request to the Mattermost API.
    async fn api_delete(&self, path: &str) -> Result<()> {
        let url = format!("{}{}{}", self.server_url, MM_API_BASE, path);
        let response = self
            .http
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .with_context(|| format!("DELETE {path} failed"))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Mattermost DELETE {path} failed: {body}");
        }

        Ok(())
    }

    /// Fetch channels the bot is a member of, optionally filtered by team.
    pub async fn get_bot_channels(&self, team_id: Option<&str>) -> Result<Vec<Value>> {
        let path = if let Some(tid) = team_id {
            format!("/users/me/teams/{tid}/channels")
        } else {
            "/users/me/channels".to_string()
        };
        let arr = self.api_get(&path).await?;
        Ok(arr.as_array().cloned().unwrap_or_default())
    }
}

/// Parse Mattermost timestamp (milliseconds since epoch) to `DateTime<Utc>`.
fn parse_mm_ts(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(ms / 1000, ((ms % 1000) as u32) * 1_000_000).unwrap_or_else(Utc::now)
}

#[async_trait]
impl brainwires_network::channels::Channel for MattermostChannel {
    fn channel_type(&self) -> &str {
        "mattermost"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::THREADS
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
            | ChannelCapabilities::MENTIONS
            | ChannelCapabilities::TYPING_INDICATOR
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let text = match &message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            _ => return Ok(MessageId::new("unsupported")),
        };

        let mut body = json!({
            "channel_id": target.channel_id,
            "message": text,
        });

        // Reply in thread if thread_id is set
        if let Some(ref tid) = message.thread_id {
            body["root_id"] = json!(tid.0);
        }

        // If replying to a specific message, use that as root_id
        if let Some(ref reply_to) = message.reply_to
            && body["root_id"].is_null()
        {
            body["root_id"] = json!(reply_to.0);
        }

        let resp = self.api_post("/posts", &body).await?;
        let post_id = resp
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(MessageId::new(post_id))
    }

    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()> {
        let text = match &message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            _ => return Ok(()),
        };
        let path = format!("/posts/{}", id.0);
        let body = json!({ "id": id.0, "message": text });
        self.api_put(&path, &body).await?;
        Ok(())
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        let path = format!("/posts/{}", id.0);
        self.api_delete(&path).await
    }

    async fn send_typing(&self, target: &ConversationId) -> Result<()> {
        let body = json!({ "channel_id": target.channel_id });
        self.api_post("/users/me/typing", &body).await.map(|_| ())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        let emoji_name = emoji.trim_matches(':');
        let body = json!({
            "user_id": self.bot_user_id,
            "post_id": id.0,
            "emoji_name": emoji_name,
        });
        self.api_post("/reactions", &body).await.map(|_| ())
    }

    async fn get_history(
        &self,
        target: &ConversationId,
        limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        let channel_id = &target.channel_id;
        let path = format!("/channels/{channel_id}/posts?per_page={limit}");

        let resp = self.api_get(&path).await?;
        let order = resp
            .get("order")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let posts = resp
            .get("posts")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let mut messages = Vec::new();
        for post_id_val in &order {
            let post_id = match post_id_val.as_str() {
                Some(s) => s,
                None => continue,
            };
            let post = match posts.get(post_id) {
                Some(p) => p,
                None => continue,
            };

            let text = post
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let author = post
                .get("user_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let ts_ms = post.get("create_at").and_then(|v| v.as_i64()).unwrap_or(0);
            let root_id = post
                .get("root_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| ThreadId(s.to_string()));

            messages.push(ChannelMessage {
                id: MessageId::new(post_id.to_string()),
                conversation: target.clone(),
                author,
                content: MessageContent::Text(text),
                thread_id: root_id,
                reply_to: None,
                timestamp: parse_mm_ts(ts_ms),
                attachments: Vec::new(),
                metadata: HashMap::new(),
            });
        }

        Ok(messages)
    }
}
