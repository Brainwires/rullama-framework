//! Nextcloud Talk REST client + [`Channel`] implementation.
//!
//! The Talk OCS API is reached at:
//!
//! - `GET  /ocs/v2.php/apps/spreed/api/v1/chat/{roomToken}` — fetch
//!   messages. Query parameters:
//!   * `lookIntoFuture=1` — return only messages newer than
//!     `lastKnownMessageId`.
//!   * `lastKnownMessageId=<n>` — polling cursor.
//!   * `format=json` — canonical JSON body.
//! - `POST /ocs/v2.php/apps/spreed/api/v1/chat/{roomToken}` — send a
//!   text message; form-encoded `message=<text>`.
//!
//! Every request must carry `OCS-APIRequest: true`; without it the
//! server returns 406. Auth is HTTP Basic with a Nextcloud app password.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
    ThreadId,
};

/// Structured Spreed message row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreedMessage {
    /// Integer message id — monotonically increasing within a room.
    pub id: u64,
    /// Message type: `"comment"`, `"system"`, `"command"`, `"comment_deleted"`.
    #[serde(default, rename = "messageType")]
    pub message_type: String,
    /// Display name of the sender.
    #[serde(default, rename = "actorDisplayName")]
    pub actor_display_name: String,
    /// Stable id of the sender (username).
    #[serde(default, rename = "actorId")]
    pub actor_id: String,
    /// Message body — may contain `{mention-…}` placeholders.
    #[serde(default)]
    pub message: String,
    /// Unix epoch seconds.
    #[serde(default)]
    pub timestamp: i64,
    /// Optional parent message id for threaded replies.
    #[serde(default)]
    pub parent: Option<ParentRef>,
}

/// Parent message reference for threaded replies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentRef {
    /// The parent message id.
    pub id: u64,
}

/// OCS envelope wrapping every Spreed response.
#[derive(Debug, Clone, Deserialize)]
pub struct OcsEnvelope<T> {
    /// The OCS metadata + data payload.
    pub ocs: OcsBody<T>,
}

/// The OCS body.
#[derive(Debug, Clone, Deserialize)]
pub struct OcsBody<T> {
    /// Metadata — status code, message, etc. We rely on HTTP status;
    /// the field is retained for future observability.
    #[serde(default)]
    pub meta: OcsMeta,
    /// The payload.
    pub data: T,
}

/// OCS metadata block.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OcsMeta {
    /// Short status string.
    #[serde(default)]
    pub status: String,
    /// Numeric status code (OCS-specific, separate from HTTP status).
    #[serde(default)]
    pub statuscode: i64,
}

/// Request body for the MCP `send_message` tool.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SendMessageRequest {
    /// Target room token.
    pub room_token: String,
    /// Message body.
    pub message: String,
    /// Optional parent message id for a threaded reply.
    pub reply_to: Option<u64>,
}

/// [`Channel`] implementation over a Nextcloud Talk instance.
pub struct NextcloudTalkChannel {
    http: reqwest::Client,
    server_url: String,
    /// Precomputed `Basic <b64>` header value. Kept opaque — never logged.
    basic_auth: String,
    /// Short host fragment used for session ids.
    host_fragment: String,
}

impl NextcloudTalkChannel {
    /// Construct from server url + username + app password.
    pub fn new(
        server_url: impl Into<String>,
        username: impl Into<String>,
        app_password: impl Into<String>,
    ) -> Self {
        let server_url = server_url.into();
        let username = username.into();
        let app_password = app_password.into();
        let basic_auth = format!("Basic {}", B64.encode(format!("{username}:{app_password}")));
        let host_fragment = server_url
            .split("://")
            .nth(1)
            .unwrap_or(&server_url)
            .split('/')
            .next()
            .unwrap_or("")
            .to_string();
        Self {
            http: reqwest::Client::new(),
            server_url,
            basic_auth,
            host_fragment,
        }
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Host fragment used for building session ids — tests inspect this.
    pub fn host_fragment(&self) -> &str {
        &self.host_fragment
    }

    fn endpoint(&self, room_token: &str) -> String {
        format!(
            "{}/ocs/v2.php/apps/spreed/api/v1/chat/{}",
            self.server_url.trim_end_matches('/'),
            urlencoding::encode(room_token)
        )
    }

    /// Poll a room for messages newer than `last_known`.
    ///
    /// `last_known = 0` means "all recent messages" — sufficient on a
    /// fresh boot.
    pub async fn poll_room(&self, room_token: &str, last_known: u64) -> Result<Vec<SpreedMessage>> {
        let url = format!(
            "{}?lookIntoFuture=1&lastKnownMessageId={}&format=json&timeout=0&includeLastKnown=0",
            self.endpoint(room_token),
            last_known
        );
        let resp = self
            .http
            .get(&url)
            .header("Authorization", &self.basic_auth)
            .header("OCS-APIRequest", "true")
            .header("Accept", "application/json")
            .send()
            .await
            .context("GET Spreed chat")?;
        let status = resp.status();
        // Nextcloud returns 304 Not Modified when no new messages
        // arrived within the (0s) timeout window — treat as empty.
        if status.as_u16() == 304 {
            return Ok(Vec::new());
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            anyhow::bail!("Nextcloud Talk rate-limited (429)");
        }
        if !status.is_success() {
            anyhow::bail!("Spreed chat returned {status}");
        }
        let env: OcsEnvelope<Vec<SpreedMessage>> =
            resp.json().await.context("parse Spreed response")?;
        Ok(env.ocs.data)
    }
}

#[async_trait]
impl Channel for NextcloudTalkChannel {
    fn channel_type(&self) -> &str {
        "nextcloud_talk"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MENTIONS
            | ChannelCapabilities::THREADS
            | ChannelCapabilities::DELETE_MESSAGES
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let url = self.endpoint(&target.channel_id);
        let body = build_send_body(message);
        let resp = self
            .http
            .post(&url)
            .header("Authorization", &self.basic_auth)
            .header("OCS-APIRequest", "true")
            .header("Accept", "application/json")
            .form(&body)
            .send()
            .await
            .context("POST Spreed chat")?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Spreed send returned {status}: {} bytes", text.len());
        }
        let env: OcsEnvelope<SpreedMessage> =
            resp.json().await.context("parse Spreed send response")?;
        Ok(MessageId::new(env.ocs.data.id.to_string()))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        anyhow::bail!("edit_message is not yet implemented for nextcloud_talk")
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        anyhow::bail!("delete_message is not yet implemented for nextcloud_talk")
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        anyhow::bail!("add_reaction is not yet implemented for nextcloud_talk")
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        Ok(Vec::new())
    }
}

/// Build the form-encoded body for `POST .../chat/{room}`.
///
/// Returns a vector of `(key, value)` tuples rather than a serde `Value`
/// because Spreed accepts form-URL-encoded input, not JSON.
pub fn build_send_body(message: &ChannelMessage) -> Vec<(String, String)> {
    let mut body: Vec<(String, String)> = Vec::new();
    body.push(("message".into(), extract_text(message)));
    if let Some(reply) = &message.reply_to
        && let Ok(id) = reply.0.parse::<u64>()
    {
        body.push(("replyTo".into(), id.to_string()));
    }
    body
}

/// Convert a Spreed message into a [`ChannelMessage`], filtering out
/// system rows and command echoes.
pub fn spreed_to_channel(
    msg: &SpreedMessage,
    room_token: &str,
    host_fragment: &str,
) -> Option<ChannelMessage> {
    // System messages (join/leave, room renamed, etc.) are not forwarded.
    if msg.message_type != "comment" && msg.message_type != "comment_deleted" {
        return None;
    }
    if msg.message.is_empty() {
        return None;
    }
    let ts = Utc
        .timestamp_opt(msg.timestamp, 0)
        .single()
        .unwrap_or_else(Utc::now);
    let session_id = format!("nextcloud:{host_fragment}:{room_token}:{}", msg.actor_id);
    let mut metadata = HashMap::new();
    metadata.insert("nextcloud.room_token".into(), room_token.to_string());
    metadata.insert("nextcloud.actor_id".into(), msg.actor_id.clone());
    metadata.insert("nextcloud.session_id".into(), session_id);
    Some(ChannelMessage {
        id: MessageId::new(msg.id.to_string()),
        conversation: ConversationId {
            platform: "nextcloud_talk".into(),
            channel_id: room_token.to_string(),
            server_id: Some(host_fragment.to_string()),
        },
        author: if msg.actor_display_name.is_empty() {
            msg.actor_id.clone()
        } else {
            msg.actor_display_name.clone()
        },
        content: MessageContent::Text(msg.message.clone()),
        thread_id: msg.parent.as_ref().map(|p| ThreadId::new(p.id.to_string())),
        reply_to: msg
            .parent
            .as_ref()
            .map(|p| MessageId::new(p.id.to_string())),
        timestamp: ts,
        attachments: Vec::new(),
        metadata,
    })
}

fn extract_text(message: &ChannelMessage) -> String {
    match &message.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::RichText { markdown, .. } => markdown.clone(),
        MessageContent::Media(m) => match &m.caption {
            Some(c) => format!("{c}\n{}", m.url),
            None => m.url.clone(),
        },
        MessageContent::Embed(e) => {
            let mut parts = Vec::new();
            if let Some(t) = &e.title {
                parts.push(format!("**{t}**"));
            }
            if let Some(d) = &e.description {
                parts.push(d.clone());
            }
            parts.join("\n")
        }
        MessageContent::Mixed(items) => items
            .iter()
            .map(|c| {
                let stub = ChannelMessage {
                    id: message.id.clone(),
                    conversation: message.conversation.clone(),
                    author: message.author.clone(),
                    content: c.clone(),
                    thread_id: None,
                    reply_to: None,
                    timestamp: message.timestamp,
                    attachments: Vec::new(),
                    metadata: HashMap::new(),
                };
                extract_text(&stub)
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Convenience helper used by tests — `ms_since_epoch` → chrono.
pub fn parse_ts(secs: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(secs, 0).single().unwrap_or_else(Utc::now)
}

/// Build the structured OCS response shape Talk expects from mock servers.
pub fn ocs_wrap(data: serde_json::Value) -> serde_json::Value {
    json!({ "ocs": { "meta": { "status": "ok", "statuscode": 200 }, "data": data } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample(id: u64, msg_type: &str, text: &str, parent: Option<u64>) -> SpreedMessage {
        SpreedMessage {
            id,
            message_type: msg_type.into(),
            actor_display_name: "Alice".into(),
            actor_id: "alice".into(),
            message: text.into(),
            timestamp: 1_700_000_000,
            parent: parent.map(|p| ParentRef { id: p }),
        }
    }

    #[test]
    fn comment_maps_to_channel_message() {
        let s = sample(10, "comment", "hello", None);
        let m = spreed_to_channel(&s, "room-abc", "cloud.example.com").expect("parsed");
        assert_eq!(m.id.0, "10");
        assert_eq!(m.conversation.channel_id, "room-abc");
        assert_eq!(
            m.conversation.server_id.as_deref(),
            Some("cloud.example.com")
        );
        assert_eq!(m.author, "Alice");
        match m.content {
            MessageContent::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn system_messages_dropped() {
        let s = sample(11, "system", "alice joined", None);
        assert!(spreed_to_channel(&s, "r", "h").is_none());
    }

    #[test]
    fn parent_carries_over_as_thread_ref() {
        let s = sample(12, "comment", "reply", Some(7));
        let m = spreed_to_channel(&s, "r", "h").unwrap();
        assert_eq!(m.thread_id.as_ref().unwrap().0, "7");
        assert_eq!(m.reply_to.as_ref().unwrap().0, "7");
    }

    #[test]
    fn send_body_carries_reply_to_when_numeric() {
        let msg = ChannelMessage {
            id: MessageId::new("x"),
            conversation: ConversationId {
                platform: "nextcloud_talk".into(),
                channel_id: "r".into(),
                server_id: None,
            },
            author: "bot".into(),
            content: MessageContent::Text("hi".into()),
            thread_id: None,
            reply_to: Some(MessageId::new("42")),
            timestamp: Utc::now(),
            attachments: Vec::new(),
            metadata: HashMap::new(),
        };
        let body = build_send_body(&msg);
        assert!(body.contains(&("message".to_string(), "hi".to_string())));
        assert!(body.contains(&("replyTo".to_string(), "42".to_string())));
    }

    #[test]
    fn ocs_envelope_parses() {
        let v = ocs_wrap(json!([{
            "id": 1,
            "messageType": "comment",
            "actorId": "a",
            "actorDisplayName": "A",
            "message": "hi",
            "timestamp": 1700000000,
        }]));
        let env: OcsEnvelope<Vec<SpreedMessage>> = serde_json::from_value(v).unwrap();
        assert_eq!(env.ocs.data.len(), 1);
    }

    #[test]
    fn caps_and_channel_type() {
        let c =
            NextcloudTalkChannel::new("https://cloud.example.com", "alice", "app-pass-placeholder");
        assert_eq!(c.channel_type(), "nextcloud_talk");
        let caps = c.capabilities();
        assert!(caps.contains(ChannelCapabilities::RICH_TEXT));
        assert!(caps.contains(ChannelCapabilities::THREADS));
        assert!(!caps.contains(ChannelCapabilities::REACTIONS));
        assert_eq!(c.host_fragment(), "cloud.example.com");
    }

    #[test]
    fn host_fragment_strips_scheme_and_path() {
        let c = NextcloudTalkChannel::new("https://nc.example.org/cloud/", "u", "p");
        assert_eq!(c.host_fragment(), "nc.example.org");
    }

    #[test]
    fn parse_ts_roundtrip() {
        let t = parse_ts(1_700_000_000);
        assert_eq!(t.timestamp(), 1_700_000_000);
    }
}
