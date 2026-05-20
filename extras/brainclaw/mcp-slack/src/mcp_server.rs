//! MCP server exposing Slack channel operations as tools.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::slack::SlackChannel;

/// MCP server wrapping a `SlackChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct SlackMcpServer {
    channel: Arc<SlackChannel>,
    tool_router: ToolRouter<Self>,
}

impl SlackMcpServer {
    /// Create a new MCP server wrapping the given Slack channel.
    pub fn new(channel: Arc<SlackChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdin/stdout (MCP standard I/O transport).
    pub async fn serve_stdio(channel: Arc<SlackChannel>) -> Result<()> {
        tracing::info!("Starting Slack MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// -- Tool request types -------------------------------------------------------

/// Request to send a message to a Slack channel.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// The Slack channel ID to send to (e.g., "C0123456789").
    pub channel_id: String,
    /// The text content to send.
    pub content: String,
    /// Optional team/workspace ID.
    pub server_id: Option<String>,
    /// Optional thread timestamp to reply in a thread.
    pub thread_ts: Option<String>,
}

/// Request to edit a Slack message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditMessageRequest {
    /// The Slack channel ID the message is in.
    pub channel_id: String,
    /// The message timestamp (ts) to edit.
    pub message_ts: String,
    /// The new text content.
    pub content: String,
    /// Optional team/workspace ID.
    pub server_id: Option<String>,
}

/// Request to delete a Slack message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMessageRequest {
    /// The Slack channel ID the message is in.
    pub channel_id: String,
    /// The message timestamp (ts) to delete.
    pub message_ts: String,
}

/// Request to fetch message history.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetHistoryRequest {
    /// The Slack channel ID to fetch history from.
    pub channel_id: String,
    /// Maximum number of messages to fetch (1-200).
    pub limit: Option<u8>,
    /// Optional team/workspace ID.
    pub server_id: Option<String>,
}

/// Request to add a reaction to a message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// The Slack channel ID the message is in.
    pub channel_id: String,
    /// The message timestamp (ts) to react to.
    pub message_ts: String,
    /// The emoji name to react with (without colons, e.g., "thumbsup").
    pub emoji: String,
}

// -- Tool implementations -----------------------------------------------------

#[tool_router(router = tool_router)]
impl SlackMcpServer {
    #[tool(
        description = "Send a message to a Slack channel. Returns the timestamp (ts) of the sent message."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "slack".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
        };
        let message = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conversation.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: req
                .thread_ts
                .map(brainwires_network::channels::ThreadId::new),
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };

        let msg_id = self
            .channel
            .send_message(&conversation, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok(format!("{{\"message_ts\": \"{}\"}}", msg_id))
    }

    #[tool(description = "Edit a previously sent Slack message.")]
    async fn edit_message(
        &self,
        Parameters(req): Parameters<EditMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "slack".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
        };
        let message = ChannelMessage {
            id: MessageId::new(&req.message_ts),
            conversation,
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };
        let id = MessageId::new(&req.message_ts);
        self.channel
            .edit_message(&id, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"edited\"}".to_string())
    }

    #[tool(description = "Delete a Slack message.")]
    async fn delete_message(
        &self,
        Parameters(req): Parameters<DeleteMessageRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.channel_id, req.message_ts);
        let id = MessageId::new(composite_id);
        self.channel
            .delete_message(&id)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"deleted\"}".to_string())
    }

    #[tool(
        description = "Fetch recent message history from a Slack channel. Returns up to `limit` messages (default 25, max 200)."
    )]
    async fn get_history(
        &self,
        Parameters(req): Parameters<GetHistoryRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "slack".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
        };
        let limit = req.limit.unwrap_or(25) as usize;
        let messages = self
            .channel
            .get_history(&conversation, limit)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&messages).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Add an emoji reaction to a Slack message.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.channel_id, req.message_ts);
        let id = MessageId::new(composite_id);
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"reacted\"}".to_string())
    }
}

// -- ServerHandler ------------------------------------------------------------

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SlackMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-slack", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Slack Channel — MCP Tool Server");
        info.instructions = Some(
            "Slack channel adapter MCP server. \
             Use send_message to send messages, edit_message to edit, \
             delete_message to delete, get_history for message history, \
             and add_reaction for emoji reactions."
                .into(),
        );
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_request_types_serialize() {
        let req = SendMessageRequest {
            channel_id: "C0123456789".to_string(),
            content: "Hello world".to_string(),
            server_id: Some("T0123456789".to_string()),
            thread_ts: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("C0123456789"));
        assert!(json.contains("Hello world"));
    }

    #[test]
    fn edit_request_serialize() {
        let req = EditMessageRequest {
            channel_id: "C01".to_string(),
            message_ts: "1234567890.123456".to_string(),
            content: "Updated".to_string(),
            server_id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: EditMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.message_ts, "1234567890.123456");
    }

    #[test]
    fn delete_request_serialize() {
        let req = DeleteMessageRequest {
            channel_id: "C01".to_string(),
            message_ts: "1234567890.123456".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: DeleteMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.message_ts, "1234567890.123456");
    }

    #[test]
    fn get_history_request_defaults() {
        let req = GetHistoryRequest {
            channel_id: "C01".to_string(),
            limit: None,
            server_id: None,
        };
        assert!(req.limit.is_none());
    }

    #[test]
    fn add_reaction_request_serialize() {
        let req = AddReactionRequest {
            channel_id: "C01".to_string(),
            message_ts: "1234567890.123456".to_string(),
            emoji: "thumbsup".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("C01"));
        assert!(json.contains("thumbsup"));
    }

    #[test]
    fn send_with_thread_ts() {
        let req = SendMessageRequest {
            channel_id: "C01".to_string(),
            content: "Thread reply".to_string(),
            server_id: None,
            thread_ts: Some("1234567890.000000".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("1234567890.000000"));
    }
}
