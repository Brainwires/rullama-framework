//! MCP server exposing Mattermost channel operations as tools.

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

use crate::mattermost::MattermostChannel;

/// MCP server wrapping a `MattermostChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct MattermostMcpServer {
    channel: Arc<MattermostChannel>,
    tool_router: ToolRouter<Self>,
}

impl MattermostMcpServer {
    /// Create a new MCP server wrapping the given Mattermost channel.
    pub fn new(channel: Arc<MattermostChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdin/stdout (MCP standard I/O transport).
    pub async fn serve_stdio(channel: Arc<MattermostChannel>) -> Result<()> {
        tracing::info!("Starting Mattermost MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// -- Tool request types -------------------------------------------------------

/// Request to send a message to a Mattermost channel.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// The Mattermost channel ID to send to.
    pub channel_id: String,
    /// The text content to send.
    pub content: String,
    /// Optional team/workspace ID.
    pub server_id: Option<String>,
    /// Optional post ID to reply in a thread.
    pub thread_id: Option<String>,
}

/// Request to edit a Mattermost post.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditMessageRequest {
    /// The post ID to edit.
    pub post_id: String,
    /// The new text content.
    pub content: String,
}

/// Request to delete a Mattermost post.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMessageRequest {
    /// The post ID to delete.
    pub post_id: String,
}

/// Request to fetch message history from a Mattermost channel.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetHistoryRequest {
    /// The Mattermost channel ID to fetch history from.
    pub channel_id: String,
    /// Maximum number of messages to fetch (1-200, default 25).
    pub limit: Option<u8>,
    /// Optional team/workspace ID.
    pub server_id: Option<String>,
}

/// Request to add a reaction to a Mattermost post.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// The post ID to react to.
    pub post_id: String,
    /// The emoji name to react with (without colons, e.g., "thumbsup").
    pub emoji: String,
}

// -- Tool implementations -----------------------------------------------------

#[tool_router(router = tool_router)]
impl MattermostMcpServer {
    #[tool(
        description = "Send a message to a Mattermost channel. Returns the post ID of the sent message."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "mattermost".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
        };
        let message = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conversation.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: req
                .thread_id
                .map(brainwires_network::channels::ThreadId::new),
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };

        let post_id = self
            .channel
            .send_message(&conversation, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok(format!("{{\"post_id\": \"{}\"}}", post_id.0))
    }

    #[tool(description = "Edit a previously sent Mattermost post.")]
    async fn edit_message(
        &self,
        Parameters(req): Parameters<EditMessageRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(&req.post_id);
        let placeholder_conversation = ConversationId {
            platform: "mattermost".to_string(),
            channel_id: String::new(),
            server_id: None,
        };
        let message = ChannelMessage {
            id: id.clone(),
            conversation: placeholder_conversation,
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };
        self.channel
            .edit_message(&id, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"edited\"}".to_string())
    }

    #[tool(description = "Delete a Mattermost post.")]
    async fn delete_message(
        &self,
        Parameters(req): Parameters<DeleteMessageRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(&req.post_id);
        self.channel
            .delete_message(&id)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"deleted\"}".to_string())
    }

    #[tool(
        description = "Fetch recent message history from a Mattermost channel. Returns up to `limit` messages (default 25, max 200)."
    )]
    async fn get_history(
        &self,
        Parameters(req): Parameters<GetHistoryRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "mattermost".to_string(),
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

    #[tool(description = "Add an emoji reaction to a Mattermost post.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(&req.post_id);
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"reacted\"}".to_string())
    }
}

// -- ServerHandler ------------------------------------------------------------

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MattermostMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-mattermost", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Mattermost Channel — MCP Tool Server");
        info.instructions = Some(
            "Mattermost channel adapter MCP server. \
             Use send_message to send posts, edit_message to edit, \
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
    fn send_request_serializes() {
        let req = SendMessageRequest {
            channel_id: "abc123".to_string(),
            content: "Hello world".to_string(),
            server_id: None,
            thread_id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("abc123"));
        assert!(json.contains("Hello world"));
    }

    #[test]
    fn send_with_thread_id() {
        let req = SendMessageRequest {
            channel_id: "C01".to_string(),
            content: "Thread reply".to_string(),
            server_id: None,
            thread_id: Some("root_post_id".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("root_post_id"));
    }

    #[test]
    fn get_history_defaults() {
        let req = GetHistoryRequest {
            channel_id: "C01".to_string(),
            limit: None,
            server_id: None,
        };
        assert!(req.limit.is_none());
    }
}
