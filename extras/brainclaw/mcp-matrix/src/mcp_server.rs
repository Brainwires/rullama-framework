//! MCP server exposing Matrix channel operations as tools.

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

use crate::matrix::MatrixChannel;

/// MCP server wrapping a `MatrixChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct MatrixMcpServer {
    channel: Arc<MatrixChannel>,
    tool_router: ToolRouter<Self>,
}

impl MatrixMcpServer {
    pub fn new(channel: Arc<MatrixChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    pub async fn serve_stdio(channel: Arc<MatrixChannel>) -> Result<()> {
        tracing::info!("Starting Matrix MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// ── Tool request types ─────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// Matrix room ID (e.g. "!roomId:server.org").
    pub room_id: String,
    /// Text content to send (Markdown supported).
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditMessageRequest {
    /// Matrix room ID the message is in.
    pub room_id: String,
    /// Event ID of the message to edit.
    pub event_id: String,
    /// New text content (Markdown supported).
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMessageRequest {
    /// Matrix room ID the message is in.
    pub room_id: String,
    /// Event ID of the message to delete.
    pub event_id: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// Matrix room ID the message is in.
    pub room_id: String,
    /// Event ID of the message to react to.
    pub event_id: String,
    /// Unicode emoji to react with.
    pub emoji: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendTypingRequest {
    /// Matrix room ID to show typing in.
    pub room_id: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetHistoryRequest {
    /// Matrix room ID to fetch history from.
    pub room_id: String,
    /// Maximum number of messages to return (default 25).
    pub limit: Option<usize>,
}

// ── Tool implementations ───────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl MatrixMcpServer {
    #[tool(
        description = "Send a text message to a Matrix room. Returns the event ID of the sent message."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "matrix".to_string(),
            channel_id: req.room_id.clone(),
            server_id: None,
        };
        let message = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conversation.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };

        let event_id = self
            .channel
            .send_message(&conversation, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok(format!("{{\"event_id\": \"{}\"}}", event_id))
    }

    #[tool(description = "Edit a previously sent Matrix message.")]
    async fn edit_message(
        &self,
        Parameters(req): Parameters<EditMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "matrix".to_string(),
            channel_id: req.room_id.clone(),
            server_id: None,
        };
        let message = ChannelMessage {
            id: MessageId::new(&req.event_id),
            conversation,
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };
        let id = MessageId::new(&req.event_id);
        self.channel
            .edit_message(&id, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"edited\"}".to_string())
    }

    #[tool(description = "Delete (redact) a Matrix message.")]
    async fn delete_message(
        &self,
        Parameters(req): Parameters<DeleteMessageRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.room_id, req.event_id);
        let id = MessageId::new(composite_id);
        self.channel
            .delete_message(&id)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"deleted\"}".to_string())
    }

    #[tool(description = "Add an emoji reaction to a Matrix message.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.room_id, req.event_id);
        let id = MessageId::new(composite_id);
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"reacted\"}".to_string())
    }

    #[tool(description = "Send a typing indicator to a Matrix room.")]
    async fn send_typing(
        &self,
        Parameters(req): Parameters<SendTypingRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "matrix".to_string(),
            channel_id: req.room_id,
            server_id: None,
        };
        self.channel
            .send_typing(&conversation)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"typing\"}".to_string())
    }

    #[tool(description = "Fetch recent message history from a Matrix room.")]
    async fn get_history(
        &self,
        Parameters(req): Parameters<GetHistoryRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "matrix".to_string(),
            channel_id: req.room_id,
            server_id: None,
        };
        let limit = req.limit.unwrap_or(25);
        let messages = self
            .channel
            .get_history(&conversation, limit)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&messages).map_err(|e| format!("Serialization failed: {}", e))
    }
}

// ── ServerHandler ────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MatrixMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-matrix", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Matrix Channel — MCP Tool Server");
        info.instructions = Some(
            "Matrix channel adapter MCP server. \
             Use send_message to send messages to rooms, edit_message to edit, \
             delete_message to redact, add_reaction for emoji reactions, \
             send_typing for typing indicators, and get_history for recent messages."
                .into(),
        );
        info
    }
}
