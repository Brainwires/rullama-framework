//! MCP server exposing Telegram channel operations as tools.

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

use crate::telegram::TelegramChannel;

/// MCP server wrapping a `TelegramChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct TelegramMcpServer {
    channel: Arc<TelegramChannel>,
    tool_router: ToolRouter<Self>,
}

impl TelegramMcpServer {
    /// Create a new MCP server wrapping the given Telegram channel.
    pub fn new(channel: Arc<TelegramChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdin/stdout (MCP standard I/O transport).
    pub async fn serve_stdio(channel: Arc<TelegramChannel>) -> Result<()> {
        tracing::info!("Starting Telegram MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// -- Tool request types --

/// Request to send a message to a Telegram chat.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// The Telegram chat ID to send to.
    pub chat_id: String,
    /// The text content to send.
    pub content: String,
}

/// Request to edit a Telegram message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditMessageRequest {
    /// The Telegram chat ID the message is in.
    pub chat_id: String,
    /// The message ID to edit.
    pub message_id: String,
    /// The new text content.
    pub content: String,
}

/// Request to delete a Telegram message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMessageRequest {
    /// The Telegram chat ID the message is in.
    pub chat_id: String,
    /// The message ID to delete.
    pub message_id: String,
}

/// Request to send a typing indicator.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendTypingRequest {
    /// The Telegram chat ID to show typing in.
    pub chat_id: String,
}

/// Request to add a reaction to a message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// The Telegram chat ID the message is in.
    pub chat_id: String,
    /// The message ID to react to.
    pub message_id: String,
    /// The emoji to react with (Unicode emoji).
    pub emoji: String,
}

// -- Tool implementations --

#[tool_router(router = tool_router)]
impl TelegramMcpServer {
    #[tool(description = "Send a message to a Telegram chat. Returns the ID of the sent message.")]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "telegram".to_string(),
            channel_id: req.chat_id,
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

        let msg_id = self
            .channel
            .send_message(&conversation, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok(format!("{{\"message_id\": \"{}\"}}", msg_id))
    }

    #[tool(description = "Edit a previously sent Telegram message.")]
    async fn edit_message(
        &self,
        Parameters(req): Parameters<EditMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "telegram".to_string(),
            channel_id: req.chat_id,
            server_id: None,
        };
        let message = ChannelMessage {
            id: MessageId::new(&req.message_id),
            conversation,
            author: "bot".to_string(),
            content: MessageContent::Text(req.content),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };
        let id = MessageId::new(&req.message_id);
        self.channel
            .edit_message(&id, &message)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"edited\"}".to_string())
    }

    #[tool(description = "Delete a Telegram message.")]
    async fn delete_message(
        &self,
        Parameters(req): Parameters<DeleteMessageRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.chat_id, req.message_id);
        let id = MessageId::new(composite_id);
        self.channel
            .delete_message(&id)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"deleted\"}".to_string())
    }

    #[tool(
        description = "Send a typing indicator to a Telegram chat. The indicator lasts ~5 seconds or until a message is sent."
    )]
    async fn send_typing(
        &self,
        Parameters(req): Parameters<SendTypingRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "telegram".to_string(),
            channel_id: req.chat_id,
            server_id: None,
        };
        self.channel
            .send_typing(&conversation)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"typing\"}".to_string())
    }

    #[tool(description = "Add an emoji reaction to a Telegram message.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.chat_id, req.message_id);
        let id = MessageId::new(composite_id);
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"reacted\"}".to_string())
    }
}

// -- ServerHandler --

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TelegramMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-telegram", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Telegram Channel — MCP Tool Server");
        info.instructions = Some(
            "Telegram channel adapter MCP server. \
             Use send_message to send messages, edit_message to edit, \
             delete_message to delete, send_typing for typing indicators, \
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
            chat_id: "-1001234567890".to_string(),
            content: "Hello world".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("-1001234567890"));
        assert!(json.contains("Hello world"));
    }

    #[test]
    fn edit_request_serialize() {
        let req = EditMessageRequest {
            chat_id: "111".to_string(),
            message_id: "222".to_string(),
            content: "Updated".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: EditMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.message_id, "222");
    }

    #[test]
    fn delete_request_serialize() {
        let req = DeleteMessageRequest {
            chat_id: "111".to_string(),
            message_id: "333".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: DeleteMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.message_id, "333");
    }

    #[test]
    fn add_reaction_request_serialize() {
        let req = AddReactionRequest {
            chat_id: "111".to_string(),
            message_id: "222".to_string(),
            emoji: "\u{1f44d}".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("111"));
        assert!(json.contains("\u{1f44d}"));
    }
}
