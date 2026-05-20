//! MCP server exposing Discord channel operations as tools.

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

use crate::discord::DiscordChannel;

/// MCP server wrapping a `DiscordChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct DiscordMcpServer {
    channel: Arc<DiscordChannel>,
    tool_router: ToolRouter<Self>,
}

impl DiscordMcpServer {
    /// Create a new MCP server wrapping the given Discord channel.
    pub fn new(channel: Arc<DiscordChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdin/stdout (MCP standard I/O transport).
    pub async fn serve_stdio(channel: Arc<DiscordChannel>) -> Result<()> {
        tracing::info!("Starting Discord MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// ── Tool request types ─────────────────────────────────────────────────

/// Request to send a message to a Discord channel.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// The Discord channel ID to send to.
    pub channel_id: String,
    /// The text content to send.
    pub content: String,
    /// Optional guild/server ID.
    pub server_id: Option<String>,
}

/// Request to edit a Discord message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditMessageRequest {
    /// The Discord channel ID the message is in.
    pub channel_id: String,
    /// The message ID to edit.
    pub message_id: String,
    /// The new text content.
    pub content: String,
    /// Optional guild/server ID.
    pub server_id: Option<String>,
}

/// Request to delete a Discord message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMessageRequest {
    /// The Discord channel ID the message is in.
    pub channel_id: String,
    /// The message ID to delete.
    pub message_id: String,
}

/// Request to fetch message history.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetHistoryRequest {
    /// The Discord channel ID to fetch history from.
    pub channel_id: String,
    /// Maximum number of messages to fetch (1-100).
    pub limit: Option<u8>,
    /// Optional guild/server ID.
    pub server_id: Option<String>,
}

/// Request to list channels in a Discord guild.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListChannelsRequest {
    /// The Discord guild (server) ID to list channels for.
    pub guild_id: String,
}

/// Request to send a typing indicator.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendTypingRequest {
    /// The Discord channel ID to show typing in.
    pub channel_id: String,
    /// Optional guild/server ID.
    pub server_id: Option<String>,
}

/// Request to add a reaction to a message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// The Discord channel ID the message is in.
    pub channel_id: String,
    /// The message ID to react to.
    pub message_id: String,
    /// The emoji to react with (Unicode emoji or custom emoji string).
    pub emoji: String,
}

// ── Tool implementations ───────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl DiscordMcpServer {
    #[tool(
        description = "Send a message to a Discord channel. Returns the ID of the sent message."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "discord".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
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

    #[tool(description = "Edit a previously sent Discord message.")]
    async fn edit_message(
        &self,
        Parameters(req): Parameters<EditMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "discord".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
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

    #[tool(description = "Delete a Discord message.")]
    async fn delete_message(
        &self,
        Parameters(req): Parameters<DeleteMessageRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.channel_id, req.message_id);
        let id = MessageId::new(composite_id);
        self.channel
            .delete_message(&id)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"deleted\"}".to_string())
    }

    #[tool(
        description = "Fetch recent message history from a Discord channel. Returns up to `limit` messages (default 25, max 100)."
    )]
    async fn get_history(
        &self,
        Parameters(req): Parameters<GetHistoryRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "discord".to_string(),
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

    #[tool(
        description = "List text channels in a Discord guild (server). Requires the bot to be a member of the guild. Returns channel IDs, names, and types."
    )]
    async fn list_channels(
        &self,
        Parameters(req): Parameters<ListChannelsRequest>,
    ) -> Result<String, String> {
        use serenity::model::id::GuildId;

        let guild_id: u64 = req
            .guild_id
            .parse()
            .map_err(|_| format!("Invalid guild ID: {}", req.guild_id))?;
        let guild_id = GuildId::new(guild_id);

        let channels = self
            .channel
            .http()
            .get_channels(guild_id)
            .await
            .map_err(|e| format!("Discord API error: {e}"))?;

        let items: Vec<serde_json::Value> = channels
            .iter()
            .map(|ch| {
                serde_json::json!({
                    "id": ch.id.to_string(),
                    "name": ch.name,
                    "kind": format!("{:?}", ch.kind),
                    "position": ch.position,
                    "parent_id": ch.parent_id.map(|p| p.to_string()),
                })
            })
            .collect();

        serde_json::to_string(&serde_json::json!({ "channels": items }))
            .map_err(|e| format!("Serialization error: {e}"))
    }

    #[tool(
        description = "Send a typing indicator to a Discord channel. The indicator lasts ~10 seconds or until a message is sent."
    )]
    async fn send_typing(
        &self,
        Parameters(req): Parameters<SendTypingRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "discord".to_string(),
            channel_id: req.channel_id,
            server_id: req.server_id,
        };
        self.channel
            .send_typing(&conversation)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"typing\"}".to_string())
    }

    #[tool(description = "Add an emoji reaction to a Discord message.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.channel_id, req.message_id);
        let id = MessageId::new(composite_id);
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"reacted\"}".to_string())
    }
}

// ── ServerHandler ────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DiscordMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-discord", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Discord Channel — MCP Tool Server");
        info.instructions = Some(
            "Discord channel adapter MCP server. \
             Use send_message to send messages, edit_message to edit, \
             delete_message to delete, get_history for message history, \
             send_typing for typing indicators, and add_reaction for emoji reactions."
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
            channel_id: "123456".to_string(),
            content: "Hello world".to_string(),
            server_id: Some("789".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("123456"));
        assert!(json.contains("Hello world"));
    }

    #[test]
    fn edit_request_serialize() {
        let req = EditMessageRequest {
            channel_id: "111".to_string(),
            message_id: "222".to_string(),
            content: "Updated".to_string(),
            server_id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: EditMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.message_id, "222");
    }

    #[test]
    fn delete_request_serialize() {
        let req = DeleteMessageRequest {
            channel_id: "111".to_string(),
            message_id: "333".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: DeleteMessageRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.message_id, "333");
    }

    #[test]
    fn get_history_request_defaults() {
        let req = GetHistoryRequest {
            channel_id: "456".to_string(),
            limit: None,
            server_id: None,
        };
        assert!(req.limit.is_none());
    }

    #[test]
    fn add_reaction_request_serialize() {
        let req = AddReactionRequest {
            channel_id: "111".to_string(),
            message_id: "222".to_string(),
            emoji: "\u{1f44d}".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("111"));
        assert!(json.contains("\u{1f44d}"));
    }
}
