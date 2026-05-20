//! MCP server exposing WhatsApp channel operations as tools.

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

use crate::whatsapp::WhatsAppChannel;

/// MCP server wrapping a `WhatsAppChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct WhatsAppMcpServer {
    channel: Arc<WhatsAppChannel>,
    tool_router: ToolRouter<Self>,
}

impl WhatsAppMcpServer {
    pub fn new(channel: Arc<WhatsAppChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    pub async fn serve_stdio(channel: Arc<WhatsAppChannel>) -> Result<()> {
        tracing::info!("Starting WhatsApp MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// ── Tool request types ─────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// Recipient phone number in E.164 format without '+' (e.g. "15551234567").
    pub phone_number: String,
    /// Text content to send.
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// Recipient phone number (E.164 without '+').
    pub phone_number: String,
    /// Message ID to react to.
    pub message_id: String,
    /// Unicode emoji character.
    pub emoji: String,
}

// ── Tool implementations ───────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl WhatsAppMcpServer {
    #[tool(
        description = "Send a text message to a WhatsApp phone number. Returns the sent message ID."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "whatsapp".to_string(),
            channel_id: req.phone_number.clone(),
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

    #[tool(description = "Add an emoji reaction to a WhatsApp message.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let composite_id = format!("{}:{}", req.phone_number, req.message_id);
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
impl ServerHandler for WhatsAppMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-whatsapp", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires WhatsApp Channel — MCP Tool Server");
        info.instructions = Some(
            "WhatsApp Business channel adapter MCP server. \
             Use send_message to send a text message to a phone number, \
             and add_reaction to react to a message with an emoji."
                .into(),
        );
        info
    }
}
