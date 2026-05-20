//! MCP stdio server exposing iMessage / BlueBubbles tools.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};

use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::imessage::{ImessageChannel, SendTextRequest};

/// Request type for the `react` MCP tool.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReactRequest {
    /// Target chat guid.
    pub chat_guid: String,
    /// Original message guid to react to.
    pub message_guid: String,
    /// Emoji or tapback string.
    pub emoji: String,
}

/// Stdio MCP server wrapping an [`ImessageChannel`].
#[derive(Clone)]
pub struct ImessageMcpServer {
    channel: Arc<ImessageChannel>,
    tool_router: ToolRouter<Self>,
}

impl ImessageMcpServer {
    /// Construct from a shared channel handle.
    pub fn new(channel: Arc<ImessageChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Run on stdio.
    pub async fn serve_stdio(channel: Arc<ImessageChannel>) -> Result<()> {
        tracing::info!("Starting iMessage MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl ImessageMcpServer {
    #[tool(description = "Send a plain-text iMessage to a chat GUID via BlueBubbles.")]
    async fn send_text(
        &self,
        Parameters(req): Parameters<SendTextRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "imessage".to_string(),
            channel_id: req.chat_guid,
            server_id: None,
        };
        let msg = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conversation.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.text),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: Vec::new(),
            metadata: std::collections::HashMap::new(),
        };
        let id = self
            .channel
            .send_message(&conversation, &msg)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format!("{{\"guid\":\"{id}\"}}"))
    }

    #[tool(description = "Add a tapback reaction (emoji) to a prior iMessage.")]
    async fn react(&self, Parameters(req): Parameters<ReactRequest>) -> Result<String, String> {
        self.channel
            .react(&req.chat_guid, &req.message_guid, &req.emoji)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok("{\"ok\":true}".into())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ImessageMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-imessage", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires iMessage (BlueBubbles) — MCP Tool Server");
        info.instructions =
            Some("iMessage adapter over BlueBubbles. Use `send_text` with a chat GUID.".into());
        info
    }
}
