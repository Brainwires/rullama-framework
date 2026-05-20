//! MCP stdio server for LINE.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};

use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::line::{LineChannel, SendMessageRequest};

/// Stdio MCP server wrapping a [`LineChannel`].
#[derive(Clone)]
pub struct LineMcpServer {
    channel: Arc<LineChannel>,
    tool_router: ToolRouter<Self>,
}

impl LineMcpServer {
    /// Construct from a shared channel handle.
    pub fn new(channel: Arc<LineChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Run on stdio.
    pub async fn serve_stdio(channel: Arc<LineChannel>) -> Result<()> {
        tracing::info!("Starting LINE MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl LineMcpServer {
    #[tool(
        description = "Send a plain-text message to a LINE user. Uses reply API when a fresh reply token is cached, otherwise push."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "line".to_string(),
            channel_id: req.to,
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
        Ok(format!("{{\"id\":\"{id}\"}}"))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for LineMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-line", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires LINE — MCP Tool Server");
        info.instructions = Some(
            "LINE Messaging API channel adapter. Use `send_message` with the target user id."
                .into(),
        );
        info
    }
}
