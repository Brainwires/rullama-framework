//! MCP stdio server for Nextcloud Talk.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};

use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::nextcloud_talk::{NextcloudTalkChannel, SendMessageRequest};

/// Stdio MCP server wrapping a [`NextcloudTalkChannel`].
#[derive(Clone)]
pub struct NextcloudTalkMcpServer {
    channel: Arc<NextcloudTalkChannel>,
    tool_router: ToolRouter<Self>,
}

impl NextcloudTalkMcpServer {
    /// Construct from a shared channel handle.
    pub fn new(channel: Arc<NextcloudTalkChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Run on stdio.
    pub async fn serve_stdio(channel: Arc<NextcloudTalkChannel>) -> Result<()> {
        tracing::info!("Starting Nextcloud Talk MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl NextcloudTalkMcpServer {
    #[tool(
        description = "Send a chat message to a Nextcloud Talk room. Returns the numeric message id."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "nextcloud_talk".to_string(),
            channel_id: req.room_token,
            server_id: None,
        };
        let msg = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conversation.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.message),
            thread_id: None,
            reply_to: req.reply_to.map(|n| MessageId::new(n.to_string())),
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
impl ServerHandler for NextcloudTalkMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info =
            Implementation::new("brainwires-nextcloud-talk", env!("CARGO_PKG_VERSION"))
                .with_title("Brainwires Nextcloud Talk — MCP Tool Server");
        info.instructions =
            Some("Nextcloud Talk channel adapter. Use `send_message` with a room token.".into());
        info
    }
}
