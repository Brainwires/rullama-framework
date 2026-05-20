//! MCP stdio tool server for Teams.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};

use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::teams::{SendMessageRequest, TeamsChannel};

/// MCP server wrapping a `TeamsChannel`.
#[derive(Clone)]
pub struct TeamsMcpServer {
    channel: Arc<TeamsChannel>,
    tool_router: ToolRouter<Self>,
}

impl TeamsMcpServer {
    /// Construct.
    pub fn new(channel: Arc<TeamsChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdio.
    pub async fn serve_stdio(channel: Arc<TeamsChannel>) -> Result<()> {
        tracing::info!("Starting Teams MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl TeamsMcpServer {
    #[tool(
        description = "Send a markdown message to a Teams conversation. Requires a prior inbound activity to have recorded the conversation's serviceUrl."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "teams".to_string(),
            channel_id: req.conversation_id,
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
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };
        let id = self
            .channel
            .send_message(&conversation, &msg)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format!("{{\"id\": \"{}\"}}", id))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TeamsMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-teams", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Microsoft Teams — MCP Tool Server");
        info.instructions = Some("Teams channel adapter. Use `send_message`.".into());
        info
    }
}
