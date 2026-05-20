//! MCP server exposing Google Chat operations as stdio tools.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};

use brainwires_network::channels::{
    Channel, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::google_chat::{GoogleChatChannel, SendMessageRequest};

/// Stdio MCP server wrapping a `GoogleChatChannel`.
#[derive(Clone)]
pub struct GoogleChatMcpServer {
    channel: Arc<GoogleChatChannel>,
    tool_router: ToolRouter<Self>,
}

impl GoogleChatMcpServer {
    /// Construct from a shared channel handle.
    pub fn new(channel: Arc<GoogleChatChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Run on stdio.
    pub async fn serve_stdio(channel: Arc<GoogleChatChannel>) -> Result<()> {
        tracing::info!("Starting Google Chat MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl GoogleChatMcpServer {
    #[tool(
        description = "Send a text message to a Google Chat space. Returns the message name (`spaces/.../messages/...`)."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "google_chat".to_string(),
            channel_id: req.space_id,
            server_id: None,
        };
        let thread_id = req
            .thread_name
            .map(brainwires_network::channels::ThreadId::new);
        let msg = ChannelMessage {
            id: MessageId::new("pending"),
            conversation: conversation.clone(),
            author: "bot".to_string(),
            content: MessageContent::Text(req.text),
            thread_id,
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
        Ok(format!("{{\"name\": \"{}\"}}", id))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for GoogleChatMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-google-chat", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Google Chat — MCP Tool Server");
        info.instructions = Some(
            "Google Chat channel adapter. Use `send_message` with the target space id.".into(),
        );
        info
    }
}
