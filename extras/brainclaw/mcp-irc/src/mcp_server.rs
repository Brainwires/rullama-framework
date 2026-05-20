//! MCP stdio tool server for IRC.

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

use crate::irc_client::IrcChannel;

/// Request shape for `send_message`.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// IRC channel name (`#chat`) or user nick for a PM.
    pub target: String,
    /// Plain-text body. Auto-chunked at 400 bytes per line.
    pub text: String,
}

/// Stdio MCP server wrapping an `IrcChannel`.
#[derive(Clone)]
pub struct IrcMcpServer {
    channel: Arc<IrcChannel>,
    tool_router: ToolRouter<Self>,
}

impl IrcMcpServer {
    /// Construct.
    pub fn new(channel: Arc<IrcChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Serve on stdio.
    pub async fn serve_stdio(channel: Arc<IrcChannel>) -> Result<()> {
        tracing::info!("Starting IRC MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl IrcMcpServer {
    #[tool(description = "Send a PRIVMSG to a channel or user nick on the connected IRC network.")]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let (channel_id, _is_pm) = if req.target.starts_with('#') || req.target.starts_with('&') {
            (req.target.clone(), false)
        } else {
            (format!("pm:{}", req.target), true)
        };
        let conversation = ConversationId {
            platform: "irc".to_string(),
            channel_id,
            server_id: Some(self.channel.server().to_string()),
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
impl ServerHandler for IrcMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-irc", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires IRC — MCP Tool Server");
        info.instructions = Some("IRC channel adapter. Use `send_message`.".into());
        info
    }
}
