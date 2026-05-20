//! MCP stdio server for Feishu.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};

use crate::feishu::{FeishuChannel, SendMessageRequest};

/// Stdio MCP server wrapping a [`FeishuChannel`].
#[derive(Clone)]
pub struct FeishuMcpServer {
    channel: Arc<FeishuChannel>,
    tool_router: ToolRouter<Self>,
}

impl FeishuMcpServer {
    /// Construct from a shared channel handle.
    pub fn new(channel: Arc<FeishuChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    /// Run on stdio.
    pub async fn serve_stdio(channel: Arc<FeishuChannel>) -> Result<()> {
        tracing::info!("Starting Feishu MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl FeishuMcpServer {
    #[tool(
        description = "Send a text message via Feishu / Lark. Specify `receive_id_type` = open_id | chat_id | user_id | union_id | email."
    )]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let id = self
            .channel
            .post_text(&req.receive_id, &req.receive_id_type, &req.text)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format!("{{\"message_id\":\"{id}\"}}"))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for FeishuMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-feishu", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Feishu / Lark — MCP Tool Server");
        info.instructions = Some("Feishu / Lark channel adapter. Use `send_message`.".into());
        info
    }
}
