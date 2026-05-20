//! MCP server exposing Signal channel operations as tools.

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

use crate::signal::SignalChannel;

/// MCP server wrapping a `SignalChannel` to expose its operations as tools.
#[derive(Clone)]
pub struct SignalMcpServer {
    channel: Arc<SignalChannel>,
    tool_router: ToolRouter<Self>,
}

impl SignalMcpServer {
    pub fn new(channel: Arc<SignalChannel>) -> Self {
        Self {
            channel,
            tool_router: Self::tool_router(),
        }
    }

    pub async fn serve_stdio(channel: Arc<SignalChannel>) -> Result<()> {
        tracing::info!("Starting Signal MCP server on stdio");
        let server = Self::new(channel);
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;
        Ok(())
    }
}

// -- Tool request types -------------------------------------------------------

/// Request to send a Signal message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SendMessageRequest {
    /// Recipient: phone number ("+14155552671") or group ID ("group.abc123==").
    pub recipient: String,
    /// The text content to send.
    pub content: String,
}

/// Request to add a reaction to a Signal message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddReactionRequest {
    /// Composite message ID in format "recipient:author:timestamp".
    pub message_id: String,
    /// The emoji to react with (e.g. "👍").
    pub emoji: String,
}

// -- Tool implementations -----------------------------------------------------

#[tool_router(router = tool_router)]
impl SignalMcpServer {
    #[tool(description = "Send a Signal message to a phone number or group. \
                       Use '+E164' for direct messages or 'group.<base64id>' for groups.")]
    async fn send_message(
        &self,
        Parameters(req): Parameters<SendMessageRequest>,
    ) -> Result<String, String> {
        let conversation = ConversationId {
            platform: "signal".to_string(),
            channel_id: req.recipient,
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

        Ok(format!("{{\"message_id\": \"{}\"}}", msg_id.0))
    }

    #[tool(description = "Add an emoji reaction to a Signal message. \
                       The message_id must be in 'recipient:author:timestamp' format.")]
    async fn add_reaction(
        &self,
        Parameters(req): Parameters<AddReactionRequest>,
    ) -> Result<String, String> {
        let id = MessageId::new(&req.message_id);
        self.channel
            .add_reaction(&id, &req.emoji)
            .await
            .map_err(|e| format!("{:#}", e))?;

        Ok("{\"status\": \"reacted\"}".to_string())
    }
}

// -- ServerHandler ------------------------------------------------------------

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SignalMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-signal", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Signal Channel — MCP Tool Server");
        info.instructions = Some(
            "Signal channel adapter MCP server. \
             Use send_message to send messages to phone numbers or groups, \
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
    fn send_request_serializes() {
        let req = SendMessageRequest {
            recipient: "+14155552671".to_string(),
            content: "Hello from BrainClaw".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("+14155552671"));
        assert!(json.contains("Hello from BrainClaw"));
    }

    #[test]
    fn group_recipient_format() {
        let req = SendMessageRequest {
            recipient: "group.abc123==".to_string(),
            content: "Hello group".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("group.abc123=="));
    }
}
