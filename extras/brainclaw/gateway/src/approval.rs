//! Interactive tool approval via chat.
//!
//! When `require_tool_approval` is enabled in the BrainClaw config, tool calls
//! matching the configured list are intercepted before execution. An approval
//! request is sent to the user's current channel ("⚠️ Run `bash`? Reply **yes**
//! or **no**"), and the agent pauses until the user responds or the timeout
//! (default 60 s) elapses.
//!
//! # Architecture
//!
//! - [`ApprovalRegistry`] holds a `DashMap<(platform, user_id), Sender<bool>>`
//!   of pending approval requests.
//! - [`ChatApprovalHook`] implements [`ToolPreHook`] and is attached to each
//!   per-user [`ChatAgent`].  Before each agent turn, `handle_message()` updates
//!   two shared `RwLock` cells: the current conversation target and the channel
//!   mpsc sender, so the hook knows where to send the approval prompt.
//! - `AgentInboundHandler::handle_message()` checks `ApprovalRegistry` at the
//!   start: if there is a pending sender for this user, it resolves it with the
//!   yes/no answer instead of routing the message to the agent.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::{ToolContext, ToolUse};
use brainwires_network::channels::{ChannelMessage, ConversationId, MessageContent, MessageId};
use brainwires_tools::{PreHookDecision, ToolPreHook};
use chrono::Utc;
use dashmap::DashMap;
use tokio::sync::{RwLock, mpsc, oneshot};
use uuid::Uuid;

/// Approval timeout — if the user does not respond within this window, the
/// tool call is rejected automatically.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Shared registry of pending tool-approval requests.
///
/// Key: `(platform, user_id)` — at most one pending approval per user at
/// any given time.
#[derive(Default)]
pub struct ApprovalRegistry {
    pending: DashMap<(String, String), oneshot::Sender<bool>>,
}

impl ApprovalRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pending approval for `(platform, user_id)`.
    ///
    /// Returns the receiver end of the oneshot channel.  The hook awaits this
    /// to learn the user's reply.
    pub fn register(&self, platform: String, user_id: String) -> oneshot::Receiver<bool> {
        let (tx, rx) = oneshot::channel();
        self.pending.insert((platform, user_id), tx);
        rx
    }

    /// Attempt to resolve a pending approval for `(platform, user_id)`.
    ///
    /// Called by `handle_message()` when it detects a yes/no response.
    /// Returns `true` if there was a pending approval, `false` if none.
    pub fn resolve(&self, platform: &str, user_id: &str, approved: bool) -> bool {
        if let Some((_, tx)) = self
            .pending
            .remove(&(platform.to_string(), user_id.to_string()))
        {
            let _ = tx.send(approved);
            true
        } else {
            false
        }
    }

    /// Returns true if there is a pending approval waiting for this user.
    pub fn is_pending(&self, platform: &str, user_id: &str) -> bool {
        self.pending
            .contains_key(&(platform.to_string(), user_id.to_string()))
    }
}

/// Per-user `ToolPreHook` that sends an approval request to the user's channel
/// and waits for their reply before allowing a tool to execute.
pub struct ChatApprovalHook {
    /// The platform / user identifying this session.
    platform: String,
    user_id: String,
    /// Tools that require approval.  Empty set = all tools require approval.
    tools_requiring_approval: HashSet<String>,
    /// The user's most-recent conversation target.  Updated by
    /// `handle_message()` before each `agent.process_message()` call.
    current_conversation: Arc<RwLock<Option<ConversationId>>>,
    /// The mpsc sender for the current channel.  Updated by
    /// `handle_message()` before each `agent.process_message()` call.
    current_sender: Arc<RwLock<Option<mpsc::Sender<String>>>>,
    /// The UUID of the current channel (for logging).
    current_channel_id: Arc<RwLock<Option<Uuid>>>,
    /// Shared approval registry for registering / resolving requests.
    registry: Arc<ApprovalRegistry>,
    /// How long to wait for a response before auto-rejecting.
    timeout: Duration,
}

impl ChatApprovalHook {
    pub fn new(
        platform: String,
        user_id: String,
        tools_requiring_approval: HashSet<String>,
        current_conversation: Arc<RwLock<Option<ConversationId>>>,
        current_sender: Arc<RwLock<Option<mpsc::Sender<String>>>>,
        current_channel_id: Arc<RwLock<Option<Uuid>>>,
        registry: Arc<ApprovalRegistry>,
    ) -> Self {
        Self {
            platform,
            user_id,
            tools_requiring_approval,
            current_conversation,
            current_sender,
            current_channel_id,
            registry,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

#[async_trait]
impl ToolPreHook for ChatApprovalHook {
    async fn before_execute(
        &self,
        tool_use: &ToolUse,
        _context: &ToolContext,
    ) -> Result<PreHookDecision> {
        // Check if this tool requires approval
        if !self.tools_requiring_approval.is_empty()
            && !self.tools_requiring_approval.contains(&tool_use.name)
        {
            return Ok(PreHookDecision::Allow);
        }

        // Read the current conversation / sender
        let conversation = {
            let guard = self.current_conversation.read().await;
            match guard.clone() {
                Some(c) => c,
                None => {
                    tracing::warn!(
                        tool = %tool_use.name,
                        "Approval requested but no conversation target — allowing"
                    );
                    return Ok(PreHookDecision::Allow);
                }
            }
        };

        let sender = {
            let guard = self.current_sender.read().await;
            match guard.clone() {
                Some(s) => s,
                None => {
                    tracing::warn!(
                        tool = %tool_use.name,
                        "Approval requested but no channel sender — allowing"
                    );
                    return Ok(PreHookDecision::Allow);
                }
            }
        };

        let channel_id = self.current_channel_id.read().await.unwrap_or(Uuid::nil());

        // Format the approval prompt
        let input_preview = {
            let s = serde_json::to_string(&tool_use.input).unwrap_or_default();
            if s.len() > 200 {
                format!("{}…", &s[..200])
            } else {
                s
            }
        };

        let prompt_text = if input_preview == "{}" || input_preview.is_empty() {
            format!(
                "⚠️ Agent wants to run tool `{}`. Reply **yes** to allow or **no** to cancel.",
                tool_use.name
            )
        } else {
            format!(
                "⚠️ Agent wants to run tool `{}` with: `{}`. Reply **yes** to allow or **no** to cancel.",
                tool_use.name, input_preview
            )
        };

        let approval_msg = ChannelMessage {
            id: MessageId::new(Uuid::new_v4().to_string()),
            conversation: conversation.clone(),
            author: "assistant".to_string(),
            content: MessageContent::Text(prompt_text),
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };

        let event = brainwires_network::channels::ChannelEvent::MessageReceived(approval_msg);
        let json = serde_json::to_string(&event)?;

        // Register the pending approval before sending (avoid race)
        let rx = self
            .registry
            .register(self.platform.clone(), self.user_id.clone());

        if let Err(e) = sender.send(json).await {
            tracing::error!(channel_id = %channel_id, error = %e, "Failed to send approval request");
            self.registry.resolve(&self.platform, &self.user_id, false);
            return Ok(PreHookDecision::Reject(
                "Could not send approval request".to_string(),
            ));
        }

        tracing::info!(
            channel_id = %channel_id,
            tool = %tool_use.name,
            "Approval request sent — waiting for user response"
        );

        // Wait for user response with timeout
        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(true)) => {
                tracing::info!(tool = %tool_use.name, "Tool call approved by user");
                Ok(PreHookDecision::Allow)
            }
            Ok(Ok(false)) => {
                tracing::info!(tool = %tool_use.name, "Tool call rejected by user");
                Ok(PreHookDecision::Reject(
                    "Tool call rejected by user".to_string(),
                ))
            }
            Ok(Err(_)) => Ok(PreHookDecision::Reject(
                "Approval channel closed".to_string(),
            )),
            Err(_) => {
                // Timeout — clean up stale entry
                self.registry.resolve(&self.platform, &self.user_id, false);
                tracing::warn!(tool = %tool_use.name, "Approval timed out (60 s)");
                Ok(PreHookDecision::Reject(
                    "Approval timed out (60 s) — tool call cancelled".to_string(),
                ))
            }
        }
    }
}
