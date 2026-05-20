//! Message routing logic between channels and agent sessions.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use uuid::Uuid;

use brainwires_network::channels::events::ChannelEvent;
use brainwires_network::channels::message::MessageContent;

use crate::channel_registry::ChannelRegistry;
use crate::session::SessionManager;

/// Trait for handling inbound channel events.
///
/// Implement this trait to provide custom inbound message handling (e.g.,
/// forwarding messages to an agent framework). The default implementation
/// ([`MessageRouter`]) logs events and manages sessions but does not
/// invoke any agent.
#[async_trait]
pub trait InboundHandler: Send + Sync {
    /// Handle an inbound event from the given channel.
    async fn handle_inbound(&self, channel_id: Uuid, event: &ChannelEvent) -> Result<()>;
}

/// Routes messages between channel adapters and agent sessions.
pub struct MessageRouter {
    /// Session manager reference.
    sessions: Arc<SessionManager>,
    /// Channel registry reference.
    channels: Arc<ChannelRegistry>,
}

impl MessageRouter {
    /// Create a new message router with references to the session manager and channel registry.
    pub fn new(sessions: Arc<SessionManager>, channels: Arc<ChannelRegistry>) -> Self {
        Self { sessions, channels }
    }

    /// Route an inbound event from a channel adapter to the appropriate agent session.
    ///
    /// 1. Extracts the user from the event
    /// 2. Gets or creates a session
    /// 3. Converts the message to agent-compatible format
    /// 4. Logs the message (agent forwarding is a future phase)
    pub async fn route_inbound(&self, channel_id: Uuid, event: &ChannelEvent) -> Result<()> {
        // Update heartbeat for the channel that sent this event
        self.channels.touch_heartbeat(&channel_id);

        match event {
            ChannelEvent::MessageReceived(msg) => {
                // Extract text content for logging
                let text = match &msg.content {
                    MessageContent::Text(t) => t.clone(),
                    MessageContent::RichText { fallback_plain, .. } => fallback_plain.clone(),
                    _ => "[non-text content]".to_string(),
                };

                // Resolve user session from conversation platform info
                let user = brainwires_network::channels::ChannelUser {
                    platform: msg.conversation.platform.clone(),
                    platform_user_id: msg.author.clone(),
                    display_name: msg.author.clone(),
                    username: None,
                    avatar_url: None,
                };
                let session = self.sessions.get_or_create_session(&user);

                tracing::info!(
                    channel_id = %channel_id,
                    session_id = %session.id,
                    platform = %msg.conversation.platform,
                    author = %msg.author,
                    "Inbound message: {}",
                    text
                );
            }
            ChannelEvent::MessageEdited(msg) => {
                tracing::info!(
                    channel_id = %channel_id,
                    message_id = %msg.id,
                    "Message edited in {}",
                    msg.conversation.platform
                );
            }
            ChannelEvent::MessageDeleted {
                message_id,
                conversation,
            } => {
                tracing::info!(
                    channel_id = %channel_id,
                    message_id = %message_id,
                    "Message deleted in {}",
                    conversation.platform
                );
            }
            ChannelEvent::ReactionAdded {
                message_id,
                user,
                emoji,
            } => {
                tracing::debug!(
                    channel_id = %channel_id,
                    message_id = %message_id,
                    user = %user.display_name,
                    emoji = %emoji,
                    "Reaction added"
                );
            }
            ChannelEvent::ReactionRemoved {
                message_id,
                user,
                emoji,
            } => {
                tracing::debug!(
                    channel_id = %channel_id,
                    message_id = %message_id,
                    user = %user.display_name,
                    emoji = %emoji,
                    "Reaction removed"
                );
            }
            ChannelEvent::TypingStarted { conversation, user } => {
                tracing::trace!(
                    channel_id = %channel_id,
                    platform = %conversation.platform,
                    user = %user.display_name,
                    "Typing started"
                );
            }
            ChannelEvent::PresenceChanged { user, status } => {
                tracing::trace!(
                    channel_id = %channel_id,
                    user = %user.display_name,
                    status = ?status,
                    "Presence changed"
                );
            }
            ChannelEvent::ThreadCreated {
                parent_message_id,
                thread_id,
            } => {
                tracing::info!(
                    channel_id = %channel_id,
                    parent_message_id = %parent_message_id,
                    thread_id = %thread_id,
                    "Thread created"
                );
            }
        }

        Ok(())
    }

    /// Route an outbound message from an agent session back to the correct channel.
    ///
    /// Finds the appropriate channel connection for the given platform and sends
    /// the message through its WebSocket sender.
    pub async fn route_outbound(
        &self,
        _session_id: Uuid,
        message: String,
        platform: &str,
    ) -> Result<()> {
        let channel_ids = self.channels.find_by_type(platform);

        if channel_ids.is_empty() {
            bail!(
                "No connected channel for platform '{}' to route outbound message",
                platform
            );
        }

        // Send to the first matching channel (future: smarter routing)
        for channel_id in &channel_ids {
            if let Some(tx) = self.channels.get_sender(channel_id) {
                if tx.send(message.clone()).await.is_ok() {
                    tracing::info!(
                        channel_id = %channel_id,
                        platform = %platform,
                        "Outbound message routed"
                    );
                    return Ok(());
                }
                tracing::warn!(
                    channel_id = %channel_id,
                    "Channel sender dropped, skipping"
                );
            }
        }

        bail!("All channels for platform '{}' have disconnected", platform);
    }
}

#[async_trait]
impl InboundHandler for MessageRouter {
    async fn handle_inbound(&self, channel_id: Uuid, event: &ChannelEvent) -> Result<()> {
        self.route_inbound(channel_id, event).await
    }
}
