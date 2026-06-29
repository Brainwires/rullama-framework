//! The `Channel` trait that all messaging channel adapters must implement.

use anyhow::Result;
use async_trait::async_trait;

use super::capabilities::ChannelCapabilities;
use super::identity::ConversationId;
use super::message::{ChannelMessage, MessageId};

/// The universal contract for a messaging channel adapter.
///
/// Every channel implementation (Discord, Telegram, Slack, etc.) must implement
/// this trait to integrate with the Brainwires gateway.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Unique channel type identifier (e.g., "discord", "telegram").
    fn channel_type(&self) -> &str;

    /// What this channel supports.
    fn capabilities(&self) -> ChannelCapabilities;

    /// Send a message to a conversation.
    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId>;

    /// Edit a previously sent message.
    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()>;

    /// Delete a message.
    async fn delete_message(&self, id: &MessageId) -> Result<()>;

    /// Send a typing indicator to a conversation.
    async fn send_typing(&self, target: &ConversationId) -> Result<()>;

    /// React to a message with an emoji.
    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()>;

    /// Get conversation history, returning up to `limit` messages.
    async fn get_history(
        &self,
        target: &ConversationId,
        limit: usize,
    ) -> Result<Vec<ChannelMessage>>;
}
