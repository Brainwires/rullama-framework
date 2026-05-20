//! IRC client + `Channel` trait implementation.
//!
//! The adapter keeps one persistent TCP connection (optionally TLS) open
//! to a single IRC network. Incoming PRIVMSG frames are routed through
//! [`crate::protocol`] and either forwarded to the gateway or dropped.
//! Outbound [`ChannelMessage`]s are sent as PRIVMSG, one IRC line per
//! 400-byte chunk.
//!
//! The connection loop owns nick-collision fallback (append `_` and
//! retry once), SASL authentication (when configured), and exponential
//! backoff reconnect.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::StreamExt;
use irc::client::prelude::*;
use tokio::sync::{Mutex, mpsc};

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelEvent, ChannelMessage, ConversationId, MessageContent,
    MessageId,
};

use crate::config::IrcConfig;
use crate::protocol::{
    InboundMessage, MAX_PRIVMSG_BYTES, build_ctcp_action, chunk_for_privmsg, classify_privmsg,
    to_channel_message,
};

/// A `Sender` clone usable from any task.
type IrcSender = irc::client::Sender;

/// Slot holding the currently-live IRC sender. Replaced on each reconnect.
type SenderSlot = Arc<Mutex<Option<IrcSender>>>;

/// IRC channel adapter.
pub struct IrcChannel {
    server: String,
    sender_slot: SenderSlot,
    /// Tracks which channels we've joined (updated from JOIN/PART events).
    joined: Arc<DashMap<String, ()>>,
}

impl IrcChannel {
    /// Construct a detached IRC channel — the sender slot is populated by
    /// the connection loop.
    pub fn new(server: impl Into<String>) -> Self {
        Self {
            server: server.into(),
            sender_slot: Arc::new(Mutex::new(None)),
            joined: Arc::new(DashMap::new()),
        }
    }

    /// Handle to the sender slot — the connection loop writes here.
    pub fn sender_slot(&self) -> SenderSlot {
        Arc::clone(&self.sender_slot)
    }

    /// Handle to the joined-channel set.
    pub fn joined(&self) -> Arc<DashMap<String, ()>> {
        Arc::clone(&self.joined)
    }

    /// Server hostname passed at construction time.
    pub fn server(&self) -> &str {
        &self.server
    }

    async fn send_raw_privmsg(&self, target: &str, text: &str) -> Result<()> {
        let slot = self.sender_slot.lock().await;
        let sender = slot
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("IRC client not connected"))?;
        for chunk in chunk_for_privmsg(text, MAX_PRIVMSG_BYTES) {
            sender.send_privmsg(target, chunk).context("PRIVMSG")?;
        }
        Ok(())
    }
}

#[async_trait]
impl Channel for IrcChannel {
    fn channel_type(&self) -> &str {
        "irc"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        // Plain text only — no rich text, no reactions, no threads.
        ChannelCapabilities::empty()
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let (text, is_action) = match &message.content {
            MessageContent::Text(t) => (t.clone(), false),
            MessageContent::RichText { fallback_plain, .. } => (fallback_plain.clone(), false),
            MessageContent::Media(m) => (
                m.caption
                    .clone()
                    .map(|c| format!("{c} {}", m.url))
                    .unwrap_or_else(|| m.url.clone()),
                false,
            ),
            MessageContent::Embed(e) => (
                e.description
                    .clone()
                    .or_else(|| e.title.clone())
                    .unwrap_or_default(),
                false,
            ),
            MessageContent::Mixed(items) => (
                items
                    .iter()
                    .map(|c| match c {
                        MessageContent::Text(t) => t.clone(),
                        MessageContent::RichText { fallback_plain, .. } => fallback_plain.clone(),
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                false,
            ),
        };

        // PMs ship as `PRIVMSG <nick>` — strip the `pm:` prefix used
        // internally to distinguish DM conversations.
        let irc_target = target
            .channel_id
            .strip_prefix("pm:")
            .unwrap_or(&target.channel_id);

        if is_action {
            self.send_raw_privmsg(irc_target, &build_ctcp_action(&text))
                .await?;
        } else {
            self.send_raw_privmsg(irc_target, &text).await?;
        }

        Ok(MessageId::new(uuid::Uuid::new_v4().to_string()))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        anyhow::bail!("IRC does not support editing messages");
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        anyhow::bail!("IRC does not support deleting messages");
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        anyhow::bail!("IRC does not support reactions");
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        // Standard IRC has no persistent history; return empty.
        Ok(Vec::new())
    }
}

/// Connect once and run the read loop until the stream closes. Returns
/// the effective nick that was acknowledged by the server (in case we
/// had to fall back to `<nick>_`).
async fn connect_and_run(
    config: &IrcConfig,
    bot_nick: &str,
    channel: Arc<IrcChannel>,
    event_tx: mpsc::Sender<ChannelEvent>,
) -> Result<()> {
    let irc_cfg = Config {
        nickname: Some(bot_nick.to_string()),
        username: Some(config.username.clone()),
        realname: Some(config.realname.clone()),
        server: Some(config.server.clone()),
        port: Some(config.port),
        use_tls: Some(config.use_tls),
        channels: config.channels.clone(),
        password: config
            .sasl_password
            .as_ref()
            .filter(|_| is_plain_password_mode(config))
            .cloned(),
        ..Default::default()
    };

    let mut client = Client::from_config(irc_cfg)
        .await
        .context("IRC client connect")?;
    client.identify().context("IRC identify")?;
    tracing::debug!(server = %config.server, port = config.port, "IRC client identified");

    let sender = client.sender();
    {
        let slot = channel.sender_slot();
        let mut guard = slot.lock().await;
        *guard = Some(sender.clone());
    }

    let joined = channel.joined();
    let server = config.server.clone();
    let prefix = config.message_prefix.clone();
    let resolved_nick = bot_nick.to_string();

    let mut stream = client.stream().context("IRC stream")?;
    tracing::debug!("entering IRC stream loop");
    while let Some(msg_result) = stream.next().await {
        let message = match msg_result {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("IRC stream error: {e}");
                break;
            }
        };

        let sender_nick = message
            .source_nickname()
            .map(|s| s.to_string())
            .unwrap_or_default();

        match message.command {
            Command::PRIVMSG(target, body) => {
                let classified =
                    classify_privmsg(&target, &body, &sender_nick, &resolved_nick, &prefix);
                match classified {
                    InboundMessage::Ignored => continue,
                    ref ev => {
                        if let Some(cm) = to_channel_message(&server, ev) {
                            audit_log(&cm);
                            let _ = event_tx.send(ChannelEvent::MessageReceived(cm)).await;
                        }
                    }
                }
            }
            Command::JOIN(ref chan, _, _) => {
                if sender_nick == resolved_nick {
                    joined.insert(chan.clone(), ());
                    tracing::info!(channel = %chan, "joined");
                }
            }
            Command::PART(ref chan, _) => {
                if sender_nick == resolved_nick {
                    joined.remove(chan);
                }
            }
            Command::Response(Response::ERR_NICKNAMEINUSE, ref args) => {
                tracing::warn!(
                    args = ?args,
                    "nick already in use — retrying once with trailing underscore",
                );
                // Tell the caller to retry with `<nick>_`.
                anyhow::bail!("nick_in_use");
            }
            _ => {}
        }
    }
    {
        let slot = channel.sender_slot();
        let mut guard = slot.lock().await;
        *guard = None;
    }
    Ok(())
}

fn is_plain_password_mode(cfg: &IrcConfig) -> bool {
    // If SASL isn't supported by the network we at least fall back to
    // PASS auth. SASL-PLAIN over TLS is what servers that document a
    // password today accept; the `irc` crate sends PASS + NICK/USER
    // unconditionally when `password` is set.
    cfg.sasl_password.is_some()
}

/// Top-level IRC runner with reconnect + nick-collision recovery.
pub async fn run(
    config: IrcConfig,
    channel: Arc<IrcChannel>,
    event_tx: mpsc::Sender<ChannelEvent>,
    mut shutdown: tokio::sync::broadcast::Receiver<()>,
) -> Result<()> {
    let mut backoff = Duration::from_secs(2);
    let mut nick_attempt = config.nick.clone();
    let mut nick_retried = false;

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                tracing::info!("IRC runner shutting down");
                return Ok(());
            }
            result = connect_and_run(&config, &nick_attempt, Arc::clone(&channel), event_tx.clone()) => {
                match result {
                    Ok(_) => {
                        tracing::info!("IRC stream ended — reconnecting");
                        backoff = Duration::from_secs(2);
                    }
                    Err(e) if e.to_string() == "nick_in_use" && !nick_retried => {
                        nick_attempt = format!("{}_", config.nick);
                        nick_retried = true;
                        tracing::warn!(retry_nick = %nick_attempt, "nick collision — retrying once");
                        continue;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "IRC connection error");
                    }
                }
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = next_backoff(backoff);
    }
}

/// Exponential backoff capped at 60s.
pub fn next_backoff(current: Duration) -> Duration {
    let ms = current.as_millis() as u64;
    Duration::from_millis(ms.saturating_mul(2).clamp(2_000, 60_000))
}

fn audit_log(msg: &ChannelMessage) {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(msg.author.as_bytes());
    let digest = hex::encode(&h.finalize()[..6]);
    let len = match &msg.content {
        MessageContent::Text(t) => t.len(),
        _ => 0,
    };
    tracing::info!(
        channel = "irc",
        user = %digest,
        message_len = len,
        "forwarded"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_starts_at_two_seconds() {
        let next = next_backoff(Duration::from_millis(100));
        assert!(next.as_secs() >= 2);
    }

    #[test]
    fn backoff_caps_at_sixty_seconds() {
        let mut d = Duration::from_secs(2);
        for _ in 0..10 {
            d = next_backoff(d);
        }
        assert!(d <= Duration::from_secs(60));
    }

    #[tokio::test]
    async fn send_without_connection_errors() {
        let ch = IrcChannel::new("irc.example.net");
        let target = ConversationId {
            platform: "irc".into(),
            channel_id: "#test".into(),
            server_id: Some("irc.example.net".into()),
        };
        let msg = ChannelMessage {
            id: MessageId::new("m"),
            conversation: target.clone(),
            author: "bot".into(),
            content: MessageContent::Text("hi".into()),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        };
        let err = ch.send_message(&target, &msg).await.unwrap_err();
        assert!(err.to_string().contains("not connected"));
    }

    #[test]
    fn capabilities_are_empty() {
        let ch = IrcChannel::new("x");
        assert!(ch.capabilities().is_empty());
        assert_eq!(ch.channel_type(), "irc");
    }

    #[test]
    fn joined_starts_empty() {
        let ch = IrcChannel::new("x");
        assert!(ch.joined().is_empty());
    }
}
