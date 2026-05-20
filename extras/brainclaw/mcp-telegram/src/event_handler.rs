//! Teloxide dispatcher that converts Telegram updates to `ChannelEvent`.

use std::sync::Arc;

use teloxide::prelude::*;
use teloxide::respond;
use teloxide::types::{ChatKind, Message, MessageEntityKind};
use tokio::sync::mpsc;

use brainwires_network::channels::ChannelEvent;

use crate::config::TelegramConfig;
use crate::telegram::telegram_message_to_channel_message;

/// Returns true if the message should be forwarded to the gateway.
///
/// In private chats, always forward.  In group/supergroup chats, forward only
/// when `group_mention_required` is false OR the message contains a mention of
/// the configured bot username, a `mention` entity, or matches a keyword pattern.
fn should_forward(msg: &Message, config: &TelegramConfig) -> bool {
    // Private chats always respond
    if matches!(msg.chat.kind, ChatKind::Private(_)) {
        return true;
    }
    // Group without filtering
    if !config.group_mention_required {
        return true;
    }

    // Check @mention entities
    if let Some(entities) = msg.entities() {
        for entity in entities {
            match &entity.kind {
                MessageEntityKind::Mention => {
                    // Telegram bots in groups are @mentioned via a Mention entity.
                    // If we have a configured username, verify it matches.
                    if let Some(ref bot_uname) = config.bot_username {
                        if let Some(text) = msg.text() {
                            let start = entity.offset;
                            let end = start + entity.length;
                            let mention_text = text.get(start..end).unwrap_or("");
                            // Mention text starts with @; strip it for comparison
                            let name = mention_text.trim_start_matches('@');
                            if name.eq_ignore_ascii_case(bot_uname) {
                                return true;
                            }
                        }
                    } else {
                        // No configured username — any @mention triggers a response
                        return true;
                    }
                }
                MessageEntityKind::TextMention { user } if !user.is_bot => {}
                _ => {}
            }
        }
    }

    // Check keyword patterns
    if let Some(text) = msg.text() {
        let lower = text.to_lowercase();
        for pattern in &config.mention_patterns {
            if lower.contains(pattern.to_lowercase().as_str()) {
                return true;
            }
        }
    }

    false
}

/// Starts the teloxide update dispatcher, forwarding events over the provided sender.
///
/// This function blocks until the bot is shut down.
pub async fn run_dispatcher(
    bot: Bot,
    event_tx: mpsc::Sender<ChannelEvent>,
    config: TelegramConfig,
) {
    let event_tx = Arc::new(event_tx);
    let config = Arc::new(config);

    let message_handler = {
        let tx = Arc::clone(&event_tx);
        let cfg = Arc::clone(&config);
        Update::filter_message().endpoint(move |msg: Message| {
            let tx = Arc::clone(&tx);
            let cfg = Arc::clone(&cfg);
            async move {
                // Skip bot messages to avoid loops
                if let Some(ref from) = msg.from
                    && from.is_bot
                {
                    return respond(());
                }

                // Group mention filter
                if !should_forward(&msg, &cfg) {
                    tracing::debug!(
                        chat_id = msg.chat.id.0,
                        "Skipping group message — bot not mentioned"
                    );
                    return respond(());
                }

                let channel_message = telegram_message_to_channel_message(&msg);
                let event = ChannelEvent::MessageReceived(channel_message);

                if let Err(e) = tx.send(event).await {
                    tracing::error!("Failed to forward message event: {}", e);
                }

                respond(())
            }
        })
    };

    let edited_message_handler = {
        let tx = Arc::clone(&event_tx);
        Update::filter_edited_message().endpoint(move |msg: Message| {
            let tx = Arc::clone(&tx);
            async move {
                let channel_message = telegram_message_to_channel_message(&msg);
                let event = ChannelEvent::MessageEdited(channel_message);

                if let Err(e) = tx.send(event).await {
                    tracing::error!("Failed to forward edited message event: {}", e);
                }

                respond(())
            }
        })
    };

    let handler = dptree::entry()
        .branch(message_handler)
        .branch(edited_message_handler);

    Dispatcher::builder(bot, handler)
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}
