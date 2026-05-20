//! Matrix SDK event handler setup.
//!
//! Registers event handlers on the `matrix_sdk::Client` to forward incoming
//! room messages to the gateway via an mpsc channel.

use std::collections::HashMap;
use std::sync::Arc;

use matrix_sdk::{
    Client, Room,
    event_handler::Ctx,
    ruma::events::room::message::{MessageType, OriginalSyncRoomMessageEvent},
};
use tokio::sync::mpsc;

use brainwires_network::channels::{
    ChannelEvent, ChannelMessage, ConversationId, MessageContent, MessageId,
};

/// Register all Matrix event handlers on the client.
///
/// After calling this, start `client.sync()` in a background task.
pub fn register_handlers(client: &Client, event_tx: mpsc::Sender<ChannelEvent>) {
    client.add_event_handler_context(Arc::new(event_tx));
    client.add_event_handler(handle_room_message);
}

/// Handler for `m.room.message` events.
async fn handle_room_message(
    ev: OriginalSyncRoomMessageEvent,
    room: Room,
    Ctx(event_tx): Ctx<Arc<mpsc::Sender<ChannelEvent>>>,
) {
    // Ignore messages sent by our own bot user
    if Some(ev.sender.clone()) == room.client().user_id().map(|u| u.to_owned()) {
        return;
    }

    let text = match &ev.content.msgtype {
        MessageType::Text(t) => t.body.clone(),
        MessageType::Notice(n) => n.body.clone(),
        MessageType::Emote(e) => format!("* {}", e.body),
        other => format!("[{}]", other.msgtype()),
    };

    let msg = ChannelMessage {
        id: MessageId::new(ev.event_id.to_string()),
        conversation: ConversationId {
            platform: "matrix".to_string(),
            channel_id: room.room_id().to_string(),
            server_id: Some(room.client().homeserver().to_string()),
        },
        author: ev.sender.to_string(),
        content: MessageContent::Text(text),
        thread_id: None,
        reply_to: None,
        timestamp: ev
            .origin_server_ts
            .to_system_time()
            .map(chrono::DateTime::from)
            .unwrap_or_else(chrono::Utc::now),
        attachments: vec![],
        metadata: HashMap::new(),
    };

    let event = ChannelEvent::MessageReceived(msg);
    if let Err(e) = event_tx.try_send(event) {
        tracing::warn!(error = %e, "Failed to forward Matrix event to channel");
    }
}
