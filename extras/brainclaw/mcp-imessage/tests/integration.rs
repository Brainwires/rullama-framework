//! Integration tests: mock BlueBubbles server (axum) + ingress loop.
//!
//! Exercises:
//! 1. End-to-end: a message surfaced by the mock server is forwarded as
//!    a `ChannelEvent::MessageReceived`.
//! 2. Cursor discipline: a second tick for the same batch forwards nothing.
//! 3. Outbound: `Channel::send_message` hits `/api/v1/message/text` with
//!    the expected JSON body.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
};
use brainwires_imessage_channel::imessage::{BbChat, BbHandle, BbMessage, ImessageChannel};
use brainwires_imessage_channel::ingress::Ingress;
use brainwires_network::channels::{
    Channel, ChannelEvent, ChannelMessage, ConversationId, MessageContent, MessageId,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

#[derive(Clone, Default)]
struct MockState {
    messages: Arc<Mutex<Vec<BbMessage>>>,
    last_sent: Arc<Mutex<Option<serde_json::Value>>>,
    last_password: Arc<Mutex<Option<String>>>,
}

#[derive(Debug, Deserialize)]
struct AuthQ {
    password: String,
}

async fn list_messages(
    State(state): State<MockState>,
    Query(q): Query<AuthQ>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    *state.last_password.lock().unwrap() = Some(q.password);
    let msgs = state.messages.lock().unwrap().clone();
    Ok(Json(json!({ "data": msgs })))
}

async fn send_text(
    State(state): State<MockState>,
    Query(q): Query<AuthQ>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    *state.last_password.lock().unwrap() = Some(q.password);
    *state.last_sent.lock().unwrap() = Some(body);
    Json(json!({ "status": 200, "data": { "guid": "sent-guid-1" } }))
}

async fn start_mock(state: MockState) -> String {
    let app = Router::new()
        .route("/api/v1/message", get(list_messages))
        .route("/api/v1/message/text", post(send_text))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}", addr)
}

fn sample_inbound(guid: &str, chat: &str, handle: &str, text: &str) -> BbMessage {
    BbMessage {
        guid: guid.into(),
        text: Some(text.into()),
        date_created_ms: Some(1_700_000_000_000),
        chats: vec![BbChat { guid: chat.into() }],
        handle: Some(BbHandle {
            address: Some(handle.into()),
        }),
        is_from_me: false,
    }
}

#[tokio::test]
async fn ingress_forwards_new_messages_and_tracks_cursor() {
    let state = MockState::default();
    state.messages.lock().unwrap().push(sample_inbound(
        "msg-1",
        "chat-A",
        "+15551112222",
        "hi there",
    ));
    let base = start_mock(state.clone()).await;

    let channel = Arc::new(ImessageChannel::new(base.clone(), "fake-password"));
    let td = tempfile::tempdir().unwrap();
    let cursor: PathBuf = td.path().join("imessage.json");
    let ingress = Ingress::new(
        Arc::clone(&channel),
        vec![],
        Duration::from_millis(50),
        cursor.clone(),
    )
    .unwrap();

    let (tx, mut rx) = mpsc::channel(8);
    let n = ingress.tick(&tx).await.unwrap();
    assert_eq!(n, 1);
    let got = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();
    match got {
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.conversation.channel_id, "chat-A");
            assert_eq!(m.author, "+15551112222");
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Second tick with the same mock state must forward nothing.
    let n2 = ingress.tick(&tx).await.unwrap();
    assert_eq!(n2, 0);
    assert!(
        rx.try_recv().is_err(),
        "no new message should have been forwarded"
    );

    // Cursor was persisted.
    let snap = ingress.cursor_snapshot().await;
    assert_eq!(
        snap.last_guid.get("chat-A").map(|s| s.as_str()),
        Some("msg-1")
    );

    // Password was forwarded on the request.
    let pw = state.last_password.lock().unwrap().clone();
    assert_eq!(pw.as_deref(), Some("fake-password"));
}

#[tokio::test]
async fn send_message_hits_text_endpoint() {
    let state = MockState::default();
    let base = start_mock(state.clone()).await;
    let channel = ImessageChannel::new(base, "pw-demo");
    let msg = ChannelMessage {
        id: MessageId::new("pending"),
        conversation: ConversationId {
            platform: "imessage".into(),
            channel_id: "chat-A".into(),
            server_id: None,
        },
        author: "bot".into(),
        content: MessageContent::Text("hello world".into()),
        thread_id: None,
        reply_to: None,
        timestamp: chrono::Utc::now(),
        attachments: Vec::new(),
        metadata: Default::default(),
    };
    let id = channel.send_message(&msg.conversation, &msg).await.unwrap();
    assert!(!id.0.is_empty());
    let sent = state.last_sent.lock().unwrap().clone().unwrap();
    assert_eq!(sent["chatGuid"], "chat-A");
    assert_eq!(sent["message"], "hello world");
}
