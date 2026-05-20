//! End-to-end tests against a mock Spreed server (axum).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

type FormBody = BTreeMap<String, String>;
type LastPost = Arc<Mutex<Option<(String, FormBody)>>>;

use axum::{
    Form, Json, Router,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, StatusCode},
    routing::get,
};
use brainwires_network::channels::{
    Channel, ChannelEvent, ChannelMessage, ConversationId, MessageContent, MessageId,
};
use brainwires_nextcloud_talk_channel::ingress::Ingress;
use brainwires_nextcloud_talk_channel::nextcloud_talk::{
    NextcloudTalkChannel, SpreedMessage, ocs_wrap,
};
use serde_json::json;
use tokio::sync::mpsc;

#[derive(Clone, Default)]
struct MockState {
    /// All messages the server claims to have, keyed by room.
    messages: Arc<Mutex<std::collections::BTreeMap<String, Vec<SpreedMessage>>>>,
    last_post: LastPost,
    header_ocs_seen: Arc<Mutex<bool>>,
    header_auth_seen: Arc<Mutex<bool>>,
}

async fn list_chat(
    AxumPath(room): AxumPath<String>,
    State(state): State<MockState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    *state.header_ocs_seen.lock().unwrap() =
        headers.get("OCS-APIRequest").and_then(|v| v.to_str().ok()) == Some("true");
    *state.header_auth_seen.lock().unwrap() = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.starts_with("Basic "))
        .unwrap_or(false);
    if !*state.header_ocs_seen.lock().unwrap() {
        return Err(StatusCode::NOT_ACCEPTABLE);
    }
    let msgs = state
        .messages
        .lock()
        .unwrap()
        .get(&room)
        .cloned()
        .unwrap_or_default();
    let v = serde_json::to_value(&msgs).unwrap();
    Ok(Json(ocs_wrap(v)))
}

async fn post_chat(
    AxumPath(room): AxumPath<String>,
    State(state): State<MockState>,
    headers: HeaderMap,
    Form(body): Form<FormBody>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if headers.get("OCS-APIRequest").and_then(|v| v.to_str().ok()) != Some("true") {
        return Err(StatusCode::NOT_ACCEPTABLE);
    }
    *state.last_post.lock().unwrap() = Some((room, body.clone()));
    // Return a freshly-allocated id.
    let echo = SpreedMessage {
        id: 9999,
        message_type: "comment".into(),
        actor_display_name: "bot".into(),
        actor_id: "bot".into(),
        message: body.get("message").cloned().unwrap_or_default(),
        timestamp: 1_700_000_000,
        parent: None,
    };
    Ok(Json(ocs_wrap(serde_json::to_value(echo).unwrap())))
}

async fn start_mock(state: MockState) -> String {
    let app = Router::new()
        .route(
            "/ocs/v2.php/apps/spreed/api/v1/chat/{room}",
            get(list_chat).post(post_chat),
        )
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}", addr)
}

#[tokio::test]
async fn ingress_forwards_once_and_tracks_cursor() {
    let state = MockState::default();
    let msgs = vec![
        SpreedMessage {
            id: 10,
            message_type: "system".into(),
            actor_display_name: "srv".into(),
            actor_id: "srv".into(),
            message: "alice joined".into(),
            timestamp: 1_700_000_000,
            parent: None,
        },
        SpreedMessage {
            id: 11,
            message_type: "comment".into(),
            actor_display_name: "Alice".into(),
            actor_id: "alice".into(),
            message: "hello".into(),
            timestamp: 1_700_000_001,
            parent: None,
        },
    ];
    state.messages.lock().unwrap().insert("room-1".into(), msgs);

    let base = start_mock(state.clone()).await;
    let channel = Arc::new(NextcloudTalkChannel::new(
        base.clone(),
        "alice",
        "app-password-placeholder",
    ));
    let td = tempfile::tempdir().unwrap();
    let cursor: PathBuf = td.path().join("nextcloud.json");
    let ingress = Ingress::new(
        Arc::clone(&channel),
        vec!["room-1".into()],
        Duration::from_millis(50),
        cursor.clone(),
    )
    .unwrap();

    let (tx, mut rx) = mpsc::channel(8);
    let n = ingress.tick(&tx).await.unwrap();
    assert_eq!(n, 1, "only the comment should be forwarded");
    let got = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();
    match got {
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.id.0, "11");
            assert_eq!(m.conversation.channel_id, "room-1");
            assert_eq!(m.author, "Alice");
        }
        other => panic!("unexpected: {other:?}"),
    }
    // Second tick must be a no-op even though the mock still returns
    // both rows.
    let n2 = ingress.tick(&tx).await.unwrap();
    assert_eq!(n2, 0);
    let snap = ingress.cursor_snapshot().await;
    assert_eq!(snap.last_message_id.get("room-1").copied(), Some(11));

    assert!(
        *state.header_ocs_seen.lock().unwrap(),
        "OCS-APIRequest missing"
    );
    assert!(
        *state.header_auth_seen.lock().unwrap(),
        "Authorization missing"
    );
}

#[tokio::test]
async fn send_message_posts_form_body() {
    let state = MockState::default();
    let base = start_mock(state.clone()).await;
    let channel = NextcloudTalkChannel::new(base, "alice", "placeholder");
    let msg = ChannelMessage {
        id: MessageId::new("pending"),
        conversation: ConversationId {
            platform: "nextcloud_talk".into(),
            channel_id: "room-2".into(),
            server_id: None,
        },
        author: "bot".into(),
        content: MessageContent::Text("hello there".into()),
        thread_id: None,
        reply_to: None,
        timestamp: chrono::Utc::now(),
        attachments: Vec::new(),
        metadata: Default::default(),
    };
    let id = channel.send_message(&msg.conversation, &msg).await.unwrap();
    assert_eq!(id.0, "9999");
    let (room, form) = state.last_post.lock().unwrap().clone().unwrap();
    assert_eq!(room, "room-2");
    assert_eq!(form.get("message").map(|s| s.as_str()), Some("hello there"));
}

#[tokio::test]
async fn send_message_without_ocs_header_would_fail() {
    // Sanity: the mock returns 406 when the OCS header is absent. We
    // exercise this by pointing a plain client at the endpoint without
    // the header, and confirming the server enforces it.
    let state = MockState::default();
    let base = start_mock(state).await;
    let resp = reqwest::Client::new()
        .get(format!("{base}/ocs/v2.php/apps/spreed/api/v1/chat/rx"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_ACCEPTABLE);
}

#[allow(dead_code)]
fn sanity_use_json() {
    // Keep `json!`/`reqwest` imports tied into the binary if tests refactor.
    let _ = json!({ "x": 1 });
}
