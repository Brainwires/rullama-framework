//! Integration tests: webhook + outbound via mock LINE server.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    Json, Router,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use brainwires_line_channel::line::{LineChannel, ReplyTokenStore};
use brainwires_line_channel::webhook::{WebhookState, serve, sign_body};
use brainwires_network::channels::{
    Channel, ChannelEvent, ChannelMessage, ConversationId, MessageContent, MessageId,
};
use serde_json::json;
use tokio::sync::mpsc;

async fn start_webhook(
    secret: &str,
    tx: mpsc::Sender<ChannelEvent>,
    tokens: Arc<ReplyTokenStore>,
) -> String {
    let state = WebhookState::new(secret.to_string(), tx, tokens);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let listen = addr.to_string();
    let listen_clone = listen.clone();
    tokio::spawn(async move {
        let _ = serve(state, &listen_clone).await;
    });
    for _ in 0..20 {
        if tokio::net::TcpStream::connect(&listen).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    format!("http://{}/webhook", listen)
}

#[tokio::test]
async fn webhook_forwards_text_event_when_signature_is_valid() {
    let secret = "super-secret-for-tests";
    let (tx, mut rx) = mpsc::channel(4);
    let tokens = Arc::new(ReplyTokenStore::default());
    let url = start_webhook(secret, tx, Arc::clone(&tokens)).await;
    let body = json!({
        "events": [{
            "type": "message",
            "replyToken": "rt-demo",
            "source": {"userId": "Uabc"},
            "message": {"type":"text","id":"m1","text":"hi line"}
        }]
    })
    .to_string();
    let sig = sign_body(secret, body.as_bytes());
    let resp = reqwest::Client::new()
        .post(&url)
        .header("X-Line-Signature", sig)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv())
        .await
        .unwrap()
        .unwrap();
    match ev {
        ChannelEvent::MessageReceived(m) => assert_eq!(m.conversation.channel_id, "Uabc"),
        _ => panic!(),
    }
    assert!(tokens.take_fresh("Uabc").is_some());
}

#[tokio::test]
async fn webhook_rejects_missing_signature() {
    let (tx, _rx) = mpsc::channel(4);
    let tokens = Arc::new(ReplyTokenStore::default());
    let url = start_webhook("s", tx, tokens).await;
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&json!({"events":[]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn webhook_rejects_wrong_signature() {
    let (tx, _rx) = mpsc::channel(4);
    let tokens = Arc::new(ReplyTokenStore::default());
    let url = start_webhook("s1", tx, tokens).await;
    let body = b"{\"events\":[]}".to_vec();
    let sig = sign_body("s2-wrong", &body);
    let resp = reqwest::Client::new()
        .post(&url)
        .header("X-Line-Signature", sig)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[derive(Clone, Default)]
struct MockLine {
    last: Arc<Mutex<Option<(String, serde_json::Value)>>>,
}

async fn start_mock_line(state: MockLine) -> String {
    let app = Router::new()
        .route(
            "/message/reply",
            post({
                let s = state.clone();
                move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                    let s = s.clone();
                    async move {
                        assert!(headers.contains_key("authorization"));
                        *s.last.lock().unwrap() = Some(("reply".into(), body));
                        StatusCode::OK
                    }
                }
            }),
        )
        .route(
            "/message/push",
            post({
                let s = state.clone();
                move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                    let s = s.clone();
                    async move {
                        assert!(headers.contains_key("authorization"));
                        *s.last.lock().unwrap() = Some(("push".into(), body));
                        StatusCode::OK
                    }
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}", addr)
}

#[tokio::test]
async fn outbound_uses_push_when_no_reply_token() {
    let mock = MockLine::default();
    let base = start_mock_line(mock.clone()).await;
    let chan = LineChannel::new("test-access-token-placeholder").with_api_base(base);
    let msg = ChannelMessage {
        id: MessageId::new("pending"),
        conversation: ConversationId {
            platform: "line".into(),
            channel_id: "Uzzz".into(),
            server_id: None,
        },
        author: "bot".into(),
        content: MessageContent::Text("out-1".into()),
        thread_id: None,
        reply_to: None,
        timestamp: chrono::Utc::now(),
        attachments: Vec::new(),
        metadata: Default::default(),
    };
    chan.send_message(&msg.conversation, &msg).await.unwrap();
    let (endpoint, body) = mock.last.lock().unwrap().clone().unwrap();
    assert_eq!(endpoint, "push");
    assert_eq!(body["to"], "Uzzz");
    assert_eq!(body["messages"][0]["text"], "out-1");
}

#[tokio::test]
async fn outbound_uses_reply_when_fresh_token_cached() {
    let mock = MockLine::default();
    let base = start_mock_line(mock.clone()).await;
    let chan = LineChannel::new("fake-access-token").with_api_base(base);
    chan.reply_tokens().remember("Uyyy", "reply-tok-1".into());
    let msg = ChannelMessage {
        id: MessageId::new("p"),
        conversation: ConversationId {
            platform: "line".into(),
            channel_id: "Uyyy".into(),
            server_id: None,
        },
        author: "bot".into(),
        content: MessageContent::Text("out-2".into()),
        thread_id: None,
        reply_to: None,
        timestamp: chrono::Utc::now(),
        attachments: Vec::new(),
        metadata: Default::default(),
    };
    chan.send_message(&msg.conversation, &msg).await.unwrap();
    let (endpoint, body) = mock.last.lock().unwrap().clone().unwrap();
    assert_eq!(endpoint, "reply");
    assert_eq!(body["replyToken"], "reply-tok-1");
}
