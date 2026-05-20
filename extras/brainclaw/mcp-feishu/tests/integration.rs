//! Integration tests: webhook + outbound via mock Feishu open-platform.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{Json, Router, http::HeaderMap, routing::post};
use brainwires_feishu_channel::feishu::FeishuChannel;
use brainwires_feishu_channel::oauth::TenantTokenMinter;
use brainwires_feishu_channel::webhook::{WebhookState, serve, sign};
use brainwires_network::channels::{
    Channel, ChannelEvent, ChannelMessage, ConversationId, MessageContent, MessageId,
};
use serde_json::json;
use tokio::sync::mpsc;

async fn start_webhook(token: &str, tx: mpsc::Sender<ChannelEvent>) -> String {
    let state = WebhookState::new(token.to_string(), tx);
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
async fn url_verification_challenge_is_echoed() {
    let (tx, _rx) = mpsc::channel(4);
    let url = start_webhook("vt-test", tx).await;
    let body = json!({"type":"url_verification","challenge":"xyz"});
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let j: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(j["challenge"], "xyz");
}

#[tokio::test]
async fn webhook_forwards_signed_message_event() {
    let token = "shared-verification-token";
    let (tx, mut rx) = mpsc::channel(4);
    let url = start_webhook(token, tx).await;
    let content = serde_json::to_string(&json!({"text":"hi feishu"})).unwrap();
    let body_v = json!({
        "schema":"2.0",
        "header":{"event_type":"im.message.receive_v1"},
        "event":{
            "sender":{"sender_id":{"open_id":"ou_a"}},
            "message":{
                "chat_id":"oc_a",
                "message_id":"om_42",
                "message_type":"text",
                "create_time":"1700000000000",
                "content": content
            }
        }
    });
    let body = serde_json::to_vec(&body_v).unwrap();
    let ts = "1700000000";
    let nonce = "n-1";
    let sig = sign(token, ts, nonce, &body);
    let resp = reqwest::Client::new()
        .post(&url)
        .header("Content-Type", "application/json")
        .header("X-Lark-Request-Timestamp", ts)
        .header("X-Lark-Request-Nonce", nonce)
        .header("X-Lark-Signature", sig)
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
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.conversation.channel_id, "oc_a");
            match m.content {
                MessageContent::Text(t) => assert_eq!(t, "hi feishu"),
                _ => panic!(),
            }
        }
        _ => panic!(),
    }
}

#[tokio::test]
async fn webhook_rejects_missing_signature_headers() {
    let (tx, _rx) = mpsc::channel(4);
    let url = start_webhook("t", tx).await;
    // A valid JSON body that is NOT url_verification.
    let body = json!({
        "schema":"2.0",
        "header":{"event_type":"im.message.receive_v1"},
        "event":{"message":{}}
    });
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn webhook_rejects_wrong_signature() {
    let (tx, _rx) = mpsc::channel(4);
    let url = start_webhook("token-a", tx).await;
    let body = b"{}".to_vec();
    let sig = sign("token-wrong", "1", "n", &body);
    let resp = reqwest::Client::new()
        .post(&url)
        .header("X-Lark-Request-Timestamp", "1")
        .header("X-Lark-Request-Nonce", "n")
        .header("X-Lark-Signature", sig)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[derive(Clone, Default)]
struct MockFeishu {
    last_body: Arc<Mutex<Option<serde_json::Value>>>,
    last_query: Arc<Mutex<Option<String>>>,
}

async fn start_mock_feishu(state: MockFeishu) -> String {
    let app = Router::new()
        .route(
            "/open-apis/auth/v3/tenant_access_token/internal",
            post(
                |_: HeaderMap, Json(_body): Json<serde_json::Value>| async move {
                    Json(json!({
                        "code": 0,
                        "msg": "ok",
                        "tenant_access_token": "t-token-fake",
                        "expire": 7200,
                    }))
                },
            ),
        )
        .route(
            "/open-apis/im/v1/messages",
            post({
                let s = state.clone();
                move |uri: axum::http::Uri,
                      _headers: HeaderMap,
                      Json(body): Json<serde_json::Value>| {
                    let s = s.clone();
                    async move {
                        *s.last_body.lock().unwrap() = Some(body);
                        *s.last_query.lock().unwrap() = uri.query().map(|q| q.to_string());
                        Json(json!({
                            "code": 0,
                            "msg": "ok",
                            "data": {"message_id": "om_new_99"}
                        }))
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
async fn outbound_posts_to_messages_endpoint() {
    let mock = MockFeishu::default();
    let base = start_mock_feishu(mock.clone()).await;
    let minter = Arc::new(
        TenantTokenMinter::new("cli_a", "app-secret-placeholder").with_base_url(base.clone()),
    );
    let channel = FeishuChannel::new(minter).with_base_url(base);
    let msg = ChannelMessage {
        id: MessageId::new("pending"),
        conversation: ConversationId {
            platform: "feishu".into(),
            channel_id: "oc_xyz".into(),
            server_id: None,
        },
        author: "bot".into(),
        content: MessageContent::Text("hello".into()),
        thread_id: None,
        reply_to: None,
        timestamp: chrono::Utc::now(),
        attachments: Vec::new(),
        metadata: Default::default(),
    };
    let id = channel.send_message(&msg.conversation, &msg).await.unwrap();
    assert_eq!(id.0, "om_new_99");
    let body = mock.last_body.lock().unwrap().clone().unwrap();
    assert_eq!(body["receive_id"], "oc_xyz");
    assert_eq!(body["msg_type"], "text");
    assert!(body["content"].as_str().unwrap().contains("hello"));
    let q = mock.last_query.lock().unwrap().clone().unwrap();
    assert!(q.contains("receive_id_type=chat_id"));
}
