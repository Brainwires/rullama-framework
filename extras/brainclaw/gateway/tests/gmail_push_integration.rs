//! End-to-end integration test for Gmail push (OpenClaw parity P3.1).
//!
//! This test boots a real gateway on an ephemeral port with the
//! `email-push` feature and wires in:
//!
//! - A mock Google Gmail REST API server that serves canned `history`
//!   and `messages/{id}` responses.
//! - A mock Google OpenID Connect JWKs endpoint that serves a single
//!   test RSA public key.
//! - A deterministic `InboundHandler` that records any inbound
//!   `ChannelEvent::MessageReceived` so we can assert dispatch.
//!
//! Then it POSTs a hand-signed Pub/Sub envelope to
//! `/webhooks/gmail-push` and checks:
//!
//! 1. The response is 204 No Content.
//! 2. The mock handler received the expected inbound message.
//! 3. A second POST with the same Pub/Sub messageId is de-duped (no
//!    second inbound dispatch).

#![cfg(feature = "email-push")]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use axum::extract::{Path, State};
use axum::routing::get;
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use tokio::net::TcpListener;
use uuid::Uuid;

use brainwires_gateway::channel_registry::ChannelRegistry;
use brainwires_gateway::config::GatewayConfig;
use brainwires_gateway::gmail_push::{GmailCursorStore, GmailPushRegistry};
use brainwires_gateway::router::InboundHandler;
use brainwires_gateway::server::Gateway;
use brainwires_gateway::session::SessionManager;
use brainwires_network::channels::events::ChannelEvent;
use brainwires_tools::gmail_push::{GmailPushConfig, GmailPushHandler};

// ── Test fixtures ────────────────────────────────────────────────────────

const TEST_PRIVATE_KEY_PEM: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCNtkupX+Bhiopj
yZ0cYPrhkmum2Y7BYCnnsW8bxZQdW5EPwYp4YhSh/tPszGmUNvY/fkz0g1Eg7zj4
kw4TMEgJzcckom9uK0FPjdS79ivKrMwYyTH3UpeLNWrfAionwYDO90hmTIXIGjv6
cm83weeJI0WNffOJKhkIEbpPByVLWcym1y75JTEmM1m2ntUF66ZuZS3NSIObXlOS
oH/48PG1nXIT+3Fr97x4t894JjifBzhXj9nwA1J2zd4bLbtqS7QH/iplzxNakEaG
2KFiDtbMSIXFSQ31LssxiiW8OSASQveZRHyjvs/MiLbVXP5cwOjsNcMc3KmqH4R6
DpyJ6Yr3AgMBAAECggEAAaBdfEX+7RpUyJy9l7VJ5oEyiMQyHhMMor2dFlcCrQGU
3SGYsBVZbPCKncr3d+gs6av5QHcqB7Q8gFHZFVb2WgfJaNNlGavCa0zSlmOsEKlC
CxNyu4bFpCc1S6Dv5evpHrb6LD5ll1csaA1HLH8mY/hf1HtlFPvC33OrTTvk7YPy
dM3jVRyU9lzkoEqr1CsL+WEu2+yV2nRV2i4esvqEsO0ZJTXQvHIXcj01d1yB4Nd/
Dk4mIleKeSEYHlN9LJTSZL84hbANdB74KVeGSqJLwKkaGK/zVbXIHlKBvS1pmfLj
llLI+JMxpElJ7twp6ue2jQjRG7lxp6SqdVt4MvBSjQKBgQDBYHU4C+UgG/Fai4mj
K7yZdzmHbIypp7B/a4fKxOC24PRGa9TMz0dYepIlW6ZNrxKxS0mSUsRxEX4aDWJe
HqXzD8u+AJ29cQwJica8+hYKVFHYP35oInjuZJvOpHyqTf8mUM6tJPi40FDwNzoW
SDEKShxCIPGWW69dCRpMnMMQdQKBgQC7mqwdReJGYWFBh8ihtkMyQkcUjt7BmnFs
9vJKErEZ1ex6kd8TBgeAEpJzLXgto/yA/Tq6fEkY3yJNL3PWb6tSRo1RmHJZ4W5c
FNTBAzLPbuFgDR8akLMVqQnGNghR6CzU5dvfQXRjg5ZgEKGXrXK6W7+3XouXci3d
Wbi7VonAOwKBgFBuiXL9Z5j6ZmIN5frLh0+hynjsinlKeVwWYs3RI9KNMK1VzpY9
pORFXyJQw5ROPI0nznshF/obl4LIjGCviMDXkhv+b53LNoGFH/ecYax8M+qpRi+U
Hw6xJClIO14uwPCz7bMQzK86Xl/76Jo5/sPT3XsX7sRmcENXNOwmy++9AoGAEzpW
G6X2/Bms+ydsk853cqZCXMQL5rHqoC1rRdZGmoxHcYST5YI/sIu2wOFPKPZeweWy
aDymzUrJXDnZ2IeXepZKk6tZRQcK5Zso9yNZyNLnfI27u2BLSpQJsWwGTEbMmYF5
mJc/05dACVaLCV24nYsbyjKBgiMsujwg5+qFsdMCgYBe8vr0sicDdWf2CT1Q9MJw
RCLt1SG/Cs2fYuyP3msyzU1OgP+H5F7zwGlWcq8h9koRKxAN6R+4wmT6amEbQ3DP
MPQpqzow7HOkGnyPodjy4bl1P/SC3TUIZjEIMzCf0FBMPXGMHnypvpaC4udEnUIw
fUXevnTFL54FiMU+ykGryA==
-----END PRIVATE KEY-----
"#;

/// Matching JWK modulus (n) for the private key above.
const TEST_N: &str = "jbZLqV_gYYqKY8mdHGD64ZJrptmOwWAp57FvG8WUHVuRD8GKeGIUof7T7MxplDb2P35M9INRIO84-JMOEzBICc3HJKJvbitBT43Uu_YryqzMGMkx91KXizVq3wIqJ8GAzvdIZkyFyBo7-nJvN8HniSNFjX3ziSoZCBG6TwclS1nMptcu-SUxJjNZtp7VBeumbmUtzUiDm15TkqB_-PDxtZ1yE_txa_e8eLfPeCY4nwc4V4_Z8ANSds3eGy27aku0B_4qZc8TWpBGhtihYg7WzEiFxUkN9S7LMYolvDkgEkL3mUR8o77PzIi21Vz-XMDo7DXDHNypqh-Eeg6ciemK9w";

/// Matching JWK exponent (e) — 65537.
const TEST_E: &str = "AQAB";

const TEST_KID: &str = "gmail-push-test-kid";
const TEST_MAILBOX: &str = "alice@example.com";
const TEST_AUDIENCE: &str = "https://gateway.example.com/webhooks/gmail-push";

// ── Recording handler ────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct RecorderState {
    events: Arc<Mutex<Vec<ChannelEvent>>>,
}

struct RecorderHandler(RecorderState);

#[async_trait]
impl InboundHandler for RecorderHandler {
    async fn handle_inbound(&self, _channel_id: Uuid, event: &ChannelEvent) -> Result<()> {
        self.0.events.lock().unwrap().push(event.clone());
        Ok(())
    }
}

// ── Mock Google mock servers ─────────────────────────────────────────────

#[derive(Clone)]
struct MockGmail {
    history_json: Arc<serde_json::Value>,
    messages: Arc<HashMap<String, serde_json::Value>>,
}

async fn mock_history(
    State(state): State<MockGmail>,
    Path(_email): Path<String>,
) -> axum::Json<serde_json::Value> {
    axum::Json((*state.history_json).clone())
}

async fn mock_message(
    State(state): State<MockGmail>,
    Path((_email, id)): Path<(String, String)>,
) -> axum::response::Response {
    match state.messages.get(&id) {
        Some(v) => axum::Json(v.clone()).into_response(),
        None => (axum::http::StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

use axum::response::IntoResponse;

async fn start_mock_gmail() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    // Canned history: one messageAdded with id "msg-1".
    let history_json = serde_json::json!({
        "history": [
            {
                "id": "101",
                "messagesAdded": [
                    { "message": { "id": "msg-1", "threadId": "thread-1" } }
                ]
            }
        ],
        "historyId": "101"
    });

    // Canned messages/msg-1: from/to/subject/text body.
    let body_b64 = URL_SAFE_NO_PAD.encode(b"hello from integration test");
    let msg1 = serde_json::json!({
        "id": "msg-1",
        "threadId": "thread-1",
        "labelIds": ["INBOX"],
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                { "name": "From", "value": "carol@example.com" },
                { "name": "To",   "value": TEST_MAILBOX },
                { "name": "Subject", "value": "integration hello" }
            ],
            "body": { "data": body_b64 }
        }
    });

    let mut messages = HashMap::new();
    messages.insert("msg-1".to_string(), msg1);

    let state = MockGmail {
        history_json: Arc::new(history_json),
        messages: Arc::new(messages),
    };

    let app = axum::Router::new()
        .route("/gmail/v1/users/{email}/history", get(mock_history))
        .route("/gmail/v1/users/{email}/messages/{id}", get(mock_message))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let jh = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (addr, jh)
}

async fn start_mock_jwks() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let jwks = serde_json::json!({
        "keys": [
            {
                "kid": TEST_KID,
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "n": TEST_N,
                "e": TEST_E,
            }
        ]
    });
    let app = axum::Router::new().route(
        "/certs",
        get(move || {
            let jwks = jwks.clone();
            async move { axum::Json(jwks) }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let jh = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (addr, jh)
}

/// Build a Pub/Sub HTTP push body for our test mailbox.
fn make_pubsub_body(message_id: &str) -> Vec<u8> {
    let inner = serde_json::json!({
        "emailAddress": TEST_MAILBOX,
        "historyId": 100u64,
    });
    let data_b64 = STANDARD.encode(serde_json::to_vec(&inner).unwrap());
    let outer = serde_json::json!({
        "message": {
            "data": data_b64,
            "messageId": message_id,
            "publishTime": "2025-01-01T00:00:00Z",
        },
        "subscription": "projects/test/subscriptions/gmail-push-test"
    });
    serde_json::to_vec(&outer).unwrap()
}

/// Sign a Google-like push JWT with our test keypair.
fn sign_push_jwt(audience: &str) -> String {
    let claims = serde_json::json!({
        "iss": "https://accounts.google.com",
        "aud": audience,
        "sub": "pubsub@system.gserviceaccount.com",
        "email": "pubsub@system.gserviceaccount.com",
        "exp": (chrono::Utc::now().timestamp() + 3600) as usize,
    });
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(TEST_KID.to_string());
    let key = EncodingKey::from_rsa_pem(TEST_PRIVATE_KEY_PEM.as_bytes()).unwrap();
    encode(&header, &claims, &key).unwrap()
}

// ── Main test body ───────────────────────────────────────────────────────

#[tokio::test]
async fn gmail_push_webhook_dispatches_and_dedupes() {
    // 1. Stand up mock Google servers.
    let (gmail_addr, _gmail_jh) = start_mock_gmail().await;
    let (jwks_addr, _jwks_jh) = start_mock_jwks().await;

    // 2. Build a GmailPushHandler pointed at the mocks.
    let push_cfg = GmailPushConfig {
        project_id: "test-project".into(),
        topic_name: "projects/test-project/topics/t".into(),
        push_audience: TEST_AUDIENCE.into(),
        watched_label_ids: vec!["INBOX".into()],
        oauth_token: "fake-oauth".into(),
        gmail_base_url: Some(format!("http://{gmail_addr}")),
    };
    let handler = Arc::new(
        GmailPushHandler::new(push_cfg).with_jwks_url(format!("http://{jwks_addr}/certs")),
    );

    // 3. Build the registry with a tempfile cursor store and register
    //    the mailbox.
    let tmp = tempfile::tempdir().unwrap();
    let cursors = Arc::new(
        GmailCursorStore::load(tmp.path().join("cursor.json"))
            .await
            .unwrap(),
    );
    // Seed the cursor so fetch_new_messages asks for history since 99 —
    // safer than 0 which triggers "404 historyId too old" behaviour on
    // real Gmail. Our mock always returns the same canned history.
    cursors.put(TEST_MAILBOX, 99).await.unwrap();
    let registry = Arc::new(GmailPushRegistry::new(Arc::clone(&cursors)));
    registry.register(TEST_MAILBOX, Arc::clone(&handler)).await;

    // 4. Stand up the gateway with a recording handler.
    let recorder = RecorderState::default();
    let handler_for_state: Arc<dyn InboundHandler> = Arc::new(RecorderHandler(recorder.clone()));

    let sessions = Arc::new(SessionManager::new());
    let channels = Arc::new(ChannelRegistry::new());

    let gw_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gw_addr = gw_listener.local_addr().unwrap();
    drop(gw_listener);

    let mut gw_config = default_gateway_config();
    gw_config.host = "127.0.0.1".into();
    gw_config.port = gw_addr.port();

    let gateway = Gateway::with_handler(gw_config, Arc::clone(&handler_for_state))
        .with_shared_state(sessions, channels)
        .with_gmail_push(Arc::clone(&registry));

    let gw_task = tokio::spawn(async move {
        let _ = gateway.run().await;
    });

    // Let the gateway bind.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    // 5. POST a valid push; expect 204 and a single inbound dispatch.
    let body = make_pubsub_body("pubsub-abc-1");
    let jwt = sign_push_jwt(TEST_AUDIENCE);

    let client = reqwest::Client::new();
    let url = format!("http://{gw_addr}/webhooks/gmail-push");
    let resp = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {jwt}"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204, "expected 204 on first push");

    // Small pause to let the async dispatch complete.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    {
        let events = recorder.events.lock().unwrap();
        assert_eq!(events.len(), 1, "expected exactly one dispatch");
        match &events[0] {
            ChannelEvent::MessageReceived(m) => {
                assert_eq!(m.conversation.platform, "email");
                assert_eq!(m.conversation.channel_id, TEST_MAILBOX);
                assert_eq!(m.author, "carol@example.com");
                if let brainwires_network::channels::message::MessageContent::Text(t) = &m.content {
                    assert!(t.contains("integration hello"));
                    assert!(t.contains("hello from integration test"));
                } else {
                    panic!("expected text content");
                }
            }
            other => panic!("expected MessageReceived, got {other:?}"),
        }
    }

    // 6. Re-POST the same message id — de-dup should kick in.
    let resp2 = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {jwt}"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status(), 204, "expected 204 on redelivery");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    {
        let events = recorder.events.lock().unwrap();
        assert_eq!(events.len(), 1, "dedup should have suppressed redelivery");
    }

    // 7. A push with a bad JWT is rejected with 401.
    let resp3 = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, "Bearer not-a-jwt")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(make_pubsub_body("pubsub-xyz"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp3.status(), 401);

    // 8. Shutdown.
    gw_task.abort();
}

fn default_gateway_config() -> GatewayConfig {
    GatewayConfig {
        host: "127.0.0.1".into(),
        port: 0,
        max_connections: 16,
        session_timeout: std::time::Duration::from_secs(300),
        auth_tokens: vec![],
        webhook_enabled: false,
        webhook_path: "/webhook".into(),
        admin_enabled: false,
        admin_path: "/admin".into(),
        allowed_origins: vec![],
        webchat_enabled: false,
        webchat_jwt_secret: None,
        webchat_session_history_limit: 0,
        max_attachment_size_mb: 10,
        strip_system_spoofing: true,
        redact_secrets_in_output: true,
        max_messages_per_minute: 60,
        max_tool_calls_per_minute: 60,
        admin_token: None,
        webhook_secret: None,
        channels_enabled: true,
        allowed_channel_types: vec![],
        allowed_channel_ids: vec![],
    }
}
