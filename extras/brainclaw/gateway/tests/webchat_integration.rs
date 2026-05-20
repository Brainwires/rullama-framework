//! End-to-end integration tests for the JWT-gated WebChat channel.
//!
//! These tests boot a real gateway instance against an ephemeral port, wire
//! in a deterministic mock inbound handler that echoes user messages back
//! through the registered channel, and then exercise the `/webchat/ws`
//! protocol from the browser's perspective using `tokio-tungstenite`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use uuid::Uuid;

use brainwires_gateway::channel_registry::ChannelRegistry;
use brainwires_gateway::config::GatewayConfig;
use brainwires_gateway::router::InboundHandler;
use brainwires_gateway::server::Gateway;
use brainwires_gateway::session::SessionManager;
use brainwires_gateway::webchat::{Hs256Verifier, WebChatHistory, issue_hs256_jwt};
use brainwires_network::channels::events::ChannelEvent;
use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};

/// Mock inbound handler that echoes any `MessageReceived` back as an
/// assistant reply. This mirrors what `AgentInboundHandler` does but
/// keeps the test hermetic (no provider / no tool executor).
struct EchoHandler {
    channels: Arc<ChannelRegistry>,
}

#[async_trait]
impl InboundHandler for EchoHandler {
    async fn handle_inbound(&self, channel_id: Uuid, event: &ChannelEvent) -> Result<()> {
        if let ChannelEvent::MessageReceived(msg) = event {
            let text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                _ => return Ok(()),
            };
            let reply = ChannelEvent::MessageReceived(ChannelMessage {
                id: MessageId::new(Uuid::new_v4().to_string()),
                conversation: msg.conversation.clone(),
                author: "assistant".to_string(),
                content: MessageContent::Text(format!("echo: {text}")),
                thread_id: None,
                reply_to: Some(msg.id.clone()),
                timestamp: chrono::Utc::now(),
                attachments: vec![],
                metadata: std::collections::HashMap::new(),
            });
            let json = serde_json::to_string(&reply)?;
            if let Some(tx) = self.channels.get_sender(&channel_id) {
                let _ = tx.send(json).await;
            }
        }
        Ok(())
    }
}

struct TestServer {
    addr: SocketAddr,
    secret: Vec<u8>,
    _history: Arc<WebChatHistory>,
}

/// Start a fresh, isolated gateway instance on an ephemeral port. Each
/// test uses its own server so that port races between parallel tests
/// cannot flake.
async fn start_server() -> TestServer {
    // Pick an ephemeral port by binding + releasing; then hand the port
    // number to the gateway to re-bind. This is racy in theory but
    // reliable in practice on Linux for the duration of these tests.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let config = GatewayConfig {
        host: "127.0.0.1".to_string(),
        port: addr.port(),
        admin_token: None,
        webchat_enabled: true,
        ..Default::default()
    };

    let sessions = Arc::new(SessionManager::new());
    let channels = Arc::new(ChannelRegistry::new());
    let handler = Arc::new(EchoHandler {
        channels: channels.clone(),
    });
    let history = Arc::new(WebChatHistory::new());
    let secret = b"integration-test-secret".to_vec();
    let verifier = Arc::new(Hs256Verifier::new(secret.clone()));

    let gateway = Gateway::with_handler(config, handler)
        .with_shared_state(sessions, channels)
        .with_webchat_verifier(verifier, history.clone());

    tokio::spawn(async move {
        let _ = gateway.run().await;
    });

    // Wait for the server to bind.
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    TestServer {
        addr,
        secret,
        _history: history,
    }
}

async fn connect_ws(
    addr: SocketAddr,
    token: Option<&str>,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    tokio_tungstenite::tungstenite::Error,
> {
    let url = match token {
        Some(t) => format!("ws://{}/webchat/ws?token={}", addr, t),
        None => format!("ws://{}/webchat/ws", addr),
    };
    let req = url.into_client_request()?;
    let (stream, _resp) = connect_async(req).await?;
    Ok(stream)
}

#[tokio::test]
async fn connects_and_echoes_user_message() {
    let s = start_server().await;
    let token = issue_hs256_jwt("alice", &s.secret, 60).unwrap();
    let mut ws = connect_ws(s.addr, Some(&token)).await.expect("connect");

    // First frame must be session echo.
    let first = ws.next().await.expect("session frame").expect("frame ok");
    let text = match first {
        WsMessage::Text(t) => t.to_string(),
        other => panic!("unexpected frame: {other:?}"),
    };
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "session");
    assert_eq!(v["id"], "webchat:alice");

    // Send a user message.
    ws.send(WsMessage::Text(
        r#"{"type":"message","content":"hi there"}"#.into(),
    ))
    .await
    .unwrap();

    // Await assistant reply.
    let reply = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout")
        .expect("frame")
        .expect("frame ok");
    let reply_text = match reply {
        WsMessage::Text(t) => t.to_string(),
        other => panic!("unexpected frame: {other:?}"),
    };
    let v: serde_json::Value = serde_json::from_str(&reply_text).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"], "echo: hi there");
    ws.close(None).await.ok();
}

#[tokio::test]
async fn rejects_connection_without_token() {
    let s = start_server().await;
    let err = connect_ws(s.addr, None)
        .await
        .expect_err("expected rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("401") || msg.to_lowercase().contains("unauthorized"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn rejects_connection_with_expired_token() {
    let s = start_server().await;
    let token = issue_hs256_jwt("bob", &s.secret, 0).unwrap();
    // Ensure clock moved past `exp`.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let err = connect_ws(s.addr, Some(&token))
        .await
        .expect_err("expected rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("401") || msg.to_lowercase().contains("unauthorized"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn oversized_payload_emits_error_frame() {
    let s = start_server().await;
    let token = issue_hs256_jwt("carol", &s.secret, 60).unwrap();
    let mut ws = connect_ws(s.addr, Some(&token)).await.expect("connect");

    // Drain session frame.
    let _ = ws.next().await;

    // Send a frame above MAX_INBOUND_FRAME_BYTES (256KB) but well within
    // the WebSocket-level max-frame threshold that tungstenite applies
    // on the server side by default.
    let huge = "x".repeat(260 * 1024);
    let payload = format!(r#"{{"type":"message","content":"{}"}}"#, huge);
    ws.send(WsMessage::Text(payload.into())).await.unwrap();

    let reply = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout")
        .expect("frame")
        .expect("frame ok");
    let reply_text = match reply {
        WsMessage::Text(t) => t.to_string(),
        other => panic!("unexpected frame: {other:?}"),
    };
    let v: serde_json::Value = serde_json::from_str(&reply_text).unwrap();
    assert_eq!(v["type"], "error");
    assert!(v["message"].as_str().unwrap().contains("payload"));
    ws.close(None).await.ok();
}

#[tokio::test]
async fn resume_returns_session_history() {
    let s = start_server().await;
    let token = issue_hs256_jwt("dave", &s.secret, 60).unwrap();

    // First connection: send a message so history accrues.
    {
        let mut ws = connect_ws(s.addr, Some(&token)).await.expect("connect");
        let _ = ws.next().await; // session frame
        ws.send(WsMessage::Text(
            r#"{"type":"message","content":"first"}"#.into(),
        ))
        .await
        .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), ws.next()).await;
        ws.close(None).await.ok();
    }

    // Give the write bridge a tick to record the assistant entry in history.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Second connection: resume and expect history frame with our message.
    let mut ws = connect_ws(s.addr, Some(&token)).await.expect("reconnect");
    let _ = ws.next().await; // session frame

    ws.send(WsMessage::Text(
        r#"{"type":"resume","session_id":"webchat:dave"}"#.into(),
    ))
    .await
    .unwrap();

    let reply = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout")
        .expect("frame")
        .expect("frame ok");
    let reply_text = match reply {
        WsMessage::Text(t) => t.to_string(),
        other => panic!("unexpected frame: {other:?}"),
    };
    let v: serde_json::Value = serde_json::from_str(&reply_text).unwrap();
    assert_eq!(v["type"], "history");
    assert_eq!(v["session_id"], "webchat:dave");
    let messages = v["messages"].as_array().expect("messages array");
    assert!(!messages.is_empty(), "history should not be empty");
    // Should contain at least one "user" entry.
    let roles: Vec<&str> = messages.iter().filter_map(|m| m["role"].as_str()).collect();
    assert!(roles.contains(&"user"));
    ws.close(None).await.ok();
}
