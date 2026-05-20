//! Integration tests for the M4 A2A bridge.
//!
//! Runs against the public lib API of `brainwires-home`. The first two
//! tests exercise the bridge directly; the third extends the M3
//! `test_full_handshake_in_process` shape into a real `message/send`
//! round-trip through the axum router and a real WebRTC data channel.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::{
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
};
use brainwires_a2a::{
    A2aError, JsonRpcRequest, JsonRpcResponse, Message as A2aMessage, RequestId, Role as A2aRole,
    SendMessageRequest,
};
use brainwires_core::{
    ChatOptions, ChatResponse, Message as CoreMessage, Provider, StreamChunk, Tool, ToolContext,
    Usage,
};
use brainwires_home::{HomeServer, a2a::A2aBridge};
use brainwires_inference::ChatAgent;
use brainwires_tool_builtins::BuiltinToolExecutor;
use brainwires_tool_runtime::ToolRegistry;
use futures::stream;
use serde_json::Value;
use tokio::sync::mpsc;
use tower::ServiceExt;

// ---------- echo provider for offline tests ----------

struct EchoProvider;

#[async_trait::async_trait]
impl Provider for EchoProvider {
    fn name(&self) -> &str {
        "echo"
    }

    async fn chat(
        &self,
        messages: &[CoreMessage],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let last = messages
            .iter()
            .rev()
            .find_map(|m| m.text().map(|s| s.to_string()))
            .unwrap_or_default();
        Ok(ChatResponse {
            message: CoreMessage::assistant(format!("echo: {last}")),
            usage: Usage::new(0, 0),
            finish_reason: Some("stop".to_string()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [CoreMessage],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> futures::stream::BoxStream<'a, Result<StreamChunk>> {
        let last = messages
            .iter()
            .rev()
            .find_map(|m| m.text().map(|s| s.to_string()))
            .unwrap_or_default();
        Box::pin(stream::iter(vec![
            Ok(StreamChunk::Text(format!("echo: {last}"))),
            Ok(StreamChunk::Done),
        ]))
    }
}

fn echo_bridge() -> Arc<A2aBridge> {
    let provider = Arc::new(EchoProvider);
    let executor = Arc::new(BuiltinToolExecutor::new(
        ToolRegistry::new(),
        ToolContext::default(),
    ));
    let agent = ChatAgent::new(provider, executor, ChatOptions::default());
    Arc::new(A2aBridge::new(agent))
}

// ---------- direct dispatch tests ----------

#[tokio::test]
async fn test_message_send_routes_to_agent() {
    let bridge = echo_bridge();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "message/send".to_string(),
        params: Some(
            serde_json::to_value(SendMessageRequest {
                tenant: None,
                message: A2aMessage::user_text("hello home"),
                configuration: None,
                metadata: None,
            })
            .unwrap(),
        ),
        id: RequestId::Number(11),
    };
    let resp: JsonRpcResponse = bridge.dispatch(req).await;
    assert!(resp.error.is_none(), "got error: {:?}", resp.error);
    assert_eq!(resp.id, RequestId::Number(11));
    let msg: A2aMessage = serde_json::from_value(resp.result.unwrap()).unwrap();
    assert_eq!(msg.role, A2aRole::Agent);
    assert_eq!(
        msg.parts.first().and_then(|p| p.text.as_deref()),
        Some("echo: hello home"),
    );
}

#[tokio::test]
async fn test_unknown_method_returns_method_not_found() {
    let bridge = echo_bridge();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "not_a_real_method".to_string(),
        params: None,
        id: RequestId::String("call-77".to_string()),
    };
    let resp = bridge.dispatch(req).await;
    let err: A2aError = resp.error.expect("error must be set");
    assert_eq!(err.code, brainwires_a2a::error::METHOD_NOT_FOUND);
    assert_eq!(resp.id, RequestId::String("call-77".to_string()));
}

// ---------- end-to-end: real WebRTC handshake + message/send ----------

async fn body_json(resp: axum::response::Response) -> Value {
    let body = resp.into_body();
    let bytes = to_bytes(body, 1 << 20).await.expect("collect body");
    serde_json::from_slice(&bytes).expect("body is valid JSON")
}

fn json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn empty_request(method: Method, uri: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

/// Full handshake: PWA-side peer drives the real axum router, opens the
/// `"a2a"` data channel, sends a `message/send` request, and asserts the
/// reply is an A2A `ROLE_AGENT` message produced by the echo provider.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_full_handshake_with_a2a() -> Result<()> {
    use webrtc::data_channel::DataChannelEvent;

    let server = HomeServer::builder()
        .long_poll_timeout(Duration::from_secs(3))
        .with_agent(echo_bridge())
        .build()?;
    let app = server.router();

    // Step 1: create the session.
    let resp = app
        .clone()
        .oneshot(empty_request(Method::POST, "/signal/session"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let id = v["session_id"].as_str().unwrap().to_string();

    // Step 2: PWA builds a peer + opens the canonical data channel.
    let pwa = brainwires_home::webrtc::build_peer(vec![]).await?;
    let dc = brainwires_home::webrtc::open_a2a_channel(&pwa).await?;

    // Step 3: forward PWA local ICE candidates → home.
    let mut pwa_events = pwa.subscribe();
    let app_for_ice = app.clone();
    let id_for_ice = id.clone();
    let pwa_ice_relay = tokio::spawn(async move {
        loop {
            match pwa_events.recv().await {
                Ok(brainwires_home::webrtc::PeerEvent::LocalIceCandidate {
                    candidate,
                    sdp_mid,
                    sdp_mline_index,
                }) => {
                    let body = serde_json::json!({
                        "candidate": candidate,
                        "sdpMid": sdp_mid,
                        "sdpMLineIndex": sdp_mline_index,
                    });
                    let _ = app_for_ice
                        .clone()
                        .oneshot(json_request(
                            Method::POST,
                            &format!("/signal/ice/{id_for_ice}"),
                            body,
                        ))
                        .await;
                }
                Ok(brainwires_home::webrtc::PeerEvent::ConnectionState(s))
                    if matches!(
                        s,
                        webrtc::peer_connection::RTCPeerConnectionState::Connected
                            | webrtc::peer_connection::RTCPeerConnectionState::Failed
                            | webrtc::peer_connection::RTCPeerConnectionState::Closed
                    ) =>
                {
                    break;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                _ => continue,
            }
        }
    });

    // Step 4: pump home → PWA ICE candidates.
    let app_for_pull = app.clone();
    let id_for_pull = id.clone();
    let pwa_pc_clone = pwa.pc.clone();
    let home_to_pwa_relay = tokio::spawn(async move {
        let mut cursor: usize = 0;
        for _ in 0..40 {
            let resp = app_for_pull
                .clone()
                .oneshot(empty_request(
                    Method::GET,
                    &format!("/signal/ice/{id_for_pull}?since={cursor}"),
                ))
                .await
                .unwrap();
            if resp.status() != StatusCode::OK {
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
            let v = body_json(resp).await;
            let new_cursor = v["cursor"].as_u64().unwrap_or(cursor as u64) as usize;
            if let Some(arr) = v["candidates"].as_array() {
                for c in arr {
                    if let Some(cand_str) = c.get("candidate").and_then(|x| x.as_str()) {
                        if cand_str.is_empty() {
                            continue;
                        }
                        let init = webrtc::peer_connection::RTCIceCandidateInit {
                            candidate: cand_str.to_string(),
                            sdp_mid: c
                                .get("sdpMid")
                                .and_then(|x| x.as_str())
                                .map(|s| s.to_string()),
                            sdp_mline_index: c
                                .get("sdpMLineIndex")
                                .and_then(|x| x.as_u64())
                                .map(|x| x as u16),
                            username_fragment: None,
                            url: None,
                        };
                        let _ = pwa_pc_clone.add_ice_candidate(init).await;
                    }
                }
            }
            cursor = new_cursor;
        }
    });

    // Step 5: PWA → home offer.
    let offer = pwa
        .pc
        .create_offer(None)
        .await
        .map_err(|e| anyhow::anyhow!("create_offer: {e}"))?;
    pwa.pc
        .set_local_description(offer.clone())
        .await
        .map_err(|e| anyhow::anyhow!("set_local_description: {e}"))?;

    let resp = app
        .clone()
        .oneshot(json_request(
            Method::POST,
            &format!("/signal/offer/{id}"),
            serde_json::json!({ "sdp": offer.sdp, "type": "offer" }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Step 6: GET the answer + apply it.
    let resp = app
        .clone()
        .oneshot(empty_request(Method::GET, &format!("/signal/answer/{id}")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    let answer_sdp = v["sdp"].as_str().unwrap().to_string();
    let answer = webrtc::peer_connection::RTCSessionDescription::answer(answer_sdp)
        .map_err(|e| anyhow::anyhow!("answer: {e}"))?;
    pwa.pc
        .set_remote_description(answer)
        .await
        .map_err(|e| anyhow::anyhow!("set_remote_description(answer): {e}"))?;

    // Step 7: send a message/send request, capture reply.
    let (got_tx, mut got_rx) = mpsc::channel::<String>(1);
    let dc_for_reader = dc.clone();
    let reader = tokio::spawn(async move {
        let send_msg = SendMessageRequest {
            tenant: None,
            message: A2aMessage::user_text("ping the agent"),
            configuration: None,
            metadata: None,
        };
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "message/send".to_string(),
            params: Some(serde_json::to_value(&send_msg).unwrap()),
            id: RequestId::Number(42),
        };
        let frame = serde_json::to_string(&req).unwrap();
        let mut sent = false;
        loop {
            match dc_for_reader.poll().await {
                Some(DataChannelEvent::OnOpen) => {
                    if !sent {
                        let _ = dc_for_reader.send_text(&frame).await;
                        sent = true;
                    }
                }
                Some(DataChannelEvent::OnMessage(msg)) => {
                    let s = String::from_utf8_lossy(&msg.data).into_owned();
                    let _ = got_tx.send(s).await;
                    break;
                }
                Some(DataChannelEvent::OnClose) | None => break,
                _ => continue,
            }
        }
    });

    let reply_text = tokio::time::timeout(Duration::from_secs(20), got_rx.recv())
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for message/send reply"))?
        .ok_or_else(|| anyhow::anyhow!("data channel closed before reply"))?;

    let resp: JsonRpcResponse = serde_json::from_str(&reply_text)
        .map_err(|e| anyhow::anyhow!("reply not a JsonRpcResponse ({e}): {reply_text}"))?;
    assert_eq!(resp.id, RequestId::Number(42));
    assert!(
        resp.error.is_none(),
        "agent returned error: {:?}",
        resp.error
    );
    let agent_msg: A2aMessage =
        serde_json::from_value(resp.result.expect("result")).expect("result is an A2A Message");
    assert_eq!(agent_msg.role, A2aRole::Agent);
    assert_eq!(
        agent_msg.parts.first().and_then(|p| p.text.as_deref()),
        Some("echo: ping the agent"),
    );

    // Cleanup.
    let _ = dc.close().await;
    let _ = pwa.pc.close().await;
    let _ = app
        .oneshot(empty_request(Method::DELETE, &format!("/signal/{id}")))
        .await;
    let _ = reader.await;
    let _ = pwa_ice_relay.await;
    let _ = home_to_pwa_relay.await;
    Ok(())
}
