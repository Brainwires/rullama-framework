//! A2A bridge: inbound JSON-RPC over WebRTC → brainwires-agent → reply.
//!
//! For M4 we handle the `message/send` method end-to-end (the A2A 0.3
//! `SendMessage` method, per the constants in `brainwires_a2a::jsonrpc`).
//! Streaming (`message/stream`) lands with the browser-side
//! home-provider.js in M9.
//!
//! # Why `ChatAgent` and not `TaskAgent`?
//!
//! The Phase-2 plan's `M4` writeup names "TaskAgent" as the target, but the
//! framework's [`brainwires_agent::TaskAgent`] is the heavy autonomous-loop
//! agent (execution graph, validation loop, file-lock manager, agent
//! communication hub). A chat-PWA dial-home daemon serving the A2A
//! `message/send` method is exactly what
//! [`brainwires_agent::ChatAgent::process_message`] was built for —
//! provider + tool registry + a single text in / text out call. The plan
//! explicitly calls out that the agent type is "TaskAgent (or whatever the
//! actual type is called — could also be Agent, ChatAgent, etc.)", so we
//! adopt `ChatAgent` here. M11+ can swap in `TaskAgent` if the home daemon
//! grows autonomous-loop responsibilities.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use brainwires_a2a::{
    A2aError, JsonRpcRequest, JsonRpcResponse, Message as A2aMessage, Part as A2aPart, RequestId,
    Role as A2aRole, SendMessageRequest, jsonrpc::METHOD_MESSAGE_SEND,
};
use brainwires_inference::ChatAgent;
use serde_json::{Value, json};
use tokio::sync::Mutex;

/// Smoke-test JSON-RPC method retained from M3. Lets the in-process
/// integration test verify the data channel without paying the cost of
/// spinning up a real provider.
pub const METHOD_SYSTEM_PING: &str = "system/ping";

/// Bridge from inbound JSON-RPC frames to a long-lived [`ChatAgent`].
///
/// The home daemon is single-user by design, so one agent serves every
/// session. The agent is wrapped in an [`Mutex`] because [`ChatAgent`]'s
/// completion loop takes `&mut self` (it appends to its own history), and
/// concurrent inbound `message/send` calls from a single PWA tab are
/// already serialized by the WebRTC data channel.
#[derive(Clone)]
pub struct A2aBridge {
    agent: Arc<Mutex<ChatAgent>>,
}

impl A2aBridge {
    /// Build a bridge that owns this agent.
    pub fn new(agent: ChatAgent) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
        }
    }

    /// Build a bridge from an already-`Arc<Mutex<ChatAgent>>`-wrapped agent.
    /// Useful when the same agent needs to be observed from outside the
    /// bridge (e.g. tests inspecting conversation history).
    pub fn from_shared(agent: Arc<Mutex<ChatAgent>>) -> Self {
        Self { agent }
    }

    /// Route a single inbound JSON-RPC request to the appropriate handler.
    ///
    /// Always returns a [`JsonRpcResponse`] — internal errors are wrapped as
    /// JSON-RPC `error` objects so the wire stays well-formed even when the
    /// underlying provider fails.
    pub async fn dispatch(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone();
        match req.method.as_str() {
            // A2A 0.3 spells this "SendMessage" (see brainwires_a2a's
            // `METHOD_MESSAGE_SEND`). The chat PWA's outbound JSON-RPC client
            // also sends the human-readable "message/send" alias, so we
            // accept both — they map to the same handler.
            METHOD_MESSAGE_SEND | "message/send" => self.handle_message_send(req).await,
            METHOD_SYSTEM_PING | "ping" => self.handle_ping(id),
            other => JsonRpcResponse::error(id, A2aError::method_not_found(other)),
        }
    }

    async fn handle_message_send(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone();
        let params = match req.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    A2aError::invalid_params("message/send requires params"),
                );
            }
        };
        let send: SendMessageRequest = match serde_json::from_value(params) {
            Ok(s) => s,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    A2aError::invalid_params(format!("malformed message/send params: {e}")),
                );
            }
        };

        let user_text = extract_text(&send.message);
        if user_text.is_empty() {
            return JsonRpcResponse::error(
                id,
                A2aError::invalid_params("message has no text part"),
            );
        }

        let reply_text = {
            let mut agent = self.agent.lock().await;
            match agent.process_message(&user_text).await {
                Ok(t) => t,
                Err(e) => {
                    return JsonRpcResponse::error(id, A2aError::internal(format!("agent: {e}")));
                }
            }
        };

        let agent_msg = A2aMessage {
            message_id: uuid::Uuid::new_v4().to_string(),
            role: A2aRole::Agent,
            parts: vec![A2aPart {
                text: Some(reply_text),
                raw: None,
                url: None,
                data: None,
                media_type: None,
                filename: None,
                metadata: None,
            }],
            context_id: send.message.context_id.clone(),
            task_id: None,
            reference_task_ids: None,
            metadata: None,
            extensions: None,
        };

        match serde_json::to_value(&agent_msg) {
            Ok(v) => JsonRpcResponse::success(id, v),
            Err(e) => {
                JsonRpcResponse::error(id, A2aError::internal(format!("serialize reply: {e}")))
            }
        }
    }

    fn handle_ping(&self, id: RequestId) -> JsonRpcResponse {
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        JsonRpcResponse::success(id, json!({ "ok": true, "ts": ts_ms }))
    }
}

/// Concatenate every text part of an A2A message in order. Empty if there
/// are no text parts.
fn extract_text(msg: &A2aMessage) -> String {
    let mut out = String::new();
    for part in &msg.parts {
        if let Some(t) = part.text.as_deref() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(t);
        }
    }
    out
}

// ---------- M3 compatibility shim ----------

/// Build a JSON-RPC 2.0 reply to a `ping` request.
///
/// Retained as a free function for backwards compatibility with the M3
/// `webrtc.rs` dispatcher path that didn't yet have a bridge in scope.
/// New code should construct an [`A2aBridge`] and call
/// [`A2aBridge::dispatch`] instead.
pub fn handle_jsonrpc_ping(req_id: Value) -> Value {
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": { "ok": true, "ts": ts_ms },
    })
}

// ---------- tests ----------

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared test scaffolding for the bridge.
    //!
    //! Builds a `ChatAgent` whose provider just echoes the most recent user
    //! message back as `"echo: <text>"`. Zero network, zero tools, ~25 LOC.

    use std::sync::Arc;

    use anyhow::Result;
    use brainwires_core::{
        ChatOptions, ChatResponse, Message as CoreMessage, Provider, StreamChunk, Tool,
        ToolContext, Usage,
    };
    use brainwires_inference::ChatAgent;
    use brainwires_tool_builtins::BuiltinToolExecutor;
    use brainwires_tool_runtime::ToolRegistry;
    use futures::stream;

    /// Provider that replies `"echo: <last user text>"`. Intended for tests.
    pub struct EchoProvider;

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

    /// Build a `ChatAgent` whose provider is [`EchoProvider`] and whose
    /// tool registry is empty. Suitable for any test that just needs the
    /// bridge to run end-to-end without a real LLM.
    pub fn echo_chat_agent() -> ChatAgent {
        let provider = Arc::new(EchoProvider);
        let executor = Arc::new(BuiltinToolExecutor::new(
            ToolRegistry::new(),
            ToolContext::default(),
        ));
        ChatAgent::new(provider, executor, ChatOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_reply_matches_id_and_has_ok_true() {
        let reply = handle_jsonrpc_ping(json!(42));
        assert_eq!(reply["jsonrpc"], "2.0");
        assert_eq!(reply["id"], json!(42));
        assert_eq!(reply["result"]["ok"], json!(true));
        assert!(reply["result"]["ts"].as_u64().is_some());
    }

    #[test]
    fn ping_reply_supports_string_ids() {
        let reply = handle_jsonrpc_ping(json!("abc"));
        assert_eq!(reply["id"], json!("abc"));
    }

    #[tokio::test]
    async fn dispatch_routes_message_send_through_agent() {
        let bridge = A2aBridge::new(test_support::echo_chat_agent());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "message/send".to_string(),
            params: Some(
                serde_json::to_value(SendMessageRequest {
                    tenant: None,
                    message: A2aMessage::user_text("hello agent"),
                    configuration: None,
                    metadata: None,
                })
                .unwrap(),
            ),
            id: RequestId::Number(1),
        };
        let resp = bridge.dispatch(req).await;
        assert_eq!(resp.id, RequestId::Number(1));
        assert!(resp.error.is_none(), "got error: {:?}", resp.error);
        let result = resp.result.expect("result is set");
        let msg: A2aMessage = serde_json::from_value(result).expect("result is a Message");
        assert_eq!(msg.role, A2aRole::Agent);
        assert_eq!(
            msg.parts.first().and_then(|p| p.text.as_deref()),
            Some("echo: hello agent"),
        );
    }

    #[tokio::test]
    async fn dispatch_uses_send_message_alias_constant() {
        // The A2A 0.3 spec spells the method "SendMessage" — make sure
        // dispatch routes the spec form too, not just the "/" alias.
        let bridge = A2aBridge::new(test_support::echo_chat_agent());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: METHOD_MESSAGE_SEND.to_string(),
            params: Some(
                serde_json::to_value(SendMessageRequest {
                    tenant: None,
                    message: A2aMessage::user_text("via spec name"),
                    configuration: None,
                    metadata: None,
                })
                .unwrap(),
            ),
            id: RequestId::String("req-A".to_string()),
        };
        let resp = bridge.dispatch(req).await;
        assert!(resp.error.is_none(), "got error: {:?}", resp.error);
        let msg: A2aMessage = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(msg.parts[0].text.as_deref(), Some("echo: via spec name"),);
    }

    #[tokio::test]
    async fn dispatch_unknown_method_returns_method_not_found() {
        let bridge = A2aBridge::new(test_support::echo_chat_agent());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "not_real".to_string(),
            params: None,
            id: RequestId::Number(7),
        };
        let resp = bridge.dispatch(req).await;
        assert!(resp.result.is_none());
        let err = resp.error.expect("error must be set");
        assert_eq!(err.code, brainwires_a2a::error::METHOD_NOT_FOUND);
        assert_eq!(resp.id, RequestId::Number(7));
    }

    #[tokio::test]
    async fn dispatch_message_send_without_params_is_invalid_params() {
        let bridge = A2aBridge::new(test_support::echo_chat_agent());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "message/send".to_string(),
            params: None,
            id: RequestId::Number(2),
        };
        let resp = bridge.dispatch(req).await;
        let err = resp.error.expect("error must be set");
        assert_eq!(err.code, brainwires_a2a::error::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn dispatch_ping_returns_ok_true() {
        let bridge = A2aBridge::new(test_support::echo_chat_agent());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "system/ping".to_string(),
            params: None,
            id: RequestId::Number(3),
        };
        let resp = bridge.dispatch(req).await;
        assert!(resp.error.is_none());
        let r = resp.result.expect("result");
        assert_eq!(r["ok"], json!(true));
        assert!(r["ts"].as_u64().is_some());
    }

    #[test]
    fn extract_text_concatenates_text_parts() {
        let msg = A2aMessage {
            message_id: "m".to_string(),
            role: A2aRole::User,
            parts: vec![
                A2aPart {
                    text: Some("a".to_string()),
                    raw: None,
                    url: None,
                    data: None,
                    media_type: None,
                    filename: None,
                    metadata: None,
                },
                A2aPart {
                    text: None,
                    raw: None,
                    url: Some("https://example.com".to_string()),
                    data: None,
                    media_type: None,
                    filename: None,
                    metadata: None,
                },
                A2aPart {
                    text: Some("b".to_string()),
                    raw: None,
                    url: None,
                    data: None,
                    media_type: None,
                    filename: None,
                    metadata: None,
                },
            ],
            context_id: None,
            task_id: None,
            reference_task_ids: None,
            metadata: None,
            extensions: None,
        };
        assert_eq!(extract_text(&msg), "a\nb");
    }
}
