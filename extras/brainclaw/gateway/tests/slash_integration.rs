//! End-to-end style tests for slash-command interception through
//! `AgentInboundHandler`. Verifies the gateway intercepts `/new`, replies to
//! the channel, and forwards `\/new` to the agent as literal content.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::{ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool, Usage};
use brainwires_gateway::agent_handler::AgentInboundHandler;
use brainwires_gateway::channel_registry::{ChannelRegistry, ConnectedChannel};
use brainwires_gateway::session::SessionManager;
use brainwires_network::channels::ChannelCapabilities;
use brainwires_network::channels::events::ChannelEvent;
use brainwires_network::channels::identity::ConversationId;
use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};
use brainwires_tools::{BuiltinToolExecutor, ToolExecutor, ToolRegistry};
use chrono::Utc;
use futures::stream;
use std::sync::Mutex as StdMutex;
use tokio::sync::mpsc;
use uuid::Uuid;

/// A mock provider that records each user message it receives and replies
/// with a fixed canned text.
struct RecordingProvider {
    seen_user_text: Arc<StdMutex<Vec<String>>>,
    response_text: String,
}

impl RecordingProvider {
    fn new(response_text: &str) -> (Arc<Self>, Arc<StdMutex<Vec<String>>>) {
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let provider = Arc::new(Self {
            seen_user_text: seen.clone(),
            response_text: response_text.to_string(),
        });
        (provider, seen)
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn chat(
        &self,
        messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        if let Some(last) = messages.last()
            && let Some(t) = last.text()
        {
            self.seen_user_text.lock().unwrap().push(t.to_string());
        }
        Ok(ChatResponse {
            message: Message::assistant(&self.response_text),
            usage: Usage::new(3, 5),
            finish_reason: Some("stop".to_string()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> futures::stream::BoxStream<'a, Result<StreamChunk>> {
        if let Some(last) = messages.last()
            && let Some(t) = last.text()
        {
            self.seen_user_text.lock().unwrap().push(t.to_string());
        }
        let text = self.response_text.clone();
        Box::pin(stream::iter(vec![
            Ok(StreamChunk::Text(text)),
            Ok(StreamChunk::Done),
        ]))
    }
}

fn make_executor() -> Arc<dyn ToolExecutor> {
    let registry = ToolRegistry::new();
    Arc::new(BuiltinToolExecutor::new(
        registry,
        brainwires_core::ToolContext::default(),
    ))
}

fn make_message(platform: &str, author: &str, text: &str) -> ChannelMessage {
    ChannelMessage {
        id: MessageId::new(Uuid::new_v4().to_string()),
        conversation: ConversationId {
            platform: platform.to_string(),
            channel_id: "general".to_string(),
            server_id: None,
        },
        author: author.to_string(),
        content: MessageContent::Text(text.to_string()),
        thread_id: None,
        reply_to: None,
        timestamp: Utc::now(),
        attachments: vec![],
        metadata: HashMap::new(),
    }
}

fn register_channel(channels: &Arc<ChannelRegistry>) -> (Uuid, mpsc::Receiver<String>) {
    let (tx, rx) = mpsc::channel::<String>(16);
    let channel_id = Uuid::new_v4();
    channels.register(ConnectedChannel {
        id: channel_id,
        channel_type: "test".to_string(),
        capabilities: ChannelCapabilities::empty(),
        connected_at: Utc::now(),
        last_heartbeat: Utc::now(),
        message_tx: tx,
    });
    (channel_id, rx)
}

fn extract_text(ev: &ChannelEvent) -> Option<String> {
    match ev {
        ChannelEvent::MessageReceived(m) => match &m.content {
            MessageContent::Text(t) => Some(t.clone()),
            _ => None,
        },
        _ => None,
    }
}

#[tokio::test]
async fn slash_new_is_intercepted_and_hello_reaches_agent() {
    let sessions = Arc::new(SessionManager::new());
    let channels = Arc::new(ChannelRegistry::new());
    let (provider, seen) = RecordingProvider::new("hi from agent");
    let handler = AgentInboundHandler::new(
        sessions,
        channels.clone(),
        provider,
        make_executor(),
        ChatOptions::default(),
    );

    let (channel_id, mut rx) = register_channel(&channels);

    // `/new` should be handled by the slash interceptor (no agent call).
    let new_msg = make_message("test", "user-1", "/new");
    handler
        .dispatch_message(channel_id, new_msg)
        .await
        .expect("dispatch /new");

    let reply_json = rx.recv().await.expect("reply for /new");
    let reply_ev: ChannelEvent = serde_json::from_str(&reply_json).unwrap();
    let reply_text = extract_text(&reply_ev).expect("text reply");
    assert_eq!(reply_text, "Session reset.");
    assert!(
        seen.lock().unwrap().is_empty(),
        "agent should not receive /new",
    );

    // Plain text should still reach the agent.
    let hello_msg = make_message("test", "user-1", "hello");
    handler
        .dispatch_message(channel_id, hello_msg)
        .await
        .expect("dispatch hello");
    let _agent_reply = rx.recv().await.expect("agent reply for hello");
    let seen_snapshot = seen.lock().unwrap().clone();
    assert!(
        seen_snapshot.iter().any(|t| t == "hello"),
        "expected agent to receive `hello`, got {seen_snapshot:?}",
    );
}

#[tokio::test]
async fn escaped_slash_is_forwarded_to_agent() {
    let sessions = Arc::new(SessionManager::new());
    let channels = Arc::new(ChannelRegistry::new());
    let (provider, seen) = RecordingProvider::new("ack");
    let handler = AgentInboundHandler::new(
        sessions,
        channels.clone(),
        provider,
        make_executor(),
        ChatOptions::default(),
    );

    let (channel_id, mut rx) = register_channel(&channels);

    let msg = make_message("test", "user-2", "\\/new");
    handler
        .dispatch_message(channel_id, msg)
        .await
        .expect("dispatch escaped");
    let _ = rx.recv().await.expect("agent replied");
    let seen_snapshot = seen.lock().unwrap().clone();
    assert!(
        seen_snapshot.iter().any(|t| t == "/new"),
        "expected agent to receive literal `/new`, got {seen_snapshot:?}",
    );
}

#[tokio::test]
async fn unknown_slash_is_rejected_not_forwarded() {
    let sessions = Arc::new(SessionManager::new());
    let channels = Arc::new(ChannelRegistry::new());
    let (provider, seen) = RecordingProvider::new("ack");
    let handler = AgentInboundHandler::new(
        sessions,
        channels.clone(),
        provider,
        make_executor(),
        ChatOptions::default(),
    );

    let (channel_id, mut rx) = register_channel(&channels);

    let msg = make_message("test", "user-3", "/notarealcommand");
    handler
        .dispatch_message(channel_id, msg)
        .await
        .expect("dispatch unknown");
    let reply_json = rx.recv().await.expect("reply for unknown");
    let reply_ev: ChannelEvent = serde_json::from_str(&reply_json).unwrap();
    let text = extract_text(&reply_ev).expect("text");
    assert!(text.contains("Unknown command"));
    assert!(text.contains("/help"));
    assert!(seen.lock().unwrap().is_empty());
}
