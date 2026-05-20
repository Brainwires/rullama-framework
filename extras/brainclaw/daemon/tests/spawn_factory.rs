//! Integration tests for [`BrainClawSpawnFactory`].
//!
//! Exercises the daemon's concrete `SessionSpawnFactory` through the public
//! `GatewaySessionBroker` path so we test the same wiring the
//! `sessions_spawn` tool actually hits at runtime.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use tokio::sync::Mutex;

use brainclaw::session_spawn::BrainClawSpawnFactory;
use brainwires_agent::ChatAgent;
use brainwires_core::{
    ChatOptions, ChatResponse, Message, Provider, Role, StreamChunk, Tool, ToolContext, Usage,
};
use brainwires_gateway::sessions_broker::{GatewaySessionBroker, SessionRegistry};
use brainwires_tools::{
    BuiltinToolExecutor, SessionBroker, SessionId, SpawnRequest, ToolExecutor, ToolRegistry,
};

// A provider that echoes the last user message back as `"echo: <text>"`.
// Deterministic and tool-free — lets us assert on the first_reply content.
struct EchoProvider;

#[async_trait]
impl Provider for EchoProvider {
    fn name(&self) -> &str {
        "echo"
    }

    async fn chat(
        &self,
        messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> anyhow::Result<ChatResponse> {
        let user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.text())
            .unwrap_or_default();
        Ok(ChatResponse {
            message: Message::assistant(format!("echo: {user}")),
            usage: Usage::new(1, 1),
            finish_reason: Some("stop".to_string()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> futures::stream::BoxStream<'a, anyhow::Result<StreamChunk>> {
        let user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.text())
            .unwrap_or_default();
        let text = format!("echo: {user}");
        Box::pin(stream::iter(vec![
            Ok(StreamChunk::Text(text)),
            Ok(StreamChunk::Done),
        ]))
    }
}

fn make_factory() -> (Arc<BrainClawSpawnFactory>, Arc<SessionRegistry>) {
    let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
    let registry_inner = ToolRegistry::new();
    let executor: Arc<dyn ToolExecutor> = Arc::new(BuiltinToolExecutor::new(
        registry_inner,
        ToolContext::default(),
    ));
    let options = ChatOptions::default();
    let factory = Arc::new(BrainClawSpawnFactory::new(provider, executor, options));
    let reg = Arc::new(SessionRegistry::new());
    (factory, reg)
}

async fn seed_parent(reg: &SessionRegistry) {
    let _: tokio::sync::mpsc::UnboundedReceiver<String> = reg
        .register(
            SessionId::new("discord:alice"),
            "discord".into(),
            "alice".into(),
            None,
            // No ChatAgent attached to the parent — the broker does not
            // need one for `sessions_list`'s parent entry.
            None::<Arc<Mutex<ChatAgent>>>,
        )
        .await;
}

#[tokio::test]
async fn spawn_wait_for_first_reply_returns_echoed_content() {
    let (factory, reg) = make_factory();
    seed_parent(&reg).await;
    let broker = GatewaySessionBroker::new((*reg).clone(), factory);

    let req = SpawnRequest {
        prompt: "hi".to_string(),
        wait_for_first_reply: true,
        wait_timeout_secs: 5,
        ..Default::default()
    };
    let spawned = broker
        .spawn(&SessionId::new("discord:alice"), req)
        .await
        .expect("spawn ok");

    assert!(!spawned.id.as_str().is_empty());
    assert!(spawned.id.as_str().starts_with("spawn-"));
    let reply = spawned.first_reply.expect("first_reply populated");
    assert_eq!(reply.role, "assistant");
    assert!(
        reply.content.contains("echo: hi"),
        "expected echoed content, got {:?}",
        reply.content
    );

    // Session must be registered with parent pointer.
    let child = reg
        .get(&spawned.id)
        .await
        .expect("spawned session registered");
    assert_eq!(child.parent.as_ref().unwrap().as_str(), "discord:alice");
}

#[tokio::test]
async fn spawn_no_wait_returns_immediately_with_registered_session() {
    let (factory, reg) = make_factory();
    seed_parent(&reg).await;
    let broker = GatewaySessionBroker::new((*reg).clone(), factory);

    let req = SpawnRequest {
        prompt: "hello".to_string(),
        wait_for_first_reply: false,
        ..Default::default()
    };
    let spawned = broker
        .spawn(&SessionId::new("discord:alice"), req)
        .await
        .expect("spawn ok");

    assert!(spawned.first_reply.is_none());
    assert!(reg.get(&spawned.id).await.is_some());
}

#[tokio::test]
async fn sessions_list_sees_parent_and_spawned_child() {
    let (factory, reg) = make_factory();
    seed_parent(&reg).await;
    let broker = GatewaySessionBroker::new((*reg).clone(), factory);

    let req = SpawnRequest {
        prompt: "spawn-probe".to_string(),
        wait_for_first_reply: true,
        wait_timeout_secs: 5,
        ..Default::default()
    };
    let spawned = broker
        .spawn(&SessionId::new("discord:alice"), req)
        .await
        .expect("spawn ok");

    let list = broker.list().await.expect("list ok");
    let ids: Vec<&str> = list.iter().map(|s| s.id.as_str()).collect();
    assert!(
        ids.contains(&"discord:alice"),
        "parent missing from list: {ids:?}"
    );
    assert!(
        ids.contains(&spawned.id.as_str()),
        "child missing from list: {ids:?}"
    );
}

#[tokio::test]
async fn spawn_with_model_override_returns_error() {
    let (factory, reg) = make_factory();
    seed_parent(&reg).await;
    let broker = GatewaySessionBroker::new((*reg).clone(), factory);

    let req = SpawnRequest {
        prompt: "hi".to_string(),
        model: Some("claude-opus".to_string()),
        ..Default::default()
    };
    let err = broker
        .spawn(&SessionId::new("discord:alice"), req)
        .await
        .expect_err("model override should error");
    assert!(err.to_string().contains("`model` override"));
}

#[tokio::test]
async fn spawn_with_tools_override_returns_error() {
    let (factory, reg) = make_factory();
    seed_parent(&reg).await;
    let broker = GatewaySessionBroker::new((*reg).clone(), factory);

    let req = SpawnRequest {
        prompt: "hi".to_string(),
        tools: Some(vec!["web_search".to_string()]),
        ..Default::default()
    };
    let err = broker
        .spawn(&SessionId::new("discord:alice"), req)
        .await
        .expect_err("tools override should error");
    assert!(err.to_string().contains("`tools` override"));
}
