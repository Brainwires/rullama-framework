//! End-to-end-ish integration test for [`GatewaySessionBroker`].
//!
//! Builds a minimal `SessionRegistry` with two fake sessions, wires a stub
//! spawn factory, and exercises list / history / send / spawn through the
//! `SessionBroker` trait. The `wait_for_first_reply` branch is skipped
//! because it is time-based and flaky under heavy CI load.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};

use brainwires_agent::ChatAgent;
use brainwires_gateway::sessions_broker::{
    GatewaySessionBroker, SessionRegistry, SessionSpawnFactory,
};
use brainwires_tools::{SessionBroker, SessionId, SessionMessage, SpawnRequest};

struct NoopFactory;

#[async_trait]
impl SessionSpawnFactory for NoopFactory {
    async fn spawn(
        &self,
        _new_session_id: &SessionId,
        _parent: &SessionId,
        _req: &SpawnRequest,
        _on_assistant_reply: Arc<dyn Fn(SessionMessage) + Send + Sync>,
        _inbound_rx: mpsc::UnboundedReceiver<String>,
    ) -> anyhow::Result<Option<Arc<Mutex<ChatAgent>>>> {
        // Register the session but don't materialise a ChatAgent — keeps
        // the test hermetic and decoupled from any real provider.
        Ok(None)
    }
}

async fn seeded_registry() -> SessionRegistry {
    let reg = SessionRegistry::new();
    let _ = reg
        .register(
            SessionId::new("alice"),
            "discord".into(),
            "alice".into(),
            None,
            None,
        )
        .await;
    let _ = reg
        .register(
            SessionId::new("bob"),
            "telegram".into(),
            "bob".into(),
            None,
            None,
        )
        .await;
    reg
}

#[tokio::test]
async fn sessions_list_reports_two_entries() {
    let reg = seeded_registry().await;
    let broker = GatewaySessionBroker::new(reg, Arc::new(NoopFactory));
    let summaries = broker.list().await.expect("list ok");
    assert_eq!(summaries.len(), 2);
    let ids: Vec<&str> = summaries.iter().map(|s| s.id.as_str()).collect();
    assert!(ids.contains(&"alice"));
    assert!(ids.contains(&"bob"));
}

#[tokio::test]
async fn sessions_history_on_empty_returns_empty() {
    let reg = seeded_registry().await;
    let broker = GatewaySessionBroker::new(reg, Arc::new(NoopFactory));
    let msgs = broker
        .history(&SessionId::new("alice"), Some(50))
        .await
        .expect("history ok");
    assert!(msgs.is_empty());
}

#[tokio::test]
async fn sessions_send_pushes_into_inbound_queue() {
    let reg = SessionRegistry::new();
    let mut rx = reg
        .register(
            SessionId::new("alice"),
            "discord".into(),
            "alice".into(),
            None,
            None,
        )
        .await;
    let broker = GatewaySessionBroker::new(reg, Arc::new(NoopFactory));
    broker
        .send(&SessionId::new("alice"), "ping".to_string())
        .await
        .expect("send ok");
    // Fire-and-forget: the message is immediately in the queue.
    let got = rx.try_recv().expect("inbound queue had the message");
    assert_eq!(got, "ping");
}

#[tokio::test]
async fn sessions_spawn_registers_child_with_parent_pointer() {
    let reg = SessionRegistry::new();
    let _ = reg
        .register(
            SessionId::new("parent"),
            "discord".into(),
            "alice".into(),
            None,
            None,
        )
        .await;
    let broker = GatewaySessionBroker::new(reg.clone(), Arc::new(NoopFactory));

    let req = SpawnRequest {
        prompt: "research the openclaw parity gap".into(),
        model: None,
        system: None,
        tools: None,
        // Intentionally skip the wait_for_first_reply branch — timing flake risk.
        wait_for_first_reply: false,
        wait_timeout_secs: 60,
    };
    let spawned = broker
        .spawn(&SessionId::new("parent"), req)
        .await
        .expect("spawn ok");
    assert!(spawned.first_reply.is_none());
    assert!(spawned.id.as_str().starts_with("spawn-"));

    // Parent + child now in the registry.
    assert_eq!(reg.len().await, 2);

    let child = reg.get(&spawned.id).await.expect("child registered");
    assert_eq!(child.parent.as_ref().unwrap().as_str(), "parent");
    assert_eq!(child.channel, "spawned");
    assert!(child.peer.starts_with("spawned-by-"));
}
