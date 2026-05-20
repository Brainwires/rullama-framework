//! Gateway-side [`SessionBroker`] implementation.
//!
//! The agent (inside `ChatAgent`) invokes `sessions_list` / `sessions_history`
//! / `sessions_send` / `sessions_spawn` tools. The tool implementation in
//! `brainwires-tools` is a framework crate — it cannot know about the
//! gateway's session registry, so it calls through the abstract
//! [`brainwires_tools::SessionBroker`] trait. This module is the host-side
//! implementation over the gateway's real `ChatAgent` sessions.
//!
//! # Shape of the registry
//!
//! We keep a small, dedicated registry (`Arc<RwLock<HashMap<_, _>>>`) rather
//! than bolting extra indices onto the existing `(platform, user_id) ->
//! Arc<Mutex<ChatAgent>>` map in `agent_handler.rs`. This gives us:
//!
//! * a single canonical [`SessionId`] per session (agnostic to channel),
//! * parent/child pointers for spawned sessions,
//! * an inbound mpsc queue per session so `sessions_send` is fire-and-forget
//!   without touching the channel adapters.
//!
//! The broker does *not* itself construct `ChatAgent`s — that requires
//! references to the gateway's provider, executor, default chat options, etc.
//! The gateway supplies a [`SessionSpawnFactory`] trait object so the broker
//! can remain agent-agnostic and trivially unit-testable.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};

use anyhow::Result;
use brainwires_agent::ChatAgent;
use brainwires_core::{ContentBlock, MessageContent, Role, Tool, ToolContext, ToolResult, ToolUse};
use brainwires_tools::{
    SessionBroker, SessionId, SessionMessage, SessionSummary, SessionsTool, SpawnRequest,
    SpawnedSession, ToolExecutor,
};

/// Per-session record held in the broker's registry.
pub struct SessionHandle {
    /// Stable session id (what [`SessionBroker`] methods take).
    pub id: SessionId,
    /// Originating channel name — `"discord"`, `"web"`, `"spawned"`, etc.
    pub channel: String,
    /// Peer handle — remote user id, or `"spawned-by-<parent>"`.
    pub peer: String,
    /// When the session first appeared in the registry.
    pub created_at: DateTime<Utc>,
    /// Last time a message was pushed or pulled.
    pub last_active: Arc<RwLock<DateTime<Utc>>>,
    /// Parent session, if this was spawned via `sessions_spawn`.
    pub parent: Option<SessionId>,
    /// The live agent. `None` for test doubles / spawn-factory failures that
    /// don't actually materialise a ChatAgent.
    pub agent: Option<Arc<Mutex<ChatAgent>>>,
    /// Inbound queue used by `sessions_send`. Drained by whoever owns the
    /// session (typically the spawn driver / gateway's message loop).
    pub inbound_tx: mpsc::UnboundedSender<String>,
    /// Outbound one-shot listener used to implement
    /// `SpawnRequest::wait_for_first_reply`. Set at spawn time when the
    /// caller asked to wait; fired when the spawn driver produces the first
    /// assistant message.
    pub first_reply_tx: Arc<Mutex<Option<oneshot::Sender<SessionMessage>>>>,
}

/// Factory trait for constructing the [`ChatAgent`] behind a spawned session.
///
/// The gateway implements this by capturing its shared `provider`, `executor`,
/// and `default_options` at `AgentInboundHandler` construction time. Kept
/// abstract so the broker can be unit-tested with a stub factory that just
/// returns `Ok(None)`.
#[async_trait]
pub trait SessionSpawnFactory: Send + Sync {
    /// Materialise a new chat agent for `new_session_id`, seeded with
    /// `req.prompt` as the first user message.
    ///
    /// The factory MUST push `req.prompt` into the returned agent's
    /// conversation and start processing it in a background task. Any
    /// assistant reply(s) it produces should be forwarded to
    /// `on_assistant_reply` so the broker can satisfy
    /// [`SpawnRequest::wait_for_first_reply`].
    ///
    /// `inbound_rx` is the receiver side of the new session's inbound mpsc
    /// queue. The factory takes ownership so it can drive the spawned
    /// ChatAgent with any follow-up `sessions_send` messages the parent
    /// pushes after the initial prompt — without the factory draining this
    /// queue, `sessions_send` on a spawned session would silently pile up.
    ///
    /// Returning `Ok(None)` means "session registered but no ChatAgent
    /// materialised" — used by test doubles and in production when the
    /// gateway simply records the intent without actually running. The
    /// `inbound_rx` is dropped in that case.
    async fn spawn(
        &self,
        new_session_id: &SessionId,
        parent: &SessionId,
        req: &SpawnRequest,
        on_assistant_reply: Arc<dyn Fn(SessionMessage) + Send + Sync>,
        inbound_rx: mpsc::UnboundedReceiver<String>,
    ) -> anyhow::Result<Option<Arc<Mutex<ChatAgent>>>>;
}

/// Shared in-memory registry of sessions. Cheap to clone.
#[derive(Clone, Default)]
pub struct SessionRegistry {
    inner: Arc<RwLock<HashMap<SessionId, Arc<SessionHandle>>>>,
}

impl SessionRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a session. Returns the inbound `Receiver` so the caller
    /// can drain messages pushed via `sessions_send` and forward them into
    /// whatever actually drives the agent (usually `ChatAgent::process_message`).
    pub async fn register(
        &self,
        id: SessionId,
        channel: String,
        peer: String,
        parent: Option<SessionId>,
        agent: Option<Arc<Mutex<ChatAgent>>>,
    ) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        let now = Utc::now();
        let handle = Arc::new(SessionHandle {
            id: id.clone(),
            channel,
            peer,
            created_at: now,
            last_active: Arc::new(RwLock::new(now)),
            parent,
            agent,
            inbound_tx: tx,
            first_reply_tx: Arc::new(Mutex::new(None)),
        });
        self.inner.write().await.insert(id, handle);
        rx
    }

    /// Remove a session from the registry.
    pub async fn unregister(&self, id: &SessionId) {
        self.inner.write().await.remove(id);
    }

    /// Fetch a session handle by id.
    pub async fn get(&self, id: &SessionId) -> Option<Arc<SessionHandle>> {
        self.inner.read().await.get(id).cloned()
    }

    /// Number of sessions currently registered.
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Whether the registry is empty.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }

    /// Bump `last_active` on a registered session (no-op if unknown).
    pub async fn touch(&self, id: &SessionId) {
        if let Some(handle) = self.get(id).await {
            *handle.last_active.write().await = Utc::now();
        }
    }
}

impl SessionHandle {
    /// Update `last_active` to the current UTC timestamp.
    pub async fn touch(&self) {
        *self.last_active.write().await = Utc::now();
    }
}

/// The concrete [`SessionBroker`] the gateway hands to `SessionsTool`.
pub struct GatewaySessionBroker {
    registry: SessionRegistry,
    spawn_factory: Arc<dyn SessionSpawnFactory>,
}

impl GatewaySessionBroker {
    /// Construct a new broker over `registry`, using `spawn_factory` to
    /// materialise `ChatAgent`s for `sessions_spawn`.
    pub fn new(registry: SessionRegistry, spawn_factory: Arc<dyn SessionSpawnFactory>) -> Self {
        Self {
            registry,
            spawn_factory,
        }
    }

    /// Generate a fresh session id for spawned sessions. Uses `uuid::Uuid`
    /// for uniqueness with a `spawn-` prefix so logs stay grep-friendly.
    fn new_spawned_id() -> SessionId {
        SessionId::new(format!("spawn-{}", uuid::Uuid::new_v4()))
    }
}

#[async_trait]
impl SessionBroker for GatewaySessionBroker {
    async fn list(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let map = self.registry.inner.read().await;
        let mut out = Vec::with_capacity(map.len());
        for (id, handle) in map.iter() {
            let last_active = *handle.last_active.read().await;
            let message_count = if let Some(ref agent) = handle.agent {
                agent.try_lock().map(|g| g.message_count()).unwrap_or(0)
            } else {
                0
            };
            out.push(SessionSummary {
                id: id.clone(),
                channel: handle.channel.clone(),
                peer: handle.peer.clone(),
                created_at: handle.created_at,
                last_active,
                message_count,
                parent: handle.parent.clone(),
            });
        }
        Ok(out)
    }

    async fn history(
        &self,
        id: &SessionId,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<SessionMessage>> {
        let handle = self
            .registry
            .get(id)
            .await
            .ok_or_else(|| anyhow::anyhow!("unknown session: {id}"))?;
        let Some(agent) = handle.agent.clone() else {
            return Ok(Vec::new());
        };
        let agent = agent.lock().await;
        let messages = agent.messages();
        // NOTE: ChatAgent does not currently track per-message timestamps,
        // so we fall back to Utc::now() for every entry. Callers that need
        // monotonic timestamps should tag messages at push time at a higher
        // layer. (Tracked as a follow-up — see TODO below.)
        let fallback_ts = Utc::now();
        let take = limit.unwrap_or(messages.len()).min(messages.len());
        let start = messages.len().saturating_sub(take);
        let mut out = Vec::with_capacity(take);
        for msg in &messages[start..] {
            out.push(SessionMessage {
                role: role_to_string(&msg.role),
                content: content_to_string(&msg.content),
                timestamp: fallback_ts,
            });
        }
        Ok(out)
    }

    async fn send(&self, id: &SessionId, text: String) -> anyhow::Result<()> {
        let handle = self
            .registry
            .get(id)
            .await
            .ok_or_else(|| anyhow::anyhow!("unknown session: {id}"))?;
        // Fire-and-forget: bounce off the mpsc unbounded queue. The session
        // driver (spawn factory or gateway message loop) drains it
        // asynchronously — we do NOT await a reply here.
        handle
            .inbound_tx
            .send(text)
            .map_err(|e| anyhow::anyhow!("session {id} inbound channel closed: {e}"))?;
        *handle.last_active.write().await = Utc::now();
        Ok(())
    }

    async fn spawn(&self, parent: &SessionId, req: SpawnRequest) -> anyhow::Result<SpawnedSession> {
        // 1. Mint the new id up front so we can return it synchronously
        //    regardless of `wait_for_first_reply`.
        let new_id = Self::new_spawned_id();

        // 2. Install the one-shot first-reply listener if the caller asked
        //    us to block on the first assistant message. We do this BEFORE
        //    registering the handle so there's no race where the factory
        //    fires a reply before the listener is armed.
        let (first_tx, first_rx) = if req.wait_for_first_reply {
            let (tx, rx) = oneshot::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // 3. Register the session *without* an agent — the factory hands us
        //    one in a moment. The returned inbound receiver is passed to
        //    the factory so it can drain follow-up `sessions_send` messages
        //    after the initial prompt is processed.
        let inbound_rx = self
            .registry
            .register(
                new_id.clone(),
                "spawned".to_string(),
                format!("spawned-by-{parent}"),
                Some(parent.clone()),
                None,
            )
            .await;

        // 4. Stash the first-reply sender on the handle so the factory can
        //    fire it when its agent produces its first assistant message.
        if let Some(tx) = first_tx
            && let Some(handle) = self.registry.get(&new_id).await
        {
            *handle.first_reply_tx.lock().await = Some(tx);
        }

        // 5. Build the `on_assistant_reply` callback. It pulls the latest
        //    first-reply sender off the handle and fires it (once).
        let registry = self.registry.clone();
        let new_id_for_cb = new_id.clone();
        let on_reply: Arc<dyn Fn(SessionMessage) + Send + Sync> = Arc::new(move |msg| {
            let registry = registry.clone();
            let id = new_id_for_cb.clone();
            tokio::spawn(async move {
                if let Some(handle) = registry.get(&id).await {
                    *handle.last_active.write().await = Utc::now();
                    let mut slot = handle.first_reply_tx.lock().await;
                    if let Some(tx) = slot.take() {
                        let _ = tx.send(msg);
                    }
                }
            });
        });

        // 6. Invoke the factory. It may attach an agent to the handle; we
        //    surface factory errors as spawn failures but leave the handle
        //    registered so `sessions_list` still shows the attempted spawn.
        let agent = self
            .spawn_factory
            .spawn(&new_id, parent, &req, on_reply, inbound_rx)
            .await?;

        if let Some(agent) = agent {
            // Re-insert the handle with the agent attached. We do a
            // minimal swap here rather than bolting interior mutability
            // onto SessionHandle.
            if let Some(existing) = self.registry.get(&new_id).await {
                let replaced = Arc::new(SessionHandle {
                    id: existing.id.clone(),
                    channel: existing.channel.clone(),
                    peer: existing.peer.clone(),
                    created_at: existing.created_at,
                    last_active: existing.last_active.clone(),
                    parent: existing.parent.clone(),
                    agent: Some(agent),
                    inbound_tx: existing.inbound_tx.clone(),
                    first_reply_tx: existing.first_reply_tx.clone(),
                });
                self.registry
                    .inner
                    .write()
                    .await
                    .insert(new_id.clone(), replaced);
            }
        }

        // 7. If the caller wanted the first reply, wait on the one-shot.
        let first_reply = if let Some(rx) = first_rx {
            let timeout = std::time::Duration::from_secs(req.wait_timeout_secs);
            match tokio::time::timeout(timeout, rx).await {
                Ok(Ok(msg)) => Some(msg),
                // Timeout or sender dropped — return None, not an error;
                // the caller still got a valid session id they can poll.
                _ => None,
            }
        } else {
            None
        };

        Ok(SpawnedSession {
            id: new_id,
            first_reply,
        })
    }
}

fn role_to_string(role: &Role) -> String {
    match role {
        Role::User => "user".into(),
        Role::Assistant => "assistant".into(),
        Role::System => "system".into(),
        Role::Tool => "tool".into(),
    }
}

fn content_to_string(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => {
            let mut out = String::new();
            for (i, block) in blocks.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                match block {
                    ContentBlock::Text { text } => out.push_str(text),
                    ContentBlock::ToolUse { name, input, .. } => {
                        out.push_str(&format!("[tool_use: {name} input={input}]"));
                    }
                    ContentBlock::ToolResult {
                        content, is_error, ..
                    } => {
                        if *is_error == Some(true) {
                            out.push_str(&format!("[tool_error: {content}]"));
                        } else {
                            out.push_str(&format!("[tool_result: {content}]"));
                        }
                    }
                    ContentBlock::Image { .. } => out.push_str("[image]"),
                }
            }
            out
        }
    }
}

/// [`ToolExecutor`] that composes a [`SessionsTool`] on top of an inner
/// executor. The four session tools (`sessions_list`, `sessions_history`,
/// `sessions_send`, `sessions_spawn`) are handled in-crate; every other tool
/// name falls through to the inner executor unchanged.
///
/// Construct one per agent session so the "self" session id is correct for
/// the recursion check in `sessions_send` and the parent-pointer wiring in
/// `sessions_spawn`.
pub struct SessionsExecutor {
    inner: Arc<dyn ToolExecutor>,
    sessions: Arc<SessionsTool>,
    /// Tool names intercepted by this executor.
    intercepted: Vec<String>,
}

impl SessionsExecutor {
    /// Wrap `inner` with a [`SessionsTool`] bound to `broker` and
    /// `current_session_id` (the caller's own session).
    pub fn new(
        inner: Arc<dyn ToolExecutor>,
        broker: Arc<dyn SessionBroker>,
        current_session_id: Option<SessionId>,
    ) -> Self {
        let sessions = Arc::new(SessionsTool::new(broker, current_session_id));
        Self {
            inner,
            sessions,
            intercepted: vec![
                brainwires_tools::sessions::TOOL_SESSIONS_LIST.to_string(),
                brainwires_tools::sessions::TOOL_SESSIONS_HISTORY.to_string(),
                brainwires_tools::sessions::TOOL_SESSIONS_SEND.to_string(),
                brainwires_tools::sessions::TOOL_SESSIONS_SPAWN.to_string(),
            ],
        }
    }
}

#[async_trait]
impl ToolExecutor for SessionsExecutor {
    async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> Result<ToolResult> {
        if self.intercepted.iter().any(|n| n == &tool_use.name) {
            Ok(self
                .sessions
                .execute(&tool_use.id, &tool_use.name, &tool_use.input, context)
                .await)
        } else {
            self.inner.execute(tool_use, context).await
        }
    }

    fn available_tools(&self) -> Vec<Tool> {
        let mut tools = self.inner.available_tools();
        tools.extend(SessionsTool::get_tools());
        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubFactory;

    #[async_trait]
    impl SessionSpawnFactory for StubFactory {
        async fn spawn(
            &self,
            _new_session_id: &SessionId,
            _parent: &SessionId,
            _req: &SpawnRequest,
            _on_assistant_reply: Arc<dyn Fn(SessionMessage) + Send + Sync>,
            _inbound_rx: mpsc::UnboundedReceiver<String>,
        ) -> anyhow::Result<Option<Arc<Mutex<ChatAgent>>>> {
            // No actual ChatAgent construction — the session is registered
            // but has no live agent. Good enough for broker-layer unit
            // tests; the integration test exercises the plumbing.
            Ok(None)
        }
    }

    #[tokio::test]
    async fn list_returns_registered_sessions() {
        let reg = SessionRegistry::new();
        let _ = reg
            .register(
                SessionId::new("a"),
                "discord".into(),
                "alice".into(),
                None,
                None,
            )
            .await;
        let _ = reg
            .register(
                SessionId::new("b"),
                "telegram".into(),
                "bob".into(),
                None,
                None,
            )
            .await;
        let broker = GatewaySessionBroker::new(reg, Arc::new(StubFactory));
        let sessions = broker.list().await.unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn history_on_empty_agent_returns_empty() {
        let reg = SessionRegistry::new();
        let _ = reg
            .register(
                SessionId::new("a"),
                "discord".into(),
                "alice".into(),
                None,
                None,
            )
            .await;
        let broker = GatewaySessionBroker::new(reg, Arc::new(StubFactory));
        let msgs = broker.history(&SessionId::new("a"), None).await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn history_unknown_session_errors() {
        let reg = SessionRegistry::new();
        let broker = GatewaySessionBroker::new(reg, Arc::new(StubFactory));
        let err = broker
            .history(&SessionId::new("ghost"), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown session"));
    }

    #[tokio::test]
    async fn send_pushes_into_inbound_queue() {
        let reg = SessionRegistry::new();
        let mut rx = reg
            .register(
                SessionId::new("a"),
                "discord".into(),
                "alice".into(),
                None,
                None,
            )
            .await;
        let broker = GatewaySessionBroker::new(reg, Arc::new(StubFactory));
        broker
            .send(&SessionId::new("a"), "hello".to_string())
            .await
            .unwrap();
        // Fire-and-forget: the message is already in the queue before send() returns.
        let got = rx.try_recv().unwrap();
        assert_eq!(got, "hello");
    }

    #[tokio::test]
    async fn spawn_registers_child_with_parent() {
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
        let broker = GatewaySessionBroker::new(reg.clone(), Arc::new(StubFactory));
        let req = SpawnRequest {
            prompt: "do research".into(),
            ..Default::default()
        };
        let spawned = broker.spawn(&SessionId::new("parent"), req).await.unwrap();
        assert!(spawned.id.as_str().starts_with("spawn-"));
        assert!(spawned.first_reply.is_none());
        // Registry should now have two sessions, the child pointing at parent.
        assert_eq!(reg.len().await, 2);
        let child = reg.get(&spawned.id).await.unwrap();
        assert_eq!(child.parent.as_ref().unwrap().as_str(), "parent");
        assert_eq!(child.channel, "spawned");
    }
}
