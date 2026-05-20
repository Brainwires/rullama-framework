//! [`SessionsTool`] — bundles `sessions_list`, `sessions_history`,
//! `sessions_send`, and `sessions_spawn` over a [`SessionBroker`].

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

use brainwires_stores::SessionId;
use brainwires_stores::session::broker::{SessionBroker, SpawnRequest};

/// Tool name for the list-sessions tool.
pub const TOOL_SESSIONS_LIST: &str = "sessions_list";
/// Tool name for the session-history tool.
pub const TOOL_SESSIONS_HISTORY: &str = "sessions_history";
/// Tool name for the send-to-session tool.
pub const TOOL_SESSIONS_SEND: &str = "sessions_send";
/// Tool name for the spawn-session tool.
pub const TOOL_SESSIONS_SPAWN: &str = "sessions_spawn";

/// Metadata key the host may set on [`ToolContext::metadata`] to carry the
/// caller's own session id. Read by the spawn tool so the new session's
/// `parent` pointer is correct.
pub const CTX_METADATA_SESSION_ID: &str = "session_id";

/// Maximum number of messages `sessions_history` will return in a single call.
/// Protects the agent's context window against pathological transcripts.
pub const MAX_HISTORY_LIMIT: usize = 500;
const DEFAULT_HISTORY_LIMIT: usize = 50;

/// Bundle of four session-control tools, all backed by a single
/// [`SessionBroker`].
///
/// Construct one per agent session so the tool call sites know which session
/// is "self" (for the `sessions_send` recursion check and for `sessions_spawn`
/// parent-pointer wiring when the executor does not plumb the session id
/// through [`ToolContext::metadata`]).
pub struct SessionsTool {
    broker: Arc<dyn SessionBroker>,
    /// Caller's own session id. Consulted when [`ToolContext::metadata`]
    /// does not contain `session_id`; `None` means the host did not tell us
    /// — the spawn/self-send checks then fall back to returning a clear
    /// error in the tool result.
    current_session_id: Option<SessionId>,
}

impl SessionsTool {
    /// Construct a new `SessionsTool`.
    pub fn new(broker: Arc<dyn SessionBroker>, current_session_id: Option<SessionId>) -> Self {
        Self {
            broker,
            current_session_id,
        }
    }

    /// Return the four tool definitions this bundle exposes to the LLM.
    pub fn get_tools() -> Vec<Tool> {
        vec![
            Self::list_tool(),
            Self::history_tool(),
            Self::send_tool(),
            Self::spawn_tool(),
        ]
    }

    // ── Tool schemas ────────────────────────────────────────────────────

    fn list_tool() -> Tool {
        Tool {
            name: TOOL_SESSIONS_LIST.to_string(),
            description:
                "List every live chat session currently managed by the host — including the \
                 caller's own session and any sessions the caller (or its peers) have spawned. \
                 Use this to discover session ids before calling sessions_history or sessions_send. \
                 Returns a JSON array of session summaries (id, channel, peer, timestamps, \
                 message_count, optional parent)."
                    .to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn history_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "session_id".to_string(),
            json!({
                "type": "string",
                "description": "The target session id (from sessions_list)."
            }),
        );
        props.insert(
            "limit".to_string(),
            json!({
                "type": "number",
                "description": format!(
                    "Max messages to return (default {DEFAULT_HISTORY_LIMIT}, \
                     hard-capped at {MAX_HISTORY_LIMIT})."
                ),
            }),
        );
        Tool {
            name: TOOL_SESSIONS_HISTORY.to_string(),
            description: "Return a target session's recent transcript as a JSON array of \
                 {role, content, timestamp} objects (newest last). Use this to catch up \
                 on what a spawned sub-session has produced, or to read another user's \
                 ongoing conversation before intervening."
                .to_string(),
            input_schema: ToolInputSchema::object(props, vec!["session_id".to_string()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn send_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "session_id".to_string(),
            json!({
                "type": "string",
                "description": "Target session id. Must not equal the caller's own session \
                                (self-send is rejected to prevent recursion)."
            }),
        );
        props.insert(
            "text".to_string(),
            json!({
                "type": "string",
                "description": "The user-role message to inject into the target session's \
                                inbound queue."
            }),
        );
        Tool {
            name: TOOL_SESSIONS_SEND.to_string(),
            description:
                "Inject a user-role message into another session's inbound queue. Fire-and-forget: \
                 returns {\"ok\": true} as soon as the message is queued; the target session \
                 processes it asynchronously. Use this to nudge a spawned sub-session, relay \
                 information between two user sessions, or ask a peer session a follow-up \
                 question."
                    .to_string(),
            input_schema: ToolInputSchema::object(
                props,
                vec!["session_id".to_string(), "text".to_string()],
            ),
            // Sending on behalf of the agent into another live session is a
            // cross-user side effect; the host should gate it with its
            // normal approval policy.
            requires_approval: true,
            ..Default::default()
        }
    }

    fn spawn_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "prompt".to_string(),
            json!({
                "type": "string",
                "description": "Initial user message to seed the new session with."
            }),
        );
        props.insert(
            "model".to_string(),
            json!({
                "type": "string",
                "description": "Optional model override (e.g. 'claude-opus-4-7'). Omit to inherit from parent."
            }),
        );
        props.insert(
            "system".to_string(),
            json!({
                "type": "string",
                "description": "Optional system prompt for the sub-session. Omit to inherit."
            }),
        );
        props.insert(
            "tools".to_string(),
            json!({
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional allow-list of tool names the sub-session may invoke. Omit to inherit the parent's toolset."
            }),
        );
        props.insert(
            "wait_for_first_reply".to_string(),
            json!({
                "type": "boolean",
                "description": "If true, block this tool call until the sub-session produces \
                                its first assistant message (or wait_timeout_secs elapses). \
                                Default false — return immediately with just the session id.",
                "default": false
            }),
        );
        props.insert(
            "wait_timeout_secs".to_string(),
            json!({
                "type": "number",
                "description": "Seconds to wait when wait_for_first_reply is true (default 60).",
                "default": 60
            }),
        );

        Tool {
            name: TOOL_SESSIONS_SPAWN.to_string(),
            description:
                "Spawn a new chat sub-session as a child of the current session, seeded with \
                 `prompt`. Returns {session_id, first_reply?}. Use this to delegate a focused \
                 task (e.g. 'spawn a research sub-session and return in 5m') — the parent can \
                 later inspect progress via sessions_history or push updates via sessions_send."
                    .to_string(),
            input_schema: ToolInputSchema::object(props, vec!["prompt".to_string()]),
            requires_approval: true,
            ..Default::default()
        }
    }

    // ── Execution ───────────────────────────────────────────────────────

    /// Dispatch a tool call by name. Returns a [`ToolResult`] (never errors
    /// out to an `anyhow::Result`; broker failures become `ToolResult::error`
    /// so the LLM sees them as tool output rather than an executor crash).
    pub async fn execute(
        &self,
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        match tool_name {
            TOOL_SESSIONS_LIST => self.exec_list(tool_use_id).await,
            TOOL_SESSIONS_HISTORY => self.exec_history(tool_use_id, input).await,
            TOOL_SESSIONS_SEND => self.exec_send(tool_use_id, input, context).await,
            TOOL_SESSIONS_SPAWN => self.exec_spawn(tool_use_id, input, context).await,
            other => ToolResult::error(
                tool_use_id.to_string(),
                format!("Unknown sessions tool: {other}"),
            ),
        }
    }

    async fn exec_list(&self, tool_use_id: &str) -> ToolResult {
        match self.broker.list().await {
            Ok(summaries) => match serde_json::to_string(&summaries) {
                Ok(body) => ToolResult::success(tool_use_id.to_string(), body),
                Err(e) => ToolResult::error(
                    tool_use_id.to_string(),
                    format!("Failed to serialize session list: {e}"),
                ),
            },
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("sessions_list failed: {e}"),
            ),
        }
    }

    async fn exec_history(&self, tool_use_id: &str, input: &Value) -> ToolResult {
        #[derive(Deserialize)]
        struct In {
            session_id: Option<String>,
            #[serde(default)]
            limit: Option<usize>,
        }
        let raw: In = match serde_json::from_value(input.clone()) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    format!("Invalid sessions_history input: {e}"),
                );
            }
        };
        let sid = match raw.session_id.filter(|s| !s.is_empty()) {
            Some(s) => SessionId(s),
            None => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    "sessions_history requires a non-empty `session_id`".to_string(),
                );
            }
        };
        let limit = Some(
            raw.limit
                .unwrap_or(DEFAULT_HISTORY_LIMIT)
                .min(MAX_HISTORY_LIMIT),
        );
        match self.broker.history(&sid, limit).await {
            Ok(msgs) => match serde_json::to_string(&msgs) {
                Ok(body) => ToolResult::success(tool_use_id.to_string(), body),
                Err(e) => ToolResult::error(
                    tool_use_id.to_string(),
                    format!("Failed to serialize session history: {e}"),
                ),
            },
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("sessions_history failed: {e}"),
            ),
        }
    }

    async fn exec_send(
        &self,
        tool_use_id: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        #[derive(Deserialize)]
        struct In {
            session_id: Option<String>,
            text: Option<String>,
        }
        let raw: In = match serde_json::from_value(input.clone()) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    format!("Invalid sessions_send input: {e}"),
                );
            }
        };
        let sid = match raw.session_id.filter(|s| !s.is_empty()) {
            Some(s) => SessionId(s),
            None => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    "sessions_send requires a non-empty `session_id`".to_string(),
                );
            }
        };
        let text = match raw.text {
            Some(t) if !t.is_empty() => t,
            _ => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    "sessions_send requires a non-empty `text`".to_string(),
                );
            }
        };

        if let Some(self_id) = self.resolve_current_session_id(context)
            && self_id == sid
        {
            return ToolResult::error(
                tool_use_id.to_string(),
                "sessions_send cannot target the caller's own session — that would recurse. \
                 Use a spawned sub-session id, or address a peer session from sessions_list."
                    .to_string(),
            );
        }

        match self.broker.send(&sid, text).await {
            Ok(()) => ToolResult::success(tool_use_id.to_string(), json!({"ok": true}).to_string()),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("sessions_send failed: {e}"),
            ),
        }
    }

    async fn exec_spawn(
        &self,
        tool_use_id: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let req: SpawnRequest = match serde_json::from_value(input.clone()) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    format!("Invalid sessions_spawn input: {e}"),
                );
            }
        };
        if req.prompt.is_empty() {
            return ToolResult::error(
                tool_use_id.to_string(),
                "sessions_spawn requires a non-empty `prompt`".to_string(),
            );
        }
        let parent = match self.resolve_current_session_id(context) {
            Some(id) => id,
            None => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    "sessions_spawn could not determine the caller's session id — \
                     host must set ToolContext::metadata[\"session_id\"] or pass \
                     current_session_id into SessionsTool::new."
                        .to_string(),
                );
            }
        };

        match self.broker.spawn(&parent, req).await {
            Ok(spawned) => match serde_json::to_string(&spawned) {
                Ok(body) => ToolResult::success(tool_use_id.to_string(), body),
                Err(e) => ToolResult::error(
                    tool_use_id.to_string(),
                    format!("Failed to serialize spawned session: {e}"),
                ),
            },
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("sessions_spawn failed: {e}"),
            ),
        }
    }

    fn resolve_current_session_id(&self, context: &ToolContext) -> Option<SessionId> {
        context
            .metadata
            .get(CTX_METADATA_SESSION_ID)
            .filter(|s| !s.is_empty())
            .map(|s| SessionId(s.clone()))
            .or_else(|| self.current_session_id.clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use brainwires_stores::session::broker::{SessionMessage, SessionSummary, SpawnedSession};
    use chrono::{TimeZone, Utc};
    use std::sync::Mutex;

    /// Hand-rolled test double — no mocking framework.
    struct MockBroker {
        list_ret: Mutex<Vec<SessionSummary>>,
        history_ret: Mutex<Vec<SessionMessage>>,
        // Captured inputs on each call.
        history_calls: Mutex<Vec<(SessionId, Option<usize>)>>,
        send_calls: Mutex<Vec<(SessionId, String)>>,
        spawn_calls: Mutex<Vec<(SessionId, SpawnRequest)>>,
        spawn_ret: Mutex<Option<SpawnedSession>>,
    }

    impl MockBroker {
        fn new() -> Self {
            Self {
                list_ret: Mutex::new(Vec::new()),
                history_ret: Mutex::new(Vec::new()),
                history_calls: Mutex::new(Vec::new()),
                send_calls: Mutex::new(Vec::new()),
                spawn_calls: Mutex::new(Vec::new()),
                spawn_ret: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl SessionBroker for MockBroker {
        async fn list(&self) -> anyhow::Result<Vec<SessionSummary>> {
            Ok(self.list_ret.lock().unwrap().clone())
        }

        async fn history(
            &self,
            id: &SessionId,
            limit: Option<usize>,
        ) -> anyhow::Result<Vec<SessionMessage>> {
            self.history_calls.lock().unwrap().push((id.clone(), limit));
            Ok(self.history_ret.lock().unwrap().clone())
        }

        async fn send(&self, id: &SessionId, text: String) -> anyhow::Result<()> {
            self.send_calls.lock().unwrap().push((id.clone(), text));
            Ok(())
        }

        async fn spawn(
            &self,
            parent: &SessionId,
            req: SpawnRequest,
        ) -> anyhow::Result<SpawnedSession> {
            self.spawn_calls
                .lock()
                .unwrap()
                .push((parent.clone(), req.clone()));
            Ok(self
                .spawn_ret
                .lock()
                .unwrap()
                .clone()
                .unwrap_or(SpawnedSession {
                    id: SessionId("spawned-1".into()),
                    first_reply: None,
                }))
        }
    }

    fn fixed_ts() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, 19, 12, 0, 0).unwrap()
    }

    fn ctx_with_session(session: &str) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.metadata
            .insert(CTX_METADATA_SESSION_ID.to_string(), session.to_string());
        ctx
    }

    #[test]
    fn list_tool_schema_shape() {
        let tools = SessionsTool::get_tools();
        let list = tools
            .iter()
            .find(|t| t.name == TOOL_SESSIONS_LIST)
            .expect("list tool present");
        // No required inputs.
        let required = list.input_schema.required.clone().unwrap_or_default();
        assert!(
            required.is_empty(),
            "sessions_list must have no required inputs, got {required:?}"
        );
        assert!(!list.description.is_empty());
        // Four tools exposed.
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&TOOL_SESSIONS_LIST));
        assert!(names.contains(&TOOL_SESSIONS_HISTORY));
        assert!(names.contains(&TOOL_SESSIONS_SEND));
        assert!(names.contains(&TOOL_SESSIONS_SPAWN));
    }

    #[tokio::test]
    async fn history_tool_rejects_missing_session_id() {
        let broker = Arc::new(MockBroker::new());
        let tool = SessionsTool::new(broker.clone(), Some(SessionId("self".into())));
        let ctx = ctx_with_session("self");
        let result = tool
            .execute("call-1", TOOL_SESSIONS_HISTORY, &json!({}), &ctx)
            .await;
        assert!(result.is_error, "expected error result, got {result:?}");
        assert!(
            result.content.to_lowercase().contains("session_id"),
            "error should mention session_id, got: {}",
            result.content
        );
        assert!(
            broker.history_calls.lock().unwrap().is_empty(),
            "broker must not be called for invalid input"
        );
    }

    #[tokio::test]
    async fn history_tool_clamps_limit() {
        let broker = Arc::new(MockBroker::new());
        broker.history_ret.lock().unwrap().push(SessionMessage {
            role: "user".into(),
            content: "hi".into(),
            timestamp: fixed_ts(),
        });
        let tool = SessionsTool::new(broker.clone(), Some(SessionId("self".into())));
        let ctx = ctx_with_session("self");
        let input = json!({"session_id": "target", "limit": 9999});
        let result = tool
            .execute("call-1", TOOL_SESSIONS_HISTORY, &input, &ctx)
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.content);
        let calls = broker.history_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, SessionId("target".into()));
        assert_eq!(
            calls[0].1,
            Some(MAX_HISTORY_LIMIT),
            "limit must be clamped to MAX_HISTORY_LIMIT ({MAX_HISTORY_LIMIT})",
        );
    }

    #[tokio::test]
    async fn send_tool_self_send_rejected() {
        let broker = Arc::new(MockBroker::new());
        let tool = SessionsTool::new(broker.clone(), Some(SessionId("me".into())));
        let ctx = ctx_with_session("me");
        let input = json!({"session_id": "me", "text": "hello"});
        let result = tool
            .execute("call-1", TOOL_SESSIONS_SEND, &input, &ctx)
            .await;
        assert!(result.is_error);
        assert!(
            result.content.to_lowercase().contains("recurs"),
            "error should mention recursion, got: {}",
            result.content
        );
        assert!(broker.send_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn send_tool_forwards_to_broker_when_distinct() {
        let broker = Arc::new(MockBroker::new());
        let tool = SessionsTool::new(broker.clone(), Some(SessionId("me".into())));
        let ctx = ctx_with_session("me");
        let input = json!({"session_id": "peer", "text": "ping"});
        let result = tool
            .execute("call-1", TOOL_SESSIONS_SEND, &input, &ctx)
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.content);
        let calls = broker.send_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, SessionId("peer".into()));
        assert_eq!(calls[0].1, "ping");
        assert!(result.content.contains("\"ok\""));
    }

    #[tokio::test]
    async fn spawn_tool_passes_through() {
        let broker = Arc::new(MockBroker::new());
        let tool = SessionsTool::new(broker.clone(), Some(SessionId("parent".into())));
        let ctx = ctx_with_session("parent");
        let input = json!({
            "prompt": "research the openclaw parity gap",
            "model": "claude-opus-4-7",
            "system": "you are a research agent",
            "tools": ["fetch_url", "query_codebase"],
            "wait_for_first_reply": true,
            "wait_timeout_secs": 30u64,
        });
        let result = tool
            .execute("call-1", TOOL_SESSIONS_SPAWN, &input, &ctx)
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.content);
        let calls = broker.spawn_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, SessionId("parent".into()));
        let req = &calls[0].1;
        assert_eq!(req.prompt, "research the openclaw parity gap");
        assert_eq!(req.model.as_deref(), Some("claude-opus-4-7"));
        assert_eq!(req.system.as_deref(), Some("you are a research agent"));
        assert_eq!(
            req.tools.as_deref(),
            Some(["fetch_url".to_string(), "query_codebase".to_string()].as_slice())
        );
        assert!(req.wait_for_first_reply);
        assert_eq!(req.wait_timeout_secs, 30);
    }

    #[tokio::test]
    async fn spawn_tool_errors_without_parent_session() {
        let broker = Arc::new(MockBroker::new());
        // No metadata, no current_session_id → parent unknown.
        let tool = SessionsTool::new(broker.clone(), None);
        let ctx = ToolContext::default();
        let input = json!({"prompt": "x"});
        let result = tool
            .execute("call-1", TOOL_SESSIONS_SPAWN, &input, &ctx)
            .await;
        assert!(result.is_error);
        assert!(
            result.content.contains("session") && result.content.to_lowercase().contains("caller")
        );
        assert!(broker.spawn_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_returns_json_array() {
        let broker = Arc::new(MockBroker::new());
        broker.list_ret.lock().unwrap().push(SessionSummary {
            id: SessionId("s1".into()),
            channel: "discord".into(),
            peer: "alice".into(),
            created_at: fixed_ts(),
            last_active: fixed_ts(),
            message_count: 3,
            parent: None,
        });
        let tool = SessionsTool::new(broker.clone(), Some(SessionId("me".into())));
        let ctx = ctx_with_session("me");
        let result = tool
            .execute("c1", TOOL_SESSIONS_LIST, &json!({}), &ctx)
            .await;
        assert!(!result.is_error);
        assert!(result.content.starts_with('['));
        assert!(result.content.contains("\"s1\""));
        assert!(result.content.contains("\"discord\""));
    }
}
