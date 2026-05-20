//! Session-control tools exposed to the agent.
//!
//! The agent uses these to inspect or orchestrate *other* chat sessions
//! running in the same host process — listing peers, reading their history,
//! pushing a message into one, or spawning a fresh sub-session (e.g. for a
//! research task it wants to delegate and poll later).
//!
//! Session state lives outside this crate (in the gateway, typically), so
//! this module only defines the tool schemas plus a [`SessionBroker`] trait
//! that the host implements over its actual registry.

// SessionBroker / SessionId / SessionMessage / SessionSummary / SpawnRequest /
// SpawnedSession live in `brainwires-session::broker`. Depend on that crate
// directly — there is no re-export shim here.

mod sessions_tool;

pub use sessions_tool::{
    SessionsTool, TOOL_SESSIONS_HISTORY, TOOL_SESSIONS_LIST, TOOL_SESSIONS_SEND,
    TOOL_SESSIONS_SPAWN,
};
