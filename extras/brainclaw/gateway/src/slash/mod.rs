//! Slash-command interception for inbound chat messages.
//!
//! The gateway parses messages that begin with `/` into [`SlashCommand`]s
//! before they reach the agent. See [`commands`] for the parser/handler and
//! [`controller`] for the `ChatAgent`-backed `SessionController` impl.

pub mod commands;
pub mod controller;
pub mod session_state;

pub use commands::{
    CompactResult, ParseResult, SessionController, SlashCommand, SlashOutcome, StatusReport,
    TRACE_DEFAULT_TURNS, ThinkLevel, UsageReport, handle, help_entries, parse,
};
pub use controller::AgentSessionHandle;
pub use session_state::SessionSlashState;
