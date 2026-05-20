//! Per-session state tracked for slash-command handling.

use super::commands::ThinkLevel;

/// Mutable, per-user state owned by the gateway and mutated by slash-command
/// handlers. Lives in memory for the lifetime of the gateway; reset implicitly
/// on process restart.
#[derive(Debug, Clone)]
pub struct SessionSlashState {
    /// Extended-thinking budget requested for this session.
    pub think_level: ThinkLevel,
    /// Number of remaining agent turns for which verbose tracing is active.
    /// Decremented after each agent turn; zero means trace is off.
    pub trace_remaining: u32,
}

impl Default for SessionSlashState {
    fn default() -> Self {
        Self {
            think_level: ThinkLevel::Off,
            trace_remaining: 0,
        }
    }
}
