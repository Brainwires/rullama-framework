//! Dream (sleep) commands
//!
//! User-facing slash commands for the background consolidation engine
//! defined in `brainwires_dream`. The framework calls this
//! "dream" rather than "sleep"/"compaction" to signal it's an offline
//! summarise-and-extract pass, not a destructive prefix truncation.

use anyhow::Result;

use super::{CommandAction, CommandExecutor, CommandResult};

impl CommandExecutor {
    /// Execute dream-related commands. Returns `Some` when the name matched.
    pub(super) fn execute_dream_command(
        &self,
        name: &str,
        _args: &[String],
    ) -> Option<Result<CommandResult>> {
        match name {
            "dream" | "dream:status" => Some(Ok(CommandResult::Action(CommandAction::DreamStatus))),
            "dream:run" => Some(Ok(CommandResult::Action(CommandAction::DreamRun))),
            _ => None,
        }
    }
}
