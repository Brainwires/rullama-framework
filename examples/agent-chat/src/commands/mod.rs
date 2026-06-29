pub mod auth_cmd;
pub mod config_cmd;
pub mod models_cmd;

/// A parsed in-session slash command, issued from the plain REPL or TUI.
///
/// The REPL and TUI each handle these inline today; this parser extracts the
/// shared recognition logic so it can be tested and reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    /// `/help` — show available commands.
    Help,
    /// `/clear` — reset conversation history.
    Clear,
    /// `/exit` or `/quit` — leave the session.
    Exit,
    /// `/model <name>` — switch the active model.
    Model(String),
    /// An unrecognized slash command. Retains the original token (without the
    /// leading slash) so callers can echo a helpful error.
    Unknown(String),
}

/// Parse a single trimmed input line.
///
/// Returns `None` when the input is not a slash command (i.e. does not begin
/// with `/`). Recognized commands map to their `SlashCommand` variant;
/// unknown `/foo` strings produce `SlashCommand::Unknown("foo")`.
pub fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    let input = input.trim();
    let rest = input.strip_prefix('/')?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let head = parts.next().unwrap_or("");
    let tail = parts.next().unwrap_or("").trim();

    let cmd = match head {
        "help" => SlashCommand::Help,
        "clear" => SlashCommand::Clear,
        "exit" | "quit" => SlashCommand::Exit,
        "model" => SlashCommand::Model(tail.to_string()),
        other => SlashCommand::Unknown(other.to_string()),
    };
    Some(cmd)
}
