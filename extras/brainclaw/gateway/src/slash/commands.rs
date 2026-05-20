//! Slash-command parsing and handling.
//!
//! The gateway intercepts messages that begin with `/` before they reach the
//! agent. A leading `\/` escape lets users send a literal `/` message to the
//! agent (the backslash is stripped).

use async_trait::async_trait;

use super::session_state::SessionSlashState;

/// Extended-thinking budget level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkLevel {
    Off,
    Low,
    Medium,
    High,
}

impl ThinkLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkLevel::Off => "off",
            ThinkLevel::Low => "low",
            ThinkLevel::Medium => "medium",
            ThinkLevel::High => "high",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Some(ThinkLevel::Off),
            "low" => Some(ThinkLevel::Low),
            "medium" | "med" => Some(ThinkLevel::Medium),
            "high" => Some(ThinkLevel::High),
            _ => None,
        }
    }
}

/// Parsed slash command ready to be handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    New,
    Compact,
    Think(ThinkLevel),
    /// `/think` with no argument or an invalid argument; carries the raw arg.
    ThinkInvalid(String),
    Usage,
    Trace(bool),
    /// `/trace` with no argument or an invalid argument; carries the raw arg.
    TraceInvalid(String),
    Status,
    Restart,
    Help,
    /// First whitespace-separated token after the slash, lowercased.
    Unknown(String),
}

/// Result of parsing an inbound message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseResult {
    /// Not a slash command (or escaped with `\/`): forward this text to the agent.
    Forward(String),
    /// Parsed slash command ready for [`handle`].
    Command(SlashCommand),
}

/// Outcome of handling a slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashOutcome {
    /// Reply text to send back to the user via the originating channel.
    Reply(String),
    /// Forward this (unescaped) text to the agent instead of replying directly.
    Forward(String),
}

/// Reports collected from the underlying agent session.
#[derive(Debug, Clone, Default)]
pub struct UsageReport {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct CompactResult {
    pub messages_before: usize,
    pub messages_after: usize,
}

#[derive(Debug, Clone)]
pub struct StatusReport {
    pub provider: String,
    pub model: String,
    pub session_id: String,
    pub message_count: usize,
    pub think_level: ThinkLevel,
    pub trace_remaining: u32,
    pub channels_connected: Vec<String>,
}

/// Operations the slash-command handler invokes on the owning session.
#[async_trait]
pub trait SessionController: Send + Sync {
    async fn reset_session(&mut self) -> anyhow::Result<()>;
    async fn compact_session(&mut self) -> anyhow::Result<CompactResult>;
    async fn usage_report(&self) -> anyhow::Result<UsageReport>;
    async fn status_report(&self) -> StatusReport;
    async fn restart_session(&mut self) -> anyhow::Result<()>;
    /// Apply the requested thinking level. Returns `false` if the underlying
    /// provider doesn't support extended thinking at all.
    fn set_think_level(&mut self, level: ThinkLevel) -> bool;
}

/// Parse an inbound message into either a slash command or forward-through text.
pub fn parse(raw: &str) -> ParseResult {
    // Escape hatch: `\/foo` → forward `/foo` to the agent as literal content.
    if let Some(rest) = raw.strip_prefix("\\/") {
        return ParseResult::Forward(format!("/{rest}"));
    }

    let trimmed = raw.trim_start();
    if !trimmed.starts_with('/') {
        return ParseResult::Forward(raw.to_string());
    }

    let body = &trimmed[1..];
    let mut parts = body.split_whitespace();
    let name = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg1 = parts.next().unwrap_or("");

    let cmd = match name.as_str() {
        "" => SlashCommand::Unknown(String::new()),
        "new" | "reset" => SlashCommand::New,
        "compact" => SlashCommand::Compact,
        "usage" => SlashCommand::Usage,
        "status" => SlashCommand::Status,
        "restart" => SlashCommand::Restart,
        "help" => SlashCommand::Help,
        "think" => match ThinkLevel::parse(arg1) {
            Some(level) => SlashCommand::Think(level),
            None => SlashCommand::ThinkInvalid(arg1.to_string()),
        },
        "trace" => match arg1.to_ascii_lowercase().as_str() {
            "on" | "enable" | "true" => SlashCommand::Trace(true),
            "off" | "disable" | "false" => SlashCommand::Trace(false),
            _ => SlashCommand::TraceInvalid(arg1.to_string()),
        },
        other => SlashCommand::Unknown(other.to_string()),
    };

    ParseResult::Command(cmd)
}

/// Default number of agent turns `/trace on` keeps tracing active for.
pub const TRACE_DEFAULT_TURNS: u32 = 5;

/// List of commands exposed by `/help` and the admin help endpoint.
pub fn help_entries() -> &'static [(&'static str, &'static str)] {
    &[
        ("/new", "Reset the current session (alias: /reset)."),
        ("/reset", "Reset the current session (alias: /new)."),
        (
            "/compact",
            "Summarise history into a single system message.",
        ),
        (
            "/think <low|medium|high|off>",
            "Set extended-thinking budget.",
        ),
        ("/usage", "Show current session token usage."),
        (
            "/trace <on|off>",
            "Toggle verbose tracing for next 5 turns.",
        ),
        ("/status", "Show provider, model, session id, counters."),
        ("/restart", "Tear down and rebuild the session."),
        ("/help", "List available slash commands."),
    ]
}

fn format_help() -> String {
    let mut out = String::from("**Available commands**\n");
    for (cmd, desc) in help_entries() {
        out.push_str(&format!("- `{cmd}` — {desc}\n"));
    }
    out
}

fn format_usage(report: &UsageReport) -> String {
    let mut out = String::from("**Session usage**\n");
    out.push_str(&format!("- Input tokens: {}\n", report.input_tokens));
    out.push_str(&format!("- Output tokens: {}\n", report.output_tokens));
    if report.cache_read > 0 || report.cache_write > 0 {
        out.push_str(&format!(
            "- Cache read / write: {} / {}\n",
            report.cache_read, report.cache_write
        ));
    }
    if let Some(cost) = report.cost_usd {
        out.push_str(&format!("- Estimated cost: ${cost:.4} USD\n"));
    }
    out
}

fn format_status(s: &StatusReport) -> String {
    let channels = if s.channels_connected.is_empty() {
        "(none)".to_string()
    } else {
        s.channels_connected.join(", ")
    };
    format!(
        "**Session status**\n\
         - Provider: `{}`\n\
         - Model: `{}`\n\
         - Session id: `{}`\n\
         - Messages: {}\n\
         - Thinking: {}\n\
         - Trace remaining: {}\n\
         - Connected channels: {}\n",
        s.provider,
        s.model,
        s.session_id,
        s.message_count,
        s.think_level.as_str(),
        s.trace_remaining,
        channels,
    )
}

/// Execute a parsed command. Returns a reply to send back to the user, or a
/// text to forward to the agent (currently unused but reserved for future
/// pass-through commands).
pub async fn handle<C: SessionController + ?Sized>(
    cmd: SlashCommand,
    session: &mut SessionSlashState,
    controller: &mut C,
) -> SlashOutcome {
    match cmd {
        SlashCommand::New => match controller.reset_session().await {
            Ok(()) => {
                *session = SessionSlashState::default();
                tracing::info!(command = "new", "slash command handled");
                SlashOutcome::Reply("Session reset.".to_string())
            }
            Err(e) => SlashOutcome::Reply(format!("Failed to reset session: {e}")),
        },
        SlashCommand::Compact => match controller.compact_session().await {
            Ok(result) => {
                tracing::info!(
                    command = "compact",
                    before = result.messages_before,
                    after = result.messages_after,
                    "slash command handled"
                );
                SlashOutcome::Reply(format!(
                    "Compacted {} messages into summary ({} remaining).",
                    result.messages_before.saturating_sub(result.messages_after),
                    result.messages_after,
                ))
            }
            Err(e) => SlashOutcome::Reply(format!("Failed to compact session: {e}")),
        },
        SlashCommand::Think(level) => {
            let supported = controller.set_think_level(level);
            if supported {
                session.think_level = level;
                tracing::info!(
                    command = "think",
                    level = level.as_str(),
                    "slash command handled"
                );
                SlashOutcome::Reply(format!("Thinking level set to {}.", level.as_str()))
            } else {
                SlashOutcome::Reply("Thinking not supported by current provider.".to_string())
            }
        }
        SlashCommand::ThinkInvalid(arg) => SlashOutcome::Reply(format!(
            "Unknown thinking level `{arg}`. Use one of: low, medium, high, off.",
        )),
        SlashCommand::Usage => match controller.usage_report().await {
            Ok(report) => {
                tracing::info!(command = "usage", "slash command handled");
                SlashOutcome::Reply(format_usage(&report))
            }
            Err(e) => SlashOutcome::Reply(format!("Failed to read usage: {e}")),
        },
        SlashCommand::Trace(enable) => {
            if enable {
                session.trace_remaining = TRACE_DEFAULT_TURNS;
                tracing::info!(
                    command = "trace",
                    enable = true,
                    turns = TRACE_DEFAULT_TURNS,
                    "slash command handled"
                );
                SlashOutcome::Reply(format!(
                    "Trace enabled for next {TRACE_DEFAULT_TURNS} turns.",
                ))
            } else {
                session.trace_remaining = 0;
                tracing::info!(command = "trace", enable = false, "slash command handled");
                SlashOutcome::Reply("Trace disabled.".to_string())
            }
        }
        SlashCommand::TraceInvalid(arg) => {
            SlashOutcome::Reply(format!("Unknown trace value `{arg}`. Use `on` or `off`.",))
        }
        SlashCommand::Status => {
            let mut report = controller.status_report().await;
            report.think_level = session.think_level;
            report.trace_remaining = session.trace_remaining;
            tracing::info!(command = "status", "slash command handled");
            SlashOutcome::Reply(format_status(&report))
        }
        SlashCommand::Restart => match controller.restart_session().await {
            Ok(()) => {
                *session = SessionSlashState::default();
                tracing::info!(command = "restart", "slash command handled");
                SlashOutcome::Reply("Session restarted.".to_string())
            }
            Err(e) => SlashOutcome::Reply(format!("Failed to restart session: {e}")),
        },
        SlashCommand::Help => {
            tracing::info!(command = "help", "slash command handled");
            SlashOutcome::Reply(format_help())
        }
        SlashCommand::Unknown(name) => {
            let cmd = if name.is_empty() {
                "/".to_string()
            } else {
                format!("/{name}")
            };
            SlashOutcome::Reply(format!("Unknown command: {cmd}. Try /help."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    };

    struct MockController {
        reset_called: AtomicBool,
        compact_called: AtomicBool,
        restart_called: AtomicBool,
        think_called: AtomicU32,
        last_think: std::sync::Mutex<Option<ThinkLevel>>,
        think_supported: bool,
        canned_usage: UsageReport,
        canned_status: StatusReport,
        messages_before: usize,
        messages_after: usize,
    }

    #[async_trait]
    impl SessionController for MockController {
        async fn reset_session(&mut self) -> anyhow::Result<()> {
            self.reset_called.store(true, Ordering::SeqCst);
            Ok(())
        }
        async fn compact_session(&mut self) -> anyhow::Result<CompactResult> {
            self.compact_called.store(true, Ordering::SeqCst);
            Ok(CompactResult {
                messages_before: self.messages_before,
                messages_after: self.messages_after,
            })
        }
        async fn usage_report(&self) -> anyhow::Result<UsageReport> {
            Ok(self.canned_usage.clone())
        }
        async fn status_report(&self) -> StatusReport {
            self.canned_status.clone()
        }
        async fn restart_session(&mut self) -> anyhow::Result<()> {
            self.restart_called.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn set_think_level(&mut self, level: ThinkLevel) -> bool {
            self.think_called.fetch_add(1, Ordering::SeqCst);
            *self.last_think.lock().unwrap() = Some(level);
            self.think_supported
        }
    }

    fn default_status() -> StatusReport {
        StatusReport {
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            session_id: "sess-123".to_string(),
            message_count: 4,
            think_level: ThinkLevel::Off,
            trace_remaining: 0,
            channels_connected: vec!["discord".to_string()],
        }
    }

    fn new_controller(think_supported: bool) -> MockController {
        MockController {
            reset_called: AtomicBool::new(false),
            compact_called: AtomicBool::new(false),
            restart_called: AtomicBool::new(false),
            think_called: AtomicU32::new(0),
            last_think: std::sync::Mutex::new(None),
            think_supported,
            canned_usage: UsageReport::default(),
            canned_status: default_status(),
            messages_before: 0,
            messages_after: 0,
        }
    }

    #[test]
    fn parse_escapes_backslash_slash() {
        assert_eq!(parse("\\/new"), ParseResult::Forward("/new".to_string()));
        assert_eq!(
            parse("\\/hello world"),
            ParseResult::Forward("/hello world".to_string())
        );
    }

    #[test]
    fn parse_non_slash_is_forwarded() {
        assert_eq!(
            parse("hello world"),
            ParseResult::Forward("hello world".to_string())
        );
        assert_eq!(parse(""), ParseResult::Forward(String::new()));
    }

    #[test]
    fn parse_unknown_command() {
        assert_eq!(
            parse("/foo"),
            ParseResult::Command(SlashCommand::Unknown("foo".to_string()))
        );
        assert_eq!(
            parse("/"),
            ParseResult::Command(SlashCommand::Unknown(String::new()))
        );
    }

    #[test]
    fn parse_think_levels() {
        assert_eq!(
            parse("/think low"),
            ParseResult::Command(SlashCommand::Think(ThinkLevel::Low))
        );
        assert_eq!(
            parse("/think MEDIUM"),
            ParseResult::Command(SlashCommand::Think(ThinkLevel::Medium))
        );
        assert_eq!(
            parse("/think high"),
            ParseResult::Command(SlashCommand::Think(ThinkLevel::High))
        );
        assert_eq!(
            parse("/think off"),
            ParseResult::Command(SlashCommand::Think(ThinkLevel::Off))
        );
        assert_eq!(
            parse("/think bogus"),
            ParseResult::Command(SlashCommand::ThinkInvalid("bogus".to_string()))
        );
        assert_eq!(
            parse("/think"),
            ParseResult::Command(SlashCommand::ThinkInvalid(String::new()))
        );
    }

    #[test]
    fn parse_trace() {
        assert_eq!(
            parse("/trace on"),
            ParseResult::Command(SlashCommand::Trace(true))
        );
        assert_eq!(
            parse("/trace OFF"),
            ParseResult::Command(SlashCommand::Trace(false))
        );
        assert_eq!(
            parse("/trace maybe"),
            ParseResult::Command(SlashCommand::TraceInvalid("maybe".to_string()))
        );
    }

    #[test]
    fn parse_case_insensitive_cmd_name() {
        assert_eq!(parse("/NEW"), ParseResult::Command(SlashCommand::New));
        assert_eq!(parse("/Reset"), ParseResult::Command(SlashCommand::New));
        assert_eq!(parse("/HELP"), ParseResult::Command(SlashCommand::Help));
    }

    #[tokio::test]
    async fn handle_reset_calls_controller() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        let outcome = handle(SlashCommand::New, &mut state, &mut ctrl).await;
        assert_eq!(outcome, SlashOutcome::Reply("Session reset.".to_string()));
        assert!(ctrl.reset_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn handle_unknown_replies_with_help_hint() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        let outcome = handle(
            SlashCommand::Unknown("nope".to_string()),
            &mut state,
            &mut ctrl,
        )
        .await;
        match outcome {
            SlashOutcome::Reply(reply) => {
                assert!(reply.contains("/help"));
                assert!(reply.contains("/nope"));
            }
            other => panic!("expected Reply, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handle_think_unsupported_provider() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(false);
        let outcome = handle(SlashCommand::Think(ThinkLevel::High), &mut state, &mut ctrl).await;
        match outcome {
            SlashOutcome::Reply(reply) => {
                assert!(reply.to_lowercase().contains("not supported"));
            }
            other => panic!("expected Reply, got {other:?}"),
        }
        assert_eq!(state.think_level, ThinkLevel::Off);
    }

    #[tokio::test]
    async fn handle_think_supported_updates_state() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        let outcome = handle(SlashCommand::Think(ThinkLevel::High), &mut state, &mut ctrl).await;
        assert_eq!(
            outcome,
            SlashOutcome::Reply("Thinking level set to high.".to_string())
        );
        assert_eq!(state.think_level, ThinkLevel::High);
    }

    #[tokio::test]
    async fn handle_usage_formats_markdown() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        ctrl.canned_usage = UsageReport {
            input_tokens: 1234,
            output_tokens: 5678,
            cache_read: 10,
            cache_write: 20,
            cost_usd: Some(0.123456),
        };
        let outcome = handle(SlashCommand::Usage, &mut state, &mut ctrl).await;
        match outcome {
            SlashOutcome::Reply(reply) => {
                assert!(reply.contains("1234"));
                assert!(reply.contains("5678"));
                assert!(reply.contains("0.12"));
                assert!(reply.contains("Session usage"));
            }
            other => panic!("expected Reply, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handle_status_includes_required_fields() {
        let mut state = SessionSlashState {
            think_level: ThinkLevel::Low,
            trace_remaining: 3,
        };
        let mut ctrl = new_controller(true);
        let outcome = handle(SlashCommand::Status, &mut state, &mut ctrl).await;
        match outcome {
            SlashOutcome::Reply(reply) => {
                assert!(reply.contains("mock"));
                assert!(reply.contains("mock-model"));
                assert!(reply.contains("sess-123"));
                assert!(reply.contains("Messages: 4"));
                assert!(reply.contains("low"));
                assert!(reply.contains("3"));
                assert!(reply.contains("discord"));
            }
            other => panic!("expected Reply, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handle_compact_reports_counts() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        ctrl.messages_before = 40;
        ctrl.messages_after = 2;
        let outcome = handle(SlashCommand::Compact, &mut state, &mut ctrl).await;
        match outcome {
            SlashOutcome::Reply(reply) => {
                assert!(reply.contains("Compacted"));
                assert!(reply.contains("38"));
            }
            other => panic!("expected Reply, got {other:?}"),
        }
        assert!(ctrl.compact_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn handle_trace_on_sets_remaining() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        let outcome = handle(SlashCommand::Trace(true), &mut state, &mut ctrl).await;
        match outcome {
            SlashOutcome::Reply(reply) => assert!(reply.contains("Trace enabled")),
            other => panic!("expected Reply, got {other:?}"),
        }
        assert_eq!(state.trace_remaining, TRACE_DEFAULT_TURNS);
    }

    #[tokio::test]
    async fn handle_trace_off_clears_remaining() {
        let mut state = SessionSlashState {
            think_level: ThinkLevel::Off,
            trace_remaining: 10,
        };
        let mut ctrl = new_controller(true);
        let _ = handle(SlashCommand::Trace(false), &mut state, &mut ctrl).await;
        assert_eq!(state.trace_remaining, 0);
    }

    #[tokio::test]
    async fn handle_help_lists_all_commands() {
        let mut state = SessionSlashState::default();
        let mut ctrl = new_controller(true);
        let outcome = handle(SlashCommand::Help, &mut state, &mut ctrl).await;
        match outcome {
            SlashOutcome::Reply(reply) => {
                for (cmd, _) in help_entries() {
                    assert!(reply.contains(cmd), "help output missing {cmd}");
                }
            }
            other => panic!("expected Reply, got {other:?}"),
        }
    }

    // Silence unused-import lint on the Arc type in this test module.
    #[allow(dead_code)]
    fn _keep_arc_used() -> Arc<()> {
        Arc::new(())
    }
}
