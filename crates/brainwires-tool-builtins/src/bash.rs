use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;
use zeroize::Zeroizing;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Output limiting mode for proactive context management
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// No output limiting
    #[default]
    Full,
    /// Limit to first N lines (head)
    Head,
    /// Limit to last N lines (tail)
    Tail,
    /// Filter output by pattern (grep)
    Filter,
    /// Return only line count
    Count,
    /// Auto-detect best strategy based on command
    Smart,
}

/// Stderr handling mode
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StderrMode {
    /// Keep stdout and stderr separate (default)
    #[default]
    Separate,
    /// Merge stderr into stdout (2>&1)
    Combined,
    /// Only capture stderr, discard stdout
    StderrOnly,
    /// Suppress stderr (2>/dev/null)
    Suppress,
}

/// Output limiting configuration
#[derive(Debug, Clone, Default)]
pub struct OutputLimits {
    /// Maximum number of lines to return
    pub max_lines: Option<u32>,
    /// Output mode (head, tail, filter, etc.)
    pub output_mode: OutputMode,
    /// Pattern for filter mode (grep pattern)
    pub filter_pattern: Option<String>,
    /// How to handle stderr
    pub stderr_mode: StderrMode,
    /// Whether to auto-apply smart limits
    pub auto_limit: bool,
}

/// Absolute byte cap per stream (stdout, stderr). Safety net so a single
/// long line or binary blob can't blow past context limits regardless of
/// line-based output_mode. Picked to roughly match Claude Code's read tool.
const MAX_STREAM_BYTES: usize = 25_000;

/// Global sandbox mode for bash tool invocations.
///
/// Checked at command-build time so every bash tool call goes through the
/// same policy gate regardless of which agent or tool path invoked it. Opt
/// in by setting `BRAINWIRES_BASH_SANDBOX=network-deny` (or via the CLI
/// `--sandbox=network-deny` flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashSandboxMode {
    /// No sandboxing (current default).
    Off,
    /// Install a seccomp-bpf filter on the spawned bash child that denies
    /// network-related syscalls (`socket` for AF_INET/AF_INET6/AF_PACKET/
    /// AF_NETLINK/AF_VSOCK, plus `connect`, `sendto`, `sendmsg`, `sendmmsg`)
    /// with `EPERM`. AF_UNIX sockets remain allowed. Linux-only; on other
    /// platforms this is a no-op that emits a `tracing::warn!`.
    NetworkDeny,
}

impl BashSandboxMode {
    /// Read the active sandbox mode from env. `network-deny` / `networkdeny`
    /// / `1` enables; anything else (including unset) is `Off`.
    pub fn from_env() -> Self {
        match std::env::var("BRAINWIRES_BASH_SANDBOX").as_deref() {
            Ok("network-deny") | Ok("networkdeny") | Ok("1") | Ok("on") => Self::NetworkDeny,
            _ => Self::Off,
        }
    }
}

/// Outcome of resolving a sandbox policy into concrete spawn parameters.
///
/// The command string is unchanged from the caller's input in all cases;
/// enforcement happens in the child's pre-exec hook (see
/// `run_command_with_timeout`) rather than via a shell wrapper. This avoids
/// depending on `unshare -U -r -n`, which is blocked by AppArmor's
/// user-namespaces policy on default-configured Ubuntu 24.04+.
#[derive(Debug, Clone)]
pub(crate) struct SandboxedCommand {
    /// The command string to pass to `bash -c`. Identical to the caller's
    /// input; the seccomp filter operates below the shell layer.
    pub command: String,
    /// When true, install the seccomp network-deny filter on the child
    /// before `exec`. Always `false` on non-Linux.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub seccomp_network_deny: bool,
}

/// Resolve the active sandbox mode into a [`SandboxedCommand`].
///
/// On Linux with `NetworkDeny`, sets `seccomp_network_deny: true` so the
/// spawn site installs the BPF filter in `pre_exec`. On non-Linux with
/// `NetworkDeny`, emits a `tracing::warn!` so operators know sandboxing was
/// requested but not enforced (seccomp is Linux-only).
pub(crate) fn apply_sandbox(command: &str, mode: BashSandboxMode) -> SandboxedCommand {
    match mode {
        BashSandboxMode::Off => SandboxedCommand {
            command: command.to_string(),
            seccomp_network_deny: false,
        },
        BashSandboxMode::NetworkDeny => {
            #[cfg(target_os = "linux")]
            {
                SandboxedCommand {
                    command: command.to_string(),
                    seccomp_network_deny: true,
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                tracing::warn!(
                    target: "brainwires_tool_builtins::bash::sandbox",
                    "network-deny sandbox requested but only Linux seccomp is supported; running unrestricted"
                );
                SandboxedCommand {
                    command: command.to_string(),
                    seccomp_network_deny: false,
                }
            }
        }
    }
}

/// Install a seccomp-bpf filter on the current thread that denies the
/// network syscalls listed on [`BashSandboxMode::NetworkDeny`] with EPERM.
/// Called from a child process' `pre_exec` hook.
///
/// AF_UNIX (domain=1) is intentionally allowed so local IPC (e.g. writing to
/// a Unix socket, syslog) keeps working inside the sandbox. The filter
/// leaves the default action as Allow and enumerates only the narrow set of
/// syscalls we want to block.
#[cfg(target_os = "linux")]
fn install_seccomp_network_deny_filter() -> std::result::Result<(), String> {
    use seccompiler::{
        BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
        SeccompRule,
    };
    use std::convert::TryInto;

    // Address families we want to deny on `socket(2)`. Taken from
    // <bits/socket.h>; leaving AF_UNIX(=1) / AF_LOCAL un-denied.
    //   AF_INET     = 2   (IPv4)
    //   AF_INET6    = 10  (IPv6)
    //   AF_NETLINK  = 16  (kernel-user netlink)
    //   AF_PACKET   = 17  (raw packet / L2)
    //   AF_VSOCK    = 40  (VM sockets)
    const DENIED_DOMAINS: &[u64] = &[2, 10, 16, 17, 40];

    let mut socket_rules: Vec<SeccompRule> = Vec::with_capacity(DENIED_DOMAINS.len());
    for &domain in DENIED_DOMAINS {
        let cond = SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Eq, domain)
            .map_err(|e| format!("seccomp condition: {}", e))?;
        socket_rules
            .push(SeccompRule::new(vec![cond]).map_err(|e| format!("seccomp rule: {}", e))?);
    }

    // `connect`, `sendto`, `sendmsg`, `sendmmsg` get flat denies regardless
    // of address family — an empty rule vec in seccompiler means "match
    // every invocation of this syscall".
    let filter_map: std::collections::BTreeMap<i64, Vec<SeccompRule>> = [
        (libc::SYS_socket, socket_rules),
        (libc::SYS_connect, vec![]),
        (libc::SYS_sendto, vec![]),
        (libc::SYS_sendmsg, vec![]),
        (libc::SYS_sendmmsg, vec![]),
    ]
    .into_iter()
    .collect();

    let filter = SeccompFilter::new(
        filter_map,
        // Default action for any syscall we didn't enumerate: allow.
        SeccompAction::Allow,
        // Action for matched syscalls: return EPERM so userspace sees a
        // normal -EPERM rather than SIGSYS (which would kill bash).
        SeccompAction::Errno(libc::EPERM as u32),
        std::env::consts::ARCH
            .try_into()
            .map_err(|e| format!("seccomp target arch: {:?}", e))?,
    )
    .map_err(|e| format!("seccomp filter: {}", e))?;

    let program: BpfProgram = filter
        .try_into()
        .map_err(|e| format!("seccomp compile: {}", e))?;
    seccompiler::apply_filter(&program).map_err(|e| format!("seccomp apply: {}", e))?;
    Ok(())
}

/// Truncate a stream to at most `max_bytes`, preserving head and tail with
/// an explicit marker in between so the model can reason about the gap.
fn truncate_middle(s: &str, max_bytes: usize) -> std::borrow::Cow<'_, str> {
    if s.len() <= max_bytes {
        return std::borrow::Cow::Borrowed(s);
    }
    let head_bytes = max_bytes / 2;
    let tail_bytes = max_bytes - head_bytes;
    // Clamp head/tail to nearest char boundary to avoid slicing mid-UTF8.
    let mut head_end = head_bytes.min(s.len());
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len().saturating_sub(tail_bytes);
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let skipped = s.len() - head_end - (s.len() - tail_start);
    std::borrow::Cow::Owned(format!(
        "{}\n… [{} bytes truncated] …\n{}",
        &s[..head_end],
        skipped,
        &s[tail_start..],
    ))
}

/// Interactive commands that should be rejected
const INTERACTIVE_COMMANDS: &[&str] = &[
    "vim",
    "vi",
    "nvim",
    "nano",
    "emacs",
    "pico",
    "less",
    "more",
    "most",
    "top",
    "htop",
    "btop",
    "glances",
    "man",
    "info",
    "ssh",
    "telnet",
    "ftp",
    "sftp",
    "python",
    "python3",
    "node",
    "irb",
    "ghci",
    "lua",
    "mysql",
    "psql",
    "sqlite3",
    "mongo",
    "redis-cli",
];

/// Bash execution tool implementation
pub struct BashTool;

impl BashTool {
    /// Get all bash tool definitions
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::execute_command_tool()]
    }

    /// Execute command tool definition
    fn execute_command_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "command".to_string(),
            json!({
                "type": "string",
                "description": "The bash command to execute"
            }),
        );
        properties.insert(
            "timeout".to_string(),
            json!({
                "type": "number",
                "description": "Timeout in seconds (default: 30)",
                "default": 30
            }),
        );
        properties.insert(
            "max_lines".to_string(),
            json!({
                "type": "number",
                "description": "Maximum output lines. Applies head -n or tail -n based on output_mode."
            }),
        );
        properties.insert(
            "output_mode".to_string(),
            json!({
                "type": "string",
                "enum": ["full", "head", "tail", "filter", "count", "smart"],
                "description": "Output limiting mode: full (no limit), head (first N lines), tail (last N lines), filter (grep pattern), count (line count only), smart (auto-detect based on command)",
                "default": "smart"
            }),
        );
        properties.insert(
            "filter_pattern".to_string(),
            json!({
                "type": "string",
                "description": "Grep pattern to filter output (used when output_mode is 'filter')"
            }),
        );
        properties.insert(
            "stderr_mode".to_string(),
            json!({
                "type": "string",
                "enum": ["separate", "combined", "stderr_only", "suppress"],
                "description": "Stderr handling: separate (keep separate), combined (merge with stdout via 2>&1), stderr_only (discard stdout), suppress (discard stderr)",
                "default": "combined"
            }),
        );
        properties.insert(
            "auto_limit".to_string(),
            json!({
                "type": "boolean",
                "description": "Automatically apply smart output limits based on command type (default: true)",
                "default": true
            }),
        );

        Tool {
            name: "execute_command".to_string(),
            description: "Execute a bash command and return the output. Supports proactive output limiting to manage context size.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["command".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    /// Execute a bash command tool
    #[tracing::instrument(name = "tool.execute", skip(input, context), fields(tool_name))]
    pub fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "execute_command" => Self::execute_command(input, context),
            _ => Err(anyhow::anyhow!("Unknown bash tool: {}", tool_name)),
        };

        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Command execution failed: {}", e),
            ),
        }
    }

    fn execute_command(input: &Value, context: &ToolContext) -> Result<String> {
        let params = Self::parse_command_params(input)?;

        if Self::is_interactive_command(&params.command) {
            return Err(anyhow::anyhow!(
                "Interactive command detected: '{}'. Use non-interactive alternatives instead.",
                params
                    .command
                    .split_whitespace()
                    .next()
                    .unwrap_or(&params.command)
            ));
        }

        Self::validate_command(&params.command)?;

        let limits = Self::resolve_output_limits(&params);
        let transformed_command = Self::transform_command(&params.command, &limits);

        let output = Self::run_command_with_timeout(
            &transformed_command,
            &context.working_directory,
            Duration::from_secs(params.timeout),
        )?;

        Self::format_command_output(&params.command, &transformed_command, &output, &limits)
    }

    fn is_interactive_command(command: &str) -> bool {
        let first_word = command.split_whitespace().next().unwrap_or("");
        let effective_command = if first_word == "sudo" || first_word == "env" {
            command.split_whitespace().nth(1).unwrap_or("")
        } else {
            first_word
        };
        INTERACTIVE_COMMANDS.contains(&effective_command)
    }

    fn get_smart_limits(command: &str) -> OutputLimits {
        let cmd_lower = command.to_lowercase();
        let first_word = command.split_whitespace().next().unwrap_or("");

        match first_word {
            "cargo" if cmd_lower.contains("build") => OutputLimits {
                max_lines: Some(80),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "cargo" if cmd_lower.contains("test") => OutputLimits {
                max_lines: Some(100),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "cargo" if cmd_lower.contains("check") => OutputLimits {
                max_lines: Some(60),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "cargo" if cmd_lower.contains("clippy") => OutputLimits {
                max_lines: Some(80),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "npm" | "yarn" | "pnpm" | "bun" => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "make" | "cmake" | "ninja" => OutputLimits {
                max_lines: Some(100),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "go" if cmd_lower.contains("build") || cmd_lower.contains("test") => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                stderr_mode: StderrMode::Combined,
                ..Default::default()
            },
            "find" | "fd" => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "locate" => OutputLimits {
                max_lines: Some(30),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "git" if cmd_lower.contains("log") => OutputLimits {
                max_lines: Some(30),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "git" if cmd_lower.contains("diff") => OutputLimits {
                max_lines: Some(100),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "git" if cmd_lower.contains("status") => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "ps" => OutputLimits {
                max_lines: Some(30),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "docker" if cmd_lower.contains("logs") => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Tail,
                ..Default::default()
            },
            "docker" if cmd_lower.contains("ps") => OutputLimits {
                max_lines: Some(30),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "kubectl" if cmd_lower.contains("logs") => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Tail,
                ..Default::default()
            },
            "kubectl" => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "pm2" if cmd_lower.contains("logs") => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Tail,
                ..Default::default()
            },
            "journalctl" => OutputLimits {
                max_lines: Some(100),
                output_mode: OutputMode::Tail,
                ..Default::default()
            },
            "supervisorctl" if cmd_lower.contains("tail") => OutputLimits {
                max_lines: Some(100),
                output_mode: OutputMode::Tail,
                ..Default::default()
            },
            "ls" => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "tree" => OutputLimits {
                max_lines: Some(80),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            "grep" | "rg" | "ag" | "ack" => OutputLimits {
                max_lines: Some(50),
                output_mode: OutputMode::Head,
                ..Default::default()
            },
            _ => OutputLimits::default(),
        }
    }

    fn handle_streaming_commands(command: &str, limits: &OutputLimits) -> String {
        let cmd_lower = command.to_lowercase();
        let first_word = command.split_whitespace().next().unwrap_or("");
        let lines = limits.max_lines.unwrap_or(50);

        match first_word {
            "pm2" if cmd_lower.contains("logs") && !cmd_lower.contains("--nostream") => {
                if cmd_lower.contains("--lines") {
                    format!("{} --nostream", command)
                } else {
                    format!("{} --nostream --lines {}", command, lines)
                }
            }
            "journalctl" if !cmd_lower.contains("-n ") && !cmd_lower.contains("--lines") => {
                let mut result = command.to_string();
                if !cmd_lower.contains("--no-pager") {
                    result = format!("{} --no-pager", result);
                }
                format!("{} -n {}", result, lines)
            }
            "docker"
                if cmd_lower.contains("logs")
                    && (cmd_lower.contains("-f") || cmd_lower.contains("--follow")) =>
            {
                let cleaned = command
                    .replace(" -f ", " ")
                    .replace(" -f", "")
                    .replace(" --follow ", " ")
                    .replace(" --follow", "");
                if !cleaned.to_lowercase().contains("--tail") {
                    format!("{} --tail {}", cleaned, lines)
                } else {
                    cleaned
                }
            }
            "kubectl"
                if cmd_lower.contains("logs")
                    && (cmd_lower.contains("-f") || cmd_lower.contains("--follow")) =>
            {
                let cleaned = command
                    .replace(" -f ", " ")
                    .replace(" -f", "")
                    .replace(" --follow ", " ")
                    .replace(" --follow", "");
                if !cleaned.to_lowercase().contains("--tail") {
                    format!("{} --tail={}", cleaned, lines)
                } else {
                    cleaned
                }
            }
            _ => command.to_string(),
        }
    }

    fn transform_command(command: &str, limits: &OutputLimits) -> String {
        let mut cmd = Self::handle_streaming_commands(command, limits);

        if cmd == command
            && limits.max_lines.is_none()
            && limits.filter_pattern.is_none()
            && limits.stderr_mode == StderrMode::Separate
            && limits.output_mode == OutputMode::Full
        {
            return command.to_string();
        }

        match limits.stderr_mode {
            StderrMode::Combined => {
                cmd = format!("{} 2>&1", cmd);
            }
            StderrMode::StderrOnly => {
                cmd = format!("{} 2>&1 >/dev/null", cmd);
            }
            StderrMode::Suppress => {
                cmd = format!("{} 2>/dev/null", cmd);
            }
            StderrMode::Separate => {}
        }

        if let Some(pattern) = &limits.filter_pattern {
            let escaped = pattern.replace('\'', "'\\''");
            cmd = format!("{} | grep -E '{}'", cmd, escaped);
        }

        if let Some(n) = limits.max_lines {
            match limits.output_mode {
                OutputMode::Tail => {
                    cmd = format!("{} | tail -n {}", cmd, n);
                }
                OutputMode::Count => {
                    cmd = format!("{} | wc -l", cmd);
                }
                OutputMode::Head | OutputMode::Smart | OutputMode::Full | OutputMode::Filter => {
                    if limits.output_mode != OutputMode::Full {
                        cmd = format!("{} | head -n {}", cmd, n);
                    }
                }
            }
        }

        if cmd != command {
            cmd = format!("set -o pipefail; {}", cmd);
        }

        // Note: sandbox enforcement used to be layered here as a shell
        // wrapper (`unshare -U -r -n …`), but that path is unreliable under
        // AppArmor's user-namespace restrictions. Sandbox mode is now
        // resolved to a [`SandboxedCommand`] at spawn time and applied via
        // a seccomp-bpf filter in the child's pre_exec hook — the shell
        // command text is unchanged.
        cmd
    }

    fn validate_command(command: &str) -> Result<()> {
        let dangerous_patterns = vec![
            "rm -rf /",
            "mkfs",
            "> /dev/sda",
            "dd if=/dev/zero",
            ":(){ :|:& };:",
        ];
        for pattern in dangerous_patterns {
            if command.contains(pattern) {
                return Err(anyhow::anyhow!(
                    "Command contains potentially dangerous pattern: {}",
                    pattern
                ));
            }
        }
        Ok(())
    }

    /// Execute a bash command that requires sudo, piping the password via stdin.
    pub fn execute_with_sudo(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
        password: Zeroizing<String>,
    ) -> ToolResult {
        let result = match tool_name {
            "execute_command" => Self::execute_command_with_sudo(input, context, password),
            _ => Err(anyhow::anyhow!("Unknown bash tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Command execution failed: {}", e),
            ),
        }
    }

    fn execute_command_with_sudo(
        input: &Value,
        context: &ToolContext,
        password: Zeroizing<String>,
    ) -> Result<String> {
        let params = Self::parse_command_params(input)?;
        if Self::is_interactive_command(&params.command) {
            return Err(anyhow::anyhow!(
                "Interactive command detected: '{}'. Use non-interactive alternatives instead.",
                params
                    .command
                    .split_whitespace()
                    .next()
                    .unwrap_or(&params.command)
            ));
        }
        Self::validate_command(&params.command)?;
        let limits = Self::resolve_output_limits(&params);
        let transformed_command = Self::transform_command(&params.command, &limits);
        let output = Self::run_command_with_sudo(
            &transformed_command,
            &context.working_directory,
            password,
        )?;
        Self::format_command_output(&params.command, &transformed_command, &output, &limits)
    }

    fn run_command_with_sudo(
        command: &str,
        working_dir: &str,
        password: Zeroizing<String>,
    ) -> Result<CommandOutput> {
        use std::io::Write;
        use std::process::Stdio;

        let effective_command = command.strip_prefix("sudo ").unwrap_or(command);
        let sudo_command = format!(
            "sudo -S bash -o pipefail -c {}",
            shell_escape(effective_command)
        );

        // NB: sudo's own privilege-elevation path requires netlink + IPC
        // that our seccomp filter would block, so we deliberately do NOT
        // install the network-deny filter when running under sudo. The
        // caller opted into sudo; sandboxing sudo itself needs a different
        // approach (e.g. nftables in the user namespace) and is out of
        // scope for this path.
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(&sudo_command)
            .current_dir(working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn sudo command: {}", command))?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = writeln!(stdin, "{}", password.as_str());
        }
        drop(password);

        let output = child
            .wait_with_output()
            .with_context(|| format!("Failed to wait for sudo command: {}", command))?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        let filtered_stderr = stderr
            .lines()
            .filter(|line| !line.contains("[sudo] password for"))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(CommandOutput {
            stdout,
            stderr: filtered_stderr,
            exit_code,
        })
    }

    fn parse_command_params(input: &Value) -> Result<ParsedCommandParams> {
        #[derive(Deserialize)]
        struct ExecuteCommandInput {
            command: String,
            #[serde(default = "default_timeout")]
            timeout: u64,
            #[serde(default)]
            max_lines: Option<u32>,
            #[serde(default)]
            output_mode: OutputMode,
            #[serde(default)]
            filter_pattern: Option<String>,
            #[serde(default)]
            stderr_mode: StderrMode,
            #[serde(default = "default_auto_limit")]
            auto_limit: bool,
        }
        fn default_timeout() -> u64 {
            30
        }
        fn default_auto_limit() -> bool {
            true
        }

        let raw: ExecuteCommandInput = serde_json::from_value(input.clone())?;
        Ok(ParsedCommandParams {
            command: raw.command,
            timeout: raw.timeout,
            max_lines: raw.max_lines,
            output_mode: raw.output_mode,
            filter_pattern: raw.filter_pattern,
            stderr_mode: raw.stderr_mode,
            auto_limit: raw.auto_limit,
        })
    }

    fn resolve_output_limits(params: &ParsedCommandParams) -> OutputLimits {
        let mut limits = OutputLimits {
            max_lines: params.max_lines,
            output_mode: params.output_mode.clone(),
            filter_pattern: params.filter_pattern.clone(),
            stderr_mode: params.stderr_mode.clone(),
            auto_limit: params.auto_limit,
        };
        if limits.auto_limit && limits.output_mode == OutputMode::Smart {
            let smart_limits = Self::get_smart_limits(&params.command);
            if limits.max_lines.is_none() {
                limits.max_lines = smart_limits.max_lines;
            }
            if limits.output_mode == OutputMode::Smart {
                limits.output_mode = smart_limits.output_mode;
            }
            if limits.stderr_mode == StderrMode::Separate {
                limits.stderr_mode = smart_limits.stderr_mode;
            }
        }
        limits
    }

    fn format_command_output(
        original_command: &str,
        transformed_command: &str,
        output: &CommandOutput,
        limits: &OutputLimits,
    ) -> Result<String> {
        let mut result = format!("Command: {}\n", original_command);
        if transformed_command != original_command {
            result.push_str(&format!("Transformed: {}\n", transformed_command));
        }
        result.push_str(&format!("Exit Code: {}\n\n", output.exit_code));

        let stdout_capped = truncate_middle(&output.stdout, MAX_STREAM_BYTES);
        let stderr_capped = truncate_middle(&output.stderr, MAX_STREAM_BYTES);

        if limits.stderr_mode == StderrMode::Combined
            || limits.stderr_mode == StderrMode::StderrOnly
        {
            result.push_str(&format!("Output:\n{}", stdout_capped));
            if !stderr_capped.is_empty() {
                result.push_str(&format!("\n\nStderr (unmerged):\n{}", stderr_capped));
            }
        } else {
            result.push_str(&format!(
                "Stdout:\n{}\n\nStderr:\n{}",
                stdout_capped, stderr_capped
            ));
        }
        Ok(result)
    }

    fn run_command_with_timeout(
        command: &str,
        working_dir: &str,
        _timeout: Duration,
    ) -> Result<CommandOutput> {
        use std::process::Stdio;

        let sandbox = apply_sandbox(command, BashSandboxMode::from_env());

        let mut cmd = Command::new("bash");
        cmd.arg("-o")
            .arg("pipefail")
            .arg("-c")
            .arg(&sandbox.command)
            .current_dir(working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(target_os = "linux")]
        if sandbox.seccomp_network_deny {
            use std::os::unix::process::CommandExt;
            // SAFETY: pre_exec runs in the child between fork and exec. We
            // only touch the thread's seccomp filter and return; no
            // allocator / async-signal-unsafe calls beyond what seccompiler
            // performs internally (a single prctl + syscall).
            unsafe {
                cmd.pre_exec(|| {
                    install_seccomp_network_deny_filter()
                        .map_err(|e| std::io::Error::other(format!("seccomp: {}", e)))
                });
            }
        }

        let output = cmd
            .output()
            .with_context(|| format!("Failed to execute command: {}", command))?;

        Ok(CommandOutput {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

struct CommandOutput {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

struct ParsedCommandParams {
    command: String,
    timeout: u64,
    max_lines: Option<u32>,
    output_mode: OutputMode,
    filter_pattern: Option<String>,
    stderr_mode: StderrMode,
    auto_limit: bool,
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::env;

    fn create_test_context() -> ToolContext {
        ToolContext {
            working_directory: env::current_dir().unwrap().to_str().unwrap().to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_tools() {
        let tools = BashTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "execute_command");
        assert!(tools[0].requires_approval);
    }

    #[test]
    fn test_execute_simple_command() {
        let context = create_test_context();
        let input = json!({"command": "echo 'Hello World'", "timeout": 5});
        let result = BashTool::execute("bash-123", "execute_command", &input, &context);
        assert!(!result.is_error);
        assert!(result.content.contains("Hello World"));
        assert!(result.content.contains("Exit Code: 0"));
    }

    #[test]
    fn test_validate_command_dangerous_rm() {
        let result = BashTool::validate_command("rm -rf /");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_command_safe() {
        let result = BashTool::validate_command("ls -la");
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_interactive_command() {
        assert!(BashTool::is_interactive_command("vim file.txt"));
        assert!(BashTool::is_interactive_command("sudo vim file.txt"));
        assert!(!BashTool::is_interactive_command("ls -la"));
        assert!(!BashTool::is_interactive_command("cargo build"));
    }

    #[test]
    fn test_smart_limits_cargo_build() {
        let limits = BashTool::get_smart_limits("cargo build");
        assert_eq!(limits.max_lines, Some(80));
        assert_eq!(limits.output_mode, OutputMode::Head);
    }

    #[test]
    fn test_transform_command_no_limits() {
        let limits = OutputLimits::default();
        let result = BashTool::transform_command("echo test", &limits);
        assert_eq!(result, "echo test");
    }

    #[test]
    fn test_transform_command_head_limit() {
        let limits = OutputLimits {
            max_lines: Some(50),
            output_mode: OutputMode::Head,
            ..Default::default()
        };
        let result = BashTool::transform_command("cat file.txt", &limits);
        assert!(result.contains("head -n 50"));
    }

    #[test]
    fn test_truncate_middle_short_input_passthrough() {
        let s = "hello world";
        let got = truncate_middle(s, 100);
        assert_eq!(got.as_ref(), s);
    }

    #[test]
    fn test_truncate_middle_long_input_keeps_head_and_tail() {
        let s = format!("{}{}", "A".repeat(10_000), "Z".repeat(10_000));
        let got = truncate_middle(&s, 1_000);
        assert!(got.len() < s.len());
        assert!(got.contains("truncated"));
        assert!(got.starts_with('A'), "head should be preserved");
        assert!(got.ends_with('Z'), "tail should be preserved");
    }

    #[test]
    fn test_truncate_middle_respects_utf8_boundaries() {
        // Build a string with multi-byte chars straddling the midpoint.
        let s = "é".repeat(1_000); // each é is 2 bytes => 2000 bytes total
        let got = truncate_middle(&s, 100);
        assert!(got.contains("truncated"));
        // Must not panic / produce invalid UTF-8 — if we got here, we're good.
        assert!(!got.is_empty());
    }

    #[test]
    fn test_apply_sandbox_off_is_identity() {
        let got = apply_sandbox("echo hi", BashSandboxMode::Off);
        assert_eq!(got.command, "echo hi");
        assert!(!got.seccomp_network_deny);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_apply_sandbox_network_deny_sets_seccomp_flag_on_linux() {
        // New contract: command text is passed through unchanged, and the
        // sandbox flag signals that the spawn site should install the
        // seccomp filter via pre_exec. The old unshare wrapper is gone.
        let got = apply_sandbox("echo hi", BashSandboxMode::NetworkDeny);
        assert_eq!(got.command, "echo hi");
        assert!(got.seccomp_network_deny);
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn test_apply_sandbox_network_deny_is_noop_on_non_linux() {
        let got = apply_sandbox("echo hi", BashSandboxMode::NetworkDeny);
        assert_eq!(got.command, "echo hi");
        assert!(!got.seccomp_network_deny);
    }

    /// End-to-end: with BRAINWIRES_BASH_SANDBOX=network-deny, `echo` still
    /// runs (no network syscalls) but a TCP connect via bash's `/dev/tcp`
    /// builtin is blocked by the seccomp filter. `/dev/tcp` is resolved
    /// inside bash via `socket(AF_INET, …)` + `connect(…)`, both of which
    /// our filter denies with EPERM.
    ///
    /// Note: this test mutates the process env, so it can't safely run in
    /// parallel with other tests that read BRAINWIRES_BASH_SANDBOX. We
    /// restore the previous value on exit.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_network_deny_blocks_tcp_but_allows_echo() {
        use std::sync::Mutex;
        // Serialize env mutation across the whole test binary.
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();

        let prev = std::env::var("BRAINWIRES_BASH_SANDBOX").ok();
        // SAFETY: guarded by ENV_LOCK; no other thread in this test binary
        // reads this variable while the guard is held.
        unsafe {
            std::env::set_var("BRAINWIRES_BASH_SANDBOX", "network-deny");
        }

        let context = create_test_context();

        let echo_result = BashTool::execute(
            "bash-sandbox-echo",
            "execute_command",
            &json!({"command": "echo sandbox-works", "timeout": 5}),
            &context,
        );
        assert!(
            !echo_result.is_error,
            "echo should pass through the sandbox: {:?}",
            echo_result.content
        );
        assert!(
            echo_result.content.contains("sandbox-works"),
            "expected sandbox-works in output, got: {}",
            echo_result.content
        );

        // The formatter echoes the command back in the "Command:" line, so
        // any token present in the command string will appear in
        // `tcp_result.content` regardless of outcome. To avoid that, we
        // have the success branch mutate process state (exit 7) and the
        // failure branch mutate it differently (exit 0 + stdout marker)
        // so we can distinguish via the Exit Code line + a stdout token
        // that is NOT in the command text.
        //
        // Success: connects, prints nothing, exits 7.
        // Failure (expected under seccomp): bash 'exec' redirect fails,
        //   `||` branch runs, prints our marker on stdout, exits 0.
        let tcp_result = BashTool::execute(
            "bash-sandbox-tcp",
            "execute_command",
            &json!({
                "command": "{ exec 3<>/dev/tcp/1.1.1.1/80 && exit 7 ; } 2>/dev/null || printf 'denied-by-sandbox'",
                "timeout": 5,
                "stderr_mode": "separate",
            }),
            &context,
        );
        assert!(
            tcp_result.content.contains("denied-by-sandbox"),
            "expected tcp connect to be denied; got: {}",
            tcp_result.content
        );
        assert!(
            !tcp_result.content.contains("Exit Code: 7"),
            "tcp connect should NOT have succeeded under network-deny; got: {}",
            tcp_result.content
        );

        // Restore prior env so subsequent tests see the expected value.
        // SAFETY: still holding ENV_LOCK.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BRAINWIRES_BASH_SANDBOX", v),
                None => std::env::remove_var("BRAINWIRES_BASH_SANDBOX"),
            }
        }
    }

    #[test]
    fn test_bash_sandbox_mode_from_env_off_by_default() {
        // Can't mutate env safely in a multi-threaded test runner, so just
        // check the mapping logic with an explicit closure equivalent.
        // (see from_env implementation)
        // This mainly guards against a refactor that breaks the default.
        match std::env::var("BRAINWIRES_BASH_SANDBOX").as_deref() {
            Ok(_) => {} // test env set it — skip
            Err(_) => assert_eq!(BashSandboxMode::from_env(), BashSandboxMode::Off),
        }
    }

    #[test]
    fn test_format_command_output_applies_byte_cap() {
        let big = "x".repeat(MAX_STREAM_BYTES * 2);
        let output = CommandOutput {
            stdout: big,
            stderr: String::new(),
            exit_code: 0,
        };
        let limits = OutputLimits {
            stderr_mode: StderrMode::Combined,
            ..Default::default()
        };
        let formatted =
            BashTool::format_command_output("cat huge.bin", "cat huge.bin", &output, &limits)
                .unwrap();
        // Formatted output must be shorter than the raw stdout AND contain the
        // truncation marker.
        assert!(formatted.len() < MAX_STREAM_BYTES * 2 + 200);
        assert!(formatted.contains("truncated"));
    }
}
