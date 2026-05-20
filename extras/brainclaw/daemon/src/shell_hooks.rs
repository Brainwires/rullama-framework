//! User-configurable shell hooks — expose the framework's lifecycle and
//! pre-tool-execute hooks to shell scripts the user configures in
//! `brainclaw.toml` under `[hooks]`.
//!
//! ## How it works
//!
//! When a lifecycle event fires, `ShellHookRunner` serialises it to JSON and
//! passes it to the configured script via `stdin`.  The script may do anything
//! it likes — log, notify, call an external API, etc.
//!
//! **Pre-tool hooks** are special: if the script exits with a non-zero code the
//! tool call is **cancelled**.  The first line of `stdout` is used as the
//! rejection reason sent back to the agent.
//!
//! ## Config
//!
//! ```toml
//! [hooks]
//! # Runs before every tool execution.  Exit non-zero to block the call.
//! pre_tool_use  = "~/.brainclaw/hooks/pre-tool.sh"
//! # Runs after every tool execution (informational; exit code ignored).
//! post_tool_use = "~/.brainclaw/hooks/post-tool.sh"
//! # Runs when an agent session starts.
//! session_start = "~/.brainclaw/hooks/session-start.sh"
//! # Runs when an agent session ends (completed or failed).
//! session_end   = "~/.brainclaw/hooks/session-end.sh"
//! ```
//!
//! The JSON sent on stdin matches the [`LifecycleEvent`] schema; it always
//! includes at least `{ "type": "...", ... }`.

use std::process::Stdio;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::ToolContext;
use brainwires_core::ToolUse;
use brainwires_core::lifecycle::{HookResult, LifecycleEvent, LifecycleHook};
use brainwires_tools::{PreHookDecision, ToolPreHook};
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::config::HooksSection;

// ── helpers ─────────────────────────────────────────────────────────────────

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().into_owned();
    }
    path.to_string()
}

/// Serialize a `LifecycleEvent` into a JSON `Value` suitable for passing to
/// a shell script via stdin.
fn event_to_json(event: &LifecycleEvent) -> serde_json::Value {
    match event {
        LifecycleEvent::AgentStarted {
            agent_id,
            task_description,
        } => json!({
            "type": "agent_started",
            "agent_id": agent_id,
            "task_description": task_description,
        }),
        LifecycleEvent::AgentCompleted {
            agent_id,
            iterations,
            summary,
        } => json!({
            "type": "agent_completed",
            "agent_id": agent_id,
            "iterations": iterations,
            "summary": summary,
        }),
        LifecycleEvent::AgentFailed {
            agent_id,
            error,
            iterations,
        } => json!({
            "type": "agent_failed",
            "agent_id": agent_id,
            "error": error,
            "iterations": iterations,
        }),
        LifecycleEvent::ToolBeforeExecute {
            agent_id,
            tool_name,
            args,
        } => json!({
            "type": "tool_before_execute",
            "agent_id": agent_id,
            "tool_name": tool_name,
            "args": args,
        }),
        LifecycleEvent::ToolAfterExecute {
            agent_id,
            tool_name,
            success,
            duration_ms,
        } => json!({
            "type": "tool_after_execute",
            "agent_id": agent_id,
            "tool_name": tool_name,
            "success": success,
            "duration_ms": duration_ms,
        }),
        LifecycleEvent::ProviderRequest {
            agent_id,
            provider,
            model,
        } => json!({
            "type": "provider_request",
            "agent_id": agent_id,
            "provider": provider,
            "model": model,
        }),
        LifecycleEvent::ProviderResponse {
            agent_id,
            provider,
            model,
            input_tokens,
            output_tokens,
            duration_ms,
        } => json!({
            "type": "provider_response",
            "agent_id": agent_id,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_ms": duration_ms,
        }),
        LifecycleEvent::ValidationStarted { agent_id, checks } => json!({
            "type": "validation_started",
            "agent_id": agent_id,
            "checks": checks,
        }),
        LifecycleEvent::ValidationCompleted {
            agent_id,
            passed,
            issues,
        } => json!({
            "type": "validation_completed",
            "agent_id": agent_id,
            "passed": passed,
            "issues": issues,
        }),
    }
}

/// Run a shell script, passing `payload` as JSON on stdin.
///
/// Returns `(exit_ok, stdout)`.  Timeout is 10 seconds.
async fn run_script(script_path: &str, payload: &serde_json::Value) -> (bool, String) {
    let payload_str = match serde_json::to_string(payload) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to serialize hook payload");
            return (true, String::new());
        }
    };

    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(script = %script_path, error = %e, "Failed to spawn hook script");
            return (true, String::new());
        }
    };

    // Write payload to stdin
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(payload_str.as_bytes()).await;
        let _ = stdin.write_all(b"\n").await;
        // stdin is dropped here, closing the pipe
    }

    // Wait with timeout
    let result =
        tokio::time::timeout(std::time::Duration::from_secs(10), child.wait_with_output()).await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let exit_ok = output.status.success();
            if !exit_ok {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::debug!(
                    script = %script_path,
                    exit_code = output.status.code().unwrap_or(-1),
                    stderr = %stderr.trim(),
                    "Hook script exited with non-zero status"
                );
            }
            (exit_ok, stdout)
        }
        Ok(Err(e)) => {
            tracing::warn!(script = %script_path, error = %e, "Hook script error");
            (true, String::new())
        }
        Err(_) => {
            tracing::warn!(script = %script_path, "Hook script timed out after 10s");
            (true, String::new())
        }
    }
}

// ── ShellHookRunner ──────────────────────────────────────────────────────────

/// Implements `LifecycleHook` — dispatches framework lifecycle events to
/// user-configured shell scripts.
pub struct ShellHookRunner {
    pre_tool_use: Option<String>,
    post_tool_use: Option<String>,
    session_start: Option<String>,
    session_end: Option<String>,
}

impl ShellHookRunner {
    /// Build from the `[hooks]` config section.
    pub fn from_config(cfg: &HooksSection) -> Self {
        Self {
            pre_tool_use: cfg.pre_tool_use.as_deref().map(expand_tilde),
            post_tool_use: cfg.post_tool_use.as_deref().map(expand_tilde),
            session_start: cfg.session_start.as_deref().map(expand_tilde),
            session_end: cfg.session_end.as_deref().map(expand_tilde),
        }
    }

    /// Returns `true` if at least one script is configured.
    pub fn has_any(&self) -> bool {
        self.pre_tool_use.is_some()
            || self.post_tool_use.is_some()
            || self.session_start.is_some()
            || self.session_end.is_some()
    }

    /// Get the pre-tool-use script path, if configured.
    pub fn pre_tool_use_path(&self) -> Option<&str> {
        self.pre_tool_use.as_deref()
    }
}

#[async_trait]
impl LifecycleHook for ShellHookRunner {
    fn name(&self) -> &str {
        "shell-hooks"
    }

    async fn on_event(&self, event: &LifecycleEvent) -> HookResult {
        let payload = event_to_json(event);

        let script = match event {
            LifecycleEvent::AgentStarted { .. } => self.session_start.as_deref(),
            LifecycleEvent::AgentCompleted { .. } | LifecycleEvent::AgentFailed { .. } => {
                self.session_end.as_deref()
            }
            // pre_tool_use is handled by ShellPreToolHook (ToolPreHook impl)
            // but also fire through the lifecycle hook for PostToolUse
            LifecycleEvent::ToolBeforeExecute { .. } => {
                // Observational only here — blocking handled by ShellPreToolHook
                None
            }
            LifecycleEvent::ToolAfterExecute { .. } => self.post_tool_use.as_deref(),
            _ => None,
        };

        if let Some(path) = script {
            run_script(path, &payload).await;
        }

        HookResult::Continue
    }
}

// ── ShellPreToolHook ─────────────────────────────────────────────────────────

/// Implements `ToolPreHook` — runs the `pre_tool_use` script before each tool
/// execution.  Non-zero exit blocks the call.
pub struct ShellPreToolHook {
    script: String,
}

impl ShellPreToolHook {
    pub fn new(script: String) -> Self {
        Self { script }
    }
}

#[async_trait]
impl ToolPreHook for ShellPreToolHook {
    async fn before_execute(
        &self,
        tool_use: &ToolUse,
        _context: &ToolContext,
    ) -> Result<PreHookDecision> {
        let payload = json!({
            "type": "tool_before_execute",
            "tool_name": tool_use.name,
            "args": tool_use.input,
        });

        let (exit_ok, stdout) = run_script(&self.script, &payload).await;

        if exit_ok {
            Ok(PreHookDecision::Allow)
        } else {
            // Use first non-empty line of stdout as the rejection reason.
            let reason = stdout
                .lines()
                .find(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string())
                .unwrap_or_else(|| format!("Tool '{}' blocked by pre-tool hook", tool_use.name));
            Ok(PreHookDecision::Reject(reason))
        }
    }
}
