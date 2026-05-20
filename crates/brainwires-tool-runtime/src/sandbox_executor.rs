//! Sandboxed tool executor decorator.
//!
//! Wraps any [`ToolExecutor`] and intercepts calls to known-dangerous tool
//! names (`bash` / `execute_command` / `code_exec` / `execute_code`), running
//! them inside a [`brainwires_sandbox::Sandbox`] instead of on the host. All
//! other tool calls pass through unchanged.
//!
//! Sandbox errors (timeout, policy violation, docker failures) are always
//! returned as [`ToolResult::error`] so the agent loop treats them as
//! ordinary tool results rather than hard errors that abort the run.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::debug;

use brainwires_core::{Tool, ToolContext, ToolResult, ToolUse};
use brainwires_sandbox::{ExecSpec, Sandbox, SandboxError, SandboxPolicy};

use crate::executor::ToolExecutor;

/// Fallback workdir used when neither the policy's `workspace_mount` nor the
/// [`ToolContext::working_directory`] yields a usable path.
const DEFAULT_WORKDIR: &str = "/workspace";

/// Decorator that routes dangerous tool calls (`bash`, `execute_command`,
/// `code_exec`, `execute_code`) through a [`Sandbox`] and forwards everything
/// else to `inner`.
pub struct SandboxedToolExecutor<E: ToolExecutor> {
    inner: E,
    sandbox: Arc<dyn Sandbox>,
    policy: SandboxPolicy,
    default_timeout: Duration,
}

impl<E: ToolExecutor> SandboxedToolExecutor<E> {
    /// Wrap `inner` so dangerous calls are routed through `sandbox` under
    /// `policy`. Defaults to a 5-minute wall-clock timeout per sandboxed call.
    pub fn new(inner: E, sandbox: Arc<dyn Sandbox>, policy: SandboxPolicy) -> Self {
        Self {
            inner,
            sandbox,
            policy,
            default_timeout: Duration::from_secs(300),
        }
    }

    /// Override the per-call wall-clock timeout used for sandboxed commands.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Borrow the wrapped executor.
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Borrow the active sandbox policy.
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Resolve the workdir to use inside the sandbox.
    ///
    /// Preference order:
    /// 1. `policy.workspace_mount` (keeps the sandbox pinned inside a known
    ///    mount so the process can't land on a host path that doesn't exist
    ///    inside the container).
    /// 2. `ToolContext::working_directory` (the agent's cwd).
    /// 3. `/workspace` (final fallback).
    fn workdir_for(&self, context: &ToolContext) -> PathBuf {
        if let Some(ref mount) = self.policy.workspace_mount {
            return mount.clone();
        }
        if !context.working_directory.is_empty() {
            return PathBuf::from(&context.working_directory);
        }
        PathBuf::from(DEFAULT_WORKDIR)
    }

    async fn run_in_sandbox(
        &self,
        tool_use_id: &str,
        tool_name: &str,
        cmd: Vec<String>,
        workdir: PathBuf,
    ) -> ToolResult {
        // Host env is intentionally NOT inherited — any secret leakage here
        // would defeat the isolation the caller is paying for.
        let spec = ExecSpec {
            cmd,
            env: BTreeMap::new(),
            workdir,
            stdin: None,
            mounts: vec![],
            timeout: self.default_timeout,
        };

        let handle = match self.sandbox.spawn(spec).await {
            Ok(h) => h,
            Err(e) => return sandbox_error_to_result(tool_use_id, e, self.default_timeout),
        };

        match self.sandbox.wait(handle).await {
            Ok(output) => {
                debug!(
                    tool = tool_name,
                    exit_code = output.exit_code,
                    wall_time_ms = output.wall_time.as_millis() as u64,
                    "sandboxed tool call completed"
                );
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                if output.exit_code == 0 {
                    ToolResult::success(tool_use_id.to_string(), stdout)
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                    ToolResult::error(
                        tool_use_id.to_string(),
                        format!("exit {}: {}", output.exit_code, stderr),
                    )
                }
            }
            Err(e) => sandbox_error_to_result(tool_use_id, e, self.default_timeout),
        }
    }

    async fn run_bash(&self, tool_use: &ToolUse, context: &ToolContext) -> ToolResult {
        let Some(command) = tool_use.input.get("command").and_then(|v| v.as_str()) else {
            return ToolResult::error(
                tool_use.id.clone(),
                "sandbox: missing or non-string 'command' parameter".to_string(),
            );
        };
        let cmd = vec!["/bin/sh".to_string(), "-c".to_string(), command.to_string()];
        self.run_in_sandbox(&tool_use.id, &tool_use.name, cmd, self.workdir_for(context))
            .await
    }

    async fn run_code_exec(&self, tool_use: &ToolUse, context: &ToolContext) -> ToolResult {
        let Some(language) = tool_use.input.get("language").and_then(|v| v.as_str()) else {
            return ToolResult::error(
                tool_use.id.clone(),
                "sandbox: missing or non-string 'language' parameter".to_string(),
            );
        };
        let Some(code) = tool_use.input.get("code").and_then(|v| v.as_str()) else {
            return ToolResult::error(
                tool_use.id.clone(),
                "sandbox: missing or non-string 'code' parameter".to_string(),
            );
        };

        let lang = language.to_lowercase();
        let cmd = match lang.as_str() {
            "python" | "python3" => {
                vec!["python3".to_string(), "-c".to_string(), code.to_string()]
            }
            "node" | "javascript" | "js" => {
                vec!["node".to_string(), "-e".to_string(), code.to_string()]
            }
            "bash" | "sh" | "shell" => {
                vec!["/bin/sh".to_string(), "-c".to_string(), code.to_string()]
            }
            other => {
                return ToolResult::error(
                    tool_use.id.clone(),
                    format!("sandbox does not yet support language '{other}'"),
                );
            }
        };

        self.run_in_sandbox(&tool_use.id, &tool_use.name, cmd, self.workdir_for(context))
            .await
    }
}

#[async_trait]
impl<E: ToolExecutor> ToolExecutor for SandboxedToolExecutor<E> {
    async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> Result<ToolResult> {
        match tool_use.name.as_str() {
            "bash" | "execute_command" => Ok(self.run_bash(tool_use, context).await),
            "code_exec" | "execute_code" => Ok(self.run_code_exec(tool_use, context).await),
            _ => self.inner.execute(tool_use, context).await,
        }
    }

    fn available_tools(&self) -> Vec<Tool> {
        self.inner.available_tools()
    }
}

fn sandbox_error_to_result(tool_use_id: &str, err: SandboxError, timeout: Duration) -> ToolResult {
    let msg = match err {
        SandboxError::Timeout => format!("sandboxed command timed out after {:?}", timeout),
        SandboxError::PolicyViolation(reason) => format!("policy violation: {reason}"),
        other => format!("sandbox error: {other}"),
    };
    ToolResult::error(tool_use_id.to_string(), msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use brainwires_core::{ToolContext, ToolUse};
    use brainwires_sandbox::{ExecHandle, ExecOutput, ExecSpec, Sandbox, SandboxRuntime};

    struct MockSandbox {
        exit_code: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        should_timeout: AtomicBool,
        spawned_specs: Mutex<Vec<ExecSpec>>,
    }

    impl MockSandbox {
        fn new(exit_code: i32, stdout: &[u8], stderr: &[u8]) -> Arc<Self> {
            Arc::new(Self {
                exit_code,
                stdout: stdout.to_vec(),
                stderr: stderr.to_vec(),
                should_timeout: AtomicBool::new(false),
                spawned_specs: Mutex::new(Vec::new()),
            })
        }

        fn timing_out() -> Arc<Self> {
            Arc::new(Self {
                exit_code: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
                should_timeout: AtomicBool::new(true),
                spawned_specs: Mutex::new(Vec::new()),
            })
        }

        fn specs(&self) -> Vec<ExecSpec> {
            self.spawned_specs.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Sandbox for MockSandbox {
        async fn spawn(&self, spec: ExecSpec) -> brainwires_sandbox::Result<ExecHandle> {
            self.spawned_specs.lock().unwrap().push(spec);
            Ok(ExecHandle::new())
        }

        async fn wait(&self, _handle: ExecHandle) -> brainwires_sandbox::Result<ExecOutput> {
            if self.should_timeout.load(Ordering::SeqCst) {
                return Err(SandboxError::Timeout);
            }
            Ok(ExecOutput {
                exit_code: self.exit_code,
                stdout: self.stdout.clone(),
                stderr: self.stderr.clone(),
                wall_time: Duration::from_millis(1),
            })
        }

        async fn shutdown(&self) -> brainwires_sandbox::Result<()> {
            Ok(())
        }

        fn runtime(&self) -> SandboxRuntime {
            SandboxRuntime::Host
        }
    }

    struct CountingInner {
        calls: AtomicUsize,
    }

    impl CountingInner {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ToolExecutor for CountingInner {
        async fn execute(&self, tool_use: &ToolUse, _ctx: &ToolContext) -> Result<ToolResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::success(
                tool_use.id.clone(),
                "inner-executed".to_string(),
            ))
        }

        fn available_tools(&self) -> Vec<Tool> {
            Vec::new()
        }
    }

    fn ctx() -> ToolContext {
        ToolContext {
            working_directory: "/tmp".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn bash_is_routed_through_sandbox_and_inner_is_not_called() {
        let sandbox = MockSandbox::new(0, b"hello from sandbox\n", b"");
        let exec = SandboxedToolExecutor::new(
            CountingInner::new(),
            sandbox.clone() as Arc<dyn Sandbox>,
            SandboxPolicy::default(),
        );

        let tool_use = ToolUse {
            id: "t-1".to_string(),
            name: "bash".to_string(),
            input: json!({ "command": "echo hello" }),
        };

        let result = exec.execute(&tool_use, &ctx()).await.expect("execute");
        assert!(!result.is_error, "unexpected error: {}", result.content);
        assert!(result.content.contains("hello from sandbox"));
        assert_eq!(
            exec.inner().call_count(),
            0,
            "inner executor must not be called for bash"
        );

        let specs = sandbox.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].cmd,
            vec![
                "/bin/sh".to_string(),
                "-c".to_string(),
                "echo hello".to_string()
            ]
        );
        assert!(specs[0].env.is_empty(), "host env must not leak");
    }

    #[tokio::test]
    async fn non_dangerous_tool_delegates_to_inner_executor() {
        let sandbox = MockSandbox::new(0, b"should not appear", b"");
        let exec = SandboxedToolExecutor::new(
            CountingInner::new(),
            sandbox as Arc<dyn Sandbox>,
            SandboxPolicy::default(),
        );

        let tool_use = ToolUse {
            id: "t-2".to_string(),
            name: "read_file".to_string(),
            input: json!({ "path": "/etc/hosts" }),
        };

        let result = exec.execute(&tool_use, &ctx()).await.expect("execute");
        assert!(!result.is_error);
        assert_eq!(result.content, "inner-executed");
        assert_eq!(exec.inner().call_count(), 1);
    }

    #[tokio::test]
    async fn non_zero_exit_becomes_error_result_with_exit_code() {
        let sandbox = MockSandbox::new(42, b"", b"boom");
        let exec = SandboxedToolExecutor::new(
            CountingInner::new(),
            sandbox as Arc<dyn Sandbox>,
            SandboxPolicy::default(),
        );

        let tool_use = ToolUse {
            id: "t-3".to_string(),
            name: "execute_command".to_string(),
            input: json!({ "command": "false" }),
        };

        let result = exec.execute(&tool_use, &ctx()).await.expect("execute");
        assert!(result.is_error);
        assert!(
            result.content.contains("exit 42"),
            "content was: {}",
            result.content
        );
        assert!(result.content.contains("boom"));
    }

    #[tokio::test]
    async fn timeout_becomes_error_result_containing_timed_out() {
        let sandbox = MockSandbox::timing_out();
        let exec = SandboxedToolExecutor::new(
            CountingInner::new(),
            sandbox as Arc<dyn Sandbox>,
            SandboxPolicy::default(),
        )
        .with_timeout(Duration::from_millis(5));

        let tool_use = ToolUse {
            id: "t-4".to_string(),
            name: "bash".to_string(),
            input: json!({ "command": "sleep 999" }),
        };

        let result = exec.execute(&tool_use, &ctx()).await.expect("execute");
        assert!(result.is_error);
        assert!(
            result.content.contains("timed out"),
            "content was: {}",
            result.content
        );
    }
}
