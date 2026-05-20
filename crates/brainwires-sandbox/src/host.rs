//! UNSAFE — no isolation; for development and testing only.
//!
//! `HostSandbox` spawns processes directly on the host with `tokio::process`.
//! It still enforces the policy's mount whitelist (so consumers can catch
//! mis-configured policies early) and the per-spec timeout, but does not
//! apply any resource limits, namespaces, or network isolation. Do not use
//! in production.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::error::{Result, SandboxError};
use crate::{ExecHandle, ExecOutput, ExecSpec, Sandbox, SandboxPolicy, SandboxRuntime};

struct Job {
    child: Child,
    stdout_reader: JoinHandle<std::io::Result<Vec<u8>>>,
    stderr_reader: JoinHandle<std::io::Result<Vec<u8>>>,
    started: Instant,
    timeout: std::time::Duration,
}

/// Host pass-through implementation — NO isolation.
pub struct HostSandbox {
    policy: SandboxPolicy,
    jobs: Arc<Mutex<HashMap<ExecHandle, Job>>>,
}

impl HostSandbox {
    /// Build a new host sandbox. The `policy` is used only for mount
    /// validation; resource limits and network rules are ignored.
    pub fn new(policy: SandboxPolicy) -> Self {
        Self {
            policy,
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

fn spawn_reader<R: AsyncReadExt + Unpin + Send + 'static>(
    mut reader: R,
) -> JoinHandle<std::io::Result<Vec<u8>>> {
    tokio::spawn(async move {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok(buf)
    })
}

async fn join_reader(h: JoinHandle<std::io::Result<Vec<u8>>>) -> Vec<u8> {
    match h.await {
        Ok(Ok(v)) => v,
        // Reader errors or task panics shouldn't take down the caller —
        // stdout/stderr are best-effort for a timed-out or crashed child.
        _ => Vec::new(),
    }
}

#[async_trait::async_trait]
impl Sandbox for HostSandbox {
    async fn spawn(&self, spec: ExecSpec) -> Result<ExecHandle> {
        for m in &spec.mounts {
            self.policy.validate_mount(m)?;
        }

        let program = spec
            .cmd
            .first()
            .ok_or_else(|| SandboxError::PolicyViolation("empty cmd".into()))?
            .clone();

        let mut command = Command::new(&program);
        command.args(&spec.cmd[1..]);
        command.env_clear();
        for (k, v) in &spec.env {
            command.env(k, v);
        }
        command.current_dir(&spec.workdir);
        command.stdin(if spec.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = command.spawn()?;

        if let Some(bytes) = spec.stdin
            && let Some(mut stdin) = child.stdin.take()
        {
            stdin.write_all(&bytes).await?;
            stdin.shutdown().await?;
        }

        let stdout: ChildStdout = child
            .stdout
            .take()
            .ok_or_else(|| SandboxError::NotAvailable("child stdout pipe missing".into()))?;
        let stderr: ChildStderr = child
            .stderr
            .take()
            .ok_or_else(|| SandboxError::NotAvailable("child stderr pipe missing".into()))?;

        let stdout_reader = spawn_reader(stdout);
        let stderr_reader = spawn_reader(stderr);

        let handle = ExecHandle::new();
        let job = Job {
            child,
            stdout_reader,
            stderr_reader,
            started: Instant::now(),
            timeout: spec.timeout,
        };
        self.jobs.lock().await.insert(handle, job);
        Ok(handle)
    }

    async fn wait(&self, handle: ExecHandle) -> Result<ExecOutput> {
        let Job {
            mut child,
            stdout_reader,
            stderr_reader,
            started,
            timeout,
        } = self
            .jobs
            .lock()
            .await
            .remove(&handle)
            .ok_or_else(|| SandboxError::NotAvailable("unknown exec handle".into()))?;

        let status = tokio::select! {
            biased;
            res = child.wait() => res?,
            _ = tokio::time::sleep(timeout) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                // Kill only terminates the direct child. If the target was a
                // shell that forked further (e.g. `sh -c 'sleep 30'` without
                // `exec`), grandchildren may keep stdout/stderr open for
                // their full lifetime. Aborting the reader tasks bounds the
                // timeout path — partial output is discarded either way.
                stdout_reader.abort();
                stderr_reader.abort();
                tracing::debug!("HostSandbox timeout — child killed, readers aborted");
                return Err(SandboxError::Timeout);
            }
        };

        let stdout = join_reader(stdout_reader).await;
        let stderr = join_reader(stderr_reader).await;

        Ok(ExecOutput {
            exit_code: status.code().unwrap_or(-1),
            stdout,
            stderr,
            wall_time: started.elapsed(),
        })
    }

    async fn shutdown(&self) -> Result<()> {
        let mut jobs = self.jobs.lock().await;
        for (_, mut job) in jobs.drain() {
            let _ = job.child.start_kill();
            // Abort outstanding readers so dropped pipes don't leak tasks.
            job.stdout_reader.abort();
            job.stderr_reader.abort();
        }
        Ok(())
    }

    fn runtime(&self) -> SandboxRuntime {
        SandboxRuntime::Host
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Duration;

    fn base_spec(cmd: Vec<String>, timeout: Duration) -> ExecSpec {
        ExecSpec {
            cmd,
            env: BTreeMap::new(),
            workdir: PathBuf::from("/"),
            stdin: None,
            mounts: vec![],
            timeout,
        }
    }

    #[tokio::test]
    async fn echo_hello() {
        let sandbox = HostSandbox::new(SandboxPolicy::default());
        let spec = base_spec(vec!["echo".into(), "hello".into()], Duration::from_secs(5));
        let handle = sandbox.spawn(spec).await.expect("spawn");
        let out = sandbox.wait(handle).await.expect("wait");
        assert_eq!(out.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
    }

    #[tokio::test]
    async fn host_stdout_captured_on_normal_exit() {
        let sandbox = HostSandbox::new(SandboxPolicy::default());
        let spec = base_spec(
            vec!["sh".into(), "-c".into(), "echo hi".into()],
            Duration::from_secs(5),
        );
        let handle = sandbox.spawn(spec).await.expect("spawn");
        let out = sandbox.wait(handle).await.expect("wait");
        assert_eq!(out.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi");
    }

    #[tokio::test]
    async fn host_timeout_kills_child() {
        let sandbox = HostSandbox::new(SandboxPolicy::default());
        let spec = base_spec(
            vec!["sh".into(), "-c".into(), "sleep 30".into()],
            Duration::from_millis(200),
        );

        let start = std::time::Instant::now();
        let handle = sandbox.spawn(spec).await.expect("spawn");
        let err = sandbox.wait(handle).await.expect_err("should time out");
        let elapsed = start.elapsed();

        assert!(matches!(err, SandboxError::Timeout), "got {err:?}");
        assert!(
            elapsed < Duration::from_secs(2),
            "timeout path took {elapsed:?} — child was not killed"
        );
    }
}
