use crate::job::{DockerSandbox, Job, JobResult};
use chrono::Utc;
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

/// Maximum bytes of stdout/stderr kept in a `JobResult` (older content is truncated).
const TRUNCATE_BYTES: usize = 4096;

pub struct JobExecutor;

impl JobExecutor {
    /// Execute a job natively or inside a Docker sandbox and return the result.
    pub async fn run(job: &Job) -> JobResult {
        let started_at = Utc::now();
        let timer = Instant::now();

        let outcome = match &job.sandbox {
            Some(sandbox) => Self::run_docker(job, sandbox).await,
            None => Self::run_native(job).await,
        };

        let duration_secs = timer.elapsed().as_secs_f64();

        match outcome {
            Ok((exit_code, stdout, stderr)) => JobResult {
                success: exit_code == 0,
                exit_code: Some(exit_code),
                stdout: truncate_tail(&stdout, TRUNCATE_BYTES),
                stderr: truncate_tail(&stderr, TRUNCATE_BYTES),
                started_at,
                duration_secs,
                error: None,
            },
            Err(e) => JobResult {
                success: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                started_at,
                duration_secs,
                error: Some(format!("{e:#}")),
            },
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    async fn run_native(job: &Job) -> anyhow::Result<(i32, String, String)> {
        let child = Command::new(&job.command)
            .args(&job.args)
            .current_dir(&job.working_dir)
            .envs(&job.env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let out = timeout(
            Duration::from_secs(job.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("job timed out after {}s", job.timeout_secs))??;

        Ok((
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }

    async fn run_docker(
        job: &Job,
        sandbox: &DockerSandbox,
    ) -> anyhow::Result<(i32, String, String)> {
        if !Self::docker_available().await {
            anyhow::bail!(
                "docker binary not found or daemon not running; \
                 job '{}' requires sandbox (image: {})",
                job.name,
                sandbox.image
            );
        }

        let mut args = vec![
            "run".to_string(),
            "--rm".to_string(),
            format!("--memory={}m", sandbox.memory_mb),
            format!("--cpus={}", sandbox.cpu_limit),
        ];

        if !sandbox.network {
            args.push("--network=none".to_string());
        }

        // Mount the working directory at the same path inside the container
        args.extend([
            "-v".to_string(),
            format!("{}:{}", job.working_dir, job.working_dir),
            "-w".to_string(),
            job.working_dir.clone(),
        ]);

        // User-defined volume mounts
        for m in &sandbox.mounts {
            args.extend(["-v".to_string(), m.clone()]);
        }

        // Environment variables
        for (k, v) in &job.env {
            args.extend(["-e".to_string(), format!("{k}={v}")]);
        }

        // Escape-hatch extra flags (e.g. "--cap-drop=ALL")
        args.extend(sandbox.extra_args.clone());

        // Image, then the command + its arguments
        args.push(sandbox.image.clone());
        args.push(job.command.clone());
        args.extend(job.args.clone());

        let child = Command::new("docker")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let out = timeout(
            Duration::from_secs(job.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("docker job timed out after {}s", job.timeout_secs))??;

        Ok((
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }

    /// Returns `true` if the `docker` CLI is on `PATH` and the daemon responds.
    pub async fn docker_available() -> bool {
        Command::new("docker")
            .args(["version", "--format", "{{.Server.Version}}"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Keep at most `max_bytes` from the *tail* of `s`.  Prepends a truncation notice.
pub(crate) fn truncate_tail(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let approx = s.len() - max_bytes;
    // Snap forward to a valid UTF-8 char boundary
    let start = (approx..=s.len())
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(s.len());
    format!("[...truncated]\n{}", &s[start..])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{FailurePolicy, Job};
    use chrono::Utc;

    fn echo_job(args: &[&str]) -> Job {
        Job {
            id: "t".into(),
            name: "test".into(),
            cron: "* * * * *".into(),
            command: "echo".into(),
            args: args.iter().map(|s| s.to_string()).collect(),
            working_dir: "/tmp".into(),
            enabled: true,
            timeout_secs: 10,
            failure_policy: FailurePolicy::Ignore,
            sandbox: None,
            env: Default::default(),
            created_at: Utc::now(),
            last_fired_at: None,
            last_result: None,
        }
    }

    // ── truncate_tail ─────────────────────────────────────────────────────────

    #[test]
    fn short_string_unchanged() {
        let s = "hello world";
        assert_eq!(truncate_tail(s, 100), s);
    }

    #[test]
    fn exact_length_unchanged() {
        let s = "a".repeat(4096);
        assert_eq!(truncate_tail(&s, 4096), s);
    }

    #[test]
    fn long_string_gets_truncation_marker() {
        let s = "x".repeat(5000);
        let result = truncate_tail(&s, 4096);
        assert!(
            result.starts_with("[...truncated]"),
            "should start with truncation marker"
        );
        assert!(result.len() < 5000, "result should be shorter than input");
    }

    #[test]
    fn truncated_result_is_valid_utf8() {
        // 4-byte emoji — the boundary check must not split mid-codepoint
        let s = "🦀".repeat(2000); // 8000 bytes total
        let result = truncate_tail(&s, 4096);
        assert!(
            std::str::from_utf8(result.as_bytes()).is_ok(),
            "result must be valid UTF-8"
        );
    }

    #[test]
    fn truncated_result_ends_with_original_suffix() {
        let s = format!("{}{}", "a".repeat(5000), "SUFFIX");
        let result = truncate_tail(&s, 100);
        assert!(
            result.ends_with("SUFFIX"),
            "truncated result should preserve the tail"
        );
    }

    // ── JobExecutor (native) ──────────────────────────────────────────────────

    #[tokio::test]
    async fn echo_succeeds_with_correct_output() {
        let job = echo_job(&["hello", "scheduler"]);
        let result = JobExecutor::run(&job).await;
        assert!(result.success, "echo should succeed");
        assert_eq!(result.exit_code, Some(0));
        assert!(
            result.stdout.contains("hello scheduler"),
            "stdout should contain echoed text, got: {:?}",
            result.stdout
        );
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn false_command_fails() {
        let mut job = echo_job(&[]);
        job.command = "false".into();
        let result = JobExecutor::run(&job).await;
        assert!(!result.success, "false should fail");
        assert!(result.exit_code.is_some_and(|c| c != 0));
        assert!(
            result.error.is_none(),
            "error field is for launch failures only"
        );
    }

    #[tokio::test]
    async fn nonexistent_command_sets_error_field() {
        let mut job = echo_job(&[]);
        job.command = "this-binary-does-not-exist-brainwires-test".into();
        let result = JobExecutor::run(&job).await;
        assert!(!result.success);
        assert!(
            result.error.is_some(),
            "launch failure should populate the error field"
        );
    }

    #[tokio::test]
    async fn timeout_is_enforced() {
        let mut job = echo_job(&[]);
        job.command = "sleep".into();
        job.args = vec!["60".into()];
        job.timeout_secs = 1;
        let result = JobExecutor::run(&job).await;
        assert!(!result.success);
        assert!(
            result.error.as_deref().unwrap_or("").contains("timed out"),
            "error should mention timeout, got: {:?}",
            result.error
        );
        assert!(result.duration_secs < 5.0, "job should not run for 60s");
    }

    #[tokio::test]
    async fn env_vars_are_passed_to_process() {
        let mut job = echo_job(&[]);
        job.command = "sh".into();
        job.args = vec!["-c".into(), "echo $MY_TEST_VAR".into()];
        job.env.insert("MY_TEST_VAR".into(), "brainwires_ok".into());
        let result = JobExecutor::run(&job).await;
        assert!(result.success);
        assert!(
            result.stdout.contains("brainwires_ok"),
            "env var should appear in output, got: {:?}",
            result.stdout
        );
    }
}
