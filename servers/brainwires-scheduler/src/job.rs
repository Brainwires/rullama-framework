use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single scheduled job definition.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Job {
    /// Unique job identifier (UUID v4, auto-generated on creation)
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Cron expression: 5-field "min hour dom month dow" or 7-field with leading seconds
    pub cron: String,
    /// Executable to run
    pub command: String,
    /// Arguments passed to the command
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory (absolute path)
    #[serde(default = "default_working_dir")]
    pub working_dir: String,
    /// Whether the job is active
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum wall-clock seconds before the job is forcibly killed (default: 3600)
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Policy applied when the job exits non-zero
    #[serde(default)]
    pub failure_policy: FailurePolicy,
    /// Optional Docker sandbox; `None` means run natively on the host
    #[serde(default)]
    pub sandbox: Option<DockerSandbox>,
    /// Extra environment variables injected into the process
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// When this job record was created
    pub created_at: DateTime<Utc>,
    /// Start time of the most recent execution (used to compute next due time)
    #[serde(default)]
    pub last_fired_at: Option<DateTime<Utc>>,
    /// Outcome of the most recent execution
    #[serde(default)]
    pub last_result: Option<JobResult>,
}

fn default_working_dir() -> String {
    std::env::current_dir()
        .unwrap_or_else(|_| "/".into())
        .to_string_lossy()
        .into_owned()
}

fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    3600
}

/// Docker sandbox configuration applied to a single job.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DockerSandbox {
    /// Docker image to run in (e.g. `"ubuntu:24.04"`, `"rust:latest"`, `"alpine"`)
    pub image: String,
    /// Memory limit in MB (default: 512)
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,
    /// CPU limit as a decimal (default: 1.0)
    #[serde(default = "default_cpu_limit")]
    pub cpu_limit: f64,
    /// Allow outbound network access (default: false — fully isolated)
    #[serde(default)]
    pub network: bool,
    /// Volume mounts in `"host_path:container_path[:ro]"` format
    #[serde(default)]
    pub mounts: Vec<String>,
    /// Extra flags forwarded verbatim to `docker run` (escape hatch)
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_memory_mb() -> u64 {
    512
}

fn default_cpu_limit() -> f64 {
    1.0
}

/// What to do when a job exits non-zero.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Log the failure and continue scheduling (default)
    #[default]
    Ignore,
    /// Retry up to `max_retries` times with a fixed backoff between attempts
    Retry {
        #[serde(default = "default_max_retries")]
        max_retries: u32,
        /// Seconds to wait between retry attempts
        #[serde(default = "default_backoff_secs")]
        backoff_secs: u64,
    },
    /// Disable the job permanently after the first failure
    Disable,
}

fn default_max_retries() -> u32 {
    3
}

fn default_backoff_secs() -> u64 {
    60
}

/// Outcome of a single job execution.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobResult {
    pub success: bool,
    pub exit_code: Option<i32>,
    /// Last 4 KB of stdout (older output is truncated)
    pub stdout: String,
    /// Last 4 KB of stderr (older output is truncated)
    pub stderr: String,
    pub started_at: DateTime<Utc>,
    pub duration_secs: f64,
    /// Populated when the job could not be launched (e.g. binary not found, Docker unavailable)
    pub error: Option<String>,
}
