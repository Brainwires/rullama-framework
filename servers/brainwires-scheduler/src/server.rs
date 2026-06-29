use crate::daemon::{DaemonHandle, next_fire, next_fire_after};
use crate::job::{DockerSandbox, FailurePolicy, Job, JobResult};
use crate::store::JobStore;
use chrono::Utc;
use rmcp::{
    ServerHandler,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::HashMap;

// ── Server struct ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SchedulerServer {
    handle: DaemonHandle,
    tool_router: ToolRouter<Self>,
}

impl SchedulerServer {
    pub fn new(handle: DaemonHandle) -> Self {
        Self {
            handle,
            tool_router: Self::tool_router(),
        }
    }
}

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddJobRequest {
    /// Human-readable name for the job
    pub name: String,
    /// Cron expression: 5-field "min hour dom month dow" or 7-field with leading seconds
    pub cron: String,
    /// Executable to run (must be on PATH or an absolute path)
    pub command: String,
    /// Arguments passed to the command
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory (defaults to the daemon's working directory)
    pub working_dir: Option<String>,
    /// Timeout in seconds before the job is killed (default: 3600)
    pub timeout_secs: Option<u64>,
    /// What to do when the job fails (default: ignore)
    pub failure_policy: Option<FailurePolicy>,
    /// Optional Docker sandbox configuration
    pub sandbox: Option<DockerSandbox>,
    /// Extra environment variables for the job process
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobIdRequest {
    /// Job ID returned by `add_job`
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetLogsRequest {
    /// Job ID
    pub id: String,
    /// Maximum number of log entries to return (default: 5, max: 20)
    pub limit: Option<usize>,
}

// ── Tool implementations ──────────────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl SchedulerServer {
    #[tool(description = "Add a new scheduled job. Returns the generated job ID.")]
    async fn add_job(&self, Parameters(req): Parameters<AddJobRequest>) -> Result<String, String> {
        let now = Utc::now();
        let id = uuid::Uuid::new_v4().to_string();

        let job = Job {
            id: id.clone(),
            name: req.name,
            cron: req.cron.clone(),
            command: req.command,
            args: req.args,
            working_dir: req.working_dir.unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap_or_else(|_| "/".into())
                    .to_string_lossy()
                    .into_owned()
            }),
            enabled: true,
            timeout_secs: req.timeout_secs.unwrap_or(3600),
            failure_policy: req.failure_policy.unwrap_or_default(),
            sandbox: req.sandbox,
            env: req.env,
            created_at: now,
            last_fired_at: None,
            last_result: None,
        };

        // Validate cron before accepting
        if next_fire(&req.cron).is_none() {
            return Err(format!("invalid cron expression: {:?}", req.cron));
        }

        // Validate working_dir exists and is a directory
        let working_dir = job.working_dir.as_str();
        match std::fs::metadata(working_dir) {
            Ok(m) if m.is_dir() => {}
            Ok(_) => return Err(format!("working_dir is not a directory: {working_dir}")),
            Err(e) => return Err(format!("working_dir does not exist: {working_dir}: {e}")),
        }

        self.handle
            .add_job(job)
            .await
            .map_err(|e| format!("{e:#}"))?;

        let next = next_fire(&req.cron)
            .map(|t| t.to_rfc3339())
            .unwrap_or_else(|| "unknown".to_string());

        Ok(format!("Job added.\nid: {id}\nNext run: {next}"))
    }

    #[tool(description = "Permanently remove a scheduled job.")]
    async fn remove_job(
        &self,
        Parameters(req): Parameters<JobIdRequest>,
    ) -> Result<String, String> {
        // Verify the job exists before sending command
        {
            let store = self.handle.store.read().await;
            if store.get(&req.id).is_none() {
                return Err(format!("job not found: {}", req.id));
            }
        }
        self.handle
            .remove_job(&req.id)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format!("Job {} removed.", req.id))
    }

    #[tool(description = "List all scheduled jobs with their status and next scheduled run time.")]
    async fn list_jobs(&self, Parameters(()): Parameters<()>) -> Result<String, String> {
        let store = self.handle.store.read().await;
        let jobs = store.all();

        if jobs.is_empty() {
            return Ok("No jobs scheduled.".to_string());
        }

        let now = Utc::now();
        let mut lines = vec![
            format!(
                "{:<36}  {:<20}  {:<8}  {:<22}  {}",
                "ID", "NAME", "STATUS", "NEXT RUN", "LAST RESULT"
            ),
            "-".repeat(110),
        ];

        for j in jobs {
            let status = if j.enabled { "enabled" } else { "disabled" };
            let next = j
                .last_fired_at
                .and_then(|lf| next_fire_after(&j.cron, lf))
                .or_else(|| next_fire(&j.cron))
                .map(|t| {
                    let secs = (t - now).num_seconds();
                    if secs < 60 {
                        format!("in {}s", secs.max(0))
                    } else if secs < 3600 {
                        format!("in {}m", secs / 60)
                    } else {
                        format!("in {}h", secs / 3600)
                    }
                })
                .unwrap_or_else(|| "never".to_string());

            let last = match &j.last_result {
                None => "never run".to_string(),
                Some(r) if r.success => format!("✓ {:.1}s", r.duration_secs),
                Some(r) => format!("✗ {:.1}s", r.duration_secs),
            };

            lines.push(format!(
                "{:<36}  {:<20}  {:<8}  {:<22}  {}",
                j.id,
                truncate_str(&j.name, 20),
                status,
                next,
                last,
            ));
        }

        Ok(lines.join("\n"))
    }

    #[tool(description = "Get full details of a specific job including its current configuration.")]
    async fn get_job(&self, Parameters(req): Parameters<JobIdRequest>) -> Result<String, String> {
        let store = self.handle.store.read().await;
        let job = store
            .get(&req.id)
            .ok_or_else(|| format!("job not found: {}", req.id))?;

        let next = job
            .last_fired_at
            .and_then(|lf| next_fire_after(&job.cron, lf))
            .or_else(|| next_fire(&job.cron))
            .map(|t| t.to_rfc3339())
            .unwrap_or_else(|| "never".to_string());

        let last_result_summary = match &job.last_result {
            None => "never run".to_string(),
            Some(r) => format!(
                "{} (exit {}), {:.1}s, started {}",
                if r.success { "success" } else { "failed" },
                r.exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                r.duration_secs,
                r.started_at.format("%Y-%m-%d %H:%M:%S UTC"),
            ),
        };

        let sandbox_info = match &job.sandbox {
            None => "none (runs natively)".to_string(),
            Some(s) => format!(
                "Docker image={}, memory={}MB, cpu={}, network={}",
                s.image,
                s.memory_mb,
                s.cpu_limit,
                if s.network { "enabled" } else { "disabled" }
            ),
        };

        Ok(format!(
            "ID:              {}\n\
             Name:            {}\n\
             Cron:            {}\n\
             Command:         {} {}\n\
             Working dir:     {}\n\
             Enabled:         {}\n\
             Timeout:         {}s\n\
             Failure policy:  {}\n\
             Sandbox:         {}\n\
             Created:         {}\n\
             Next run:        {}\n\
             Last result:     {}",
            job.id,
            job.name,
            job.cron,
            job.command,
            job.args.join(" "),
            job.working_dir,
            job.enabled,
            job.timeout_secs,
            failure_policy_name(&job.failure_policy),
            sandbox_info,
            job.created_at.format("%Y-%m-%d %H:%M:%S UTC"),
            next,
            last_result_summary,
        ))
    }

    #[tool(description = "Enable a previously disabled job.")]
    async fn enable_job(
        &self,
        Parameters(req): Parameters<JobIdRequest>,
    ) -> Result<String, String> {
        {
            let store = self.handle.store.read().await;
            if store.get(&req.id).is_none() {
                return Err(format!("job not found: {}", req.id));
            }
        }
        self.handle
            .enable_job(&req.id)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format!("Job {} enabled.", req.id))
    }

    #[tool(description = "Disable a job without removing it. The job can be re-enabled later.")]
    async fn disable_job(
        &self,
        Parameters(req): Parameters<JobIdRequest>,
    ) -> Result<String, String> {
        {
            let store = self.handle.store.read().await;
            if store.get(&req.id).is_none() {
                return Err(format!("job not found: {}", req.id));
            }
        }
        self.handle
            .disable_job(&req.id)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format!("Job {} disabled.", req.id))
    }

    #[tool(
        description = "Trigger a job to run immediately, outside its normal cron schedule. Waits for the job to complete and returns the result."
    )]
    async fn run_job(&self, Parameters(req): Parameters<JobIdRequest>) -> Result<String, String> {
        let result = self
            .handle
            .run_now(&req.id)
            .await
            .map_err(|e| format!("{e:#}"))?;
        Ok(format_result(&result))
    }

    #[tool(description = "Retrieve stdout/stderr logs from recent executions of a job.")]
    async fn get_logs(
        &self,
        Parameters(req): Parameters<GetLogsRequest>,
    ) -> Result<String, String> {
        let limit = req.limit.unwrap_or(5).min(20);

        // Capture the log directory path while holding the lock, then drop it.
        // Log I/O must not happen while the store lock is held — it blocks the daemon.
        let log_dir = {
            let store = self.handle.store.read().await;
            match store.get(&req.id) {
                None => return Err(format!("job not found: {}", req.id)),
                Some(_) => store.log_dir_for(&req.id),
            }
        }; // lock dropped here

        let logs = JobStore::read_logs_from_dir(&log_dir, limit)
            .await
            .map_err(|e| format!("{e:#}"))?;

        if logs.is_empty() {
            return Ok("No execution logs found for this job.".to_string());
        }

        let mut out = Vec::new();
        for (i, r) in logs.iter().enumerate() {
            out.push(format!(
                "── Run {} ({}) ─────────────────────────────",
                i + 1,
                r.started_at.format("%Y-%m-%d %H:%M:%S UTC")
            ));
            out.push(format_result(r));
        }
        Ok(out.join("\n\n"))
    }

    #[tool(description = "Return overall scheduler status: uptime, job counts, and daemon health.")]
    async fn status(&self, Parameters(()): Parameters<()>) -> Result<String, String> {
        let store = self.handle.store.read().await;
        let all = store.all();
        let total = all.len();
        let enabled = all.iter().filter(|j| j.enabled).count();
        let disabled = total - enabled;
        let failed_last = all
            .iter()
            .filter(|j| j.last_result.as_ref().map(|r| !r.success).unwrap_or(false))
            .count();

        let uptime_secs = (Utc::now() - self.handle.started_at).num_seconds();
        let uptime = format_duration(uptime_secs);

        Ok(format!(
            "Scheduler status\n\
             ─────────────────\n\
             Uptime:          {uptime}\n\
             Total jobs:      {total}\n\
             Enabled:         {enabled}\n\
             Disabled:        {disabled}\n\
             Last run failed: {failed_last}",
        ))
    }
}

// ── ServerHandler ─────────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SchedulerServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("brainwires-scheduler", env!("CARGO_PKG_VERSION"))
            .with_title("Brainwires Scheduler — local cron job manager");
        info.instructions = Some(
            "Schedule and manage local cron jobs. Use add_job to create a job, \
             list_jobs to see all jobs, run_job to trigger immediately, and \
             get_logs to inspect recent outputs. Jobs optionally run inside \
             Docker sandboxes for isolation."
                .into(),
        );
        info
    }
}

// ── Formatting helpers ────────────────────────────────────────────────────────

fn format_result(r: &JobResult) -> String {
    let status = if r.success {
        format!("✓ success (exit 0) in {:.2}s", r.duration_secs)
    } else {
        format!(
            "✗ failed (exit {}) in {:.2}s",
            r.exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".to_string()),
            r.duration_secs
        )
    };

    let mut parts = vec![status];

    if let Some(err) = &r.error {
        parts.push(format!("Error: {err}"));
    }
    if !r.stdout.is_empty() {
        parts.push(format!("stdout:\n{}", r.stdout));
    }
    if !r.stderr.is_empty() {
        parts.push(format!("stderr:\n{}", r.stderr));
    }

    parts.join("\n")
}

fn failure_policy_name(p: &FailurePolicy) -> &'static str {
    match p {
        FailurePolicy::Ignore => "ignore",
        FailurePolicy::Retry { .. } => "retry",
        FailurePolicy::Disable => "disable",
    }
}

fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((byte_pos, _)) => &s[..byte_pos],
    }
}

fn format_duration(secs: i64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}
