use crate::executor::JobExecutor;
use crate::job::{FailurePolicy, Job, JobResult};
use crate::store::JobStore;
use anyhow::Result;
use chrono::{DateTime, Utc};
use cron::Schedule;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore, mpsc, oneshot, watch};
use tokio::time::{Duration, sleep};

// ── Command channel ──────────────────────────────────────────────────────────

pub enum DaemonCommand {
    AddJob(Box<Job>),
    RemoveJob(String),
    EnableJob(String),
    DisableJob(String),
    RunNow(String, oneshot::Sender<Result<JobResult>>),
}

// ── Public handle (held by the MCP server) ───────────────────────────────────

/// Cheap-to-clone handle that the MCP server uses to talk to the daemon.
#[derive(Clone)]
pub struct DaemonHandle {
    tx: mpsc::Sender<DaemonCommand>,
    /// Shared store — MCP server can read it directly for queries.
    pub store: Arc<RwLock<JobStore>>,
    /// Time the daemon (and the process) started.
    pub started_at: DateTime<Utc>,
}

impl DaemonHandle {
    pub async fn add_job(&self, job: Job) -> Result<()> {
        self.tx
            .send(DaemonCommand::AddJob(Box::new(job)))
            .await
            .map_err(|_| anyhow::anyhow!("scheduler daemon has stopped"))
    }

    pub async fn remove_job(&self, id: &str) -> Result<()> {
        self.tx
            .send(DaemonCommand::RemoveJob(id.to_string()))
            .await
            .map_err(|_| anyhow::anyhow!("scheduler daemon has stopped"))
    }

    pub async fn enable_job(&self, id: &str) -> Result<()> {
        self.tx
            .send(DaemonCommand::EnableJob(id.to_string()))
            .await
            .map_err(|_| anyhow::anyhow!("scheduler daemon has stopped"))
    }

    pub async fn disable_job(&self, id: &str) -> Result<()> {
        self.tx
            .send(DaemonCommand::DisableJob(id.to_string()))
            .await
            .map_err(|_| anyhow::anyhow!("scheduler daemon has stopped"))
    }

    /// Trigger a job immediately (outside its normal schedule) and await the result.
    /// Retry policies are intentionally bypassed — callers see the raw outcome.
    pub async fn run_now(&self, id: &str) -> Result<JobResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(DaemonCommand::RunNow(id.to_string(), reply_tx))
            .await
            .map_err(|_| anyhow::anyhow!("scheduler daemon has stopped"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("scheduler daemon has stopped"))?
    }
}

// ── Daemon ────────────────────────────────────────────────────────────────────

pub struct SchedulerDaemon {
    store: Arc<RwLock<JobStore>>,
    rx: mpsc::Receiver<DaemonCommand>,
    cancel: watch::Receiver<bool>,
    /// Limits the number of concurrently *running* jobs (not just selected per tick).
    semaphore: Arc<Semaphore>,
    /// Total permit count — used to wait for all in-flight jobs on shutdown.
    max_concurrent: usize,
}

impl SchedulerDaemon {
    /// Create the daemon and the handle that controls it.
    ///
    /// Returns `(daemon, handle, cancel_sender)`.  Send `true` on `cancel_sender`
    /// to trigger a graceful shutdown that waits for all in-flight jobs to finish.
    pub fn new(
        store: Arc<RwLock<JobStore>>,
        max_concurrent: usize,
    ) -> (Self, DaemonHandle, watch::Sender<bool>) {
        let (tx, rx) = mpsc::channel(64);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let max_concurrent = max_concurrent.max(1);

        let handle = DaemonHandle {
            tx,
            store: Arc::clone(&store),
            started_at: Utc::now(),
        };

        let daemon = Self {
            store,
            rx,
            cancel: cancel_rx,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        };

        (daemon, handle, cancel_tx)
    }

    /// Run the scheduler loop until cancelled, then wait for all in-flight jobs to finish.
    pub async fn run(mut self) {
        tracing::info!("scheduler daemon started");

        loop {
            // Drain all pending commands before sleeping
            while let Ok(cmd) = self.rx.try_recv() {
                self.handle_command(cmd).await;
            }

            if *self.cancel.borrow() {
                break;
            }

            let sleep_ms = self.next_sleep_ms().await;

            tokio::select! {
                _ = sleep(Duration::from_millis(sleep_ms)) => {}
                _ = self.cancel.changed() => {
                    if *self.cancel.borrow() { break; }
                }
                Some(cmd) = self.rx.recv() => {
                    self.handle_command(cmd).await;
                    continue;
                }
            }

            self.fire_due_jobs().await;
        }

        // Graceful shutdown: wait for all in-flight jobs to release their semaphore permits.
        // Acquiring every permit means every running task has finished.
        let in_flight = self
            .max_concurrent
            .saturating_sub(self.semaphore.available_permits());
        if in_flight > 0 {
            tracing::info!("waiting for {in_flight} in-flight job(s) to complete before exit");
            let _ = self
                .semaphore
                .acquire_many(self.max_concurrent as u32)
                .await;
        }

        tracing::info!("scheduler daemon stopped");
    }

    // ── Private ──────────────────────────────────────────────────────────────

    async fn next_sleep_ms(&self) -> u64 {
        let store = self.store.read().await;
        let now = Utc::now();
        store
            .all()
            .into_iter()
            .filter(|j| j.enabled)
            .filter_map(|j| next_fire(&j.cron))
            .map(|t| (t - now).num_milliseconds().max(0) as u64)
            .min()
            .unwrap_or(60_000)
            .min(60_000)
    }

    async fn handle_command(&mut self, cmd: DaemonCommand) {
        match cmd {
            DaemonCommand::AddJob(job) => {
                // Treat the creation moment as "last fired" so the job waits
                // for the next cron tick rather than firing immediately on add.
                let mut job = *job;
                job.last_fired_at.get_or_insert_with(Utc::now);
                let mut store = self.store.write().await;
                store.insert(job);
                persist(&mut store).await;
            }
            DaemonCommand::RemoveJob(id) => {
                let mut store = self.store.write().await;
                store.remove(&id);
                persist(&mut store).await;
            }
            DaemonCommand::EnableJob(id) => {
                let mut store = self.store.write().await;
                if let Some(j) = store.get_mut(&id) {
                    j.enabled = true;
                }
                persist(&mut store).await;
            }
            DaemonCommand::DisableJob(id) => {
                let mut store = self.store.write().await;
                if let Some(j) = store.get_mut(&id) {
                    j.enabled = false;
                }
                persist(&mut store).await;
            }
            DaemonCommand::RunNow(id, reply) => {
                let job = self.store.read().await.get(&id).cloned();
                match job {
                    None => {
                        let _ = reply.send(Err(anyhow::anyhow!("job not found: {id}")));
                    }
                    Some(job) => {
                        let store = Arc::clone(&self.store);
                        let semaphore = Arc::clone(&self.semaphore);
                        tokio::spawn(async move {
                            let _permit = semaphore.acquire_owned().await;
                            // RunNow intentionally skips retry — the caller wants the raw result
                            // immediately, not to wait potentially minutes for retries to exhaust.
                            let result = JobExecutor::run(&job).await;
                            update_store_after_run(&store, &job.id, result.clone(), false).await;
                            let _ = reply.send(Ok(result));
                        });
                    }
                }
            }
        }
    }

    async fn fire_due_jobs(&self) {
        let now = Utc::now();

        // Collect all due jobs — the semaphore enforces the actual concurrency cap.
        let due: Vec<Job> = {
            let store = self.store.read().await;
            store
                .all()
                .into_iter()
                .filter(|j| j.enabled && is_due(j, now))
                .cloned()
                .collect()
        };

        for job in due {
            // Try to acquire a concurrency slot without blocking.
            // If all slots are taken, the job stays "due" and is retried on the next tick.
            let permit = match Arc::clone(&self.semaphore).try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    tracing::debug!(
                        "concurrency limit reached — job '{}' deferred to next tick",
                        job.name
                    );
                    break;
                }
            };

            tracing::info!("firing job '{}' ({})", job.name, job.id);
            let store = Arc::clone(&self.store);
            let disable_on_failure = matches!(job.failure_policy, FailurePolicy::Disable);

            tokio::spawn(async move {
                let _permit = permit; // released when this task ends
                let result = run_with_retry(&job).await;
                update_store_after_run(&store, &job.id, result, disable_on_failure).await;
            });
        }
    }
}

// ── Retry logic ───────────────────────────────────────────────────────────────

/// Execute a job, applying `FailurePolicy::Retry` if configured.
/// Used for scheduled firings only — `RunNow` calls `JobExecutor::run` directly.
pub(crate) async fn run_with_retry(job: &Job) -> JobResult {
    let mut result = JobExecutor::run(job).await;

    if result.success {
        return result;
    }

    let (max_retries, backoff_secs) = match &job.failure_policy {
        FailurePolicy::Retry {
            max_retries,
            backoff_secs,
        } => (*max_retries, *backoff_secs),
        _ => return result,
    };

    for attempt in 1..=max_retries {
        tracing::info!(
            "job '{}' failed — retrying ({}/{}), backoff {}s",
            job.name,
            attempt,
            max_retries,
            backoff_secs,
        );
        sleep(Duration::from_secs(backoff_secs)).await;
        result = JobExecutor::run(job).await;
        if result.success {
            tracing::info!(
                "job '{}' succeeded on retry {}/{}",
                job.name,
                attempt,
                max_retries
            );
            break;
        }
    }

    result
}

// ── Shared store helpers ──────────────────────────────────────────────────────

async fn update_store_after_run(
    store: &RwLock<JobStore>,
    job_id: &str,
    result: JobResult,
    disable_on_failure: bool,
) {
    let failed = !result.success;
    let mut s = store.write().await;
    if let Some(j) = s.get_mut(job_id) {
        j.last_fired_at = Some(result.started_at);
        j.last_result = Some(result.clone());
        if failed && disable_on_failure {
            j.enabled = false;
            tracing::warn!(
                "job '{}' disabled after failure (FailurePolicy::Disable)",
                j.name
            );
        }
    }
    if let Err(e) = s.write_log(job_id, &result).await {
        tracing::error!("failed to write log for job {job_id}: {e:#}");
    }
    persist(&mut s).await;
}

async fn persist(store: &mut JobStore) {
    if let Err(e) = store.save().await {
        tracing::error!("failed to persist jobs.json: {e:#}");
    }
}

// ── Cron helpers ─────────────────────────────────────────────────────────────

/// Parse a 5-field or 7-field cron expression, returning `None` on invalid input.
pub(crate) fn parse_cron(expr: &str) -> Option<Schedule> {
    Schedule::from_str(expr)
        .ok()
        .or_else(|| Schedule::from_str(&format!("0 {expr}")).ok())
}

/// Next scheduled fire time after `now` for the given cron expression.
pub fn next_fire(expr: &str) -> Option<DateTime<Utc>> {
    parse_cron(expr)?.upcoming(Utc).next()
}

/// Next scheduled fire time strictly after `from`.
pub fn next_fire_after(expr: &str, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
    parse_cron(expr)?.after(&from).next()
}

/// Is `job` due to run at `now`?
pub(crate) fn is_due(job: &Job, now: DateTime<Utc>) -> bool {
    let last = job.last_fired_at.unwrap_or(job.created_at);
    next_fire_after(&job.cron, last)
        .map(|t| t <= now)
        .unwrap_or(false)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::FailurePolicy;

    fn make_job(cron: &str, last_fired_at: Option<DateTime<Utc>>) -> Job {
        Job {
            id: "test-id".into(),
            name: "test job".into(),
            cron: cron.into(),
            command: "echo".into(),
            args: vec!["hello".into()],
            working_dir: "/tmp".into(),
            enabled: true,
            timeout_secs: 60,
            failure_policy: FailurePolicy::Ignore,
            sandbox: None,
            env: Default::default(),
            created_at: Utc::now(),
            last_fired_at,
            last_result: None,
        }
    }

    // ── Cron parsing ──────────────────────────────────────────────────────────

    #[test]
    fn five_field_cron_parses() {
        assert!(parse_cron("* * * * *").is_some());
        assert!(parse_cron("0 9 * * 1-5").is_some());
        assert!(parse_cron("*/15 * * * *").is_some());
        assert!(parse_cron("30 2 * * *").is_some());
        assert!(parse_cron("0 0 1 * *").is_some());
    }

    #[test]
    fn seven_field_cron_parses() {
        assert!(parse_cron("0 * * * * *").is_some());
        assert!(parse_cron("0 0 9 * * 1-5").is_some());
    }

    #[test]
    fn invalid_cron_returns_none() {
        assert!(parse_cron("not a cron expression").is_none());
        assert!(parse_cron("99 99 99 99 99").is_none());
        assert!(parse_cron("").is_none());
    }

    #[test]
    fn five_field_and_seven_field_agree_on_next_fire() {
        let five = next_fire("0 9 * * *").unwrap();
        let seven = next_fire("0 0 9 * * *").unwrap();
        assert_eq!(five, seven);
    }

    // ── next_fire / next_fire_after ───────────────────────────────────────────

    #[test]
    fn next_fire_is_in_the_future() {
        assert!(next_fire("* * * * *").unwrap() > Utc::now());
    }

    #[test]
    fn next_fire_after_is_strictly_after_base() {
        let base = Utc::now();
        assert!(next_fire_after("* * * * *", base).unwrap() > base);
    }

    #[test]
    fn next_fire_after_hourly_is_within_one_hour() {
        let base = Utc::now();
        let t = next_fire_after("0 * * * *", base).unwrap();
        let diff = (t - base).num_seconds();
        assert!(diff > 0 && diff <= 3600);
    }

    // ── is_due ────────────────────────────────────────────────────────────────

    #[test]
    fn new_job_is_not_immediately_due() {
        let now = Utc::now();
        let mut job = make_job("* * * * *", None);
        job.last_fired_at = Some(now);
        assert!(!is_due(&job, now));
    }

    #[test]
    fn job_is_due_after_tick_passes() {
        let last = Utc::now() - chrono::Duration::seconds(120);
        let job = make_job("* * * * *", Some(last));
        assert!(is_due(&job, Utc::now()));
    }

    #[test]
    fn daily_job_not_due_again_within_same_day() {
        // Use a fixed reference time far from the cron's 02:00 fire, otherwise
        // the test is flaky: CI that runs at, say, 02:23 UTC sees `last_fired`
        // 30 min earlier (01:53) with the 02:00 window crossed between `last`
        // and `now`, which correctly marks the job as due.
        use chrono::TimeZone;
        let now = Utc.with_ymd_and_hms(2026, 1, 15, 12, 0, 0).unwrap(); // noon UTC
        let last = now - chrono::Duration::minutes(30); // 11:30 UTC — no 02:00 crossing
        let job = make_job("0 2 * * *", Some(last));
        assert!(!is_due(&job, now));
    }

    #[test]
    fn job_with_no_last_fired_uses_created_at() {
        let job = make_job("* * * * *", None);
        assert!(!is_due(&job, Utc::now()));
    }

    #[test]
    fn job_with_old_created_at_and_no_last_fired_is_due() {
        let mut job = make_job("* * * * *", None);
        job.created_at = Utc::now() - chrono::Duration::seconds(120);
        assert!(is_due(&job, Utc::now()));
    }

    // ── run_with_retry ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn retry_succeeds_on_second_attempt() {
        // A counter file lets the command fail once then succeed.
        let counter = std::env::temp_dir().join(format!("bw-retry-{}.txt", uuid::Uuid::new_v4()));
        let p = counter.to_string_lossy().to_string();

        let mut job = make_job("* * * * *", None);
        job.command = "sh".into();
        // Increment counter; exit 0 only when count reaches 2.
        job.args = vec![
            "-c".into(),
            format!("N=$(cat {p} 2>/dev/null || echo 0); N=$((N+1)); echo $N > {p}; [ $N -ge 2 ]"),
        ];
        job.failure_policy = FailurePolicy::Retry {
            max_retries: 3,
            backoff_secs: 0,
        };

        let result = run_with_retry(&job).await;
        let _ = tokio::fs::remove_file(&counter).await;

        assert!(result.success, "should succeed on the second attempt");
    }

    #[tokio::test]
    async fn retry_all_attempts_exhausted_returns_failure() {
        let mut job = make_job("* * * * *", None);
        job.command = "false".into();
        job.args = vec![];
        job.failure_policy = FailurePolicy::Retry {
            max_retries: 2,
            backoff_secs: 0,
        };

        let result = run_with_retry(&job).await;
        assert!(!result.success, "should fail after exhausting all retries");
    }

    #[tokio::test]
    async fn ignore_policy_runs_exactly_once() {
        // With Ignore, run_with_retry must not retry — it should return quickly.
        let mut job = make_job("* * * * *", None);
        job.command = "false".into();
        job.args = vec![];
        job.failure_policy = FailurePolicy::Ignore;

        let start = std::time::Instant::now();
        let result = run_with_retry(&job).await;

        assert!(!result.success);
        // No retry sleeps — should finish in well under a second
        assert!(
            start.elapsed().as_secs() < 2,
            "Ignore policy must not add retry delays"
        );
    }

    #[tokio::test]
    async fn disable_policy_runs_exactly_once() {
        // Disable also does not retry — only the daemon loop applies the disable action.
        let mut job = make_job("* * * * *", None);
        job.command = "false".into();
        job.failure_policy = FailurePolicy::Disable;

        let start = std::time::Instant::now();
        let result = run_with_retry(&job).await;

        assert!(!result.success);
        assert!(
            start.elapsed().as_secs() < 2,
            "Disable policy must not retry"
        );
    }

    #[tokio::test]
    async fn successful_job_runs_exactly_once_regardless_of_policy() {
        let mut job = make_job("* * * * *", None);
        // Count invocations via a temp file
        let counter = std::env::temp_dir().join(format!("bw-once-{}.txt", uuid::Uuid::new_v4()));
        let p = counter.to_string_lossy().to_string();
        job.command = "sh".into();
        job.args = vec![
            "-c".into(),
            format!("N=$(cat {p} 2>/dev/null || echo 0); echo $((N+1)) > {p}"),
        ];
        job.failure_policy = FailurePolicy::Retry {
            max_retries: 5,
            backoff_secs: 0,
        };

        let result = run_with_retry(&job).await;
        let count: u32 = tokio::fs::read_to_string(&counter)
            .await
            .unwrap_or_default()
            .trim()
            .parse()
            .unwrap_or(0);
        let _ = tokio::fs::remove_file(&counter).await;

        assert!(result.success);
        assert_eq!(
            count, 1,
            "successful job should run exactly once, not retry"
        );
    }
}
