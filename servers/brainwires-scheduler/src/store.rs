use crate::job::{Job, JobResult};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Persistent job registry backed by `jobs.json` with per-job execution logs.
pub struct JobStore {
    jobs_file: PathBuf,
    logs_dir: PathBuf,
    jobs: HashMap<String, Job>,
    /// Maximum number of log files kept per job (older entries are pruned)
    pub log_retention: usize,
}

impl JobStore {
    /// Open (or create) the store at the given directory.
    pub async fn open(jobs_dir: &Path) -> Result<Self> {
        fs::create_dir_all(jobs_dir).await?;
        let logs_dir = jobs_dir.join("logs");
        fs::create_dir_all(&logs_dir).await?;
        let jobs_file = jobs_dir.join("jobs.json");

        let jobs = if jobs_file.exists() {
            let text = fs::read_to_string(&jobs_file)
                .await
                .context("reading jobs.json")?;
            serde_json::from_str::<Vec<Job>>(&text)
                .context("parsing jobs.json")?
                .into_iter()
                .map(|j| (j.id.clone(), j))
                .collect()
        } else {
            HashMap::new()
        };

        Ok(Self {
            jobs_file,
            logs_dir,
            jobs,
            log_retention: 20,
        })
    }

    pub fn get(&self, id: &str) -> Option<&Job> {
        self.jobs.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Job> {
        self.jobs.get_mut(id)
    }

    /// All jobs sorted by creation time.
    pub fn all(&self) -> Vec<&Job> {
        let mut v: Vec<&Job> = self.jobs.values().collect();
        v.sort_by_key(|j| j.created_at);
        v
    }

    pub fn insert(&mut self, job: Job) {
        self.jobs.insert(job.id.clone(), job);
    }

    pub fn remove(&mut self, id: &str) -> Option<Job> {
        self.jobs.remove(id)
    }

    /// Flush the current job list to disk.
    pub async fn save(&self) -> Result<()> {
        let jobs: Vec<&Job> = self.all();
        let text = serde_json::to_string_pretty(&jobs)?;
        fs::write(&self.jobs_file, text).await?;
        Ok(())
    }

    /// Write a single execution result as a timestamped JSON file.
    pub async fn write_log(&self, job_id: &str, result: &JobResult) -> Result<()> {
        let job_log_dir = self.logs_dir.join(job_id);
        fs::create_dir_all(&job_log_dir).await?;

        // Use wall-clock write time for the filename so concurrent or rapid writes
        // never collide even if result.started_at values are identical.
        let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ").to_string();
        let log_file = job_log_dir.join(format!("{ts}.json"));
        fs::write(&log_file, serde_json::to_string_pretty(result)?).await?;

        self.prune_logs(job_id).await
    }

    /// Remove oldest log files beyond the retention limit.
    async fn prune_logs(&self, job_id: &str) -> Result<()> {
        let job_log_dir = self.logs_dir.join(job_id);
        let mut entries = fs::read_dir(&job_log_dir).await?;
        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            files.push(entry.path());
        }
        files.sort();
        if files.len() > self.log_retention {
            for old in &files[..files.len() - self.log_retention] {
                let _ = fs::remove_file(old).await;
            }
        }
        Ok(())
    }

    /// Return the log directory path for a job without doing any I/O.
    /// Callers can capture this while holding the lock, then drop the lock before reading files.
    pub fn log_dir_for(&self, job_id: &str) -> PathBuf {
        self.logs_dir.join(job_id)
    }

    /// Read the most recent `limit` execution logs from a given log directory, newest first.
    /// This is a free function so callers can invoke it *without* holding the store lock.
    pub async fn read_logs_from_dir(log_dir: &Path, limit: usize) -> Result<Vec<JobResult>> {
        if !log_dir.exists() {
            return Ok(Vec::new());
        }
        let mut entries = fs::read_dir(log_dir).await?;
        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            files.push(entry.path());
        }
        files.sort();
        files.reverse();
        files.truncate(limit);

        let mut results = Vec::new();
        for f in files {
            if let Ok(text) = fs::read_to_string(&f).await
                && let Ok(r) = serde_json::from_str::<JobResult>(&text)
            {
                results.push(r);
            }
        }
        Ok(results)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{FailurePolicy, Job, JobResult};
    use chrono::Utc;

    fn temp_dir() -> PathBuf {
        let id = uuid::Uuid::new_v4();
        std::env::temp_dir().join(format!("bw-scheduler-test-{id}"))
    }

    fn sample_job(id: &str) -> Job {
        Job {
            id: id.into(),
            name: format!("Job {id}"),
            cron: "* * * * *".into(),
            command: "echo".into(),
            args: vec![],
            working_dir: "/tmp".into(),
            enabled: true,
            timeout_secs: 60,
            failure_policy: FailurePolicy::Ignore,
            sandbox: None,
            env: Default::default(),
            created_at: Utc::now(),
            last_fired_at: None,
            last_result: None,
        }
    }

    fn sample_result(success: bool) -> JobResult {
        JobResult {
            success,
            exit_code: if success { Some(0) } else { Some(1) },
            stdout: "some output".into(),
            stderr: String::new(),
            started_at: Utc::now(),
            duration_secs: 0.1,
            error: None,
        }
    }

    // ── Round-trip persistence ────────────────────────────────────────────────

    #[tokio::test]
    async fn jobs_survive_reopen() {
        let dir = temp_dir();
        {
            let mut store = JobStore::open(&dir).await.unwrap();
            store.insert(sample_job("alpha"));
            store.insert(sample_job("beta"));
            store.save().await.unwrap();
        }
        // Re-open and check
        let store = JobStore::open(&dir).await.unwrap();
        let ids: Vec<&str> = store.all().iter().map(|j| j.id.as_str()).collect();
        assert!(ids.contains(&"alpha"), "alpha should survive reopen");
        assert!(ids.contains(&"beta"), "beta should survive reopen");
        let _ = fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn empty_store_opens_cleanly() {
        let dir = temp_dir();
        let store = JobStore::open(&dir).await.unwrap();
        assert_eq!(store.all().len(), 0);
        let _ = fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn insert_and_remove() {
        let dir = temp_dir();
        let mut store = JobStore::open(&dir).await.unwrap();
        store.insert(sample_job("x"));
        assert!(store.get("x").is_some());
        store.remove("x");
        assert!(store.get("x").is_none());
        let _ = fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn all_returns_sorted_by_created_at() {
        let dir = temp_dir();
        let mut store = JobStore::open(&dir).await.unwrap();
        let mut j1 = sample_job("first");
        j1.created_at = Utc::now() - chrono::Duration::seconds(10);
        let j2 = sample_job("second"); // created_at ≈ now
        store.insert(j2); // insert out of order
        store.insert(j1);
        let all = store.all();
        assert_eq!(all[0].id, "first", "oldest job should be first");
        assert_eq!(all[1].id, "second");
        let _ = fs::remove_dir_all(&dir).await;
    }

    // ── Log write / read / retention ──────────────────────────────────────────

    #[tokio::test]
    async fn write_and_read_log() {
        let dir = temp_dir();
        let store = JobStore::open(&dir).await.unwrap();
        let result = sample_result(true);
        store.write_log("job1", &result).await.unwrap();

        let logs = JobStore::read_logs_from_dir(&store.log_dir_for("job1"), 5)
            .await
            .unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].success);
        let _ = fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn log_retention_prunes_oldest_files() {
        let dir = temp_dir();
        let mut store = JobStore::open(&dir).await.unwrap();
        store.log_retention = 3;

        let result = sample_result(true);
        for _ in 0..5 {
            // Ensure distinct filenames by sleeping 1 ms between writes
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
            store.write_log("job1", &result).await.unwrap();
        }

        let logs = JobStore::read_logs_from_dir(&store.log_dir_for("job1"), 10)
            .await
            .unwrap();
        assert_eq!(logs.len(), 3, "only the 3 most recent logs should be kept");
        let _ = fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn read_logs_returns_newest_first() {
        let dir = temp_dir();
        let store = JobStore::open(&dir).await.unwrap();

        // Write two results with distinct timestamps
        let mut r1 = sample_result(true);
        r1.started_at = Utc::now() - chrono::Duration::seconds(10);
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        let r2 = sample_result(false); // started_at ≈ now

        store.write_log("j", &r1).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        store.write_log("j", &r2).await.unwrap();

        let logs = JobStore::read_logs_from_dir(&store.log_dir_for("j"), 10)
            .await
            .unwrap();
        assert_eq!(logs.len(), 2);
        // newest first → logs[0] is r2 (failed), logs[1] is r1 (success)
        assert!(!logs[0].success, "newest log should be first");
        assert!(logs[1].success, "oldest log should be second");
        let _ = fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn read_logs_from_nonexistent_dir_returns_empty() {
        let dir = temp_dir().join("does-not-exist");
        let logs = JobStore::read_logs_from_dir(&dir, 10).await.unwrap();
        assert!(logs.is_empty());
    }

    #[tokio::test]
    async fn log_dir_for_returns_correct_path() {
        let dir = temp_dir();
        let store = JobStore::open(&dir).await.unwrap();
        let log_dir = store.log_dir_for("myjob");
        assert!(log_dir.ends_with("logs/myjob"));
        let _ = fs::remove_dir_all(&dir).await;
    }
}
