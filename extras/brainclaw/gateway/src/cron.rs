//! Cron job data types and persistent store.
//!
//! `CronJob` describes a scheduled task; `CronStore` persists jobs as JSON
//! files under a configurable directory.  The background runner lives in the
//! brainclaw daemon crate so it can reference `AgentInboundHandler`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use cron::Schedule;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// A single scheduled cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    /// Unique job identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// 5-field cron expression (min hour day month weekday).
    pub schedule: String,
    /// Prompt text that will be sent to the agent.
    pub prompt: String,
    /// Target platform (e.g. "discord", "telegram", "webchat").
    pub target_platform: String,
    /// Target channel ID within the platform (room, channel, DM peer, etc.).
    pub target_channel_id: String,
    /// Target user ID used to look up or create an agent session.
    pub target_user_id: String,
    /// Whether the job is active.
    pub enabled: bool,
    /// UTC timestamp of the last execution (None if never run).
    pub last_run: Option<DateTime<Utc>>,
}

impl CronJob {
    /// Parse this job's schedule and return the next scheduled run after `after`.
    pub fn next_run_after(&self, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
        // Prepend a wildcard seconds field so the cron crate gets a 6-field expression.
        let expr = format!("0 {}", self.schedule);
        let sched = Schedule::from_str(&expr).ok()?;
        sched.after(&after).next()
    }

    /// Validate the cron expression. Returns `Ok(())` or an error message.
    pub fn validate_schedule(expr: &str) -> Result<()> {
        let full = format!("0 {expr}");
        Schedule::from_str(&full)
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("Invalid cron expression: {e}"))
    }
}

/// Persistent store for cron jobs backed by JSON files.
pub struct CronStore {
    dir: PathBuf,
    jobs: RwLock<HashMap<Uuid, CronJob>>,
}

impl CronStore {
    /// Open (or create) the store at `dir`.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create cron dir: {}", dir.display()))?;

        let mut jobs = HashMap::new();
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("Failed to read cron dir: {}", dir.display()))?
        {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                match std::fs::read_to_string(&path)
                    .and_then(|s| Ok(serde_json::from_str::<CronJob>(&s)?))
                {
                    Ok(job) => {
                        jobs.insert(job.id, job);
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "Failed to load cron job");
                    }
                }
            }
        }

        tracing::info!(count = jobs.len(), path = %dir.display(), "Cron jobs loaded");

        Ok(Self {
            dir,
            jobs: RwLock::new(jobs),
        })
    }

    /// Return a snapshot of all jobs.
    pub async fn list(&self) -> Vec<CronJob> {
        let mut jobs: Vec<CronJob> = self.jobs.read().await.values().cloned().collect();
        jobs.sort_by_key(|j| j.name.clone());
        jobs
    }

    /// Get a single job by ID.
    pub async fn get(&self, id: Uuid) -> Option<CronJob> {
        self.jobs.read().await.get(&id).cloned()
    }

    /// Add or replace a job and persist it to disk.
    pub async fn upsert(&self, job: CronJob) -> Result<()> {
        let path = self.job_path(job.id);
        let json = serde_json::to_string_pretty(&job).context("Failed to serialize cron job")?;
        std::fs::write(&path, json)
            .with_context(|| format!("Failed to write cron job: {}", path.display()))?;
        self.jobs.write().await.insert(job.id, job);
        Ok(())
    }

    /// Delete a job and remove its file.
    pub async fn delete(&self, id: Uuid) -> Result<bool> {
        let removed = self.jobs.write().await.remove(&id).is_some();
        if removed {
            let path = self.job_path(id);
            if path.exists() {
                std::fs::remove_file(&path).with_context(|| {
                    format!("Failed to delete cron job file: {}", path.display())
                })?;
            }
        }
        Ok(removed)
    }

    /// Update `last_run` for a job and persist the change.
    pub async fn record_run(&self, id: Uuid, at: DateTime<Utc>) -> Result<()> {
        if let Some(job) = self.jobs.write().await.get_mut(&id) {
            job.last_run = Some(at);
            let path = self.job_path(id);
            let json = serde_json::to_string_pretty(job).context("Failed to serialize cron job")?;
            std::fs::write(&path, json)
                .with_context(|| format!("Failed to write cron job: {}", path.display()))?;
        }
        Ok(())
    }

    /// Shared accessor (Arc wrapper constructor helper).
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn job_path(&self, id: Uuid) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
}
