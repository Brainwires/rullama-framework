//! Structured webhook event logging with daily rotation.
//!
//! Adapted from the brainwires-deploy daemon's `WebhookLogger` with
//! autonomy-specific event actions.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

/// Actions logged for webhook events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WebhookAction {
    /// Event received and investigation started.
    InvestigationStarted,
    /// Investigation completed, fix being applied.
    FixApplied,
    /// PR was created for the fix.
    PrCreated,
    /// Event was a ping (health check).
    Ping,
    /// Event was ignored (not configured or unsupported type).
    Ignored,
    /// Duplicate investigation skipped.
    DuplicateSkipped,
    /// Signature verification failed.
    Unauthorized,
    /// Payload could not be parsed.
    InvalidPayload,
    /// Deployment started (push event).
    DeploymentStarted,
    /// Event received but no matching repo config.
    NotConfigured,
}

impl std::fmt::Display for WebhookAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvestigationStarted => write!(f, "INVESTIGATION_STARTED"),
            Self::FixApplied => write!(f, "FIX_APPLIED"),
            Self::PrCreated => write!(f, "PR_CREATED"),
            Self::Ping => write!(f, "PING"),
            Self::Ignored => write!(f, "IGNORED"),
            Self::DuplicateSkipped => write!(f, "DUPLICATE_SKIPPED"),
            Self::Unauthorized => write!(f, "UNAUTHORIZED"),
            Self::InvalidPayload => write!(f, "INVALID_PAYLOAD"),
            Self::DeploymentStarted => write!(f, "DEPLOYMENT_STARTED"),
            Self::NotConfigured => write!(f, "NOT_CONFIGURED"),
        }
    }
}

/// A logged webhook event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Timestamp of the event.
    pub timestamp: DateTime<Utc>,
    /// GitHub event type (issues, push, etc.).
    pub event_type: String,
    /// Repository name.
    pub repo: String,
    /// Git ref (branch or tag).
    #[serde(default)]
    pub git_ref: String,
    /// Action taken.
    pub action: WebhookAction,
    /// Human-readable message.
    pub message: String,
    /// Client IP address, if available.
    #[serde(default)]
    pub client_ip: Option<String>,
}

/// Logger that writes webhook events to daily-rotated plain-text and JSONL files.
///
/// Automatically creates new log files when the date changes and cleans up
/// files older than the configured retention period.
pub struct WebhookLogger {
    log_dir: PathBuf,
    keep_days: u32,
    current_date: Option<NaiveDate>,
    current_file: Option<File>,
    current_jsonl_file: Option<File>,
}

impl WebhookLogger {
    /// Create a new webhook logger.
    pub fn new(log_dir: PathBuf, keep_days: u32) -> Self {
        Self {
            log_dir,
            keep_days,
            current_date: None,
            current_file: None,
            current_jsonl_file: None,
        }
    }

    /// Initialize the logger: create log directory and clean up old logs.
    pub fn init(&mut self) -> anyhow::Result<()> {
        fs::create_dir_all(&self.log_dir)?;
        self.cleanup_old_logs()?;
        Ok(())
    }

    /// Log a webhook event.
    pub fn log_event(&mut self, event: &WebhookEvent) -> anyhow::Result<()> {
        self.ensure_current_file(&event.timestamp)?;

        // Write plain-text line
        if let Some(ref mut file) = self.current_file {
            writeln!(
                file,
                "[{}] {} {} repo={} ref={} {}",
                event.timestamp.format("%H:%M:%S%.3f"),
                event.action,
                event.event_type,
                event.repo,
                event.git_ref,
                event.message,
            )?;
        }

        // Write JSONL line
        if let Some(ref mut file) = self.current_jsonl_file {
            let json = serde_json::to_string(event)?;
            writeln!(file, "{json}")?;
        }

        Ok(())
    }

    fn ensure_current_file(&mut self, timestamp: &DateTime<Utc>) -> anyhow::Result<()> {
        let date = timestamp.date_naive();
        if self.current_date == Some(date) {
            return Ok(());
        }

        let date_str = date.format("%Y-%m-%d").to_string();

        let log_path = self.log_dir.join(format!("webhooks_{date_str}.log"));
        self.current_file = Some(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)?,
        );

        let jsonl_path = self.log_dir.join(format!("webhooks_{date_str}.jsonl"));
        self.current_jsonl_file = Some(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&jsonl_path)?,
        );

        self.current_date = Some(date);
        Ok(())
    }

    fn cleanup_old_logs(&self) -> anyhow::Result<()> {
        let cutoff = Utc::now().date_naive() - chrono::Duration::days(self.keep_days as i64);

        if let Ok(entries) = fs::read_dir(&self.log_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(date_str) = extract_date_from_filename(&name)
                    && let Ok(date) = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                    && date < cutoff
                {
                    let _ = fs::remove_file(entry.path());
                    tracing::debug!("Cleaned up old webhook log: {name}");
                }
            }
        }

        Ok(())
    }
}

fn extract_date_from_filename(name: &str) -> Option<String> {
    // Pattern: webhooks_YYYY-MM-DD.log or .jsonl
    let name = name.strip_prefix("webhooks_")?;
    let date = name
        .strip_suffix(".log")
        .or_else(|| name.strip_suffix(".jsonl"))?;
    if date.len() == 10 {
        Some(date.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_date_from_log_filename() {
        assert_eq!(
            extract_date_from_filename("webhooks_2026-03-15.log"),
            Some("2026-03-15".to_string())
        );
        assert_eq!(
            extract_date_from_filename("webhooks_2026-03-15.jsonl"),
            Some("2026-03-15".to_string())
        );
        assert_eq!(extract_date_from_filename("other.log"), None);
    }

    #[test]
    fn webhook_event_serialization() {
        let event = WebhookEvent {
            timestamp: Utc::now(),
            event_type: "issues".to_string(),
            repo: "user/repo".to_string(),
            git_ref: String::new(),
            action: WebhookAction::InvestigationStarted,
            message: "Investigating issue #42".to_string(),
            client_ip: Some("127.0.0.1".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: WebhookEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.repo, "user/repo");
    }
}
