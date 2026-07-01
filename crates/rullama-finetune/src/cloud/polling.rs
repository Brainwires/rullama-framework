use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info};

use super::FineTuneProvider;
use crate::error::TrainingError;
use crate::types::{TrainingJobId, TrainingJobStatus};

/// Default initial polling interval in seconds.
const DEFAULT_POLL_INITIAL_SECS: u64 = 10;
/// Default maximum polling interval in seconds.
const DEFAULT_POLL_MAX_SECS: u64 = 300;
/// Default total polling timeout in seconds (24 hours).
const DEFAULT_POLL_TIMEOUT_SECS: u64 = 86_400;

/// Exponential-backoff job status poller.
pub struct JobPoller {
    /// Initial polling interval.
    pub initial_interval: Duration,
    /// Maximum polling interval.
    pub max_interval: Duration,
    /// Backoff multiplier.
    pub multiplier: f64,
    /// Maximum total polling time before timeout.
    pub timeout: Duration,
}

impl Default for JobPoller {
    fn default() -> Self {
        Self {
            initial_interval: Duration::from_secs(DEFAULT_POLL_INITIAL_SECS),
            max_interval: Duration::from_secs(DEFAULT_POLL_MAX_SECS),
            multiplier: 1.5,
            timeout: Duration::from_secs(DEFAULT_POLL_TIMEOUT_SECS), // 24 hours
        }
    }
}

impl JobPoller {
    /// Create a new job poller with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial polling interval.
    pub fn with_initial_interval(mut self, interval: Duration) -> Self {
        self.initial_interval = interval;
        self
    }

    /// Set the maximum polling interval.
    pub fn with_max_interval(mut self, interval: Duration) -> Self {
        self.max_interval = interval;
        self
    }

    /// Set the overall polling timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Poll a training job until it reaches a terminal state.
    ///
    /// Calls `progress_callback` on each status update with the latest progress.
    pub async fn poll_until_complete<F>(
        &self,
        provider: &dyn FineTuneProvider,
        job_id: &TrainingJobId,
        mut progress_callback: F,
    ) -> Result<TrainingJobStatus, TrainingError>
    where
        F: FnMut(&TrainingJobStatus),
    {
        let start = std::time::Instant::now();
        let mut interval = self.initial_interval;

        loop {
            if start.elapsed() > self.timeout {
                return Err(TrainingError::Other(format!(
                    "Polling timeout after {:?}",
                    self.timeout
                )));
            }

            let status = provider.get_job_status(job_id).await?;

            debug!("Job {} status: {:?}", job_id, status);
            progress_callback(&status);

            if status.is_terminal() {
                info!("Job {} reached terminal state", job_id);
                return Ok(status);
            }

            sleep(interval).await;

            // Exponential backoff
            let next = Duration::from_secs_f64(interval.as_secs_f64() * self.multiplier);
            interval = next.min(self.max_interval);
        }
    }

    /// Poll with a simple logging callback.
    pub async fn poll_with_logging(
        &self,
        provider: &dyn FineTuneProvider,
        job_id: &TrainingJobId,
    ) -> Result<TrainingJobStatus, TrainingError> {
        self.poll_until_complete(provider, job_id, |status| match status {
            TrainingJobStatus::Running { progress } => {
                info!(
                    "Training progress: {:.1}% (step {}/{})",
                    progress.completion_fraction() * 100.0,
                    progress.step,
                    progress.total_steps
                );
            }
            TrainingJobStatus::Queued => info!("Job is queued..."),
            TrainingJobStatus::Validating => info!("Validating files..."),
            status => info!("Status: {:?}", status),
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poller_defaults() {
        let poller = JobPoller::default();
        assert_eq!(poller.initial_interval, Duration::from_secs(10));
        assert_eq!(poller.max_interval, Duration::from_secs(300));
        assert!((poller.multiplier - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_poller_builder() {
        let poller = JobPoller::new()
            .with_initial_interval(Duration::from_secs(5))
            .with_max_interval(Duration::from_secs(60))
            .with_timeout(Duration::from_secs(3600));

        assert_eq!(poller.initial_interval, Duration::from_secs(5));
        assert_eq!(poller.max_interval, Duration::from_secs(60));
        assert_eq!(poller.timeout, Duration::from_secs(3600));
    }
}
