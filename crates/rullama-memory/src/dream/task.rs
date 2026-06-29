//! Scheduled dream task — wraps [`DreamConsolidator`](crate::dream::consolidator::DreamConsolidator) for periodic execution.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Mutex;

use super::consolidator::{DreamConsolidator, DreamSessionStore};
use super::metrics::DreamReport;

/// A scheduled task that runs the dream consolidation cycle.
///
/// Designed to be driven by the autonomy scheduler's cron system or called
/// manually for on-demand consolidation.
pub struct DreamTask {
    consolidator: Arc<Mutex<DreamConsolidator>>,
    session_store: Arc<dyn DreamSessionStore>,
    /// Cron expression controlling how often the dream cycle runs
    /// (e.g. `"0 3 * * *"` for 3 AM daily).
    cron_expr: String,
}

impl DreamTask {
    /// Create a new dream task.
    pub fn new(
        consolidator: Arc<Mutex<DreamConsolidator>>,
        session_store: Arc<dyn DreamSessionStore>,
        cron_expr: impl Into<String>,
    ) -> Self {
        Self {
            consolidator,
            session_store,
            cron_expr: cron_expr.into(),
        }
    }

    /// Run one consolidation cycle immediately.
    pub async fn run_once(&self) -> Result<DreamReport> {
        let mut consolidator = self.consolidator.lock().await;
        consolidator.run_cycle(&*self.session_store).await
    }

    /// Return the cron expression for this task.
    pub fn cron_expr(&self) -> &str {
        &self.cron_expr
    }
}
