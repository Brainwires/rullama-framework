//! Time Tracking
//!
//! Types and functions for task time tracking and statistics.

use brainwires_core::TaskStatus;

/// Statistics about tasks
#[derive(Debug, Clone, Default)]
pub struct TaskStats {
    /// Total number of tasks.
    pub total: usize,
    /// Number of pending tasks.
    pub pending: usize,
    /// Number of in-progress tasks.
    pub in_progress: usize,
    /// Number of completed tasks.
    pub completed: usize,
    /// Number of failed tasks.
    pub failed: usize,
    /// Number of blocked tasks.
    pub blocked: usize,
    /// Number of skipped tasks.
    pub skipped: usize,
}

/// Time tracking information for a single task
#[derive(Debug, Clone)]
pub struct TaskTimeInfo {
    /// Task identifier.
    pub task_id: String,
    /// Human-readable task description.
    pub description: String,
    /// Current task status.
    pub status: TaskStatus,
    /// Unix timestamp when the task started.
    pub started_at: Option<i64>,
    /// Unix timestamp when the task completed.
    pub completed_at: Option<i64>,
    /// Total duration in seconds (for completed tasks).
    pub duration_secs: Option<i64>,
    /// Elapsed seconds since start (for in-progress tasks).
    pub elapsed_secs: Option<i64>,
}

impl TaskTimeInfo {
    /// Format duration as human-readable string
    pub fn format_duration(&self) -> String {
        if let Some(duration) = self.duration_secs {
            format_duration_secs(duration)
        } else if let Some(elapsed) = self.elapsed_secs {
            format!("{}...", format_duration_secs(elapsed))
        } else {
            "-".to_string()
        }
    }
}

/// Time statistics for all tasks
#[derive(Debug, Clone, Default)]
pub struct TimeStats {
    /// Sum of all completed task durations in seconds.
    pub total_duration_secs: i64,
    /// Number of completed tasks.
    pub completed_tasks: usize,
    /// Average task duration in seconds.
    pub average_duration_secs: Option<i64>,
    /// Total elapsed time for in-progress tasks in seconds.
    pub current_elapsed_secs: i64,
    /// Number of currently in-progress tasks.
    pub in_progress_tasks: usize,
}

impl TimeStats {
    /// Format total duration as human-readable string
    pub fn format_total(&self) -> String {
        format_duration_secs(self.total_duration_secs)
    }

    /// Format average duration as human-readable string
    pub fn format_average(&self) -> String {
        if let Some(avg) = self.average_duration_secs {
            format_duration_secs(avg)
        } else {
            "-".to_string()
        }
    }

    /// Format current elapsed time (in-progress tasks)
    pub fn format_elapsed(&self) -> String {
        format_duration_secs(self.current_elapsed_secs)
    }
}

/// Format duration in seconds as human-readable string (e.g., "2m 34s", "1h 5m")
pub fn format_duration_secs(secs: i64) -> String {
    if secs < 0 {
        return "-".to_string();
    }

    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}
