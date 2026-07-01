use thiserror::Error;

/// Errors returned by the analytics subsystem — sinks, queries, exporters.
#[derive(Debug, Error)]
pub enum AnalyticsError {
    /// SQLite driver error from the persistent sink or query layer.
    #[cfg(feature = "sqlite")]
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Filesystem / pipe I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The collector's sink channel was dropped before the event could be delivered.
    #[error("Analytics sink channel closed")]
    ChannelClosed,

    /// JSON serialization / deserialization of an event payload failed.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Catch-all for lower-level failures surfaced as `anyhow::Error`.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Convenience result alias used throughout the telemetry crate.
pub type AnalyticsResult<T> = Result<T, AnalyticsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_closed_display() {
        let e = AnalyticsError::ChannelClosed;
        assert!(e.to_string().contains("closed"));
    }

    #[test]
    fn io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let analytics_err: AnalyticsError = io_err.into();
        assert!(analytics_err.to_string().contains("I/O"));
    }

    #[test]
    fn serde_error_from() {
        let serde_err: Result<serde_json::Value, _> = serde_json::from_str("{bad}");
        let analytics_err: AnalyticsError = serde_err.unwrap_err().into();
        assert!(analytics_err.to_string().contains("Serialization"));
    }

    #[test]
    fn other_error_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("custom failure");
        let analytics_err: AnalyticsError = anyhow_err.into();
        assert!(analytics_err.to_string().contains("custom failure"));
    }

    #[test]
    fn analytics_result_ok() {
        let result: AnalyticsResult<i32> = Ok(42);
        assert!(matches!(result, Ok(42)));
    }

    #[test]
    fn analytics_result_err() {
        let result: AnalyticsResult<i32> = Err(AnalyticsError::ChannelClosed);
        assert!(result.is_err());
    }
}
