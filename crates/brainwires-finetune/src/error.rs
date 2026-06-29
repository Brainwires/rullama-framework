use thiserror::Error;

/// Errors that can occur during training operations.
#[derive(Error, Debug)]
pub enum TrainingError {
    /// Dataset-related error.
    #[error("Dataset error: {0}")]
    Dataset(#[from] brainwires_datasets::DatasetError),

    /// API request error.
    #[error("API error: {message} (status: {status_code})")]
    Api {
        /// Error message.
        message: String,
        /// HTTP status code.
        status_code: u16,
    },

    /// Provider-specific error.
    #[error("Provider error: {0}")]
    Provider(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Job not found.
    #[error("Job not found: {0}")]
    JobNotFound(String),

    /// Job execution failed.
    #[error("Job failed: {0}")]
    JobFailed(String),

    /// Dataset upload error.
    #[error("Upload error: {0}")]
    Upload(String),

    /// Training backend error.
    #[error("Training backend error: {0}")]
    Backend(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// HTTP request error.
    #[error("HTTP error: {0}")]
    #[cfg(feature = "cloud")]
    Http(#[from] reqwest::Error),

    /// Feature unsupported by a provider.
    #[error("{provider}: {feature} is unsupported")]
    NotImplemented {
        /// Provider name.
        provider: String,
        /// Feature description.
        feature: String,
    },

    /// Other unclassified error.
    #[error("{0}")]
    Other(String),
}

/// Result type alias for training operations.
pub type TrainingResult<T> = Result<T, TrainingError>;
