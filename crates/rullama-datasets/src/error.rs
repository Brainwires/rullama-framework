use thiserror::Error;

/// Errors that can occur during dataset operations.
#[derive(Error, Debug)]
pub enum DatasetError {
    /// I/O error reading or writing data.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization error.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Dataset validation failed.
    #[error("Validation error: {message}")]
    Validation {
        /// Description of the validation failure.
        message: String,
    },

    /// Format conversion error between providers.
    #[error("Format conversion error: {message}")]
    FormatConversion {
        /// Description of the conversion failure.
        message: String,
    },

    /// Tokenizer encoding or decoding error.
    #[error("Tokenizer error: {message}")]
    Tokenizer {
        /// Description of the tokenizer failure.
        message: String,
    },

    /// Index is out of bounds for the dataset.
    #[error("Index out of bounds: {index} (len: {len})")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The actual length of the dataset.
        len: usize,
    },

    /// The dataset is empty.
    #[error("Empty dataset")]
    EmptyDataset,

    /// Generic error with a message.
    #[error("{0}")]
    Other(
        /// Error message.
        String,
    ),
}

/// Convenience alias for `Result<T, DatasetError>`.
pub type DatasetResult<T> = Result<T, DatasetError>;
