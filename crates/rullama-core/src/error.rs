//! Framework error types

use thiserror::Error;

/// Core framework errors
#[derive(Error, Debug)]
pub enum FrameworkError {
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Provider error.
    #[error("Provider error: {0}")]
    Provider(String),

    /// Provider authentication failure.
    #[error("Provider authentication failed for {provider}: {message}")]
    ProviderAuth {
        /// Provider name that failed authentication.
        provider: String,
        /// Detailed error message.
        message: String,
    },

    /// Provider model error.
    #[error("Provider model error ({provider}/{model}): {message}")]
    ProviderModel {
        /// Provider name.
        provider: String,
        /// Model identifier.
        model: String,
        /// Detailed error message.
        message: String,
    },

    /// Embedding dimension mismatch between expected and actual vectors.
    #[error("Embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimension {
        /// Expected dimension count.
        expected: usize,
        /// Actual dimension count received.
        got: usize,
    },

    /// Tool execution error.
    #[error("Tool execution error: {0}")]
    ToolExecution(String),

    /// Agent error.
    #[error("Agent error: {0}")]
    Agent(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Storage schema mismatch or migration error.
    #[error("Storage schema error in {store}: {message}")]
    StorageSchema {
        /// Name of the storage store.
        store: String,
        /// Detailed error message.
        message: String,
    },

    /// Training configuration error.
    #[error("Training configuration error for {parameter}: {message}")]
    TrainingConfig {
        /// Parameter name that is invalid.
        parameter: String,
        /// Detailed error message.
        message: String,
    },

    /// Permission denied error.
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Catch-all for other errors.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl FrameworkError {
    /// Create a provider authentication error
    pub fn provider_auth(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ProviderAuth {
            provider: provider.into(),
            message: message.into(),
        }
    }

    /// Create a provider model error
    pub fn provider_model(
        provider: impl Into<String>,
        model: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::ProviderModel {
            provider: provider.into(),
            model: model.into(),
            message: message.into(),
        }
    }

    /// Create an embedding dimension mismatch error
    pub fn embedding_dimension(expected: usize, got: usize) -> Self {
        Self::EmbeddingDimension { expected, got }
    }

    /// Create a storage schema error
    pub fn storage_schema(store: impl Into<String>, message: impl Into<String>) -> Self {
        Self::StorageSchema {
            store: store.into(),
            message: message.into(),
        }
    }

    /// Create a training configuration error
    pub fn training_config(parameter: impl Into<String>, message: impl Into<String>) -> Self {
        Self::TrainingConfig {
            parameter: parameter.into(),
            message: message.into(),
        }
    }
}

/// Result type alias using FrameworkError
pub type FrameworkResult<T> = Result<T, FrameworkError>;
