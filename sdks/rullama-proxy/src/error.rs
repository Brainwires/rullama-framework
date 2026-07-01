use std::io;

/// Errors that can occur in the proxy framework.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("transport error: {0}")]
    Transport(String),

    #[error("connection failed: {0}")]
    Connection(String),

    #[error("upstream unreachable: {0}")]
    UpstreamUnreachable(String),

    #[error("middleware rejected request: {0}")]
    MiddlewareRejected(String),

    #[error("conversion error: {0}")]
    Conversion(String),

    #[error("format detection failed: no matching format for content")]
    FormatDetectionFailed,

    #[error("unsupported conversion: {src} -> {dst}")]
    UnsupportedConversion { src: String, dst: String },

    #[error("configuration error: {0}")]
    Config(String),

    #[error("timeout after {0:?}")]
    Timeout(std::time::Duration),

    #[error("request too large: {size} bytes exceeds limit of {limit} bytes")]
    RequestTooLarge { size: usize, limit: usize },

    #[error("rate limited")]
    RateLimited,

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] http::Error),

    #[error("shutdown signal received")]
    Shutdown,

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type ProxyResult<T> = Result<T, ProxyError>;
