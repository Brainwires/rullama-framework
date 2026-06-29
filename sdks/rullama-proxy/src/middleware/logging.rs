//! Structured JSONL traffic logging middleware.

use crate::error::ProxyResult;
use crate::middleware::{LayerAction, ProxyLayer};
use crate::types::{ProxyRequest, ProxyResponse};

const DEFAULT_MAX_BODY_LOG_BYTES: usize = 4096;

/// Logs request/response pairs as structured JSON via `tracing`.
pub struct LoggingLayer {
    /// Whether to include body content in logs.
    pub log_bodies: bool,
    /// Maximum body bytes to include in log output.
    pub max_body_log_bytes: usize,
}

impl LoggingLayer {
    pub fn new() -> Self {
        Self {
            log_bodies: false,
            max_body_log_bytes: DEFAULT_MAX_BODY_LOG_BYTES,
        }
    }

    pub fn with_bodies(mut self, enabled: bool) -> Self {
        self.log_bodies = enabled;
        self
    }

    pub fn with_max_body_bytes(mut self, max: usize) -> Self {
        self.max_body_log_bytes = max;
        self
    }
}

impl Default for LoggingLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ProxyLayer for LoggingLayer {
    async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction> {
        let body_preview = if self.log_bodies {
            let bytes = request.body.as_bytes();
            let len = bytes.len().min(self.max_body_log_bytes);
            String::from_utf8_lossy(&bytes[..len]).into_owned()
        } else {
            format!("[{} bytes]", request.body.len())
        };

        tracing::info!(
            request_id = %request.id,
            method = %request.method,
            uri = %request.uri,
            transport = ?request.transport,
            body = %body_preview,
            "proxy request"
        );

        Ok(LayerAction::Forward(request))
    }

    async fn on_response(&self, response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        let body_preview = if self.log_bodies {
            let bytes = response.body.as_bytes();
            let len = bytes.len().min(self.max_body_log_bytes);
            String::from_utf8_lossy(&bytes[..len]).into_owned()
        } else {
            format!("[{} bytes]", response.body.len())
        };

        tracing::info!(
            request_id = %response.id,
            status = %response.status,
            body = %body_preview,
            "proxy response"
        );

        Ok(response)
    }

    fn name(&self) -> &str {
        "logging"
    }
}
