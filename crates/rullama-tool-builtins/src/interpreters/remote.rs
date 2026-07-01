//! Remote sandbox execution backend.
//!
//! Delegates code execution to any REST sandbox API that speaks the
//! rullama execution protocol:
//!
//! ```text
//! POST {base_url}/execute
//! Content-Type: application/json
//! Authorization: Bearer <api_key>
//!
//! { "language": "python", "code": "print(1+1)", ... }   ← ExecutionRequest
//!
//! 200 OK
//! { "success": true, "stdout": "2\n", ... }              ← ExecutionResult
//! ```
//!
//! The same wire format is compatible with [E2B](https://e2b.dev),
//! [Modal](https://modal.com) custom endpoints, [Daytona](https://daytona.io)
//! workspaces, and any home-grown sandbox service.
//!
//! # Example
//!
//! ```rust,no_run
//! use rullama_tool_builtins::interpreters::remote::{RemoteSandboxConfig, RemoteSandboxExecutor};
//! use rullama_tool_builtins::interpreters::{ExecutionRequest, Language};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = RemoteSandboxConfig::new("https://sandbox.example.com", "sk-my-api-key");
//! let executor = RemoteSandboxExecutor::new(config)?;
//!
//! let result = executor.execute(ExecutionRequest {
//!     language: Language::JavaScript,
//!     code: "console.log('hello from the cloud')".to_string(),
//!     ..Default::default()
//! }).await?;
//!
//! println!("{}", result.stdout);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

use super::types::{ExecutionRequest, ExecutionResult};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the remote sandbox executor.
#[derive(Debug, Clone)]
pub struct RemoteSandboxConfig {
    /// Base URL of the sandbox API (e.g. `"https://sandbox.example.com"`).
    ///
    /// The executor appends `/execute` to this URL.
    pub base_url: String,

    /// API key sent as `Authorization: Bearer <api_key>`.
    pub api_key: String,

    /// HTTP request timeout (default: 60 s — longer than local execution to
    /// account for cold-start latency on remote runtimes).
    pub timeout: Duration,

    /// Additional HTTP headers to include in every request.
    ///
    /// Useful for vendor-specific metadata (e.g. E2B team IDs, Modal
    /// workspace tokens, custom tracing headers).
    pub extra_headers: HashMap<String, String>,
}

impl RemoteSandboxConfig {
    /// Create a minimal config with a base URL and API key.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            timeout: Duration::from_secs(60),
            extra_headers: HashMap::new(),
        }
    }

    /// Override the HTTP timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add an extra header sent with every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }
}

// ── Executor ──────────────────────────────────────────────────────────────────

/// Async executor that forwards [`ExecutionRequest`]s to a remote REST sandbox.
///
/// A single `RemoteSandboxExecutor` holds a connection-pooled [`reqwest::Client`]
/// and can be cheaply cloned (the client is `Arc`-backed internally).
#[derive(Clone)]
pub struct RemoteSandboxExecutor {
    config: RemoteSandboxConfig,
    client: Client,
}

impl RemoteSandboxExecutor {
    /// Build an executor from the given config.
    ///
    /// Returns an error if the HTTP client cannot be constructed (e.g. bad TLS
    /// configuration on the current platform).
    pub fn new(config: RemoteSandboxConfig) -> anyhow::Result<Self> {
        let mut default_headers = HeaderMap::new();
        default_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let bearer = HeaderValue::from_str(&format!("Bearer {}", config.api_key))
            .map_err(|e| anyhow::anyhow!("Invalid API key (bad header value): {e}"))?;
        default_headers.insert(AUTHORIZATION, bearer);

        for (name, value) in &config.extra_headers {
            let header_name = name
                .parse::<HeaderName>()
                .map_err(|e| anyhow::anyhow!("Invalid header name '{name}': {e}"))?;
            let header_value = HeaderValue::from_str(value)
                .map_err(|e| anyhow::anyhow!("Invalid header value for '{name}': {e}"))?;
            default_headers.insert(header_name, header_value);
        }

        let client = Client::builder()
            .default_headers(default_headers)
            .timeout(config.timeout)
            .build()?;

        Ok(Self { config, client })
    }

    /// Execute code on the remote sandbox.
    ///
    /// POSTs the `request` as JSON to `{base_url}/execute` and deserialises
    /// the response body as an [`ExecutionResult`].
    pub async fn execute(&self, request: ExecutionRequest) -> anyhow::Result<ExecutionResult> {
        let url = format!("{}/execute", self.config.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Remote sandbox request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Remote sandbox returned HTTP {status}: {body}");
        }

        let result: ExecutionResult = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to deserialize sandbox response: {e}"))?;

        Ok(result)
    }

    /// Run a health-check against `{base_url}/health`.
    ///
    /// Returns `true` if the endpoint responds with any 2xx status.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.config.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }

    /// Return the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_trims_trailing_slash() {
        let cfg = RemoteSandboxConfig::new("https://sandbox.example.com/", "key");
        assert_eq!(cfg.base_url, "https://sandbox.example.com");
    }

    #[test]
    fn test_config_default_timeout() {
        let cfg = RemoteSandboxConfig::new("https://sandbox.example.com", "key");
        assert_eq!(cfg.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_config_builder_pattern() {
        let cfg = RemoteSandboxConfig::new("https://sandbox.example.com", "key")
            .with_timeout(Duration::from_secs(120))
            .with_header("X-Team-Id", "team-123");

        assert_eq!(cfg.timeout, Duration::from_secs(120));
        assert_eq!(cfg.extra_headers.get("X-Team-Id").unwrap(), "team-123");
    }

    #[test]
    fn test_executor_construction() {
        let cfg = RemoteSandboxConfig::new("https://sandbox.example.com", "valid-key");
        assert!(RemoteSandboxExecutor::new(cfg).is_ok());
    }

    #[test]
    fn test_executor_base_url() {
        let cfg = RemoteSandboxConfig::new("https://sandbox.example.com", "key");
        let executor = RemoteSandboxExecutor::new(cfg).unwrap();
        assert_eq!(executor.base_url(), "https://sandbox.example.com");
    }
}
