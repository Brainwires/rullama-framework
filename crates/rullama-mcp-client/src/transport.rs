use anyhow::{Context, Result};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::types::{JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};

/// Transport layer for MCP communication
#[derive(Debug)]
pub enum Transport {
    /// Standard I/O transport.
    Stdio(StdioTransport),
    /// Stateless HTTP transport (MCP 2026 spec).
    #[cfg(feature = "http")]
    Http(HttpTransport),
}

impl Transport {
    /// Send a JSON-RPC request
    pub async fn send_request(&mut self, request: &JsonRpcRequest) -> Result<()> {
        match self {
            Transport::Stdio(transport) => transport.send_request(request).await,
            #[cfg(feature = "http")]
            Transport::Http(transport) => transport.send_request(request).await,
        }
    }

    /// Receive a JSON-RPC response
    pub async fn receive_response(&mut self) -> Result<JsonRpcResponse> {
        match self {
            Transport::Stdio(transport) => transport.receive_response().await,
            #[cfg(feature = "http")]
            Transport::Http(transport) => transport.receive_response().await,
        }
    }

    /// Receive any JSON-RPC message (response or notification)
    /// This is used for bidirectional communication where servers can send notifications
    pub async fn receive_message(&mut self) -> Result<JsonRpcMessage> {
        match self {
            Transport::Stdio(transport) => transport.receive_message().await,
            #[cfg(feature = "http")]
            Transport::Http(transport) => transport.receive_message().await,
        }
    }

    /// Close the transport
    pub async fn close(&mut self) -> Result<()> {
        match self {
            Transport::Stdio(transport) => transport.close().await,
            #[cfg(feature = "http")]
            Transport::Http(_) => Ok(()), // HTTP is stateless; nothing to close
        }
    }
}

/// Stdio transport for communicating with MCP servers via stdin/stdout
#[derive(Debug)]
pub struct StdioTransport {
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    child: Arc<Mutex<Child>>,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a command
    pub async fn new(command: &str, args: &[String]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context(format!("Failed to spawn MCP server: {}", command))?;

        let stdin = child.stdin.take().context("Failed to get stdin handle")?;

        let stdout = child.stdout.take().context("Failed to get stdout handle")?;

        Ok(Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
            child: Arc::new(Mutex::new(child)),
        })
    }

    /// Send a JSON-RPC request via stdin
    pub async fn send_request(&mut self, request: &JsonRpcRequest) -> Result<()> {
        let json =
            serde_json::to_string(request).context("Failed to serialize JSON-RPC request")?;

        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(json.as_bytes())
            .await
            .context("Failed to write to stdin")?;
        stdin
            .write_all(b"\n")
            .await
            .context("Failed to write newline")?;
        stdin.flush().await.context("Failed to flush stdin")?;

        Ok(())
    }

    /// Receive a JSON-RPC response from stdout
    pub async fn receive_response(&mut self) -> Result<JsonRpcResponse> {
        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();

        stdout
            .read_line(&mut line)
            .await
            .context("Failed to read from stdout")?;

        if line.is_empty() {
            anyhow::bail!("EOF reached, server closed");
        }

        serde_json::from_str(&line).context("Failed to parse JSON-RPC response")
    }

    /// Receive any JSON-RPC message from stdout (response or notification)
    /// Discriminates based on presence of "id" field:
    /// - If "id" is present and not null: Response
    /// - If "id" is missing or null: Notification
    pub async fn receive_message(&mut self) -> Result<JsonRpcMessage> {
        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();

        match stdout.read_line(&mut line).await {
            Ok(0) => {
                // EOF - server closed connection
                anyhow::bail!("MCP server closed connection (EOF on stdout)");
            }
            Ok(_) => {
                // Successfully read a line
            }
            Err(e) => {
                // Check for specific error types
                let error_msg = if e.kind() == std::io::ErrorKind::BrokenPipe {
                    "MCP server process terminated unexpectedly (broken pipe). The server may have crashed during tool execution. Check stderr output for panic messages.".to_string()
                } else if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    "MCP server process exited unexpectedly (unexpected EOF)".to_string()
                } else {
                    format!(
                        "Failed to read from MCP server stdout: {} (kind: {:?})",
                        e,
                        e.kind()
                    )
                };
                anyhow::bail!("{}", error_msg);
            }
        }

        if line.is_empty() {
            anyhow::bail!("MCP server returned empty response");
        }

        // Parse as generic JSON first to check structure
        let value: serde_json::Value =
            serde_json::from_str(&line).context("Failed to parse JSON-RPC message")?;

        // Discriminate based on "id" field
        // Responses have a non-null "id", notifications either lack "id" or have null
        let has_valid_id = value.get("id").map(|id| !id.is_null()).unwrap_or(false);

        if has_valid_id {
            // This is a response
            let response: JsonRpcResponse =
                serde_json::from_value(value).context("Failed to parse as JSON-RPC response")?;
            Ok(JsonRpcMessage::Response(response))
        } else {
            // This is a notification
            let notification: JsonRpcNotification = serde_json::from_value(value)
                .context("Failed to parse as JSON-RPC notification")?;
            Ok(JsonRpcMessage::Notification(notification))
        }
    }

    /// Close the transport and kill the child process
    pub async fn close(&mut self) -> Result<()> {
        let mut child = self.child.lock().await;

        child
            .kill()
            .await
            .context("Failed to kill MCP server process")?;

        Ok(())
    }
}

// ── HTTP client transport ──────────────────────────────────────────────────

/// Stateless HTTP transport for MCP clients (MCP 2026 spec, Streamable HTTP).
///
/// Each JSON-RPC round-trip is a single `POST /mcp` request. The pending
/// request body is buffered in `pending`; `receive_response` / `receive_message`
/// flush it by posting to the server and awaiting the HTTP response.
///
/// Server-initiated notifications arrive via `GET /mcp/events` (SSE), but
/// for the stateless request/response pattern used by most MCP clients only
/// the POST endpoint is required.
#[cfg(feature = "http")]
#[derive(Debug)]
pub struct HttpTransport {
    /// reqwest client (shared, keep-alive).
    client: reqwest::Client,
    /// Full URL of the `POST /mcp` endpoint.
    mcp_url: String,
    /// Buffered outgoing request body (set by `send_request`, consumed by `receive_*`).
    pending: Option<String>,
}

#[cfg(feature = "http")]
impl HttpTransport {
    /// Create a new HTTP transport pointing at the given base URL.
    ///
    /// # Arguments
    /// * `base_url` — server base URL, e.g. `"http://127.0.0.1:3001"`.
    ///   The `/mcp` path is appended automatically.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base = base_url.into().trim_end_matches('/').to_string();
        Self {
            client: reqwest::Client::new(),
            mcp_url: format!("{}/mcp", base),
            pending: None,
        }
    }

    /// Buffer the serialised request; the actual POST happens in `receive_*`.
    pub async fn send_request(&mut self, request: &JsonRpcRequest) -> Result<()> {
        let body =
            serde_json::to_string(request).context("Failed to serialize JSON-RPC request")?;
        self.pending = Some(body);
        Ok(())
    }

    /// POST the buffered request and parse the response body as [`JsonRpcResponse`].
    pub async fn receive_response(&mut self) -> Result<JsonRpcResponse> {
        let msg = self.post_and_receive().await?;
        match msg {
            JsonRpcMessage::Response(r) => Ok(r),
            JsonRpcMessage::Notification(_) => {
                anyhow::bail!("Expected JSON-RPC response, got notification")
            }
        }
    }

    /// POST the buffered request and return the raw [`JsonRpcMessage`].
    pub async fn receive_message(&mut self) -> Result<JsonRpcMessage> {
        self.post_and_receive().await
    }

    async fn post_and_receive(&mut self) -> Result<JsonRpcMessage> {
        let body = self
            .pending
            .take()
            .context("receive called before send_request")?;

        let resp = self
            .client
            .post(&self.mcp_url)
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .context("HTTP POST to MCP server failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("MCP server returned HTTP {}", resp.status());
        }

        let text = resp
            .text()
            .await
            .context("Failed to read MCP response body")?;
        let value: serde_json::Value =
            serde_json::from_str(&text).context("Failed to parse MCP response as JSON")?;

        let has_valid_id = value.get("id").map(|id| !id.is_null()).unwrap_or(false);
        if has_valid_id {
            let response: JsonRpcResponse =
                serde_json::from_value(value).context("Failed to parse as JSON-RPC response")?;
            Ok(JsonRpcMessage::Response(response))
        } else {
            let notification: JsonRpcNotification = serde_json::from_value(value)
                .context("Failed to parse as JSON-RPC notification")?;
            Ok(JsonRpcMessage::Notification(notification))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_stdio_transport_echo() {
        // Test with echo command (simple test)
        let result = StdioTransport::new("echo", &["test".to_string()]).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_json_rpc_serialization() {
        let request =
            JsonRpcRequest::new(1, "initialize".to_string(), Some(json!({"test": "value"})))
                .unwrap();

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("jsonrpc"));
        assert!(json.contains("2.0"));
        assert!(json.contains("initialize"));
    }
}
