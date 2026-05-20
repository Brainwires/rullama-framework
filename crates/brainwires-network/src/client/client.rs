use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::error::AgentNetworkClientError;
use super::protocol;

/// MCP relay client that communicates with a subprocess over stdio.
pub struct AgentNetworkClient {
    /// Child process handle.
    child: Child,
    /// Buffered writer to the child's stdin.
    stdin: BufWriter<ChildStdin>,
    /// Buffered reader from the child's stdout.
    stdout: BufReader<ChildStdout>,
    /// Monotonically increasing request ID counter.
    request_id: AtomicU64,
    /// Whether the initialize handshake has completed.
    initialized: bool,
}

impl AgentNetworkClient {
    /// Connect to a relay process using default MCP server arguments.
    pub async fn connect(binary_path: &str) -> Result<Self, AgentNetworkClientError> {
        Self::connect_with_args(binary_path, &["chat", "--mcp-server"]).await
    }

    /// Connect to a relay process with custom arguments.
    pub async fn connect_with_args(
        binary_path: &str,
        args: &[&str],
    ) -> Result<Self, AgentNetworkClientError> {
        let mut child = Command::new(binary_path)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(AgentNetworkClientError::SpawnFailed)?;

        let stdin = child.stdin.take().ok_or_else(|| {
            AgentNetworkClientError::Protocol("Failed to capture stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            AgentNetworkClientError::Protocol("Failed to capture stdout".to_string())
        })?;

        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            request_id: AtomicU64::new(1),
            initialized: false,
        })
    }

    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a JSON-RPC request and read the response.
    pub async fn send_request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, AgentNetworkClientError> {
        let id = self.next_id();
        let request = brainwires_mcp_client::JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: serde_json::json!(id),
            method: method.to_string(),
            params,
        };

        let json = serde_json::to_string(&request)?;
        self.stdin
            .write_all(format!("{json}\n").as_bytes())
            .await
            .map_err(AgentNetworkClientError::Io)?;
        self.stdin
            .flush()
            .await
            .map_err(AgentNetworkClientError::Io)?;

        // Read response
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .await
            .map_err(AgentNetworkClientError::Io)?;

        if bytes == 0 {
            return Err(AgentNetworkClientError::ProcessExited);
        }

        let response = protocol::parse_response(line.trim())?;
        protocol::extract_result(response)
    }

    /// Perform the MCP initialize handshake with the relay process.
    pub async fn initialize(&mut self) -> Result<serde_json::Value, AgentNetworkClientError> {
        let id = self.next_id();
        let request = protocol::build_initialize_request(id);

        let json = serde_json::to_string(&request)?;
        self.stdin
            .write_all(format!("{json}\n").as_bytes())
            .await
            .map_err(AgentNetworkClientError::Io)?;
        self.stdin
            .flush()
            .await
            .map_err(AgentNetworkClientError::Io)?;

        // Read initialize response
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .await
            .map_err(AgentNetworkClientError::Io)?;

        if bytes == 0 {
            return Err(AgentNetworkClientError::ProcessExited);
        }

        let response = protocol::parse_response(line.trim())?;
        let result = protocol::extract_result(response)?;

        // Send initialized notification
        let notif = protocol::build_initialized_notification();
        self.stdin
            .write_all(format!("{notif}\n").as_bytes())
            .await
            .map_err(AgentNetworkClientError::Io)?;
        self.stdin
            .flush()
            .await
            .map_err(AgentNetworkClientError::Io)?;

        self.initialized = true;
        Ok(result)
    }

    /// Call a tool on the relay server by name with the given arguments.
    pub async fn call_tool(
        &mut self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, AgentNetworkClientError> {
        if !self.initialized {
            return Err(AgentNetworkClientError::NotInitialized);
        }

        self.send_request(
            "tools/call",
            Some(serde_json::json!({
                "name": name,
                "arguments": args
            })),
        )
        .await
    }

    /// List all tools available on the relay server.
    pub async fn list_tools(&mut self) -> Result<serde_json::Value, AgentNetworkClientError> {
        if !self.initialized {
            return Err(AgentNetworkClientError::NotInitialized);
        }
        self.send_request("tools/list", None).await
    }

    /// Shut down the relay client and terminate the child process.
    pub async fn shutdown(mut self) -> Result<(), AgentNetworkClientError> {
        // Close stdin to signal EOF to the child process
        drop(self.stdin);
        // Wait for child to exit (with timeout)
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), self.child.wait()).await;
        Ok(())
    }

    /// Check whether the client has completed initialization.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}
