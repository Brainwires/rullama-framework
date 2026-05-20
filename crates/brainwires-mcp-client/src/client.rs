use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, mpsc};

use crate::config::McpServerConfig;
use crate::transport::{StdioTransport, Transport};
use crate::types::*;

/// MCP Client - manages connections to MCP servers
pub struct McpClient {
    connections: Arc<RwLock<HashMap<String, McpConnection>>>,
    request_id: Arc<AtomicU64>,
    client_name: String,
    client_version: String,
}

/// Active connection to an MCP server
struct McpConnection {
    #[allow(dead_code)]
    server_name: String,
    transport: Transport,
    server_info: ServerInfo,
    capabilities: ServerCapabilities,
    /// Channel for forwarding notifications received during requests
    _notification_tx: Option<mpsc::UnboundedSender<JsonRpcNotification>>,
}

impl McpClient {
    /// Create a new MCP client with the given name and version.
    pub fn new(client_name: impl Into<String>, client_version: impl Into<String>) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            request_id: Arc::new(AtomicU64::new(1)),
            client_name: client_name.into(),
            client_version: client_version.into(),
        }
    }

    /// Connect to an MCP server
    pub async fn connect(&self, config: &McpServerConfig) -> Result<()> {
        // Spawn server process
        let transport = StdioTransport::new(&config.command, &config.args).await?;
        let mut transport = Transport::Stdio(transport);

        // Send initialize request
        let init_result = self.initialize(&mut transport).await?;

        // Create connection
        let connection = McpConnection {
            server_name: config.name.clone(),
            transport,
            server_info: init_result.server_info,
            capabilities: init_result.capabilities,
            _notification_tx: None,
        };

        // Store connection
        self.connections
            .write()
            .await
            .insert(config.name.clone(), connection);

        Ok(())
    }

    /// Disconnect from an MCP server
    pub async fn disconnect(&self, server_name: &str) -> Result<()> {
        let mut connections = self.connections.write().await;
        if let Some(mut connection) = connections.remove(server_name) {
            connection.transport.close().await?;
        }
        Ok(())
    }

    /// Check if connected to a server
    pub async fn is_connected(&self, server_name: &str) -> bool {
        self.connections.read().await.contains_key(server_name)
    }

    /// Get list of connected servers
    pub async fn list_connected(&self) -> Vec<String> {
        self.connections.read().await.keys().cloned().collect()
    }

    /// Initialize handshake with server
    async fn initialize(&self, transport: &mut Transport) -> Result<InitializeResult> {
        let request = JsonRpcRequest::new(
            self.next_request_id(),
            "initialize".to_string(),
            Some(InitializeParams {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ClientCapabilities::default(),
                client_info: ClientInfo {
                    name: self.client_name.clone(),
                    version: self.client_version.clone(),
                },
            }),
        )
        .context("Failed to serialize initialize params")?;

        transport.send_request(&request).await?;
        let response = transport.receive_response().await?;

        if let Some(error) = response.error {
            anyhow::bail!(
                "Initialize failed: {} (code: {})",
                error.message,
                error.code
            );
        }

        let result: InitializeResult = serde_json::from_value(
            response
                .result
                .context("Missing result in initialize response")?,
        )
        .context("Failed to parse initialize result")?;

        // Send initialized notification
        transport
            .send_request(&JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: serde_json::Value::Null,
                method: "notifications/initialized".to_string(),
                params: None,
            })
            .await?;

        Ok(result)
    }

    /// List available tools from a server
    pub async fn list_tools(&self, server_name: &str) -> Result<Vec<McpTool>> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        let request =
            JsonRpcRequest::new(self.next_request_id(), "tools/list".to_string(), None::<()>)
                .context("Failed to serialize tools/list params")?;

        connection.transport.send_request(&request).await?;
        let response = connection.transport.receive_response().await?;

        if let Some(error) = response.error {
            anyhow::bail!(
                "tools/list failed: {} (code: {})",
                error.message,
                error.code
            );
        }

        let result: ListToolsResult =
            serde_json::from_value(response.result.context("Missing result")?)?;

        Ok(result.tools)
    }

    /// Call a tool on a server
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Option<serde_json::Value>,
    ) -> Result<CallToolResult> {
        self.call_tool_with_notifications(server_name, tool_name, arguments, None)
            .await
    }

    /// Call a tool on a server with notification forwarding
    /// If notification_tx is provided, any notifications received while waiting for the response
    /// will be forwarded through that channel
    pub async fn call_tool_with_notifications(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Option<serde_json::Value>,
        notification_tx: Option<mpsc::UnboundedSender<JsonRpcNotification>>,
    ) -> Result<CallToolResult> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        // Convert Value to JsonObject (Map<String, Value>) as required by rmcp
        let arguments_obj = arguments.and_then(|v| {
            if let serde_json::Value::Object(map) = v {
                Some(map)
            } else {
                None
            }
        });

        let request_id = self.next_request_id();
        let request = JsonRpcRequest::new(
            request_id,
            "tools/call".to_string(),
            Some({
                let mut params = CallToolParams::new(tool_name.to_string());
                params.arguments = arguments_obj;
                params
            }),
        )
        .context("Failed to serialize tools/call params")?;

        connection.transport.send_request(&request).await?;

        // Wait for response, forwarding any notifications that arrive
        loop {
            let message = connection.transport.receive_message().await?;

            match message {
                JsonRpcMessage::Response(response) => {
                    // Check if this is the response we're waiting for
                    // (in a simple single-request-at-a-time model, it should be)
                    if let Some(error) = response.error {
                        anyhow::bail!(
                            "tools/call failed: {} (code: {})",
                            error.message,
                            error.code
                        );
                    }

                    let result: CallToolResult =
                        serde_json::from_value(response.result.context("Missing result")?)?;

                    return Ok(result);
                }
                JsonRpcMessage::Notification(notification) => {
                    // Forward notification to caller if they provided a channel
                    if let Some(ref tx) = notification_tx {
                        let _ = tx.send(notification);
                    }
                    // Continue waiting for the response
                }
            }
        }
    }

    /// List available resources from a server
    pub async fn list_resources(&self, server_name: &str) -> Result<Vec<McpResource>> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        let request = JsonRpcRequest::new(
            self.next_request_id(),
            "resources/list".to_string(),
            None::<()>,
        )
        .context("Failed to serialize resources/list params")?;

        connection.transport.send_request(&request).await?;
        let response = connection.transport.receive_response().await?;

        if let Some(error) = response.error {
            anyhow::bail!(
                "resources/list failed: {} (code: {})",
                error.message,
                error.code
            );
        }

        let result: ListResourcesResult =
            serde_json::from_value(response.result.context("Missing result")?)?;

        Ok(result.resources)
    }

    /// Read a resource from a server
    pub async fn read_resource(&self, server_name: &str, uri: &str) -> Result<ReadResourceResult> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        let request = JsonRpcRequest::new(
            self.next_request_id(),
            "resources/read".to_string(),
            Some(ReadResourceParams {
                uri: uri.to_string(),
            }),
        )
        .context("Failed to serialize resources/read params")?;

        connection.transport.send_request(&request).await?;
        let response = connection.transport.receive_response().await?;

        if let Some(error) = response.error {
            anyhow::bail!(
                "resources/read failed: {} (code: {})",
                error.message,
                error.code
            );
        }

        let result: ReadResourceResult =
            serde_json::from_value(response.result.context("Missing result")?)?;

        Ok(result)
    }

    /// List available prompts from a server
    pub async fn list_prompts(&self, server_name: &str) -> Result<Vec<McpPrompt>> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        let request = JsonRpcRequest::new(
            self.next_request_id(),
            "prompts/list".to_string(),
            None::<()>,
        )
        .context("Failed to serialize prompts/list params")?;

        connection.transport.send_request(&request).await?;
        let response = connection.transport.receive_response().await?;

        if let Some(error) = response.error {
            anyhow::bail!(
                "prompts/list failed: {} (code: {})",
                error.message,
                error.code
            );
        }

        let result: ListPromptsResult =
            serde_json::from_value(response.result.context("Missing result")?)?;

        Ok(result.prompts)
    }

    /// Get a prompt from a server
    pub async fn get_prompt(
        &self,
        server_name: &str,
        prompt_name: &str,
        arguments: Option<serde_json::Value>,
    ) -> Result<GetPromptResult> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        let request = JsonRpcRequest::new(
            self.next_request_id(),
            "prompts/get".to_string(),
            Some(GetPromptParams {
                name: prompt_name.to_string(),
                arguments,
            }),
        )
        .context("Failed to serialize prompts/get params")?;

        connection.transport.send_request(&request).await?;
        let response = connection.transport.receive_response().await?;

        if let Some(error) = response.error {
            anyhow::bail!(
                "prompts/get failed: {} (code: {})",
                error.message,
                error.code
            );
        }

        let result: GetPromptResult =
            serde_json::from_value(response.result.context("Missing result")?)?;

        Ok(result)
    }

    /// Get server info for a connection
    pub async fn get_server_info(&self, server_name: &str) -> Result<ServerInfo> {
        let connections = self.connections.read().await;
        let connection = connections
            .get(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        Ok(connection.server_info.clone())
    }

    /// Get server capabilities for a connection
    pub async fn get_capabilities(&self, server_name: &str) -> Result<ServerCapabilities> {
        let connections = self.connections.read().await;
        let connection = connections
            .get(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        Ok(connection.capabilities.clone())
    }

    /// Get next request ID
    fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a cancellation request to the MCP server
    /// This follows the JSON-RPC 2.0 cancellation protocol using `$/cancelRequest`
    pub async fn cancel_request(&self, server_name: &str, request_id: u64) -> Result<()> {
        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_name)
            .context(format!("Not connected to server: {}", server_name))?;

        // Send cancellation notification (no id since it's a notification)
        let cancel_notification = JsonRpcNotification::new(
            "$/cancelRequest",
            Some(serde_json::json!({ "id": request_id })),
        )
        .context("Failed to serialize cancel request params")?;

        // Convert to JsonRpcRequest for sending (with null id for notification)
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: serde_json::Value::Null,
            method: cancel_notification.method,
            params: cancel_notification.params,
        };

        connection.transport.send_request(&request).await?;

        Ok(())
    }
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new("brainwires", env!("CARGO_PKG_VERSION"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = McpClient::new("test", "0.1.0");
        assert_eq!(client.request_id.load(Ordering::SeqCst), 1);
        assert_eq!(client.client_name, "test");
        assert_eq!(client.client_version, "0.1.0");
    }

    #[test]
    fn test_request_id_increment() {
        let client = McpClient::new("test", "0.1.0");
        assert_eq!(client.next_request_id(), 1);
        assert_eq!(client.next_request_id(), 2);
        assert_eq!(client.next_request_id(), 3);
    }
}
