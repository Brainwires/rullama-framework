use anyhow::Result;
use brainwires_mcp_client::{InitializeParams, InitializeResult, JsonRpcRequest, JsonRpcResponse};
use serde_json::{Value, json};
use tracing;

use crate::connection::{ClientInfo, RequestContext};
use crate::error::AgentNetworkError;
use crate::handler::McpHandler;
use crate::mcp_transport::{ServerTransport, StdioServerTransport};
use crate::middleware::{Middleware, MiddlewareChain};

/// MCP server that processes JSON-RPC requests via a transport.
pub struct McpServer<H: McpHandler> {
    handler: H,
    middleware: MiddlewareChain,
    transport: Box<dyn ServerTransport>,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl<H: McpHandler> McpServer<H> {
    /// Create a new server with the given handler and stdio transport.
    pub fn new(handler: H) -> Self {
        Self {
            handler,
            middleware: MiddlewareChain::new(),
            transport: Box::new(StdioServerTransport::new()),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Set a custom transport.
    pub fn with_transport(mut self, transport: impl ServerTransport + 'static) -> Self {
        self.transport = Box::new(transport);
        self
    }

    /// Add a middleware to the processing pipeline.
    pub fn with_middleware(mut self, mw: impl Middleware) -> Self {
        self.middleware.add(mw);
        self
    }

    /// Attach an analytics collector to record McpRequest events.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<brainwires_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }

    /// Run the server event loop until the transport closes.
    pub async fn run(mut self) -> Result<()> {
        let mut ctx = RequestContext::new(json!(null));
        tracing::info!("MCP Relay server starting");

        loop {
            let line = match self.transport.read_request().await {
                Ok(Some(line)) => line,
                Ok(None) => {
                    tracing::debug!("Transport closed (EOF)");
                    break;
                }
                Err(e) => {
                    tracing::error!("Transport read error: {}", e);
                    break;
                }
            };

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let error = AgentNetworkError::ParseError(e.to_string());
                    let response = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: json!(null),
                        result: None,
                        error: Some(error.to_json_rpc_error()),
                    };
                    self.write_response(&response).await?;
                    continue;
                }
            };

            ctx.request_id = request.id.clone();

            // Run middleware chain
            if let Err(err) = self.middleware.process_request(&request, &mut ctx).await {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(err),
                };
                self.write_response(&response).await?;
                continue;
            }

            // Dispatch to handler
            let response = self.handle_request(&request, &mut ctx).await;

            // Run response middleware
            let mut response = response;
            self.middleware.process_response(&mut response, &ctx).await;

            self.write_response(&response).await?;
        }

        self.handler.on_shutdown().await?;
        tracing::info!("MCP Relay server shut down");
        Ok(())
    }

    async fn handle_request(
        &self,
        request: &JsonRpcRequest,
        ctx: &mut RequestContext,
    ) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request, ctx).await,
            "notifications/initialized" => {
                // Client confirming initialization - no response needed but we return success
                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: Some(json!({})),
                    error: None,
                }
            }
            "tools/list" => self.handle_list_tools(request).await,
            "tools/call" => self.handle_call_tool(request, ctx).await,
            _ => {
                let error = AgentNetworkError::MethodNotFound(request.method.clone());
                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(error.to_json_rpc_error()),
                }
            }
        }
    }

    async fn handle_initialize(
        &self,
        request: &JsonRpcRequest,
        ctx: &mut RequestContext,
    ) -> JsonRpcResponse {
        let params: InitializeParams = match request
            .params
            .as_ref()
            .and_then(|p| serde_json::from_value(p.clone()).ok())
        {
            Some(p) => p,
            None => {
                // Allow initialize without params for compatibility
                InitializeParams {
                    protocol_version: "2024-11-05".to_string(),
                    capabilities: Default::default(),
                    client_info: brainwires_mcp_client::ClientInfo {
                        name: "unknown".to_string(),
                        version: "0.7.0".to_string(),
                    },
                }
            }
        };

        ctx.client_info = Some(ClientInfo {
            name: params.client_info.name.clone(),
            version: params.client_info.version.clone(),
        });
        ctx.set_initialized();

        if let Err(e) = self.handler.on_initialize(&params).await {
            tracing::error!("Handler on_initialize failed: {}", e);
        }

        let info = self.handler.server_info();
        let capabilities = self.handler.capabilities();

        let result = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities,
            server_info: info,
        };

        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: serde_json::to_value(result).ok(),
            error: None,
        }
    }

    async fn handle_list_tools(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tool_defs = self.handler.list_tools();

        let tools: Vec<Value> = tool_defs
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();

        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({ "tools": tools })),
            error: None,
        }
    }

    async fn handle_call_tool(
        &self,
        request: &JsonRpcRequest,
        ctx: &RequestContext,
    ) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => p,
            None => {
                let error =
                    AgentNetworkError::InvalidParams("Missing params for tools/call".to_string());
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(error.to_json_rpc_error()),
                };
            }
        };

        let tool_name = match params.get("name").and_then(|n| n.as_str()) {
            Some(name) => name,
            None => {
                let error =
                    AgentNetworkError::InvalidParams("Missing 'name' in tools/call".to_string());
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(error.to_json_rpc_error()),
                };
            }
        };

        let args = params.get("arguments").cloned().unwrap_or(json!({}));

        #[cfg(feature = "telemetry")]
        let _started = std::time::Instant::now();

        let (response, _success) = match self.handler.call_tool(tool_name, args, ctx).await {
            Ok(result) => {
                let result_value = serde_json::to_value(result).unwrap_or(json!({}));
                (
                    JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id.clone(),
                        result: Some(result_value),
                        error: None,
                    },
                    true,
                )
            }
            Err(e) => {
                let error = AgentNetworkError::Internal(e);
                (
                    JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id.clone(),
                        result: None,
                        error: Some(error.to_json_rpc_error()),
                    },
                    false,
                )
            }
        };

        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use brainwires_telemetry::AnalyticsEvent;
            collector.record(AnalyticsEvent::McpRequest {
                session_id: None,
                server_name: self.handler.server_info().name.clone(),
                tool_name: tool_name.to_string(),
                success: _success,
                duration_ms: _started.elapsed().as_millis() as u64,
                timestamp: chrono::Utc::now(),
            });
        }

        response
    }

    async fn write_response(&mut self, response: &JsonRpcResponse) -> Result<()> {
        let json = serde_json::to_string(response)?;
        self.transport.write_response(&json).await
    }
}
