use brainwires_mcp_client::{JsonRpcRequest, JsonRpcResponse};
use serde_json::{Value, json};

use super::error::AgentNetworkClientError;

/// Build a JSON-RPC initialize request with standard client info.
pub fn build_initialize_request(id: u64) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(id),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "brainwires-relay-client",
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
    }
}

/// Build the initialized notification string to send after handshake.
pub fn build_initialized_notification() -> String {
    serde_json::to_string(&json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }))
    .expect("Failed to serialize initialized notification")
}

/// Build a JSON-RPC request to list available tools.
pub fn build_tools_list_request(id: u64) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(id),
        method: "tools/list".to_string(),
        params: None,
    }
}

/// Build a JSON-RPC request to call a tool by name with arguments.
pub fn build_tools_call_request(id: u64, name: &str, args: Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(id),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": name,
            "arguments": args
        })),
    }
}

/// Parse a JSON-RPC response from a raw line of text.
pub fn parse_response(line: &str) -> Result<JsonRpcResponse, AgentNetworkClientError> {
    serde_json::from_str(line).map_err(|e| {
        AgentNetworkClientError::Protocol(format!("Failed to parse response: {e}: {line}"))
    })
}

/// Extract the result value from a JSON-RPC response, returning errors as needed.
pub fn extract_result(response: JsonRpcResponse) -> Result<Value, AgentNetworkClientError> {
    if let Some(error) = response.error {
        return Err(AgentNetworkClientError::JsonRpc {
            code: error.code,
            message: error.message,
        });
    }
    Ok(response.result.unwrap_or(json!(null)))
}
