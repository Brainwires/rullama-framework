use brainwires_mcp_client::JsonRpcError;

/// Errors that can occur in the agent network layer.
#[derive(Debug, thiserror::Error)]
pub enum AgentNetworkError {
    /// JSON-RPC parse error.
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Requested method does not exist.
    #[error("Method not found: {0}")]
    MethodNotFound(String),
    /// Invalid parameters supplied.
    #[error("Invalid params: {0}")]
    InvalidParams(String),
    /// Internal server error.
    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
    /// Transport-level error.
    #[error("Transport error: {0}")]
    Transport(String),
    /// Requested tool does not exist.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    /// Request was rate-limited.
    #[error("Rate limited")]
    RateLimited,
    /// Request was not authorized.
    #[error("Unauthorized")]
    Unauthorized,
}

impl AgentNetworkError {
    /// Convert to a JSON-RPC error with the appropriate code.
    pub fn to_json_rpc_error(&self) -> JsonRpcError {
        match self {
            AgentNetworkError::ParseError(msg) => JsonRpcError {
                code: -32700,
                message: msg.clone(),
                data: None,
            },
            AgentNetworkError::MethodNotFound(method) => JsonRpcError {
                code: -32601,
                message: format!("Method not found: {method}"),
                data: None,
            },
            AgentNetworkError::InvalidParams(msg) => JsonRpcError {
                code: -32602,
                message: msg.clone(),
                data: None,
            },
            AgentNetworkError::Internal(err) => JsonRpcError {
                code: -32603,
                message: err.to_string(),
                data: None,
            },
            AgentNetworkError::Transport(msg) => JsonRpcError {
                code: -32000,
                message: format!("Transport error: {msg}"),
                data: None,
            },
            AgentNetworkError::ToolNotFound(name) => JsonRpcError {
                code: -32001,
                message: format!("Tool not found: {name}"),
                data: None,
            },
            AgentNetworkError::RateLimited => JsonRpcError {
                code: -32002,
                message: "Rate limited".to_string(),
                data: None,
            },
            AgentNetworkError::Unauthorized => JsonRpcError {
                code: -32003,
                message: "Unauthorized".to_string(),
                data: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_code_is_minus_32700() {
        let err = AgentNetworkError::ParseError("bad JSON".to_string());
        assert_eq!(err.to_json_rpc_error().code, -32700);
    }

    #[test]
    fn method_not_found_code_is_minus_32601() {
        let err = AgentNetworkError::MethodNotFound("unknownMethod".to_string());
        let rpc = err.to_json_rpc_error();
        assert_eq!(rpc.code, -32601);
        assert!(rpc.message.contains("unknownMethod"));
    }

    #[test]
    fn invalid_params_code_is_minus_32602() {
        let err = AgentNetworkError::InvalidParams("missing field".to_string());
        assert_eq!(err.to_json_rpc_error().code, -32602);
    }

    #[test]
    fn internal_error_code_is_minus_32603() {
        let err = AgentNetworkError::Internal(anyhow::anyhow!("database down"));
        let rpc = err.to_json_rpc_error();
        assert_eq!(rpc.code, -32603);
        assert!(rpc.message.contains("database down"));
    }

    #[test]
    fn transport_error_code_is_minus_32000() {
        let err = AgentNetworkError::Transport("connection reset".to_string());
        let rpc = err.to_json_rpc_error();
        assert_eq!(rpc.code, -32000);
        assert!(rpc.message.contains("connection reset"));
    }

    #[test]
    fn tool_not_found_code_is_minus_32001() {
        let err = AgentNetworkError::ToolNotFound("my_tool".to_string());
        let rpc = err.to_json_rpc_error();
        assert_eq!(rpc.code, -32001);
        assert!(rpc.message.contains("my_tool"));
    }

    #[test]
    fn rate_limited_code_is_minus_32002() {
        let err = AgentNetworkError::RateLimited;
        assert_eq!(err.to_json_rpc_error().code, -32002);
    }

    #[test]
    fn unauthorized_code_is_minus_32003() {
        let err = AgentNetworkError::Unauthorized;
        assert_eq!(err.to_json_rpc_error().code, -32003);
    }

    #[test]
    fn display_messages_are_non_empty() {
        let errors: Vec<AgentNetworkError> = vec![
            AgentNetworkError::ParseError("x".to_string()),
            AgentNetworkError::MethodNotFound("m".to_string()),
            AgentNetworkError::InvalidParams("p".to_string()),
            AgentNetworkError::Transport("t".to_string()),
            AgentNetworkError::ToolNotFound("tool".to_string()),
            AgentNetworkError::RateLimited,
            AgentNetworkError::Unauthorized,
        ];
        for e in errors {
            assert!(!e.to_string().is_empty());
        }
    }
}
