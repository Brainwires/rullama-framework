//! JSON-RPC 2.0 envelope types and method constants.

use serde::{Deserialize, Serialize};

use crate::error::A2aError;

// ---------------------------------------------------------------------------
// Method constants
// ---------------------------------------------------------------------------

/// Send a message to an agent.
pub const METHOD_MESSAGE_SEND: &str = "SendMessage";
/// Stream a message to an agent.
pub const METHOD_MESSAGE_STREAM: &str = "SendStreamingMessage";
/// Get a task by ID.
pub const METHOD_TASKS_GET: &str = "GetTask";
/// Cancel a task.
pub const METHOD_TASKS_CANCEL: &str = "CancelTask";
/// Resubscribe to task updates.
pub const METHOD_TASKS_RESUBSCRIBE: &str = "SubscribeToTask";
/// List tasks.
pub const METHOD_TASKS_LIST: &str = "ListTasks";
/// Set push notification configuration.
pub const METHOD_PUSH_CONFIG_SET: &str = "CreateTaskPushNotificationConfig";
/// Get push notification configuration.
pub const METHOD_PUSH_CONFIG_GET: &str = "GetTaskPushNotificationConfig";
/// List push notification configurations.
pub const METHOD_PUSH_CONFIG_LIST: &str = "ListTaskPushNotificationConfigs";
/// Delete push notification configuration.
pub const METHOD_PUSH_CONFIG_DELETE: &str = "DeleteTaskPushNotificationConfig";
/// Get authenticated extended agent card.
pub const METHOD_EXTENDED_CARD: &str = "GetExtendedAgentCard";

// ---------------------------------------------------------------------------
// Request ID
// ---------------------------------------------------------------------------

/// JSON-RPC request identifier (string or number).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String identifier.
    String(String),
    /// Numeric identifier.
    Number(i64),
}

// ---------------------------------------------------------------------------
// JSON-RPC envelope types
// ---------------------------------------------------------------------------

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version (always "2.0").
    pub jsonrpc: String,
    /// Method name.
    pub method: String,
    /// Request parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Request identifier.
    pub id: RequestId,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// Protocol version (always "2.0").
    pub jsonrpc: String,
    /// Result on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<A2aError>,
    /// Request identifier echoed back.
    pub id: RequestId,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: RequestId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    /// Create an error response.
    pub fn error(id: RequestId, error: A2aError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(error),
            id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{A2aError, TASK_NOT_FOUND};

    // --- RequestId ---

    #[test]
    fn request_id_string_roundtrip() {
        let id = RequestId::String("req-1".to_string());
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#""req-1""#);
        let back: RequestId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn request_id_number_roundtrip() {
        let id = RequestId::Number(42);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "42");
        let back: RequestId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    // --- Method constants ---

    #[test]
    fn method_constants_are_non_empty() {
        assert!(!METHOD_MESSAGE_SEND.is_empty());
        assert!(!METHOD_MESSAGE_STREAM.is_empty());
        assert!(!METHOD_TASKS_GET.is_empty());
        assert!(!METHOD_TASKS_CANCEL.is_empty());
        assert!(!METHOD_TASKS_RESUBSCRIBE.is_empty());
        assert!(!METHOD_TASKS_LIST.is_empty());
        assert!(!METHOD_PUSH_CONFIG_SET.is_empty());
        assert!(!METHOD_PUSH_CONFIG_GET.is_empty());
        assert!(!METHOD_PUSH_CONFIG_LIST.is_empty());
        assert!(!METHOD_PUSH_CONFIG_DELETE.is_empty());
        assert!(!METHOD_EXTENDED_CARD.is_empty());
    }

    // --- JsonRpcRequest ---

    #[test]
    fn jsonrpc_request_serializes_correctly() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: METHOD_MESSAGE_SEND.to_string(),
            params: Some(serde_json::json!({"key": "value"})),
            id: RequestId::Number(1),
        };
        let json = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["method"], METHOD_MESSAGE_SEND);
        assert_eq!(v["id"], 1);
        assert_eq!(v["params"]["key"], "value");
    }

    #[test]
    fn jsonrpc_request_omits_params_when_none() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: METHOD_TASKS_GET.to_string(),
            params: None,
            id: RequestId::String("x".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("params"));
    }

    #[test]
    fn jsonrpc_request_roundtrip() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "TestMethod".to_string(),
            params: Some(serde_json::json!(42)),
            id: RequestId::Number(99),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.method, "TestMethod");
        assert_eq!(back.id, RequestId::Number(99));
    }

    // --- JsonRpcResponse ---

    #[test]
    fn success_response_has_result_no_error() {
        let id = RequestId::Number(1);
        let resp = JsonRpcResponse::success(id.clone(), serde_json::json!("ok"));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert_eq!(resp.id, id);
        assert_eq!(resp.jsonrpc, "2.0");
    }

    #[test]
    fn error_response_has_error_no_result() {
        let id = RequestId::String("e1".to_string());
        let err = A2aError::new(TASK_NOT_FOUND, "not found");
        let resp = JsonRpcResponse::error(id.clone(), err);
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        assert_eq!(resp.id, id);
    }

    #[test]
    fn success_response_roundtrip() {
        let id = RequestId::Number(7);
        let resp = JsonRpcResponse::success(id, serde_json::json!({"status": "done"}));
        let json = serde_json::to_string(&resp).unwrap();
        let back: JsonRpcResponse = serde_json::from_str(&json).unwrap();
        assert!(back.result.is_some());
        assert!(back.error.is_none());
    }

    #[test]
    fn error_response_roundtrip() {
        let id = RequestId::Number(8);
        let err = A2aError::internal("crash");
        let resp = JsonRpcResponse::error(id, err);
        let json = serde_json::to_string(&resp).unwrap();
        let back: JsonRpcResponse = serde_json::from_str(&json).unwrap();
        assert!(back.error.is_some());
        assert!(back.result.is_none());
    }

    #[test]
    fn id_echoed_in_response() {
        let id = RequestId::String("correlation-abc".to_string());
        let resp = JsonRpcResponse::success(id.clone(), serde_json::json!(null));
        assert_eq!(resp.id, id);
    }
}
