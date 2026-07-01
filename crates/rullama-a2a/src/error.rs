//! A2A error types and JSON-RPC error codes.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// JSON-RPC error codes (spec-defined)
// ---------------------------------------------------------------------------

/// Invalid JSON payload.
pub const JSON_PARSE_ERROR: i32 = -32700;
/// Request payload validation error.
pub const INVALID_REQUEST: i32 = -32600;
/// Method not found.
pub const METHOD_NOT_FOUND: i32 = -32601;
/// Invalid parameters.
pub const INVALID_PARAMS: i32 = -32602;
/// Internal error.
pub const INTERNAL_ERROR: i32 = -32603;
/// Task not found.
pub const TASK_NOT_FOUND: i32 = -32001;
/// Task cannot be canceled.
pub const TASK_NOT_CANCELABLE: i32 = -32002;
/// Push notification is not supported.
pub const PUSH_NOT_SUPPORTED: i32 = -32003;
/// This operation is not supported.
pub const UNSUPPORTED_OPERATION: i32 = -32004;
/// Incompatible content types.
pub const CONTENT_TYPE_NOT_SUPPORTED: i32 = -32005;
/// Invalid agent response.
pub const INVALID_AGENT_RESPONSE: i32 = -32006;
/// Authenticated Extended Card is not configured.
pub const EXTENDED_CARD_NOT_CONFIGURED: i32 = -32007;
/// Extension support is required but not available.
pub const EXTENSION_SUPPORT_REQUIRED: i32 = -32008;
/// Protocol version is not supported.
pub const VERSION_NOT_SUPPORTED: i32 = -32009;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// A2A protocol error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2aError {
    /// Numeric error code.
    pub code: i32,
    /// Human-readable error message.
    pub message: String,
    /// Optional additional data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl A2aError {
    /// Create a new error from code and message.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Attach extra data to the error.
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Task not found error.
    pub fn task_not_found(task_id: &str) -> Self {
        Self::new(TASK_NOT_FOUND, format!("Task not found: {task_id}"))
    }

    /// Task not cancelable error.
    pub fn task_not_cancelable(task_id: &str) -> Self {
        Self::new(
            TASK_NOT_CANCELABLE,
            format!("Task cannot be canceled: {task_id}"),
        )
    }

    /// Push notifications not supported error.
    pub fn push_not_supported() -> Self {
        Self::new(PUSH_NOT_SUPPORTED, "Push notifications are not supported")
    }

    /// Unsupported operation error.
    pub fn unsupported_operation(detail: &str) -> Self {
        Self::new(
            UNSUPPORTED_OPERATION,
            format!("Unsupported operation: {detail}"),
        )
    }

    /// Content type not supported error.
    pub fn content_type_not_supported(detail: &str) -> Self {
        Self::new(
            CONTENT_TYPE_NOT_SUPPORTED,
            format!("Content type not supported: {detail}"),
        )
    }

    /// Invalid request error.
    pub fn invalid_request(detail: impl Into<String>) -> Self {
        Self::new(INVALID_REQUEST, detail)
    }

    /// Internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(INTERNAL_ERROR, message)
    }

    /// Method not found error.
    pub fn method_not_found(method: &str) -> Self {
        Self::new(METHOD_NOT_FOUND, format!("Method not found: {method}"))
    }

    /// Invalid params error.
    pub fn invalid_params(detail: impl Into<String>) -> Self {
        Self::new(INVALID_PARAMS, detail)
    }

    /// Parse error.
    pub fn parse_error(detail: impl Into<String>) -> Self {
        Self::new(JSON_PARSE_ERROR, detail)
    }

    /// Extended card not configured.
    pub fn extended_card_not_configured() -> Self {
        Self::new(
            EXTENDED_CARD_NOT_CONFIGURED,
            "Authenticated Extended Card is not configured",
        )
    }

    /// Extension support is required but not available.
    pub fn extension_support_required() -> Self {
        Self::new(
            EXTENSION_SUPPORT_REQUIRED,
            "Extension support is required but not available",
        )
    }

    /// Protocol version is not supported.
    pub fn version_not_supported() -> Self {
        Self::new(VERSION_NOT_SUPPORTED, "Protocol version is not supported")
    }
}

impl std::fmt::Display for A2aError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A2A error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for A2aError {}

impl From<serde_json::Error> for A2aError {
    fn from(err: serde_json::Error) -> Self {
        Self::parse_error(err.to_string())
    }
}

impl From<anyhow::Error> for A2aError {
    fn from(err: anyhow::Error) -> Self {
        Self::internal(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sets_code_and_message() {
        let e = A2aError::new(TASK_NOT_FOUND, "missing");
        assert_eq!(e.code, TASK_NOT_FOUND);
        assert_eq!(e.message, "missing");
        assert!(e.data.is_none());
    }

    #[test]
    fn with_data_attaches_payload() {
        let e = A2aError::new(INTERNAL_ERROR, "boom").with_data(serde_json::json!({"detail": "x"}));
        assert!(e.data.is_some());
    }

    #[test]
    fn constructors_use_correct_codes() {
        assert_eq!(A2aError::task_not_found("t1").code, TASK_NOT_FOUND);
        assert_eq!(
            A2aError::task_not_cancelable("t1").code,
            TASK_NOT_CANCELABLE
        );
        assert_eq!(A2aError::push_not_supported().code, PUSH_NOT_SUPPORTED);
        assert_eq!(
            A2aError::unsupported_operation("x").code,
            UNSUPPORTED_OPERATION
        );
        assert_eq!(
            A2aError::content_type_not_supported("x").code,
            CONTENT_TYPE_NOT_SUPPORTED
        );
        assert_eq!(A2aError::invalid_request("x").code, INVALID_REQUEST);
        assert_eq!(A2aError::internal("x").code, INTERNAL_ERROR);
        assert_eq!(A2aError::method_not_found("m").code, METHOD_NOT_FOUND);
        assert_eq!(A2aError::invalid_params("x").code, INVALID_PARAMS);
        assert_eq!(A2aError::parse_error("x").code, JSON_PARSE_ERROR);
        assert_eq!(
            A2aError::extended_card_not_configured().code,
            EXTENDED_CARD_NOT_CONFIGURED
        );
        assert_eq!(
            A2aError::extension_support_required().code,
            EXTENSION_SUPPORT_REQUIRED
        );
        assert_eq!(
            A2aError::version_not_supported().code,
            VERSION_NOT_SUPPORTED
        );
    }

    #[test]
    fn task_not_found_includes_id_in_message() {
        let e = A2aError::task_not_found("abc-123");
        assert!(e.message.contains("abc-123"));
    }

    #[test]
    fn display_includes_code_and_message() {
        let e = A2aError::new(INTERNAL_ERROR, "oops");
        let s = e.to_string();
        assert!(s.contains(&INTERNAL_ERROR.to_string()));
        assert!(s.contains("oops"));
    }

    #[test]
    fn serde_roundtrip_without_data() {
        let original = A2aError::task_not_found("t42");
        let json = serde_json::to_string(&original).unwrap();
        let decoded: A2aError = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.code, original.code);
        assert_eq!(decoded.message, original.message);
        assert!(decoded.data.is_none());
    }

    #[test]
    fn serde_roundtrip_with_data() {
        let original = A2aError::internal("err").with_data(serde_json::json!({"k": 1}));
        let json = serde_json::to_string(&original).unwrap();
        let decoded: A2aError = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.code, original.code);
        assert!(decoded.data.is_some());
    }

    #[test]
    fn from_serde_json_error() {
        let bad: Result<serde_json::Value, _> = serde_json::from_str("{bad json}");
        let a2a_err: A2aError = bad.unwrap_err().into();
        assert_eq!(a2a_err.code, JSON_PARSE_ERROR);
    }

    #[test]
    fn from_anyhow_error() {
        let anyhow_err = anyhow::anyhow!("something went wrong");
        let a2a_err: A2aError = anyhow_err.into();
        assert_eq!(a2a_err.code, INTERNAL_ERROR);
        assert!(a2a_err.message.contains("something went wrong"));
    }

    #[test]
    fn data_field_omitted_when_none_in_json() {
        let e = A2aError::new(TASK_NOT_FOUND, "x");
        let json = serde_json::to_string(&e).unwrap();
        assert!(!json.contains("data"));
    }
}
