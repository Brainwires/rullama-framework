use serde_json::Value;
use std::collections::HashMap;

/// Information about a connected MCP client.
#[derive(Debug, Clone, Default)]
pub struct ClientInfo {
    /// Client name.
    pub name: String,
    /// Client version.
    pub version: String,
}

/// Context for an MCP request.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Connected client info, if available.
    pub client_info: Option<ClientInfo>,
    /// JSON-RPC request ID.
    pub request_id: Value,
    /// Whether the connection has been initialized.
    pub initialized: bool,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
}

impl RequestContext {
    /// Create a new request context with the given ID.
    pub fn new(request_id: Value) -> Self {
        Self {
            client_info: None,
            request_id,
            initialized: false,
            metadata: HashMap::new(),
        }
    }

    /// Set the client info.
    pub fn with_client_info(mut self, info: ClientInfo) -> Self {
        self.client_info = Some(info);
        self
    }

    /// Mark this context as initialized.
    pub fn set_initialized(&mut self) {
        self.initialized = true;
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: String, value: Value) {
        self.metadata.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_context_is_not_initialized() {
        let ctx = RequestContext::new(json!(1));
        assert!(!ctx.initialized);
        assert!(ctx.client_info.is_none());
        assert!(ctx.metadata.is_empty());
    }

    #[test]
    fn set_initialized_marks_context() {
        let mut ctx = RequestContext::new(json!(1));
        ctx.set_initialized();
        assert!(ctx.initialized);
    }

    #[test]
    fn metadata_get_set_roundtrip() {
        let mut ctx = RequestContext::new(json!(1));
        ctx.set_metadata("foo".to_string(), json!("bar"));
        assert_eq!(ctx.get_metadata("foo"), Some(&json!("bar")));
    }

    #[test]
    fn get_missing_metadata_returns_none() {
        let ctx = RequestContext::new(json!(1));
        assert!(ctx.get_metadata("missing").is_none());
    }

    #[test]
    fn with_client_info_sets_name_and_version() {
        let ctx = RequestContext::new(json!(1)).with_client_info(ClientInfo {
            name: "test-client".to_string(),
            version: "1.0".to_string(),
        });
        let info = ctx.client_info.unwrap();
        assert_eq!(info.name, "test-client");
        assert_eq!(info.version, "1.0");
    }

    #[test]
    fn request_id_preserved() {
        let ctx = RequestContext::new(json!("req-42"));
        assert_eq!(ctx.request_id, json!("req-42"));
    }

    #[test]
    fn metadata_overwrite_works() {
        let mut ctx = RequestContext::new(json!(1));
        ctx.set_metadata("key".to_string(), json!(1));
        ctx.set_metadata("key".to_string(), json!(2));
        assert_eq!(ctx.get_metadata("key"), Some(&json!(2)));
    }
}
