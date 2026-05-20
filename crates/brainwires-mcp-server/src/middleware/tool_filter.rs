use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcError, JsonRpcRequest};
use std::collections::HashSet;

use super::{Middleware, MiddlewareResult};
use crate::connection::RequestContext;

/// Tool filtering mode.
pub enum FilterMode {
    /// Only allow listed tools.
    AllowList(HashSet<String>),
    /// Deny listed tools, allow all others.
    DenyList(HashSet<String>),
}

/// Middleware that filters tool calls by allow/deny lists.
pub struct ToolFilterMiddleware {
    mode: FilterMode,
}

impl ToolFilterMiddleware {
    /// Create a filter that only allows the specified tools.
    pub fn allow_only(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            mode: FilterMode::AllowList(tools.into_iter().map(|t| t.into()).collect()),
        }
    }

    /// Create a filter that denies the specified tools.
    pub fn deny(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            mode: FilterMode::DenyList(tools.into_iter().map(|t| t.into()).collect()),
        }
    }

    fn is_allowed(&self, tool_name: &str) -> bool {
        match &self.mode {
            FilterMode::AllowList(allowed) => allowed.contains(tool_name),
            FilterMode::DenyList(denied) => !denied.contains(tool_name),
        }
    }
}

#[async_trait]
impl Middleware for ToolFilterMiddleware {
    async fn process_request(
        &self,
        request: &JsonRpcRequest,
        _ctx: &mut RequestContext,
    ) -> MiddlewareResult {
        // Only filter tools/call
        if request.method != "tools/call" {
            return MiddlewareResult::Continue;
        }

        let tool_name = request
            .params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown");

        if self.is_allowed(tool_name) {
            MiddlewareResult::Continue
        } else {
            MiddlewareResult::Reject(JsonRpcError {
                code: -32001,
                message: format!("Tool '{tool_name}' is not allowed by filter policy"),
                data: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allow_list() {
        let filter = ToolFilterMiddleware::allow_only(["agent_spawn", "agent_list"]);
        assert!(filter.is_allowed("agent_spawn"));
        assert!(filter.is_allowed("agent_list"));
        assert!(!filter.is_allowed("bash"));
    }

    #[test]
    fn test_deny_list() {
        let filter = ToolFilterMiddleware::deny(["bash", "write_file"]);
        assert!(!filter.is_allowed("bash"));
        assert!(!filter.is_allowed("write_file"));
        assert!(filter.is_allowed("agent_spawn"));
    }
}
