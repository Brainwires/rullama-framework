//! Tier-A feature cases for MCP protocol shapes and the Tool system.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{Tool, ToolInputSchema};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_mcp_client::JsonRpcRequest;
use serde_json::json;

use crate::registry::TierACase;

// ── mcp.jsonrpc_request_shape ──────────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::mcp_and_tools::jsonrpc_request_shape",
        crate_name: "rullama-mcp-client",
        description: "JsonRpcRequest::new always sets jsonrpc=\"2.0\" and propagates method + id + params",
        factory: || Box::new(JsonRpcRequestShapeCase),
    }
}

struct JsonRpcRequestShapeCase;

#[async_trait]
impl EvaluationCase for JsonRpcRequestShapeCase {
    fn name(&self) -> &str {
        "feature.mcp.jsonrpc_request_shape"
    }
    fn category(&self) -> &str {
        "feature.mcp"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let req = JsonRpcRequest::new(
            json!(42),
            "tools/list".to_string(),
            Some(json!({"cursor": "abc"})),
        )?;
        if req.jsonrpc != "2.0" {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("jsonrpc field = {:?}, expected \"2.0\"", req.jsonrpc),
            ));
        }
        if req.method != "tools/list" {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("method = {:?}, expected \"tools/list\"", req.method),
            ));
        }
        if req.id != json!(42) {
            return Ok(TrialResult::failure(0, 0, "id field not propagated"));
        }
        // params=None must round-trip as None
        let req2 = JsonRpcRequest::new(json!(7), "ping".to_string(), None::<()>)?;
        if req2.params.is_some() {
            return Ok(TrialResult::failure(
                0,
                0,
                "None params became Some after construction",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── tools.input_schema_object ──────────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::mcp_and_tools::tool_input_schema_object",
        crate_name: "rullama-core",
        description: "ToolInputSchema::object builds a schema with the given properties and required list",
        factory: || Box::new(ToolInputSchemaObjectCase),
    }
}

struct ToolInputSchemaObjectCase;

#[async_trait]
impl EvaluationCase for ToolInputSchemaObjectCase {
    fn name(&self) -> &str {
        "feature.tools.input_schema_object"
    }
    fn category(&self) -> &str {
        "feature.tools"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mut props = HashMap::new();
        props.insert("command".to_string(), json!({"type": "string"}));
        let schema = ToolInputSchema::object(props.clone(), vec!["command".to_string()]);
        // The schema must serialise to JSON without panicking.
        let s = serde_json::to_string(&schema)?;
        if !s.contains("command") {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("serialised schema missing 'command' property: {s}"),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── tools.tool_struct_default ──────────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::mcp_and_tools::tool_struct_default",
        crate_name: "rullama-core",
        description: "Tool::default() yields an empty-named tool with no required approvals",
        factory: || Box::new(ToolStructDefaultCase),
    }
}

struct ToolStructDefaultCase;

#[async_trait]
impl EvaluationCase for ToolStructDefaultCase {
    fn name(&self) -> &str {
        "feature.tools.tool_struct_default"
    }
    fn category(&self) -> &str {
        "feature.tools"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let t = Tool::default();
        if t.requires_approval {
            return Ok(TrialResult::failure(
                0,
                0,
                "Tool::default().requires_approval is true — must default to false",
            ));
        }
        if !t.allowed_callers.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                "Tool::default().allowed_callers must be empty",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
