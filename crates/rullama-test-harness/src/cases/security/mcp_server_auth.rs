//! Tier-B adversarial cases for `rullama_mcp_server::middleware::AuthMiddleware`.
//!
//! Invariants:
//! - `initialize` MUST skip auth — required by the MCP protocol, since
//!   the client hasn't presented credentials yet. A regression that
//!   tightens this to require a token before initialize would break
//!   every MCP client.
//! - Any non-initialize method WITHOUT a matching token MUST be rejected
//!   with JSON-RPC error code -32003.
//! - A valid token in `_auth_token` request params MUST be accepted AND
//!   cached in the request metadata for subsequent requests.
//! - An invalid token MUST be rejected, even if it differs only in
//!   case (case-sensitive comparison).

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_mcp_client::JsonRpcRequest;
use rullama_mcp_server::connection::RequestContext;
use rullama_mcp_server::middleware::auth::AuthMiddleware;
use rullama_mcp_server::middleware::{Middleware, MiddlewareResult};
use serde_json::json;

use crate::registry::SecurityCase;

fn make_request(method: &str, params: Option<serde_json::Value>) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        method: method.to_string(),
        params,
    }
}

// ── sec.mcp_server.auth_skips_initialize ───────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.mcp_server.auth_skips_initialize",
        crate_name: "rullama-mcp-server",
        invariant: "AuthMiddleware MUST allow `initialize` through without a token (MCP protocol requirement)",
        factory: || Box::new(AuthSkipsInitializeCase),
    }
}

struct AuthSkipsInitializeCase;

#[async_trait]
impl EvaluationCase for AuthSkipsInitializeCase {
    fn name(&self) -> &str {
        "sec.mcp_server.auth_skips_initialize"
    }
    fn category(&self) -> &str {
        "security.mcp_server"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mw = AuthMiddleware::new("the-server-token");
        let req = make_request("initialize", None);
        let mut ctx = RequestContext::new(json!(1));
        match mw.process_request(&req, &mut ctx).await {
            MiddlewareResult::Continue => Ok(TrialResult::success(0, 0)),
            MiddlewareResult::Reject(err) => Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "AuthMiddleware rejected `initialize` ({}): {}",
                    err.code, err.message
                ),
            )),
        }
    }
}

// ── sec.mcp_server.auth_rejects_missing_token ──────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.mcp_server.auth_rejects_missing_token",
        crate_name: "rullama-mcp-server",
        invariant: "Any non-initialize call without an auth token MUST be rejected with JSON-RPC -32003",
        factory: || Box::new(AuthRejectsMissingTokenCase),
    }
}

struct AuthRejectsMissingTokenCase;

#[async_trait]
impl EvaluationCase for AuthRejectsMissingTokenCase {
    fn name(&self) -> &str {
        "sec.mcp_server.auth_rejects_missing_token"
    }
    fn category(&self) -> &str {
        "security.mcp_server"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mw = AuthMiddleware::new("secret");
        let req = make_request("tools/list", None);
        let mut ctx = RequestContext::new(json!(1));
        match mw.process_request(&req, &mut ctx).await {
            MiddlewareResult::Reject(err) if err.code == -32003 => {
                Ok(TrialResult::success(0, 0))
            }
            MiddlewareResult::Reject(err) => Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "expected JSON-RPC error code -32003, got {} ({})",
                    err.code, err.message
                ),
            )),
            MiddlewareResult::Continue => Ok(TrialResult::failure(
                0,
                0,
                "AuthMiddleware allowed a tokenless tools/list — auth bypass",
            )),
        }
    }
}

// ── sec.mcp_server.auth_rejects_wrong_token ────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.mcp_server.auth_rejects_wrong_token",
        crate_name: "rullama-mcp-server",
        invariant: "Wrong token in params rejected; case-sensitive comparison",
        factory: || Box::new(AuthRejectsWrongTokenCase),
    }
}

struct AuthRejectsWrongTokenCase;

#[async_trait]
impl EvaluationCase for AuthRejectsWrongTokenCase {
    fn name(&self) -> &str {
        "sec.mcp_server.auth_rejects_wrong_token"
    }
    fn category(&self) -> &str {
        "security.mcp_server"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mw = AuthMiddleware::new("correct-Token-42");
        // Wrong case — auth must be case-sensitive.
        let req = make_request(
            "tools/list",
            Some(json!({"_auth_token": "correct-token-42"})),
        );
        let mut ctx = RequestContext::new(json!(1));
        match mw.process_request(&req, &mut ctx).await {
            MiddlewareResult::Reject(_) => Ok(TrialResult::success(0, 0)),
            MiddlewareResult::Continue => Ok(TrialResult::failure(
                0,
                0,
                "AuthMiddleware accepted a case-mismatched token — comparison must be case-sensitive",
            )),
        }
    }
}

// ── sec.mcp_server.auth_caches_token_in_metadata ───────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.mcp_server.auth_caches_token_in_metadata",
        crate_name: "rullama-mcp-server",
        invariant: "A valid `_auth_token` is cached in request metadata for subsequent requests",
        factory: || Box::new(AuthCachesTokenCase),
    }
}

struct AuthCachesTokenCase;

#[async_trait]
impl EvaluationCase for AuthCachesTokenCase {
    fn name(&self) -> &str {
        "sec.mcp_server.auth_caches_token_in_metadata"
    }
    fn category(&self) -> &str {
        "security.mcp_server"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mw = AuthMiddleware::new("the-token");
        let req = make_request("tools/list", Some(json!({"_auth_token": "the-token"})));
        let mut ctx = RequestContext::new(json!(1));
        let first = mw.process_request(&req, &mut ctx).await;
        match first {
            MiddlewareResult::Continue => {}
            MiddlewareResult::Reject(err) => {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "AuthMiddleware rejected the canonical token ({}): {}",
                        err.code, err.message
                    ),
                ));
            }
        }
        // Metadata should now carry auth_token; next request without it must
        // still pass via the cached value.
        let req2 = make_request("tools/list", None);
        match mw.process_request(&req2, &mut ctx).await {
            MiddlewareResult::Continue => Ok(TrialResult::success(0, 0)),
            MiddlewareResult::Reject(err) => Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "second request rejected despite cached metadata token ({}): {}",
                    err.code, err.message
                ),
            )),
        }
    }
}
