//! HTTP/REST route handling — dispatches REST endpoints to handler methods.

use std::sync::Arc;

use crate::error::A2aError;
use crate::params::*;
use crate::push_notification::TaskPushNotificationConfig;
use crate::server::handler::A2aHandler;
use crate::streaming::{SendMessageResponse, StreamResponse};
use crate::task::Task;

/// Result of routing a REST request.
pub enum RestResult {
    /// Non-streaming JSON response.
    Json(serde_json::Value),
    /// Streaming SSE response — returns a stream of events.
    Stream(std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamResponse, A2aError>> + Send>>),
}

/// Dispatch a REST request based on method and path.
///
/// `method` is the HTTP method (GET, POST, DELETE).
/// `path` is the request path (e.g. `/message:send`, `/tasks/abc123`).
/// `body` is the request body (for POST/PUT).
pub async fn dispatch_rest<H: A2aHandler>(
    handler: &Arc<H>,
    method: &str,
    path: &str,
    body: &[u8],
) -> Result<RestResult, A2aError> {
    // Strip optional tenant prefix: /{tenant}/path → path
    let (_, path) = strip_tenant(path);

    match (method, path) {
        ("POST", "/message:send") => {
            let req: SendMessageRequest = serde_json::from_slice(body)?;
            let result: SendMessageResponse = handler.on_send_message(req).await?;
            Ok(RestResult::Json(serde_json::to_value(result)?))
        }
        ("POST", "/message:stream") => {
            let req: SendMessageRequest = serde_json::from_slice(body)?;
            let stream = handler.on_send_streaming_message(req).await?;
            Ok(RestResult::Stream(stream))
        }
        ("POST", p) if p.ends_with(":cancel") => {
            // POST /tasks/{id}:cancel
            let id = p
                .strip_prefix("/tasks/")
                .and_then(|s| s.strip_suffix(":cancel"))
                .ok_or_else(|| A2aError::method_not_found(p))?;
            let mut req: CancelTaskRequest = if body.is_empty() {
                CancelTaskRequest {
                    tenant: None,
                    id: String::new(),
                    metadata: None,
                }
            } else {
                serde_json::from_slice(body)?
            };
            req.id = id.to_string();
            let task: Task = handler.on_cancel_task(req).await?;
            Ok(RestResult::Json(serde_json::to_value(task)?))
        }
        ("POST", p) if p.starts_with("/tasks/") && p.contains("/pushNotificationConfigs") => {
            // POST /tasks/{task_id}/pushNotificationConfigs
            let config: TaskPushNotificationConfig = serde_json::from_slice(body)?;
            let result = handler.on_create_push_config(config).await?;
            Ok(RestResult::Json(serde_json::to_value(result)?))
        }
        ("POST", p) if p.ends_with(":subscribe") => {
            // POST /tasks/{id}:subscribe
            let id = p
                .strip_prefix("/tasks/")
                .and_then(|s| s.strip_suffix(":subscribe"))
                .ok_or_else(|| A2aError::method_not_found(p))?;
            let req = SubscribeToTaskRequest {
                tenant: None,
                id: id.to_string(),
            };
            let stream = handler.on_subscribe_to_task(req).await?;
            Ok(RestResult::Stream(stream))
        }
        ("GET", "/extendedAgentCard") => {
            let req = GetExtendedAgentCardRequest::default();
            let card = handler.on_get_extended_agent_card(req).await?;
            Ok(RestResult::Json(serde_json::to_value(card)?))
        }
        ("GET", "/tasks") => {
            let req = ListTasksRequest::default();
            let result = handler.on_list_tasks(req).await?;
            Ok(RestResult::Json(serde_json::to_value(result)?))
        }
        ("GET", p) if p.starts_with("/tasks/") => {
            let remaining = &p["/tasks/".len()..];
            if remaining.contains("/pushNotificationConfigs/") {
                // GET /tasks/{task_id}/pushNotificationConfigs/{id}
                let parts: Vec<&str> = remaining.split("/pushNotificationConfigs/").collect();
                if parts.len() == 2 {
                    let req = GetTaskPushNotificationConfigRequest {
                        tenant: None,
                        task_id: parts[0].to_string(),
                        config_id: parts[1].to_string(),
                    };
                    let result = handler.on_get_push_config(req).await?;
                    return Ok(RestResult::Json(serde_json::to_value(result)?));
                }
            }
            if remaining.contains("/pushNotificationConfigs") {
                // GET /tasks/{task_id}/pushNotificationConfigs
                let task_id = remaining
                    .strip_suffix("/pushNotificationConfigs")
                    .ok_or_else(|| A2aError::method_not_found(p))?;
                let req = ListTaskPushNotificationConfigsRequest {
                    tenant: None,
                    task_id: task_id.to_string(),
                    page_size: None,
                    page_token: None,
                };
                let result = handler.on_list_push_configs(req).await?;
                return Ok(RestResult::Json(serde_json::to_value(result)?));
            }
            // GET /tasks/{id}
            let req = GetTaskRequest {
                tenant: None,
                id: remaining.to_string(),
                history_length: None,
            };
            let task = handler.on_get_task(req).await?;
            Ok(RestResult::Json(serde_json::to_value(task)?))
        }
        ("DELETE", p) if p.starts_with("/tasks/") && p.contains("/pushNotificationConfigs/") => {
            let remaining = &p["/tasks/".len()..];
            let parts: Vec<&str> = remaining.split("/pushNotificationConfigs/").collect();
            if parts.len() == 2 {
                let req = DeleteTaskPushNotificationConfigRequest {
                    tenant: None,
                    task_id: parts[0].to_string(),
                    config_id: parts[1].to_string(),
                };
                handler.on_delete_push_config(req).await?;
                return Ok(RestResult::Json(serde_json::Value::Null));
            }
            Err(A2aError::method_not_found(p))
        }
        _ => Err(A2aError::method_not_found(&format!("{method} {path}"))),
    }
}

/// Strip an optional `/{tenant}` prefix from a path. Returns (tenant, remaining_path).
fn strip_tenant(path: &str) -> (Option<String>, &str) {
    // Known non-tenant prefixes
    let known = ["/message:", "/tasks", "/extendedAgentCard", "/.well-known"];
    for prefix in &known {
        if path.starts_with(prefix) {
            return (None, path);
        }
    }

    // Try stripping /{tenant}/...
    if let Some(rest) = path.strip_prefix('/')
        && let Some(slash_pos) = rest.find('/')
    {
        let tenant = &rest[..slash_pos];
        let remaining = &rest[slash_pos..];
        return (Some(tenant.to_string()), remaining);
    }

    (None, path)
}
