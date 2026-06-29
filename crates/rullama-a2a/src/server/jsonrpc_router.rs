//! JSON-RPC method dispatch — routes JSON-RPC requests to handler methods.

use std::sync::Arc;

use crate::error::A2aError;
use crate::jsonrpc::*;
use crate::params::*;
use crate::push_notification::TaskPushNotificationConfig;
use crate::server::handler::A2aHandler;

/// Dispatch a JSON-RPC request to the appropriate handler method.
///
/// Returns `Ok(Some(response))` for non-streaming methods,
/// `Ok(None)` for streaming methods (handled separately),
/// or `Err(response)` for errors.
pub async fn dispatch<H: A2aHandler>(
    handler: &Arc<H>,
    request: &JsonRpcRequest,
) -> Result<Option<JsonRpcResponse>, JsonRpcResponse> {
    let params = request.params.clone().unwrap_or(serde_json::Value::Null);
    let id = request.id.clone();

    match request.method.as_str() {
        METHOD_MESSAGE_SEND => {
            let req: SendMessageRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_send_message(req).await {
                Ok(result) => {
                    let val = serde_json::to_value(result)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_MESSAGE_STREAM | METHOD_TASKS_RESUBSCRIBE => {
            // Streaming methods are handled by the SSE layer, not here.
            Ok(None)
        }
        METHOD_TASKS_GET => {
            let req: GetTaskRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_get_task(req).await {
                Ok(task) => {
                    let val = serde_json::to_value(task)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_TASKS_LIST => {
            let req: ListTasksRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_list_tasks(req).await {
                Ok(resp) => {
                    let val = serde_json::to_value(resp)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_TASKS_CANCEL => {
            let req: CancelTaskRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_cancel_task(req).await {
                Ok(task) => {
                    let val = serde_json::to_value(task)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_PUSH_CONFIG_SET => {
            let config: TaskPushNotificationConfig = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_create_push_config(config).await {
                Ok(result) => {
                    let val = serde_json::to_value(result)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_PUSH_CONFIG_GET => {
            let req: GetTaskPushNotificationConfigRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_get_push_config(req).await {
                Ok(result) => {
                    let val = serde_json::to_value(result)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_PUSH_CONFIG_LIST => {
            let req: ListTaskPushNotificationConfigsRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_list_push_configs(req).await {
                Ok(result) => {
                    let val = serde_json::to_value(result)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_PUSH_CONFIG_DELETE => {
            let req: DeleteTaskPushNotificationConfigRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_delete_push_config(req).await {
                Ok(()) => Ok(Some(JsonRpcResponse::success(id, serde_json::Value::Null))),
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        METHOD_EXTENDED_CARD => {
            let req: GetExtendedAgentCardRequest = serde_json::from_value(params)
                .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
            match handler.on_get_extended_agent_card(req).await {
                Ok(card) => {
                    let val = serde_json::to_value(card)
                        .map_err(|e| JsonRpcResponse::error(id.clone(), A2aError::from(e)))?;
                    Ok(Some(JsonRpcResponse::success(id, val)))
                }
                Err(e) => Err(JsonRpcResponse::error(id, e)),
            }
        }
        _ => Err(JsonRpcResponse::error(
            id,
            A2aError::method_not_found(&request.method),
        )),
    }
}
