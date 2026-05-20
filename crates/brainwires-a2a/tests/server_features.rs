//! Tests for server features: streaming dispatch, CORS, body limits,
//! cancel/subscribe, tenant stripping, and SSE response construction.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use brainwires_a2a::*;
use futures::{Stream, StreamExt};
use tokio::sync::Mutex;

/// Test handler with task storage and optional push notification support.
struct FullTestHandler {
    card: AgentCard,
    tasks: Mutex<Vec<Task>>,
}

impl FullTestHandler {
    fn new() -> Self {
        Self {
            card: AgentCard {
                name: "Full Test Agent".into(),
                description: "Tests all dispatch paths".into(),
                version: "1.0.0".into(),
                supported_interfaces: vec![],
                capabilities: AgentCapabilities {
                    streaming: Some(true),
                    push_notifications: Some(false),
                    extended_agent_card: None,
                    extensions: None,
                },
                skills: vec![],
                default_input_modes: vec!["text/plain".into()],
                default_output_modes: vec!["text/plain".into()],
                provider: None,
                security_schemes: None,
                security_requirements: None,
                documentation_url: None,
                icon_url: None,
                signatures: None,
            },
            tasks: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl A2aHandler for FullTestHandler {
    fn agent_card(&self) -> &AgentCard {
        &self.card
    }

    async fn on_send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError> {
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            context_id: req.message.context_id.clone(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(Message::agent_text("Done")),
                timestamp: None,
            },
            artifacts: None,
            history: Some(vec![req.message]),
            metadata: None,
        };
        self.tasks.lock().await.push(task.clone());
        Ok(SendMessageResponse {
            task: Some(task),
            message: None,
        })
    }

    async fn on_send_streaming_message(
        &self,
        _req: SendMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
    {
        let stream = async_stream::stream! {
            yield Ok(StreamResponse {
                task: None,
                message: None,
                status_update: Some(TaskStatusUpdateEvent {
                    task_id: "stream-t".into(),
                    context_id: "ctx".into(),
                    status: TaskStatus {
                        state: TaskState::Working,
                        message: None,
                        timestamp: None,
                    },
                    trace_id: None,
                    sequence: None,
                    metadata: None,
                }),
                artifact_update: None,
            });
            yield Ok(StreamResponse {
                task: None,
                message: None,
                status_update: Some(TaskStatusUpdateEvent {
                    task_id: "stream-t".into(),
                    context_id: "ctx".into(),
                    status: TaskStatus {
                        state: TaskState::Completed,
                        message: Some(Message::agent_text("Streamed")),
                        timestamp: None,
                    },
                    trace_id: None,
                    sequence: None,
                    metadata: None,
                }),
                artifact_update: None,
            });
        };
        Ok(Box::pin(stream))
    }

    async fn on_get_task(&self, req: GetTaskRequest) -> Result<Task, A2aError> {
        let tasks = self.tasks.lock().await;
        tasks
            .iter()
            .find(|t| t.id == req.id)
            .cloned()
            .ok_or_else(|| A2aError::task_not_found(&req.id))
    }

    async fn on_list_tasks(&self, _req: ListTasksRequest) -> Result<ListTasksResponse, A2aError> {
        let tasks = self.tasks.lock().await;
        Ok(ListTasksResponse {
            tasks: tasks.clone(),
            next_page_token: String::new(),
            page_size: tasks.len() as i32,
            total_size: tasks.len() as i32,
        })
    }

    async fn on_cancel_task(&self, req: CancelTaskRequest) -> Result<Task, A2aError> {
        let mut tasks = self.tasks.lock().await;
        if let Some(task) = tasks.iter_mut().find(|t| t.id == req.id) {
            task.status.state = TaskState::Canceled;
            Ok(task.clone())
        } else {
            Err(A2aError::task_not_found(&req.id))
        }
    }

    async fn on_subscribe_to_task(
        &self,
        req: SubscribeToTaskRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
    {
        let task = self
            .on_get_task(GetTaskRequest {
                tenant: None,
                id: req.id,
                history_length: None,
            })
            .await?;

        let stream = async_stream::stream! {
            yield Ok(StreamResponse {
                task: Some(task),
                message: None,
                status_update: None,
                artifact_update: None,
            });
        };
        Ok(Box::pin(stream))
    }
}

// ---- JSON-RPC Streaming dispatch ----

#[tokio::test]
async fn test_jsonrpc_streaming_dispatch_returns_none() {
    let handler = Arc::new(FullTestHandler::new());

    // SendStreamingMessage should return Ok(None) from dispatch (handled by SSE layer)
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SendStreamingMessage".into(),
        params: Some(serde_json::json!({
            "message": {"messageId": "1", "role": "ROLE_USER", "parts": [{"text": "hi"}]}
        })),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Ok(None) => {} // Correct — streaming handled separately
        other => panic!("Expected Ok(None) for streaming method, got {other:?}"),
    }
}

#[tokio::test]
async fn test_jsonrpc_resubscribe_returns_none() {
    let handler = Arc::new(FullTestHandler::new());

    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SubscribeToTask".into(),
        params: Some(serde_json::json!({"id": "t-1"})),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Ok(None) => {}
        other => panic!("Expected Ok(None) for resubscribe, got {other:?}"),
    }
}

// ---- JSON-RPC cancel task ----

#[tokio::test]
async fn test_jsonrpc_cancel_task() {
    let handler = Arc::new(FullTestHandler::new());

    // Create a task first
    let send_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SendMessage".into(),
        params: Some(
            serde_json::to_value(&SendMessageRequest {
                tenant: None,
                message: Message::user_text("Create me"),
                configuration: None,
                metadata: None,
            })
            .unwrap(),
        ),
        id: RequestId::Number(1),
    };
    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &send_req)
        .await
        .unwrap()
        .unwrap();
    let smr: SendMessageResponse = serde_json::from_value(result.result.unwrap()).unwrap();
    let task = smr.task.unwrap();

    // Cancel it
    let cancel_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "CancelTask".into(),
        params: Some(serde_json::json!({"id": task.id})),
        id: RequestId::Number(2),
    };
    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &cancel_req)
        .await
        .unwrap()
        .unwrap();
    let canceled: Task = serde_json::from_value(result.result.unwrap()).unwrap();
    assert_eq!(canceled.status.state, TaskState::Canceled);
}

// ---- JSON-RPC list tasks ----

#[tokio::test]
async fn test_jsonrpc_list_tasks() {
    let handler = Arc::new(FullTestHandler::new());

    // Create two tasks
    for msg in ["First", "Second"] {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "SendMessage".into(),
            params: Some(
                serde_json::to_value(&SendMessageRequest {
                    tenant: None,
                    message: Message::user_text(msg),
                    configuration: None,
                    metadata: None,
                })
                .unwrap(),
            ),
            id: RequestId::Number(1),
        };
        brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req)
            .await
            .unwrap();
    }

    let list_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "ListTasks".into(),
        params: Some(serde_json::json!({})),
        id: RequestId::Number(3),
    };
    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &list_req)
        .await
        .unwrap()
        .unwrap();
    let resp: ListTasksResponse = serde_json::from_value(result.result.unwrap()).unwrap();
    assert_eq!(resp.tasks.len(), 2);
}

// ---- JSON-RPC invalid params ----

#[tokio::test]
async fn test_jsonrpc_invalid_params() {
    let handler = Arc::new(FullTestHandler::new());

    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SendMessage".into(),
        params: Some(serde_json::json!({"wrong": "params"})),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Err(resp) => {
            assert!(resp.error.is_some());
        }
        _ => panic!("Expected error for invalid params"),
    }
}

// ---- REST streaming dispatch ----

#[tokio::test]
async fn test_rest_streaming_message() {
    let handler = Arc::new(FullTestHandler::new());
    let body = serde_json::to_vec(&SendMessageRequest {
        tenant: None,
        message: Message::user_text("Stream please"),
        configuration: None,
        metadata: None,
    })
    .unwrap();

    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/message:stream",
        &body,
    )
    .await;

    match result {
        Ok(brainwires_a2a::server::rest_router::RestResult::Stream(stream)) => {
            let items: Vec<_> = stream.collect().await;
            assert_eq!(items.len(), 2);
            // First event: Working
            let first = items[0].as_ref().unwrap();
            assert!(first.status_update.is_some());
            assert_eq!(
                first.status_update.as_ref().unwrap().status.state,
                TaskState::Working
            );
            // Second event: Completed
            let second = items[1].as_ref().unwrap();
            assert!(second.status_update.is_some());
            assert_eq!(
                second.status_update.as_ref().unwrap().status.state,
                TaskState::Completed
            );
        }
        _ => panic!("Expected Stream result"),
    }
}

// ---- REST cancel task ----

#[tokio::test]
async fn test_rest_cancel_task() {
    let handler = Arc::new(FullTestHandler::new());

    // Create a task
    let body = serde_json::to_vec(&SendMessageRequest {
        tenant: None,
        message: Message::user_text("Cancel me"),
        configuration: None,
        metadata: None,
    })
    .unwrap();
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/message:send",
        &body,
    )
    .await
    .unwrap();
    let smr: SendMessageResponse = match result {
        brainwires_a2a::server::rest_router::RestResult::Json(v) => {
            serde_json::from_value(v).unwrap()
        }
        _ => panic!("Expected JSON"),
    };
    let task = smr.task.unwrap();

    // Cancel it
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        &format!("/tasks/{}:cancel", task.id),
        &[],
    )
    .await
    .unwrap();
    let canceled: Task = match result {
        brainwires_a2a::server::rest_router::RestResult::Json(v) => {
            serde_json::from_value(v).unwrap()
        }
        _ => panic!("Expected JSON"),
    };
    assert_eq!(canceled.status.state, TaskState::Canceled);
}

// ---- REST get single task ----

#[tokio::test]
async fn test_rest_get_single_task() {
    let handler = Arc::new(FullTestHandler::new());

    // Create a task
    let body = serde_json::to_vec(&SendMessageRequest {
        tenant: None,
        message: Message::user_text("Get me"),
        configuration: None,
        metadata: None,
    })
    .unwrap();
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/message:send",
        &body,
    )
    .await
    .unwrap();
    let smr: SendMessageResponse = match result {
        brainwires_a2a::server::rest_router::RestResult::Json(v) => {
            serde_json::from_value(v).unwrap()
        }
        _ => panic!("Expected JSON"),
    };
    let task = smr.task.unwrap();

    // Get it
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "GET",
        &format!("/tasks/{}", task.id),
        &[],
    )
    .await
    .unwrap();
    let fetched: Task = match result {
        brainwires_a2a::server::rest_router::RestResult::Json(v) => {
            serde_json::from_value(v).unwrap()
        }
        _ => panic!("Expected JSON"),
    };
    assert_eq!(fetched.id, task.id);
}

// ---- REST subscribe to task (streaming) ----

#[tokio::test]
async fn test_rest_subscribe_to_task() {
    let handler = Arc::new(FullTestHandler::new());

    // Create a task first
    let body = serde_json::to_vec(&SendMessageRequest {
        tenant: None,
        message: Message::user_text("Subscribe test"),
        configuration: None,
        metadata: None,
    })
    .unwrap();
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/message:send",
        &body,
    )
    .await
    .unwrap();
    let smr: SendMessageResponse = match result {
        brainwires_a2a::server::rest_router::RestResult::Json(v) => {
            serde_json::from_value(v).unwrap()
        }
        _ => panic!("Expected JSON"),
    };
    let task = smr.task.unwrap();

    // Subscribe (now POST instead of GET)
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        &format!("/tasks/{}:subscribe", task.id),
        &[],
    )
    .await
    .unwrap();
    match result {
        brainwires_a2a::server::rest_router::RestResult::Stream(stream) => {
            let items: Vec<_> = stream.collect().await;
            assert_eq!(items.len(), 1);
            let item = items[0].as_ref().unwrap();
            assert!(item.task.is_some());
            assert_eq!(item.task.as_ref().unwrap().id, task.id);
        }
        _ => panic!("Expected Stream"),
    }
}

// ---- REST tenant stripping ----

#[tokio::test]
async fn test_rest_tenant_prefix_stripped() {
    let handler = Arc::new(FullTestHandler::new());

    // With tenant prefix: /my-tenant/tasks
    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "GET",
        "/my-tenant/tasks",
        &[],
    )
    .await;
    // Should work — tenant is stripped, routes to /tasks
    match result {
        Ok(brainwires_a2a::server::rest_router::RestResult::Json(_)) => {}
        _ => panic!("Expected JSON result for tenant-prefixed path"),
    }
}

// ---- REST unknown route ----

#[tokio::test]
async fn test_rest_unknown_route() {
    let handler = Arc::new(FullTestHandler::new());

    let result =
        brainwires_a2a::server::rest_router::dispatch_rest(&handler, "GET", "/nonexistent", &[])
            .await;
    match result {
        Err(e) => assert_eq!(e.code, brainwires_a2a::error::METHOD_NOT_FOUND),
        Ok(_) => panic!("Expected error for unknown route"),
    }
}

// ---- REST extended agent card ----

#[tokio::test]
async fn test_rest_extended_card_not_configured() {
    let handler = Arc::new(FullTestHandler::new());

    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "GET",
        "/extendedAgentCard",
        &[],
    )
    .await;
    match result {
        Err(e) => assert_eq!(e.code, brainwires_a2a::error::EXTENDED_CARD_NOT_CONFIGURED),
        Ok(_) => panic!("Expected extended card not configured error"),
    }
}

// ---- SSE response construction ----

#[tokio::test]
async fn test_sse_response_jsonrpc_format() {
    use brainwires_a2a::server::sse_response;

    let event = StreamResponse {
        task: None,
        message: None,
        status_update: Some(TaskStatusUpdateEvent {
            task_id: "t-1".into(),
            context_id: "ctx".into(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
                timestamp: None,
            },
            trace_id: None,
            sequence: None,
            metadata: None,
        }),
        artifact_update: None,
    };

    let stream: Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> =
        Box::pin(futures::stream::once(async { Ok(event) }));

    let sse_stream = sse_response::stream_to_sse(RequestId::Number(42), stream);
    let frames: Vec<_> = sse_stream.collect().await;
    assert_eq!(frames.len(), 1);

    let frame = frames[0].as_ref().unwrap();
    let data = frame.data_ref().unwrap();
    let text = String::from_utf8_lossy(data);
    assert!(text.starts_with("data: "));
    assert!(text.ends_with("\n\n"));

    // Should be valid JSON-RPC
    let inner = text.trim_start_matches("data: ").trim();
    let resp: JsonRpcResponse = serde_json::from_str(inner).unwrap();
    assert_eq!(resp.id, RequestId::Number(42));
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());
}

#[tokio::test]
async fn test_sse_response_rest_format() {
    use brainwires_a2a::server::sse_response;

    let event = StreamResponse {
        task: None,
        message: None,
        status_update: Some(TaskStatusUpdateEvent {
            task_id: "t-1".into(),
            context_id: "ctx".into(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: None,
                timestamp: None,
            },
            trace_id: None,
            sequence: None,
            metadata: None,
        }),
        artifact_update: None,
    };

    let stream: Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> =
        Box::pin(futures::stream::once(async { Ok(event) }));

    let sse_stream = sse_response::stream_to_sse_rest(stream);
    let frames: Vec<_> = sse_stream.collect().await;
    assert_eq!(frames.len(), 1);

    let frame = frames[0].as_ref().unwrap();
    let data = frame.data_ref().unwrap();
    let text = String::from_utf8_lossy(data);
    assert!(text.starts_with("data: "));

    // Should be raw StreamResponse JSON, NOT wrapped in JSON-RPC
    let inner = text.trim_start_matches("data: ").trim();
    let event: StreamResponse = serde_json::from_str(inner).unwrap();
    assert!(event.status_update.is_some());
    assert_eq!(event.status_update.unwrap().task_id, "t-1");
}

#[tokio::test]
async fn test_sse_response_error_event() {
    use brainwires_a2a::server::sse_response;

    let err = A2aError::task_not_found("t-99");
    let stream: Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> =
        Box::pin(futures::stream::once(async { Err(err) }));

    let sse_stream = sse_response::stream_to_sse(RequestId::Number(1), stream);
    let frames: Vec<_> = sse_stream.collect().await;
    assert_eq!(frames.len(), 1);

    let frame = frames[0].as_ref().unwrap();
    let data = frame.data_ref().unwrap();
    let text = String::from_utf8_lossy(data);
    let inner = text.trim_start_matches("data: ").trim();
    let resp: JsonRpcResponse = serde_json::from_str(inner).unwrap();
    assert!(resp.error.is_some());
    assert_eq!(
        resp.error.unwrap().code,
        brainwires_a2a::error::TASK_NOT_FOUND
    );
}

// ---- JSON-RPC push notification CRUD ----

#[tokio::test]
async fn test_jsonrpc_push_config_set_unsupported() {
    let handler = Arc::new(FullTestHandler::new());

    let config = TaskPushNotificationConfig {
        tenant: None,
        config_id: None,
        task_id: "t-1".into(),
        url: "https://example.com/hook".into(),
        token: None,
        authentication: None,
        created_at: None,
    };
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "CreateTaskPushNotificationConfig".into(),
        params: Some(serde_json::to_value(&config).unwrap()),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Err(resp) => {
            let err = resp.error.unwrap();
            assert_eq!(err.code, brainwires_a2a::error::PUSH_NOT_SUPPORTED);
        }
        _ => panic!("Expected push not supported error"),
    }
}

#[tokio::test]
async fn test_jsonrpc_push_config_get_unsupported() {
    let handler = Arc::new(FullTestHandler::new());

    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "GetTaskPushNotificationConfig".into(),
        params: Some(serde_json::json!({"taskId": "t-1", "configId": "cfg-1"})),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_jsonrpc_push_config_list_unsupported() {
    let handler = Arc::new(FullTestHandler::new());

    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "ListTaskPushNotificationConfigs".into(),
        params: Some(serde_json::json!({"taskId": "t-1"})),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_jsonrpc_push_config_delete_unsupported() {
    let handler = Arc::new(FullTestHandler::new());

    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "DeleteTaskPushNotificationConfig".into(),
        params: Some(serde_json::json!({"taskId": "t-1", "configId": "cfg-1"})),
        id: RequestId::Number(1),
    };

    let result = brainwires_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    assert!(result.is_err());
}

// ---- REST push notification ----

#[tokio::test]
async fn test_rest_push_config_create_unsupported() {
    let handler = Arc::new(FullTestHandler::new());
    let config = TaskPushNotificationConfig {
        tenant: None,
        config_id: None,
        task_id: "t-1".into(),
        url: "https://example.com/hook".into(),
        token: None,
        authentication: None,
        created_at: None,
    };
    let body = serde_json::to_vec(&config).unwrap();

    let result = brainwires_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/tasks/t-1/pushNotificationConfigs",
        &body,
    )
    .await;
    match result {
        Err(e) => assert_eq!(e.code, brainwires_a2a::error::PUSH_NOT_SUPPORTED),
        Ok(_) => panic!("Expected push not supported error"),
    }
}

// ---- Error types ----

#[test]
fn test_error_constructors() {
    let e = A2aError::task_not_found("t-1");
    assert_eq!(e.code, brainwires_a2a::error::TASK_NOT_FOUND);
    assert!(e.message.contains("t-1"));

    let e = A2aError::task_not_cancelable("t-2");
    assert_eq!(e.code, brainwires_a2a::error::TASK_NOT_CANCELABLE);

    let e = A2aError::push_not_supported();
    assert_eq!(e.code, brainwires_a2a::error::PUSH_NOT_SUPPORTED);

    let e = A2aError::unsupported_operation("grpc");
    assert_eq!(e.code, brainwires_a2a::error::UNSUPPORTED_OPERATION);

    let e = A2aError::content_type_not_supported("video/mp4");
    assert_eq!(e.code, brainwires_a2a::error::CONTENT_TYPE_NOT_SUPPORTED);

    let e = A2aError::invalid_request("too big");
    assert_eq!(e.code, brainwires_a2a::error::INVALID_REQUEST);

    let e = A2aError::internal("oops");
    assert_eq!(e.code, brainwires_a2a::error::INTERNAL_ERROR);

    let e = A2aError::method_not_found("foo/bar");
    assert_eq!(e.code, brainwires_a2a::error::METHOD_NOT_FOUND);

    let e = A2aError::invalid_params("missing field");
    assert_eq!(e.code, brainwires_a2a::error::INVALID_PARAMS);

    let e = A2aError::parse_error("bad json");
    assert_eq!(e.code, brainwires_a2a::error::JSON_PARSE_ERROR);

    let e = A2aError::extended_card_not_configured();
    assert_eq!(e.code, brainwires_a2a::error::EXTENDED_CARD_NOT_CONFIGURED);
}

#[test]
fn test_error_with_data() {
    let e = A2aError::internal("test").with_data(serde_json::json!({"detail": "extra"}));
    assert!(e.data.is_some());
    assert_eq!(e.data.unwrap()["detail"], "extra");
}

#[test]
fn test_error_display() {
    let e = A2aError::internal("something broke");
    let display = format!("{e}");
    assert!(display.contains("something broke"));
    assert!(display.contains(&e.code.to_string()));
}
