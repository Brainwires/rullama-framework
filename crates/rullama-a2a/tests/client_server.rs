//! In-process client <-> server integration test for JSON-RPC and REST.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use rullama_a2a::*;
use futures::Stream;
use tokio::sync::Mutex;

/// Simple test handler that stores tasks in memory.
struct TestHandler {
    card: AgentCard,
    tasks: Mutex<Vec<Task>>,
}

impl TestHandler {
    fn new() -> Self {
        Self {
            card: AgentCard {
                name: "Test Agent".into(),
                description: "Integration test agent".into(),
                version: "0.7.0".into(),
                supported_interfaces: vec![],
                capabilities: AgentCapabilities::default(),
                skills: vec![AgentSkill {
                    id: "echo".into(),
                    name: "Echo".into(),
                    description: "Echoes back".into(),
                    tags: vec!["echo".into()],
                    examples: None,
                    input_modes: None,
                    output_modes: None,
                    security_requirements: None,
                }],
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
impl A2aHandler for TestHandler {
    fn agent_card(&self) -> &AgentCard {
        &self.card
    }

    async fn on_send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError> {
        // Echo: create a task with the message
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            context_id: req.message.context_id.clone(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(Message::agent_text("Echo received")),
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
        req: SendMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
    {
        let task_id = uuid::Uuid::new_v4().to_string();
        let context_id = req
            .message
            .context_id
            .clone()
            .unwrap_or_else(|| "default".into());

        let stream = async_stream::stream! {
            yield Ok(StreamResponse {
                task: None,
                message: None,
                status_update: Some(TaskStatusUpdateEvent {
                    task_id: task_id.clone(),
                    context_id: context_id.clone(),
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
                    task_id,
                    context_id,
                    status: TaskStatus {
                        state: TaskState::Completed,
                        message: Some(Message::agent_text("Done")),
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

#[tokio::test]
async fn test_jsonrpc_send_message() {
    let handler = Arc::new(TestHandler::new());
    let req = rullama_a2a::jsonrpc::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SendMessage".into(),
        params: Some(
            serde_json::to_value(&SendMessageRequest {
                tenant: None,
                message: Message::user_text("Hello test"),
                configuration: None,
                metadata: None,
            })
            .unwrap(),
        ),
        id: RequestId::Number(1),
    };

    let result = rullama_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Ok(Some(resp)) => {
            assert!(
                resp.error.is_none(),
                "Expected success, got error: {:?}",
                resp.error
            );
            assert!(resp.result.is_some());
            // Verify the result is a SendMessageResponse with a task
            let smr: SendMessageResponse = serde_json::from_value(resp.result.unwrap()).unwrap();
            assert!(smr.task.is_some());
            assert_eq!(smr.task.unwrap().status.state, TaskState::Completed);
        }
        Ok(None) => panic!("Expected Some response for non-streaming method"),
        Err(resp) => panic!("Expected Ok, got Err: {:?}", resp.error),
    }
}

#[tokio::test]
async fn test_jsonrpc_task_not_found() {
    let handler = Arc::new(TestHandler::new());
    let req = rullama_a2a::jsonrpc::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "GetTask".into(),
        params: Some(serde_json::json!({
            "id": "nonexistent-task"
        })),
        id: RequestId::Number(2),
    };

    let result = rullama_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Err(resp) => {
            let err = resp.error.unwrap();
            assert_eq!(err.code, rullama_a2a::error::TASK_NOT_FOUND);
        }
        _ => panic!("Expected task not found error"),
    }
}

#[tokio::test]
async fn test_jsonrpc_method_not_found() {
    let handler = Arc::new(TestHandler::new());
    let req = rullama_a2a::jsonrpc::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "nonexistent/method".into(),
        params: None,
        id: RequestId::Number(3),
    };

    let result = rullama_a2a::server::jsonrpc_router::dispatch(&handler, &req).await;
    match result {
        Err(resp) => {
            let err = resp.error.unwrap();
            assert_eq!(err.code, rullama_a2a::error::METHOD_NOT_FOUND);
        }
        _ => panic!("Expected method not found error"),
    }
}

#[tokio::test]
async fn test_rest_dispatch_send_message() {
    let handler = Arc::new(TestHandler::new());
    let body = serde_json::to_vec(&SendMessageRequest {
        tenant: None,
        message: Message::user_text("REST test"),
        configuration: None,
        metadata: None,
    })
    .unwrap();

    let result = rullama_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/message:send",
        &body,
    )
    .await;

    match result {
        Ok(rullama_a2a::server::rest_router::RestResult::Json(val)) => {
            // Should be a SendMessageResponse with a Task
            let smr: SendMessageResponse = serde_json::from_value(val).unwrap();
            assert!(smr.task.is_some());
            assert_eq!(smr.task.unwrap().status.state, TaskState::Completed);
        }
        Ok(rullama_a2a::server::rest_router::RestResult::Stream(_)) => {
            panic!("Expected JSON, got stream");
        }
        Err(e) => panic!("Expected success: {e}"),
    }
}

#[tokio::test]
async fn test_rest_dispatch_get_tasks() {
    let handler = Arc::new(TestHandler::new());

    // First create a task
    let body = serde_json::to_vec(&SendMessageRequest {
        tenant: None,
        message: Message::user_text("Create task"),
        configuration: None,
        metadata: None,
    })
    .unwrap();
    let _ = rullama_a2a::server::rest_router::dispatch_rest(
        &handler,
        "POST",
        "/message:send",
        &body,
    )
    .await
    .unwrap();

    // List tasks
    let result =
        rullama_a2a::server::rest_router::dispatch_rest(&handler, "GET", "/tasks", &[]).await;

    match result {
        Ok(rullama_a2a::server::rest_router::RestResult::Json(val)) => {
            let resp: ListTasksResponse = serde_json::from_value(val).unwrap();
            assert_eq!(resp.tasks.len(), 1);
        }
        _ => panic!("Expected JSON response"),
    }
}

#[tokio::test]
async fn test_agent_card_discovery_handler() {
    let handler = TestHandler::new();
    let card = handler.agent_card();
    assert_eq!(card.name, "Test Agent");
    assert_eq!(card.skills.len(), 1);
    assert_eq!(card.skills[0].id, "echo");
}

#[tokio::test]
async fn test_push_notification_default_unsupported() {
    let handler = TestHandler::new();
    let result = handler
        .on_create_push_config(TaskPushNotificationConfig {
            tenant: None,
            config_id: None,
            task_id: "t-1".into(),
            url: "https://example.com".into(),
            token: None,
            authentication: None,
            created_at: None,
        })
        .await;
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().code,
        rullama_a2a::error::PUSH_NOT_SUPPORTED
    );
}

#[tokio::test]
async fn test_extended_card_default_not_configured() {
    let handler = TestHandler::new();
    let result = handler
        .on_get_extended_agent_card(GetExtendedAgentCardRequest::default())
        .await;
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().code,
        rullama_a2a::error::EXTENDED_CARD_NOT_CONFIGURED
    );
}
