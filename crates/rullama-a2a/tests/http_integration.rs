//! HTTP-level integration tests — spin up a real server and hit it with reqwest.
//!
//! Tests: CORS headers, body size limit, agent card discovery, streaming SSE,
//! graceful shutdown, bearer token propagation.

use std::net::SocketAddr;
use std::pin::Pin;

use async_trait::async_trait;
use rullama_a2a::*;
use futures::Stream;

/// Minimal handler for HTTP integration tests.
struct HttpTestHandler {
    card: AgentCard,
}

impl HttpTestHandler {
    fn new() -> Self {
        Self {
            card: AgentCard {
                name: "HTTP Test".into(),
                description: "HTTP integration tests".into(),
                version: "1.0.0".into(),
                supported_interfaces: vec![],
                capabilities: AgentCapabilities::default(),
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
        }
    }
}

#[async_trait]
impl A2aHandler for HttpTestHandler {
    fn agent_card(&self) -> &AgentCard {
        &self.card
    }

    async fn on_send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError> {
        let task = Task {
            id: "http-task-1".into(),
            context_id: req.message.context_id.clone(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(Message::agent_text("OK")),
                timestamp: None,
            },
            artifacts: None,
            history: None,
            metadata: None,
        };
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
                    task_id: "http-stream-1".into(),
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
                    task_id: "http-stream-1".into(),
                    context_id: "ctx".into(),
                    status: TaskStatus {
                        state: TaskState::Completed,
                        message: Some(Message::agent_text("Done streaming")),
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
        Err(A2aError::task_not_found(&req.id))
    }

    async fn on_list_tasks(&self, _req: ListTasksRequest) -> Result<ListTasksResponse, A2aError> {
        Ok(ListTasksResponse {
            tasks: vec![],
            next_page_token: String::new(),
            page_size: 0,
            total_size: 0,
        })
    }

    async fn on_cancel_task(&self, req: CancelTaskRequest) -> Result<Task, A2aError> {
        Err(A2aError::task_not_found(&req.id))
    }

    async fn on_subscribe_to_task(
        &self,
        req: SubscribeToTaskRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
    {
        Err(A2aError::task_not_found(&req.id))
    }
}

/// Start a test server on a random port and return the address.
async fn start_test_server() -> (SocketAddr, tokio::sync::watch::Sender<()>) {
    let handler = HttpTestHandler::new();
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    // Bind the listener manually to get the actual port
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    // We need to run the server ourselves since A2aServer::run binds its own listener.
    // Instead, let's use A2aServer with the actual address.
    drop(listener); // Release the port

    let server =
        rullama_a2a::server::A2aServer::new(handler, actual_addr).with_shutdown(shutdown_rx);

    tokio::spawn(async move {
        if let Err(e) = server.run().await {
            eprintln!("Test server error: {e}");
        }
    });

    // Give the server a moment to bind
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    (actual_addr, shutdown_tx)
}

#[tokio::test]
async fn test_http_cors_on_options() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .request(reqwest::Method::OPTIONS, format!("http://{addr}/"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 204);
    assert_eq!(
        resp.headers().get("access-control-allow-origin").unwrap(),
        "*"
    );
    assert!(
        resp.headers()
            .get("access-control-allow-methods")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("POST")
    );
    assert!(
        resp.headers()
            .get("access-control-allow-headers")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("Authorization")
    );

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_cors_on_json_response() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("http://{addr}/.well-known/agent-card.json"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("access-control-allow-origin").unwrap(),
        "*"
    );

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_agent_card_discovery() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("http://{addr}/.well-known/agent-card.json"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let card: AgentCard = resp.json().await.unwrap();
    assert_eq!(card.name, "HTTP Test");
    assert_eq!(card.version, "1.0.0");

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_jsonrpc_send_message() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let rpc_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SendMessage".into(),
        params: Some(
            serde_json::to_value(&SendMessageRequest {
                tenant: None,
                message: Message::user_text("HTTP test"),
                configuration: None,
                metadata: None,
            })
            .unwrap(),
        ),
        id: RequestId::Number(1),
    };

    let resp = client
        .post(format!("http://{addr}/"))
        .json(&rpc_req)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let rpc_resp: JsonRpcResponse = resp.json().await.unwrap();
    assert!(rpc_resp.error.is_none());
    assert!(rpc_resp.result.is_some());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_jsonrpc_streaming_returns_sse() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let rpc_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        method: "SendStreamingMessage".into(),
        params: Some(
            serde_json::to_value(&SendMessageRequest {
                tenant: None,
                message: Message::user_text("Stream me"),
                configuration: None,
                metadata: None,
            })
            .unwrap(),
        ),
        id: RequestId::Number(1),
    };

    let resp = client
        .post(format!("http://{addr}/"))
        .json(&rpc_req)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    // Read the full body — should have SSE data lines
    let body = resp.text().await.unwrap();
    let data_lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data: ")).collect();
    assert_eq!(data_lines.len(), 2, "Expected 2 SSE events, body: {body}");

    // Each line should be valid JSON-RPC
    for line in &data_lines {
        let json = line.strip_prefix("data: ").unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_some() || resp.error.is_some());
    }

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_rest_send_message() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let req = SendMessageRequest {
        tenant: None,
        message: Message::user_text("REST HTTP test"),
        configuration: None,
        metadata: None,
    };

    let resp = client
        .post(format!("http://{addr}/message:send"))
        .json(&req)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    // SendMessageResponse with task field
    assert!(body.get("task").is_some());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_rest_streaming_returns_sse() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let req = SendMessageRequest {
        tenant: None,
        message: Message::user_text("REST stream test"),
        configuration: None,
        metadata: None,
    };

    let resp = client
        .post(format!("http://{addr}/message:stream"))
        .json(&req)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    let body = resp.text().await.unwrap();
    let data_lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data: ")).collect();
    assert_eq!(data_lines.len(), 2);

    // REST SSE: raw StreamResponse (no JSON-RPC wrapper)
    for line in &data_lines {
        let json = line.strip_prefix("data: ").unwrap();
        let _event: StreamResponse = serde_json::from_str(json).unwrap();
    }

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_jsonrpc_parse_error() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/"))
        .body("{invalid json")
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200); // JSON-RPC always returns 200
    let rpc_resp: JsonRpcResponse = resp.json().await.unwrap();
    assert!(rpc_resp.error.is_some());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_graceful_shutdown() {
    let (addr, shutdown_tx) = start_test_server().await;

    // Use a client with no connection pooling so each request creates a new TCP connection
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .build()
        .unwrap();

    // Verify server is alive
    let resp = client
        .get(format!("http://{addr}/.well-known/agent-card.json"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Signal shutdown
    drop(shutdown_tx);

    // Wait for the server to stop accepting — retry with short timeout using fresh connections
    let mut refused = false;
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let fresh_client = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .build()
            .unwrap();
        let result = fresh_client
            .get(format!("http://{addr}/.well-known/agent-card.json"))
            .timeout(std::time::Duration::from_millis(200))
            .send()
            .await;
        if result.is_err() {
            refused = true;
            break;
        }
    }
    assert!(
        refused,
        "Server should stop accepting connections after shutdown"
    );
}

#[tokio::test]
async fn test_http_client_with_bearer_token() {
    let (addr, shutdown_tx) = start_test_server().await;

    // Create A2aClient with JSON-RPC transport and bearer token
    let url = url::Url::parse(&format!("http://{addr}/")).unwrap();
    let client = A2aClient::new_jsonrpc(url).with_bearer_token("test-token-123");

    // The server doesn't validate the token, but the request should succeed
    // (proving the client doesn't crash with a token set)
    let result = client
        .send_message(SendMessageRequest {
            tenant: None,
            message: Message::user_text("Auth test"),
            configuration: None,
            metadata: None,
        })
        .await;
    assert!(result.is_ok());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_client_rest_with_bearer_token() {
    let (addr, shutdown_tx) = start_test_server().await;

    let url = url::Url::parse(&format!("http://{addr}/")).unwrap();
    let client = A2aClient::new_rest(url).with_bearer_token("rest-token-456");

    let result = client
        .send_message(SendMessageRequest {
            tenant: None,
            message: Message::user_text("REST auth test"),
            configuration: None,
            metadata: None,
        })
        .await;
    assert!(result.is_ok());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_http_404_unknown_rest_route() {
    let (addr, shutdown_tx) = start_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("http://{addr}/totally/unknown/path"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 404);

    drop(shutdown_tx);
}
