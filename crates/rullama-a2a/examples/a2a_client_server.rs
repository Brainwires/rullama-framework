//! A2A Client/Server — message and task lifecycle.
//!
//! Demonstrates:
//! - Setting up an `A2aServer` with a custom `A2aHandler`
//! - Connecting an `A2aClient` (JSON-RPC transport)
//! - Sending a message and receiving a task back
//! - Getting, listing, and canceling tasks
//! - Graceful shutdown via a watch channel
//!
//! ```bash
//! cargo run -p rullama-a2a --example a2a_client_server --features full
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::Stream;
use url::Url;

use rullama_a2a::{
    A2aClient, A2aError, A2aHandler, A2aServer, AgentCapabilities, AgentCard, AgentProvider,
    AgentSkill, Artifact, CancelTaskRequest, GetTaskRequest, ListTasksRequest, ListTasksResponse,
    Message, Part, Role, SendMessageRequest, SendMessageResponse, StreamResponse,
    SubscribeToTaskRequest, Task, TaskState, TaskStatus,
};

// ---------------------------------------------------------------------------
// Handler implementation
// ---------------------------------------------------------------------------

/// A demo handler that creates tasks from incoming messages.
struct DemoHandler {
    card: AgentCard,
    tasks: Arc<Mutex<HashMap<String, Task>>>,
}

impl DemoHandler {
    fn new() -> Self {
        let card = AgentCard {
            name: "demo-agent".to_string(),
            description: "A demo A2A agent for the client/server example.".to_string(),
            version: "0.7.0".to_string(),
            supported_interfaces: vec![],
            capabilities: AgentCapabilities {
                streaming: Some(false),
                push_notifications: Some(false),
                extended_agent_card: Some(false),
                extensions: None,
            },
            skills: vec![AgentSkill {
                id: "echo".to_string(),
                name: "Echo".to_string(),
                description: "Echoes the user message back.".to_string(),
                tags: vec!["demo".to_string()],
                examples: Some(vec!["Say hello".to_string()]),
                input_modes: None,
                output_modes: None,
                security_requirements: None,
            }],
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            provider: Some(AgentProvider {
                url: "https://example.com".to_string(),
                organization: "Demo".to_string(),
            }),
            security_schemes: None,
            security_requirements: None,
            documentation_url: None,
            icon_url: None,
            signatures: None,
        };

        Self {
            card,
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl A2aHandler for DemoHandler {
    fn agent_card(&self) -> &AgentCard {
        &self.card
    }

    async fn on_send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError> {
        // Extract text from the user message
        let user_text = req
            .message
            .parts
            .first()
            .and_then(|p| p.text.as_deref())
            .unwrap_or("(no text)");

        // Create a task with an agent reply
        let task_id = uuid::Uuid::new_v4().to_string();
        let context_id = req
            .message
            .context_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let agent_reply = Message {
            message_id: uuid::Uuid::new_v4().to_string(),
            role: Role::Agent,
            parts: vec![Part {
                text: Some(format!("Echo: {user_text}")),
                raw: None,
                url: None,
                data: None,
                media_type: None,
                filename: None,
                metadata: None,
            }],
            context_id: Some(context_id.clone()),
            task_id: Some(task_id.clone()),
            reference_task_ids: None,
            metadata: None,
            extensions: None,
        };

        let task = Task {
            id: task_id.clone(),
            context_id: Some(context_id),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(agent_reply),
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
            },
            artifacts: Some(vec![Artifact {
                artifact_id: "artifact-1".to_string(),
                name: Some("echo-result".to_string()),
                description: Some("The echoed message".to_string()),
                parts: vec![Part {
                    text: Some(format!("Echo: {user_text}")),
                    raw: None,
                    url: None,
                    data: None,
                    media_type: Some("text/plain".to_string()),
                    filename: None,
                    metadata: None,
                }],
                metadata: None,
                extensions: None,
            }]),
            history: None,
            metadata: None,
        };

        // Store the task
        self.tasks.lock().unwrap().insert(task_id, task.clone());

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
        Err(A2aError::unsupported_operation("streaming not enabled"))
    }

    async fn on_get_task(&self, req: GetTaskRequest) -> Result<Task, A2aError> {
        self.tasks
            .lock()
            .unwrap()
            .get(&req.id)
            .cloned()
            .ok_or_else(|| A2aError::task_not_found(&req.id))
    }

    async fn on_list_tasks(&self, _req: ListTasksRequest) -> Result<ListTasksResponse, A2aError> {
        let tasks: Vec<Task> = self.tasks.lock().unwrap().values().cloned().collect();
        let count = tasks.len() as i32;
        Ok(ListTasksResponse {
            tasks,
            next_page_token: String::new(),
            page_size: count,
            total_size: count,
        })
    }

    async fn on_cancel_task(&self, req: CancelTaskRequest) -> Result<Task, A2aError> {
        let mut tasks = self.tasks.lock().unwrap();
        let task = tasks
            .get_mut(&req.id)
            .ok_or_else(|| A2aError::task_not_found(&req.id))?;
        task.status.state = TaskState::Canceled;
        task.status.timestamp = Some(chrono::Utc::now().to_rfc3339());
        Ok(task.clone())
    }

    async fn on_subscribe_to_task(
        &self,
        _req: SubscribeToTaskRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
    {
        Err(A2aError::unsupported_operation("streaming not enabled"))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== A2A Client/Server Example ===\n");

    // Step 1: Start the server
    println!("--- Start Server ---");

    let addr: SocketAddr = "127.0.0.1:0".parse()?;
    let handler = DemoHandler::new();
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    // Bind to port 0 so the OS picks an available port
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let actual_addr = listener.local_addr()?;
    drop(listener); // Release so the server can bind

    let server = A2aServer::new(handler, actual_addr).with_shutdown(shutdown_rx);

    println!("  Server will listen on {actual_addr}");

    let server_handle = tokio::spawn(async move {
        if let Err(e) = server.run().await {
            eprintln!("  Server error: {e}");
        }
    });

    // Give the server a moment to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    println!("  Server started\n");

    // Step 2: Create a client
    println!("--- Create Client ---");

    let base_url = Url::parse(&format!("http://{actual_addr}"))?;
    let client = A2aClient::new_jsonrpc(base_url);
    println!("  Client created (JSON-RPC transport)\n");

    // Step 3: Send a message
    println!("--- Send Message ---");

    let user_msg = Message::user_text("Hello from the A2A client!");
    let req = SendMessageRequest {
        tenant: None,
        message: user_msg,
        configuration: None,
        metadata: None,
    };

    let response = client.send_message(req).await?;

    if let Some(task) = &response.task {
        println!("  Response task ID: {}", task.id);
        println!("  Task state:       {:?}", task.status.state);
        if let Some(msg) = &task.status.message {
            let text = msg
                .parts
                .first()
                .and_then(|p| p.text.as_deref())
                .unwrap_or("(none)");
            println!("  Agent reply:      {text}");
        }
        if let Some(artifacts) = &task.artifacts {
            println!("  Artifacts:        {} item(s)", artifacts.len());
        }
    }
    println!();

    // Step 4: Get the task by ID
    println!("--- Get Task ---");

    if let Some(task) = &response.task {
        let fetched = client
            .get_task(GetTaskRequest {
                tenant: None,
                id: task.id.clone(),
                history_length: None,
            })
            .await?;
        println!(
            "  Fetched task: {} (state={:?})",
            fetched.id, fetched.status.state
        );
    }
    println!();

    // Step 5: List all tasks
    println!("--- List Tasks ---");

    let list = client.list_tasks(ListTasksRequest::default()).await?;
    println!("  Total tasks: {}", list.total_size);
    for t in &list.tasks {
        println!("    {} — {:?}", t.id, t.status.state);
    }
    println!();

    // Step 6: Cancel the task
    println!("--- Cancel Task ---");

    if let Some(task) = &response.task {
        let canceled = client
            .cancel_task(CancelTaskRequest {
                tenant: None,
                id: task.id.clone(),
                metadata: None,
            })
            .await?;
        println!(
            "  Canceled task: {} (state={:?})",
            canceled.id, canceled.status.state
        );
    }
    println!();

    // Step 7: Shutdown
    println!("--- Shutdown ---");
    drop(shutdown_tx);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server_handle).await;
    println!("  Server shut down");

    println!("\nDone.");
    Ok(())
}
