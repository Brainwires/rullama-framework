# brainwires-a2a

Full Rust implementation of the [Agent-to-Agent (A2A)](https://github.com/a2a-protocol/a2a) protocol — the open standard (Google / Linux Foundation) for interoperable agent communication.

Covers all three protocol bindings: **JSON-RPC 2.0**, **HTTP/REST**, and **gRPC**.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `client` | yes (via `native`) | HTTP client for JSON-RPC and REST (reqwest) |
| `server` | yes (via `native`) | HTTP server for JSON-RPC and REST (hyper) |
| `native` | **yes** | Both `client` and `server` |
| `grpc` | no | Proto types (prost + tonic) |
| `grpc-client` | no | gRPC client transport |
| `grpc-server` | no | gRPC server service |
| `full` | no | Everything |

Types are always available with no features enabled — useful if you only need the data model.

## Quick start

### Client

```rust
use brainwires_a2a::{A2aClient, Message, SendMessageRequest};
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Discover the agent
    let card = A2aClient::discover("https://agent.example.com").await?;
    println!("Connected to: {}", card.name);

    // Create a client (JSON-RPC is the primary binding)
    let client = A2aClient::new_jsonrpc(Url::parse("https://agent.example.com")?);

    // Send a message
    let response = client.send_message(SendMessageRequest {
        tenant: None,
        message: Message::user_text("Hello, agent!"),
        configuration: None,
        metadata: None,
    }).await?;

    println!("{response:?}");
    Ok(())
}
```

### Server

Implement the `A2aHandler` trait once — the server routes JSON-RPC, REST, and gRPC requests to it automatically.

```rust
use std::net::SocketAddr;
use std::pin::Pin;

use async_trait::async_trait;
use brainwires_a2a::*;
use futures::Stream;

struct MyAgent { card: AgentCard }

#[async_trait]
impl A2aHandler for MyAgent {
    fn agent_card(&self) -> &AgentCard { &self.card }

    async fn on_send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError> {
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            context_id: req.message.context_id.clone(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(Message::agent_text("Hello back!")),
                timestamp: None,
            },
            artifacts: None,
            history: Some(vec![req.message]),
            metadata: None,
            kind: "task".into(),
        };
        Ok(SendMessageResponse::Task(task))
    }

    async fn on_send_streaming_message(
        &self,
        _req: SendMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, A2aError>> + Send>>, A2aError> {
        Err(A2aError::unsupported_operation("streaming"))
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, A2aError>> + Send>>, A2aError> {
        Err(A2aError::task_not_found(&req.id))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let card = AgentCard {
        name: "My Agent".into(),
        description: "Example A2A agent".into(),
        version: "1.0.0".into(),
        supported_interfaces: None,
        capabilities: AgentCapabilities {
            streaming: Some(false),
            push_notifications: Some(false),
            extended_agent_card: None,
            extensions: None,
        },
        skills: vec![AgentSkill {
            id: "chat".into(),
            name: "Chat".into(),
            description: "Basic chat".into(),
            tags: vec!["chat".into()],
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
    };

    let addr: SocketAddr = "0.0.0.0:8080".parse()?;
    let server = A2aServer::new(MyAgent { card }, addr);
    server.run().await?;
    Ok(())
}
```

## Protocol bindings

### JSON-RPC 2.0 (primary)

The primary binding used by the official Python SDK. Requests are `POST /` with a JSON-RPC body. Streaming methods (`message/stream`, `tasks/resubscribe`) return `text/event-stream` (SSE) where each `data:` line is a JSON-RPC response.

**Methods:**

| Method | Description |
|--------|-------------|
| `message/send` | Send a message, get Task or Message back |
| `message/stream` | Send a message, stream SSE events |
| `tasks/get` | Get a task by ID |
| `tasks/list` | List tasks with filters |
| `tasks/cancel` | Cancel a running task |
| `tasks/resubscribe` | Re-subscribe to task updates (SSE) |
| `tasks/pushNotificationConfig/set` | Create/update push config |
| `tasks/pushNotificationConfig/get` | Get push config |
| `tasks/pushNotificationConfig/list` | List push configs |
| `tasks/pushNotificationConfig/delete` | Delete push config |
| `agent/authenticatedExtendedCard` | Get extended agent card |

### HTTP/REST

RESTful endpoints derived from `google.api.http` annotations in the proto spec. All endpoints also accept an optional `/{tenant}/` prefix.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/.well-known/agent-card.json` | Agent card discovery |
| POST | `/message:send` | Send message |
| POST | `/message:stream` | Stream message (SSE) |
| GET | `/tasks/{id}` | Get task |
| GET | `/tasks` | List tasks |
| POST | `/tasks/{id}:cancel` | Cancel task |
| GET | `/tasks/{id}:subscribe` | Subscribe to updates (SSE) |
| POST | `/tasks/{task_id}/pushNotificationConfigs` | Create push config |
| GET | `/tasks/{task_id}/pushNotificationConfigs/{id}` | Get push config |
| GET | `/tasks/{task_id}/pushNotificationConfigs` | List push configs |
| DELETE | `/tasks/{task_id}/pushNotificationConfigs/{id}` | Delete push config |
| GET | `/extendedAgentCard` | Get extended agent card |

### gRPC

Generated from the official `a2a.proto` (`lf.a2a.v1.A2AService`) via `tonic-build`. Enable with the `grpc`, `grpc-client`, or `grpc-server` features.

The gRPC server runs on a separate port and can be enabled alongside HTTP:

```rust
let server = A2aServer::new(handler, http_addr)
    .with_grpc(grpc_addr);
server.run().await?;
```

#### Build requirements

`build.rs` automatically runs `tonic_build` against `proto/a2a.proto` when any
`grpc*` feature is enabled — no manual codegen step. Requirements on the build host:

- `protoc` (protobuf compiler) on `$PATH`. Install via:
  - Debian/Ubuntu: `apt-get install -y protobuf-compiler`
  - macOS: `brew install protobuf`
  - Alpine: `apk add protoc protobuf-dev`
- The proto import root is `proto/` (anchored to the crate root). Google's
  `api/http.proto` and `api/annotations.proto` are vendored under `proto/google/`
  so no well-known-types package is required.

Default-feature builds (`native` only) skip the gRPC codegen entirely and have
no `protoc` requirement.

## Transport selection (client)

```rust
// JSON-RPC (default, compatible with Python SDK)
let client = A2aClient::new_jsonrpc(url);

// REST
let client = A2aClient::new_rest(url);

// gRPC (requires grpc-client feature)
let client = A2aClient::new_grpc("http://localhost:50051").await?;
```

All `A2aClient` methods work identically regardless of transport.

## Error codes

Spec-defined JSON-RPC error codes are available as constants:

| Code | Constant | Meaning |
|------|----------|---------|
| -32700 | `JSON_PARSE_ERROR` | Invalid JSON payload |
| -32600 | `INVALID_REQUEST` | Request validation error |
| -32601 | `METHOD_NOT_FOUND` | Method not found |
| -32602 | `INVALID_PARAMS` | Invalid parameters |
| -32603 | `INTERNAL_ERROR` | Internal error |
| -32001 | `TASK_NOT_FOUND` | Task not found |
| -32002 | `TASK_NOT_CANCELABLE` | Task cannot be canceled |
| -32003 | `PUSH_NOT_SUPPORTED` | Push notifications not supported |
| -32004 | `UNSUPPORTED_OPERATION` | Operation not supported |
| -32005 | `CONTENT_TYPE_NOT_SUPPORTED` | Incompatible content types |
| -32006 | `INVALID_AGENT_RESPONSE` | Invalid agent response |
| -32007 | `EXTENDED_CARD_NOT_CONFIGURED` | Extended card not configured |

## Streaming Event Correlation

`TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` carry optional trace correlation fields for cross-system diagnostics:

| Field | JSON key | Description |
|-------|----------|-------------|
| `trace_id: Option<Uuid>` | `traceId` | Matches the `trace_id` generated by the originating `TaskAgent` and stamped in `AuditEvent.metadata["trace_id"]` |
| `sequence: Option<u64>` | `sequence` | Monotonically increasing counter within the trace — use to reorder out-of-order events and detect gaps |

Both fields are `skip_serializing_if = None` so existing clients and serialized payloads are fully wire-compatible.

## Cargo.toml

```toml
# Types only (no networking)
brainwires-a2a = { version = "0.11", default-features = false }

# Client + server (JSON-RPC + REST)
brainwires-a2a = "0.11"

# Everything including gRPC
brainwires-a2a = { version = "0.11", features = ["full"] }
```

Or via the `brainwires` facade crate:

```toml
brainwires = { version = "0.11", features = ["a2a"] }
```

## License

MIT OR Apache-2.0
