# brainwires-mcp

[![Crates.io](https://img.shields.io/crates/v/brainwires-mcp.svg)](https://crates.io/crates/brainwires-mcp)
[![Documentation](https://img.shields.io/docsrs/brainwires-mcp)](https://docs.rs/brainwires-mcp)
[![License](https://img.shields.io/crates/l/brainwires-mcp.svg)](LICENSE)

MCP client, transport, and protocol types for the Brainwires Agent Framework.

## Overview

`brainwires-mcp` provides a full MCP (Model Context Protocol) client implementation for connecting to external MCP servers. It handles the stdio transport layer, JSON-RPC 2.0 protocol, bidirectional notifications, and persistent server configuration.

**Design principles:**

- **Multi-server connections** — manage concurrent connections to multiple MCP servers from a single `McpClient`
- **rmcp-backed protocol types** — backward-compatible aliases wrapping the official `rmcp` crate types
- **Stdio transport** — newline-delimited JSON over stdin/stdout to spawned server processes
- **Bidirectional notifications** — receive server-initiated notifications (progress updates, etc.) during tool calls
- **Offline config persistence** — `McpConfigManager` saves server configurations to `~/.brainwires/mcp-config.json`

```text
  ┌───────────────────────────────────────────────────────────────────┐
  │                        brainwires-mcp                             │
  │                                                                   │
  │  ┌─── McpClient ──────────────────────────────────────────────┐  │
  │  │  connections: HashMap<String, McpConnection>                │  │
  │  │  request_id: AtomicU64                                     │  │
  │  │                                                             │  │
  │  │  connect() ──► StdioTransport::new(cmd, args)              │  │
  │  │                     │                                       │  │
  │  │                     ▼                                       │  │
  │  │              ┌─────────────┐     ┌───────────────────┐     │  │
  │  │              │  Transport  │────►│  MCP Server        │     │  │
  │  │              │  (Stdio)    │◄────│  (child process)   │     │  │
  │  │              └─────────────┘     └───────────────────┘     │  │
  │  │                     │                                       │  │
  │  │          JSON-RPC 2.0 over newline-delimited JSON           │  │
  │  │                                                             │  │
  │  │  Request ──► {"jsonrpc":"2.0","id":1,"method":"tools/list"} │  │
  │  │  Response ◄─ {"jsonrpc":"2.0","id":1,"result":{...}}        │  │
  │  │  Notif    ◄─ {"jsonrpc":"2.0","method":"notifications/..."}│  │
  │  └─────────────────────────────────────────────────────────────┘  │
  │                                                                   │
  │  ┌─── McpConfigManager ───────────────────────────────────────┐  │
  │  │  load() / save() ◄──► ~/.brainwires/mcp-config.json        │  │
  │  │  add_server() / remove_server() / get_server()             │  │
  │  └─────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-mcp-client = "0.11"
```

Connect to an MCP server and call a tool:

```rust
use brainwires_mcp::{McpClient, McpServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = McpClient::new("my-app", "1.0.0");

    // Connect to a server
    let config = McpServerConfig {
        name: "filesystem".into(),
        command: "npx".into(),
        args: vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
        env: None,
    };
    client.connect(&config).await?;

    // List available tools
    let tools = client.list_tools("filesystem").await?;
    for tool in &tools {
        println!("Tool: {}", tool.name);
    }

    // Call a tool
    let result = client.call_tool(
        "filesystem",
        "read_file",
        Some(serde_json::json!({"path": "/tmp/example.txt"})),
    ).await?;

    client.disconnect("filesystem").await?;
    Ok(())
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `native` | Yes | Enables `rmcp` protocol types, `dirs` for config paths, `tokio/process` + `tokio/io-util` for stdio transport |
| `wasm` | No | WASM-compatible build (no transport, no config persistence, forwards `brainwires-core/wasm`) |

```toml
# Default (native with full MCP client)
brainwires-mcp-client = "0.11"

# WASM target (JSON-RPC types only, no client or transport)
brainwires-mcp-client = { version = "0.11", default-features = false, features = ["wasm"] }
```

## Architecture

### McpClient

The central struct for managing MCP server connections. Thread-safe via `Arc<RwLock<>>` internals.

| Field | Type | Description |
|-------|------|-------------|
| `connections` | `Arc<RwLock<HashMap<String, McpConnection>>>` | Active server connections keyed by name |
| `request_id` | `Arc<AtomicU64>` | Monotonically incrementing request ID counter |
| `client_name` | `String` | Client name sent during initialization |
| `client_version` | `String` | Client version sent during initialization |

**Lifecycle methods:**

| Method | Description |
|--------|-------------|
| `new(name, version)` | Create a new client with the given identity |
| `connect(config)` | Spawn a server process, perform initialize handshake, store connection |
| `disconnect(server_name)` | Close transport and remove connection |
| `is_connected(server_name)` | Check if a named server is connected |
| `list_connected()` | Get names of all connected servers |

**Tool operations:**

| Method | Description |
|--------|-------------|
| `list_tools(server_name)` | List available tools (`tools/list`) |
| `call_tool(server_name, tool_name, arguments)` | Call a tool and return the result (`tools/call`) |
| `call_tool_with_notifications(server_name, tool_name, arguments, notification_tx)` | Call a tool with a channel for receiving server notifications during execution |

**Resource operations:**

| Method | Description |
|--------|-------------|
| `list_resources(server_name)` | List available resources (`resources/list`) |
| `read_resource(server_name, uri)` | Read a resource by URI (`resources/read`) |

**Prompt operations:**

| Method | Description |
|--------|-------------|
| `list_prompts(server_name)` | List available prompts (`prompts/list`) |
| `get_prompt(server_name, prompt_name, arguments)` | Get a prompt with optional arguments (`prompts/get`) |

**Server info:**

| Method | Description |
|--------|-------------|
| `get_server_info(server_name)` | Get the server's name and version |
| `get_capabilities(server_name)` | Get the server's declared capabilities |
| `cancel_request(server_name, request_id)` | Send a `$/cancelRequest` notification to cancel a pending request |

### Transport

The transport layer handles sending/receiving JSON-RPC messages over process stdio.

**`Transport` enum:**

| Variant | Description |
|---------|-------------|
| `Stdio(StdioTransport)` | Newline-delimited JSON over stdin/stdout of a child process |

**`StdioTransport` struct:**

| Field | Type | Description |
|-------|------|-------------|
| `stdin` | `Arc<Mutex<ChildStdin>>` | Write handle to the server process stdin |
| `stdout` | `Arc<Mutex<BufReader<ChildStdout>>>` | Buffered read handle from the server process stdout |
| `child` | `Arc<Mutex<Child>>` | Handle to the spawned server process |

**Methods:**

| Method | Description |
|--------|-------------|
| `new(command, args)` | Spawn a child process and capture stdin/stdout |
| `send_request(request)` | Serialize and write a JSON-RPC request followed by a newline |
| `receive_response()` | Read and parse a single JSON-RPC response line |
| `receive_message()` | Read any JSON-RPC message — discriminates responses (has `id`) from notifications (no `id`) |
| `close()` | Kill the child process |

### JSON-RPC Types

Always available (no feature gate). These implement the JSON-RPC 2.0 wire format.

**`JsonRpcRequest`:**

| Field | Type | Description |
|-------|------|-------------|
| `jsonrpc` | `String` | Always `"2.0"` |
| `id` | `Value` | Request identifier (numeric or null for notifications) |
| `method` | `String` | Method name (e.g., `"tools/list"`) |
| `params` | `Option<Value>` | Method parameters |

Constructors: `new(id, method, params) -> Result` (fallible serialization), `new_unchecked(id, method, params)` (panics on failure).

**`JsonRpcResponse`:**

| Field | Type | Description |
|-------|------|-------------|
| `jsonrpc` | `String` | Always `"2.0"` |
| `id` | `Value` | Matching request identifier |
| `result` | `Option<Value>` | Success result (mutually exclusive with `error`) |
| `error` | `Option<JsonRpcError>` | Error object (mutually exclusive with `result`) |

**`JsonRpcError`:**

| Field | Type | Description |
|-------|------|-------------|
| `code` | `i32` | Error code (e.g., `-32600` for Invalid Request) |
| `message` | `String` | Human-readable error message |
| `data` | `Option<Value>` | Additional error data |

**`JsonRpcNotification`:**

| Field | Type | Description |
|-------|------|-------------|
| `jsonrpc` | `String` | Always `"2.0"` |
| `method` | `String` | Notification method (e.g., `"notifications/progress"`) |
| `params` | `Option<Value>` | Notification parameters |

Constructors: `new(method, params) -> Result`, `new_unchecked(method, params)`.

**`JsonRpcMessage` enum:**

| Variant | Description |
|---------|-------------|
| `Response(JsonRpcResponse)` | A response to a previous request |
| `Notification(JsonRpcNotification)` | A server-initiated notification |

Helper methods: `is_response()`, `is_notification()`, `as_response()`, `as_notification()`.

### MCP Protocol Types

Native-only types (require `rmcp`). Backward-compatible aliases for the official `rmcp` crate types, plus custom types for initialization, list results, and prompt structures.

**rmcp aliases:**

| Alias | Wraps | Description |
|-------|-------|-------------|
| `McpTool` | `rmcp::model::Tool` | Tool definition (name, description, input schema) |
| `McpResource` | `rmcp::model::Resource` | Resource definition (uri, name, description) |
| `McpPrompt` | `rmcp::model::Prompt` | Prompt definition (name, description, arguments) |
| `CallToolParams` | `rmcp::model::CallToolRequestParam` | Tool call parameters (name, arguments) |
| `CallToolResult` | `rmcp::model::CallToolResult` | Tool call result (content, isError) |
| `Content` | `rmcp::model::Content` | Content item (text, image, resource) |
| `ServerCapabilities` | `rmcp::model::ServerCapabilities` | Server capability declaration |
| `ClientCapabilities` | `rmcp::model::ClientCapabilities` | Client capability declaration |

**Custom types:**

| Type | Fields | Description |
|------|--------|-------------|
| `InitializeParams` | `protocol_version`, `capabilities`, `client_info` | Initialize request params |
| `InitializeResult` | `protocol_version`, `capabilities`, `server_info` | Initialize response result |
| `ServerInfo` | `name`, `version` | Server identity |
| `ClientInfo` | `name`, `version` | Client identity |
| `ListToolsResult` | `tools: Vec<McpTool>` | Response to `tools/list` |
| `ListResourcesResult` | `resources: Vec<McpResource>` | Response to `resources/list` |
| `ListPromptsResult` | `prompts: Vec<McpPrompt>` | Response to `prompts/list` |
| `ReadResourceParams` | `uri: String` | Request params for `resources/read` |
| `ReadResourceResult` | `contents: Vec<ResourceContent>` | Response to `resources/read` |
| `ResourceContent` | enum: `Text { uri, mime_type, text }`, `Blob { uri, mime_type, blob }` | Resource content item |
| `GetPromptParams` | `name`, `arguments` | Request params for `prompts/get` |
| `GetPromptResult` | `description`, `messages: Vec<PromptMessage>` | Response to `prompts/get` |
| `PromptMessage` | `role`, `content: PromptContent` | A message in a prompt |
| `PromptContent` | enum: `Text { text }`, `Image { data, mime_type }`, `Resource { resource }` | Prompt message content |
| `PromptArgument` | `name`, `description`, `required` | Prompt argument definition |
| `ToolResultContent` | enum: `Text { text }`, `Image { data, mime_type }`, `Resource { resource }` | Tool result content item |

### Progress & Notifications

**`ProgressParams`:**

| Field | Type | Description |
|-------|------|-------------|
| `progress_token` | `String` | Token identifying which request this progress is for |
| `progress` | `f64` | Current progress value |
| `total` | `Option<f64>` | Total expected value (for percentage calculation) |
| `message` | `Option<String>` | Human-readable progress message |

**`McpNotification` enum:**

| Variant | Description |
|---------|-------------|
| `Progress(ProgressParams)` | Progress update for a long-running operation |
| `Unknown { method, params }` | Unhandled notification type |

Parse with `McpNotification::from_notification(&json_rpc_notification)`.

### Configuration

**`McpServerConfig`:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Unique server name (used as connection key) |
| `command` | `String` | Executable to spawn (e.g., `"npx"`, `"node"`, `"python"`) |
| `args` | `Vec<String>` | Command-line arguments |
| `env` | `Option<HashMap<String, String>>` | Environment variables (optional) |

**`McpConfigManager` (native only):**

| Method | Description |
|--------|-------------|
| `new()` | Create a new empty config manager |
| `load()` | Load from `~/.brainwires/mcp-config.json` (creates default if missing) |
| `save()` | Write current config to disk |
| `add_server(config)` | Add a server config (errors on duplicate name) |
| `remove_server(name)` | Remove a server config by name |
| `get_servers()` | Get all server configs |
| `get_server(name)` | Get a specific server config by name |

Config file location: `~/.brainwires/mcp-config.json`

Config file format:

```json
{
  "servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "env": null
    }
  ]
}
```

## Usage Examples

### Create a client and connect to a server

```rust
use brainwires_mcp::{McpClient, McpServerConfig};

let client = McpClient::new("my-agent", "0.2.0");

let config = McpServerConfig {
    name: "weather".into(),
    command: "node".into(),
    args: vec!["weather-server.js".into()],
    env: None,
};

client.connect(&config).await?;
assert!(client.is_connected("weather").await);

let servers = client.list_connected().await;
// ["weather"]
```

### List and call tools

```rust
// List tools
let tools = client.list_tools("weather").await?;
for tool in &tools {
    println!("{}: {}", tool.name, tool.description.as_deref().unwrap_or(""));
}

// Call a tool
let result = client.call_tool(
    "weather",
    "get_forecast",
    Some(serde_json::json!({"city": "Seattle"})),
).await?;
```

### Call a tool with notification forwarding

```rust
use tokio::sync::mpsc;
use brainwires_mcp::McpNotification;

let (tx, mut rx) = mpsc::unbounded_channel();

// Spawn a task to handle notifications
tokio::spawn(async move {
    while let Some(notif) = rx.recv().await {
        let parsed = McpNotification::from_notification(&notif);
        match parsed {
            McpNotification::Progress(p) => {
                println!("Progress: {}/{}", p.progress, p.total.unwrap_or(1.0));
            }
            McpNotification::Unknown { method, .. } => {
                println!("Notification: {}", method);
            }
        }
    }
});

let result = client.call_tool_with_notifications(
    "weather",
    "bulk_forecast",
    Some(serde_json::json!({"cities": ["Seattle", "Portland", "Vancouver"]})),
    Some(tx),
).await?;
```

### Read resources

```rust
let resources = client.list_resources("filesystem").await?;
for res in &resources {
    println!("{}: {}", res.uri, res.name);
}

let content = client.read_resource("filesystem", "file:///tmp/data.txt").await?;
for item in &content.contents {
    match item {
        brainwires_mcp::types::ResourceContent::Text { text, .. } => {
            println!("{}", text);
        }
        brainwires_mcp::types::ResourceContent::Blob { blob, .. } => {
            println!("(binary: {} bytes)", blob.len());
        }
    }
}
```

### Get prompts

```rust
let prompts = client.list_prompts("code-server").await?;

let result = client.get_prompt(
    "code-server",
    "review_code",
    Some(serde_json::json!({"language": "rust", "file": "src/main.rs"})),
).await?;

println!("Prompt: {}", result.description);
for msg in &result.messages {
    println!("[{}] {:?}", msg.role, msg.content);
}
```

### Manage server config with McpConfigManager

```rust
use brainwires_mcp::{McpConfigManager, McpServerConfig};

// Load or create config
let mut manager = McpConfigManager::load()?;

// Add a server
manager.add_server(McpServerConfig {
    name: "filesystem".into(),
    command: "npx".into(),
    args: vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
    env: None,
})?;

// Query servers
let servers = manager.get_servers();
let fs_server = manager.get_server("filesystem");

// Remove a server
manager.remove_server("filesystem")?;
```

### Build custom JSON-RPC requests

```rust
use brainwires_mcp::{JsonRpcRequest, JsonRpcNotification};

// Create a request
let request = JsonRpcRequest::new(
    42,
    "custom/method".to_string(),
    Some(serde_json::json!({"key": "value"})),
)?;

// Create a notification (no id)
let notification = JsonRpcNotification::new(
    "custom/event",
    Some(serde_json::json!({"status": "ready"})),
)?;
```

## Integration

Use via the `brainwires` facade crate with the `mcp` feature, or depend on `brainwires-mcp` directly:

```toml
# Via facade
[dependencies]
brainwires = { version = "0.11", features = ["mcp"] }

# Direct
[dependencies]
brainwires-mcp-client = "0.11"
```

The `prelude` module re-exports the most commonly used types:

```rust
use brainwires_mcp::prelude::*;
```

## License

Licensed under either MIT or Apache-2.0 at your option. See [LICENSE-MIT](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-APACHE).
