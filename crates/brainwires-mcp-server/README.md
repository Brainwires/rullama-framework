# brainwires-mcp-server

[![Crates.io](https://img.shields.io/crates/v/brainwires-mcp-server.svg)](https://crates.io/crates/brainwires-mcp-server)
[![Documentation](https://img.shields.io/docsrs/brainwires-mcp-server)](https://docs.rs/brainwires-mcp-server)
[![License](https://img.shields.io/crates/l/brainwires-mcp-server.svg)](LICENSE)

MCP server framework with composable middleware for the Brainwires Agent Framework.

## Overview

`brainwires-mcp-server` provides everything needed to build an MCP-compliant tool server:

- **`McpServer`** — Async event loop that reads JSON-RPC requests, runs the middleware chain, and dispatches to your handler
- **`McpHandler`** — Trait defining your server's identity, capabilities, and tool dispatch
- **`McpToolRegistry`** — Declarative tool registration with automatic dispatch by tool name
- **`MiddlewareChain`** — Composable onion-model middleware pipeline
- **Built-in middlewares** — Auth, logging, rate limiting, and tool filtering included

This crate was extracted from `brainwires-network` so that consumers who only need to build MCP servers don't have to pull in the full networking stack.

```text
  JSON-RPC request
       │
       ▼
  McpServer::run()
       │
       ▼
  ┌─────────────────────────────────────────┐
  │           MiddlewareChain               │
  │  AuthMiddleware → LoggingMiddleware →   │
  │  RateLimitMiddleware → ToolFilter       │
  └──────────────────────┬──────────────────┘
                         │
                         ▼
                   McpHandler::call_tool()
                   (or list_tools / initialize)
```

## Quick Start

```toml
[dependencies]
brainwires-mcp-server = "0.11"
```

Minimal server:

```rust
use brainwires_mcp_server::{
    McpServer, McpHandler, McpToolDef, McpToolRegistry,
    StdioServerTransport, LoggingMiddleware, RequestContext,
};
use brainwires_mcp::types::{ServerInfo, ServerCapabilities, CallToolResult};
use async_trait::async_trait;
use serde_json::Value;

struct MyHandler;

#[async_trait]
impl McpHandler for MyHandler {
    fn server_info(&self) -> ServerInfo {
        ServerInfo { name: "my-server".into(), version: "0.1.0".into() }
    }

    fn capabilities(&self) -> ServerCapabilities {
        ServerCapabilities::default()
    }

    fn list_tools(&self) -> Vec<McpToolDef> {
        vec![
            McpToolDef::new("greet", "Say hello to someone")
                .with_string_param("name", "Name to greet", true),
        ]
    }

    async fn call_tool(
        &self,
        name: &str,
        args: Value,
        _ctx: &RequestContext,
    ) -> anyhow::Result<CallToolResult> {
        match name {
            "greet" => {
                let name = args["name"].as_str().unwrap_or("world");
                Ok(CallToolResult::text(format!("Hello, {}!", name)))
            }
            _ => anyhow::bail!("Unknown tool: {}", name),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server = McpServer::new(MyHandler)
        .with_transport(StdioServerTransport::new())
        .with_middleware(LoggingMiddleware::new());

    server.run().await
}
```

## With Auth and Rate Limiting

```rust
use brainwires_mcp_server::{
    McpServer, StdioServerTransport,
    AuthMiddleware, RateLimitMiddleware, ToolFilterMiddleware,
};

let server = McpServer::new(MyHandler)
    .with_transport(StdioServerTransport::new())
    .with_middleware(AuthMiddleware::bearer("my-secret-token"))
    .with_middleware(RateLimitMiddleware::new(100))        // 100 req/min
    .with_middleware(ToolFilterMiddleware::deny(["bash"])); // block bash tool

server.run().await?;
```

## Using McpToolRegistry

`McpToolRegistry` handles dispatch automatically — register handlers and call `dispatch()`:

```rust
use brainwires_mcp_server::{McpToolRegistry, McpToolDef, ToolHandler};
use brainwires_mcp::types::CallToolResult;
use serde_json::Value;

let mut registry = McpToolRegistry::new();

registry.register(
    McpToolDef::new("echo", "Echo back the input")
        .with_string_param("message", "Message to echo", true),
    |args: Value| async move {
        let msg = args["message"].as_str().unwrap_or("");
        Ok(CallToolResult::text(msg.to_string()))
    },
);

// In your McpHandler::call_tool():
// registry.dispatch(tool_name, args).await
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `McpServer<H>` | Server lifecycle — wires handler, middleware chain, and transport |
| `McpHandler` | Trait: `server_info()`, `capabilities()`, `list_tools()`, `call_tool()` |
| `McpToolRegistry` | Declarative registry with `register()` and `dispatch()` |
| `McpToolDef` | Tool definition (name, description, input schema) |
| `ToolHandler` | Boxed async fn `(Value) -> Result<CallToolResult>` |
| `ServerTransport` | Trait for request/response I/O |
| `StdioServerTransport` | Stdio transport (reads from stdin, writes to stdout) |
| `MiddlewareChain` | Ordered middleware list; processed in insertion order |
| `Middleware` | Trait: `handle(request, next) -> MiddlewareResult` |
| `MiddlewareResult` | Continue or short-circuit with a response |
| `RequestContext` | Per-request info: client ID, remote address, auth token |
| `ClientInfo` | Client identity attached to the request context |

### Middleware

| Type | Description |
|------|-------------|
| `AuthMiddleware` | Bearer token validation; rejects with JSON-RPC `-32001` on failure |
| `LoggingMiddleware` | Structured `tracing` spans for every request |
| `RateLimitMiddleware` | Token-bucket rate limiter; rejects with `-32029` when over budget |
| `ToolFilterMiddleware` | Allow-list or deny-list; rejects denied tools with `-32004` |

## Integration

Use via the `brainwires` facade crate:

```toml
[dependencies]
brainwires = { version = "0.11", features = ["mcp-server-framework"] }
```

Or use standalone — `brainwires-mcp-server` depends only on `brainwires-mcp`.

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.
