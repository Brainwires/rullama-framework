# Architecture

The framework follows a layered, modular design. Every package is independently
installable and has a clear role in the dependency graph.

## Design Philosophy

1. **Zero-dep core** -- `@rullama/core` has no external dependencies. It
   defines all foundational types (messages, tools, errors, lifecycle hooks) so
   every other package can depend on it without pulling in heavy libraries.
2. **Layered architecture** -- Higher-level packages compose lower-level ones.
   You only install what you need.
3. **Interface-driven** -- Core abstractions (`Provider`, `StorageBackend`,
   `ToolExecutor`) are TypeScript interfaces. Swap implementations without
   changing consuming code.
4. **Deno-native** -- Built for Deno with JSR publishing, no Node.js polyfills.

## Package Dependency Graph

```
            @rullama/core
           /    |    |    \
          /     |    |     \
 providers  storage  mcp  permissions
     |        |       |
     +--------+-------+
     |
   agents -----> tool-system
     |               |
agent-network    cognition
     |
    a2a
```

`core` is the root. `agents` pulls in `providers`, `storage`, `mcp`, and
`tool-system`. The `agent-network` and `a2a` packages are leaf-level consumers.

## Package Overview

| Package                   | Install                            | Description                                                                 |
| ------------------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| `@rullama/core`        | `deno add @rullama/core`        | Messages, tools, errors, lifecycle hooks, output parsers, working set       |
| `@rullama/provider`      | `deno add @rullama/provider`      | AI chat providers (Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex)      |
| `@rullama/agent`         | `deno add @rullama/agent`         | Agent coordination: comms, locks, task manager, contract-net, saga, market  |
| `@rullama/inference`     | `deno add @rullama/inference`     | Agent runtime, task agents (TaskAgent, ChatAgent, Judge, Planner, Validator)|
| `@rullama/mdap`          | `deno add @rullama/mdap`          | MDAP / MAKER voting                                                          |
| `@rullama/tool-runtime`  | `deno add @rullama/tool-runtime`  | Tool registry, executor, sanitization, smart routing, OpenAPI/OAuth         |
| `@rullama/tool-builtins` | `deno add @rullama/tool-builtins` | Built-in tools (bash, file, git, web, search, calendar, sessions)           |
| `@rullama/storage`       | `deno add @rullama/storage`       | Backend-agnostic storage, domain stores, tiered memory                      |
| `@rullama/knowledge`     | `deno add @rullama/knowledge`     | Prompting techniques, knowledge graph, RAG interfaces, code analysis        |
| `@rullama/mcp-client`    | `deno add @rullama/mcp-client`    | Model Context Protocol client (stdio transport)                             |
| `@rullama/mcp-server`    | `deno add @rullama/mcp-server`    | MCP server framework (its own package since v0.11.0)                        |
| `@rullama/network`       | `deno add @rullama/network`       | Middleware, routing, discovery, remote bridge                               |
| `@rullama/a2a`           | `deno add @rullama/a2a`           | Google A2A protocol (JSON-RPC + REST, SSE streaming)                        |
| `@rullama/permission`    | `deno add @rullama/permission`    | Capability profiles, policy engine, audit logging, trust, anomaly detection |
| `@rullama/skills`        | `deno add @rullama/skills`        | SKILL.md parsing, skill registry, routing, execution                        |

## Key Types from Core

These types appear throughout the framework:

| Type                     | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `Message`                | Chat message (user, assistant, tool result)         |
| `ChatOptions`            | Model parameters (max_tokens, temperature, etc.)    |
| `ChatResponse`           | Provider response (message + usage + finish reason) |
| `Tool`                   | Tool definition (name, description, input schema)   |
| `ToolUse` / `ToolResult` | Tool call request and response                      |
| `FrameworkError`         | Typed error hierarchy                               |
| `LifecycleHook`          | Event interception for the framework lifecycle      |
| `Provider`               | Interface all chat providers implement              |

## Runtime boundary — what stays Rust-only

The Deno port is deliberately a subset. A handful of crates in the Rust
framework are Rust-only on purpose, and Deno consumers should drive the Rust
binary for those concerns instead of trying to approximate them:

- **`rullama-hardware`** — GPIO, USB, BLE, CPAL audio, Zigbee, Z-Wave,
  Matter. Needs OS kernel access; not reachable from Deno without FFI.
- **`rullama-sandbox`, `rullama-sandbox-proxy`** — Bollard Docker
  orchestration and Hyper-based egress proxy. Run the Rust sidecar.
- **`local_llm` provider** — llama-cpp FFI. Use `OllamaChatProvider` for local
  inference from Deno.
- **`interpreters` / `orchestrator` tools** — Rhai, Boa, RustPython embedded
  runtimes.
- **`sandbox_executor` / `code_exec`** — depend on the Rust sandbox crate.
- **`browser` tool** — pairs with the Rust Thalora headless browser.
- **LanceDB / ONNX / tantivy RAG** — native indexing and embedding stays in
  Rust. The Deno `@rullama/knowledge` package keeps its client role and talks
  to a Rust RAG service over the existing `RagClient` interface.
- **Burn-based local training** — Deno ships `@rullama/finetune` with cloud
  backends only (OpenAI, Together, Fireworks).

Communication across the boundary goes through `@rullama/network` /
`@rullama/mcp-server` (MCP server, IPC transport) or `@rullama/a2a`
(Google A2A protocol).

For per-file detail, see [parity.md](./parity.md) — it links to each
`SKIPPED.md` under the corresponding package.

## Further Reading

- [Getting Started](./getting-started.md) for a hands-on quickstart
- [Extensibility](./extensibility.md) for how to implement custom providers,
  storage, and tools
- [Parity](./parity.md) — crate-by-crate diff against the Rust workspace
