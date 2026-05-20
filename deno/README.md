# Brainwires Framework — Deno/TypeScript Port

A modular, Deno-native TypeScript port of the [Brainwires Agent Framework](https://github.com/Brainwires/brainwires-framework). Build autonomous AI agents with tool use, multi-provider support, inter-agent communication, and fine-grained permissions — all running on Deno.

## Packages

| Package | JSR | Description |
|---------|-----|-------------|
| `@brainwires/core` | `deno add @brainwires/core` | Foundation types, messages, tools, errors, lifecycle hooks |
| `@brainwires/providers` | `deno add @brainwires/providers` | AI chat providers (Anthropic, OpenAI, Google, Ollama, etc.) |
| `@brainwires/agents` | `deno add @brainwires/agents` | Agent runtime, task agents, coordination patterns |
| `@brainwires/mcp` | `deno add @brainwires/mcp` | Model Context Protocol client |
| `@brainwires/a2a` | `deno add @brainwires/a2a` | Agent-to-Agent protocol (Google A2A) |
| `@brainwires/storage` | `deno add @brainwires/storage` | Backend-agnostic storage with domain stores |
| `@brainwires/permissions` | `deno add @brainwires/permissions` | Capability profiles, policy engine, audit, trust |
| `@brainwires/tools` | `deno add @brainwires/tools` | Tool registry, built-in tools (bash, files, git, web, search) |
| `@brainwires/knowledge` | `deno add @brainwires/knowledge` | Prompting techniques, knowledge graph, RAG interfaces |
| `@brainwires/network` | `deno add @brainwires/network` | MCP server framework, middleware, routing, discovery |
| `@brainwires/session` | `deno add @brainwires/session` | Pluggable session persistence (in-memory, Deno KV) |
| `@brainwires/resilience` | `deno add @brainwires/resilience` | Retry / budget / circuit-breaker / cache provider decorators |
| `@brainwires/telemetry` | `deno add @brainwires/telemetry` | Analytics events, sinks, Prometheus metrics, billing hooks |
| `@brainwires/reasoning` | `deno add @brainwires/reasoning` | Plan parser, complexity/router/validator/retrieval scorers |
| `@brainwires/training` | `deno add @brainwires/training` | Cloud fine-tuning (OpenAI, Together, Fireworks) |

The core 10 packages are at **0.5.0**; the five new packages start at **0.1.0**. All are published to JSR under the `@brainwires` scope.

## Documentation & Examples

- **[Documentation](./docs/)** — Guides covering architecture, each subsystem, and extensibility
- **[Examples](./examples/)** — 43 runnable TypeScript examples ported from the Rust crates

## Package Dependency Diagram

```
                     @brainwires/core
                    /    |    |    \
                   /     |    |     \
          providers  storage  mcp  permissions
              |        |       |
              +--------+-------+
              |
            agents -----> tools
              |             |
           network      knowledge
              |
             a2a
```

`core` has zero external dependencies. Every other package depends on `core`. The `agents` package pulls in `providers`, `storage`, `mcp`, `tools`, and skills (absorbed). The `network` and `a2a` packages are leaf-level consumers.

## Quick Start

### 1. Create a provider and send a message

```ts
import { Message, ChatOptions } from "@brainwires/core";
import { AnthropicChatProvider } from "@brainwires/providers";

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
  "anthropic",
);

const messages = [Message.user("What is the Deno runtime?")];
const options = new ChatOptions({ max_tokens: 1024 });

const response = await provider.chat(messages, undefined, options);
console.log(response.content);
```

### 2. Register tools and run an agent

```ts
import { ChatOptions, Message } from "@brainwires/core";
import { AnthropicChatProvider } from "@brainwires/providers";
import { ToolRegistry, BashTool, FileOpsTool } from "@brainwires/tools";
import { TaskAgent, AgentContext, spawnTaskAgent } from "@brainwires/agents";

// Set up tools
const registry = new ToolRegistry();
registry.registerTools(BashTool.getTools());
registry.registerTools(FileOpsTool.getTools());

// Create provider
const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
  "anthropic",
);

// Build agent context and run
const context = new AgentContext({ tools: registry.allTools() });
const result = await spawnTaskAgent({
  agentId: "demo-agent",
  provider,
  context,
  systemPrompt: "You are a helpful coding assistant.",
  taskDescription: "List the files in the current directory.",
});

console.log(`Success: ${result.success}, Output: ${result.output}`);
```

### 3. Connect to an MCP server

```ts
import { McpClient } from "@brainwires/mcp";

const client = McpClient.createDefault();
await client.connect("my-server", {
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
});

const tools = await client.listTools("my-server");
console.log("Available tools:", tools.map((t) => t.name));
```

## What's Ported vs What's Not

Per-file detail lives in [docs/parity.md](./docs/parity.md). Summary:

| Rust Crate | Deno Package | Status |
|------------|-------------|--------|
| `brainwires-core` | `@brainwires/core` | Faithful (+ event.ts, workflow_state.ts, output parsers) |
| `brainwires-providers` | `@brainwires/providers` | Chat providers + Relay + 7 audio HTTP clients. `local_llm` (llama-cpp) stays Rust-only. |
| `brainwires-agent` | `@brainwires/agents` | Runtime, task agent, coordination, MDAP, skills, seal, eval, system_prompts, roles |
| `brainwires-mcp` | `@brainwires/mcp` | Client + stdio transport + JSON-RPC |
| `brainwires-mcp-server` | folded into `@brainwires/network` | Server framework + middleware pipeline |
| `brainwires-a2a` | `@brainwires/a2a` | JSON-RPC + REST (no gRPC by design) |
| `brainwires-storage` | `@brainwires/storage` | In-memory + Postgres/MySQL/Qdrant/SurrealDB/Pinecone/Weaviate/Milvus + domain stores |
| `brainwires-permissions` | `@brainwires/permissions` | Capabilities, policy, audit, trust, anomaly |
| `brainwires-tools` | `@brainwires/tools` | Bash, file_ops, git, web, search, validation, openapi, oauth, calendar, sessions, tool_search, tool_embedding, semantic_search. `interpreters`/`orchestrator`/`sandbox_executor`/`code_exec`/`browser` stay Rust-only (see packages/tools/tools/SKIPPED.md). |
| `brainwires-knowledge` | `@brainwires/knowledge` | Prompting + code analysis implemented; RAG / BKS / PKS stay as client interfaces. |
| `brainwires-network` | `@brainwires/network` | MCP server, middleware, routing, discovery, remote bridge |
| `brainwires-session` | `@brainwires/session` | InMemory + DenoKv backends (replaces the Rust SQLite backend) |
| `brainwires-resilience` | `@brainwires/resilience` | Retry / budget / circuit-breaker / memory-cache decorators |
| `brainwires-telemetry` | `@brainwires/telemetry` | Analytics events, sinks, Prometheus metrics, billing hook (SQLite sink + tracing-crate layer intentionally omitted) |
| `brainwires-reasoning` | `@brainwires/reasoning` | Tier-1 slice: plan parser, complexity, router, validator, retrieval. `strategies`, `strategy_selector`, `summarizer`, `relevance_scorer`, `entity_enhancer` deferred. |
| `brainwires-training` | `@brainwires/training` | Cloud-only: OpenAI, Together, Fireworks + JobPoller + TrainingManager. Bedrock/Vertex (vendor SDKs) and the local Burn-based path intentionally stay Rust-only. |
| `brainwires-hardware` | — | Runtime boundary — Rust-only (GPIO/USB/BLE/CPAL audio/Zigbee/Z-Wave/Matter). |
| `brainwires-sandbox` · `-sandbox-proxy` | — | Infra sidecars (Bollard Docker, Hyper HTTP proxy) — drive from the Rust binary. |

## Installation

Install any package with `deno add`:

```sh
deno add @brainwires/core
deno add @brainwires/providers
deno add @brainwires/agents
# ... etc.
```

Or import directly from JSR in your source:

```ts
import { Message, ChatOptions } from "jsr:@brainwires/core@0.5.0";
```

## Rust Crate Documentation

For full API documentation of the underlying Rust crates, see the [crates README](../crates/README.md) and the per-crate docs on [docs.rs](https://docs.rs).

## License

Same license as the parent Brainwires Framework repository.
