# Architecture

The framework follows a layered, modular design. Every package is independently
installable and has a clear role in the dependency graph.

## Design Philosophy

1. **Zero-dep core** -- `@rullama/core` has no external dependencies. It defines
   all foundational types (messages, tools, errors, lifecycle hooks) so every
   other package can depend on it without pulling in heavy libraries.
2. **Layered architecture** -- Higher-level packages compose lower-level ones.
   You only install what you need.
3. **Interface-driven** -- Core abstractions (`Provider`, `StorageBackend`,
   `ToolExecutor`) are TypeScript interfaces. Swap implementations without
   changing consuming code.
4. **Deno-native** -- Built for Deno with JSR publishing, no Node.js polyfills.

## Package Dependency Graph

Edges are the actual `@rullama/*` imports in each package's non-test sources
(v0.12.0). Arrows point from a package to what it depends on.

```
                               core
      ┌──────────┬──────────┬────┼──────────┬────────────┬──────────┐
  provider  provider-speech  agent  storage  telemetry  reasoning  session
      │                        │      │         │
      │                        │    stores   permission ◄──────────────┐
      │                        │      │         │                      │
      │                        │    memory      │                      │
      │                        │                │                      │
      │                        │         tool-runtime ◄──── seal ──────┘
      │                        │           │       │
      │                        │           │  tool-builtins ──► rag
      │                        │           │
      └──────────► inference ◄─┴───────────┘

  mcp-client ◄── mcp-server ◄── network (network also imports core)

  call-policy ──► core

  Leaf packages (no @rullama/* imports): a2a, eval, finetune, knowledge,
  mcp-client, mdap, prompting, rag, skills
```

Per-package edges:

| Package                    | Depends on                                                                      |
| -------------------------- | ------------------------------------------------------------------------------- |
| `@rullama/agent`           | core                                                                            |
| `@rullama/call-policy`     | core                                                                            |
| `@rullama/inference`       | core, agent, tool-runtime                                                       |
| `@rullama/mcp-server`      | mcp-client                                                                      |
| `@rullama/memory`          | stores                                                                          |
| `@rullama/network`         | core, mcp-server                                                                |
| `@rullama/permission`      | telemetry                                                                       |
| `@rullama/provider`        | core                                                                            |
| `@rullama/provider-speech` | core                                                                            |
| `@rullama/reasoning`       | core                                                                            |
| `@rullama/seal`            | core, permission, tool-runtime                                                  |
| `@rullama/session`         | core                                                                            |
| `@rullama/storage`         | core                                                                            |
| `@rullama/stores`          | core, storage                                                                   |
| `@rullama/telemetry`       | core                                                                            |
| `@rullama/tool-builtins`   | core, rag, tool-runtime                                                         |
| `@rullama/tool-runtime`    | core, permission                                                                |
| all others                 | none (a2a, eval, finetune, knowledge, mcp-client, mdap, prompting, rag, skills) |

Two edges are new in v0.12.0: `tool-runtime → permission` (the enforcing
executor consults the policy engine and capability profiles) and
`tool-builtins → tool-runtime` (the built-in executor wraps itself in that
enforcement).

## Package Overview

| Package                    | Install                             | Description                                                                                                                                                             |
| -------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@rullama/core`            | `deno add @rullama/core`            | Messages, tools, errors, lifecycle hooks, output parsers, working set, confidence, paths, file context                                                                  |
| `@rullama/a2a`             | `deno add @rullama/a2a`             | Google A2A protocol v1.0 client + types (JSON-RPC + REST, SSE streaming; no server transport, no gRPC)                                                                  |
| `@rullama/agent`           | `deno add @rullama/agent`           | Coordination primitives: `CommunicationHub`, `FileLockManager`, `TaskManager`/`TaskQueue`, `ExecutionGraph`, contract-net, saga, optimistic, market, wait-queue         |
| `@rullama/call-policy`     | `deno add @rullama/call-policy`     | Provider decorators: retry, budget, circuit breaker, response cache                                                                                                     |
| `@rullama/eval`            | `deno add @rullama/eval`            | Evaluation harness: trial runner with Wilson CI, suites, recorder, adversarial/regression/stability cases, ranking metrics                                              |
| `@rullama/finetune`        | `deno add @rullama/finetune`        | Cloud fine-tuning (OpenAI, Together, Fireworks); local training stays Rust-side                                                                                         |
| `@rullama/inference`       | `deno add @rullama/inference`       | LLM workhorses: `runAgentLoop`, `TaskAgent`, `AgentContext` (enforcing by default), `AgentPool`, `ValidatorAgent`, `PlanExecutorAgent`, judge/planner prompts + parsers |
| `@rullama/knowledge`       | `deno add @rullama/knowledge`       | Knowledge-graph types (`Thought`, `Entity`, `Relationship`) and the `BrainClient` interface; no implementation ships                                                    |
| `@rullama/mcp-client`      | `deno add @rullama/mcp-client`      | Model Context Protocol client (stdio transport) + on-disk server config                                                                                                 |
| `@rullama/mcp-server`      | `deno add @rullama/mcp-server`      | MCP tool server framework: `McpServer`, `McpToolRegistry`, middleware chain, stdio transport                                                                            |
| `@rullama/mdap`            | `deno add @rullama/mdap`            | MDAP / MAKER voting: first-to-ahead-by-k consensus, red-flag validation, decomposition, scaling laws                                                                    |
| `@rullama/memory`          | `deno add @rullama/memory`          | Tiered memory (hot/warm/cold) with retention and multi-factor scoring                                                                                                   |
| `@rullama/network`         | `deno add @rullama/network`         | Agent identity, message envelopes, routing, peer table, discovery, remote bridge; an unauthenticated transport layer                                                    |
| `@rullama/permission`      | `deno add @rullama/permission`      | Capability profiles, policy engine, audit logging, trust management, approval types                                                                                     |
| `@rullama/prompting`       | `deno add @rullama/prompting`       | 15 adaptive prompting techniques, task clustering, prompt generation, temperature optimization                                                                          |
| `@rullama/provider`        | `deno add @rullama/provider`        | Chat providers (Anthropic, OpenAI Chat + Responses, Google, Bedrock, Vertex, Ollama), factory, stream parsers, model listing                                            |
| `@rullama/provider-speech` | `deno add @rullama/provider-speech` | TTS/STT/ASR HTTP clients (Azure, Cartesia, Deepgram, ElevenLabs, Fish, Google TTS, Murf)                                                                                |
| `@rullama/rag`             | `deno add @rullama/rag`             | `RagClient` interface + request/response types; code analysis (`RepoMap`, call graphs) under `@rullama/rag/code-analysis`                                               |
| `@rullama/reasoning`       | `deno add @rullama/reasoning`       | Tier-1 local scorers (complexity, router, validator, retrieval) and the plan parser                                                                                     |
| `@rullama/seal`            | `deno add @rullama/seal`            | SEAL self-evolving learning loop (coreference, query-core extraction, pattern learning)                                                                                 |
| `@rullama/session`         | `deno add @rullama/session`         | Pluggable session persistence (in-memory, Deno KV)                                                                                                                      |
| `@rullama/skills`          | `deno add @rullama/skills`          | SKILL.md parsing, `SkillRegistry`, `SkillRouter`, `SkillExecutor`                                                                                                       |
| `@rullama/storage`         | `deno add @rullama/storage`         | `StorageBackend` / `VectorDatabase` interfaces, `InMemoryStorageBackend`, Postgres/MySQL/SurrealDB/Qdrant/Pinecone/Weaviate/Milvus adapters, cached embeddings          |
| `@rullama/stores`          | `deno add @rullama/stores`          | Domain stores over `StorageBackend`: message, conversation, task, plan, agent-state, plan templates                                                                     |
| `@rullama/telemetry`       | `deno add @rullama/telemetry`       | Analytics events, sinks, Prometheus metrics, billing hooks, anomaly detection                                                                                           |
| `@rullama/tool-builtins`   | `deno add @rullama/tool-builtins`   | Built-in tools (bash, file ops, git, web, search, semantic search, calendar, sessions) + `createBuiltinExecutor()`                                                      |
| `@rullama/tool-runtime`    | `deno add @rullama/tool-runtime`    | Tool registry, `ToolExecutor`, `EnforcingExecutor` / `enforce()`, sanitization, guards, `safeFetch`, smart routing, transactions, OpenAPI/OAuth                         |

## Key Types from Core

These types appear throughout the framework:

| Type                     | Purpose                                                   |
| ------------------------ | --------------------------------------------------------- |
| `Message`                | Chat message (user, assistant, tool result)               |
| `ChatOptions`            | Model parameters (max_tokens, temperature, model…)        |
| `ChatResponse`           | Provider response (`message` + `usage` + `finish_reason`) |
| `Tool`                   | Tool definition (name, description, input schema)         |
| `ToolUse` / `ToolResult` | Tool call request and response                            |
| `ToolContext`            | Working directory + metadata passed to every tool         |
| `PermissionMode`         | `"read-only"` / `"auto"` / `"full"`                       |
| `FrameworkError`         | Typed error hierarchy                                     |
| `LifecycleHook`          | Event interception for the framework lifecycle            |
| `Provider`               | Interface all chat providers implement                    |

## Runtime boundary — what stays Rust-only

The Deno port is deliberately a subset. A handful of crates in the Rust
framework are Rust-only on purpose, and Deno consumers should drive the Rust
binary for those concerns instead of trying to approximate them:

- **`rullama-hardware`** — GPIO, USB, BLE, CPAL audio, Zigbee, Z-Wave, Matter.
  Needs OS kernel access; not reachable from Deno without FFI.
- **`rullama-sandbox`, `rullama-sandbox-proxy`** — Bollard Docker orchestration
  and Hyper-based egress proxy. Run the Rust sidecar.
- **`local_llm` provider** — llama-cpp FFI. Use `OllamaChatProvider` for local
  inference from Deno.
- **`interpreters` / `orchestrator` tools** — Rhai, Boa, RustPython embedded
  runtimes.
- **`sandbox_executor` / `code_exec`** — depend on the Rust sandbox crate.
- **`browser` tool** — pairs with the Rust Thalora headless browser.
- **LanceDB / ONNX / tantivy RAG** — native indexing and embedding stays in
  Rust. The Deno `@rullama/rag` package keeps its client role and talks to a
  Rust RAG service over the `RagClient` interface; `@rullama/knowledge` is the
  matching type contract for the knowledge graph (`BrainClient`).
- **Burn-based local training** — Deno ships `@rullama/finetune` with cloud
  backends only (OpenAI, Together, Fireworks).
- **`ChatAgent`, `CycleOrchestrator`** — the Rust `rullama-inference` crate's
  chat agent and Plan → Work → Judge orchestrator are not ported; Deno ships
  `TaskAgent` plus the judge/planner prompt builders and parsers.

Communication across the boundary goes through `@rullama/mcp-server` /
`@rullama/mcp-client` (MCP over stdio), `@rullama/network` (relay + remote
bridge) or `@rullama/a2a` (Google A2A protocol).

For per-file detail, see [parity.md](./parity.md) — it links to each
`SKIPPED.md` under the corresponding package.

## Further Reading

- [Getting Started](./getting-started.md) for a hands-on quickstart
- [Extensibility](./extensibility.md) for how to implement custom providers,
  storage, and tools
- [Parity](./parity.md) — crate-by-crate diff against the Rust workspace
