# rullama — Deno/TypeScript Port

A modular, Deno-native TypeScript port of the
[rullama](https://github.com/Brainwires/rullama-framework). Build autonomous AI
agents with tool use, multi-provider support, inter-agent communication, and
fine-grained permissions — all running on Deno.

## Packages (v0.12.0)

All 27 packages publish to JSR under the `@rullama/*` scope, versioned in
lockstep with the Rust crates (`0.12.0`). The shape mirrors the Rust workspace
1:1 (the singular-crate-name restructure landed in v0.11.0): mcp-client /
mcp-server split, finetune-not-training, etc. No transitional shims — v0.11.0
was a clean break from 0.10.x.

| Package                    | Description                                                                                                                                                    |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@rullama/core`            | Foundation types — messages, tools, errors, lifecycle, confidence, paths, file_context                                                                         |
| `@rullama/a2a`             | Agent-to-Agent protocol (Google A2A) — JSON-RPC + REST                                                                                                         |
| `@rullama/agent`           | Coordination primitives: communication, locks, task manager, contract-net, saga, market, three-state, wait-queue                                               |
| `@rullama/inference`       | LLM workhorses: `runAgentLoop`, `TaskAgent`, `AgentContext` (enforcing by default), `AgentPool`, `ValidatorAgent`, judge/planner prompts + parsers             |
| `@rullama/mdap`            | MAKER voting framework — k-of-n consensus, decomposition, red-flag validation                                                                                  |
| `@rullama/seal`            | Self-Evolving Agentic Learning loop                                                                                                                            |
| `@rullama/skills`          | SKILL.md skills system (parser, registry, executor, router)                                                                                                    |
| `@rullama/eval`            | Evaluation harness (trial runner, regression, adversarial, ranking metrics)                                                                                    |
| `@rullama/provider`        | LLM chat providers (Anthropic, OpenAI, Google, Bedrock, Vertex, Ollama)                                                                                        |
| `@rullama/provider-speech` | TTS/STT/ASR clients (Azure, Cartesia, Deepgram, ElevenLabs, Fish, Google TTS, Murf)                                                                            |
| `@rullama/call-policy`     | Provider decorators — retry / budget / circuit-breaker / cache                                                                                                 |
| `@rullama/mcp-client`      | Model Context Protocol client (stdio transport) + `McpConfigManager`                                                                                           |
| `@rullama/mcp-server`      | MCP tool-server framework (initialize / tools/list / tools/call) + middleware pipeline + stdio transport                                                       |
| `@rullama/network`         | Agent-to-agent networking: identity, routing, discovery, peer table, remote bridge (unauthenticated transport)                                                 |
| `@rullama/storage`         | StorageBackend trait + Postgres/MySQL/Qdrant/SurrealDB/Pinecone/Weaviate/Milvus + embeddings                                                                   |
| `@rullama/stores`          | Domain stores: message, conversation, task, plan, agent state, plan templates                                                                                  |
| `@rullama/memory`          | Tiered memory (hot/warm/cold) + multi-factor retention scoring                                                                                                 |
| `@rullama/session`         | Pluggable session persistence (in-memory, Deno KV)                                                                                                             |
| `@rullama/knowledge`       | Knowledge-graph types (Thought / Entity / Relationship) + the `BrainClient` interface (no implementation)                                                      |
| `@rullama/prompting`       | 15 prompting techniques + task clustering + temperature optimization                                                                                           |
| `@rullama/rag`             | RAG client interface + code analysis (symbol extraction, repo maps, call graphs) under `@rullama/rag/code-analysis`                                            |
| `@rullama/tool-runtime`    | Tool execution framework: registry, `ToolExecutor`, `EnforcingExecutor` / `enforce()`, guards + `safeFetch`, sanitization, router, transaction, OpenAPI, OAuth |
| `@rullama/tool-builtins`   | Built-in tools (bash, file ops, git, web, search, semantic search, calendar, sessions) + `createBuiltinExecutor()`                                             |
| `@rullama/permission`      | Capability profiles, policy engine, audit, trust — enforced by `tool-runtime` since v0.12.0                                                                    |
| `@rullama/telemetry`       | Analytics events, sinks, Prometheus metrics, billing hooks, anomaly detection                                                                                  |
| `@rullama/reasoning`       | Plan parser, complexity/router/validator/retrieval scorers                                                                                                     |
| `@rullama/finetune`        | Cloud fine-tuning (OpenAI, Together, Fireworks)                                                                                                                |

v0.11.0 is a breaking release. The pre-rename package names (`providers`,
`permissions`, `agents`, `mcp`, `resilience`, `training`, `tools`) are **not**
published as tombstones — consumers must update imports to the new names.

## Documentation & Examples

- **[Documentation](./docs/)** — Guides covering architecture, each subsystem,
  and extensibility
- **[Examples](./examples/)** — Runnable TypeScript examples ported from the
  Rust crates

## Package Dependency Diagram

Edges are the real `@rullama/*` imports in each package (non-test files):

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

  No @rullama/* imports: a2a, eval, finetune, knowledge, mcp-client, mdap,
  prompting, rag, skills
```

New in v0.12.0: `tool-runtime → permission` (the enforcing executor applies the
policy engine and capability profiles) and `tool-builtins → tool-runtime` (the
built-in executor wraps itself in that enforcement). `inference` uses `core`,
`agent` and `tool-runtime`; the provider is passed in by the caller.

## Quick Start

```ts
import { Task } from "@rullama/core";
import { CommunicationHub, FileLockManager } from "@rullama/agent";
import { AgentContext, TaskAgent } from "@rullama/inference";
import { AnthropicChatProvider } from "@rullama/provider";
import { createBuiltinExecutor } from "@rullama/tool-builtins";

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
);

// bash, file ops, git, web fetch and code search behind the enforcing
// executor (policy engine, capability profile, output filtering).
const executor = createBuiltinExecutor({ mode: "auto" });

// AgentContext enforces permissions by default; pass `false` as the sixth
// argument to opt out.
const context = new AgentContext(
  Deno.cwd(),
  executor,
  new CommunicationHub(),
  new FileLockManager(),
);

const task = new Task("demo-task", "List the files in the current directory.");
const agent = new TaskAgent("demo-agent", task, provider, context, {
  systemPrompt: "You are a helpful coding assistant.",
});
const result = await agent.execute();

console.log(`Success: ${result.success}\nSummary: ${result.summary}`);
```

## What's Ported vs What's Not

Per-file detail lives in [docs/parity.md](./docs/parity.md). Runtime-boundary
crates that stay Rust-only:

- **`rullama-hardware`** — kernel access
  (GPIO/USB/BLE/ALSA/Zigbee/Z-Wave/Matter)
- **`rullama-sandbox` / -sandbox-proxy** — Bollard Docker / Hyper HTTP proxy
- Within `@rullama/tool-builtins` — `interpreters`, `code_exec`,
  `sandbox_executor`, `browser`, `email`, `system` (see `SKIPPED.md`)
- Within `@rullama/inference` — `ChatAgent`, `JudgeAgent` / `PlannerAgent`
  classes and the `CycleOrchestrator` loop (Deno ships the prompts + parsers)
- Local LLM inference (llama.cpp, Candle) — use `OllamaChatProvider` instead

## Installation

```sh
deno add jsr:@rullama/core jsr:@rullama/provider jsr:@rullama/inference
# … etc per package needed
```

## License

Same license as the parent rullama repository.
