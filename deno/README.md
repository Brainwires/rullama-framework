# Brainwires Framework — Deno/TypeScript Port

A modular, Deno-native TypeScript port of the
[Brainwires Agent Framework](https://github.com/Brainwires/brainwires-framework).
Build autonomous AI agents with tool use, multi-provider support, inter-agent
communication, and fine-grained permissions — all running on Deno.

## Packages (v0.11.0)

All 27 packages publish to JSR under the `@brainwires/*` scope. The shape
mirrors the Rust workspace 1:1 (post-v0.11.0 restructure): singular crate names,
mcp-client / mcp-server split, finetune-not-training, etc. No transitional
shims — v0.11.0 is a clean break from 0.10.x.

| Package                       | Description                                                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `@brainwires/core`            | Foundation types — messages, tools, errors, lifecycle, confidence, paths, file_context                           |
| `@brainwires/a2a`             | Agent-to-Agent protocol (Google A2A) — JSON-RPC + REST                                                           |
| `@brainwires/agent`           | Coordination primitives: communication, locks, task manager, contract-net, saga, market, three-state, wait-queue |
| `@brainwires/inference`       | LLM workhorses: TaskAgent / ChatAgent / Judge / Planner / Validator / CycleOrchestrator / runtime                |
| `@brainwires/mdap`            | MAKER voting framework — k-of-n consensus, decomposition, red-flag validation                                    |
| `@brainwires/seal`            | Self-Evolving Agentic Learning loop                                                                              |
| `@brainwires/skills`          | SKILL.md skills system (parser, registry, executor, router)                                                      |
| `@brainwires/eval`            | Evaluation harness (trial runner, regression, adversarial, ranking metrics)                                      |
| `@brainwires/provider`        | LLM chat providers (Anthropic, OpenAI, Google, Bedrock, Vertex, Ollama, Brainwires Relay)                        |
| `@brainwires/provider-speech` | TTS/STT/ASR clients (Azure, Cartesia, Deepgram, ElevenLabs, Fish, Google TTS, Murf)                              |
| `@brainwires/call-policy`     | Provider decorators — retry / budget / circuit-breaker / cache                                                   |
| `@brainwires/mcp-client`      | Model Context Protocol client                                                                                    |
| `@brainwires/mcp-server`      | MCP server framework + middleware pipeline + stdio transport                                                     |
| `@brainwires/network`         | Agent-to-agent networking: identity, routing, discovery, peer table, remote bridge                               |
| `@brainwires/storage`         | StorageBackend trait + Postgres/MySQL/Qdrant/SurrealDB/Pinecone/Weaviate/Milvus + embeddings                     |
| `@brainwires/stores`          | Domain stores: message, conversation, task, plan, template, lock, image                                          |
| `@brainwires/memory`          | Tiered memory (hot/warm/cold) + multi-factor retention scoring                                                   |
| `@brainwires/session`         | Pluggable session persistence (in-memory, Deno KV)                                                               |
| `@brainwires/knowledge`       | BrainClient + entity/relationship graph + BKS/PKS thought storage                                                |
| `@brainwires/prompting`       | 15 prompting techniques + task clustering + temperature optimization                                             |
| `@brainwires/rag`             | RAG client interface + code-analysis (symbol extraction, repo maps)                                              |
| `@brainwires/tool-runtime`    | Tool execution framework: registry, executor, sanitization, router, transaction, OpenAPI, OAuth, validation      |
| `@brainwires/tool-builtins`   | Built-in tools: bash, file ops, git, web, search, semantic search, calendar, sessions                            |
| `@brainwires/permission`      | Capability profiles, policy engine, audit, trust                                                                 |
| `@brainwires/telemetry`       | Analytics events, sinks, Prometheus metrics, billing hooks, anomaly detection                                    |
| `@brainwires/reasoning`       | Plan parser, complexity/router/validator/retrieval scorers                                                       |
| `@brainwires/finetune`        | Cloud fine-tuning (OpenAI, Together, Fireworks)                                                                  |

v0.11.0 is a breaking release. The pre-rename package names (`providers`,
`permissions`, `agents`, `mcp`, `resilience`, `training`, `tools`) are **not**
published as tombstones — consumers must update imports to the new names.

## Documentation & Examples

- **[Documentation](./docs/)** — Guides covering architecture, each subsystem,
  and extensibility
- **[Examples](./examples/)** — Runnable TypeScript examples ported from the
  Rust crates

## Package Dependency Diagram

```
                          core (zero deps)
                            │
        ┌──────────┬────────┼─────────┬────────┬───────────┐
   call-policy  permission  │   provider   storage    telemetry
                            │      │          │          │
                          mcp-client          │          │
                            │                 │          │
                       mcp-server          stores       memory
                                              │          │
                                           session    knowledge
                                                         │
                                                     prompting, rag

                tool-runtime ── tool-builtins ── skills
                      │
                  inference (needs provider + tool-runtime + call-policy)
                      │
                  agent (coordination)
                  mdap, seal, eval (independent)
```

## Quick Start

```ts
import { ChatOptions, Message } from "@brainwires/core";
import { AnthropicChatProvider } from "@brainwires/provider";
import { BashTool, FileOpsTool } from "@brainwires/tool-builtins";
import { ToolRegistry } from "@brainwires/tool-runtime";
import { AgentContext, spawnTaskAgent, TaskAgent } from "@brainwires/inference";

const registry = new ToolRegistry();
registry.registerTools(BashTool.getTools());
registry.registerTools(FileOpsTool.getTools());

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
  "anthropic",
);

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

## What's Ported vs What's Not

Per-file detail lives in [docs/parity.md](./docs/parity.md). Runtime-boundary
crates that stay Rust-only:

- **`brainwires-hardware`** — kernel access
  (GPIO/USB/BLE/ALSA/Zigbee/Z-Wave/Matter)
- **`brainwires-sandbox` / -sandbox-proxy** — Bollard Docker / Hyper HTTP proxy
- Within `@brainwires/tool-builtins` — `interpreters`, `code_exec`,
  `sandbox_executor`, `browser`, `email`, `system` (see `SKIPPED.md`)
- Local LLM inference (llama.cpp, Candle) — use `OllamaChatProvider` instead

## Installation

```sh
deno add @brainwires/core @brainwires/provider @brainwires/inference
# … etc per package needed
```

## License

Same license as the parent Brainwires Framework repository.
