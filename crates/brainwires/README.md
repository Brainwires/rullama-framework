# brainwires

[![Crates.io](https://img.shields.io/crates/v/brainwires.svg)](https://crates.io/crates/brainwires)
[![Documentation](https://img.shields.io/docsrs/brainwires)](https://docs.rs/brainwires)
[![License](https://img.shields.io/crates/l/brainwires.svg)](LICENSE)

Unified facade crate for the Brainwires Agent Framework — build any AI application in Rust.

## Overview

`brainwires` is the single entry point for the entire framework. It re-exports 19 sub-crates as feature-gated modules and provides a `prelude` that pulls in the most commonly needed types. Add one dependency, enable the features you need, and you're ready to go.

`brainwires-core` (messages, tools, providers, tasks, errors) is **always available** — no feature flag required. Everything else is opt-in.

```text
                             ┌─────────────┐
                             │  brainwires │  (facade)
                             └──────┬──────┘
           ┌──────────┬─────────┬───┴───┬─────────┬─────────┐
           │          │         │       │         │         │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼───┐ ┌─▼────┐ ┌──▼───┐ ┌───▼────┐
    │  core   │ │ tooling │ │ agents│ │ mcp  │ │ mdap │ │storage │
    │ (always)│ │         │ │       │ │      │ │      │ │        │
    └─────────┘ └─────────┘ └───────┘ └──────┘ └──────┘ └────────┘
           ┌──────────┬─────────┬───────┬─────────┬─────────┐
           │          │         │       │         │         │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼───┐ ┌─▼────┐ ┌──▼───┐ ┌───▼────┐
    │prompting│ │permiss- │ │  rag  │ │seal  │ │relay │ │provid- │
    │         │ │  ions   │ │       │ │      │ │      │ │  ers   │
    └─────────┘ └─────────┘ └───────┘ └──────┘ └──────┘ └────────┘
           ┌──────────┬─────────┬───────┬─────────┐
           │          │         │       │         │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼───┐ ┌─▼────┐ ┌──▼───┐
    │ skills  │ │  eval   │ │ proxy │ │ a2a  │ │ mesh │
    └─────────┘ └─────────┘ └───────┘ └──────┘ └──────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires = "0.11"  # default features: tools + agents
```

Then import via the prelude:

```rust
use brainwires::prelude::*;

let messages = vec![
    Message::system("You are a helpful assistant."),
    Message::user("Hello!"),
];

let options = ChatOptions::deterministic(1024);
let response = provider.chat(&messages, None, &options).await?;
```

## Features

Source of truth: [`Cargo.toml`](Cargo.toml). Listed in rough capability order.

| Feature | Default | Activates | Description |
|---------|---------|-----------|-------------|
| `tools` | **yes** | `brainwires-tool-runtime` + `brainwires-tool-builtins` | File, bash, git, search, web, and validation tools |
| `agents` | **yes** | `brainwires-agent` | Agent runtime, communication hub, task manager, validation loop |
| `inference` | **yes** | `brainwires-inference` | LLM-driven workhorses (ChatAgent, PlannerAgent, JudgeAgent, TaskAgent, CycleOrchestrator) |
| `wasm` | no | `brainwires-core/wasm` | WASM-safe build of `brainwires-core` (no native deps) |
| `storage` | no | `brainwires-storage` | Unified database layer (9 backends) |
| `memory` | no | `brainwires-stores` (memory) | Conversation message stores, fact stores, mental-model stores |
| `tiered` | no | `brainwires-memory` | TieredMemory orchestration (hot/warm/cold) + multi-factor search |
| `mcp` | no | `brainwires-mcp-client` | MCP client for connecting to external MCP servers |
| `mcp-server` | no | `rmcp` + `schemars` + `tokio-util` | Low-level MCP server re-exports |
| `mcp-server-framework` | no | `brainwires-mcp-server` | Higher-level MCP server framework with middleware |
| `a2a` | no | `brainwires-a2a` | Agent-to-Agent protocol (JSON-RPC 2.0, HTTP, gRPC) |
| `agent-network` | no | `brainwires-network` | 5-layer networking stack (IPC, TCP, A2A, pub/sub) |
| `mesh` | no | `brainwires-network/mesh` | Mesh networking for distributed agents (implies `agent-network`) |
| `mdap` | no | `brainwires-mdap` | Multi-Dimensional Adaptive Planning with k-agent voting |
| `prompting` | no | `brainwires-prompting` | Prompt generation, technique library, temperature optimizer |
| `knowledge` | no | `brainwires-knowledge` | Persistent knowledge caches — BKS/PKS behavioral + personal stores, entity graphs |
| `dream` | no | `brainwires-memory/dream` | Offline consolidation / replay passes over tiered memory (implies `tiered`) |
| `rag` | no | `brainwires-rag` + `brainwires-storage` | Semantic code search with vector + BM25 hybrid search |
| `rag-full-languages` | no | `rag` | Full tree-sitter language pack (alias for `rag`) |
| `permissions` | no | `brainwires-permission` | Capability profiles, trust levels, policy engine, audit logging |
| `orchestrator` | no | `brainwires-tool-runtime/orchestrator` + `brainwires-tool-builtins/orchestrator` | Tool orchestration layer (implies `tools`) |
| `interpreters` | no | `brainwires-tool-builtins/interpreters` | Sandboxed Rhai / Lua / JS code execution |
| `system` | no | `brainwires-tool-builtins/system` | System-level tool primitives |
| `openapi` | no | `brainwires-tool-runtime/openapi` | Auto-generate tools from OpenAPI 3.x specs |
| `providers` | no | `brainwires-provider` | AI providers (Anthropic, OpenAI, Google, Ollama) |
| `chat` | no | `brainwires-provider` | Chat helpers built on `providers` |
| `bedrock` | no | `brainwires-provider/bedrock` | AWS Bedrock provider (implies `providers`) |
| `vertex-ai` | no | `brainwires-provider/vertex-ai` | Google Vertex AI provider (implies `providers`) |
| `llama-cpp-2` | no | `brainwires-provider/llama-cpp-2` | Local LLM inference (implies `providers`) |
| `reasoning` | no | `brainwires-reasoning` | Reasoning strategies (planners, validators, routers, scorers) |
| `seal` | no | `brainwires-seal` | Self-Evolving Autonomous Learner |
| `skills` | no | `brainwires-skills` | Pluggable skills system (SKILL.md routing) |
| `eval` | no | `brainwires-eval` | Evaluation framework for benchmarking agents |
| `otel` | no | `brainwires-agent/otel` | OpenTelemetry span export for agent traces |
| `telemetry` | no | `brainwires-telemetry` | OutcomeMetrics, Prometheus export, billing hooks |
| `audio` | no | `brainwires-hardware/audio` | Audio capture, STT, TTS (16 cloud providers + local Whisper) |
| `vad` | no | `brainwires-hardware/vad` | WebRTC voice activity detection (`EnergyVad` always available with `audio`) |
| `wake-word` | no | `brainwires-hardware/wake-word` | Wake word detection — `EnergyTriggerDetector` (zero deps) |
| `voice-assistant` | no | `brainwires-hardware/voice-assistant` | Full voice assistant pipeline (implies `audio` + `vad` + `wake-word`) |
| `gpio` | no | `brainwires-hardware/gpio` | GPIO pin control with safety allow-lists (Linux) |
| `bluetooth` | no | `brainwires-hardware/bluetooth` | BLE advertisement scanning and adapter enumeration |
| `network-hardware` | no | `brainwires-hardware/network` | NIC enumeration, IP config, ARP discovery, port scanning |
| `camera` | no | `brainwires-hardware/camera` | Webcam/camera frame capture (V4L2/AVFoundation/MSMF) |
| `usb` | no | `brainwires-hardware/usb` | Raw USB device enumeration and transfers (no libusb) |
| `training` | no | `brainwires-finetune` | Cloud fine-tuning APIs |
| `training-cloud` | no | `brainwires-finetune/cloud` | Cloud fine-tuning (alias for `training`) |
| `datasets` | no | `brainwires-finetune/datasets-full` | Training data pipelines (JSONL, tokenization, dedup) |

> Local PEFT (LoRA / QLoRA / DoRA via Burn) and training-from-scratch live
> in the sibling `rullama` workspace as `rullama-finetune` and
> `rullama-training`. They had `training-local` / `training-full` facade
> features prior to v0.11 — depend on the rullama crates directly now.

### Recommended profile

If you're unsure which features to pick, start with:

```toml
[dependencies]
brainwires = { version = "0.11", features = ["agent-full", "reasoning", "providers"] }
```

That gives you the full agent runtime (communication hub, validation loop,
task manager), capability-based permissions, prompt generation, the reasoning
scorers and strategy selector, and the Anthropic / OpenAI / Google / Ollama
providers — the smallest feature set that ships a complete chat-agent app.
Add `storage + rag` when you need persistence, `mcp` or `a2a` when you need
interop, and `seal + knowledge` when you want self-improving behavior.

### Convenience Features

| Feature | Enables | Use Case |
|---------|---------|----------|
| `agent-full` | `agents` + `permissions` + `prompting` + `tools` | Complete agent workflow with permissions |
| `researcher` | `providers` + `agents` + `storage` + `rag` + `training` + `datasets` | Full research workflow |
| `learning` | `seal` + `knowledge` + `brainwires-agent/seal-knowledge` + `brainwires-agent/seal-feedback` | Full learning subsystem with knowledge integration |
| `full` | Everything | Kitchen sink — all sub-crates and cross-crate features |

## Prelude

`use brainwires::prelude::*` brings in the most commonly needed types, grouped by subsystem:

**Core** (always available):
`Message`, `Role`, `ContentBlock`, `ChatResponse`, `StreamChunk`, `Usage`, `Tool`, `ToolUse`, `ToolResult`, `ToolContext`, `ToolInputSchema`, `Provider`, `ChatOptions`, `Task`, `TaskStatus`, `TaskPriority`, `PlanMetadata`, `PlanStatus`, `PermissionMode`, `EntityType`, `EdgeType`, `GraphNode`, `GraphEdge`, `EmbeddingProvider`, `VectorStore`, `WorkingSet`, `FrameworkError`, `FrameworkResult`

**Tools** (`tools` feature):
`BashTool`, `FileOpsTool`, `GitTool`, `SearchTool`, `WebTool`, `ValidationTool`, `ToolRegistry`, `ToolCategory`, `ToolErrorCategory`, `RetryStrategy`

**Agents** (`agents` feature):
`AgentRuntime`, `AgentExecutionResult`, `run_agent_loop`, `CommunicationHub`, `FileLockManager`, `TaskManager`, `TaskQueue`, `ValidationConfig`, `AccessControlManager`, `GitCoordinator`, `PlanExecutorAgent`

**Storage** (`storage` feature):
`CachedEmbeddingProvider`

**Memory** (`memory` feature):
`TieredMemory` (re-exported from `brainwires-memory`)

**MCP** (`mcp` feature):
`McpClient`, `McpConfigManager`, `McpServerConfig`

**MDAP** (`mdap` feature):
`Composer`, `MdapEstimate`, `MicroagentConfig`, `FirstToAheadByKVoter`

**Knowledge** (`knowledge` feature):
`BehavioralKnowledgeCache`, `PersonalKnowledgeCache`, `BehavioralTruth`, `TruthCategory`

**Prompting** (`prompting` feature):
`PromptGenerator`, `PromptingTechnique`, `TechniqueLibrary`, `TemperatureOptimizer`, `TaskClusterManager`

**Permissions** (`permissions` feature):
`AgentCapabilities`, `PolicyEngine`, `TrustLevel`, `TrustManager`, `AuditLogger`, `PermissionsConfig`

## Usage Examples

### Agent Workflow

```toml
[dependencies]
brainwires = { version = "0.11", features = ["agent-full"] }
```

```rust
use brainwires::prelude::*;

// Set up the agent runtime
let hub = CommunicationHub::new();
let lock_manager = FileLockManager::new();
let runtime = AgentRuntime::new(hub, lock_manager);

// Define validation checks
let validation = ValidationConfig {
    checks: vec![ValidationCheck::FileExistence, ValidationCheck::Syntax],
    working_directory: "/my/project".into(),
    max_retries: 3,
    enabled: true,
    working_set_files: vec![],
};
```

### MCP Server with RAG

```toml
[dependencies]
brainwires = { version = "0.11", features = ["rag", "mcp-server"] }
```

```rust
use brainwires::rag::mcp_server::RagMcpServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    RagMcpServer::serve_stdio().await?;
    Ok(())
}
```

### RAG Pipeline

```toml
[dependencies]
brainwires = { version = "0.11", features = ["rag"] }
```

```rust
use brainwires::rag::RagClient;

let client = RagClient::new(None).await?;
client.index("/path/to/project", None, None).await?;

let results = client.query("authentication logic", 10, 0.7).await?;
for result in results {
    println!("{}: {:.2}", result.file_path, result.score);
}
```

### Learning System

```toml
[dependencies]
brainwires = { version = "0.11", features = ["learning"] }
```

```rust
use brainwires::prelude::*;

let cache = BehavioralKnowledgeCache::new();
let truth = BehavioralTruth::new("always_use_async", TruthCategory::Pattern);
cache.store(truth);
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
