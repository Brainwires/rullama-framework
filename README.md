# rullama

[![CI](https://github.com/Brainwires/rullama-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Brainwires/rullama-framework/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/brainwires.svg)](https://crates.io/crates/rullama)
[![Documentation](https://docs.rs/rullama/badge.svg)](https://docs.rs/rullama)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)
[![Lines of Code](https://img.shields.io/badge/loc-112k-blue)](#)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/rullama-framework/blob/main/LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.91%2B%20%7C%202024%20edition-orange)](https://www.rust-lang.org/)

A modular Rust framework for building AI agents with multi-provider support, tool orchestration, MCP integration, and pluggable agent networking.

**Warning:** This is an early-stage project under active development. Expect breaking changes and rapid iteration as we build towards a 1.0 release.

## Overview

rullama is a workspace of 32 framework crates plus 18 extras (including the 7-crate `brainclaw` set) that provide everything needed to build, train, deploy, and coordinate AI agents. Each framework crate is independently publishable to crates.io and usable standalone, but they compose together through the `rullama` facade crate for a batteries-included experience.

**[Full feature list](FEATURES.md)** | **Key capabilities:**

- **Multi-provider AI** — Anthropic, OpenAI, Google, Ollama, and local LLMs behind a unified `Provider` trait
- **Agent orchestration** — hierarchical task decomposition, multi-agent coordination with file locks, MDAP voting
- **MCP protocol** — full client and server support via `rmcp`, exposing agents as MCP tools
- **Agent networking** — 5-layer protocol stack (IPC, TCP, A2A, Pub/Sub) with pluggable transports, routing, and discovery
- **Training pipelines** — cloud fine-tuning (6 providers) and local LoRA/QLoRA/DoRA via Burn
- **RAG & code search** — AST-aware chunking, hybrid vector + keyword search, Git-aware indexing
- **Audio** — speech-to-text, text-to-speech, hardware capture/playback
- **Security** — encrypted storage (ChaCha20-Poly1305), permission policies, content trust tagging

## Crate Map

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                          rullama                             │
  │                        (facade crate)                           │
  │                                                                 │
  │  ┌────────────┐ ┌─────────────┐ ┌────────────┐ ┌─────────────┐  │
  │  │   agent    │ │  provider   │ │  storage   │ │ mcp-client  │  │
  │  │ tool-runtime│ │   speech   │ │   stores   │ │ mcp-server  │  │
  │  │tool-builtins│ │             │ │   memory   │ │  network    │  │
  │  └─────┬──────┘ └──────┬──────┘ └──────┬─────┘ └──────┬──────┘  │
  │        │               │               │              │         │
  │        └───────────────┴───────────────┴──────────────┘         │
  │                              │                                  │
  │                       ┌──────▼──────┐                           │
  │                       │    core     │                           │
  │                       │ permission  │                           │
  │                       │ call-policy │                           │
  │                       └─────────────┘                           │
  │                                                                 │
  │  ┌────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐   │
  │  │ knowledge  │ │  reasoning  │ │telemetry │ │   hardware   │   │
  │  │    rag     │ │   sandbox   │ │   a2a    │ │    finetune  │   │
  │  │ prompting  │ │             │ │          │ │finetune-local│   │
  │  └────────────┘ └─────────────┘ └──────────┘ └──────────────┘   │
  └─────────────────────────────────────────────────────────────────┘
```

### Framework Crates

| Crate | Description |
|-------|-------------|
| [**rullama**](crates/rullama/README.md) | Facade crate — re-exports every other framework crate behind feature flags |
| [**rullama-core**](crates/rullama-core/README.md) | Core types, traits, and error handling shared by all crates |
| [**rullama-provider**](crates/rullama-provider/README.md) | Multi-provider LLM interface (Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex AI, local llama.cpp / Candle) |
| [**rullama-provider-speech**](crates/rullama-provider-speech/README.md) | Speech (TTS / STT) providers (Azure, Cartesia, Deepgram, ElevenLabs, Fish, Google, Murf, browser web-speech) |
| [**rullama-tool-runtime**](crates/rullama-tool-runtime/README.md) | Tool framework — `ToolExecutor`, `ToolRegistry`, validation, smart router, sandbox/orchestrator/sessions/oauth/openapi |
| [**rullama-tool-builtins**](crates/rullama-tool-builtins/README.md) | Built-in tool implementations: bash, file_ops, git, web, search, code_exec, browser, email, calendar, system, semantic_search |
| [**rullama-agent**](crates/rullama-agent/README.md) | Agent coordination primitives + multi-agent patterns — communication, locks, queues, git coordination, contract net, saga, optimistic concurrency, market allocation, workflow graph |
| [**rullama-inference**](crates/rullama-inference/README.md) | LLM-driven workhorses — `ChatAgent`, `TaskAgent`, planner / judge / validator helpers, cycle orchestrator, validation loop, summarization, system-prompt registry |
| [**rullama-mdap**](crates/rullama-mdap/README.md) | Multi-Dimensional Adaptive Planning (MAKER voting framework) |
| [**rullama-seal**](crates/rullama-seal/README.md) | Self-Evolving Agentic Learning — coreference, query-core extraction, learned-pattern store, reflection |
| [**rullama-skills**](crates/rullama-skills/README.md) | SKILL.md skills system — manifest parsing, registry, smart routing, sandboxed execution |
| [**rullama-eval**](crates/rullama-eval/README.md) | Evaluation harness — fixtures, regression suites, stability tests, adversarial cases, NDCG / MRR / precision@k |
| [**rullama-knowledge**](crates/rullama-knowledge/README.md) | Knowledge layer — BKS / PKS, BrainClient, entity graph |
| [**rullama-rag**](crates/rullama-rag/README.md) | Codebase indexing + hybrid retrieval (vector + BM25), AST-aware chunking via tree-sitter, Git history search |
| [**rullama-prompting**](crates/rullama-prompting/README.md) | Adaptive prompting — technique library, K-means task clustering, BKS/PKS-aware generator, SEAL feedback hook |
| [**rullama-storage**](crates/rullama-storage/README.md) | Substrate — `StorageBackend` trait, 9 backends, embeddings, BM25 keyword search, file-context primitives |
| [**rullama-stores**](crates/rullama-stores/README.md) | Schema + CRUD for the opinionated minimum store set: sessions, conversations, tasks, plans, locks, images, templates + tier schema stores |
| [**rullama-memory**](crates/rullama-memory/README.md) | Tiered memory **orchestration** — `TieredMemory` multi-factor adaptive search + offline `dream` consolidation engine. Built on `rullama-stores`. |
| [**rullama-permission**](crates/rullama-permission/README.md) | Permission policies (auto, ask, reject) for tool execution |
| [**rullama-mcp-client**](crates/rullama-mcp-client/README.md) | MCP client — connect to external MCP servers and use their tools |
| [**rullama-mcp-server**](crates/rullama-mcp-server/README.md) | MCP server framework with composable middleware; `http` feature adds stateless HTTP+SSE transport, Server Cards (SEP-1649), RFC9728, and Tasks (SEP-1686); `oauth` feature adds JWT validation middleware |
| [**rullama-network**](crates/rullama-network/README.md) | Agent networking — IPC, remote bridge, mesh, WebRTC, LAN discovery |
| [**rullama-reasoning**](crates/rullama-reasoning/README.md) | Reasoning scorers — complexity, entity enhancer, relevance, retrieval classifier, router, strategy selector, summarizer, validator |
| [**rullama-call-policy**](crates/rullama-call-policy/README.md) | Policies on outbound provider calls — retry with backoff, circuit breaker, budget caps, response cache, error classification |
| [**rullama-hardware**](crates/rullama-hardware/README.md) | Hardware I/O — audio (STT/TTS), GPIO, Bluetooth, camera/webcam, raw USB |
| [**rullama-finetune**](crates/rullama-finetune/README.md) | Cloud fine-tune APIs (OpenAI, Anthropic, Together, Fireworks, Anyscale, Bedrock, Vertex AI) + dataset pipelines. Local PEFT (LoRA / QLoRA / DoRA), training-from-scratch, and the pure-wgpu Gemma 4 inference engine all live in the sibling `rullama` workspace. |
| [**rullama-telemetry**](crates/rullama-telemetry/README.md) | OutcomeMetrics, Prometheus export, anomaly detection, billing-hook trait |
| [**rullama-a2a**](crates/rullama-a2a/README.md) | Agent-to-Agent protocol — JSON-RPC 2.0, HTTP/REST, and gRPC bindings |
| [**rullama-sandbox**](crates/rullama-sandbox/README.md) | Container-backed sandbox executor for untrusted tool code |
| [**rullama-sandbox-proxy**](crates/rullama-sandbox-proxy/README.md) | Out-of-process sandbox-executor proxy for isolating untrusted code |

### Extras

| Crate | Description |
|-------|-------------|
| [**rullama-proxy**](extras/rullama-proxy/README.md) | HTTP proxy for AI API request routing |
| [**rullama-brain-server**](extras/rullama-brain-server/README.md) | MCP server binary exposing the `rullama-knowledge::knowledge` subsystem (BKS/PKS, thoughts, entity graphs) |
| [**rullama-rag-server**](extras/rullama-rag-server/README.md) | MCP server binary exposing the `rullama-knowledge::rag` subsystem (codebase indexing + hybrid search) |
| [**agent-chat**](extras/agent-chat/README.md) | Simplified AI chat client with TUI and plain modes |
| [**reload-daemon**](extras/reload-daemon/README.md) | MCP server for killing and restarting AI coding clients |
| [**audio-demo-ffi**](extras/audio-demo-ffi/README.md) | UniFFI bindings (cdylib) exposing rullama-hardware (audio) to C#, Kotlin, Swift, Python |
| [**audio-demo**](extras/audio-demo/README.md) | Cross-platform Avalonia GUI for TTS/STT demo across all audio providers |
| [**brainclaw**](extras/brainclaw/daemon/README.md) | Self-hosted personal AI assistant daemon — multi-provider, per-user sessions, secure gateway |
| [**rullama-gateway**](extras/brainclaw/gateway/README.md) | WebSocket/HTTP channel hub — routes channel adapters to AI agent sessions |
| [**rullama-discord-channel**](extras/brainclaw/mcp-discord/README.md) | Discord channel adapter — reference `Channel` trait implementation, optional MCP server mode |
| [**rullama-telegram-channel**](extras/brainclaw/mcp-telegram/README.md) | Telegram channel adapter — teloxide-based, optional MCP server mode |
| [**rullama-slack-channel**](extras/brainclaw/mcp-slack/README.md) | Slack channel adapter — Socket Mode (no public URL), optional MCP server mode |
| [**rullama-skill-registry**](extras/brainclaw/mcp-skill-registry/README.md) | Skill registry HTTP server — SQLite FTS5, publish/search/download endpoints |
| [**brainclaw-mcp-github**](extras/brainclaw/mcp-github/README.md) | GitHub channel adapter — webhook receiver, REST API, MCP server mode |
| [**rullama-memory-server**](extras/rullama-memory-server/README.md) | Mem0-compatible memory REST API backed by rullama knowledge |
| [**claude-brain**](extras/claude-brain/README.md) | rullama context management for Claude Code — persistent context across compaction |
| [**rullama-cli**](extras/rullama-cli/README.md) | AI-powered agentic CLI tool for autonomous coding assistance |
| [**rullama-issues**](extras/rullama-issues/README.md) | MCP-native issue tracking server |
| [**rullama-scheduler**](extras/rullama-scheduler/README.md) | MCP server for cron scheduling |
| [**rullama-autonomy**](extras/rullama-autonomy/README.md) | Autonomous agent operations |
| [**rullama-wasm**](extras/rullama-wasm/README.md) | WASM browser bindings |
| [**rullama-billing**](extras/rullama-billing/README.md) | Billing and cost accounting hooks for agent telemetry |
| [**rullama-docs**](extras/rullama-docs/README.md) | Documentation tooling and reference site generation |
| [**voice-assistant**](extras/voice-assistant/README.md) | End-to-end voice assistant binary using the `rullama-hardware` pipeline |

### Workspace layout

- **`crates/`** — the framework. Cohesive, independently-publishable libraries.
- **`extras/`** — applications and libraries that **consume** the framework: binaries, demos, MCP servers, and integration helpers.

**Allowed dependency arrows:** `crates/ → crates/` and `extras/ → crates/`.

### Brands, repos, and the engine/harness boundary

**rullama is the open-source platform**; **[rullama](../rullama) is the app**
(`rullama.com`) that runs on it. Two names, one downward dependency:

- The platform holds both the inference **engine** (`rullama-engine` — the
  Rust → WASM + WebGPU inference path, moving in from the old `rullama` crate) and
  the agent **harness** (the `rullama-*` crates here). They stay separate,
  joined by the `Provider` seam; the engine is a first-party WebGPU provider.
- The **rullama product family** (the PWA, `rullama-native` — a shipping paid
  .NET/Avalonia desktop+mobile app, and `rullama-cli`) consumes the platform
  three ways: in-browser via the engine's wasm bundle, natively via an
  OpenAI-compatible `/v1/chat/completions` endpoint (existing `openai_chat`
  provider, base-URL swap), and in-process via a C-ABI shim (rullama-native). The
  PWA supersedes the old `rullama-studio` and the Candle
  `extras/rullama-chat-pwa` (both retire).
- **brainclaw** is extracting to its own product repo, and **rullama-cli** is
  extracting *and being renamed `rullama-cli`* (it joins the rullama product
  family — app + CLI). Both depend on published `rullama` crates.

See the canonical reference:
[`docs/ARCHITECTURE-engine-harness.md`](docs/ARCHITECTURE-engine-harness.md).

**Forbidden:** `crates/ → extras/` (the framework cannot depend on its consumers) and `extras/ → extras/` (extras are siblings of equal standing, not a hierarchy). If an `extras/` library starts being depended on by another `extras/` entry, that's a signal it belongs in `crates/`.

Enforcement: `cargo xtask lint-deps` walks every `Cargo.toml` and rejects forbidden arrows. See [`docs/adr/ADR-0004-framework-extras-boundary.md`](docs/adr/ADR-0004-framework-extras-boundary.md) for the rationale.

The `deprecated/` directory holds historical crates that have been merged or retired; nothing in the active workspace depends on it.

## Getting Started

### Requirements

- **Rust 1.91+** (edition 2024)
- **Cargo** (comes with Rust)

> **Note:** This framework uses `edition = "2024"` which requires Rust 1.91 or newer. Check your version with `rustc --version` and update with `rustup update stable` if needed.

### Using the Facade Crate

The simplest way to use the framework is through the `rullama` facade crate, which re-exports everything behind feature flags:

```toml
[dependencies]
rullama = "0.11"  # defaults: tools + agents
```

Enable only what you need:

```toml
[dependencies]
rullama = { version = "0.11", features = ["provider", "rag"] }
```

### Using Individual Crates

Each crate is independently publishable and usable:

```toml
[dependencies]
rullama-core = "0.11"
rullama-provider = "0.11"
rullama-agent = "0.11"
```

### Minimal Example

```rust
use rullama::prelude::*;
use rullama::providers::{ChatProviderFactory, ProviderConfig, ProviderType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a provider via the factory
    let config = ProviderConfig {
        provider: ProviderType::Anthropic,
        model: "claude-sonnet-4-6".into(),
        api_key: Some("your-api-key".into()),
        base_url: None,
    };
    let provider = ChatProviderFactory::create(&config)?;

    // Send a message
    let messages = vec![Message::user("Hello, what can you do?")];
    let options = ChatOptions::default();
    let response = provider.chat(&messages, None, &options).await?;

    println!("{}", response.message.content);
    Ok(())
}
```

## Feature Flags

The `rullama` facade crate exposes feature flags corresponding to each sub-crate:

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `core` | Always | Core types and traits (not feature-gated) |
| `tools` | Yes | Tool definitions, execution, and interpreters (`rullama-tool-runtime` + `rullama-tool-builtins`) |
| `agents` | Yes | Multi-agent orchestration, communication hub, file/resource locks (`rullama-agent`) |
| `inference` | Yes | LLM-driven workhorses — ChatAgent, PlannerAgent, JudgeAgent, TaskAgent, CycleOrchestrator (`rullama-inference`) |
| `providers` | No | AI provider integrations |
| `storage` | No | Vector storage and semantic search |
| `mcp` | No | MCP client support |
| `agent-network` | No | Agent networking — IPC, remote bridge, channels, 5-layer protocol stack (`rullama-network`) |
| `mcp-server-framework` | No | MCP server building blocks (McpServer, McpHandler, middleware pipeline) |
| `rag` | No | RAG engine with code search |
| `audio` | No | Audio capture, STT, TTS |
| `training` | No | Cloud fine-tuning (local PEFT lives in `rullama-finetune`) |
| `datasets` | No | Training data pipelines (delegates to `rullama-finetune`) |
| `telemetry` | No | OutcomeMetrics, Prometheus export, billing hooks |
| `reasoning` | No | Reasoning strategies (re-exports from core) |
| `mesh` | No | Mesh networking (via `agent-network` mesh feature) |
| `a2a` | No | Agent-to-Agent protocol |
| `wasm` | No | WASM core bindings |
| `researcher` | No | Bundle: providers + agents + storage + rag + training + datasets |

## Building

```bash
# Build all crates (debug)
cargo build

# Build all crates (release)
cargo build --release

# Build a specific crate
cargo build -p rullama-agent

# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p rullama-core
```

## Dependency DAG

```text
  rullama (facade)
  ├── rullama-agent
  │   ├── rullama-core
  │   ├── rullama-call-policy
  │   ├── rullama-tool-runtime
  │   ├── rullama-tool-builtins
  │   ├── rullama-storage (seal feature — for PatternStore)
  │   ├── rullama-knowledge (seal-knowledge feature)
  │   └── rullama-permission (seal-feedback feature)
  ├── rullama-knowledge
  │   ├── rullama-core
  │   └── rullama-storage
  ├── rullama-rag
  │   ├── rullama-core
  │   └── rullama-storage
  ├── rullama-prompting
  │   ├── rullama-core
  │   └── rullama-knowledge (knowledge feature)
  ├── rullama-storage
  │   └── rullama-core
  ├── rullama-stores
  │   ├── rullama-core
  │   └── rullama-storage
  ├── rullama-memory
  │   ├── rullama-core
  │   ├── rullama-storage
  │   └── rullama-stores (memory feature)
  ├── rullama-tool-runtime
  │   ├── rullama-core
  │   ├── rullama-stores (sessions feature — SessionBroker)
  │   ├── rullama-rag (rag feature)
  │   └── rullama-sandbox (sandbox feature)
  ├── rullama-tool-builtins
  │   ├── rullama-tool-runtime
  │   └── rullama-rag (rag feature)
  ├── rullama-mcp-client
  │   └── rullama-core
  ├── rullama-mcp-server
  │   ├── rullama-core
  │   └── rullama-mcp-client
  ├── rullama-network
  │   ├── rullama-core
  │   ├── rullama-mcp-client
  │   └── rullama-a2a (a2a-transport feature)
  ├── rullama-finetune          (cloud only — local PEFT + wgpu Gemma 4 live in rullama)
  │   ├── rullama-core
  │   └── rullama-provider (cloud feature)
  ├── rullama-telemetry
  │   └── rullama-core
  └── rullama-hardware
      ├── rullama-provider (audio feature)
      └── rullama-provider-speech (audio feature)
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
