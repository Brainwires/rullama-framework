# Brainwires Framework

[![CI](https://github.com/Brainwires/brainwires-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Brainwires/brainwires-framework/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/brainwires.svg)](https://crates.io/crates/brainwires)
[![Documentation](https://docs.rs/brainwires/badge.svg)](https://docs.rs/brainwires)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)
[![Lines of Code](https://img.shields.io/badge/loc-112k-blue)](#)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.91%2B%20%7C%202024%20edition-orange)](https://www.rust-lang.org/)

A modular Rust framework for building AI agents with multi-provider support, tool orchestration, MCP integration, and pluggable agent networking.

**Warning:** This is an early-stage project under active development. Expect breaking changes and rapid iteration as we build towards a 1.0 release.

## Overview

The Brainwires Framework is a workspace of 32 framework crates plus 18 extras (including the 7-crate `brainclaw` set) that provide everything needed to build, train, deploy, and coordinate AI agents. Each framework crate is independently publishable to crates.io and usable standalone, but they compose together through the `brainwires` facade crate for a batteries-included experience.

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
  │                          brainwires                             │
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
| [**brainwires**](crates/brainwires/README.md) | Facade crate — re-exports every other framework crate behind feature flags |
| [**brainwires-core**](crates/brainwires-core/README.md) | Core types, traits, and error handling shared by all crates |
| [**brainwires-provider**](crates/brainwires-provider/README.md) | Multi-provider LLM interface (Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex AI, local llama.cpp / Candle) |
| [**brainwires-provider-speech**](crates/brainwires-provider-speech/README.md) | Speech (TTS / STT) providers (Azure, Cartesia, Deepgram, ElevenLabs, Fish, Google, Murf, browser web-speech) |
| [**brainwires-tool-runtime**](crates/brainwires-tool-runtime/README.md) | Tool framework — `ToolExecutor`, `ToolRegistry`, validation, smart router, sandbox/orchestrator/sessions/oauth/openapi |
| [**brainwires-tool-builtins**](crates/brainwires-tool-builtins/README.md) | Built-in tool implementations: bash, file_ops, git, web, search, code_exec, browser, email, calendar, system, semantic_search |
| [**brainwires-agent**](crates/brainwires-agent/README.md) | Agent coordination primitives + multi-agent patterns — communication, locks, queues, git coordination, contract net, saga, optimistic concurrency, market allocation, workflow graph |
| [**brainwires-inference**](crates/brainwires-inference/README.md) | LLM-driven workhorses — `ChatAgent`, `TaskAgent`, planner / judge / validator helpers, cycle orchestrator, validation loop, summarization, system-prompt registry |
| [**brainwires-mdap**](crates/brainwires-mdap/README.md) | Multi-Dimensional Adaptive Planning (MAKER voting framework) |
| [**brainwires-seal**](crates/brainwires-seal/README.md) | Self-Evolving Agentic Learning — coreference, query-core extraction, learned-pattern store, reflection |
| [**brainwires-skills**](crates/brainwires-skills/README.md) | SKILL.md skills system — manifest parsing, registry, smart routing, sandboxed execution |
| [**brainwires-eval**](crates/brainwires-eval/README.md) | Evaluation harness — fixtures, regression suites, stability tests, adversarial cases, NDCG / MRR / precision@k |
| [**brainwires-knowledge**](crates/brainwires-knowledge/README.md) | Knowledge layer — BKS / PKS, BrainClient, entity graph |
| [**brainwires-rag**](crates/brainwires-rag/README.md) | Codebase indexing + hybrid retrieval (vector + BM25), AST-aware chunking via tree-sitter, Git history search |
| [**brainwires-prompting**](crates/brainwires-prompting/README.md) | Adaptive prompting — technique library, K-means task clustering, BKS/PKS-aware generator, SEAL feedback hook |
| [**brainwires-storage**](crates/brainwires-storage/README.md) | Substrate — `StorageBackend` trait, 9 backends, embeddings, BM25 keyword search, file-context primitives |
| [**brainwires-stores**](crates/brainwires-stores/README.md) | Schema + CRUD for the opinionated minimum store set: sessions, conversations, tasks, plans, locks, images, templates + tier schema stores |
| [**brainwires-memory**](crates/brainwires-memory/README.md) | Tiered memory **orchestration** — `TieredMemory` multi-factor adaptive search + offline `dream` consolidation engine. Built on `brainwires-stores`. |
| [**brainwires-permission**](crates/brainwires-permission/README.md) | Permission policies (auto, ask, reject) for tool execution |
| [**brainwires-mcp-client**](crates/brainwires-mcp-client/README.md) | MCP client — connect to external MCP servers and use their tools |
| [**brainwires-mcp-server**](crates/brainwires-mcp-server/README.md) | MCP server framework with composable middleware; `http` feature adds stateless HTTP+SSE transport, Server Cards (SEP-1649), RFC9728, and Tasks (SEP-1686); `oauth` feature adds JWT validation middleware |
| [**brainwires-network**](crates/brainwires-network/README.md) | Agent networking — IPC, remote bridge, mesh, WebRTC, LAN discovery |
| [**brainwires-reasoning**](crates/brainwires-reasoning/README.md) | Reasoning scorers — complexity, entity enhancer, relevance, retrieval classifier, router, strategy selector, summarizer, validator |
| [**brainwires-call-policy**](crates/brainwires-call-policy/README.md) | Policies on outbound provider calls — retry with backoff, circuit breaker, budget caps, response cache, error classification |
| [**brainwires-hardware**](crates/brainwires-hardware/README.md) | Hardware I/O — audio (STT/TTS), GPIO, Bluetooth, camera/webcam, raw USB |
| [**brainwires-finetune**](crates/brainwires-finetune/README.md) | Cloud fine-tune APIs (OpenAI, Anthropic, Together, Fireworks, Anyscale, Bedrock, Vertex AI) + dataset pipelines. Local PEFT (LoRA / QLoRA / DoRA), training-from-scratch, and the pure-wgpu Gemma 4 inference engine all live in the sibling `rullama` workspace. |
| [**brainwires-telemetry**](crates/brainwires-telemetry/README.md) | OutcomeMetrics, Prometheus export, anomaly detection, billing-hook trait |
| [**brainwires-a2a**](crates/brainwires-a2a/README.md) | Agent-to-Agent protocol — JSON-RPC 2.0, HTTP/REST, and gRPC bindings |
| [**brainwires-sandbox**](crates/brainwires-sandbox/README.md) | Container-backed sandbox executor for untrusted tool code |
| [**brainwires-sandbox-proxy**](crates/brainwires-sandbox-proxy/README.md) | Out-of-process sandbox-executor proxy for isolating untrusted code |

### Extras

| Crate | Description |
|-------|-------------|
| [**brainwires-proxy**](extras/brainwires-proxy/README.md) | HTTP proxy for AI API request routing |
| [**brainwires-brain-server**](extras/brainwires-brain-server/README.md) | MCP server binary exposing the `brainwires-knowledge::knowledge` subsystem (BKS/PKS, thoughts, entity graphs) |
| [**brainwires-rag-server**](extras/brainwires-rag-server/README.md) | MCP server binary exposing the `brainwires-knowledge::rag` subsystem (codebase indexing + hybrid search) |
| [**agent-chat**](extras/agent-chat/README.md) | Simplified AI chat client with TUI and plain modes |
| [**reload-daemon**](extras/reload-daemon/README.md) | MCP server for killing and restarting AI coding clients |
| [**audio-demo-ffi**](extras/audio-demo-ffi/README.md) | UniFFI bindings (cdylib) exposing brainwires-hardware (audio) to C#, Kotlin, Swift, Python |
| [**audio-demo**](extras/audio-demo/README.md) | Cross-platform Avalonia GUI for TTS/STT demo across all audio providers |
| [**brainclaw**](extras/brainclaw/daemon/README.md) | Self-hosted personal AI assistant daemon — multi-provider, per-user sessions, secure gateway |
| [**brainwires-gateway**](extras/brainclaw/gateway/README.md) | WebSocket/HTTP channel hub — routes channel adapters to AI agent sessions |
| [**brainwires-discord-channel**](extras/brainclaw/mcp-discord/README.md) | Discord channel adapter — reference `Channel` trait implementation, optional MCP server mode |
| [**brainwires-telegram-channel**](extras/brainclaw/mcp-telegram/README.md) | Telegram channel adapter — teloxide-based, optional MCP server mode |
| [**brainwires-slack-channel**](extras/brainclaw/mcp-slack/README.md) | Slack channel adapter — Socket Mode (no public URL), optional MCP server mode |
| [**brainwires-skill-registry**](extras/brainclaw/mcp-skill-registry/README.md) | Skill registry HTTP server — SQLite FTS5, publish/search/download endpoints |
| [**brainclaw-mcp-github**](extras/brainclaw/mcp-github/README.md) | GitHub channel adapter — webhook receiver, REST API, MCP server mode |
| [**brainwires-memory-server**](extras/brainwires-memory-server/README.md) | Mem0-compatible memory REST API backed by Brainwires knowledge |
| [**claude-brain**](extras/claude-brain/README.md) | Brainwires context management for Claude Code — persistent context across compaction |
| [**brainwires-cli**](extras/brainwires-cli/README.md) | AI-powered agentic CLI tool for autonomous coding assistance |
| [**brainwires-issues**](extras/brainwires-issues/README.md) | MCP-native issue tracking server |
| [**brainwires-scheduler**](extras/brainwires-scheduler/README.md) | MCP server for cron scheduling |
| [**brainwires-autonomy**](extras/brainwires-autonomy/README.md) | Autonomous agent operations |
| [**brainwires-wasm**](extras/brainwires-wasm/README.md) | WASM browser bindings |
| [**brainwires-billing**](extras/brainwires-billing/README.md) | Billing and cost accounting hooks for agent telemetry |
| [**brainwires-docs**](extras/brainwires-docs/README.md) | Documentation tooling and reference site generation |
| [**voice-assistant**](extras/voice-assistant/README.md) | End-to-end voice assistant binary using the `brainwires-hardware` pipeline |

### Workspace layout

- **`crates/`** — the framework. Cohesive, independently-publishable libraries.
- **`extras/`** — applications and libraries that **consume** the framework: binaries, demos, MCP servers, and integration helpers.

**Allowed dependency arrows:** `crates/ → crates/` and `extras/ → crates/`.

### Brands, repos, and the engine/harness boundary

**brainwires is the open-source platform**; **[rullama](../rullama) is the app**
(`rullama.com`) that runs on it. Two names, one downward dependency:

- The platform holds both the inference **engine** (`brainwires-engine` — the
  Rust → WASM + WebGPU inference path, moving in from the old `rullama` crate) and
  the agent **harness** (the `brainwires-*` crates here). They stay separate,
  joined by the `Provider` seam; the engine is a first-party WebGPU provider.
- The **rullama product family** (the PWA, `rullama-native` — a shipping paid
  .NET/Avalonia desktop+mobile app, and `rullama-cli`) consumes the platform
  three ways: in-browser via the engine's wasm bundle, natively via an
  OpenAI-compatible `/v1/chat/completions` endpoint (existing `openai_chat`
  provider, base-URL swap), and in-process via a C-ABI shim (rullama-native). The
  PWA supersedes the old `brainwires-studio` and the Candle
  `extras/brainwires-chat-pwa` (both retire).
- **brainclaw** is extracting to its own product repo, and **brainwires-cli** is
  extracting *and being renamed `rullama-cli`* (it joins the rullama product
  family — app + CLI). Both depend on published `brainwires` crates.

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

The simplest way to use the framework is through the `brainwires` facade crate, which re-exports everything behind feature flags:

```toml
[dependencies]
brainwires = "0.11"  # defaults: tools + agents
```

Enable only what you need:

```toml
[dependencies]
brainwires = { version = "0.11", features = ["provider", "rag"] }
```

### Using Individual Crates

Each crate is independently publishable and usable:

```toml
[dependencies]
brainwires-core = "0.11"
brainwires-provider = "0.11"
brainwires-agent = "0.11"
```

### Minimal Example

```rust
use brainwires::prelude::*;
use brainwires::providers::{ChatProviderFactory, ProviderConfig, ProviderType};

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

The `brainwires` facade crate exposes feature flags corresponding to each sub-crate:

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `core` | Always | Core types and traits (not feature-gated) |
| `tools` | Yes | Tool definitions, execution, and interpreters (`brainwires-tool-runtime` + `brainwires-tool-builtins`) |
| `agents` | Yes | Multi-agent orchestration, communication hub, file/resource locks (`brainwires-agent`) |
| `inference` | Yes | LLM-driven workhorses — ChatAgent, PlannerAgent, JudgeAgent, TaskAgent, CycleOrchestrator (`brainwires-inference`) |
| `providers` | No | AI provider integrations |
| `storage` | No | Vector storage and semantic search |
| `mcp` | No | MCP client support |
| `agent-network` | No | Agent networking — IPC, remote bridge, channels, 5-layer protocol stack (`brainwires-network`) |
| `mcp-server-framework` | No | MCP server building blocks (McpServer, McpHandler, middleware pipeline) |
| `rag` | No | RAG engine with code search |
| `audio` | No | Audio capture, STT, TTS |
| `training` | No | Cloud fine-tuning (local PEFT lives in `rullama-finetune`) |
| `datasets` | No | Training data pipelines (delegates to `brainwires-finetune`) |
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
cargo build -p brainwires-agent

# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p brainwires-core
```

## Dependency DAG

```text
  brainwires (facade)
  ├── brainwires-agent
  │   ├── brainwires-core
  │   ├── brainwires-call-policy
  │   ├── brainwires-tool-runtime
  │   ├── brainwires-tool-builtins
  │   ├── brainwires-storage (seal feature — for PatternStore)
  │   ├── brainwires-knowledge (seal-knowledge feature)
  │   └── brainwires-permission (seal-feedback feature)
  ├── brainwires-knowledge
  │   ├── brainwires-core
  │   └── brainwires-storage
  ├── brainwires-rag
  │   ├── brainwires-core
  │   └── brainwires-storage
  ├── brainwires-prompting
  │   ├── brainwires-core
  │   └── brainwires-knowledge (knowledge feature)
  ├── brainwires-storage
  │   └── brainwires-core
  ├── brainwires-stores
  │   ├── brainwires-core
  │   └── brainwires-storage
  ├── brainwires-memory
  │   ├── brainwires-core
  │   ├── brainwires-storage
  │   └── brainwires-stores (memory feature)
  ├── brainwires-tool-runtime
  │   ├── brainwires-core
  │   ├── brainwires-stores (sessions feature — SessionBroker)
  │   ├── brainwires-rag (rag feature)
  │   └── brainwires-sandbox (sandbox feature)
  ├── brainwires-tool-builtins
  │   ├── brainwires-tool-runtime
  │   └── brainwires-rag (rag feature)
  ├── brainwires-mcp-client
  │   └── brainwires-core
  ├── brainwires-mcp-server
  │   ├── brainwires-core
  │   └── brainwires-mcp-client
  ├── brainwires-network
  │   ├── brainwires-core
  │   ├── brainwires-mcp-client
  │   └── brainwires-a2a (a2a-transport feature)
  ├── brainwires-finetune          (cloud only — local PEFT + wgpu Gemma 4 live in rullama)
  │   ├── brainwires-core
  │   └── brainwires-provider (cloud feature)
  ├── brainwires-telemetry
  │   └── brainwires-core
  └── brainwires-hardware
      ├── brainwires-provider (audio feature)
      └── brainwires-provider-speech (audio feature)
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
