# Brainwires Framework — Crate Dependency Tree

Crates organized in layers. Arrows (`->`) indicate internal dependencies. For
standalone apps built on the framework, see [`extras/`](../extras/README.md).

```
brainwires  (facade — re-exports every framework crate via feature flags)
│
├─── Foundation (no internal deps)
│    └── brainwires-core               Core types, traits, messages, tools, tasks, embeddings
│
├─── Substrate
│    ├── brainwires-storage            StorageBackend trait, 9 backends, embeddings, BM25, file context
│    │   └─> core
│    └── brainwires-telemetry          OutcomeMetrics, Prometheus export, anomaly detection, billing-hook trait
│        └─> core
│
├─── Provider + call policy
│    ├── brainwires-call-policy        Retry, circuit breaker, budget caps, response cache, error classification
│    │   └─> core
│    ├── brainwires-provider           LLM clients (Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex AI, llama.cpp, Candle)
│    │   └─> core
│    │   └─> telemetry (opt, "telemetry" feature)
│    ├── brainwires-provider-speech    Speech (TTS / STT) clients (Azure, Cartesia, Deepgram, ElevenLabs, Fish, Google, Murf, web-speech)
│    │   └─> core
│    └── brainwires-hardware           Audio, GPIO, Bluetooth, camera, USB, Matter, home automation
│        └─> core
│        └─> provider (opt, "audio" feature)
│        └─> provider-speech (opt, "audio" feature)
│
├─── Stores (schema + CRUD)
│    └── brainwires-stores             Sessions, conversations, tasks, plans, locks, images, templates, tier schemas
│        └─> core
│        └─> storage
│
├─── Memory orchestration
│    └── brainwires-memory             TieredMemory adaptive search + dream offline consolidation
│        └─> core
│        └─> storage
│        └─> stores ("memory" feature)
│        └─> telemetry (opt, "telemetry" feature)
│
├─── Protocols
│    ├── brainwires-mcp-client         MCP client (rmcp-backed)
│    │   └─> core
│    ├── brainwires-mcp-server         MCP server framework with middleware; opt HTTP+SSE, OAuth
│    │   └─> core
│    │   └─> mcp-client (shared protocol types)
│    └── brainwires-a2a                Agent-to-Agent protocol (JSON-RPC, REST, gRPC)
│        └─> core
│
├─── Intelligence
│    ├── brainwires-knowledge          BKS / PKS, BrainClient, entity graph
│    │   └─> core
│    │   └─> storage
│    ├── brainwires-rag                Codebase indexing + hybrid retrieval (vector + BM25), AST chunking, Git history
│    │   └─> core
│    │   └─> storage
│    └── brainwires-prompting          Adaptive prompting — technique library, K-means clustering, BKS/PKS-aware generator
│        └─> core
│        └─> knowledge (opt, "knowledge" feature)
│
├─── Tools
│    ├── brainwires-tool-runtime       ToolExecutor, ToolRegistry, validation, smart router, sessions, oauth, openapi
│    │   └─> core
│    │   └─> stores (opt, "sessions" feature — pulls SessionBroker)
│    │   └─> rag (opt, "rag" feature)
│    │   └─> sandbox (opt, "sandbox" feature)
│    └── brainwires-tool-builtins      Concrete tools: bash, file_ops, git, web, search, code_exec, browser, email, calendar, system, semantic_search
│        └─> tool-runtime
│        └─> rag (opt, "rag" feature)
│
├─── Sandbox
│    ├── brainwires-sandbox            Container-backed sandbox executor
│    │   └─> core
│    └── brainwires-sandbox-proxy      Out-of-process sandbox-executor proxy
│        └─> core
│        └─> sandbox
│
├─── Permissions
│    └── brainwires-permission         Permission policies, audit logging, trust profiles
│        └─> core
│
├─── Reasoning
│    └── brainwires-reasoning          Planners, validators, routers, strategies, scorers, output parsers
│        └─> core
│        └─> tool-runtime (uses ToolCategory in router.rs)
│
├─── Agency
│    ├── brainwires-agent              Agent runtime, communication hub, task decomposition, MDAP, SEAL (with PatternStore), skills, eval
│    │   └─> core
│    │   └─> call-policy
│    │   └─> tool-runtime
│    │   └─> tool-builtins
│    │   └─> storage (opt, "seal" feature — for PatternStore)
│    │   └─> knowledge (opt, "seal-knowledge" feature)
│    │   └─> permission (opt, "seal-feedback" feature)
│    └── brainwires-network            IPC, TCP, remote bridge, 5-layer protocol stack, mesh, LAN discovery
│        └─> core
│        └─> mcp-client
│        └─> a2a (opt, "a2a-transport" feature)
│
└─── Fine-tuning + training
     └── brainwires-finetune           Cloud fine-tune APIs (OpenAI, Anthropic, Together, Fireworks, Anyscale, Bedrock, Vertex AI) + dataset pipelines
         └─> core
         └─> provider (opt, "cloud" feature)

Local PEFT (LoRA / QLoRA / DoRA) and training-from-scratch live in the
sibling `rullama` workspace as `rullama-finetune` and `rullama-training`
since v0.11 — they used to live here as separate local-training crates,
moved out alongside the rest of the wgpu inference engine.
```

## Three-layer storage architecture

```
brainwires-storage    substrate (StorageBackend trait, backends)
        ▲
        │
brainwires-stores     schema + CRUD (sessions, tasks, plans,
                      conversations, locks, images, tier rows)
        ▲
        │
brainwires-memory     orchestration (TieredMemory, dream)
```

`brainwires-stores` is the framework's **opinionated minimum store
set** — schema only, generic over `StorageBackend`. Anyone building an
agent system on the framework gets a coherent set of primitives
without having to invent or copy them. See ADR-0005.

## Longest Dependency Chain

With the `seal` and `dream` features active, the longest leaf-to-leaf
chain is 5 hops:

```
core -> storage -> stores -> memory -> agent -> brainwires
core -> storage -> stores -> memory       ↑    (facade)
core -> storage -> rag -> tool-builtins ──┘
```

Without the optional features the chain collapses to
`core -> tool-runtime -> agent` for the default agent build.

## Feature Presets (facade crate)

See [`crates/brainwires/README.md`](brainwires/README.md) for the full
feature table. Convenience presets:

| Preset | Includes |
|--------|----------|
| `agent-full` | agents, permission, prompting, tools |
| `researcher` | provider, agents, storage, rag, training, datasets |
| `learning` | seal, knowledge, permission, seal-knowledge, seal-feedback |
| `full` | everything |
