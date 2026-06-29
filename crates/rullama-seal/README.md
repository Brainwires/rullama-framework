# SEAL — Self-Evolving Agentic Learning

This module (inside `rullama-agent`) implements the SEAL framework for enhancing conversational question answering and agent decision-making. It provides coreference resolution, structured query extraction, self-evolving pattern learning, and post-execution reflection — enabling agents to understand implicit references, build reusable knowledge, and correct their own mistakes without retraining.

> Inspired by: **SEAL: Self-Evolving Agentic Learning for Conversational Question Answering over Knowledge Graphs** (Wang et al., arXiv:2512.04868, December 2024)

## Crate boundary

SEAL spent part of the 0.10 cycle folded into `rullama-agent` behind a `seal` feature flag; in 0.11 it moved back out into the standalone `rullama-seal` crate. The dependencies it needs — `ResponseConfidence` (now in `rullama-core`), `ToolOutcome` / `ToolErrorCategory` (in `rullama-tool-runtime`), `RelationshipGraph` (in `rullama-knowledge`) — are all addressable from a leaf crate, so the standalone shape is back to being the right one. Optional integrations use the `knowledge`, `feedback`, and `mdap` features.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `seal` | Core SEAL pipeline (coreference, query extraction, learning, reflection) |
| `seal-mdap` | MDAP metric recording via `mdap` feature |
| `seal-knowledge` | BKS/PKS knowledge system integration via `rullama-knowledge` |
| `seal-feedback` | Audit feedback bridge via `rullama-permission` |

```toml
# Core SEAL
rullama-seal = "0.11"

# With knowledge integration
rullama-seal = { version = "0.11", features = ["knowledge"] }

# Via the rullama facade
rullama = { version = "0.11", features = ["seal"] }
```

## Architecture

```text
User Query
    │
    ▼
┌─── Coreference Resolution ─────────────────────────────────────┐
│  detect_references() → resolve() → rewrite_with_resolutions()  │
│  "What uses it?" → "What uses [main.rs]?"                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─── Query Core Extraction ──────────────────────────────────────┐
│  classify() → build_expression() → QueryCore                   │
│  S-expression: (JOIN DependsOn ?dep "main.rs")                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─── Learning Coordinator ───────────────────────────────────────┐
│  Local Memory (per-session)  │  Global Memory (cross-session)  │
│  process_query() → match pattern or create new                 │
│  record_outcome() → update reliability scores                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─── Reflection Module ──────────────────────────────────────────┐
│  analyze() → detect issues → suggest fixes → attempt_correction│
│  Errors: EmptyResult, Overflow, EntityNotFound, RelationMismatch│
└────────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust,ignore
use rullama_agent::seal::{SealProcessor, SealConfig, DialogState};
use rullama_core::graph::{EntityStoreT, RelationshipGraphT};

let mut processor = SealProcessor::with_defaults();
processor.init_conversation("session-001");

let result = processor.process(
    "What uses it?",
    &dialog_state,
    &entity_store,
    Some(&graph),
)?;

println!("Resolved: {}", result.resolved_query);
```

## Components

- **`SealProcessor`** — Main orchestrator chaining all pipeline stages
- **`CoreferenceResolver`** — Salience-weighted anaphora resolution
- **`QueryCoreExtractor`** — NL → structured S-expression queries
- **`LearningCoordinator`** — Dual-level memory (local + global) with pattern learning
- **`ReflectionModule`** — Post-execution error detection and correction
- **`SealKnowledgeCoordinator`** — BKS/PKS bidirectional bridge (requires `seal-knowledge`)
- **`FeedbackBridge`** — Audit log → learning signal converter (requires `seal-feedback`)
