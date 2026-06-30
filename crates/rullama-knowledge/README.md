# rullama-knowledge

[![Crates.io](https://img.shields.io/crates/v/rullama-knowledge.svg)](https://crates.io/crates/rullama-knowledge)
[![Documentation](https://img.shields.io/docsrs/rullama-knowledge)](https://docs.rs/rullama-knowledge)
[![License](https://img.shields.io/crates/l/rullama-knowledge.svg)](LICENSE)

**Knowledge layer for [rullama](https://github.com/Brainwires/rullama-framework)** —
knowledge graphs, behavioral/personal knowledge systems (BKS/PKS), a brain client
for persistent thoughts, and entity extraction.

> This crate is now **just the knowledge subsystem**. The prompting, RAG,
> spectral, and code-analysis subsystems that used to live here were split into
> dedicated crates:
> - Adaptive prompting → [`rullama-prompting`](../rullama-prompting)
> - RAG / hybrid retrieval / spectral / code-analysis → [`rullama-rag`](../rullama-rag)
> - Offline memory consolidation ("dream") → [`rullama-stores`](../rullama-stores) (`dream` feature)
>
> Depend on those crates directly for that functionality.

## What it provides

- **BrainClient** — persistent thought storage with semantic search.
- **Entity & Relationship Graph** — entity types, co-occurrence, and impact
  analysis over a knowledge graph.
- **BKS** (Behavioral Knowledge System) — shared truths with confidence scoring.
- **PKS** (Personal Knowledge System) — user-scoped facts.
- **Thoughts & fact extraction** — `Thought` / `ThoughtCategory` / `ThoughtSource`
  with automatic categorization and tag extraction.

## Quick start

```toml
[dependencies]
rullama-knowledge = "0.12"
```

```rust
use rullama_knowledge::{BrainClient, Thought, ThoughtCategory};
```

Key exports: `BrainClient`, `Thought` / `ThoughtCategory` / `ThoughtSource`,
`DispositionTrait`, `MemoryBankConfig`, plus the entity and relationship-graph
types (and a `prelude`).

## Features

| Feature | Enables |
|---|---|
| `knowledge` *(default)* | knowledge graph, entities, thoughts, brain client (native persistence via rusqlite + reqwest) |
| `telemetry` | DreamCycle event recording via `rullama-telemetry` |
| `native` | `knowledge` + `telemetry` |
| `wasm` | lightweight WASM build (`rullama-core/wasm`) |
| `alt-folder-name` | alternative on-disk storage folder name |

## License

Licensed under either of MIT or Apache-2.0 at your option.
