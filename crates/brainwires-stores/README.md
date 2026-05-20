# brainwires-stores

[![Crates.io](https://img.shields.io/crates/v/brainwires-stores.svg)](https://crates.io/crates/brainwires-stores)
[![Documentation](https://docs.rs/brainwires-stores/badge.svg)](https://docs.rs/brainwires-stores)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Schema + CRUD for the Brainwires Agent Framework's opinionated
minimum data-store set.

## What this crate is

The framework is opinionated about a small set of stores every agent
system needs — sessions, conversations, tasks, plans, locks, images,
templates, and the hot/warm/cold memory tiers. This crate defines them
all as **schema + CRUD only**, generic over the
[`brainwires_storage::StorageBackend`] trait so you can swap backends
(LanceDB, in-memory, your own) without touching store code.

It deliberately contains no orchestration, no engines, no
pipelines — just rows and the operations that move them in and out.
Multi-tier search, promotion / demotion, and offline consolidation
live in the separate
[`brainwires-memory`](https://crates.io/crates/brainwires-memory)
crate, which depends on the schema types here.

## Features

| Flag           | Default | What it pulls in                                                                                  |
|----------------|---------|---------------------------------------------------------------------------------------------------|
| `session`      | yes     | `SessionStore` trait + `InMemorySessionStore` + (with `sqlite`) `SqliteSessionStore`              |
| `task`         | yes     | `TaskStore` + `AgentStateStore`                                                                   |
| `plan`         | yes     | `PlanStore` + `TemplateStore`                                                                     |
| `conversation` | yes     | `ConversationStore` (catalog metadata: id, title, model, message count)                           |
| `memory`       | no      | tier schema stores: `MessageStore`, `SummaryStore`, `FactStore`, `MentalModelStore`, `TierMetadataStore` + `tier_types` |
| `lock`         | no      | `LockStore` (rusqlite-backed coordination locks)                                                  |
| `image`        | no      | `ImageStore` with sha256 hashing                                                                  |
| `sqlite`       | no      | rusqlite backend for `SqliteSessionStore` and `LockStore`                                         |

`default = ["session", "task", "plan", "conversation"]` covers the most
common agent-runtime needs. Set `default-features = false` and opt
into `memory` / `lock` / `image` only when you need them.

## How the three storage layers fit together

```
brainwires-storage    StorageBackend trait, backends (LanceDB, …),
                      embeddings, BM25, file-context primitives.
                      ─── substrate ───
              ▲
              │
brainwires-stores     Row schemas + CRUD for the opinionated minimum
                      set (this crate). Built on the substrate.
                      ─── schema ───
              ▲
              │
brainwires-memory     TieredMemory orchestration over the tier
                      schema stores; multi-factor adaptive search;
                      offline `dream` consolidation engine.
                      ─── orchestration ───
```

## SessionStore example

```rust
use brainwires_stores::{InMemorySessionStore, SessionId, SessionStore};
use brainwires_core::Message;

let store = InMemorySessionStore::new();
let id = SessionId::from("user-123");
store.save(&id, &[Message::user("hello")]).await?;
let transcript = store.load(&id).await?;
```

## License

MIT OR Apache-2.0
