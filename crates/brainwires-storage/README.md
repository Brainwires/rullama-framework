# brainwires-storage

[![Crates.io](https://img.shields.io/crates/v/brainwires-storage.svg)](https://crates.io/crates/brainwires-storage)
[![Documentation](https://img.shields.io/docsrs/brainwires-storage)](https://docs.rs/brainwires-storage)
[![License](https://img.shields.io/crates/l/brainwires-storage.svg)](LICENSE)

Backend-agnostic storage primitives — `StorageBackend` trait, embeddings,
BM25 keyword search, file-context primitives, and image-storage types —
for the Brainwires Agent Framework.

> **Crate boundary (v0.11).** Domain-shaped stores moved out of this
> crate so the surface stays focused on primitives:
>
> - **`brainwires-stores`** — schema + CRUD for the opinionated minimum
>   store set: `MessageStore`, `SummaryStore`, `FactStore`,
>   `MentalModelStore`, `TierMetadataStore`, `PlanStore`,
>   `TemplateStore` / `PlanTemplate`, `LockStore` / `LockRecord` /
>   `LockStats`, `ImageStore`, `ConversationStore`, `TaskStore`,
>   `AgentStateStore`, `PersistentTaskManager`.
> - **`brainwires-memory`** — tiered hot/warm/cold agent memory
>   orchestration: `TieredMemory`, `TieredMemoryConfig`,
>   `TieredMemoryStats`, `MultiFactorScore`, `CanonicalWriteToken`,
>   `TieredSearchResult`, dream consolidation.

## Overview

`brainwires-storage` is the persistent-backend foundation: a generic
trait-and-types layer that other crates build their domain stores on
top of. It provides the `StorageBackend` and `VectorDatabase` traits
plus their concrete backend impls (LanceDB, Postgres, MySQL, Surreal,
Qdrant, Pinecone, Milvus, Weaviate, NornicDB), text embeddings via
FastEmbed (LRU-cached), BM25 keyword search, file-context primitives,
and image-storage types reused by the concrete `ImageStore` in
`brainwires-cli`.

**Design principles:**

- **Backend-agnostic** — domain stores are generic over `StorageBackend`; swap databases by changing a feature flag, not your application code
- **One struct, one connection** — each database backend is a single struct (e.g. `LanceDatabase`, `PostgresDatabase`) that implements one or both core traits and shares a single connection for all operations
- **Semantic-first retrieval** — all stores embed content via all-MiniLM-L6-v2 (384 dimensions) and search by vector similarity, so queries match meaning rather than keywords
- **Hybrid search** — document retrieval combines vector similarity with BM25 keyword scoring via Reciprocal Rank Fusion (RRF) for best-of-both-worlds accuracy
- **Three-tier memory** — hot (full messages with TTL), warm (compressed summaries), cold (extracted facts) with automatic demotion/promotion based on importance and access patterns
- **Memory safety** — contradiction detection flags conflicting facts for human review; canonical write tokens gate long-lived writes; session TTL auto-expires ephemeral data
- **Cross-process coordination** — SQLite-backed locks with WAL mode, stale lock detection via PID/hostname, and automatic cleanup for multi-instance deployments
- **Feature-gated portability** — pure types and logic compile everywhere; native-only modules (LanceDB, Arrow, SQLite) are behind the `native` feature for WASM compatibility

```text
  +-----------------------------------------------------------------------+
  |                        brainwires-storage                              |
  |                                                                        |
  |  +--- Unified Database Layer (databases/) -------------------------+  |
  |  |                                                                  |  |
  |  |  Core Traits:                                                    |  |
  |  |    StorageBackend --- generic CRUD + vector search               |  |
  |  |    VectorDatabase --- RAG embedding storage + hybrid search      |  |
  |  |                                                                  |  |
  |  |  Backends:                                                       |  |
  |  |    LanceDatabase ---- StorageBackend + VectorDatabase (default)  |  |
  |  |    PostgresDatabase - StorageBackend + VectorDatabase            |  |
  |  |    MySqlDatabase ---- StorageBackend only                        |  |
  |  |    SurrealDatabase -- StorageBackend + VectorDatabase            |  |
  |  |    QdrantDatabase --- VectorDatabase only                        |  |
  |  |    PineconeDatabase - VectorDatabase only                        |  |
  |  |    MilvusDatabase --- VectorDatabase only                        |  |
  |  |    WeaviateDatabase - VectorDatabase only                        |  |
  |  |    NornicDatabase --- VectorDatabase only                        |  |
  |  |                                                                  |  |
  |  |  Supporting modules:                                             |  |
  |  |    types.rs --- Record, FieldDef, Filter, ScoredRecord           |  |
  |  |    capabilities.rs --- BackendCapabilities discovery              |  |
  |  |    sql/ --- shared SQL dialect layer|  |
  |  |    bm25_helpers.rs --- BM25 scoring for client-side keyword search|  |
  |  +------------------------------------------------------------------+  |
  |                                                                        |
  |  +--- Core Infrastructure -------------------------------------------+  |
  |  |  CachedEmbeddingProvider -- all-MiniLM-L6-v2 w/ LRU cache (1000) |  |
  |  +------------------------------------------------------------------+  |
  |                                                                        |
  |  +--- Domain Stores (moved out of this crate) ----------------------+  |
  |  |  See `brainwires-stores`  --- MessageStore, SummaryStore,         |  |
  |  |                                FactStore, MentalModelStore,       |  |
  |  |                                TierMetadataStore, PlanStore,      |  |
  |  |                                TemplateStore, LockStore,          |  |
  |  |                                ImageStore, ConversationStore,     |  |
  |  |                                TaskStore                          |  |
  |  |  See `brainwires-memory`  --- TieredMemory orchestration,         |  |
  |  |                                multi-factor search, dream cycles  |  |
  |  +------------------------------------------------------------------+  |
  |                                                                        |
  |  +--- Document Management ------------------------------------------+  |
  |  |  DocumentProcessor --- PDF, DOCX, Markdown, plain text            |  |
  |  |  DocumentChunker --- paragraph/sentence-aware segmentation        |  |
  |  |  DocumentStore --- hybrid search (vector + BM25 via RRF)         |  |
  |  |  DocumentMetadataStore --- hash-based deduplication               |  |
  |  +------------------------------------------------------------------+  |
  |  Note: EntityStore and RelationshipGraph moved to brainwires-knowledge |
  +------------------------------------------------------------------------+
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-storage = "0.11"
# Domain store types (MessageStore, LockStore, TemplateStore, …) live here:
brainwires-stores = { version = "0.11", features = ["memory"] }
```

Store and search conversation messages:

```rust
use brainwires_storage::{LanceDatabase, CachedEmbeddingProvider};
use brainwires_stores::{MessageStore, MessageMetadata};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize storage — one struct, one connection
    let db = Arc::new(LanceDatabase::new("~/.brainwires/db").await?);
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    db.initialize(embeddings.dimension()).await?;

    let store = MessageStore::new(db.clone(), embeddings.clone());

    // Store a message
    store.add(MessageMetadata {
        message_id: "msg-001".into(),
        conversation_id: "conv-001".into(),
        role: "assistant".into(),
        content: "The auth module uses JWT tokens with RS256 signing".into(),
        token_count: Some(42),
        model_id: Some("claude-opus-4-6".into()),
        images: None,
        created_at: chrono::Utc::now().timestamp(),
        expires_at: None,
    }).await?;

    // Semantic search across all conversations
    let results = store.search("how does authentication work?", 5, 0.7).await?;
    for (msg, score) in &results {
        println!("[{:.2}] {}: {}", score, msg.role, msg.content);
    }

    Ok(())
}
```

## Database Backends

The `databases/` module provides a unified abstraction layer. Each database is a single struct implementing one or both core traits:

- **`StorageBackend`** — generic CRUD + vector search for domain stores (messages, conversations, tasks, etc.)
- **`VectorDatabase`** — RAG-style embedding storage with hybrid search for codebase indexing

### Trait implementation matrix

| Database | Struct | `StorageBackend` | `VectorDatabase` | Feature flag |
|----------|--------|:---:|:---:|--------------|
| LanceDB | `LanceDatabase` | YES | YES | `lance-backend` (default via `native`) |
| PostgreSQL + pgvector | `PostgresDatabase` | YES | YES | `postgres-backend` |
| MySQL / MariaDB | `MySqlDatabase` | YES | NO | `mysql-backend` |
| SurrealDB | `SurrealDatabase` | YES | YES | `surrealdb-backend` |
| Qdrant | `QdrantDatabase` | NO | YES | `qdrant-backend` |
| Pinecone | `PineconeDatabase` | NO | YES | `pinecone-backend` |
| Milvus | `MilvusDatabase` | NO | YES | `milvus-backend` |
| Weaviate | `WeaviateDatabase` | NO | YES | `weaviate-backend` |
| NornicDB | `NornicDatabase` | NO | YES | `nornicdb-backend` |

### Connection sharing

Backends that implement both traits share a single connection. This means domain stores and the RAG subsystem can use the same database instance without opening separate connections:

```rust
use brainwires_storage::{LanceDatabase, StorageBackend};
use brainwires_storage::databases::VectorDatabase;
use std::sync::Arc;

let db = Arc::new(LanceDatabase::new("/path/to/db").await?);

// Domain stores use the StorageBackend trait
let messages = MessageStore::new(db.clone(), embeddings.clone());
let conversations = ConversationStore::new(db.clone());

// RAG system uses the VectorDatabase trait on the same connection
let rag = RagClient::with_vector_db(db.clone());
```

### Module structure

```text
databases/
  mod.rs              -- top-level module, re-exports
  traits.rs           -- StorageBackend + VectorDatabase trait definitions
  types.rs            -- Record, FieldDef, FieldValue, Filter, ScoredRecord
  capabilities.rs     -- BackendCapabilities runtime discovery
  bm25_helpers.rs     -- shared BM25 scoring for client-side keyword search
  sql/                -- shared SQL dialect layer
    mod.rs            -- SqlDialect trait
    postgres.rs       -- PostgreSQL dialect
    mysql.rs          -- MySQL dialect
    surrealdb.rs      -- SurrealDB dialect
  lance/              -- LanceDB backend (default)
    mod.rs            -- LanceDatabase struct
    arrow_convert.rs  -- Arrow <-> Record conversion
  postgres/           -- PostgreSQL + pgvector backend
  mysql/              -- MySQL / MariaDB backend
  surrealdb/          -- SurrealDB backend (MTREE vector search)
  qdrant/             -- Qdrant backend
  pinecone/           -- Pinecone backend
  milvus/             -- Milvus backend
  weaviate/           -- Weaviate backend
  nornicdb/           -- NornicDB backend
    mod.rs            -- NornicDatabase struct
    transport.rs      -- REST/Bolt/gRPC transport layer
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `native` | Yes | Enables LanceDB backend, FastEmbed, SQLite locks, and all native-only stores |
| `lance-backend` | Yes (via `native`) | LanceDB embedded vector database |
| `postgres-backend` | No | PostgreSQL + pgvector (both traits) |
| `mysql-backend` | No | MySQL / MariaDB (StorageBackend only) |
| `surrealdb-backend` | No | SurrealDB with native MTREE vector search (both traits) |
| `qdrant-backend` | No | Qdrant vector search server |
| `pinecone-backend` | No | Pinecone managed cloud vectors |
| `milvus-backend` | No | Milvus open-source vectors |
| `weaviate-backend` | No | Weaviate vector search engine |
| `nornicdb-backend` | No | NornicDB graph + vector database |
| `nornicdb-bolt` | No | NornicDB with Neo4j Bolt transport |
| `nornicdb-grpc` | No | NornicDB with Qdrant-compatible gRPC |
| `nornicdb-full` | No | NornicDB with all transports |
| `wasm` | No | WASM-compatible compilation (pure types only) |

```toml
# Default (LanceDB + full native functionality)
brainwires-storage = "0.11"

# WASM-compatible (pure types and logic only)
brainwires-storage = { version = "0.11", default-features = false, features = ["wasm"] }

# With Qdrant backend (in addition to LanceDB)
brainwires-storage = { version = "0.11", features = ["qdrant-backend"] }

# PostgreSQL as primary backend
brainwires-storage = { version = "0.11", features = ["postgres-backend"] }

# MySQL / MariaDB backend
brainwires-storage = { version = "0.11", features = ["mysql-backend"] }

# SurrealDB backend (native vector search)
brainwires-storage = { version = "0.11", features = ["surrealdb-backend"] }

# NornicDB with all transports
brainwires-storage = { version = "0.11", features = ["nornicdb-full"] }
```

**Module availability by feature:**

| Module | Always | `native` | Backend-specific |
|--------|--------|----------|------------------|
| `databases` (traits, types, capabilities) | Yes | -- | -- |
| `image_types` | Yes | -- | -- |
| `template_store` | Yes | -- | -- |
| `databases::lance` | -- | Yes | `lance-backend` |
| `databases::qdrant` | -- | -- | `qdrant-backend` |
| `databases::postgres` | -- | -- | `postgres-backend` |
| `databases::mysql` | -- | -- | `mysql-backend` |
| `databases::surrealdb` | -- | -- | `surrealdb-backend` |
| `databases::pinecone` | -- | -- | `pinecone-backend` |
| `databases::milvus` | -- | -- | `milvus-backend` |
| `databases::weaviate` | -- | -- | `weaviate-backend` |
| `databases::nornicdb` | -- | -- | `nornicdb-backend` |
| `databases::sql` | -- | -- | any SQL backend |
| `embeddings` | -- | Yes | -- |
| `message_store`, `conversation_store` | -- | Yes | -- |
| `task_store`, `plan_store`, `lock_store` | -- | Yes | -- |
| `document_store`, `document_processor` | -- | Yes | -- |
| `image_store` | -- | Yes | -- |
| `tiered_memory`, `summary_store`, `fact_store` | -- | Yes | -- |
| `tier_metadata_store`, `file_context` | -- | Yes | -- |
| `bm25_search`, `glob_utils`, `paths` | -- | -- | `lance-backend` |

## Architecture

### StorageBackend trait

Backend-agnostic storage operations. Domain stores are generic over this trait.

| Method | Description |
|--------|-------------|
| `ensure_table(name, schema)` | Ensure table exists (idempotent) |
| `insert(table, records)` | Insert one or more records |
| `query(table, filter, limit)` | Query with optional filter |
| `delete(table, filter)` | Delete matching records |
| `count(table, filter)` | Count matching records |
| `vector_search(table, column, vector, limit, filter)` | Vector similarity search |

### VectorDatabase trait

RAG-style embedding storage used by the codebase indexing subsystem.

| Method | Description |
|--------|-------------|
| `initialize(dimension)` | Initialize collections |
| `store_embeddings(embeddings, metadata, contents, root_path)` | Store embeddings with metadata |
| `search(vector, text, limit, min_score, project, root, hybrid)` | Vector/hybrid search |
| `search_filtered(...)` | Search with extension/language/path filters |
| `search_with_embeddings(...)` | Search returning raw embedding vectors |
| `delete_by_file(path)` | Delete embeddings for a file |
| `clear()` | Clear all embeddings |
| `get_statistics()` | Get storage statistics |
| `flush()` | Flush changes to disk |
| `count_by_root_path(root)` | Count embeddings per project |
| `get_indexed_files(root)` | List indexed file paths |

### CachedEmbeddingProvider

Text embedding with LRU caching, backed by FastEmbed (all-MiniLM-L6-v2, 384 dimensions). Implements the `brainwires_core::EmbeddingProvider` trait.

| Method | Description |
|--------|-------------|
| `new()` | Create provider with default model |
| `embed(text)` | Embed single text -> `Vec<f32>` |
| `embed_cached(text)` | Embed with LRU cache (1000 entries) -> `Vec<f32>` |
| `embed_batch(texts)` | Embed multiple texts -> `Vec<Vec<f32>>` |
| `dimension()` | Get embedding dimension (384) |
| `cache_len()` | Get current cache size |
| `clear_cache()` | Clear the LRU cache |

### MessageStore

Conversation messages with vector search and TTL expiry support.

| Method | Description |
|--------|-------------|
| `new(client, embeddings)` | Create store |
| `add(message)` | Add a single message |
| `add_batch(messages)` | Add multiple messages |
| `get(message_id)` | Get message by ID |
| `get_by_conversation(conversation_id)` | Get all messages in a conversation |
| `search(query, limit, min_score)` | Semantic search across all messages |
| `search_conversation(conversation_id, query, limit, min_score)` | Search within a conversation |
| `delete(message_id)` | Delete a single message |
| `delete_by_conversation(conversation_id)` | Delete all messages in a conversation |
| `delete_expired()` | Delete TTL-expired messages -> count |

**`MessageMetadata`:**

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | `String` | Unique message identifier |
| `conversation_id` | `String` | Parent conversation |
| `role` | `String` | Message role (user, assistant, system) |
| `content` | `String` | Message content |
| `token_count` | `Option<i32>` | Token count estimate |
| `model_id` | `Option<String>` | Model that generated the message |
| `images` | `Option<String>` | JSON-encoded image references |
| `created_at` | `i64` | Unix timestamp |
| `expires_at` | `Option<i64>` | TTL expiry timestamp (session tier) |

### ConversationStore

Conversation metadata with create-or-update semantics.

| Method | Description |
|--------|-------------|
| `new(client)` | Create store |
| `create(id, title, model_id, message_count)` | Create or update conversation |
| `get(conversation_id)` | Get by ID |
| `list(limit)` | List conversations sorted by recency |
| `update(conversation_id, title, message_count)` | Update metadata |
| `delete(conversation_id)` | Delete conversation |

### TieredMemory

Three-tier memory hierarchy with adaptive search and automatic demotion/promotion.

| Method | Description |
|--------|-------------|
| `new(hot_store, client, embeddings, config)` | Create with custom configuration |
| `with_defaults(hot_store, client, embeddings)` | Create with default thresholds |
| `add_message(message, importance)` | Add to hot tier with Session authority |
| `add_canonical_message(message, importance, token)` | Add canonical message (no TTL) |
| `evict_expired()` | Delete expired session messages -> count |
| `record_access(message_id)` | Update access tracking for scoring |
| `search_adaptive(query, conversation_id)` | Similarity-based search across tiers |
| `search_adaptive_multi_factor(query, conversation_id)` | Blended scoring (similarity + recency + importance) |
| `demote_to_warm(message_id, summary)` | Compress message to summary |
| `demote_to_cold(summary_id, fact)` | Extract fact from summary |
| `promote_to_hot(message_id)` | Restore full message from warm tier |
| `get_demotion_candidates(tier, count)` | Get candidates for demotion |
| `get_stats()` | Get tier counts |

**`MemoryTier` enum:** `Hot`, `Warm`, `Cold`.

**`MemoryAuthority` enum:** `Ephemeral`, `Session`, `Canonical`.

### DocumentStore

Document ingestion with hybrid search (vector + BM25 via Reciprocal Rank Fusion).

| Method | Description |
|--------|-------------|
| `new(client, embeddings, bm25_base_path)` | Create with default chunking |
| `index_file(file_path, scope)` | Index document from file |
| `index_bytes(bytes, file_name, file_type, scope)` | Index document from bytes |
| `search(request)` | Hybrid or vector-only search |
| `delete_document(document_id)` | Delete document and chunks |
| `list_by_conversation(conversation_id)` | List documents in conversation |
| `list_by_project(project_id)` | List documents in project |

**`DocumentScope` enum:** `Conversation(String)`, `Project(String)`, `Global`.

**`DocumentType` enum:** `Pdf`, `Markdown`, `PlainText`, `Docx`, `Unknown`.

### ImageStore

Analyzed image storage with semantic search over LLM-generated descriptions.

| Method | Description |
|--------|-------------|
| `new(client, embeddings)` | Create store |
| `store(metadata, storage)` | Store image with metadata |
| `store_from_bytes(bytes, analysis, conversation_id, format)` | Store from raw bytes |
| `get(image_id)` | Get image metadata |
| `search(request)` | Semantic search on analysis text |
| `delete(image_id)` | Delete image |

**`ImageFormat` enum:** `Png`, `Jpeg`, `Gif`, `Webp`, `Svg`.

**`ImageStorage` enum:** `Base64(String)`, `FilePath(String)`, `Url(String)`.

### LockStore

SQLite-backed cross-process lock coordination with stale lock detection.

| Method | Description |
|--------|-------------|
| `new_default()` | Use `~/.brainwires/locks.db` |
| `new_with_path(db_path)` | Use custom database path |
| `try_acquire(lock_type, resource_path, agent_id, timeout)` | Acquire lock (idempotent per agent) |
| `release(lock_type, resource_path, agent_id)` | Release a lock |
| `release_all_for_agent(agent_id)` | Release all locks held by agent |
| `is_locked(lock_type, resource_path)` | Check lock status |
| `cleanup_stale()` | Remove expired and dead-process locks |

**Lock types:** `file_read`, `file_write`, `build`, `test`, `build_test`.

### TemplateStore

JSON file-based reusable plan template storage with `{{variable}}` substitution.

| Method | Description |
|--------|-------------|
| `new(data_dir)` | Create store (creates `templates.json`) |
| `save(template)` | Save a template |
| `get(template_id)` | Get by ID |
| `search(query)` | Search name, description, tags |
| `delete(template_id)` | Delete template |

## Usage Examples

### Connection sharing across domain stores and RAG

```rust
use brainwires_storage::{
    LanceDatabase, CachedEmbeddingProvider, MessageStore, ConversationStore,
};
use brainwires_storage::databases::VectorDatabase;
use std::sync::Arc;

let db = Arc::new(LanceDatabase::new("~/.brainwires/db").await?);
let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
db.initialize(embeddings.dimension()).await?;

// All stores share the same LanceDatabase connection
let messages = MessageStore::new(db.clone(), embeddings.clone());
let conversations = ConversationStore::new(db.clone());

// The same `db` can also be passed to the RAG subsystem as a VectorDatabase
// let rag = RagClient::with_vector_db(db.clone());
```

### Store and search conversation messages

```rust
use brainwires_storage::{LanceDatabase, CachedEmbeddingProvider, MessageStore, MessageMetadata};
use std::sync::Arc;

let db = Arc::new(LanceDatabase::new("~/.brainwires/db").await?);
let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
db.initialize(embeddings.dimension()).await?;

let store = MessageStore::new(db.clone(), embeddings.clone());

// Add messages
store.add(MessageMetadata {
    message_id: "msg-001".into(),
    conversation_id: "conv-001".into(),
    role: "assistant".into(),
    content: "We should use B-tree indexes for the user lookup table".into(),
    token_count: Some(35),
    model_id: None,
    images: None,
    created_at: chrono::Utc::now().timestamp(),
    expires_at: None,
}).await?;

// Semantic search
let results = store.search("database indexing strategy", 5, 0.7).await?;
for (msg, score) in &results {
    println!("[{:.2}] {}", score, msg.content);
}

// Search within a conversation
let results = store.search_conversation("conv-001", "indexing", 3, 0.6).await?;
```

### Use tiered memory for infinite context

```rust
use brainwires_storage::{
    TieredMemory, TieredMemoryConfig, MessageStore, MessageMetadata,
    MemoryTier, LanceDatabase, CachedEmbeddingProvider,
};
use std::sync::Arc;

let db = Arc::new(LanceDatabase::new("~/.brainwires/db").await?);
let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
db.initialize(embeddings.dimension()).await?;

let hot_store = Arc::new(MessageStore::new(db.clone(), embeddings.clone()));

let config = TieredMemoryConfig {
    hot_retention_hours: 12,
    warm_retention_hours: 168,
    max_hot_messages: 500,
    session_ttl_hours: 24,
    ..TieredMemoryConfig::default()
};

let mut memory = TieredMemory::new(hot_store, db.clone(), embeddings.clone(), config);

// Add message to hot tier
memory.add_message(MessageMetadata {
    message_id: "msg-042".into(),
    conversation_id: "conv-001".into(),
    role: "assistant".into(),
    content: "JWT tokens expire after 15 minutes".into(),
    token_count: Some(20),
    model_id: None,
    images: None,
    created_at: chrono::Utc::now().timestamp(),
    expires_at: None,
}, 0.8).await?;

// Search across all tiers with multi-factor scoring
let results = memory.search_adaptive_multi_factor("token expiration", Some("conv-001")).await?;
for result in &results {
    println!("[{:?} {:.2}] {}", result.tier, result.score, result.content);
}
```

### Index and search documents with hybrid retrieval

```rust
use brainwires_storage::{
    DocumentStore, DocumentScope, DocumentSearchRequest, DocumentType,
    LanceDatabase, CachedEmbeddingProvider,
};
use std::sync::Arc;
use std::path::Path;

let db = Arc::new(LanceDatabase::new("~/.brainwires/db").await?);
let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
db.initialize(embeddings.dimension()).await?;

let store = DocumentStore::new(db.clone(), embeddings.clone(), "~/.brainwires/bm25");

// Index a file
let metadata = store.index_file(
    Path::new("docs/architecture.md"),
    DocumentScope::Project("my-project".into()),
).await?;
println!("Indexed: {} ({} chunks)", metadata.title.unwrap_or_default(), metadata.chunk_count);

// Hybrid search (vector + BM25)
let results = store.search(DocumentSearchRequest {
    query: "authentication flow".into(),
    limit: 10,
    min_score: 0.5,
    conversation_id: None,
    project_id: Some("my-project".into()),
    file_types: None,
    use_hybrid: true,
}).await?;

for result in &results {
    println!("[{:.2}] {} (chunk {})", result.score, result.document_id, result.chunk_index);
}
```

### Coordinate multi-process access with locks

```rust
use brainwires_stores::LockStore;
use std::time::Duration;

let locks = LockStore::new_default().await?;

// Acquire a write lock with 30-second timeout
let acquired = locks.try_acquire(
    "file_write",
    "src/main.rs",
    "agent-001",
    Some(Duration::from_secs(30)),
).await?;

if acquired {
    // Do exclusive work on file...
    locks.release("file_write", "src/main.rs", "agent-001").await?;
}

// Cleanup stale locks from dead processes
let cleaned = locks.cleanup_stale().await?;
println!("Cleaned {} stale locks", cleaned);
```

## Integration

Use via the `brainwires` facade crate with the `storage` feature, or depend on `brainwires-storage` directly:

```toml
# Via facade
[dependencies]
brainwires = { version = "0.11", features = ["storage"] }

# Direct
[dependencies]
brainwires-storage = "0.11"
```

The crate re-exports all components at the top level:

```rust
use brainwires_storage::{
    // Always available
    StorageBackend, BackendCapabilities,
    FieldDef, FieldType, FieldValue, Filter, Record, ScoredRecord, record_get,
    ImageFormat, ImageMetadata, ImageSearchRequest, ImageSearchResult, ImageStorage,
    PlanTemplate, TemplateStore,
};

// Database backends (feature-gated)
#[cfg(feature = "lance-backend")]
use brainwires_storage::LanceDatabase;

// Native-only stores
#[cfg(feature = "native")]
use brainwires_storage::{
    CachedEmbeddingProvider, FastEmbedManager,
    ConversationMetadata, ConversationStore,
    MessageMetadata, MessageStore,
    TaskMetadata, TaskStore, AgentStateMetadata, AgentStateStore,
    PlanStore,
    LockStore, LockRecord, LockStats,
    ImageStore,
    SummaryStore, FactStore, TierMetadataStore,
    CanonicalWriteToken, MemoryAuthority, MemoryTier,
    TieredMemory, TieredMemoryConfig, TieredSearchResult,
    FileChunk, FileContent, FileContextManager,
};

// VectorDatabase trait (from databases module)
use brainwires_storage::databases::VectorDatabase;
```

A `prelude` module is also available for convenient imports:

```rust
use brainwires_storage::prelude::*;
```

## License

Licensed under either MIT or Apache-2.0 at your option. See [LICENSE-MIT](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-APACHE).
