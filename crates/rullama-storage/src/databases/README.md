# databases/ — Unified Database Layer

Backend-agnostic database abstraction for the Brainwires storage system.

## Bring Your Own Database

The two core traits — `StorageBackend` and `VectorDatabase` — are always available with no feature flags required. Implement one or both on your own struct to plug any database into the framework.

### StorageBackend (CRUD + vector search for domain stores)

```rust
use anyhow::Result;
use async_trait::async_trait;
use rullama_storage::databases::{
    StorageBackend, FieldDef, Filter, Record, ScoredRecord,
};

pub struct MyDatabase {
    // your connection pool, client, etc.
}

#[async_trait]
impl StorageBackend for MyDatabase {
    async fn ensure_table(&self, table_name: &str, schema: &[FieldDef]) -> Result<()> {
        // Create the table if it doesn't exist. Must be idempotent.
        todo!()
    }

    async fn insert(&self, table_name: &str, records: Vec<Record>) -> Result<()> {
        // Insert rows. Each Record is Vec<(column_name, FieldValue)>.
        todo!()
    }

    async fn query(
        &self,
        table_name: &str,
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<Record>> {
        // Return matching rows. None filter = return all (up to limit).
        todo!()
    }

    async fn delete(&self, table_name: &str, filter: &Filter) -> Result<()> {
        // Delete matching rows.
        todo!()
    }

    // count() has a default impl that calls query().len() — override for efficiency.

    async fn vector_search(
        &self,
        table_name: &str,
        vector_column: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredRecord>> {
        // Return rows ranked by cosine/L2 similarity to `vector`.
        // If your DB has no native vector search, compute client-side.
        todo!()
    }
}
```

### VectorDatabase (RAG embedding storage + hybrid search)

```rust
use anyhow::Result;
use async_trait::async_trait;
use rullama_storage::databases::{
    VectorDatabase, ChunkMetadata, DatabaseStats, SearchResult,
};

#[async_trait]
impl VectorDatabase for MyDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        // Create collections/tables for the given embedding dimension (e.g. 384).
        todo!()
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        root_path: &str,
    ) -> Result<usize> {
        // Store embedding vectors with their source metadata.
        // root_path scopes embeddings to a project directory.
        // Return the number of embeddings stored.
        todo!()
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
    ) -> Result<Vec<SearchResult>> {
        // Vector similarity search. If hybrid=true, combine with BM25 keyword
        // scoring (use bm25_helpers module or your DB's native full-text search).
        todo!()
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        // Same as search() but with additional metadata filters.
        todo!()
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        // Delete all embeddings for a given source file. Return count deleted.
        todo!()
    }

    async fn clear(&self) -> Result<()> {
        // Drop all embeddings.
        todo!()
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        // Return collection/table stats (counts, sizes, etc.)
        todo!()
    }

    async fn flush(&self) -> Result<()> {
        // Persist any buffered writes. No-op if your backend auto-flushes.
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        // Count embeddings scoped to a project root.
        todo!()
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        // Return unique file paths indexed under this root.
        todo!()
    }

    // search_with_embeddings() has a default impl that delegates to search()
    // and returns empty embedding vectors. Override if your DB can return
    // stored vectors (used by the spectral diversity reranker).
}
```

### Wiring it up

Wrap your implementation in `Arc` and pass it wherever the framework expects a backend:

```rust
use std::sync::Arc;

let db = Arc::new(MyDatabase::new(/* ... */).await?);

// Domain stores (messages, conversations, tasks, etc.)
let messages = MessageStore::new(db.clone(), embeddings.clone());
let conversations = ConversationStore::new(db.clone());

// RAG subsystem
let rag = RagClient::with_vector_db(db.clone());

// Cognition layer
let brain = BrainClient::with_backend(db.clone());
```

If your struct implements both traits, everything shares one connection — no duplication.

## Type Reference

All types are in `types.rs` and re-exported from `databases::*`:

| Type | Purpose |
|------|---------|
| `FieldDef` | Column definition (name, type, nullable) |
| `FieldType` | Column types: `Utf8`, `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, `Float64`, `Boolean`, `Vector(dim)` |
| `FieldValue` | Typed nullable column value (with accessor methods like `.as_str()`, `.as_i64()`, `.as_vector()`) |
| `Record` | A row: `Vec<(String, FieldValue)>` |
| `ScoredRecord` | A row + similarity score (from vector search) |
| `Filter` | Query filter: `Eq`, `Ne`, `Lt`, `Gt`, `In`, `And`, `Or`, `NotNull`, `IsNull`, `Raw(String)` |
| `ChunkMetadata` | Source metadata for RAG embeddings (file path, line range, language) |
| `SearchResult` | RAG search hit (content, metadata, score) |
| `DatabaseStats` | Collection statistics (counts, sizes) |

`Filter::Raw(String)` is an escape hatch for backend-specific filter expressions that don't map to the structured variants.

## Built-in Backends

| Database | Struct | `StorageBackend` | `VectorDatabase` | Feature flag |
|----------|--------|:---:|:---:|--------------|
| LanceDB | `LanceDatabase` | YES | YES | `lance-backend` (default) |
| PostgreSQL + pgvector | `PostgresDatabase` | YES | YES | `postgres-backend` |
| MySQL / MariaDB | `MySqlDatabase` | YES | NO | `mysql-backend` |
| SurrealDB | `SurrealDatabase` | YES | YES | `surrealdb-backend` |
| Qdrant | `QdrantDatabase` | NO | YES | `qdrant-backend` |
| Pinecone | `PineconeDatabase` | NO | YES | `pinecone-backend` |
| Milvus | `MilvusDatabase` | NO | YES | `milvus-backend` |
| Weaviate | `WeaviateDatabase` | NO | YES | `weaviate-backend` |
| NornicDB | `NornicDatabase` | NO | YES | `nornicdb-backend` |

## Module Layout

```
databases/
  traits.rs           -- StorageBackend + VectorDatabase trait definitions
  types.rs            -- Record, FieldDef, FieldValue, Filter, ScoredRecord
  capabilities.rs     -- BackendCapabilities runtime discovery
  bm25_helpers.rs     -- shared BM25 scoring for client-side keyword search
  sql/                -- shared SQL dialect layer (SqlDialect trait)
    postgres.rs       -- PostgreSQL dialect
    mysql.rs          -- MySQL dialect
    surrealdb.rs      -- SurrealDB dialect
  lance/              -- LanceDB backend (default)
  postgres/           -- PostgreSQL + pgvector backend
  mysql/              -- MySQL / MariaDB backend
  surrealdb/          -- SurrealDB backend
  qdrant/             -- Qdrant backend
  pinecone/           -- Pinecone backend
  milvus/             -- Milvus backend
  weaviate/           -- Weaviate backend
  nornicdb/           -- NornicDB backend (multi-transport: REST/Bolt/gRPC)
```
