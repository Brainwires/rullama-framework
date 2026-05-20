#![deny(missing_docs)]
//! Brainwires Storage — backend-agnostic persistent storage primitives for
//! the Brainwires Agent Framework.
//!
//! This crate provides the generic abstractions: a `StorageBackend` trait,
//! the per-backend connections (`LanceDatabase`, `PostgresDatabase`, ...),
//! embeddings (`CachedEmbeddingProvider`), BM25 keyword search, file
//! chunking, and image metadata types. Domain-shaped stores
//! (`ConversationStore`, `MessageStore`, `PlanStore`, `LockStore`, ...) and
//! the tiered hot/warm/cold memory orchestration moved out:
//!
//! - **`brainwires-memory`** — `MessageStore`, `SummaryStore`, `FactStore`,
//!   `MentalModelStore`, `TierMetadataStore`, `TieredMemory`.
//! - **`brainwires-cli` `crate::storage`** — `ConversationStore`,
//!   `TaskStore` / `AgentStateStore`, `PlanStore`, `TemplateStore`,
//!   `LockStore`, `ImageStore`, `PersistentTaskManager`.
//!
//! # Unified Database Layer ([`databases`] module)
//!
//! One struct per database, one shared connection, implementing one or both
//! of the core traits:
//!
//! - [`StorageBackend`] — generic CRUD + vector search for domain stores
//! - [`VectorDatabase`](databases::traits::VectorDatabase) — RAG embedding storage
//!   with hybrid search
//!
//! ### Database backends
//!
//! | Backend | Struct | `StorageBackend` | `VectorDatabase` | Feature |
//! |---------|--------|:---:|:---:|---------|
//! | LanceDB | `LanceDatabase` | YES | YES | `lance-backend` (default) |
//! | PostgreSQL | `PostgresDatabase` | YES | YES | `postgres-backend` |
//! | MySQL | `MySqlDatabase` | YES | NO | `mysql-backend` |
//! | SurrealDB | `SurrealDatabase` | YES | YES | `surrealdb-backend` |
//! | Qdrant | `QdrantDatabase` | NO | YES | `qdrant-backend` |
//! | Pinecone | `PineconeDatabase` | NO | YES | `pinecone-backend` |
//! | Milvus | `MilvusDatabase` | NO | YES | `milvus-backend` |
//! | Weaviate | `WeaviateDatabase` | NO | YES | `weaviate-backend` |
//! | NornicDB | `NornicDatabase` | NO | YES | `nornicdb-backend` |
//!
//! Backends that implement both traits share a single connection — construct
//! once, wrap in `Arc`, and pass to both domain stores and RAG subsystem.
//!
//! # Core Infrastructure
//!
//! - **`FastEmbedManager`** — text embeddings via FastEmbed ONNX model
//!   (all-MiniLM-L6-v2, 384 dimensions)
//! - **`CachedEmbeddingProvider`** — LRU-cached embedding provider (1000 entries)
//! - **`BM25Search`** — keyword search via Tantivy
//! - **`FileContextManager`** — file chunking / context primitives
//!
//! # Image Types
//!
//! - **`ImageFormat`**, **`ImageMetadata`**, **`ImageSearchRequest`**,
//!   **`ImageSearchResult`**, **`ImageStorage`** — pure types reused by
//!   the `ImageStore` that lives in `brainwires-cli::storage`.
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `native` | Yes | LanceDB backend + FastEmbed + file context + native primitives |
//! | `lance-backend` | Yes (via `native`) | LanceDB embedded vector database |
//! | `postgres-backend` | No | PostgreSQL + pgvector |
//! | `mysql-backend` | No | MySQL / MariaDB |
//! | `surrealdb-backend` | No | SurrealDB with native MTREE vector search |
//! | `qdrant-backend` | No | Qdrant vector search |
//! | `pinecone-backend` | No | Pinecone cloud vectors |
//! | `milvus-backend` | No | Milvus vectors |
//! | `weaviate-backend` | No | Weaviate search engine |
//! | `nornicdb-backend` | No | NornicDB graph + vector |
//! | `wasm` | No | WASM-compatible (pure types only) |

// Re-export core types
pub use brainwires_core;

// ── Always available (pure types/logic) ──────────────────────────────────

/// Structured error taxonomy. See [`StorageError`].
///
/// Public APIs on this crate return `anyhow::Result<T>` so backends don't
/// break across a trait rewrite; callers recover the typed variant via
/// `err.downcast_ref::<StorageError>()`.
pub mod error;

pub use error::StorageError;

/// Image-storage type definitions (concrete `ImageStore` lives in `brainwires-cli`).
pub mod image_types;

/// Unified database layer — one struct per database, shared connection,
/// implementing [`StorageBackend`](databases::traits::StorageBackend) and/or
/// [`VectorDatabase`](databases::traits::VectorDatabase).
pub mod databases;

/// BM25 keyword search using Tantivy. Used by Lance backend for hybrid
/// vector + keyword search; consumed by `brainwires-rag` and other
/// indexers.
#[cfg(feature = "lance-backend")]
pub mod bm25_search;
/// Glob pattern matching utilities. Used by every database backend's
/// path/include filtering.
#[cfg(feature = "lance-backend")]
pub mod glob_utils;

// Phase 9 moves:
//   paths + file_context → brainwires-core (cross-cutting utilities, no storage internals)
//   hnsw_wasm            → deleted (zero consumers anywhere in the workspace)

/// Embedding provider for vector operations.
#[cfg(feature = "native")]
pub mod embeddings;

// ── Re-exports (always available) ────────────────────────────────────────

pub use databases::BackendCapabilities;
pub use databases::traits::StorageBackend;
pub use databases::types::record_get;
pub use databases::types::{FieldDef, FieldType, FieldValue, Filter, Record, ScoredRecord};

#[cfg(feature = "lance-backend")]
pub use databases::LanceDatabase;

pub use image_types::{
    ImageFormat, ImageMetadata, ImageSearchRequest, ImageSearchResult, ImageStorage,
};

// ── Re-exports (native only) ─────────────────────────────────────────────

#[cfg(feature = "native")]
pub use embeddings::{CachedEmbeddingProvider, EmbeddingProvider, FastEmbedManager};
// FileChunk / FileContent / FileContextManager moved to brainwires-core in Phase 9.

/// Prelude module for convenient imports of the primitive surface.
pub mod prelude {
    pub use super::databases::{
        BackendCapabilities, FieldDef, FieldType, FieldValue, Filter, Record, ScoredRecord,
        StorageBackend,
    };

    #[cfg(feature = "native")]
    pub use super::embeddings::{CachedEmbeddingProvider, EmbeddingProvider, FastEmbedManager};
    // file_context moved to brainwires-core in Phase 9.
    pub use super::image_types::{
        ImageFormat, ImageMetadata, ImageSearchRequest, ImageSearchResult, ImageStorage,
    };
}
