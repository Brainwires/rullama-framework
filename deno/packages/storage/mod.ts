/**
 * @module @rullama/storage
 *
 * Backend-agnostic persistent storage substrate. Equivalent to Rust's
 * `rullama-storage` crate.
 *
 * Provides:
 * - `StorageBackend` and `VectorDatabase` interfaces
 * - `InMemoryStorageBackend` for testing
 * - Embedding provider wrapper
 * - Concrete database adapters: Postgres / MySQL / Qdrant / SurrealDB /
 *   Pinecone / Weaviate / Milvus
 *
 * Domain stores moved to `@rullama/stores`. Tiered memory orchestration
 * moved to `@rullama/memory`. No transitional re-exports — update imports.
 */

// -- Core types -------------------------------------------------------------
export {
  type BackendCapabilities,
  defaultCapabilities,
  type FieldDef,
  type FieldType,
  FieldTypes,
  type FieldValue,
  fieldValueAsBool,
  fieldValueAsF32,
  fieldValueAsF64,
  fieldValueAsI32,
  fieldValueAsI64,
  fieldValueAsStr,
  fieldValueAsVector,
  FieldValues,
  type Filter,
  Filters,
  optionalField,
  type Record,
  recordGet,
  requiredField,
  type ScoredRecord,
} from "./types.ts";

// -- Traits / interfaces ----------------------------------------------------
export { type StorageBackend, type VectorDatabase } from "./traits.ts";

// -- In-memory backend ------------------------------------------------------
export { InMemoryStorageBackend } from "./memory_backend.ts";

// -- Embedding provider -----------------------------------------------------
export {
  CachedEmbeddingProvider,
  type EmbeddingProvider,
} from "./embeddings.ts";

// -- Database backends ------------------------------------------------------
export {
  MilvusDatabase,
  type MySqlConfig,
  MySqlDatabase,
  PineconeDatabase,
  type PostgresConfig,
  PostgresDatabase,
  QdrantDatabase,
  type SurrealConfig,
  SurrealDatabase,
  WeaviateDatabase,
} from "./backends/mod.ts";
