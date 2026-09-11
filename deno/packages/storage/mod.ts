/**
 * Backend-agnostic persistent storage substrate for rullama. Defines the
 * `StorageBackend` (typed tables, structured `Filter`s, vector search) and
 * `VectorDatabase` (RAG embedding store with hybrid search) interfaces plus
 * the schema/record/filter types they share, and ships `InMemoryStorageBackend`
 * for tests, `CachedEmbeddingProvider` (LRU memo over any `EmbeddingProvider`)
 * and seven concrete adapters: Postgres and MySQL (`npm:pg` / `npm:mysql2`),
 * SurrealDB (`npm:surrealdb`), and the fetch-only Qdrant, Pinecone, Weaviate
 * and Milvus clients. `Raw` filters are refused unless a backend is built with
 * `allowRawFilters: true`.
 *
 * Domain stores live in `@rullama/stores` and tiered memory in
 * `@rullama/memory`; there are no transitional re-exports here.
 * Equivalent to Rust's `rullama-storage` crate.
 *
 * @module
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

// Core types that appear in VectorDatabase / backend signatures.
export type { ChunkMetadata, DatabaseStats, SearchResult } from "@rullama/core";

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
