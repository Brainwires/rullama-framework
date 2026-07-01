/**
 * Unified database traits for the rullama storage layer.
 *
 * Two interfaces define the database capabilities:
 * - StorageBackend -- generic CRUD + vector search for domain stores
 * - VectorDatabase -- RAG-style embedding storage with hybrid search
 *
 * Equivalent to Rust's `databases/traits.rs` in rullama-storage.
 * @module
 */

import type {
  ChunkMetadata,
  DatabaseStats,
  SearchResult,
} from "@rullama/core";
import type { FieldDef, Filter, Record, ScoredRecord } from "./types.ts";

/**
 * Backend-agnostic storage operations.
 *
 * Domain stores (MessageStore, ConversationStore, etc.) depend on this
 * interface so they can work with any supported database backend.
 */
export interface StorageBackend {
  /**
   * Ensure a table exists with the given schema.
   * Implementations should be idempotent.
   */
  ensureTable(tableName: string, schema: FieldDef[]): Promise<void>;

  /** Insert one or more records into a table. */
  insert(tableName: string, records: Record[]): Promise<void>;

  /**
   * Query records matching an optional filter.
   * Pass undefined for filter to return all rows (up to limit).
   */
  query(
    tableName: string,
    filter?: Filter,
    limit?: number,
  ): Promise<Record[]>;

  /** Delete records matching a filter. */
  delete(tableName: string, filter: Filter): Promise<void>;

  /**
   * Count records matching an optional filter.
   * Default implementation: count via query.
   */
  count(tableName: string, filter?: Filter): Promise<number>;

  /**
   * Vector similarity search.
   * Returns up to `limit` records ordered by descending similarity to `vector`.
   */
  vectorSearch(
    tableName: string,
    vectorColumn: string,
    vector: number[],
    limit: number,
    filter?: Filter,
  ): Promise<ScoredRecord[]>;
}

/**
 * Trait for vector database operations used by the RAG subsystem.
 *
 * Implementations handle connection management, BM25 keyword indexing,
 * and hybrid search fusion internally.
 */
export interface VectorDatabase {
  /** Initialize the database and create collections if needed. */
  initialize(dimension: number): Promise<void>;

  /** Store embeddings with metadata. */
  storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    rootPath: string,
  ): Promise<number>;

  /** Search for similar vectors. */
  search(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
  ): Promise<SearchResult[]>;

  /** Search with additional filters (extensions, languages, path patterns). */
  searchFiltered(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
    fileExtensions?: string[],
    languages?: string[],
    pathPatterns?: string[],
  ): Promise<SearchResult[]>;

  /** Delete embeddings for a specific file. */
  deleteByFile(filePath: string): Promise<number>;

  /** Clear all embeddings. */
  clear(): Promise<void>;

  /** Get statistics about the stored data. */
  getStatistics(): Promise<DatabaseStats>;

  /** Flush/save changes to disk. */
  flush(): Promise<void>;

  /** Count embeddings for a specific root path. */
  countByRootPath(rootPath: string): Promise<number>;

  /** Get unique file paths indexed for a specific root path. */
  getIndexedFiles(rootPath: string): Promise<string[]>;

  /**
   * Search and return results together with their embedding vectors.
   * Default: delegates to search() and returns empty embedding vectors.
   */
  searchWithEmbeddings(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
  ): Promise<[SearchResult[], number[][]]>;
}
