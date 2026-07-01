// deno-lint-ignore-file no-explicit-any

/** Result from a vector similarity search.
 * Equivalent to Rust's `VectorSearchResult` in rullama-core. */
export interface VectorSearchResult {
  id: string;
  score: number;
  content: string;
  metadata: any;
}

/** Interface for vector database operations.
 * Equivalent to Rust's `VectorStore` trait in rullama-core. */
export interface VectorStore {
  /** Initialize the store (create tables/collections if needed). */
  initialize(dimension: number): Promise<void>;

  /** Insert embeddings with associated content and metadata. Returns count stored. */
  upsert(
    ids: string[],
    embeddings: number[][],
    contents: string[],
    metadata: any[],
  ): Promise<number>;

  /** Search for similar vectors. Returns up to limit results with score >= minScore. */
  search(
    queryVector: number[],
    limit: number,
    minScore: number,
  ): Promise<VectorSearchResult[]>;

  /** Delete items by their IDs. Returns count deleted. */
  delete(ids: string[]): Promise<number>;

  /** Delete all stored data. */
  clear(): Promise<void>;

  /** Get the number of stored items. */
  count(): Promise<number>;
}
