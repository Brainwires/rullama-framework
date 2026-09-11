/** A single search result from vector or hybrid search.
 * Equivalent to Rust's `SearchResult` in rullama-core. */
export interface SearchResult {
  /** Path of the file the chunk came from. */
  file_path: string;
  /** Root directory of the indexed tree the file belongs to. */
  root_path?: string;
  /** Text of the matching chunk. */
  content: string;
  /** Combined relevance score used for ranking. */
  score: number;
  /** Cosine-similarity score from the vector search. */
  vector_score: number;
  /** Keyword (BM25/full-text) score; undefined for pure vector search. */
  keyword_score?: number;
  /** First line of the chunk in the source file (1-based). */
  start_line: number;
  /** Last line of the chunk in the source file (1-based). */
  end_line: number;
  /** Programming language of the file. */
  language: string;
  /** Project name the file was indexed under. */
  project?: string;
  /** Unix timestamp (seconds) when the chunk was indexed. */
  indexed_at: number;
}

/** Metadata stored with each code chunk in the vector database.
 * Equivalent to Rust's `ChunkMetadata` in rullama-core. */
export interface ChunkMetadata {
  /** Path of the file the chunk came from. */
  file_path: string;
  /** Root directory of the indexed tree the file belongs to. */
  root_path?: string;
  /** Project name the file was indexed under. */
  project?: string;
  /** First line of the chunk in the source file (1-based). */
  start_line: number;
  /** Last line of the chunk in the source file (1-based). */
  end_line: number;
  /** Programming language of the file, if detected. */
  language?: string;
  /** File extension (without the dot). */
  extension?: string;
  /** Content hash of the whole file, used to detect changes on re-index. */
  file_hash: string;
  /** Unix timestamp (seconds) when the chunk was indexed. */
  indexed_at: number;
}

/** Statistics about the vector database contents.
 * Equivalent to Rust's `DatabaseStats` in rullama-core. */
export interface DatabaseStats {
  /** Number of stored points (chunks). */
  total_points: number;
  /** Number of stored embedding vectors (equal to `total_points` for every current backend). */
  total_vectors: number;
  /** `[language, chunk count]` pairs. */
  language_breakdown: [string, number][];
}
