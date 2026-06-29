/** A single search result from vector or hybrid search.
 * Equivalent to Rust's `SearchResult` in rullama-core. */
export interface SearchResult {
  file_path: string;
  root_path?: string;
  content: string;
  score: number;
  vector_score: number;
  keyword_score?: number;
  start_line: number;
  end_line: number;
  language: string;
  project?: string;
  indexed_at: number;
}

/** Metadata stored with each code chunk in the vector database.
 * Equivalent to Rust's `ChunkMetadata` in rullama-core. */
export interface ChunkMetadata {
  file_path: string;
  root_path?: string;
  project?: string;
  start_line: number;
  end_line: number;
  language?: string;
  extension?: string;
  file_hash: string;
  indexed_at: number;
}

/** Statistics about the vector database contents.
 * Equivalent to Rust's `DatabaseStats` in rullama-core. */
export interface DatabaseStats {
  total_points: number;
  total_vectors: number;
  language_breakdown: [string, number][];
}
