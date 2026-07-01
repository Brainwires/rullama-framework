/**
 * RAG module -- Retrieval-Augmented Generation client, types, and requests.
 *
 * Provides types for semantic code search with hybrid BM25+vector search,
 * file indexing, git history search, and advanced filtered search.
 */

// ---------------------------------------------------------------------------
// IndexRequest / IndexResponse
// ---------------------------------------------------------------------------

/** Request to index a codebase. */
export interface IndexRequest {
  /** Path to the codebase directory to index. */
  path: string;
  /** Optional project name (for multi-project support). */
  project?: string;
  /** Optional glob patterns to include (e.g., ["**\/*.rs", "**\/*.toml"]). */
  includePatterns?: string[];
  /** Optional glob patterns to exclude (e.g., ["**\/target\/**"]). */
  excludePatterns?: string[];
  /** Maximum file size in bytes to index (default: 1MB). */
  maxFileSize?: number;
}

/** Default max file size: 1 MB. */
export const DEFAULT_MAX_FILE_SIZE = 1_048_576;

/** Indexing mode used. */
export type IndexingMode = "full" | "incremental";

/** Response from indexing operation. */
export interface IndexResponse {
  /** Indexing mode used. */
  mode: IndexingMode;
  /** Number of files successfully indexed. */
  filesIndexed: number;
  /** Number of code chunks created. */
  chunksCreated: number;
  /** Number of embeddings generated. */
  embeddingsGenerated: number;
  /** Time taken in milliseconds. */
  durationMs: number;
  /** Any errors encountered (non-fatal). */
  errors: string[];
  /** Number of files updated (incremental mode only). */
  filesUpdated: number;
  /** Number of files removed (incremental mode only). */
  filesRemoved: number;
}

// ---------------------------------------------------------------------------
// QueryRequest / QueryResponse
// ---------------------------------------------------------------------------

/** Request to query the codebase. */
export interface QueryRequest {
  /** The question or search query. */
  query: string;
  /** Optional path to filter by specific indexed codebase. */
  path?: string;
  /** Optional project name to filter by. */
  project?: string;
  /** Number of results to return (default: 10). */
  limit?: number;
  /** Minimum similarity score 0.0-1.0 (default: 0.7). */
  minScore?: number;
  /** Enable hybrid search (vector + keyword) -- default: true. */
  hybrid?: boolean;
}

/** Default query limit. */
export const DEFAULT_LIMIT = 10;

/** Default minimum similarity score. */
export const DEFAULT_MIN_SCORE = 0.7;

/** A single search result. */
export interface SearchResult {
  /** File path of the matched code. */
  filePath: string;
  /** Root path of the indexed codebase. */
  rootPath?: string;
  /** The matched code content. */
  content: string;
  /** Combined similarity score (0.0-1.0). */
  score: number;
  /** Vector similarity score. */
  vectorScore: number;
  /** Keyword match score (if hybrid search enabled). */
  keywordScore?: number;
  /** Start line number. */
  startLine: number;
  /** End line number. */
  endLine: number;
  /** Programming language. */
  language: string;
  /** Project name if set. */
  project?: string;
  /** Unix timestamp when indexed. */
  indexedAt: number;
}

/** Response from query operation. */
export interface QueryResponse {
  /** List of search results, ordered by relevance. */
  results: SearchResult[];
  /** Time taken in milliseconds. */
  durationMs: number;
  /** The actual threshold used (may be lower than requested if adaptive). */
  thresholdUsed: number;
  /** Whether the threshold was automatically lowered. */
  thresholdLowered: boolean;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/** Statistics for a single programming language in the index. */
export interface LanguageStats {
  /** Language name. */
  language: string;
  /** Number of indexed files. */
  fileCount: number;
  /** Number of code chunks. */
  chunkCount: number;
}

/** Statistics about the indexed codebase. */
export interface StatisticsResponse {
  /** Total number of indexed files. */
  totalFiles: number;
  /** Total number of code chunks. */
  totalChunks: number;
  /** Total number of embeddings. */
  totalEmbeddings: number;
  /** Size of the vector database in bytes. */
  databaseSizeBytes: number;
  /** Breakdown by programming language. */
  languageBreakdown: LanguageStats[];
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

/** Response from clear operation. */
export interface ClearResponse {
  /** Whether the operation was successful. */
  success: boolean;
  /** Optional message. */
  message: string;
}

// ---------------------------------------------------------------------------
// AdvancedSearchRequest
// ---------------------------------------------------------------------------

/** Request to search with file type filters. */
export interface AdvancedSearchRequest {
  /** The search query. */
  query: string;
  /** Optional path to filter by. */
  path?: string;
  /** Optional project name. */
  project?: string;
  /** Number of results to return (default: 10). */
  limit?: number;
  /** Minimum similarity score (default: 0.7). */
  minScore?: number;
  /** Filter by file extensions (e.g., ["rs", "toml"]). */
  fileExtensions?: string[];
  /** Filter by programming languages. */
  languages?: string[];
  /** Filter by file path patterns (glob). */
  pathPatterns?: string[];
}

// ---------------------------------------------------------------------------
// Git history search
// ---------------------------------------------------------------------------

/** Request to search git history. */
export interface SearchGitHistoryRequest {
  /** The search query. */
  query: string;
  /** Path to the codebase (will discover git repo). */
  path?: string;
  /** Optional project name. */
  project?: string;
  /** Optional branch name (default: current branch). */
  branch?: string;
  /** Maximum number of commits to index/search (default: 10). */
  maxCommits?: number;
  /** Number of results to return (default: 10). */
  limit?: number;
  /** Minimum similarity score 0.0-1.0 (default: 0.7). */
  minScore?: number;
  /** Filter by commit author. */
  author?: string;
  /** Filter by commits since this date. */
  since?: string;
  /** Filter by commits until this date. */
  until?: string;
  /** Filter by file path pattern. */
  filePattern?: string;
}

/** A single git search result. */
export interface GitSearchResult {
  /** Git commit hash (SHA). */
  commitHash: string;
  /** Commit message. */
  commitMessage: string;
  /** Author name. */
  author: string;
  /** Author email. */
  authorEmail: string;
  /** Commit date (Unix timestamp). */
  commitDate: number;
  /** Combined similarity score (0.0-1.0). */
  score: number;
  /** Vector similarity score. */
  vectorScore: number;
  /** Keyword match score (if hybrid search enabled). */
  keywordScore?: number;
  /** Files changed in this commit. */
  filesChanged: string[];
  /** Diff snippet. */
  diffSnippet: string;
}

/** Response from git history search. */
export interface SearchGitHistoryResponse {
  /** List of matching commits, ordered by relevance. */
  results: GitSearchResult[];
  /** Number of commits indexed during this search. */
  commitsIndexed: number;
  /** Total commits in cache for this repo. */
  totalCachedCommits: number;
  /** Time taken in milliseconds. */
  durationMs: number;
}

// ---------------------------------------------------------------------------
// ChunkMetadata
// ---------------------------------------------------------------------------

/** Metadata for a code chunk in the vector database. */
export interface ChunkMetadata {
  /** File path of the chunk. */
  filePath: string;
  /** Root path of the indexed codebase. */
  rootPath?: string;
  /** Project name. */
  project?: string;
  /** Start line number. */
  startLine: number;
  /** End line number. */
  endLine: number;
  /** Programming language. */
  language?: string;
  /** File extension. */
  extension?: string;
  /** Hash of the source file. */
  fileHash: string;
  /** Unix timestamp when indexed. */
  indexedAt: number;
}

// ---------------------------------------------------------------------------
// RagClient interface (stub -- concrete implementations need vector DB)
// ---------------------------------------------------------------------------

/**
 * RagClient interface -- core semantic code search.
 *
 * Concrete implementations require a vector database backend and an
 * embedding provider. This interface defines the public API contract.
 */
export interface RagClient {
  /** Index a codebase directory. */
  indexCodebase(req: IndexRequest): Promise<IndexResponse>;

  /** Query the indexed codebase. */
  queryCodebase(req: QueryRequest): Promise<QueryResponse>;

  /** Get statistics about the index. */
  getStatistics(): Promise<StatisticsResponse>;

  /** Clear the index. */
  clearIndex(): Promise<ClearResponse>;

  /** Advanced search with file type filters. */
  advancedSearch(req: AdvancedSearchRequest): Promise<QueryResponse>;

  /** Search git history. */
  searchGitHistory(
    req: SearchGitHistoryRequest,
  ): Promise<SearchGitHistoryResponse>;
}

// ── Code analysis (folded into @rullama/rag in v0.11.0) ──
export {
  buildCallGraph,
  CallGraph,
  createSymbolId,
  definitionToStorageId,
  determineReferenceKind,
  findReferences,
  referenceToStorageId,
  RepoMap,
  symbolIdToStorageId,
  symbolKindDisplayName,
  visibilityFromKeywords,
} from "./code_analysis/mod.ts";

export type {
  CallEdge,
  CallGraphNode,
  Definition as CodeAnalysisDefinition,
  ExtractOptions,
  LanguageStats as CodeAnalysisLanguageStats,
  Reference as CodeAnalysisReference,
  ReferenceKind,
  SymbolId,
  SymbolKind,
  Visibility,
} from "./code_analysis/mod.ts";
