/**
 * Knowledge layer for rullama: the type contract for the "Open Brain"
 * personal-knowledge system. Provides the {@link Thought} model with its
 * category/source enums and parsers, the {@link Entity} / {@link Relationship}
 * knowledge-graph types with {@link ExtractionResult} and contradiction
 * events, the request/response shapes for thought capture, memory search and
 * PKS/BKS knowledge search, and the {@link BrainClient} interface that ties
 * them together. No `BrainClient` implementation ships here — a concrete one
 * needs a storage backend and an embedding provider. Prompting techniques
 * live in `@rullama/prompting`; RAG and code analysis in `@rullama/rag`.
 * Equivalent to Rust's `rullama-knowledge` crate.
 *
 * @module
 */

export {
  ALL_THOUGHT_CATEGORIES,
  createThought,
  parseThoughtCategory,
  parseThoughtSource,
} from "./knowledge/mod.ts";

export type {
  BksStats,
  BrainClient,
  CaptureThoughtRequest,
  CaptureThoughtResponse,
  ContradictionEvent,
  ContradictionKind,
  DeleteThoughtRequest,
  DeleteThoughtResponse,
  Entity,
  EntityType,
  ExtractionResult,
  GetThoughtRequest,
  GetThoughtResponse,
  KnowledgeResult,
  ListRecentRequest,
  ListRecentResponse,
  MemorySearchResult,
  MemoryStatsResponse,
  PksStats,
  Relationship,
  SearchKnowledgeRequest,
  SearchKnowledgeResponse,
  SearchMemoryRequest,
  SearchMemoryResponse,
  Thought,
  ThoughtCategory,
  ThoughtSource,
  ThoughtStats,
  ThoughtSummary,
} from "./knowledge/mod.ts";
