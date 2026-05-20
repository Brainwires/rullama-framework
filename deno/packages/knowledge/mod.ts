/**
 * @module @brainwires/knowledge
 *
 * Knowledge layer: BrainClient + entity/relationship/thought graph + BKS/PKS.
 * Equivalent to Rust's `brainwires-knowledge` crate.
 *
 * Prompting moved to `@brainwires/prompting`. RAG and code analysis moved
 * to `@brainwires/rag`. No transitional re-exports — update imports.
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
