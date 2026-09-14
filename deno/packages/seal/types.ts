/**
 * Shared SEAL types placed in a standalone module to avoid circular imports
 * between mod.ts and knowledge_integration.ts, plus documented aliases of the
 * `@rullama/core` / `@rullama/tool-runtime` types that appear in SEAL's
 * public signatures.
 */

import type {
  EdgeType as CoreEdgeType,
  EntityStoreT as CoreEntityStoreT,
  EntityType as CoreEntityType,
  RelationshipGraphT as CoreRelationshipGraphT,
} from "@rullama/core";
import type {
  ToolErrorCategory as RuntimeToolErrorCategory,
  ToolOutcome as RuntimeToolOutcome,
} from "@rullama/tool-runtime";
import type { QueryCore } from "./query_core.ts";
import type { ResolvedReference } from "./coreference.ts";
import type { Issue } from "./reflection.ts";

export type { ResolvedReference };

// ─── Cross-package aliases ──────────────────────────────────────────────────

/**
 * Entity category used throughout SEAL (`"file"`, `"function"`, `"type"`,
 * `"variable"`, `"error"`, `"concept"`, `"command"`, …). Alias of
 * `EntityType` from `@rullama/core`.
 */
export type EntityType = CoreEntityType;

/**
 * Relationship-graph edge category (`"contains"`, `"references"`,
 * `"depends_on"`, `"modifies"`, `"defines"`, `"co_occurs"`, …). Alias of
 * `EdgeType` from `@rullama/core`.
 */
export type EdgeType = CoreEdgeType;

/**
 * Read-only entity lookup consumed by coreference resolution and query-core
 * extraction (`entityNamesByType`, `topEntityInfo`). Alias of `EntityStoreT`
 * from `@rullama/core`; {@link InMemoryEntityStore} is a minimal implementation.
 */
export type EntityStoreT = CoreEntityStoreT;

/**
 * Relationship graph queried by the executor and reflection module
 * (`getNode`, `getNeighbors`, `getEdges`, `search`). Alias of
 * `RelationshipGraphT` from `@rullama/core`.
 */
export type RelationshipGraphT = CoreRelationshipGraphT;

/**
 * Result of one tool invocation (`toolName`, `success`, `retries`,
 * `executionTimeMs`, optional `errorCategory`) fed into global learning.
 * Alias of `ToolOutcome` from `@rullama/tool-runtime`.
 */
export type ToolOutcome = RuntimeToolOutcome;

/**
 * Classified tool failure (transient, input validation, external service, …)
 * from which {@link ToolErrorPattern} derives its category name and suggested
 * fix. Alias of `ToolErrorCategory` from `@rullama/tool-runtime`.
 */
export type ToolErrorCategory = RuntimeToolErrorCategory;

// ─── SealProcessingResult ───────────────────────────────────────────────────

/** Result of running the SEAL pipeline on a single user query. */
export interface SealProcessingResult {
  /** The user query exactly as submitted. */
  original_query: string;
  /**
   * The query after confident coreferences were rewritten to `[antecedent]`
   * markers; equal to `original_query` when nothing was resolved.
   */
  resolved_query: string;
  /** Structured query extracted from `resolved_query`, or `undefined` when the question type was not recognised. */
  query_core: QueryCore | undefined;
  /** Id of the learned `QueryPattern` that matched this query, if any. */
  matched_pattern: string | undefined;
  /** Coreference resolutions that met `SealConfig.min_coreference_confidence`. */
  resolutions: ResolvedReference[];
  /** Structural quality in `[0, 1]`: `1.0` with no issues, otherwise `0.8 - 0.1 * issues` floored at `0.3`. */
  quality_score: number;
  /** Structural issues reported by `ReflectionModule.validateQueryCore`. */
  issues: Issue[];
}

/** Create a SEAL result with only quality_score / resolved_query set. */
export function newSealProcessingResult(
  quality_score: number,
  resolved_query: string,
): SealProcessingResult {
  return {
    original_query: resolved_query,
    resolved_query,
    query_core: undefined,
    matched_pattern: undefined,
    resolutions: [],
    quality_score,
    issues: [],
  };
}
