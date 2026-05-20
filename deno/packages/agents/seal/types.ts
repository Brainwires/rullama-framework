/**
 * Shared SEAL types (ResolvedReference + SealProcessingResult) placed in a
 * standalone module to avoid circular imports between mod.ts and
 * knowledge_integration.ts.
 */

import type { QueryCore } from "./query_core.ts";
import type { ResolvedReference } from "./coreference.ts";
import type { Issue } from "./reflection.ts";

export type { ResolvedReference };

/** Result of running the SEAL pipeline on a single user query. */
export interface SealProcessingResult {
  original_query: string;
  resolved_query: string;
  query_core: QueryCore | undefined;
  matched_pattern: string | undefined;
  resolutions: ResolvedReference[];
  quality_score: number;
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
