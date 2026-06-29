/**
 * @module @rullama/memory
 *
 * Tiered-memory orchestration on top of `@rullama/storage` + `@rullama/stores`.
 * Equivalent to Rust's `rullama-memory` crate.
 *
 * Extracted from `@rullama/storage` in v0.11.0.
 */

export {
  computeMultiFactorScore,
  createTierMetadata,
  defaultTieredMemoryConfig,
  demoteTier,
  type FactType,
  type KeyFact,
  type MemoryAuthority,
  type MemoryTier,
  type MessageSummary,
  type MultiFactorScore,
  parseMemoryAuthority,
  promoteTier,
  recencyFromHours,
  recordAccess,
  retentionScore,
  TieredMemory,
  type TieredMemoryConfig,
  type TieredMemoryStats,
  type TieredSearchResult,
  type TierMetadata,
} from "./tiered_memory.ts";
