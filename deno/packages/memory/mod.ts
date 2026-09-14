/**
 * @module @rullama/memory
 *
 * Tiered-memory orchestration on top of `@rullama/storage` + `@rullama/stores`.
 * Equivalent to Rust's `rullama-memory` crate.
 *
 * Provides the hot/warm/cold tier model (`MemoryTier`, `TierMetadata`,
 * `MessageSummary`, `KeyFact`), the retention and multi-factor retrieval
 * scoring functions, and `TieredMemory`, an in-process implementation that
 * tracks messages through the tiers, evicts on capacity or TTL, and ranks
 * demotion candidates. Persistence is left to the caller.
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
  type MessageMetadata,
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
