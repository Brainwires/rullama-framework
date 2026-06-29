/**
 * Tiered Memory Storage System
 *
 * Three-tier memory hierarchy for conversation storage:
 * - Hot: Full messages -- recent, important, or recently accessed
 * - Warm: Compressed summaries -- older messages
 * - Cold: Ultra-compressed key facts -- archival storage
 *
 * Messages flow from hot -> warm -> cold based on age and importance,
 * and can be promoted back up when accessed.
 *
 * Equivalent to Rust's `rullama-memory::tiered_memory`. Moved from
 * `@rullama/storage` to `@rullama/memory` in v0.11.0 to mirror Rust.
 * @module
 */

import type { MessageMetadata } from "@rullama/stores";

const SECS_PER_HOUR = 3600;
const SIMILARITY_WEIGHT = 0.50;
const RECENCY_WEIGHT = 0.30;
const IMPORTANCE_WEIGHT = 0.20;
const DEFAULT_HOT_RETENTION_HOURS = 24;
const DEFAULT_WARM_RETENTION_HOURS = 168;
const DEFAULT_HOT_IMPORTANCE_THRESHOLD = 0.3;
const DEFAULT_WARM_IMPORTANCE_THRESHOLD = 0.1;
const DEFAULT_MAX_HOT_MESSAGES = 1000;
const DEFAULT_MAX_WARM_SUMMARIES = 5000;

// -- Memory authority -------------------------------------------------------

/** Trust level of a memory entry's origin. */
export type MemoryAuthority = "ephemeral" | "session" | "canonical";

/** Parse a MemoryAuthority from a stored string. */
export function parseMemoryAuthority(s: string): MemoryAuthority {
  switch (s) {
    case "ephemeral":
      return "ephemeral";
    case "canonical":
      return "canonical";
    default:
      return "session";
  }
}

// -- Memory tier ------------------------------------------------------------

/** Memory tier classification. */
export type MemoryTier = "hot" | "warm" | "cold";

/** Get the next cooler tier. */
export function demoteTier(tier: MemoryTier): MemoryTier | undefined {
  switch (tier) {
    case "hot":
      return "warm";
    case "warm":
      return "cold";
    case "cold":
      return undefined;
  }
}

/** Get the next hotter tier. */
export function promoteTier(tier: MemoryTier): MemoryTier | undefined {
  switch (tier) {
    case "hot":
      return undefined;
    case "warm":
      return "hot";
    case "cold":
      return "warm";
  }
}

// -- Tier metadata ----------------------------------------------------------

/** Metadata tracking for tiered storage. */
export interface TierMetadata {
  messageId: string;
  tier: MemoryTier;
  importance: number;
  lastAccessed: number;
  accessCount: number;
  createdAt: number;
  authority: MemoryAuthority;
}

/** Create new tier metadata with the given importance score. */
export function createTierMetadata(
  messageId: string,
  importance: number,
  authority: MemoryAuthority = "session",
): TierMetadata {
  const now = Math.floor(Date.now() / 1000);
  return {
    messageId,
    tier: "hot",
    importance,
    lastAccessed: now,
    accessCount: 0,
    createdAt: now,
    authority,
  };
}

/** Record an access on tier metadata (mutates). */
export function recordAccess(meta: TierMetadata): void {
  meta.lastAccessed = Math.floor(Date.now() / 1000);
  meta.accessCount += 1;
}

/** Calculate a retention score (lower = demote first). */
export function retentionScore(meta: TierMetadata): number {
  const now = Math.floor(Date.now() / 1000);
  const ageHours = (now - meta.lastAccessed) / SECS_PER_HOUR;
  const recencyFactor = Math.exp(-0.01 * ageHours);
  const accessFactor = Math.log1p(meta.accessCount) * 0.1;

  return (
    meta.importance * SIMILARITY_WEIGHT +
    recencyFactor * RECENCY_WEIGHT +
    accessFactor * IMPORTANCE_WEIGHT
  );
}

// -- Multi-factor score -----------------------------------------------------

/** Combined retrieval score blending similarity, recency, and importance. */
export interface MultiFactorScore {
  similarity: number;
  recency: number;
  importance: number;
  combined: number;
}

/** Compute a multi-factor score. */
export function computeMultiFactorScore(
  similarity: number,
  recency: number,
  importance: number,
): MultiFactorScore {
  const combined = similarity * SIMILARITY_WEIGHT +
    recency * RECENCY_WEIGHT +
    importance * IMPORTANCE_WEIGHT;
  return { similarity, recency, importance, combined };
}

/** Compute recency factor from hours since last access. */
export function recencyFromHours(hoursSinceAccess: number): number {
  return Math.exp(-0.01 * hoursSinceAccess);
}

// -- Message summary / Key fact types ---------------------------------------

/** Summary of a message for warm tier storage. */
export interface MessageSummary {
  summaryId: string;
  originalMessageId: string;
  conversationId: string;
  role: string;
  summary: string;
  keyEntities: string[];
  createdAt: number;
}

/** Type of key fact. */
export type FactType =
  | "decision"
  | "definition"
  | "requirement"
  | "code_change"
  | "configuration"
  | "other";

/** Key fact extracted from messages for cold tier storage. */
export interface KeyFact {
  factId: string;
  originalMessageIds: string[];
  conversationId: string;
  fact: string;
  factType: FactType;
  createdAt: number;
}

// -- Tiered search result ---------------------------------------------------

/** Result from adaptive search across tiers. */
export interface TieredSearchResult {
  content: string;
  score: number;
  tier: MemoryTier;
  originalMessageId?: string;
  metadata?: MessageMetadata;
  multiFactorScore?: MultiFactorScore;
}

// -- Configuration ----------------------------------------------------------

/** Configuration for tiered memory behavior. */
export interface TieredMemoryConfig {
  hotRetentionHours: number;
  warmRetentionHours: number;
  hotImportanceThreshold: number;
  warmImportanceThreshold: number;
  maxHotMessages: number;
  maxWarmSummaries: number;
  sessionTtlSecs?: number;
}

/** Default tiered memory configuration. */
export function defaultTieredMemoryConfig(): TieredMemoryConfig {
  return {
    hotRetentionHours: DEFAULT_HOT_RETENTION_HOURS,
    warmRetentionHours: DEFAULT_WARM_RETENTION_HOURS,
    hotImportanceThreshold: DEFAULT_HOT_IMPORTANCE_THRESHOLD,
    warmImportanceThreshold: DEFAULT_WARM_IMPORTANCE_THRESHOLD,
    maxHotMessages: DEFAULT_MAX_HOT_MESSAGES,
    maxWarmSummaries: DEFAULT_MAX_WARM_SUMMARIES,
  };
}

// -- Statistics -------------------------------------------------------------

/** Statistics about tiered memory usage. */
export interface TieredMemoryStats {
  hotCount: number;
  warmCount: number;
  coldCount: number;
  totalTracked: number;
}

// -- Simplified TieredMemory ------------------------------------------------

/**
 * Simplified in-memory tiered memory system.
 *
 * This is a pure-logic version suitable for testing and lightweight use.
 * For a full persistent version, combine with a StorageBackend.
 */
export class TieredMemory {
  private hotMessages: Map<string, MessageMetadata> = new Map();
  private warmSummaries: Map<string, MessageSummary> = new Map();
  private coldFacts: Map<string, KeyFact> = new Map();
  private tierMetadata: Map<string, TierMetadata> = new Map();
  readonly config: TieredMemoryConfig;

  constructor(config?: TieredMemoryConfig) {
    this.config = config ?? defaultTieredMemoryConfig();
  }

  /** Add a message to the hot tier with session authority. */
  addMessage(message: MessageMetadata, importance: number): void {
    const meta = createTierMetadata(message.messageId, importance);

    // Apply TTL if configured
    if (this.config.sessionTtlSecs !== undefined) {
      message.expiresAt = Math.floor(Date.now() / 1000) +
        this.config.sessionTtlSecs;
    }

    this.hotMessages.set(message.messageId, message);
    this.tierMetadata.set(message.messageId, meta);
  }

  /** Add a message with canonical authority. */
  addCanonicalMessage(message: MessageMetadata, importance: number): void {
    const meta = createTierMetadata(message.messageId, importance, "canonical");
    this.hotMessages.set(message.messageId, message);
    this.tierMetadata.set(message.messageId, meta);
  }

  /** Record access to a message. */
  recordAccess(messageId: string): void {
    const meta = this.tierMetadata.get(messageId);
    if (meta) {
      recordAccess(meta);
    }
  }

  /** Demote a message from hot to warm. */
  demoteToWarm(messageId: string, summary: MessageSummary): void {
    const meta = this.tierMetadata.get(messageId);
    if (meta) {
      meta.tier = "warm";
    }
    this.warmSummaries.set(summary.summaryId, summary);
  }

  /** Demote a summary from warm to cold. */
  demoteToCold(summaryId: string, fact: KeyFact): void {
    this.warmSummaries.delete(summaryId);
    this.coldFacts.set(fact.factId, fact);
  }

  /** Promote a message back to hot tier. */
  promoteToHot(messageId: string): void {
    const meta = this.tierMetadata.get(messageId);
    if (meta) {
      meta.tier = "hot";
      recordAccess(meta);
    }
  }

  /** Get demotion candidates sorted by retention score (lowest first). */
  getDemotionCandidates(tier: MemoryTier, count: number): string[] {
    const candidates: [string, number][] = [];
    for (const [id, meta] of this.tierMetadata) {
      if (meta.tier === tier) {
        candidates.push([id, retentionScore(meta)]);
      }
    }
    candidates.sort((a, b) => a[1] - b[1]);
    return candidates.slice(0, count).map(([id]) => id);
  }

  /** Delete all hot-tier messages whose TTL has expired. */
  evictExpired(): number {
    const now = Math.floor(Date.now() / 1000);
    let evicted = 0;
    for (const [id, msg] of this.hotMessages) {
      if (msg.expiresAt !== undefined && msg.expiresAt <= now) {
        const meta = this.tierMetadata.get(id);
        // Never evict canonical entries
        if (meta?.authority === "canonical") continue;
        this.hotMessages.delete(id);
        this.tierMetadata.delete(id);
        evicted++;
      }
    }
    return evicted;
  }

  /** Fallback summarization (truncate to 75 words). */
  fallbackSummarize(content: string): string {
    const words = content.split(/\s+/);
    if (words.length <= 75) return content;
    return words.slice(0, 75).join(" ") + "...";
  }

  /** Get statistics. */
  getStats(): TieredMemoryStats {
    return {
      hotCount: this.hotMessages.size,
      warmCount: this.warmSummaries.size,
      coldCount: this.coldFacts.size,
      totalTracked: this.tierMetadata.size,
    };
  }

  /** Get a hot-tier message. */
  getHotMessage(messageId: string): MessageMetadata | undefined {
    return this.hotMessages.get(messageId);
  }

  /** Get tier metadata for a message. */
  getTierMetadata(messageId: string): TierMetadata | undefined {
    return this.tierMetadata.get(messageId);
  }
}
