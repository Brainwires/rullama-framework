/**
 * Prompting Learning Coordinator
 *
 * Tracks technique effectiveness over time and adapts selections.
 * Records outcomes, computes per-technique statistics, and identifies
 * techniques that consistently perform well (promotion candidates).
 */

import type { PromptingTechnique } from "./techniques.ts";

// ---------------------------------------------------------------------------
// TechniqueEffectivenessRecord
// ---------------------------------------------------------------------------

/** Record of technique effectiveness for a specific task execution. */
export interface TechniqueEffectivenessRecord {
  /** The prompting technique that was used. */
  readonly technique: PromptingTechnique;
  /** The cluster this task belongs to. */
  readonly clusterId: string;
  /** Description of the task that was executed. */
  readonly taskDescription: string;
  /** Whether the task completed successfully. */
  readonly success: boolean;
  /** Number of iterations consumed. */
  readonly iterationsUsed: number;
  /** Quality score from 0.0 to 1.0. */
  readonly qualityScore: number;
  /** Unix timestamp (ms) of the execution. */
  readonly timestamp: number;
}

// ---------------------------------------------------------------------------
// TechniqueStats
// ---------------------------------------------------------------------------

/** Aggregated statistics for a technique in a specific cluster. */
export interface TechniqueStats {
  /** Number of successful executions. */
  successCount: number;
  /** Number of failed executions. */
  failureCount: number;
  /** Average iterations used (EMA). */
  avgIterations: number;
  /** Average quality score (EMA). */
  avgQuality: number;
  /** Unix timestamp (ms) of the last execution. */
  lastUsed: number;
}

/** Create new TechniqueStats with initial values. */
export function createTechniqueStats(): TechniqueStats {
  return {
    successCount: 0,
    failureCount: 0,
    avgIterations: 0,
    avgQuality: 0,
    lastUsed: Date.now(),
  };
}

/** Calculate reliability (success rate). */
export function statsReliability(stats: TechniqueStats): number {
  const total = stats.successCount + stats.failureCount;
  return total === 0 ? 0 : stats.successCount / total;
}

/** Total number of uses. */
export function statsTotalUses(stats: TechniqueStats): number {
  return stats.successCount + stats.failureCount;
}

/** Update stats with new outcome using EMA (alpha = 0.3). */
export function updateTechniqueStats(
  stats: TechniqueStats,
  success: boolean,
  iterations: number,
  quality: number,
): void {
  if (success) {
    stats.successCount += 1;
  } else {
    stats.failureCount += 1;
  }

  const alpha = 0.3;
  stats.avgIterations = alpha * iterations + (1 - alpha) * stats.avgIterations;
  stats.avgQuality = alpha * quality + (1 - alpha) * stats.avgQuality;
  stats.lastUsed = Date.now();
}

// ---------------------------------------------------------------------------
// ClusterSummary
// ---------------------------------------------------------------------------

/** Summary of technique performance for a cluster. */
export interface ClusterSummary {
  /** The cluster identifier. */
  readonly clusterId: string;
  /** Total number of task executions in this cluster. */
  readonly totalExecutions: number;
  /** Per-technique performance statistics. */
  readonly techniques: ReadonlyMap<PromptingTechnique, TechniqueStats>;
}

/** Get the most effective technique in a cluster summary. */
export function bestTechnique(
  summary: ClusterSummary,
  minUses: number = 3,
): [PromptingTechnique, TechniqueStats] | undefined {
  let best: [PromptingTechnique, TechniqueStats] | undefined;
  let bestReliability = -Infinity;

  for (const [technique, stats] of summary.techniques) {
    if (statsTotalUses(stats) < minUses) continue;
    const rel = statsReliability(stats);
    if (
      rel > bestReliability ||
      (rel === bestReliability && best && stats.avgQuality > best[1].avgQuality)
    ) {
      bestReliability = rel;
      best = [technique, stats];
    }
  }

  return best;
}

/** Get techniques eligible for promotion in a cluster summary. */
export function promotableTechniques(
  summary: ClusterSummary,
  threshold: number,
  minUses: number,
): PromptingTechnique[] {
  const result: PromptingTechnique[] = [];
  for (const [technique, stats] of summary.techniques) {
    if (
      statsReliability(stats) >= threshold && statsTotalUses(stats) >= minUses
    ) {
      result.push(technique);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// PromptingLearningCoordinator
// ---------------------------------------------------------------------------

/** Default minimum reliability threshold for promotion. */
const DEFAULT_PROMOTION_THRESHOLD = 0.8;

/** Default minimum uses before promotion. */
const DEFAULT_MIN_USES = 5;

/** Composite key for the stats map. */
function statsKey(clusterId: string, technique: PromptingTechnique): string {
  return `${clusterId}:${technique}`;
}

/**
 * Coordinates learning and promotion of technique effectiveness.
 *
 * Tracks which techniques work well for which task clusters, computes
 * aggregated statistics, and identifies promotion candidates.
 */
export class PromptingLearningCoordinator {
  private readonly records: TechniqueEffectivenessRecord[] = [];
  private readonly techniqueStats = new Map<string, TechniqueStats>();
  private readonly promotionThreshold: number;
  private readonly minUsesForPromotion: number;

  /** Create a coordinator; `options` tune the promotion threshold and minimum uses. */
  constructor(options?: {
    promotionThreshold?: number;
    minUsesForPromotion?: number;
  }) {
    this.promotionThreshold = options?.promotionThreshold ??
      DEFAULT_PROMOTION_THRESHOLD;
    this.minUsesForPromotion = options?.minUsesForPromotion ?? DEFAULT_MIN_USES;
  }

  /**
   * Record outcome of using specific techniques.
   *
   * Called after task completion to track which techniques worked.
   */
  recordOutcome(
    clusterId: string,
    techniques: readonly PromptingTechnique[],
    taskDescription: string,
    success: boolean,
    iterations: number,
    qualityScore: number,
  ): void {
    const timestamp = Date.now();

    for (const technique of techniques) {
      const record: TechniqueEffectivenessRecord = {
        technique,
        clusterId,
        taskDescription,
        success,
        iterationsUsed: iterations,
        qualityScore,
        timestamp,
      };
      this.records.push(record);

      // Update aggregated stats
      const key = statsKey(clusterId, technique);
      let stats = this.techniqueStats.get(key);
      if (!stats) {
        stats = createTechniqueStats();
        this.techniqueStats.set(key, stats);
      }
      updateTechniqueStats(stats, success, iterations, qualityScore);
    }
  }

  /**
   * Check if a technique should be promoted.
   *
   * Promotion criteria:
   * - Reliability >= threshold (default 80%)
   * - Total uses >= minUses (default 5)
   */
  shouldPromote(clusterId: string, technique: PromptingTechnique): boolean {
    const stats = this.techniqueStats.get(statsKey(clusterId, technique));
    if (!stats) return false;
    return (
      statsReliability(stats) >= this.promotionThreshold &&
      statsTotalUses(stats) >= this.minUsesForPromotion
    );
  }

  /** Get all techniques that are eligible for promotion. */
  getPromotionCandidates(): Array<
    { clusterId: string; technique: PromptingTechnique }
  > {
    const candidates: Array<
      { clusterId: string; technique: PromptingTechnique }
    > = [];

    for (const [key, stats] of this.techniqueStats) {
      if (
        statsReliability(stats) >= this.promotionThreshold &&
        statsTotalUses(stats) >= this.minUsesForPromotion
      ) {
        const sepIdx = key.indexOf(":");
        const clusterId = key.slice(0, sepIdx);
        const technique = key.slice(sepIdx + 1) as PromptingTechnique;
        candidates.push({ clusterId, technique });
      }
    }

    return candidates;
  }

  /** Get statistics for a specific technique in a cluster. */
  getStats(
    clusterId: string,
    technique: PromptingTechnique,
  ): TechniqueStats | undefined {
    return this.techniqueStats.get(statsKey(clusterId, technique));
  }

  /** Get all statistics. */
  getAllStats(): ReadonlyMap<string, TechniqueStats> {
    return this.techniqueStats;
  }

  /** Get recent records (last N). */
  getRecentRecords(count: number): readonly TechniqueEffectivenessRecord[] {
    return this.records.slice(-count);
  }

  /** Get statistics summary for a cluster. */
  getClusterSummary(clusterId: string): ClusterSummary {
    let totalExecutions = 0;
    const techniques = new Map<PromptingTechnique, TechniqueStats>();

    for (const [key, stats] of this.techniqueStats) {
      const sepIdx = key.indexOf(":");
      const cid = key.slice(0, sepIdx);
      if (cid === clusterId) {
        const technique = key.slice(sepIdx + 1) as PromptingTechnique;
        totalExecutions += statsTotalUses(stats);
        techniques.set(technique, { ...stats });
      }
    }

    return { clusterId, totalExecutions, techniques };
  }

  /** Clear old records, keeping only the most recent N. */
  pruneOldRecords(keepCount: number): void {
    if (this.records.length > keepCount) {
      this.records.splice(0, this.records.length - keepCount);
    }
  }

  /** Get promotion thresholds. */
  getThresholds(): { promotionThreshold: number; minUsesForPromotion: number } {
    return {
      promotionThreshold: this.promotionThreshold,
      minUsesForPromotion: this.minUsesForPromotion,
    };
  }
}
