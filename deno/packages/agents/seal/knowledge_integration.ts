/**
 * SEAL + Knowledge System integration.
 *
 * Bridges SEAL's entity-centric learning with a Knowledge System's behavioral
 * truths (BKS) and personal facts (PKS). On the Rust side this module talks to
 * concrete `BehavioralKnowledgeCache` / `PersonalKnowledgeCache` types backed
 * by LanceDB. On the Deno side the `@brainwires/knowledge` package is
 * interface-only — the concrete BKS/PKS work happens Rust-side — so this
 * module is structured as interface wiring: callers inject the cache shapes
 * (typically via an RPC bridge) and the coordinator glues them to SEAL.
 *
 * Equivalent to Rust's `brainwires_agents::seal::knowledge_integration` module.
 */

import type { QueryPattern } from "./learning.ts";
import type { LearningCoordinator, PatternHint } from "./learning.ts";
import type { QuestionType } from "./query_core.ts";
import type { ResolvedReference, SealProcessingResult } from "./types.ts";

// ─── Types expected from @brainwires/knowledge (interfaces) ────────────────

/** Category of a behavioral truth. */
export type TruthCategory =
  | "command_usage"
  | "task_strategy"
  | "error_recovery"
  | "convention"
  | "capability"
  | "other";

/** How a behavioral truth was sourced. */
export type TruthSource =
  | "manual"
  | "success_pattern"
  | "failure_pattern"
  | "user_correction";

/** A behavioral truth entry (shape matches Rust `BehavioralTruth`). */
export interface BehavioralTruth {
  category: TruthCategory;
  context_pattern: string;
  rule: string;
  rationale: string;
  source: TruthSource;
  confidence: number;
  author: string | undefined;
}

export function newBehavioralTruth(
  category: TruthCategory,
  context_pattern: string,
  rule: string,
  rationale: string,
  source: TruthSource,
  author: string | undefined,
): BehavioralTruth {
  return {
    category,
    context_pattern,
    rule,
    rationale,
    source,
    confidence: 1.0,
    author,
  };
}

/** A scored (truth, relevance) pair from a BKS lookup. */
export type ScoredTruth = [BehavioralTruth, number];

/** A personal fact entry. */
export interface PersonalFact {
  key: string;
  value: string;
  confidence: number;
  deleted: boolean;
}

/**
 * Behavioral knowledge cache — concrete implementation lives Rust-side.
 * Deno consumers inject an object matching this shape (e.g., an RPC bridge).
 */
export interface BehavioralKnowledgeCache {
  getMatchingTruthsWithScores(
    query: string,
    min_confidence: number,
    limit: number,
  ): Promise<ScoredTruth[]>;
  getReliableTruths(
    min_confidence: number,
    days: number,
  ): Promise<BehavioralTruth[]>;
  queueSubmission(truth: BehavioralTruth): Promise<void>;
}

/** Personal knowledge cache — concrete implementation lives Rust-side. */
export interface PersonalKnowledgeCache {
  getAllFacts(): Promise<PersonalFact[]>;
  upsertFactSimple(
    key: string,
    value: string,
    confidence: number,
    local_only: boolean,
  ): Promise<void>;
}

// ─── Config ─────────────────────────────────────────────────────────────────

export type EntityResolutionStrategy =
  | { kind: "seal_first" }
  | { kind: "pks_context_first" }
  | { kind: "hybrid"; seal_weight: number; pks_weight: number };

/** Configuration for SEAL + Knowledge integration. */
export interface IntegrationConfig {
  enabled: boolean;
  seal_to_knowledge: boolean;
  knowledge_to_seal: boolean;
  min_seal_quality_for_bks_boost: number;
  min_seal_quality_for_pks_boost: number;
  pattern_promotion_threshold: number;
  min_pattern_uses: number;
  cache_bks_in_seal: boolean;
  entity_resolution_strategy: EntityResolutionStrategy;
  seal_weight: number;
  bks_weight: number;
  pks_weight: number;
}

export const DEFAULT_PATTERN_PROMOTION_THRESHOLD = 0.8;

export function defaultIntegrationConfig(): IntegrationConfig {
  return {
    enabled: true,
    seal_to_knowledge: true,
    knowledge_to_seal: true,
    min_seal_quality_for_bks_boost: 0.7,
    min_seal_quality_for_pks_boost: 0.5,
    pattern_promotion_threshold: DEFAULT_PATTERN_PROMOTION_THRESHOLD,
    min_pattern_uses: 5,
    cache_bks_in_seal: true,
    entity_resolution_strategy: {
      kind: "hybrid",
      seal_weight: 0.6,
      pks_weight: 0.4,
    },
    seal_weight: 0.5,
    bks_weight: 0.3,
    pks_weight: 0.2,
  };
}

export function integrationConfigSealToKnowledgeOnly(): IntegrationConfig {
  return { ...defaultIntegrationConfig(), knowledge_to_seal: false };
}

export function integrationConfigDisabled(): IntegrationConfig {
  return { ...defaultIntegrationConfig(), enabled: false };
}

export function validateIntegrationConfig(c: IntegrationConfig): void {
  if (c.min_seal_quality_for_bks_boost < 0 || c.min_seal_quality_for_bks_boost > 1) {
    throw new Error("min_seal_quality_for_bks_boost must be between 0.0 and 1.0");
  }
  if (c.pattern_promotion_threshold < 0 || c.pattern_promotion_threshold > 1) {
    throw new Error("pattern_promotion_threshold must be between 0.0 and 1.0");
  }
  const sum = c.seal_weight + c.bks_weight + c.pks_weight;
  if (Math.abs(sum - 1.0) > 0.01) {
    throw new Error(
      `Confidence weights must sum to 1.0 (got: ${sum.toFixed(2)})`,
    );
  }
}

// ─── Coordinator ────────────────────────────────────────────────────────────

/**
 * Coordinator bridging SEAL with a knowledge system. Deno callers supply
 * cache implementations — usually thin proxies to a Rust-side BKS/PKS server.
 */
export class SealKnowledgeCoordinator {
  private bks_cache: BehavioralKnowledgeCache;
  private pks_cache: PersonalKnowledgeCache;
  readonly integrationConfig: IntegrationConfig;

  constructor(
    bks_cache: BehavioralKnowledgeCache,
    pks_cache: PersonalKnowledgeCache,
    config: IntegrationConfig = defaultIntegrationConfig(),
  ) {
    validateIntegrationConfig(config);
    this.bks_cache = bks_cache;
    this.pks_cache = pks_cache;
    this.integrationConfig = config;
  }

  /** PKS context for SEAL entity resolutions. */
  async getPksContext(
    seal_result: SealProcessingResult,
  ): Promise<string | undefined> {
    if (!this.integrationConfig.enabled) return undefined;
    if (
      seal_result.quality_score <
        this.integrationConfig.min_seal_quality_for_pks_boost
    ) return undefined;

    const entities = seal_result.resolutions.map((r) => r.antecedent);
    if (entities.length === 0) return undefined;

    const allFacts = await this.pks_cache.getAllFacts();
    const contextParts: string[] = [];

    for (const entity of entities) {
      const relevant = allFacts.filter((f) =>
        !f.deleted && (f.key.includes(entity) || f.value.includes(entity))
      );
      if (relevant.length === 0) continue;
      contextParts.push(`\n**${entity}:**`);
      for (const fact of relevant) {
        if (fact.confidence >= 0.5) {
          contextParts.push(
            `  - ${fact.value} (confidence: ${fact.confidence.toFixed(2)})`,
          );
        }
      }
    }

    if (contextParts.length === 0) return undefined;
    return `# PERSONAL CONTEXT\n\nRelevant facts about entities mentioned:\n${
      contextParts.join("\n")
    }`;
  }

  /** BKS context for query. */
  async getBksContext(query: string): Promise<string | undefined> {
    if (!this.integrationConfig.enabled) return undefined;
    const truths = await this.bks_cache.getMatchingTruthsWithScores(query, 0.5, 5);
    if (truths.length === 0) return undefined;

    const parts: string[] = ["# BEHAVIORAL KNOWLEDGE\n"];
    parts.push("Learned patterns that may be relevant:\n");
    for (const [truth, score] of truths) {
      parts.push(
        `\n**${truth.context_pattern}** (confidence: ${
          truth.confidence.toFixed(2)
        }, relevance: ${score.toFixed(2)}):`,
      );
      parts.push(`  Rule: ${truth.rule}`);
      parts.push(`  Why: ${truth.rationale}`);
    }
    return parts.join("\n");
  }

  /** Combine SEAL quality with optional BKS and PKS confidences. */
  harmonizeConfidence(
    seal_quality: number,
    bks_confidence: number | undefined,
    pks_confidence: number | undefined,
  ): number {
    let weightedSum = seal_quality * this.integrationConfig.seal_weight;
    let totalWeight = this.integrationConfig.seal_weight;
    if (bks_confidence !== undefined) {
      weightedSum += bks_confidence * this.integrationConfig.bks_weight;
      totalWeight += this.integrationConfig.bks_weight;
    }
    if (pks_confidence !== undefined) {
      weightedSum += pks_confidence * this.integrationConfig.pks_weight;
      totalWeight += this.integrationConfig.pks_weight;
    }
    return totalWeight > 0 ? Math.min(weightedSum / totalWeight, 1.0) : seal_quality;
  }

  /** Quality-adjusted retrieval threshold. */
  adjustRetrievalThreshold(base: number, seal_quality: number): number {
    return base * Math.max(0.7 + 0.3 * seal_quality, 0.5);
  }

  /** Check promotion criteria and, if met, queue a BKS submission. */
  async checkAndPromotePattern(
    pattern: QueryPattern,
    execution_context: string,
  ): Promise<BehavioralTruth | undefined> {
    if (
      !this.integrationConfig.enabled ||
      !this.integrationConfig.seal_to_knowledge
    ) return undefined;
    if (pattern.reliability() < this.integrationConfig.pattern_promotion_threshold) {
      return undefined;
    }
    const total = pattern.success_count + pattern.failure_count;
    if (total < this.integrationConfig.min_pattern_uses) return undefined;

    const truth = newBehavioralTruth(
      this.inferCategory(pattern.question_type),
      execution_context,
      this.generalizePatternToRule(pattern),
      `Learned from ${pattern.success_count} successful executions with ${
        (pattern.reliability() * 100).toFixed(1)
      }% reliability (SEAL pattern)`,
      "success_pattern",
      undefined,
    );
    await this.bks_cache.queueSubmission(truth);
    return truth;
  }

  /** Sync high-confidence BKS truths into SEAL's global memory. */
  async syncBksToSeal(seal_learning: LearningCoordinator): Promise<number> {
    if (
      !this.integrationConfig.enabled ||
      !this.integrationConfig.knowledge_to_seal ||
      !this.integrationConfig.cache_bks_in_seal
    ) return 0;

    const truths = await this.bks_cache.getReliableTruths(0.7, 30);
    let loaded = 0;
    for (const truth of truths) {
      const hint = this.truthToPatternHint(truth);
      if (hint !== undefined) {
        seal_learning.global.addPatternHint(hint);
        loaded += 1;
      }
    }
    return loaded;
  }

  /** Track SEAL entity resolutions as recent-entity facts in PKS. */
  async observeSealResolutions(
    resolutions: readonly ResolvedReference[],
  ): Promise<void> {
    if (!this.integrationConfig.enabled) return;
    for (const resolution of resolutions) {
      const key = `recent_entity:${resolution.antecedent}`;
      await this.pks_cache.upsertFactSimple(
        key,
        resolution.antecedent,
        resolution.confidence,
        true,
      );
    }
  }

  /** Record a tool failure pattern as a BKS entry. */
  async recordToolFailure(
    tool_name: string,
    error_message: string,
    context: string,
  ): Promise<void> {
    if (
      !this.integrationConfig.enabled ||
      !this.integrationConfig.seal_to_knowledge
    ) return;
    const truth = newBehavioralTruth(
      "error_recovery",
      context,
      `Tool '${tool_name}' commonly fails with: ${error_message}`,
      "Observed from validation failures",
      "failure_pattern",
      undefined,
    );
    await this.bks_cache.queueSubmission(truth);
  }

  /** Accessors mirroring Rust `get_pks_cache` / `get_bks_cache`. */
  getPksCache(): PersonalKnowledgeCache {
    return this.pks_cache;
  }
  getBksCache(): BehavioralKnowledgeCache {
    return this.bks_cache;
  }

  private inferCategory(qt: QuestionType): TruthCategory {
    switch (qt) {
      case "definition":
        return "command_usage";
      case "dependency":
      case "location":
      case "count":
      case "superlative":
      default:
        return "task_strategy";
    }
  }

  private generalizePatternToRule(pattern: QueryPattern): string {
    const typesStr = pattern.required_types.join(", ");
    return `For '${pattern.question_type}' queries about ${typesStr}, use pattern: ${pattern.template}`;
  }

  private truthToPatternHint(t: BehavioralTruth): PatternHint | undefined {
    return {
      context_pattern: t.context_pattern,
      rule: t.rule,
      confidence: t.confidence,
      source: "bks",
    };
  }
}
