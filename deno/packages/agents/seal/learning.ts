/**
 * Self-Evolving Learning Mechanism.
 *
 * Enables the system to learn from successful interactions without retraining.
 * Implements both local (per-session) and global (cross-session) memory.
 *
 * Equivalent to Rust's `brainwires_agents::seal::learning` module.
 */

import type { EntityType } from "@brainwires/core";
import type { ToolErrorCategory, ToolOutcome } from "@brainwires/tools";
import { categoryName, getSuggestion } from "@brainwires/tools";
import {
  type QueryCore,
  queryCoreToSexp,
  type QuestionType,
} from "./query_core.ts";

/**
 * Response confidence carrier used by SEAL learning. Mirrors the Rust
 * `brainwires_agents::confidence::ResponseConfidence` struct.
 *
 * The Deno agents package exposes a scalar-returning `extractResponseConfidence`
 * helper in mdap/planner.ts; this interface is the thin struct form that the
 * SEAL learning system records samples of.
 */
export interface ResponseConfidence {
  score: number;
}

const LOW_CONFIDENCE_THRESHOLD = 0.4;
const HIGH_CONFIDENCE_THRESHOLD = 0.7;

export function isLowConfidence(c: ResponseConfidence): boolean {
  return c.score < LOW_CONFIDENCE_THRESHOLD;
}

export function isHighConfidence(c: ResponseConfidence): boolean {
  return c.score >= HIGH_CONFIDENCE_THRESHOLD;
}

// ─── TrackedEntity ──────────────────────────────────────────────────────────

/** A tracked entity in local memory. */
export class TrackedEntity {
  name: string;
  entity_type: EntityType;
  mention_turns: number[];
  was_queried = false;
  was_modified = false;
  discovered_relations: [string, string][] = [];

  constructor(name: string, entity_type: EntityType, turn: number) {
    this.name = name;
    this.entity_type = entity_type;
    this.mention_turns = [turn];
  }

  recordMention(turn: number): void {
    if (!this.mention_turns.includes(turn)) {
      this.mention_turns.push(turn);
    }
  }

  frequency(): number {
    return this.mention_turns.length;
  }
}

/** Record of a coreference resolution. */
export interface CoreferenceRecord {
  reference: string;
  resolved_to: string;
  confidence: number;
  turn: number;
  confirmed: boolean | undefined;
}

/** Record of a query execution. */
export interface QueryRecord {
  original: string;
  resolved: string;
  question_type: QuestionType;
  query_sexp: string | undefined;
  turn: number;
  success: boolean;
  result_count: number;
  execution_time_ms: number;
}

// ─── LocalMemory ────────────────────────────────────────────────────────────

/** Local memory for a single conversation session. */
export class LocalMemory {
  conversation_id: string;
  entities: Map<string, TrackedEntity> = new Map();
  coreference_log: CoreferenceRecord[] = [];
  query_history: QueryRecord[] = [];
  focus_stack: string[] = [];
  current_turn = 0;

  constructor(conversationId: string) {
    this.conversation_id = conversationId;
  }

  nextTurn(): void {
    this.current_turn += 1;
  }

  trackEntity(name: string, entity_type: EntityType): void {
    const existing = this.entities.get(name);
    if (existing !== undefined) {
      existing.recordMention(this.current_turn);
    } else {
      this.entities.set(
        name,
        new TrackedEntity(name, entity_type, this.current_turn),
      );
    }
    this.focus_stack = this.focus_stack.filter((n) => n !== name);
    this.focus_stack.unshift(name);
    if (this.focus_stack.length > 20) {
      this.focus_stack.length = 20;
    }
  }

  recordCoreference(
    reference: string,
    resolved_to: string,
    confidence: number,
  ): void {
    this.coreference_log.push({
      reference,
      resolved_to,
      confidence,
      turn: this.current_turn,
      confirmed: undefined,
    });
  }

  recordQuery(
    original: string,
    resolved: string,
    question_type: QuestionType,
    query_sexp: string | undefined,
    success: boolean,
    result_count: number,
    execution_time_ms: number,
  ): void {
    this.query_history.push({
      original,
      resolved,
      question_type,
      query_sexp,
      turn: this.current_turn,
      success,
      result_count,
      execution_time_ms,
    });
  }

  getFrequentEntities(limit: number): TrackedEntity[] {
    return Array.from(this.entities.values())
      .sort((a, b) => b.frequency() - a.frequency())
      .slice(0, limit);
  }

  getRecentCoreferences(count: number): CoreferenceRecord[] {
    const recent = this.coreference_log.slice(-count);
    return recent.reverse();
  }

  getSuccessRate(question_type: QuestionType): number {
    const relevant = this.query_history.filter((q) =>
      q.question_type === question_type
    );
    if (relevant.length === 0) return 0.5;
    const successes = relevant.filter((q) => q.success).length;
    return successes / relevant.length;
  }
}

// ─── QueryPattern ───────────────────────────────────────────────────────────

/** A learned query pattern. */
export class QueryPattern {
  id: string;
  question_type: QuestionType;
  template: string;
  required_types: EntityType[];
  success_count = 0;
  failure_count = 0;
  avg_results = 0.0;
  created_at: number;
  last_used_at: number;

  constructor(
    question_type: QuestionType,
    template: string,
    required_types: EntityType[],
  ) {
    const now = Math.floor(Date.now() / 1000);
    this.id = crypto.randomUUID();
    this.question_type = question_type;
    this.template = template;
    this.required_types = required_types;
    this.created_at = now;
    this.last_used_at = now;
  }

  reliability(): number {
    const total = this.success_count + this.failure_count;
    if (total === 0) return 0.5;
    return this.success_count / total;
  }

  recordSuccess(result_count: number): void {
    this.success_count += 1;
    this.last_used_at = Math.floor(Date.now() / 1000);
    const alpha = 0.3;
    this.avg_results = alpha * result_count + (1 - alpha) * this.avg_results;
  }

  recordFailure(): void {
    this.failure_count += 1;
    this.last_used_at = Math.floor(Date.now() / 1000);
  }

  matchesTypes(types: EntityType[]): boolean {
    return this.required_types.every((rt) => types.includes(rt));
  }
}

/** A learned coreference resolution pattern. */
export interface ResolutionPattern {
  reference_type: string;
  entity_type: EntityType;
  context_pattern: string | undefined;
  success_count: number;
  failure_count: number;
}

// ─── ToolErrorPattern ───────────────────────────────────────────────────────

/** A learned tool error pattern for avoiding repeated failures. */
export class ToolErrorPattern {
  tool_name: string;
  error_category: string;
  occurrence_count = 1;
  last_occurred: number;
  suggested_fix: string | undefined;
  input_patterns: string[] = [];

  constructor(tool_name: string, error_category: ToolErrorCategory) {
    this.tool_name = tool_name;
    this.error_category = categoryName(error_category);
    this.last_occurred = Math.floor(Date.now() / 1000);
    this.suggested_fix = getSuggestion(error_category);
  }

  recordOccurrence(): void {
    this.occurrence_count += 1;
    this.last_occurred = Math.floor(Date.now() / 1000);
  }

  isFrequent(): boolean {
    return this.occurrence_count >= 3;
  }
}

// ─── ToolStats ──────────────────────────────────────────────────────────────

/** Tool execution statistics for learning. */
export class ToolStats {
  success_count = 0;
  failure_count = 0;
  total_retries = 0;
  avg_execution_time_ms = 0;
  last_used = 0;

  recordSuccess(retries: number, execution_time_ms: number): void {
    this.success_count += 1;
    this.total_retries += retries;
    this.last_used = Math.floor(Date.now() / 1000);
    const alpha = 0.3;
    this.avg_execution_time_ms = alpha * execution_time_ms +
      (1 - alpha) * this.avg_execution_time_ms;
  }

  recordFailure(retries: number, execution_time_ms: number): void {
    this.failure_count += 1;
    this.total_retries += retries;
    this.last_used = Math.floor(Date.now() / 1000);
    const alpha = 0.3;
    this.avg_execution_time_ms = alpha * execution_time_ms +
      (1 - alpha) * this.avg_execution_time_ms;
  }

  successRate(): number {
    const total = this.success_count + this.failure_count;
    if (total === 0) return 0.5;
    return this.success_count / total;
  }

  avgRetries(): number {
    const total = this.success_count + this.failure_count;
    if (total === 0) return 0.0;
    return this.total_retries / total;
  }
}

// ─── ConfidenceStats ────────────────────────────────────────────────────────

/** Response confidence statistics for learning prompt patterns. */
export class ConfidenceStats {
  sample_count = 0;
  confidence_sum = 0;
  low_confidence_count = 0;
  high_confidence_count = 0;

  recordSample(confidence: ResponseConfidence): void {
    this.sample_count += 1;
    this.confidence_sum += confidence.score;
    if (isLowConfidence(confidence)) this.low_confidence_count += 1;
    else if (isHighConfidence(confidence)) this.high_confidence_count += 1;
  }

  avgConfidence(): number {
    if (this.sample_count === 0) return 0.5;
    return this.confidence_sum / this.sample_count;
  }

  lowConfidenceRatio(): number {
    if (this.sample_count === 0) return 0.0;
    return this.low_confidence_count / this.sample_count;
  }
}

/** A structured hint derived from behavioral knowledge (BKS). */
export interface PatternHint {
  context_pattern: string;
  rule: string;
  confidence: number;
  source: string;
}

// ─── GlobalMemory ───────────────────────────────────────────────────────────

/** Global memory for cross-session learning. */
export class GlobalMemory {
  query_patterns: Map<QuestionType, QueryPattern[]> = new Map();
  resolution_patterns: ResolutionPattern[] = [];
  tool_error_patterns: Map<string, ToolErrorPattern> = new Map();
  tool_stats: Map<string, ToolStats> = new Map();
  confidence_stats: ConfidenceStats = new ConfidenceStats();
  pattern_hints: PatternHint[] = [];

  addPatternHint(hint: PatternHint): void {
    this.pattern_hints.push(hint);
  }

  getPatternHints(): PatternHint[] {
    return this.pattern_hints;
  }

  addPattern(pattern: QueryPattern): void {
    const patterns = this.query_patterns.get(pattern.question_type) ?? [];
    patterns.push(pattern);
    this.query_patterns.set(pattern.question_type, patterns);
  }

  getPatterns(question_type: QuestionType): QueryPattern[] {
    const patterns = this.query_patterns.get(question_type);
    if (patterns === undefined) return [];
    return [...patterns].sort((a, b) => b.reliability() - a.reliability());
  }

  getBestPattern(
    question_type: QuestionType,
    entity_types: EntityType[],
  ): QueryPattern | undefined {
    return this.getPatterns(question_type).find((p) => p.matchesTypes(entity_types));
  }

  getPatternMut(id: string): QueryPattern | undefined {
    for (const patterns of this.query_patterns.values()) {
      const p = patterns.find((x) => x.id === id);
      if (p !== undefined) return p;
    }
    return undefined;
  }

  prunePatterns(min_reliability: number, min_uses: number): void {
    for (const [qt, patterns] of this.query_patterns.entries()) {
      this.query_patterns.set(
        qt,
        patterns.filter((p) => {
          const total = p.success_count + p.failure_count;
          return total < min_uses || p.reliability() >= min_reliability;
        }),
      );
    }
  }

  recordToolOutcome(outcome: ToolOutcome): void {
    const stats = this.tool_stats.get(outcome.toolName) ?? new ToolStats();
    this.tool_stats.set(outcome.toolName, stats);

    if (outcome.success) {
      stats.recordSuccess(outcome.retries, outcome.executionTimeMs);
    } else {
      stats.recordFailure(outcome.retries, outcome.executionTimeMs);
      if (outcome.errorCategory !== undefined) {
        const key = `${outcome.toolName}:${categoryName(outcome.errorCategory)}`;
        const existing = this.tool_error_patterns.get(key);
        if (existing !== undefined) {
          existing.recordOccurrence();
        } else {
          this.tool_error_patterns.set(
            key,
            new ToolErrorPattern(outcome.toolName, outcome.errorCategory),
          );
        }
      }
    }
  }

  recordConfidence(confidence: ResponseConfidence): void {
    this.confidence_stats.recordSample(confidence);
  }

  getCommonErrors(tool_name: string): ToolErrorPattern[] {
    return Array.from(this.tool_error_patterns.values()).filter((p) =>
      p.tool_name === tool_name && p.isFrequent()
    );
  }

  getErrorPreventionHints(tool_name: string): string | undefined {
    const common = this.getCommonErrors(tool_name);
    if (common.length === 0) return undefined;
    const hints = common
      .map((e) => e.suggested_fix)
      .filter((s): s is string => s !== undefined);
    if (hints.length === 0) return undefined;
    return `Common pitfalls for ${tool_name}: ${hints.join("; ")}`;
  }

  getToolReliability(tool_name: string): number | undefined {
    const stats = this.tool_stats.get(tool_name);
    return stats === undefined ? undefined : stats.successRate();
  }
}

// ─── LearningCoordinator ────────────────────────────────────────────────────

/** Aggregate statistics emitted by {@link LearningCoordinator.getStats}. */
export interface LearningStats {
  session_queries: number;
  session_entities: number;
  session_coreferences: number;
  global_patterns: number;
  global_successes: number;
  global_failures: number;
  overall_reliability: number;
}

/** Learning coordinator that manages both local and global memory. */
export class LearningCoordinator {
  local: LocalMemory;
  global: GlobalMemory = new GlobalMemory();
  // Rust had `_learning_rate: 0.3` unused; kept for parity but not referenced.
  private min_successes = 3;

  constructor(conversationId: string) {
    this.local = new LocalMemory(conversationId);
  }

  /** Process a query — returns a matching pattern if one exists. */
  processQuery(
    _original: string,
    _resolved: string,
    core: QueryCore | undefined,
    turn: number,
  ): QueryPattern | undefined {
    this.local.current_turn = turn;
    if (core !== undefined) {
      const entityTypes = core.entities.map(([, t]) => t);
      const pattern = this.global.getBestPattern(core.question_type, entityTypes);
      if (pattern !== undefined) return pattern;
    }
    return undefined;
  }

  /** Record the outcome of a query execution. */
  recordOutcome(
    pattern_id: string | undefined,
    success: boolean,
    result_count: number,
    query_core: QueryCore | undefined,
    execution_time_ms: number,
  ): void {
    if (pattern_id !== undefined) {
      const pattern = this.global.getPatternMut(pattern_id);
      if (pattern !== undefined) {
        if (success) pattern.recordSuccess(result_count);
        else pattern.recordFailure();
      }
    }

    if (query_core !== undefined) {
      this.local.recordQuery(
        query_core.original,
        query_core.resolved ?? query_core.original,
        query_core.question_type,
        queryCoreToSexp(query_core),
        success,
        result_count,
        execution_time_ms,
      );

      if (success && pattern_id === undefined && result_count > 0) {
        this.learnPattern(query_core, result_count);
      }
    }
  }

  /** Learn a new pattern from a successful query. */
  learnPattern(query: QueryCore, result_count: number): string | undefined {
    if (result_count === 0 || result_count > 100) return undefined;

    const template = this.generalizeQuery(query);
    const required_types = query.entities.map(([, t]) => t);

    const existing = this.global.getBestPattern(query.question_type, required_types);
    if (existing !== undefined && existing.template === template) {
      return undefined;
    }

    const pattern = new QueryPattern(
      query.question_type,
      template,
      required_types,
    );
    pattern.recordSuccess(result_count);
    const id = pattern.id;
    this.global.addPattern(pattern);
    return id;
  }

  private generalizeQuery(query: QueryCore): string {
    let template = queryCoreToSexp(query);
    for (const [name, entity_type] of query.entities) {
      const placeholder = `\${${entity_type.toUpperCase()}}`;
      template = template.split(`"${name}"`).join(placeholder);
    }
    return template;
  }

  /** Build a human-readable context block for prompt injection. */
  getContextForPrompt(): string {
    let context = "";
    const frequent = this.local.getFrequentEntities(5);
    if (frequent.length > 0) {
      context += "Frequently referenced entities:\n";
      for (const e of frequent) {
        context += `- ${e.name} (${e.entity_type}): ${e.frequency()} mentions\n`;
      }
      context += "\n";
    }

    const types: QuestionType[] = ["definition", "location", "dependency"];
    const titleCase = (t: QuestionType): string =>
      t.split("_").map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join("");
    for (const qt of types) {
      const patterns = this.global.getPatterns(qt);
      const good = patterns.filter((p) =>
        p.reliability() > 0.7 && p.success_count >= this.min_successes
      ).slice(0, 2);
      if (good.length > 0) {
        context += `Effective ${titleCase(qt)} patterns:\n`;
        for (const p of good) {
          context += `- ${p.template} (${
            Math.floor(p.reliability() * 100)
          }% reliable)\n`;
        }
        context += "\n";
      }
    }
    return context;
  }

  getPromotablePatterns(
    min_reliability: number,
    min_uses: number,
  ): QueryPattern[] {
    const promotable: QueryPattern[] = [];
    for (const patterns of this.global.query_patterns.values()) {
      for (const pattern of patterns) {
        const total = pattern.success_count + pattern.failure_count;
        if (pattern.reliability() >= min_reliability && total >= min_uses) {
          promotable.push(pattern);
        }
      }
    }
    promotable.sort((a, b) => b.reliability() - a.reliability());
    return promotable;
  }

  getStats(): LearningStats {
    let total_patterns = 0;
    let total_successes = 0;
    let total_failures = 0;
    for (const patterns of this.global.query_patterns.values()) {
      total_patterns += patterns.length;
      for (const p of patterns) {
        total_successes += p.success_count;
        total_failures += p.failure_count;
      }
    }
    const total = total_successes + total_failures;
    return {
      session_queries: this.local.query_history.length,
      session_entities: this.local.entities.size,
      session_coreferences: this.local.coreference_log.length,
      global_patterns: total_patterns,
      global_successes: total_successes,
      global_failures: total_failures,
      overall_reliability: total > 0 ? total_successes / total : 0.5,
    };
  }

  // Delegation helpers.
  recordToolOutcome(outcome: ToolOutcome): void {
    this.global.recordToolOutcome(outcome);
  }
  recordConfidence(confidence: ResponseConfidence): void {
    this.global.recordConfidence(confidence);
  }
  getErrorPreventionHints(tool_name: string): string | undefined {
    return this.global.getErrorPreventionHints(tool_name);
  }
  getToolReliability(tool_name: string): number | undefined {
    return this.global.getToolReliability(tool_name);
  }
  getCommonErrors(tool_name: string): ToolErrorPattern[] {
    return this.global.getCommonErrors(tool_name);
  }
  getAvgConfidence(): number {
    return this.global.confidence_stats.avgConfidence();
  }
  hasConfidenceIssues(): boolean {
    return this.global.confidence_stats.lowConfidenceRatio() > 0.3;
  }
}
