/**
 * Self-Evolving Learning Mechanism.
 *
 * Enables the system to learn from successful interactions without retraining.
 * Implements both local (per-session) and global (cross-session) memory.
 *
 * Equivalent to Rust's `rullama_agents::seal::learning` module.
 */

import type { ResponseConfidence as CoreResponseConfidence } from "@rullama/core";
import { categoryName, getSuggestion } from "@rullama/tool-runtime";
import {
  type QueryCore,
  queryCoreToSexp,
  type QuestionType,
} from "./query_core.ts";
import type { EntityType, ToolErrorCategory, ToolOutcome } from "./types.ts";

/**
 * Response confidence carrier used by SEAL learning. Imported from
 * `@rullama/core` since v0.11.0 (was previously a local duplicate type;
 * Rust's Phase 11a centralised it in `rullama-core::confidence`).
 *
 * SEAL uses its own threshold constants (lower bars than the core
 * `isHighConfidence` / `isLowConfidence` helpers, which are intended for
 * agent runtime decisions): SEAL considers a learning sample "low" below 0.4
 * and "high" at ≥ 0.7. Use the SEAL helpers below in this module.
 */
export type ResponseConfidence = CoreResponseConfidence;

const LOW_CONFIDENCE_THRESHOLD = 0.4;
const HIGH_CONFIDENCE_THRESHOLD = 0.7;

/** SEAL's low-confidence test: `score < 0.4`. */
export function isLowConfidence(c: ResponseConfidence): boolean {
  return c.score < LOW_CONFIDENCE_THRESHOLD;
}

/** SEAL's high-confidence test: `score >= 0.7`. */
export function isHighConfidence(c: ResponseConfidence): boolean {
  return c.score >= HIGH_CONFIDENCE_THRESHOLD;
}

// ─── TrackedEntity ──────────────────────────────────────────────────────────

/** A tracked entity in local memory. */
export class TrackedEntity {
  /** Entity name as mentioned in the conversation. */
  name: string;
  /** Entity category. */
  entity_type: EntityType;
  /** Distinct turn numbers at which the entity was mentioned, in order. */
  mention_turns: number[];
  /** Whether a query targeted this entity; never set by this package (kept for Rust parity). */
  was_queried = false;
  /** Whether the entity was modified; never set by this package (kept for Rust parity). */
  was_modified = false;
  /** `[relation, other entity]` pairs; never populated by this package (kept for Rust parity). */
  discovered_relations: [string, string][] = [];

  /** Track an entity first mentioned at `turn`. */
  constructor(name: string, entity_type: EntityType, turn: number) {
    this.name = name;
    this.entity_type = entity_type;
    this.mention_turns = [turn];
  }

  /** Append `turn` to `mention_turns` unless it is already present. */
  recordMention(turn: number): void {
    if (!this.mention_turns.includes(turn)) {
      this.mention_turns.push(turn);
    }
  }

  /** Number of distinct turns in which the entity was mentioned. */
  frequency(): number {
    return this.mention_turns.length;
  }
}

/** Record of a coreference resolution. */
export interface CoreferenceRecord {
  /** Surface text of the reference (e.g. `"it"`). */
  reference: string;
  /** Name of the antecedent it was resolved to. */
  resolved_to: string;
  /** Resolution confidence in `[0, 1]`. */
  confidence: number;
  /** Turn on which the resolution happened. */
  turn: number;
  /** User confirmation of the resolution; always `undefined` when recorded (nothing in this package confirms). */
  confirmed: boolean | undefined;
}

/** Record of a query execution. */
export interface QueryRecord {
  /** Question as asked. */
  original: string;
  /** Question after coreference rewriting (same as `original` if nothing was rewritten). */
  resolved: string;
  /** Question classification of the executed core. */
  question_type: QuestionType;
  /** S-expression rendering of the core (`queryCoreToSexp`), if available. */
  query_sexp: string | undefined;
  /** Turn on which the query ran. */
  turn: number;
  /** Whether execution was deemed successful. */
  success: boolean;
  /** Number of results returned. */
  result_count: number;
  /** Wall-clock execution time in milliseconds. */
  execution_time_ms: number;
}

// ─── LocalMemory ────────────────────────────────────────────────────────────

/** Local memory for a single conversation session. */
export class LocalMemory {
  /** Identifier of the conversation this memory belongs to. */
  conversation_id: string;
  /** Tracked entities keyed by name. */
  entities: Map<string, TrackedEntity> = new Map();
  /** Every coreference resolution recorded this session, oldest first. */
  coreference_log: CoreferenceRecord[] = [];
  /** Every query outcome recorded this session, oldest first. */
  query_history: QueryRecord[] = [];
  /** Most-recently-tracked entity names first; capped at 20 entries. */
  focus_stack: string[] = [];
  /** Current turn number; stamped onto new records. */
  current_turn = 0;

  /** Create empty session memory for `conversationId`. */
  constructor(conversationId: string) {
    this.conversation_id = conversationId;
  }

  /** Increment `current_turn`. */
  nextTurn(): void {
    this.current_turn += 1;
  }

  /** Record a mention of `name` on the current turn (creating the {@link TrackedEntity} if new) and move it to the top of the focus stack. */
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

  /** Append a {@link CoreferenceRecord} for the current turn with `confirmed` unset. */
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

  /** Append a {@link QueryRecord} for the current turn. */
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

  /** Up to `limit` tracked entities, most frequently mentioned first. */
  getFrequentEntities(limit: number): TrackedEntity[] {
    return Array.from(this.entities.values())
      .sort((a, b) => b.frequency() - a.frequency())
      .slice(0, limit);
  }

  /** The last `count` coreference records, newest first. */
  getRecentCoreferences(count: number): CoreferenceRecord[] {
    const recent = this.coreference_log.slice(-count);
    return recent.reverse();
  }

  /** Fraction of recorded queries of `question_type` that succeeded; 0.5 when there are none. */
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
  /** Random UUID assigned at construction. */
  id: string;
  /** Question classification the pattern applies to. */
  question_type: QuestionType;
  /** S-expression with entity names replaced by `${TYPE}` placeholders. */
  template: string;
  /** Entity types that must all be present for the pattern to match. */
  required_types: EntityType[];
  /** Successful uses recorded via {@link recordSuccess}. */
  success_count = 0;
  /** Failed uses recorded via {@link recordFailure}. */
  failure_count = 0;
  /** Exponential moving average (alpha 0.3) of result counts on success. */
  avg_results = 0.0;
  /** Unix time (seconds) the pattern was created. */
  created_at: number;
  /** Unix time (seconds) of the last success or failure. */
  last_used_at: number;

  /** Create an unused pattern with a fresh UUID and both timestamps set to now. */
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

  /** Success ratio; 0.5 before any use. */
  reliability(): number {
    const total = this.success_count + this.failure_count;
    if (total === 0) return 0.5;
    return this.success_count / total;
  }

  /** Count a success, refresh `last_used_at` and fold `result_count` into `avg_results`. */
  recordSuccess(result_count: number): void {
    this.success_count += 1;
    this.last_used_at = Math.floor(Date.now() / 1000);
    const alpha = 0.3;
    this.avg_results = alpha * result_count + (1 - alpha) * this.avg_results;
  }

  /** Count a failure and refresh `last_used_at`. */
  recordFailure(): void {
    this.failure_count += 1;
    this.last_used_at = Math.floor(Date.now() / 1000);
  }

  /** `true` when every `required_types` entry occurs in `types`. */
  matchesTypes(types: EntityType[]): boolean {
    return this.required_types.every((rt) => types.includes(rt));
  }
}

/** A learned coreference resolution pattern. */
export interface ResolutionPattern {
  /** Reference kind the pattern was learned from (e.g. `"singular_neutral"`). */
  reference_type: string;
  /** Entity type the reference tends to resolve to. */
  entity_type: EntityType;
  /** Optional surrounding-text pattern. */
  context_pattern: string | undefined;
  /** Times the resolution was confirmed correct. */
  success_count: number;
  /** Times the resolution was wrong. */
  failure_count: number;
}

// ─── ToolErrorPattern ───────────────────────────────────────────────────────

/** A learned tool error pattern for avoiding repeated failures. */
export class ToolErrorPattern {
  /** Tool that failed. */
  tool_name: string;
  /** Category name from `@rullama/tool-runtime`'s `categoryName`. */
  error_category: string;
  /** How many times this tool/category pair has failed; starts at 1. */
  occurrence_count = 1;
  /** Unix time (seconds) of the latest occurrence. */
  last_occurred: number;
  /** Remedy from `getSuggestion(error_category)`, if the category has one. */
  suggested_fix: string | undefined;
  /** Input shapes associated with the failure; never populated by this package (kept for Rust parity). */
  input_patterns: string[] = [];

  /** Record the first occurrence of `error_category` for `tool_name`, deriving the category name and suggested fix. */
  constructor(tool_name: string, error_category: ToolErrorCategory) {
    this.tool_name = tool_name;
    this.error_category = categoryName(error_category);
    this.last_occurred = Math.floor(Date.now() / 1000);
    this.suggested_fix = getSuggestion(error_category);
  }

  /** Increment `occurrence_count` and refresh `last_occurred`. */
  recordOccurrence(): void {
    this.occurrence_count += 1;
    this.last_occurred = Math.floor(Date.now() / 1000);
  }

  /** `true` once the pattern has occurred 3 or more times. */
  isFrequent(): boolean {
    return this.occurrence_count >= 3;
  }
}

// ─── ToolStats ──────────────────────────────────────────────────────────────

/** Tool execution statistics for learning. */
export class ToolStats {
  /** Successful invocations. */
  success_count = 0;
  /** Failed invocations. */
  failure_count = 0;
  /** Sum of retries across all invocations. */
  total_retries = 0;
  /** Exponential moving average (alpha 0.3) of execution time in milliseconds. */
  avg_execution_time_ms = 0;
  /** Unix time (seconds) of the last invocation; 0 until first use. */
  last_used = 0;

  /** Count a success, add `retries`, refresh `last_used` and fold `execution_time_ms` into the average. */
  recordSuccess(retries: number, execution_time_ms: number): void {
    this.success_count += 1;
    this.total_retries += retries;
    this.last_used = Math.floor(Date.now() / 1000);
    const alpha = 0.3;
    this.avg_execution_time_ms = alpha * execution_time_ms +
      (1 - alpha) * this.avg_execution_time_ms;
  }

  /** Count a failure, add `retries`, refresh `last_used` and fold `execution_time_ms` into the average. */
  recordFailure(retries: number, execution_time_ms: number): void {
    this.failure_count += 1;
    this.total_retries += retries;
    this.last_used = Math.floor(Date.now() / 1000);
    const alpha = 0.3;
    this.avg_execution_time_ms = alpha * execution_time_ms +
      (1 - alpha) * this.avg_execution_time_ms;
  }

  /** Success ratio; 0.5 before any invocation. */
  successRate(): number {
    const total = this.success_count + this.failure_count;
    if (total === 0) return 0.5;
    return this.success_count / total;
  }

  /** Mean retries per invocation; 0 before any invocation. */
  avgRetries(): number {
    const total = this.success_count + this.failure_count;
    if (total === 0) return 0.0;
    return this.total_retries / total;
  }
}

// ─── ConfidenceStats ────────────────────────────────────────────────────────

/** Response confidence statistics for learning prompt patterns. */
export class ConfidenceStats {
  /** Samples recorded. */
  sample_count = 0;
  /** Sum of sample scores (for the running mean). */
  confidence_sum = 0;
  /** Samples with score `< 0.4`. */
  low_confidence_count = 0;
  /** Samples with score `>= 0.7`. */
  high_confidence_count = 0;

  /** Add a sample, classifying it with {@link isLowConfidence} / {@link isHighConfidence}. */
  recordSample(confidence: ResponseConfidence): void {
    this.sample_count += 1;
    this.confidence_sum += confidence.score;
    if (isLowConfidence(confidence)) this.low_confidence_count += 1;
    else if (isHighConfidence(confidence)) this.high_confidence_count += 1;
  }

  /** Mean sample score; 0.5 before any sample. */
  avgConfidence(): number {
    if (this.sample_count === 0) return 0.5;
    return this.confidence_sum / this.sample_count;
  }

  /** Fraction of samples that were low-confidence; 0 before any sample. */
  lowConfidenceRatio(): number {
    if (this.sample_count === 0) return 0.0;
    return this.low_confidence_count / this.sample_count;
  }
}

/** A structured hint derived from behavioral knowledge (BKS). */
export interface PatternHint {
  /** Situation the hint applies to (a BKS context pattern or `run:<id>` for user corrections). */
  context_pattern: string;
  /** The guidance text itself. */
  rule: string;
  /** Confidence in `[0, 1]`. */
  confidence: number;
  /** Origin label: `"bks"` when synced from knowledge, `"user_feedback"` from the feedback bridge. */
  source: string;
}

// ─── GlobalMemory ───────────────────────────────────────────────────────────

/** Global memory for cross-session learning. */
export class GlobalMemory {
  /** Learned query patterns grouped by question type. */
  query_patterns: Map<QuestionType, QueryPattern[]> = new Map();
  /** Learned coreference patterns; never populated by this package (kept for Rust parity). */
  resolution_patterns: ResolutionPattern[] = [];
  /** Tool error patterns keyed by `"<tool>:<category>"`. */
  tool_error_patterns: Map<string, ToolErrorPattern> = new Map();
  /** Per-tool execution statistics keyed by tool name. */
  tool_stats: Map<string, ToolStats> = new Map();
  /** Aggregate response-confidence statistics. */
  confidence_stats: ConfidenceStats = new ConfidenceStats();
  /** Hints injected from BKS sync or user feedback, in insertion order. */
  pattern_hints: PatternHint[] = [];

  /** Append a hint. */
  addPatternHint(hint: PatternHint): void {
    this.pattern_hints.push(hint);
  }

  /** The live hints array (not a copy). */
  getPatternHints(): PatternHint[] {
    return this.pattern_hints;
  }

  /** Register a pattern under its `question_type`. */
  addPattern(pattern: QueryPattern): void {
    const patterns = this.query_patterns.get(pattern.question_type) ?? [];
    patterns.push(pattern);
    this.query_patterns.set(pattern.question_type, patterns);
  }

  /** Patterns for `question_type`, most reliable first (a sorted copy). */
  getPatterns(question_type: QuestionType): QueryPattern[] {
    const patterns = this.query_patterns.get(question_type);
    if (patterns === undefined) return [];
    return [...patterns].sort((a, b) => b.reliability() - a.reliability());
  }

  /** The most reliable pattern for `question_type` whose required types are all in `entity_types`. */
  getBestPattern(
    question_type: QuestionType,
    entity_types: EntityType[],
  ): QueryPattern | undefined {
    return this.getPatterns(question_type).find((p) =>
      p.matchesTypes(entity_types)
    );
  }

  /** Look up a pattern by id across all question types. */
  getPatternMut(id: string): QueryPattern | undefined {
    for (const patterns of this.query_patterns.values()) {
      const p = patterns.find((x) => x.id === id);
      if (p !== undefined) return p;
    }
    return undefined;
  }

  /** Drop patterns used at least `min_uses` times whose reliability is below `min_reliability`; lightly used patterns are kept. */
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

  /** Update the tool's {@link ToolStats}; on failure with an `errorCategory`, create or bump the matching {@link ToolErrorPattern}. */
  recordToolOutcome(outcome: ToolOutcome): void {
    const stats = this.tool_stats.get(outcome.toolName) ?? new ToolStats();
    this.tool_stats.set(outcome.toolName, stats);

    if (outcome.success) {
      stats.recordSuccess(outcome.retries, outcome.executionTimeMs);
    } else {
      stats.recordFailure(outcome.retries, outcome.executionTimeMs);
      if (outcome.errorCategory !== undefined) {
        const key = `${outcome.toolName}:${
          categoryName(outcome.errorCategory)
        }`;
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

  /** Add a sample to `confidence_stats`. */
  recordConfidence(confidence: ResponseConfidence): void {
    this.confidence_stats.recordSample(confidence);
  }

  /** Error patterns for `tool_name` that have occurred 3 or more times. */
  getCommonErrors(tool_name: string): ToolErrorPattern[] {
    return Array.from(this.tool_error_patterns.values()).filter((p) =>
      p.tool_name === tool_name && p.isFrequent()
    );
  }

  /** `"Common pitfalls for <tool>: fix; fix"` built from frequent errors with a suggested fix, or `undefined` if there are none. */
  getErrorPreventionHints(tool_name: string): string | undefined {
    const common = this.getCommonErrors(tool_name);
    if (common.length === 0) return undefined;
    const hints = common
      .map((e) => e.suggested_fix)
      .filter((s): s is string => s !== undefined);
    if (hints.length === 0) return undefined;
    return `Common pitfalls for ${tool_name}: ${hints.join("; ")}`;
  }

  /** The tool's success rate, or `undefined` if it has never been recorded. */
  getToolReliability(tool_name: string): number | undefined {
    const stats = this.tool_stats.get(tool_name);
    return stats === undefined ? undefined : stats.successRate();
  }
}

// ─── LearningCoordinator ────────────────────────────────────────────────────

/** Aggregate statistics emitted by {@link LearningCoordinator.getStats}. */
export interface LearningStats {
  /** Queries recorded in local memory this session. */
  session_queries: number;
  /** Distinct entities tracked this session. */
  session_entities: number;
  /** Coreference resolutions recorded this session. */
  session_coreferences: number;
  /** Total learned query patterns across all question types. */
  global_patterns: number;
  /** Sum of `success_count` over all patterns. */
  global_successes: number;
  /** Sum of `failure_count` over all patterns. */
  global_failures: number;
  /** `global_successes / (successes + failures)`, or 0.5 when no uses were recorded. */
  overall_reliability: number;
}

/** Learning coordinator that manages both local and global memory. */
export class LearningCoordinator {
  /** Per-conversation memory. */
  local: LocalMemory;
  /** Cross-session memory (patterns, tool stats, hints). */
  global: GlobalMemory = new GlobalMemory();
  // Rust had `_learning_rate: 0.3` unused; kept for parity but not referenced.
  private min_successes = 3;

  /** Create a coordinator with fresh local memory for `conversationId` and empty global memory. */
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
      const pattern = this.global.getBestPattern(
        core.question_type,
        entityTypes,
      );
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

    const existing = this.global.getBestPattern(
      query.question_type,
      required_types,
    );
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

  /** Render the core as an S-expression with each quoted entity name replaced by `${ENTITY_TYPE}`. */
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
        context +=
          `- ${e.name} (${e.entity_type}): ${e.frequency()} mentions\n`;
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

  /** Patterns with reliability `>= min_reliability` and at least `min_uses` uses, most reliable first. */
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

  /** Snapshot of session and global counters. */
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
  /** Delegates to {@link GlobalMemory.recordToolOutcome}. */
  recordToolOutcome(outcome: ToolOutcome): void {
    this.global.recordToolOutcome(outcome);
  }
  /** Delegates to {@link GlobalMemory.recordConfidence}. */
  recordConfidence(confidence: ResponseConfidence): void {
    this.global.recordConfidence(confidence);
  }
  /** Delegates to {@link GlobalMemory.getErrorPreventionHints}. */
  getErrorPreventionHints(tool_name: string): string | undefined {
    return this.global.getErrorPreventionHints(tool_name);
  }
  /** Delegates to {@link GlobalMemory.getToolReliability}. */
  getToolReliability(tool_name: string): number | undefined {
    return this.global.getToolReliability(tool_name);
  }
  /** Delegates to {@link GlobalMemory.getCommonErrors}. */
  getCommonErrors(tool_name: string): ToolErrorPattern[] {
    return this.global.getCommonErrors(tool_name);
  }
  /** Mean recorded response confidence (0.5 before any sample). */
  getAvgConfidence(): number {
    return this.global.confidence_stats.avgConfidence();
  }
  /** `true` when more than 30% of recorded responses were low-confidence. */
  hasConfidenceIssues(): boolean {
    return this.global.confidence_stats.lowConfidenceRatio() > 0.3;
  }
}
