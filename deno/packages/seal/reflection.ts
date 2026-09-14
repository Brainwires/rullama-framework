/**
 * Reflection Module for Error Detection and Correction.
 *
 * Provides post-execution analysis to detect issues and suggest corrections.
 *
 * Equivalent to Rust's `rullama_agents::seal::reflection` module.
 */

import type { RelationshipGraphT } from "./types.ts";
import type { LearningCoordinator } from "./learning.ts";
import {
  type QueryCore,
  type QueryExecutor,
  type QueryExpr,
  type QueryOp,
  type QueryResult,
  relationName,
  relationToEdgeType,
  type RelationType,
} from "./query_core.ts";

// ─── ErrorType ──────────────────────────────────────────────────────────────

/** Classification of a problem found while validating or reflecting on a query. */
export type ErrorType =
  | { kind: "empty_result" }
  | { kind: "result_overflow" }
  | { kind: "entity_not_found"; name: string }
  | { kind: "relation_mismatch"; message: string }
  | { kind: "coreference_failure"; reference: string }
  | { kind: "schema_alignment"; message: string }
  | { kind: "timeout" }
  | { kind: "unknown"; message: string };

/** Human-readable description of an error type (matches the Rust `Display` text). */
export function errorTypeDescription(e: ErrorType): string {
  switch (e.kind) {
    case "empty_result":
      return "Query returned no results";
    case "result_overflow":
      return "Query returned too many results";
    case "entity_not_found":
      return `Entity '${e.name}' not found`;
    case "relation_mismatch":
      return `Relationship '${e.message}' does not apply`;
    case "coreference_failure":
      return `Could not resolve reference '${e.reference}'`;
    case "schema_alignment":
      return `Schema alignment issue: ${e.message}`;
    case "timeout":
      return "Query execution timed out";
    case "unknown":
      return `Unknown error: ${e.message}`;
  }
}

/** Key for grouping error patterns (used where Rust stores ErrorType in a HashMap). */
export function errorTypeKey(e: ErrorType): string {
  switch (e.kind) {
    case "entity_not_found":
      return `entity_not_found:${e.name}`;
    case "relation_mismatch":
      return `relation_mismatch:${e.message}`;
    case "coreference_failure":
      return `coreference_failure:${e.reference}`;
    case "schema_alignment":
      return `schema_alignment:${e.message}`;
    case "unknown":
      return `unknown:${e.message}`;
    default:
      return e.kind;
  }
}

// ─── Severity ───────────────────────────────────────────────────────────────

/** Issue severity, ordered `info` < `warning` < `error` < `critical`. */
export type Severity = "info" | "warning" | "error" | "critical";

const SEVERITY_ORDER: Record<Severity, number> = {
  info: 0,
  warning: 1,
  error: 2,
  critical: 3,
};

/** `true` when `a` is at least as severe as `b`. */
export function severityAtLeast(a: Severity, b: Severity): boolean {
  return SEVERITY_ORDER[a] >= SEVERITY_ORDER[b];
}

/** Comparator for severities: negative when `a` is milder than `b`, zero when equal, positive when more severe. */
export function severityCompare(a: Severity, b: Severity): number {
  return SEVERITY_ORDER[a] - SEVERITY_ORDER[b];
}

// ─── SuggestedFix ───────────────────────────────────────────────────────────

/**
 * A remedy attached to an {@link Issue}. Only `resolve_entity` is acted on by
 * `ReflectionModule.attemptCorrection`; the others are advisory.
 */
export type SuggestedFix =
  | { kind: "retry_with_query"; query: QueryCore }
  | { kind: "expand_scope"; relation: string }
  | { kind: "narrow_scope"; filter: string }
  | { kind: "resolve_entity"; original: string; suggested: string }
  | { kind: "add_relation"; from: string; to: string; relation: string }
  | { kind: "manual_intervention"; message: string };

/** Human-readable description of a suggested fix. */
export function suggestedFixDescription(f: SuggestedFix): string {
  switch (f.kind) {
    case "retry_with_query":
      return "Retry with modified query";
    case "expand_scope":
      return `Expand scope to include ${f.relation} relationships`;
    case "narrow_scope":
      return `Narrow scope with filter: ${f.filter}`;
    case "resolve_entity":
      return `Resolve '${f.original}' as '${f.suggested}'`;
    case "add_relation":
      return `Add ${f.relation} relationship from ${f.from} to ${f.to}`;
    case "manual_intervention":
      return `Manual intervention: ${f.message}`;
  }
}

// ─── Issue ──────────────────────────────────────────────────────────────────

/** An issue detected during reflection. */
export class Issue {
  /** What kind of problem this is. */
  error_type: ErrorType;
  /** How serious the problem is. */
  severity: Severity;
  /** Human-readable explanation. */
  message: string;
  /** Remedies added via {@link withFix}, in insertion order. */
  suggested_fixes: SuggestedFix[] = [];
  /** Optional origin label (e.g. the relation name), set via {@link withSource}. */
  source: string | undefined;

  /** Create an issue with no fixes and no source. */
  constructor(error_type: ErrorType, severity: Severity, message: string) {
    this.error_type = error_type;
    this.severity = severity;
    this.message = message;
  }

  /** Append a suggested fix and return `this` for chaining. */
  withFix(fix: SuggestedFix): Issue {
    this.suggested_fixes.push(fix);
    return this;
  }

  /** Set the source label and return `this` for chaining. */
  withSource(source: string): Issue {
    this.source = source;
    return this;
  }
}

/** Record of a correction attempt. */
export interface CorrectionRecord {
  /** The issue the correction addressed. */
  issue: Issue;
  /** The fix that was applied. */
  fix_applied: SuggestedFix;
  /** Whether applying the fix produced a corrected query. */
  success: boolean;
  /** Unix time (seconds) the correction was recorded. */
  timestamp: number;
}

// ─── ReflectionReport ───────────────────────────────────────────────────────

/** Reflection report. */
export class ReflectionReport {
  /** Deep copy of the analysed query core. */
  query: QueryCore;
  /** Deep copy of the analysed execution result. */
  result: QueryResult;
  /** Issues found by `ReflectionModule.analyze`. */
  issues: Issue[] = [];
  /** Quality in `[0, 1]`: 0 on an error result, 0.3 on an empty result, 0.6 on overflow, minus 0.2 per missing entity. */
  quality_score = 1.0;
  /** Set to `true` once `ReflectionModule.attemptCorrection` has run on a report with issues. */
  correction_attempted = false;
  /** Query produced by a successful `resolve_entity` correction, if any. */
  corrected_query: QueryCore | undefined;
  /** Result of re-executing `corrected_query`; never set by this module (callers may fill it in). */
  corrected_result: QueryResult | undefined;

  /** Start a report for `query`/`result` with no issues and quality 1.0. */
  constructor(query: QueryCore, result: QueryResult) {
    this.query = query;
    this.result = result;
  }

  /** `true` when `quality_score >= 0.5` and no issue is `error` or `critical`. */
  isAcceptable(): boolean {
    return this.quality_score >= 0.5 &&
      !this.issues.some((i) => severityAtLeast(i.severity, "error"));
  }

  /** The highest severity among `issues`, or `undefined` when there are none. */
  maxSeverity(): Severity | undefined {
    if (this.issues.length === 0) return undefined;
    let max = this.issues[0].severity;
    for (const i of this.issues) {
      if (severityCompare(i.severity, max) > 0) max = i.severity;
    }
    return max;
  }
}

// ─── ReflectionConfig ───────────────────────────────────────────────────────

/** Tuning knobs for {@link ReflectionModule}. */
export interface ReflectionConfig {
  /** Result count above which a `result_overflow` warning is raised. */
  max_results: number;
  /** Minimum expected result count; stored for parity with Rust but not consulted by `analyze`. */
  min_results: number;
  /** Retry budget for corrections; stored for parity with Rust but not consulted by `attemptCorrection`. */
  max_retries: number;
  /** When `false`, `attemptCorrection` returns `false` without doing anything. */
  auto_correct: boolean;
}

/** Defaults: `max_results` 100, `min_results` 1, `max_retries` 2, `auto_correct` true. */
export function defaultReflectionConfig(): ReflectionConfig {
  return {
    max_results: 100,
    min_results: 1,
    max_retries: 2,
    auto_correct: true,
  };
}

// ─── ReflectionModule ───────────────────────────────────────────────────────

/** Reflection module for analyzing and correcting query results. */
export class ReflectionModule {
  private config: ReflectionConfig;
  private error_patterns: Map<string, number> = new Map();
  private correction_history: CorrectionRecord[] = [];

  /** Create a module with empty error statistics and correction history. */
  constructor(config: ReflectionConfig = defaultReflectionConfig()) {
    this.config = config;
  }

  /** Analyze a query result and produce a reflection report. */
  analyze(
    query: QueryCore,
    result: QueryResult,
    graph: RelationshipGraphT,
  ): ReflectionReport {
    const report = new ReflectionReport(
      structuredClone(query),
      structuredClone(result),
    );

    if (result.error !== undefined) {
      report.issues.push(
        new Issue(
          { kind: "unknown", message: result.error },
          "error",
          result.error,
        ),
      );
      report.quality_score = 0;
      return report;
    }

    if (result.values.length === 0 && result.count !== 0) {
      report.issues.push(this.analyzeEmptyResult(query, graph));
      report.quality_score = 0.3;
    }

    if (result.values.length > this.config.max_results) {
      const issue = new Issue(
        { kind: "result_overflow" },
        "warning",
        `Query returned ${result.values.length} results (max: ${this.config.max_results})`,
      ).withFix({ kind: "narrow_scope", filter: "Add type or name filter" });
      report.issues.push(issue);
      report.quality_score = 0.6;
    }

    for (const [entity_name] of query.entities) {
      if (graph.getNode(entity_name) === undefined) {
        const similar = this.findSimilarEntities(entity_name, graph);
        let issue = new Issue(
          { kind: "entity_not_found", name: entity_name },
          "warning",
          `Entity '${entity_name}' not found in graph`,
        );
        if (similar.length > 0) {
          issue = issue.withFix({
            kind: "resolve_entity",
            original: entity_name,
            suggested: similar[0],
          });
        }
        report.issues.push(issue);
        report.quality_score = Math.max(report.quality_score - 0.2, 0);
      }
    }

    this.validateRelationships(query, graph, report);

    for (const issue of report.issues) {
      const key = errorTypeKey(issue.error_type);
      this.error_patterns.set(key, (this.error_patterns.get(key) ?? 0) + 1);
    }

    return report;
  }

  /**
   * Explain an empty result: an `entity_not_found` error for the first entity
   * missing from the graph, else a `relation_mismatch` error when the joined
   * relation has no edges for the constant entity, else a plain
   * `empty_result` warning suggesting scope expansion.
   */
  private analyzeEmptyResult(
    query: QueryCore,
    graph: RelationshipGraphT,
  ): Issue {
    for (const [entity_name] of query.entities) {
      if (graph.getNode(entity_name) === undefined) {
        return new Issue(
          { kind: "entity_not_found", name: entity_name },
          "error",
          `Entity '${entity_name}' not found - query cannot return results`,
        );
      }
    }

    const relationMsg = this.checkRelationshipApplicability(query.root, graph);
    if (relationMsg !== undefined) {
      return new Issue(
        { kind: "relation_mismatch", message: relationMsg },
        "error",
        relationMsg,
      ).withFix({ kind: "expand_scope", relation: "CoOccurs" });
    }

    return new Issue(
      { kind: "empty_result" },
      "warning",
      "Query returned no results",
    ).withFix({ kind: "expand_scope", relation: "All" });
  }

  /**
   * For a root `join` with a constant side, return a message when the graph
   * has no edge of the relation's type on that entity; otherwise `undefined`.
   */
  private checkRelationshipApplicability(
    expr: QueryExpr,
    graph: RelationshipGraphT,
  ): string | undefined {
    if (expr.kind !== "op" || expr.op.kind !== "join") return undefined;
    const { relation, subject, object } = expr.op;
    const subjectName = subject.kind === "constant" ? subject.value : undefined;
    const objectName = object.kind === "constant" ? object.value : undefined;
    const name = subjectName ?? objectName;
    if (name === undefined) return undefined;

    const edges = graph.getEdges(name);
    const edgeType = relationToEdgeType(relation);
    if (
      edgeType !== undefined &&
      !edges.some((e) => e.edge_type === edgeType)
    ) {
      return `No ${relationName(relation)} relationships found for '${name}'`;
    }
    return undefined;
  }

  /** Names of up to 5 graph nodes returned by `graph.search(name, 5)`. */
  private findSimilarEntities(
    name: string,
    graph: RelationshipGraphT,
  ): string[] {
    return graph.search(name, 5).map((n) => n.entity_name);
  }

  /** Walk the query's root expression with {@link validateExpr}. */
  private validateRelationships(
    query: QueryCore,
    graph: RelationshipGraphT,
    report: ReflectionReport,
  ): void {
    this.validateExpr(query.root, graph, report);
  }

  /**
   * Recursively add a `relation_mismatch` warning for every `join` whose
   * relation is `custom` (no graph edge type), descending into nested ops.
   */
  private validateExpr(
    expr: QueryExpr,
    graph: RelationshipGraphT,
    report: ReflectionReport,
  ): void {
    if (expr.kind !== "op") return;
    const op = expr.op;
    switch (op.kind) {
      case "join": {
        if (
          relationToEdgeType(op.relation) === undefined &&
          !isSpecialRelation(op.relation) &&
          op.relation.kind === "custom"
        ) {
          const name = op.relation.name;
          const issue = new Issue(
            { kind: "relation_mismatch", message: name },
            "warning",
            `Custom relationship '${name}' may not exist`,
          ).withSource(relationName(op.relation));
          report.issues.push(issue);
        }
        this.validateExpr(op.subject, graph, report);
        this.validateExpr(op.object, graph, report);
        return;
      }
      case "and":
      case "or":
        for (const e of op.exprs) this.validateExpr(e, graph, report);
        return;
      case "filter":
        this.validateExpr(op.source, graph, report);
        return;
      case "count":
        this.validateExpr(op.inner, graph, report);
        return;
      case "superlative":
        this.validateExpr(op.source, graph, report);
        return;
    }
  }

  /** Validate query core structure (before execution). */
  validateQueryCore(query: QueryCore): Issue[] {
    const issues: Issue[] = [];
    if (query.entities.length === 0) {
      issues.push(
        new Issue(
          { kind: "schema_alignment", message: "No entities in query" },
          "warning",
          "Query does not reference any entities",
        ),
      );
    }
    if (query.question_type === "unknown") {
      issues.push(
        new Issue(
          { kind: "schema_alignment", message: "Unknown question type" },
          "info",
          "Could not determine question type",
        ),
      );
    }
    return issues;
  }

  /** Attempt to correct issues in a report. */
  attemptCorrection(
    report: ReflectionReport,
    graph: RelationshipGraphT,
    _executor: QueryExecutor,
  ): boolean {
    if (!this.config.auto_correct) return false;
    if (report.issues.length === 0) return true;

    report.correction_attempted = true;
    for (const issue of report.issues) {
      if (severityCompare(issue.severity, "warning") < 0) continue;
      for (const fix of issue.suggested_fixes) {
        if (fix.kind === "resolve_entity") {
          if (graph.getNode(fix.suggested) !== undefined) {
            const corrected = this.substituteEntity(
              report.query,
              fix.original,
              fix.suggested,
            );
            if (corrected !== undefined) {
              report.corrected_query = corrected;
              this.recordCorrection(issue, fix, true);
              return true;
            }
          }
        }
        // expand_scope and others: parity with Rust — skip (would require rewrite)
      }
    }
    return false;
  }

  /**
   * Deep-copy `query` with every occurrence of `original` (in `entities` and
   * in constant expressions) replaced by `replacement`. Never returns
   * `undefined` in practice; the type mirrors the Rust `Option`.
   */
  substituteEntity(
    query: QueryCore,
    original: string,
    replacement: string,
  ): QueryCore | undefined {
    const corrected: QueryCore = structuredClone(query);
    corrected.entities = corrected.entities.map(([name, t]) =>
      name === original ? [replacement, t] as [string, typeof t] : [name, t]
    );
    corrected.root = substituteInExpr(query.root, original, replacement);
    return corrected;
  }

  /** Append a {@link CorrectionRecord} stamped with the current Unix time. */
  private recordCorrection(
    issue: Issue,
    fix: SuggestedFix,
    success: boolean,
  ): void {
    this.correction_history.push({
      issue,
      fix_applied: fix,
      success,
      timestamp: Math.floor(Date.now() / 1000),
    });
  }

  /**
   * Feed the report into learning: records a pattern-less outcome whose
   * success is `report.isAcceptable()` and whose result count is the number
   * of values in the report.
   */
  provideFeedback(
    report: ReflectionReport,
    coordinator: LearningCoordinator,
  ): void {
    const success = report.isAcceptable();
    const result_count = report.result.values.length;
    coordinator.recordOutcome(
      undefined,
      success,
      result_count,
      report.query,
      0,
    );
  }

  /** Copy of the per-error-type counters (keys from `errorTypeKey`) accumulated across `analyze` calls. */
  getErrorStats(): Map<string, number> {
    return new Map(this.error_patterns);
  }

  /** Fraction of recorded corrections that succeeded; 0 when none were recorded. */
  correctionSuccessRate(): number {
    if (this.correction_history.length === 0) return 0;
    const successes = this.correction_history.filter((r) => r.success).length;
    return successes / this.correction_history.length;
  }

  /** Test hook that exposes {@link recordCorrection} (matches Rust helper usage). */
  recordCorrectionForTest(
    issue: Issue,
    fix: SuggestedFix,
    success: boolean,
  ): void {
    this.recordCorrection(issue, fix, success);
  }
}

function isSpecialRelation(r: RelationType): boolean {
  return r.kind === "has_type" ||
    r.kind === "has_error" ||
    r.kind === "created_at" ||
    r.kind === "modified_at";
}

function substituteInExpr(
  expr: QueryExpr,
  original: string,
  replacement: string,
): QueryExpr {
  switch (expr.kind) {
    case "constant":
      if (expr.value === original) {
        return {
          kind: "constant",
          value: replacement,
          entity_type: expr.entity_type,
        };
      }
      return expr;
    case "variable":
      return expr;
    case "op":
      return { kind: "op", op: substituteInOp(expr.op, original, replacement) };
  }
}

function substituteInOp(
  op: QueryOp,
  original: string,
  replacement: string,
): QueryOp {
  switch (op.kind) {
    case "join":
      return {
        kind: "join",
        relation: op.relation,
        subject: substituteInExpr(op.subject, original, replacement),
        object: substituteInExpr(op.object, original, replacement),
      };
    case "and":
      return {
        kind: "and",
        exprs: op.exprs.map((e) => substituteInExpr(e, original, replacement)),
      };
    case "or":
      return {
        kind: "or",
        exprs: op.exprs.map((e) => substituteInExpr(e, original, replacement)),
      };
    case "filter":
      return {
        kind: "filter",
        source: substituteInExpr(op.source, original, replacement),
        predicate: op.predicate,
      };
    case "count":
      return {
        kind: "count",
        inner: substituteInExpr(op.inner, original, replacement),
      };
    case "superlative":
      return {
        kind: "superlative",
        source: substituteInExpr(op.source, original, replacement),
        property: op.property,
        direction: op.direction,
      };
    case "values":
      return op;
  }
}
