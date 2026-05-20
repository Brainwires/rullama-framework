/**
 * Reflection Module for Error Detection and Correction.
 *
 * Provides post-execution analysis to detect issues and suggest corrections.
 *
 * Equivalent to Rust's `brainwires_agents::seal::reflection` module.
 */

import type { RelationshipGraphT } from "@brainwires/core";
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

export type ErrorType =
  | { kind: "empty_result" }
  | { kind: "result_overflow" }
  | { kind: "entity_not_found"; name: string }
  | { kind: "relation_mismatch"; message: string }
  | { kind: "coreference_failure"; reference: string }
  | { kind: "schema_alignment"; message: string }
  | { kind: "timeout" }
  | { kind: "unknown"; message: string };

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

export type Severity = "info" | "warning" | "error" | "critical";

const SEVERITY_ORDER: Record<Severity, number> = {
  info: 0,
  warning: 1,
  error: 2,
  critical: 3,
};

export function severityAtLeast(a: Severity, b: Severity): boolean {
  return SEVERITY_ORDER[a] >= SEVERITY_ORDER[b];
}

export function severityCompare(a: Severity, b: Severity): number {
  return SEVERITY_ORDER[a] - SEVERITY_ORDER[b];
}

// ─── SuggestedFix ───────────────────────────────────────────────────────────

export type SuggestedFix =
  | { kind: "retry_with_query"; query: QueryCore }
  | { kind: "expand_scope"; relation: string }
  | { kind: "narrow_scope"; filter: string }
  | { kind: "resolve_entity"; original: string; suggested: string }
  | { kind: "add_relation"; from: string; to: string; relation: string }
  | { kind: "manual_intervention"; message: string };

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
  error_type: ErrorType;
  severity: Severity;
  message: string;
  suggested_fixes: SuggestedFix[] = [];
  source: string | undefined;

  constructor(error_type: ErrorType, severity: Severity, message: string) {
    this.error_type = error_type;
    this.severity = severity;
    this.message = message;
  }

  withFix(fix: SuggestedFix): Issue {
    this.suggested_fixes.push(fix);
    return this;
  }

  withSource(source: string): Issue {
    this.source = source;
    return this;
  }
}

/** Record of a correction attempt. */
export interface CorrectionRecord {
  issue: Issue;
  fix_applied: SuggestedFix;
  success: boolean;
  timestamp: number;
}

// ─── ReflectionReport ───────────────────────────────────────────────────────

/** Reflection report. */
export class ReflectionReport {
  query: QueryCore;
  result: QueryResult;
  issues: Issue[] = [];
  quality_score = 1.0;
  correction_attempted = false;
  corrected_query: QueryCore | undefined;
  corrected_result: QueryResult | undefined;

  constructor(query: QueryCore, result: QueryResult) {
    this.query = query;
    this.result = result;
  }

  isAcceptable(): boolean {
    return this.quality_score >= 0.5 &&
      !this.issues.some((i) => severityAtLeast(i.severity, "error"));
  }

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

export interface ReflectionConfig {
  max_results: number;
  min_results: number;
  max_retries: number;
  auto_correct: boolean;
}

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

  private findSimilarEntities(
    name: string,
    graph: RelationshipGraphT,
  ): string[] {
    return graph.search(name, 5).map((n) => n.entity_name);
  }

  private validateRelationships(
    query: QueryCore,
    graph: RelationshipGraphT,
    report: ReflectionReport,
  ): void {
    this.validateExpr(query.root, graph, report);
  }

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

  provideFeedback(
    report: ReflectionReport,
    coordinator: LearningCoordinator,
  ): void {
    const success = report.isAcceptable();
    const result_count = report.result.values.length;
    coordinator.recordOutcome(undefined, success, result_count, report.query, 0);
  }

  getErrorStats(): Map<string, number> {
    return new Map(this.error_patterns);
  }

  correctionSuccessRate(): number {
    if (this.correction_history.length === 0) return 0;
    const successes =
      this.correction_history.filter((r) => r.success).length;
    return successes / this.correction_history.length;
  }

  // Expose recordCorrection for tests (matches Rust helper usage).
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
        return { kind: "constant", value: replacement, entity_type: expr.entity_type };
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
