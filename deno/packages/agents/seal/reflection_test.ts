import { assert, assertEquals } from "@std/assert";
import type { GraphEdge, GraphNode, RelationshipGraphT } from "@brainwires/core";
import {
  Issue,
  ReflectionModule,
  ReflectionReport,
  severityCompare,
  suggestedFixDescription,
  type SuggestedFix,
  errorTypeDescription,
} from "./reflection.ts";
import {
  newQueryCore,
  queryConstant,
  queryResultEmpty,
  queryResultWithValues,
  queryVar,
  type QueryResultValue,
} from "./query_core.ts";

/** Minimal empty in-memory graph for tests — mirrors `RelationshipGraph::new()`. */
class EmptyGraph implements RelationshipGraphT {
  getNode(_name: string): GraphNode | undefined {
    return undefined;
  }
  getNeighbors(_name: string): GraphNode[] {
    return [];
  }
  getEdges(_name: string): GraphEdge[] {
    return [];
  }
  search(_query: string, _limit: number): GraphNode[] {
    return [];
  }
  findPath(_from: string, _to: string): string[] | undefined {
    return undefined;
  }
}

function createTestQuery() {
  return newQueryCore(
    "definition",
    queryVar("x"),
    [["main.rs", "file"]],
    "What is main.rs?",
  );
}

Deno.test("analyze empty result", () => {
  const reflection = new ReflectionModule();
  const query = createTestQuery();
  const report = reflection.analyze(query, queryResultEmpty(), new EmptyGraph());

  assert(report.issues.length > 0);
  assert(report.issues.some((i) =>
    i.error_type.kind === "empty_result" ||
    i.error_type.kind === "entity_not_found"
  ));
});

Deno.test("analyze overflow result", () => {
  const reflection = new ReflectionModule({
    max_results: 10,
    min_results: 1,
    max_retries: 2,
    auto_correct: true,
  });
  const query = createTestQuery();
  const values: QueryResultValue[] = [];
  for (let i = 0; i < 20; i++) {
    values.push({
      value: `entity_${i}`,
      entity_type: "file",
      score: 0.8,
      metadata: new Map(),
    });
  }
  const result = queryResultWithValues(values);
  const report = reflection.analyze(query, result, new EmptyGraph());
  assert(report.issues.some((i) => i.error_type.kind === "result_overflow"));
});

Deno.test("validate query core", () => {
  const reflection = new ReflectionModule();
  const query = newQueryCore("unknown", queryVar("x"), [], "Test");
  const issues = reflection.validateQueryCore(query);
  assert(issues.length > 0);
});

Deno.test("Issue creation", () => {
  const issue = new Issue({ kind: "empty_result" }, "warning", "No results")
    .withFix({ kind: "expand_scope", relation: "All" })
    .withSource("query_root");
  assertEquals(issue.severity, "warning");
  assertEquals(issue.suggested_fixes.length, 1);
  assert(issue.source !== undefined);
});

Deno.test("Severity ordering", () => {
  assert(severityCompare("info", "warning") < 0);
  assert(severityCompare("warning", "error") < 0);
  assert(severityCompare("error", "critical") < 0);
});

Deno.test("ReflectionReport is acceptable", () => {
  const query = createTestQuery();
  const result = queryResultEmpty();
  const report = new ReflectionReport(query, result);

  report.quality_score = 0.7;
  assert(report.isAcceptable());

  report.quality_score = 0.3;
  assert(!report.isAcceptable());

  report.quality_score = 0.7;
  report.issues.push(new Issue({ kind: "empty_result" }, "error", "Error"));
  assert(!report.isAcceptable());
});

Deno.test("SuggestedFix description", () => {
  const fix: SuggestedFix = {
    kind: "resolve_entity",
    original: "main",
    suggested: "main.rs",
  };
  const desc = suggestedFixDescription(fix);
  assert(desc.includes("main"));
  assert(desc.includes("main.rs"));
});

Deno.test("ErrorType description", () => {
  const desc = errorTypeDescription({ kind: "entity_not_found", name: "test.rs" });
  assert(desc.includes("test.rs"));
});

Deno.test("substitute entity", () => {
  const reflection = new ReflectionModule();
  const query = newQueryCore(
    "definition",
    queryConstant("main", "file"),
    [["main", "file"]],
    "What is main?",
  );
  const corrected = reflection.substituteEntity(query, "main", "main.rs");
  assert(corrected !== undefined);
  assert(corrected!.entities.some(([n]) => n === "main.rs"));
});

Deno.test("correction success rate", () => {
  const reflection = new ReflectionModule();
  assertEquals(reflection.correctionSuccessRate(), 0);

  reflection.recordCorrectionForTest(
    new Issue({ kind: "empty_result" }, "warning", "test"),
    { kind: "expand_scope", relation: "All" },
    true,
  );
  reflection.recordCorrectionForTest(
    new Issue({ kind: "empty_result" }, "warning", "test"),
    { kind: "expand_scope", relation: "All" },
    false,
  );
  assertEquals(reflection.correctionSuccessRate(), 0.5);
});
