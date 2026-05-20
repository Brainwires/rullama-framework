import { assert, assertEquals } from "@std/assert";
import {
  asVariable,
  isVariable,
  newQueryCore,
  queryConstant,
  QueryCoreExtractor,
  queryCoreToSexp,
  queryResultEmpty,
  queryResultError,
  queryResultWithValues,
  queryVar,
  relationInverse,
  type QueryResultValue,
} from "./query_core.ts";
import type { EntityType } from "@brainwires/core";

Deno.test("classify definition question", () => {
  const x = new QueryCoreExtractor();
  const [qtype] = x.classifyQuestion("What is EntityStore?");
  assertEquals(qtype, "definition");
});

Deno.test("classify location question", () => {
  const x = new QueryCoreExtractor();
  const [qtype] = x.classifyQuestion("Where is main defined?");
  assertEquals(qtype, "location");
});

Deno.test("classify dependency question", () => {
  const x = new QueryCoreExtractor();
  const [qtype, rel] = x.classifyQuestion("What uses EntityStore?");
  assertEquals(qtype, "dependency");
  assertEquals(rel?.kind, "depends_on");
});

Deno.test("classify count question", () => {
  const x = new QueryCoreExtractor();
  const [qtype] = x.classifyQuestion("How many functions are there?");
  assertEquals(qtype, "count");
});

Deno.test("extract dependency query", () => {
  const x = new QueryCoreExtractor();
  const entities: [string, EntityType][] = [["main.rs", "file"]];
  const core = x.extract("What uses main.rs?", entities);
  assert(core !== undefined);
  assertEquals(core!.question_type, "dependency");

  const sexp = queryCoreToSexp(core!);
  assert(sexp.includes("JOIN"));
  assert(sexp.includes("DependsOn"));
});

Deno.test("extract location query", () => {
  const x = new QueryCoreExtractor();
  const entities: [string, EntityType][] = [["process_data", "function"]];
  const core = x.extract("Where is process_data defined?", entities);
  assert(core !== undefined);
  assertEquals(core!.question_type, "location");
});

Deno.test("query expression helpers", () => {
  const v = queryVar("file");
  assert(isVariable(v));
  assertEquals(asVariable(v), "?file");

  const c = queryConstant("main.rs", "file");
  assert(!isVariable(c));
  assertEquals(asVariable(c), undefined);
});

Deno.test("query result", () => {
  const values: QueryResultValue[] = [
    { value: "test1", entity_type: "file", score: 0.9, metadata: new Map() },
    { value: "test2", entity_type: "function", score: 0.8, metadata: new Map() },
  ];
  const r = queryResultWithValues(values);
  assert(r.success);
  assertEquals(r.count, 2);
  assertEquals(r.values.length, 2);
});

Deno.test("query result error", () => {
  const r = queryResultError("Entity not found");
  assert(!r.success);
  assert(r.error !== undefined);
});

Deno.test("query result empty", () => {
  const r = queryResultEmpty();
  assert(r.success);
  assertEquals(r.values.length, 0);
});

Deno.test("relation type inverse", () => {
  assert(relationInverse({ kind: "contains" }) !== undefined);
  assert(relationInverse({ kind: "depends_on" }) !== undefined);
  assert(relationInverse({ kind: "co_occurs" }) === undefined);
});

Deno.test("query core round trip", () => {
  const core = newQueryCore(
    "definition",
    queryVar("x"),
    [["main.rs", "file"]],
    "What is main.rs?",
  );
  assertEquals(core.question_type, "definition");
  assertEquals(core.confidence, 1.0);
});
