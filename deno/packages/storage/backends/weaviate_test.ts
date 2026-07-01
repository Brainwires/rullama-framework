/**
 * Tests for Weaviate backend: GraphQL query building, filter construction, response parsing.
 *
 * These tests exercise the pure helper functions (no live Weaviate server required).
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import {
  buildAggregateQuery,
  buildBatchObject,
  buildSearchQuery,
  buildWhereFilter,
  deterministicUuid,
  parseWeaviateResult,
} from "./weaviate.ts";
import type { ChunkMetadata } from "@rullama/core";

// ---------------------------------------------------------------------------
// buildWhereFilter
// ---------------------------------------------------------------------------

Deno.test("buildWhereFilter - no params returns undefined", () => {
  assertEquals(buildWhereFilter(), undefined);
  assertEquals(buildWhereFilter(undefined, undefined, [], []), undefined);
});

Deno.test("buildWhereFilter - project only", () => {
  const filter = buildWhereFilter("my-project");
  assertEquals(filter, {
    path: ["project"],
    operator: "Equal",
    valueText: "my-project",
  });
});

Deno.test("buildWhereFilter - root_path only", () => {
  const filter = buildWhereFilter(undefined, "/root");
  assertEquals(filter, {
    path: ["root_path"],
    operator: "Equal",
    valueText: "/root",
  });
});

Deno.test("buildWhereFilter - multiple conditions uses And operator", () => {
  const filter = buildWhereFilter("proj", "/root", ["ts"], ["TypeScript"]);
  assertEquals(filter?.operator, "And");
  assertEquals((filter?.operands as unknown[])?.length, 4);
});

Deno.test("buildWhereFilter - extensions uses ContainsAny", () => {
  const filter = buildWhereFilter(undefined, undefined, ["rs", "ts"]);
  assertEquals(filter, {
    path: ["extension"],
    operator: "ContainsAny",
    valueTextArray: ["rs", "ts"],
  });
});

// ---------------------------------------------------------------------------
// buildSearchQuery
// ---------------------------------------------------------------------------

Deno.test("buildSearchQuery - nearVector without filter", () => {
  const gql = buildSearchQuery("CodeEmbedding", [1, 2, 3], 10, false, "");
  assertEquals(gql.includes("nearVector:"), true);
  assertEquals(gql.includes("limit: 10"), true);
  assertEquals(gql.includes("CodeEmbedding"), true);
  assertEquals(gql.includes("file_path"), true);
  assertEquals(gql.includes("_additional { score }"), true);
});

Deno.test("buildSearchQuery - hybrid mode", () => {
  const gql = buildSearchQuery("CodeEmbedding", [1, 0], 5, true, "test query");
  assertEquals(gql.includes("hybrid:"), true);
  assertEquals(gql.includes("test query"), true);
  assertEquals(gql.includes("alpha: 0.7"), true);
});

Deno.test("buildSearchQuery - with where filter", () => {
  const filter = {
    path: ["project"],
    operator: "Equal",
    valueText: "proj",
  };
  const gql = buildSearchQuery("CodeEmbedding", [1, 0], 5, false, "", filter);
  assertEquals(gql.includes("where:"), true);
});

Deno.test("buildSearchQuery - escapes special characters in hybrid query", () => {
  const gql = buildSearchQuery("CodeEmbedding", [1], 5, true, 'say "hello"');
  assertEquals(gql.includes('say \\"hello\\"'), true);
});

// ---------------------------------------------------------------------------
// buildAggregateQuery
// ---------------------------------------------------------------------------

Deno.test("buildAggregateQuery - without filter", () => {
  const gql = buildAggregateQuery("CodeEmbedding");
  assertEquals(gql.includes("Aggregate"), true);
  assertEquals(gql.includes("meta { count }"), true);
});

Deno.test("buildAggregateQuery - with filter", () => {
  const filter = { path: ["root_path"], operator: "Equal", valueText: "/root" };
  const gql = buildAggregateQuery("CodeEmbedding", filter);
  assertEquals(gql.includes("where:"), true);
});

// ---------------------------------------------------------------------------
// parseWeaviateResult
// ---------------------------------------------------------------------------

Deno.test("parseWeaviateResult - valid result", () => {
  const obj = {
    file_path: "/src/lib.rs",
    root_path: "/project",
    content: "use std::io;",
    start_line: 1,
    end_line: 5,
    language: "Rust",
    project: "my-proj",
    indexed_at: 999,
    _additional: { score: "0.95" },
  };
  const result = parseWeaviateResult(obj);
  assertEquals(result?.file_path, "/src/lib.rs");
  assertEquals(result?.content, "use std::io;");
  assertEquals(result?.score, 0.95);
  assertEquals(result?.start_line, 1);
  assertEquals(result?.end_line, 5);
  assertEquals(result?.language, "Rust");
  assertEquals(result?.project, "my-proj");
  assertEquals(result?.root_path, "/project");
  assertEquals(result?.indexed_at, 999);
});

Deno.test("parseWeaviateResult - missing content returns null", () => {
  assertEquals(
    parseWeaviateResult({ file_path: "/a.rs", start_line: 0, end_line: 1 }),
    null,
  );
});

Deno.test("parseWeaviateResult - missing file_path returns null", () => {
  assertEquals(
    parseWeaviateResult({ content: "hello", start_line: 0, end_line: 1 }),
    null,
  );
});

Deno.test("parseWeaviateResult - defaults for optional fields", () => {
  const obj = {
    file_path: "/x.ts",
    content: "const x = 1;",
    start_line: 0,
    end_line: 1,
  };
  const result = parseWeaviateResult(obj);
  assertEquals(result?.language, "Unknown");
  assertEquals(result?.project, undefined);
  assertEquals(result?.root_path, undefined);
  assertEquals(result?.indexed_at, 0);
  assertEquals(result?.score, 0);
});

// ---------------------------------------------------------------------------
// buildBatchObject
// ---------------------------------------------------------------------------

Deno.test("buildBatchObject - builds correct structure", () => {
  const meta: ChunkMetadata = {
    file_path: "/src/main.rs",
    root_path: "/project",
    project: "my-proj",
    start_line: 0,
    end_line: 10,
    language: "Rust",
    extension: "rs",
    file_hash: "abc123",
    indexed_at: 1000,
  };
  const obj = buildBatchObject(
    "CodeEmbedding",
    [1, 2, 3],
    meta,
    "fn main() {}",
    "/project",
  );
  assertEquals(obj.class, "CodeEmbedding");
  assertEquals(obj.vector, [1, 2, 3]);
  const props = obj.properties as Record<string, unknown>;
  assertEquals(props.file_path, "/src/main.rs");
  assertEquals(props.content, "fn main() {}");
  assertEquals(typeof obj.id, "string");
});

// ---------------------------------------------------------------------------
// deterministicUuid
// ---------------------------------------------------------------------------

Deno.test("deterministicUuid - same inputs produce same UUID", () => {
  const uuid1 = deterministicUuid("file.rs", 1, 10);
  const uuid2 = deterministicUuid("file.rs", 1, 10);
  assertEquals(uuid1, uuid2);
});

Deno.test("deterministicUuid - different inputs produce different UUID", () => {
  const uuid1 = deterministicUuid("file.rs", 1, 10);
  const uuid2 = deterministicUuid("other.rs", 1, 10);
  assertNotEquals(uuid1, uuid2);
});

Deno.test("deterministicUuid - correct format (8-4-4-4-12)", () => {
  const uuid = deterministicUuid("file.rs", 1, 10);
  assertEquals(uuid.length, 36);
  assertEquals(uuid.split("-").map((s) => s.length), [8, 4, 4, 4, 12]);
});
