/**
 * Tests for Qdrant backend: REST request building, filter conversion, response parsing.
 *
 * These tests exercise the pure helper functions (no live Qdrant server required).
 */

import { assertEquals } from "@std/assert";
import {
  buildQdrantFilter,
  buildSearchBody,
  buildUpsertBody,
  parseSearchPoint,
} from "./qdrant.ts";
import type { ChunkMetadata } from "@rullama/core";

// ---------------------------------------------------------------------------
// buildQdrantFilter
// ---------------------------------------------------------------------------

Deno.test("buildQdrantFilter - no params returns undefined", () => {
  assertEquals(buildQdrantFilter(), undefined);
  assertEquals(buildQdrantFilter(undefined, undefined, [], []), undefined);
});

Deno.test("buildQdrantFilter - project only", () => {
  const filter = buildQdrantFilter("my-project");
  assertEquals(filter, {
    must: [{ key: "project", match: { value: "my-project" } }],
  });
});

Deno.test("buildQdrantFilter - multiple conditions", () => {
  const filter = buildQdrantFilter("proj", "/root", ["ts", "rs"], [
    "TypeScript",
  ]);
  assertEquals(filter?.must?.length, 4);
  assertEquals(filter!.must![0].key, "project");
  assertEquals(filter!.must![1].key, "root_path");
  assertEquals(filter!.must![2].key, "extension");
  assertEquals(filter!.must![3].key, "language");
});

Deno.test("buildQdrantFilter - extensions as array value", () => {
  const filter = buildQdrantFilter(undefined, undefined, ["rs", "ts"]);
  assertEquals(filter, {
    must: [{ key: "extension", match: { value: ["rs", "ts"] } }],
  });
});

// ---------------------------------------------------------------------------
// buildUpsertBody
// ---------------------------------------------------------------------------

Deno.test("buildUpsertBody - single point", () => {
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
  const body = buildUpsertBody([[1, 2, 3]], [meta], ["fn main() {}"]);
  assertEquals(body.points.length, 1);
  const point = body.points[0] as Record<string, unknown>;
  assertEquals(point.id, 0);
  assertEquals(point.vector, [1, 2, 3]);
  const payload = point.payload as Record<string, unknown>;
  assertEquals(payload.file_path, "/src/main.rs");
  assertEquals(payload.content, "fn main() {}");
});

Deno.test("buildUpsertBody - multiple points", () => {
  const meta: ChunkMetadata = {
    file_path: "/a.ts",
    start_line: 0,
    end_line: 5,
    file_hash: "h1",
    indexed_at: 100,
  };
  const body = buildUpsertBody(
    [[1, 0], [0, 1]],
    [meta, { ...meta, file_path: "/b.ts" }],
    ["content-a", "content-b"],
  );
  assertEquals(body.points.length, 2);
  assertEquals((body.points[0] as Record<string, unknown>).id, 0);
  assertEquals((body.points[1] as Record<string, unknown>).id, 1);
});

// ---------------------------------------------------------------------------
// buildSearchBody
// ---------------------------------------------------------------------------

Deno.test("buildSearchBody - no filter", () => {
  const body = buildSearchBody([1, 2, 3], 10, 0.7);
  assertEquals(body.vector, [1, 2, 3]);
  assertEquals(body.limit, 10);
  assertEquals(body.score_threshold, 0.7);
  assertEquals(body.with_payload, true);
  assertEquals(body.filter, undefined);
});

Deno.test("buildSearchBody - with filter", () => {
  const filter = { must: [{ key: "project", match: { value: "p" } }] };
  const body = buildSearchBody([1, 0], 5, 0.5, filter);
  assertEquals(body.filter, filter);
});

// ---------------------------------------------------------------------------
// parseSearchPoint
// ---------------------------------------------------------------------------

Deno.test("parseSearchPoint - valid point", () => {
  const point = {
    id: 42,
    score: 0.95,
    payload: {
      file_path: "/src/lib.rs",
      root_path: "/project",
      content: "use std::io;",
      start_line: 1,
      end_line: 5,
      language: "Rust",
      project: "my-proj",
      indexed_at: 999,
    },
  };
  const result = parseSearchPoint(point);
  assertEquals(result?.file_path, "/src/lib.rs");
  assertEquals(result?.content, "use std::io;");
  assertEquals(result?.score, 0.95);
  assertEquals(result?.vector_score, 0.95);
  assertEquals(result?.start_line, 1);
  assertEquals(result?.end_line, 5);
  assertEquals(result?.language, "Rust");
  assertEquals(result?.project, "my-proj");
  assertEquals(result?.root_path, "/project");
  assertEquals(result?.indexed_at, 999);
});

Deno.test("parseSearchPoint - missing content returns null", () => {
  const result = parseSearchPoint({
    id: 1,
    score: 0.5,
    payload: { file_path: "/a.rs", start_line: 0, end_line: 1 },
  });
  assertEquals(result, null);
});

Deno.test("parseSearchPoint - missing file_path returns null", () => {
  const result = parseSearchPoint({
    id: 1,
    score: 0.5,
    payload: { content: "hello", start_line: 0, end_line: 1 },
  });
  assertEquals(result, null);
});

Deno.test("parseSearchPoint - missing payload returns null", () => {
  assertEquals(parseSearchPoint({ id: 1, score: 0.5 }), null);
});

Deno.test("parseSearchPoint - defaults for optional fields", () => {
  const point = {
    id: 1,
    score: 0.8,
    payload: {
      file_path: "/x.ts",
      content: "const x = 1;",
      start_line: 0,
      end_line: 1,
    },
  };
  const result = parseSearchPoint(point);
  assertEquals(result?.language, "Unknown");
  assertEquals(result?.project, undefined);
  assertEquals(result?.root_path, undefined);
  assertEquals(result?.indexed_at, 0);
});
