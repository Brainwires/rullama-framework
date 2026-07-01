/**
 * Tests for Milvus backend: REST request building, filter construction, response parsing.
 *
 * These tests exercise the pure helper functions (no live Milvus server required).
 */

import { assertEquals } from "@std/assert";
import {
  buildFilterExpr,
  buildInsertBody,
  buildSearchBody,
  escapeFilterValue,
  parseMilvusResult,
} from "./milvus.ts";
import type { ChunkMetadata } from "@rullama/core";

// ---------------------------------------------------------------------------
// escapeFilterValue
// ---------------------------------------------------------------------------

Deno.test("escapeFilterValue - plain string unchanged", () => {
  assertEquals(escapeFilterValue("hello"), "hello");
});

Deno.test("escapeFilterValue - escapes double quotes", () => {
  assertEquals(escapeFilterValue('say "hi"'), 'say \\"hi\\"');
});

Deno.test("escapeFilterValue - escapes backslashes", () => {
  assertEquals(escapeFilterValue("back\\slash"), "back\\\\slash");
});

// ---------------------------------------------------------------------------
// buildFilterExpr
// ---------------------------------------------------------------------------

Deno.test("buildFilterExpr - no params returns empty string", () => {
  assertEquals(buildFilterExpr(), "");
  assertEquals(buildFilterExpr(undefined, undefined, [], []), "");
});

Deno.test("buildFilterExpr - project only", () => {
  assertEquals(
    buildFilterExpr("my-proj"),
    'project == "my-proj"',
  );
});

Deno.test("buildFilterExpr - root_path only", () => {
  assertEquals(
    buildFilterExpr(undefined, "/root"),
    'root_path == "/root"',
  );
});

Deno.test("buildFilterExpr - multiple clauses joined with and", () => {
  const expr = buildFilterExpr("proj", "/root", ["ts"], ["TypeScript"]);
  assertEquals(
    expr,
    'project == "proj" and root_path == "/root" and extension in ["ts"] and language in ["TypeScript"]',
  );
});

Deno.test("buildFilterExpr - multiple extensions", () => {
  const expr = buildFilterExpr(undefined, undefined, ["rs", "ts"]);
  assertEquals(expr, 'extension in ["rs", "ts"]');
});

Deno.test("buildFilterExpr - escapes special characters", () => {
  const expr = buildFilterExpr('proj "A"');
  assertEquals(expr, 'project == "proj \\"A\\""');
});

// ---------------------------------------------------------------------------
// buildSearchBody
// ---------------------------------------------------------------------------

Deno.test("buildSearchBody - without filter", () => {
  const body = buildSearchBody("embeddings", [1, 2, 3], 10, "");
  assertEquals(body.collectionName, "embeddings");
  assertEquals(body.data, [[1, 2, 3]]);
  assertEquals(body.annsField, "embedding");
  assertEquals(body.limit, 10);
  assertEquals(body.filter, undefined);
});

Deno.test("buildSearchBody - with filter", () => {
  const body = buildSearchBody("embeddings", [1, 0], 5, 'project == "p"');
  assertEquals(body.filter, 'project == "p"');
});

Deno.test("buildSearchBody - includes output fields", () => {
  const body = buildSearchBody("col", [1], 1, "");
  const fields = body.outputFields as string[];
  assertEquals(fields.includes("file_path"), true);
  assertEquals(fields.includes("content"), true);
  assertEquals(fields.includes("language"), true);
});

// ---------------------------------------------------------------------------
// buildInsertBody
// ---------------------------------------------------------------------------

Deno.test("buildInsertBody - single entity", () => {
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
  const body = buildInsertBody("code_embeddings", [[1, 2, 3]], [meta], [
    "fn main() {}",
  ], "/project");
  assertEquals(body.collectionName, "code_embeddings");
  const data = body.data as Record<string, unknown>[];
  assertEquals(data.length, 1);
  assertEquals(data[0].file_path, "/src/main.rs");
  assertEquals(data[0].embedding, [1, 2, 3]);
  assertEquals(data[0].content, "fn main() {}");
});

Deno.test("buildInsertBody - uses rootPath fallback", () => {
  const meta: ChunkMetadata = {
    file_path: "/a.ts",
    start_line: 0,
    end_line: 5,
    file_hash: "h1",
    indexed_at: 100,
  };
  const body = buildInsertBody("col", [[1]], [meta], ["x"], "/fallback");
  const data = body.data as Record<string, unknown>[];
  assertEquals(data[0].root_path, "/fallback");
  assertEquals(data[0].language, "Unknown");
  assertEquals(data[0].project, "");
});

// ---------------------------------------------------------------------------
// parseMilvusResult
// ---------------------------------------------------------------------------

Deno.test("parseMilvusResult - valid result with cosine distance", () => {
  const item = {
    distance: 0.1,
    file_path: "/src/lib.rs",
    root_path: "/project",
    content: "use std::io;",
    start_line: 1,
    end_line: 5,
    language: "Rust",
    project: "my-proj",
    indexed_at: 999,
  };
  const result = parseMilvusResult(item, 0.0);
  assertEquals(result?.file_path, "/src/lib.rs");
  assertEquals(result?.content, "use std::io;");
  // distance 0.1 -> score 0.9
  assertEquals(result?.score, 0.9);
  assertEquals(result?.vector_score, 0.9);
  assertEquals(result?.start_line, 1);
  assertEquals(result?.end_line, 5);
  assertEquals(result?.language, "Rust");
  assertEquals(result?.project, "my-proj");
  assertEquals(result?.root_path, "/project");
  assertEquals(result?.indexed_at, 999);
});

Deno.test("parseMilvusResult - score below minScore returns null", () => {
  const item = {
    distance: 0.8,
    file_path: "/a.rs",
    content: "hello",
    start_line: 0,
    end_line: 1,
  };
  // distance 0.8 -> score 0.2, below minScore 0.5
  assertEquals(parseMilvusResult(item, 0.5), null);
});

Deno.test("parseMilvusResult - missing content returns null", () => {
  assertEquals(
    parseMilvusResult({
      distance: 0,
      file_path: "/a.rs",
      start_line: 0,
      end_line: 1,
    }, 0),
    null,
  );
});

Deno.test("parseMilvusResult - missing file_path returns null", () => {
  assertEquals(
    parseMilvusResult({
      distance: 0,
      content: "hello",
      start_line: 0,
      end_line: 1,
    }, 0),
    null,
  );
});

Deno.test("parseMilvusResult - empty project treated as undefined", () => {
  const item = {
    distance: 0,
    file_path: "/x.ts",
    content: "x",
    project: "",
    start_line: 0,
    end_line: 1,
  };
  const result = parseMilvusResult(item, 0);
  assertEquals(result?.project, undefined);
});

Deno.test("parseMilvusResult - defaults for optional fields", () => {
  const item = {
    distance: 0.05,
    file_path: "/x.ts",
    content: "const x = 1;",
    start_line: 0,
    end_line: 1,
  };
  const result = parseMilvusResult(item, 0);
  assertEquals(result?.language, "Unknown");
  assertEquals(result?.project, undefined);
  assertEquals(result?.root_path, undefined);
  assertEquals(result?.indexed_at, 0);
});
