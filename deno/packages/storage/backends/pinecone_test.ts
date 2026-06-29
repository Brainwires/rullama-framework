/**
 * Tests for Pinecone backend: request building, response parsing, filter construction.
 *
 * These tests exercise the pure helper functions (no live Pinecone server required).
 */

import { assertEquals } from "@std/assert";
import {
  buildMetadataFilter,
  buildQueryBody,
  buildUpsertBody,
  extractFilePathsFromIds,
  parseMatch,
} from "./pinecone.ts";
import type { ChunkMetadata } from "@rullama/core";

// ---------------------------------------------------------------------------
// buildMetadataFilter
// ---------------------------------------------------------------------------

Deno.test("buildMetadataFilter - no params returns undefined", () => {
  assertEquals(buildMetadataFilter(), undefined);
  assertEquals(buildMetadataFilter(undefined, undefined, [], []), undefined);
});

Deno.test("buildMetadataFilter - project only", () => {
  const filter = buildMetadataFilter("my-project");
  assertEquals(filter, { project: { $eq: "my-project" } });
});

Deno.test("buildMetadataFilter - root_path only", () => {
  const filter = buildMetadataFilter(undefined, "/root");
  assertEquals(filter, { root_path: { $eq: "/root" } });
});

Deno.test("buildMetadataFilter - multiple conditions uses $and", () => {
  const filter = buildMetadataFilter("proj", "/root", ["ts", "rs"], [
    "TypeScript",
  ]);
  assertEquals(filter, {
    $and: [
      { project: { $eq: "proj" } },
      { root_path: { $eq: "/root" } },
      { extension: { $in: ["ts", "rs"] } },
      { language: { $in: ["TypeScript"] } },
    ],
  });
});

Deno.test("buildMetadataFilter - extensions only", () => {
  const filter = buildMetadataFilter(undefined, undefined, ["rs"]);
  assertEquals(filter, { extension: { $in: ["rs"] } });
});

// ---------------------------------------------------------------------------
// buildUpsertBody
// ---------------------------------------------------------------------------

Deno.test("buildUpsertBody - single vector", () => {
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
  const body = buildUpsertBody(
    [[1, 2, 3]],
    [meta],
    ["fn main() {}"],
    "/project",
    "ns1",
  );
  assertEquals(body.namespace, "ns1");
  assertEquals(body.vectors.length, 1);
  const vec = body.vectors[0] as Record<string, unknown>;
  assertEquals(vec.id, "/project:/src/main.rs:0");
  assertEquals(vec.values, [1, 2, 3]);
  const payload = vec.metadata as Record<string, unknown>;
  assertEquals(payload.file_path, "/src/main.rs");
  assertEquals(payload.content, "fn main() {}");
});

Deno.test("buildUpsertBody - multiple vectors", () => {
  const meta: ChunkMetadata = {
    file_path: "/a.ts",
    start_line: 0,
    end_line: 5,
    file_hash: "h1",
    indexed_at: 100,
  };
  const body = buildUpsertBody(
    [[1, 0], [0, 1]],
    [meta, { ...meta, file_path: "/b.ts", start_line: 10 }],
    ["a", "b"],
    "/root",
    "",
  );
  assertEquals(body.vectors.length, 2);
});

// ---------------------------------------------------------------------------
// buildQueryBody
// ---------------------------------------------------------------------------

Deno.test("buildQueryBody - no filter", () => {
  const body = buildQueryBody([1, 2, 3], 10, "ns");
  assertEquals(body.vector, [1, 2, 3]);
  assertEquals(body.topK, 10);
  assertEquals(body.namespace, "ns");
  assertEquals(body.includeMetadata, true);
  assertEquals(body.filter, undefined);
});

Deno.test("buildQueryBody - with filter", () => {
  const filter = { project: { $eq: "p" } };
  const body = buildQueryBody([1, 0], 5, "ns", filter);
  assertEquals(body.filter, filter);
});

// ---------------------------------------------------------------------------
// parseMatch
// ---------------------------------------------------------------------------

Deno.test("parseMatch - valid match", () => {
  const match = {
    id: "root:/src/lib.rs:1",
    score: 0.95,
    metadata: {
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
  const result = parseMatch(match, 0.5);
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

Deno.test("parseMatch - score below minScore returns null", () => {
  const match = {
    id: "x",
    score: 0.3,
    metadata: {
      file_path: "/a.rs",
      content: "hello",
      start_line: 0,
      end_line: 1,
    },
  };
  assertEquals(parseMatch(match, 0.5), null);
});

Deno.test("parseMatch - missing metadata returns null", () => {
  assertEquals(parseMatch({ id: "x", score: 0.9 }, 0.0), null);
});

Deno.test("parseMatch - missing file_path returns null", () => {
  const match = {
    id: "x",
    score: 0.9,
    metadata: { content: "hello", start_line: 0, end_line: 1 },
  };
  assertEquals(parseMatch(match, 0.0), null);
});

Deno.test("parseMatch - defaults for optional fields", () => {
  const match = {
    id: "x",
    score: 0.8,
    metadata: {
      file_path: "/x.ts",
      content: "const x = 1;",
      start_line: 0,
      end_line: 1,
    },
  };
  const result = parseMatch(match, 0.0);
  assertEquals(result?.language, "Unknown");
  assertEquals(result?.project, undefined);
  assertEquals(result?.root_path, undefined);
  assertEquals(result?.indexed_at, 0);
});

// ---------------------------------------------------------------------------
// extractFilePathsFromIds
// ---------------------------------------------------------------------------

Deno.test("extractFilePathsFromIds - extracts and deduplicates", () => {
  const ids = [
    "/root:/src/main.rs:0",
    "/root:/src/main.rs:10",
    "/root:/src/lib.rs:0",
    "/other:/src/other.rs:0",
  ];
  const result = extractFilePathsFromIds(ids, "/root:");
  assertEquals(result, ["/src/lib.rs", "/src/main.rs"]);
});

Deno.test("extractFilePathsFromIds - empty list", () => {
  assertEquals(extractFilePathsFromIds([], "/root:"), []);
});

Deno.test("extractFilePathsFromIds - no matching prefix", () => {
  assertEquals(
    extractFilePathsFromIds(["/other:/a.rs:0"], "/root:"),
    [],
  );
});
