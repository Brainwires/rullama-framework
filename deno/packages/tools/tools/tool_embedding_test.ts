import { assert, assertEquals } from "@std/assert";
import type { EmbeddingProvider } from "@brainwires/core";
import { cosineSimilarity, ToolEmbeddingIndex } from "./tool_embedding.ts";

function sampleTools(): Array<[string, string]> {
  return [
    ["read_file", "Read the contents of a file from disk"],
    ["write_file", "Write content to a file on disk"],
    ["execute_command", "Execute a shell command in bash"],
    ["git_commit", "Create a git commit with a message"],
    ["optimize_png", "Optimize and compress PNG image files"],
  ];
}

/**
 * Fake embedding provider for tests — deterministic, keyword-bag based.
 * Produces a 26-dim vector counting lowercase letters in the text.
 * That's enough to give "similar" text pairs a measurably higher cosine
 * similarity than unrelated pairs, without pulling in an ML runtime.
 */
class FakeProvider implements EmbeddingProvider {
  readonly dimension = 26;
  readonly modelName = "fake-alphabet-histogram";

  private textToVec(text: string): number[] {
    const v = new Array(26).fill(0);
    for (const c of text.toLowerCase()) {
      const code = c.charCodeAt(0) - 97;
      if (code >= 0 && code < 26) v[code] += 1;
    }
    return v;
  }
  embed(text: string): Promise<number[]> {
    return Promise.resolve(this.textToVec(text));
  }
  embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.resolve(texts.map((t) => this.textToVec(t)));
  }
}

Deno.test("cosine similarity of identical vectors is 1", () => {
  const a = [1, 2, 3];
  assert(Math.abs(cosineSimilarity(a, a) - 1) < 1e-6);
});

Deno.test("cosine similarity of orthogonal vectors is 0", () => {
  const a = [1, 0];
  const b = [0, 1];
  assert(Math.abs(cosineSimilarity(a, b)) < 1e-6);
});

Deno.test("cosine similarity with zero vector is 0", () => {
  assertEquals(cosineSimilarity([1, 2], [0, 0]), 0);
});

Deno.test("build empty returns empty index", async () => {
  const idx = await ToolEmbeddingIndex.build([], new FakeProvider());
  assertEquals(idx.tool_count(), 0);
  assertEquals((await idx.search("anything", 10, 0)).length, 0);
});

Deno.test("build and search returns non-empty results", async () => {
  const idx = await ToolEmbeddingIndex.build(sampleTools(), new FakeProvider());
  assertEquals(idx.tool_count(), 5);
  const results = await idx.search("compress image", 5, 0);
  assert(results.length > 0);
});

Deno.test("min_score filters low-similarity results", async () => {
  const idx = await ToolEmbeddingIndex.build(sampleTools(), new FakeProvider());
  // Very high threshold — most results must drop.
  const results = await idx.search("random unrelated query xyz", 10, 0.999);
  assert(
    results.length <= 1,
    `expected <=1 result with min_score=0.999, got ${results.length}`,
  );
});

Deno.test("limit caps result count", async () => {
  const idx = await ToolEmbeddingIndex.build(sampleTools(), new FakeProvider());
  const results = await idx.search("file", 2, 0);
  assert(results.length <= 2);
});
