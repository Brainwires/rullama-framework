import { assert, assertEquals } from "@std/assert";
import { mrr, ndcgAtK, precisionAtK } from "./ranking_metrics.ts";

// ── ndcgAtK ───────────────────────────────────────────────────────────────

Deno.test("ndcg perfect ranking", () => {
  const scores = [0.9, 0.7, 0.4, 0.1];
  const rel = [3, 2, 1, 0];
  const n = ndcgAtK(scores, rel, 4);
  assert(Math.abs(n - 1.0) < 1e-9, `perfect ranking should yield 1.0, got ${n}`);
});

Deno.test("ndcg worst ranking", () => {
  const scores = [0.1, 0.4, 0.7, 0.9];
  const rel = [3, 2, 1, 0];
  const n = ndcgAtK(scores, rel, 4);
  assert(n < 0.65 && n > 0.0);
  const perfect = ndcgAtK([0.9, 0.7, 0.4, 0.1], rel, 4);
  assert(n < perfect);
});

Deno.test("ndcg all zero relevance", () => {
  const scores = [0.9, 0.5, 0.1];
  const rel = [0, 0, 0];
  assertEquals(ndcgAtK(scores, rel, 3), 0.0);
});

Deno.test("ndcg empty", () => {
  assertEquals(ndcgAtK([], [], 0), 0.0);
});

Deno.test("ndcg k truncates", () => {
  const scores = [0.9, 0.7, 0.5, 0.3];
  const rel = [2, 2, 0, 2];
  const k2 = ndcgAtK(scores, rel, 2);
  const k4 = ndcgAtK(scores, rel, 4);
  assert(Math.abs(k2 - 1.0) < 1e-9);
  assert(k4 < 1.0);
  assert(k2 > k4);
});

Deno.test("ndcg k zero means all", () => {
  const scores = [0.9, 0.5];
  const rel = [2, 1];
  assert(Math.abs(ndcgAtK(scores, rel, 0) - ndcgAtK(scores, rel, 2)) < 1e-9);
});

// ── mrr ───────────────────────────────────────────────────────────────────

Deno.test("mrr first is relevant", () => {
  const scores = [0.9, 0.5, 0.1];
  const rel = [1, 0, 0];
  assert(Math.abs(mrr(scores, rel) - 1.0) < 1e-9);
});

Deno.test("mrr second is relevant", () => {
  const scores = [0.9, 0.5, 0.1];
  const rel = [0, 1, 0];
  assert(Math.abs(mrr(scores, rel) - 0.5) < 1e-9);
});

Deno.test("mrr no relevant", () => {
  assertEquals(mrr([0.9, 0.5], [0, 0]), 0.0);
});

Deno.test("mrr empty", () => {
  assertEquals(mrr([], []), 0.0);
});

// ── precisionAtK ──────────────────────────────────────────────────────────

Deno.test("precision all relevant", () => {
  const scores = [0.9, 0.7, 0.5];
  const rel = [1, 1, 1];
  assert(Math.abs(precisionAtK(scores, rel, 3) - 1.0) < 1e-9);
});

Deno.test("precision half relevant", () => {
  const scores = [0.9, 0.8, 0.5, 0.1];
  const rel = [1, 1, 0, 0];
  assert(Math.abs(precisionAtK(scores, rel, 4) - 0.5) < 1e-9);
});

Deno.test("precision truncates", () => {
  const scores = [0.9, 0.8, 0.3, 0.1];
  const rel = [1, 1, 0, 0];
  assert(Math.abs(precisionAtK(scores, rel, 2) - 1.0) < 1e-9);
});

Deno.test("precision zero k means all", () => {
  const scores = [0.9, 0.5];
  const rel = [1, 0];
  assert(
    Math.abs(precisionAtK(scores, rel, 0) - precisionAtK(scores, rel, 2)) <
      1e-9,
  );
});
