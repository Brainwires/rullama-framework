import { assert, assertEquals } from "@std/assert";
import {
  complexityFromLocal,
  defaultComplexity,
  parseScore as parseComplexityScore,
  scoreHeuristic,
} from "./complexity.ts";

Deno.test("default complexity is medium", () => {
  const r = defaultComplexity();
  assertEquals(r.score, 0.5);
  assert(!r.used_local_llm);
});

Deno.test("complexity clamps to 0-1", () => {
  assertEquals(complexityFromLocal(1.5, 0.9).score, 1);
  assertEquals(complexityFromLocal(-0.5, 0.9).score, 0);
});

Deno.test("heuristic scoring", () => {
  const simple = scoreHeuristic("read a file");
  assert(simple.score < 0.5);
  const complex = scoreHeuristic(
    "refactor the architecture to implement a distributed concurrent system with multiple parallel workers",
  );
  assert(complex.score > 0.5);
});

Deno.test("parseScore picks number out of prose", () => {
  assertEquals(parseComplexityScore("0.5"), 0.5);
  assertEquals(parseComplexityScore("0.85"), 0.85);
  assertEquals(parseComplexityScore("The complexity is 0.7"), 0.7);
  assertEquals(parseComplexityScore("1.5"), 1);
  assertEquals(parseComplexityScore("no number here"), null);
});
