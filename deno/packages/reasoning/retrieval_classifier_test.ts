import { assert, assertEquals } from "@std/assert";
import {
  classificationFromFallback,
  classificationFromLocal,
  classifyHeuristic,
  parseClassification,
  retrievalScore,
  shouldRetrieve,
} from "./retrieval_classifier.ts";

Deno.test("retrieval need methods", () => {
  assert(!shouldRetrieve("none"));
  assert(!shouldRetrieve("low"));
  assert(shouldRetrieve("medium"));
  assert(shouldRetrieve("high"));
  assertEquals(retrievalScore("none"), 0);
  assert(retrievalScore("high") > retrievalScore("low"));
});

Deno.test("classification result constructors", () => {
  const local = classificationFromLocal("high", 0.9, "references earlier discussion");
  assert(local.used_local_llm);
  assertEquals(local.intent, "references earlier discussion");
  const fb = classificationFromFallback("medium", 0.7);
  assert(!fb.used_local_llm);
  assertEquals(fb.intent, null);
});

Deno.test("heuristic: reference pattern -> high", () => {
  const r = classifyHeuristic("What did we discuss earlier?", 10);
  assertEquals(r.need, "high");
});

Deno.test("heuristic: self-contained -> none", () => {
  const r = classifyHeuristic("Write a hello world function in Python", 20);
  assertEquals(r.need, "none");
});

Deno.test("heuristic: short context bumps need up", () => {
  const r = classifyHeuristic("Continue please", 2);
  assert(shouldRetrieve(r.need));
});

Deno.test("parseClassification extracts level + intent", () => {
  assertEquals(parseClassification("HIGH: references earlier discussion").need, "high");
  assertEquals(parseClassification("NONE: self-contained query").need, "none");
});
