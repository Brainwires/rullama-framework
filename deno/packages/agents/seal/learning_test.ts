import { assert, assertEquals } from "@std/assert";
import {
  GlobalMemory,
  LearningCoordinator,
  LocalMemory,
  QueryPattern,
  TrackedEntity,
} from "./learning.ts";
import { newQueryCore, queryVar } from "./query_core.ts";

Deno.test("TrackedEntity record mention", () => {
  const entity = new TrackedEntity("main.rs", "file", 1);
  assertEquals(entity.frequency(), 1);

  entity.recordMention(2);
  entity.recordMention(3);
  assertEquals(entity.frequency(), 3);

  // Duplicate mention should not increase frequency.
  entity.recordMention(2);
  assertEquals(entity.frequency(), 3);
});

Deno.test("LocalMemory tracks entities", () => {
  const local = new LocalMemory("test-conv");
  local.trackEntity("main.rs", "file");
  local.nextTurn();
  local.trackEntity("config.toml", "file");
  local.trackEntity("main.rs", "file");

  assertEquals(local.entities.size, 2);
  assertEquals(local.entities.get("main.rs")!.frequency(), 2);
  // main.rs re-mentioned last, so it should be at the top of the focus stack.
  assertEquals(local.focus_stack[0], "main.rs");
});

Deno.test("QueryPattern reliability", () => {
  const pattern = new QueryPattern("definition", "template", []);
  assertEquals(pattern.reliability(), 0.5);

  pattern.recordSuccess(5);
  pattern.recordSuccess(3);
  pattern.recordFailure();

  // 2 successes, 1 failure = 2/3.
  assert(Math.abs(pattern.reliability() - 0.666) < 0.01);
});

Deno.test("GlobalMemory patterns sorted by reliability", () => {
  const global = new GlobalMemory();

  const p1 = new QueryPattern("definition", "template1", []);
  p1.recordSuccess(5);
  p1.recordSuccess(5);

  const p2 = new QueryPattern("definition", "template2", []);
  p2.recordFailure();

  global.addPattern(p1);
  global.addPattern(p2);

  const patterns = global.getPatterns("definition");
  assertEquals(patterns.length, 2);
  assert(patterns[0].reliability() > patterns[1].reliability());
});

Deno.test("LearningCoordinator learns from successful query", () => {
  const coord = new LearningCoordinator("test-conv");
  const core = newQueryCore(
    "definition",
    queryVar("x"),
    [["main.rs", "file"]],
    "What is main.rs?",
  );
  coord.recordOutcome(undefined, true, 1, core, 0);
  const stats = coord.getStats();
  assertEquals(stats.session_queries, 1);
  assertEquals(stats.global_patterns, 1);
});

Deno.test("QueryPattern matchesTypes", () => {
  const pattern = new QueryPattern("definition", "template", ["file"]);
  assert(pattern.matchesTypes(["file"]));
  assert(pattern.matchesTypes(["file", "function"]));
  assert(!pattern.matchesTypes(["function"]));
});

Deno.test("GlobalMemory prune removes low-reliability patterns", () => {
  const global = new GlobalMemory();

  const good = new QueryPattern("definition", "good", []);
  for (let i = 0; i < 10; i++) good.recordSuccess(5);

  const bad = new QueryPattern("definition", "bad", []);
  for (let i = 0; i < 10; i++) bad.recordFailure();

  global.addPattern(good);
  global.addPattern(bad);
  assertEquals(global.getPatterns("definition").length, 2);

  global.prunePatterns(0.5, 5);
  assertEquals(global.getPatterns("definition").length, 1);
});

Deno.test("LearningCoordinator getContextForPrompt mentions frequent entities", () => {
  const coord = new LearningCoordinator("test");
  coord.local.trackEntity("main.rs", "file");
  coord.local.trackEntity("main.rs", "file");
  coord.local.trackEntity("config.toml", "file");
  const context = coord.getContextForPrompt();
  assert(context.includes("main.rs") || context.includes("Frequently"));
});
