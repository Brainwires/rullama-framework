import { assert, assertEquals } from "@std/assert";
import {
  GoalPreservationCase,
  LoopDetectionSimCase,
  longHorizonStabilitySuite,
} from "./stability_tests.ts";
import { EvaluationSuite } from "./suite.ts";

Deno.test("loop sim fires at correct step", () => {
  const c = LoopDetectionSimCase.shouldDetect(20, "read_file", 3, 5);
  assert(c.simulate(), "expected loop detection to fire");
});

Deno.test("loop sim does not fire diverse", () => {
  const c = LoopDetectionSimCase.shouldNotDetect(20, 5);
  assert(!c.simulate(), "expected no detection on diverse sequence");
});

Deno.test("loop sim fires immediately", () => {
  const c = LoopDetectionSimCase.shouldDetect(10, "write_file", 1, 3);
  assert(c.simulate());
});

Deno.test("loop sim short run no loop", () => {
  const c = LoopDetectionSimCase.shouldDetect(2, "read_file", 1, 5);
  assert(!c.simulate());
});

Deno.test("goal injection points 15iter interval10", () => {
  const c = new GoalPreservationCase(15, 10);
  assertEquals(c.expectedInjectionPoints(), [11]);
});

Deno.test("goal injection points 20iter interval5", () => {
  const c = new GoalPreservationCase(20, 5);
  assertEquals(c.expectedInjectionPoints(), [6, 11, 16]);
});

Deno.test("goal injection simulation matches expected", () => {
  const c = new GoalPreservationCase(30, 10);
  assertEquals(c.simulateInjections(), c.expectedInjectionPoints());
});

Deno.test("loop detection case succeeds when loop fires", async () => {
  const c = LoopDetectionSimCase.shouldDetect(20, "read_file", 3, 5);
  const r = await c.run(0);
  assert(r.success, `error: ${r.error}`);
});

Deno.test("loop detection case fails when no loop fires", async () => {
  const c = LoopDetectionSimCase.shouldDetect(2, "read_file", 1, 5);
  const r = await c.run(0);
  assert(!r.success);
});

Deno.test("goal preservation case succeeds", async () => {
  const c = new GoalPreservationCase(20, 5);
  const r = await c.run(0);
  assert(r.success, `error: ${r.error}`);
});

Deno.test("full stability suite runs", async () => {
  const suite = new EvaluationSuite(1);
  const cases = longHorizonStabilitySuite();
  const results = await suite.runSuite(cases);
  assert(results.case_results.size > 0);
});
