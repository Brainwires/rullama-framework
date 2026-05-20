import { assert, assertEquals } from "@std/assert";
import { AlwaysFailCase, AlwaysPassCase, StochasticCase } from "./case.ts";
import {
  EvaluationSuite,
  failingCases,
  overallSuccessRate,
} from "./suite.ts";

Deno.test("suite all pass", async () => {
  const suite = new EvaluationSuite(5);
  const result = await suite.runSuite([new AlwaysPassCase("ok")]);
  const stats = result.stats.get("ok")!;
  assertEquals(stats.n_trials, 5);
  assertEquals(stats.successes, 5);
  assert(Math.abs(stats.success_rate - 1.0) < 1e-9);
  assert(Math.abs(overallSuccessRate(result) - 1.0) < 1e-9);
});

Deno.test("suite all fail", async () => {
  const suite = new EvaluationSuite(3);
  const result = await suite.runSuite([new AlwaysFailCase("bad", "expected")]);
  const stats = result.stats.get("bad")!;
  assertEquals(stats.successes, 0);
  assertEquals(stats.success_rate, 0.0);
});

Deno.test("suite multiple cases", async () => {
  const suite = new EvaluationSuite(10);
  const result = await suite.runSuite([
    new AlwaysPassCase("pass"),
    new AlwaysFailCase("fail", "x"),
  ]);
  assert(result.stats.has("pass"));
  assert(result.stats.has("fail"));
  assert(Math.abs(overallSuccessRate(result) - 0.5) < 1e-9);
});

Deno.test("suite n_trials minimum one", async () => {
  const suite = new EvaluationSuite(0); // clamps to 1
  const result = await suite.runSuite([new AlwaysPassCase("x")]);
  assertEquals(result.stats.get("x")!.n_trials, 1);
});

Deno.test("runCase returns correct count", async () => {
  const suite = new EvaluationSuite(7);
  const results = await suite.runCase(new AlwaysPassCase("seven"));
  assertEquals(results.length, 7);
  results.forEach((r, i) => assertEquals(r.trial_id, i));
});

Deno.test("failing cases filter", async () => {
  const suite = new EvaluationSuite(10);
  const result = await suite.runSuite([
    new AlwaysPassCase("good"),
    new StochasticCase("flaky", 0.0), // always fails
  ]);
  const failing = failingCases(result, 0.5);
  assert(failing.includes("flaky"));
  assert(!failing.includes("good"));
});

Deno.test("confidence interval in suite result", async () => {
  const suite = new EvaluationSuite(50);
  const result = await suite.runSuite([new StochasticCase("ci_test", 0.8)]);
  const stats = result.stats.get("ci_test")!;
  const ci = stats.confidence_interval_95;
  assert(ci.lower < 0.85 && ci.upper > 0.65);
});
