import { assert, assertEquals } from "@std/assert";
import {
  evaluationStatsFromTrials,
  percentile,
  trialFailure,
  trialSuccess,
  trialWithMeta,
  wilsonInterval,
} from "./trial.ts";

Deno.test("trial_success builder", () => {
  const t = trialSuccess(0, 42);
  assert(t.success);
  assertEquals(t.trial_id, 0);
  assertEquals(t.duration_ms, 42);
  assertEquals(t.error, null);
});

Deno.test("trial_failure builder", () => {
  const t = trialFailure(1, 100, "timeout");
  assert(!t.success);
  assertEquals(t.error, "timeout");
});

Deno.test("trial_with_meta attaches metadata", () => {
  let t = trialSuccess(0, 10);
  t = trialWithMeta(t, "iterations", 7);
  t = trialWithMeta(t, "model", "claude-sonnet");
  assertEquals(t.metadata["iterations"], 7);
});

Deno.test("wilson_ci all successes", () => {
  const ci = wilsonInterval(10, 10);
  assert(ci.lower > 0.7, "lower bound should be well above 0 for 10/10");
  assert(Math.abs(ci.upper - 1.0) < 1e-9, "upper bound should be 1.0");
});

Deno.test("wilson_ci no successes", () => {
  const ci = wilsonInterval(0, 10);
  assertEquals(ci.lower, 0.0);
  assert(ci.upper < 0.3, "upper bound should be low for 0/10");
});

Deno.test("wilson_ci zero trials", () => {
  const ci = wilsonInterval(0, 0);
  assertEquals(ci.lower, 0.0);
  assertEquals(ci.upper, 1.0);
});

Deno.test("wilson_ci contains true rate", () => {
  const ci = wilsonInterval(70, 100);
  assert(ci.lower < 0.70 && ci.upper > 0.70);
});

Deno.test("evaluation_stats empty", () => {
  assertEquals(evaluationStatsFromTrials([]), null);
});

Deno.test("evaluation_stats all success", () => {
  const trials = Array.from({ length: 10 }, (_, i) => trialSuccess(i, 100));
  const stats = evaluationStatsFromTrials(trials)!;
  assertEquals(stats.n_trials, 10);
  assertEquals(stats.successes, 10);
  assert(Math.abs(stats.success_rate - 1.0) < 1e-9);
});

Deno.test("evaluation_stats mixed", () => {
  const trials = [
    ...Array.from({ length: 7 }, (_, i) => trialSuccess(i, 50)),
    ...Array.from({ length: 3 }, (_, i) => trialFailure(7 + i, 200, "err")),
  ];
  const stats = evaluationStatsFromTrials(trials)!;
  assertEquals(stats.successes, 7);
  assert(Math.abs(stats.success_rate - 0.7) < 1e-9);
  assert(stats.p95_duration_ms >= stats.p50_duration_ms);
  assert(stats.p50_duration_ms >= stats.mean_duration_ms * 0.5);
});

Deno.test("percentile single element", () => {
  assertEquals(percentile([42.0], 50.0), 42.0);
});

Deno.test("percentile interpolation", () => {
  const data = [0.0, 10.0, 20.0, 30.0, 40.0];
  const p50 = percentile(data, 50.0);
  assert(Math.abs(p50 - 20.0) < 1e-9);
});
