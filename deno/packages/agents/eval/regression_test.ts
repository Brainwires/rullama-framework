import { assert, assertEquals } from "@std/assert";
import type { SuiteResult } from "./suite.ts";
import {
  failingCategoryResults,
  improvedCategoryResults,
  isCiPassing,
  newCategoryBaseline,
  RegressionSuite,
} from "./regression.ts";
import {
  type EvaluationStats,
  evaluationStatsFromTrials,
  type TrialResult,
  trialFailure,
  trialSuccess,
} from "./trial.ts";

function makeStats(successes: number, total: number): EvaluationStats {
  const trials: TrialResult[] = [];
  for (let i = 0; i < total; i++) {
    trials.push(
      i < successes ? trialSuccess(i, 10) : trialFailure(i, 10, "fail"),
    );
  }
  return evaluationStatsFromTrials(trials)!;
}

function makeSuiteResult(
  case_name: string,
  successes: number,
  total: number,
): SuiteResult {
  const trials: TrialResult[] = [];
  for (let i = 0; i < total; i++) {
    trials.push(
      i < successes ? trialSuccess(i, 10) : trialFailure(i, 10, "fail"),
    );
  }
  const stats = evaluationStatsFromTrials(trials)!;
  return {
    case_results: new Map([[case_name, trials]]),
    stats: new Map([[case_name, stats]]),
  };
}

Deno.test("baseline creation", () => {
  const stats = makeStats(80, 100);
  const b = newCategoryBaseline("smoke", stats);
  assertEquals(b.category, "smoke");
  assert(Math.abs(b.baseline_success_rate - 0.8) < 1e-9);
  assertEquals(b.n_trials, 100);
});

Deno.test("check passes when no regression", () => {
  const stats = makeStats(80, 100);
  const reg = new RegressionSuite();
  reg.addBaseline("smoke", stats);

  const sr = makeSuiteResult("smoke", 80, 100);
  const result = reg.check(sr);
  assert(isCiPassing(result));
  assertEquals(failingCategoryResults(result).length, 0);
});

Deno.test("check fails on regression above threshold", () => {
  const baseline = makeStats(90, 100);
  const reg = new RegressionSuite();
  reg.addBaseline("smoke", baseline);

  const sr = makeSuiteResult("smoke", 80, 100);
  const result = reg.check(sr);
  assert(!isCiPassing(result));
  const failing = failingCategoryResults(result);
  assertEquals(failing.length, 1);
  assert(Math.abs(failing[0].regression - 0.1) < 1e-9);
});

Deno.test("check passes regression within threshold", () => {
  const baseline = makeStats(90, 100);
  const reg = new RegressionSuite({ max_regression: 0.1, min_trials: 30 });
  reg.addBaseline("smoke", baseline);

  const sr = makeSuiteResult("smoke", 82, 100);
  const result = reg.check(sr);
  assert(isCiPassing(result));
});

Deno.test("check skips low trial count", () => {
  const baseline = makeStats(90, 100);
  const reg = new RegressionSuite();
  reg.addBaseline("smoke", baseline);

  const sr = makeSuiteResult("smoke", 0, 5);
  const result = reg.check(sr);
  assert(isCiPassing(result));
  assertEquals(result.category_results.length, 0);
});

Deno.test("json roundtrip", () => {
  const stats = makeStats(75, 100);
  const reg = new RegressionSuite();
  reg.addBaseline("smoke", stats);

  const json = reg.baselinesToJson();
  const loaded = RegressionSuite.loadBaselinesFromJson(json);
  const b = loaded.getBaseline("smoke")!;
  assert(Math.abs(b.baseline_success_rate - 0.75) < 1e-9);
});

Deno.test("record baselines from suite result", () => {
  const sr = makeSuiteResult("my_case", 40, 50);
  const reg = new RegressionSuite();
  reg.recordBaselines(sr);

  assert(reg.hasBaseline("my_case"));
  const b = reg.getBaseline("my_case")!;
  assert(Math.abs(b.baseline_success_rate - 0.8) < 1e-9);
});

Deno.test("improved categories", () => {
  const baseline = makeStats(70, 100);
  const reg = new RegressionSuite();
  reg.addBaseline("smoke", baseline);

  const sr = makeSuiteResult("smoke", 90, 100);
  const result = reg.check(sr);
  assert(isCiPassing(result));
  const improved = improvedCategoryResults(result);
  assertEquals(improved.length, 1);
  assert(improved[0].regression < 0.0);
});
