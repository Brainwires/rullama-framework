import { assert, assertEquals } from "@std/assert";
import type { SuiteResult } from "./suite.ts";
import {
  analyzeSuiteForFaults,
  faultKindPriority,
  faultReportPriority,
  newCapabilityFault,
  regressionFault,
} from "./fault_report.ts";
import { RegressionSuite } from "./regression.ts";
import {
  evaluationStatsFromTrials,
  type TrialResult,
  trialFailure,
  trialSuccess,
} from "./trial.ts";

function makeSuiteResult(
  case_name: string,
  successes: number,
  total: number,
): SuiteResult {
  const trials: TrialResult[] = [];
  for (let i = 0; i < total; i++) {
    trials.push(
      i < successes ? trialSuccess(i, 1) : trialFailure(i, 1, `error_${i}`),
    );
  }
  const stats = evaluationStatsFromTrials(trials)!;
  return {
    case_results: new Map([[case_name, trials]]),
    stats: new Map([[case_name, stats]]),
  };
}

// ── FaultKind priority ─────────────────────────────────────────────────

Deno.test("priority regression scaled by drop", () => {
  assertEquals(
    faultKindPriority({
      type: "regression",
      previous_rate: 0.9,
      current_rate: 0.85,
      drop: 0.05,
    }),
    5,
  );
});

Deno.test("priority regression capped at 10", () => {
  assertEquals(
    faultKindPriority({
      type: "regression",
      previous_rate: 1.0,
      current_rate: 0.75,
      drop: 0.25,
    }),
    10,
  );
});

Deno.test("priority consistent failure", () => {
  assertEquals(
    faultKindPriority({ type: "consistent_failure", success_rate: 0.1 }),
    8,
  );
});

Deno.test("priority new capability", () => {
  assertEquals(
    faultKindPriority({ type: "new_capability", description: "x" }),
    5,
  );
});

Deno.test("priority flaky", () => {
  assertEquals(
    faultKindPriority({ type: "flaky", mean_rate: 0.5, ci_width: 0.3 }),
    4,
  );
});

// ── Constructors ───────────────────────────────────────────────────────

Deno.test("regression constructor sets fields", () => {
  const r = regressionFault("my_case", "smoke", 0.9, 0.7, ["err1"], 3, 10);
  assertEquals(r.case_name, "my_case");
  assertEquals(r.category, "smoke");
  assertEquals(r.n_failures, 3);
  assertEquals(r.n_trials, 10);
  assert(r.suggested_task_description.includes("my_case"));
  assert(r.suggested_task_description.includes("regression"));
  if (r.fault_kind.type !== "regression") throw new Error("wrong variant");
  assert(Math.abs(r.fault_kind.drop - 0.2) < 1e-9);
  assert(Math.abs(r.fault_kind.previous_rate - 0.9) < 1e-9);
  assert(Math.abs(r.fault_kind.current_rate - 0.7) < 1e-9);
});

Deno.test("new capability constructor", () => {
  const r = newCapabilityFault("new_case", "cat", "desc", 0.85, 1, 10);
  assertEquals(r.case_name, "new_case");
  assertEquals(r.fault_kind.type, "new_capability");
  assertEquals(r.sample_errors.length, 0);
});

// ── analyzeSuiteForFaults ──────────────────────────────────────────────

Deno.test("consistent failure detected", () => {
  const sr = makeSuiteResult("bad_case", 1, 20); // 5%
  const reports = analyzeSuiteForFaults(sr, null, 0.2, 0.25);
  assertEquals(reports.length, 1);
  assertEquals(reports[0].fault_kind.type, "consistent_failure");
  assertEquals(reports[0].case_name, "bad_case");
});

Deno.test("regression detected when drop exceeds tolerance", () => {
  const currentSr = makeSuiteResult("my_case", 7, 10); // 70%
  const reg = new RegressionSuite();
  const baselineTrials: TrialResult[] = [];
  for (let i = 0; i < 10; i++) {
    baselineTrials.push(
      i < 9 ? trialSuccess(i, 1) : trialFailure(i, 1, "e"),
    );
  }
  const baselineStats = evaluationStatsFromTrials(baselineTrials)!;
  reg.addBaseline("my_case", baselineStats);

  const reports = analyzeSuiteForFaults(currentSr, reg, 0.2, 0.25);
  assert(reports.some((r) => r.fault_kind.type === "regression"));
});

Deno.test("no fault when within tolerance", () => {
  const sr = makeSuiteResult("ok_case", 88, 100);
  const reg = new RegressionSuite();
  const baselineTrials: TrialResult[] = [];
  for (let i = 0; i < 100; i++) {
    baselineTrials.push(
      i < 90 ? trialSuccess(i, 1) : trialFailure(i, 1, "e"),
    );
  }
  const baselineStats = evaluationStatsFromTrials(baselineTrials)!;
  reg.addBaseline("ok_case", baselineStats);

  const reports = analyzeSuiteForFaults(sr, reg, 0.2, 0.25);
  assertEquals(reports.length, 0);
});

Deno.test("no fault for passing case without regression suite", () => {
  const sr = makeSuiteResult("good_case", 45, 50);
  const reports = analyzeSuiteForFaults(sr, null, 0.2, 0.25);
  assertEquals(reports.length, 0);
});

Deno.test("new capability when regression suite provided but no matching baseline", () => {
  const sr = makeSuiteResult("new_case", 45, 50);
  const reg = new RegressionSuite();
  const reports = analyzeSuiteForFaults(sr, reg, 0.2, 0.25);
  assert(reports.some((r) => r.fault_kind.type === "new_capability"));
});

Deno.test("results sorted by priority descending", () => {
  const badTrials: TrialResult[] = [];
  for (let i = 0; i < 10; i++) {
    badTrials.push(i < 1 ? trialSuccess(i, 1) : trialFailure(i, 1, "e"));
  }
  const flakyTrials: TrialResult[] = [];
  for (let i = 0; i < 10; i++) {
    flakyTrials.push(i < 5 ? trialSuccess(i, 1) : trialFailure(i, 1, "e"));
  }
  const sr: SuiteResult = {
    case_results: new Map([
      ["bad", badTrials],
      ["flaky", flakyTrials],
    ]),
    stats: new Map([
      ["bad", evaluationStatsFromTrials(badTrials)!],
      ["flaky", evaluationStatsFromTrials(flakyTrials)!],
    ]),
  };
  const reports = analyzeSuiteForFaults(sr, null, 0.2, 0.25);
  assert(reports.length >= 2);
  for (let i = 0; i < reports.length - 1; i++) {
    assert(faultReportPriority(reports[i]) >= faultReportPriority(reports[i + 1]));
  }
});

Deno.test("sample errors collected", () => {
  const sr = makeSuiteResult("broken", 0, 5);
  const reports = analyzeSuiteForFaults(sr, null, 0.2, 0.25);
  assert(reports.length > 0);
  assert(reports[0].sample_errors.length > 0);
  assert(reports[0].sample_errors.length <= 3);
});
