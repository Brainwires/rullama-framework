/**
 * Fault classification for eval-driven autonomous self-improvement.
 *
 * {@link analyzeSuiteForFaults} inspects a completed {@link SuiteResult} and
 * classifies per-case issues into {@link FaultReport}s sorted by priority.
 *
 * Equivalent to Rust's `brainwires_agents::eval::fault_report` module.
 */

import type { SuiteResult } from "./suite.ts";
import type { RegressionSuite } from "./regression.ts";

// ─── FaultKind ─────────────────────────────────────────────────────────────

export type FaultKind =
  | {
    type: "regression";
    previous_rate: number;
    current_rate: number;
    drop: number;
  }
  | { type: "new_capability"; description: string }
  | { type: "consistent_failure"; success_rate: number }
  | { type: "flaky"; mean_rate: number; ci_width: number };

/** Scheduling priority (higher = more urgent, max 10). */
export function faultKindPriority(k: FaultKind): number {
  switch (k.type) {
    case "regression": {
      const scaled = Math.round(k.drop * 100);
      return Math.max(1, Math.min(10, scaled));
    }
    case "consistent_failure":
      return 8;
    case "new_capability":
      return 5;
    case "flaky":
      return 4;
  }
}

/** Short label for use in reports and task IDs. */
export function faultKindLabel(k: FaultKind): string {
  switch (k.type) {
    case "regression":
      return "regression";
    case "new_capability":
      return "new_capability";
    case "consistent_failure":
      return "consistent_failure";
    case "flaky":
      return "flaky";
  }
}

// ─── FaultReport ───────────────────────────────────────────────────────────

/** A classified eval fault ready for self-improvement task generation. */
export interface FaultReport {
  case_name: string;
  category: string;
  fault_kind: FaultKind;
  sample_errors: string[];
  n_failures: number;
  n_trials: number;
  suggested_task_description: string;
}

/** Priority for a fault report (delegates to {@link faultKindPriority}). */
export function faultReportPriority(r: FaultReport): number {
  return faultKindPriority(r.fault_kind);
}

/** Construct a regression fault. */
export function regressionFault(
  case_name: string,
  category: string,
  previous_rate: number,
  current_rate: number,
  sample_errors: string[],
  n_failures: number,
  n_trials: number,
): FaultReport {
  const drop = Math.max(0.0, previous_rate - current_rate);
  const suggested = `Fix regression in eval case '${case_name}': success rate dropped from ` +
    `${(previous_rate * 100).toFixed(0)}% to ${(current_rate * 100).toFixed(0)}% ` +
    `(drop: ${(drop * 100).toFixed(0)}%). Investigate recent changes and restore reliability.`;
  return {
    case_name,
    category,
    fault_kind: { type: "regression", previous_rate, current_rate, drop },
    sample_errors,
    n_failures,
    n_trials,
    suggested_task_description: suggested,
  };
}

/** Construct a new-capability fault. */
export function newCapabilityFault(
  case_name: string,
  category: string,
  description: string,
  success_rate: number,
  n_failures: number,
  n_trials: number,
): FaultReport {
  const suggested = `Record baseline for newly-observed eval case '${case_name}' ` +
    `(${(success_rate * 100).toFixed(0)}% success rate). ` +
    `Add documentation and verify the capability is tested consistently.`;
  return {
    case_name,
    category,
    fault_kind: { type: "new_capability", description },
    sample_errors: [],
    n_failures,
    n_trials,
    suggested_task_description: suggested,
  };
}

// ─── analyzeSuiteForFaults ─────────────────────────────────────────────────

/**
 * Inspect a {@link SuiteResult} and return classified {@link FaultReport}s.
 *
 * Classification precedence (first match wins per case):
 * 1. **regression** — baseline exists and `current < baseline − 0.03`.
 * 2. **consistent_failure** — `success_rate < consistent_failure_threshold`.
 * 3. **flaky** — CI width > `flaky_ci_threshold` (requires ≥1 failure).
 * 4. **new_capability** — no baseline recorded yet and `success_rate ≥ 0.8`.
 */
export function analyzeSuiteForFaults(
  suiteResult: SuiteResult,
  regressionSuite: RegressionSuite | null,
  consistent_failure_threshold: number,
  flaky_ci_threshold: number,
): FaultReport[] {
  const reports: FaultReport[] = [];

  for (const [case_name, stats] of suiteResult.stats) {
    const n_trials = stats.n_trials;
    const n_failures = n_trials - stats.successes;
    const success_rate = stats.success_rate;
    const ci_width = stats.confidence_interval_95.upper -
      stats.confidence_interval_95.lower;

    const caseTrials = suiteResult.case_results.get(case_name) ?? [];
    const sample_errors = caseTrials
      .map((t) => t.error)
      .filter((e): e is string => e !== null)
      .slice(0, 3);

    const baseline = regressionSuite?.getBaseline(case_name) ?? null;

    // 1. Regression
    if (baseline) {
      const drop = baseline.baseline_success_rate - success_rate;
      if (drop > 0.03) {
        reports.push(
          regressionFault(
            case_name,
            case_name,
            baseline.baseline_success_rate,
            success_rate,
            sample_errors,
            n_failures,
            n_trials,
          ),
        );
        continue;
      }
    }

    // 2. Consistent failure
    if (success_rate < consistent_failure_threshold) {
      const suggested = `Fix consistently failing eval case '${case_name}' ` +
        `(success rate: ${(success_rate * 100).toFixed(0)}%). ` +
        `Review the implementation and ensure the evaluated functionality works correctly.`;
      reports.push({
        case_name,
        category: case_name,
        fault_kind: { type: "consistent_failure", success_rate },
        sample_errors,
        n_failures,
        n_trials,
        suggested_task_description: suggested,
      });
      continue;
    }

    // 3. Flaky
    if (n_failures > 0 && ci_width > flaky_ci_threshold) {
      const suggested = `Stabilize flaky eval case '${case_name}' ` +
        `(mean success: ${(success_rate * 100).toFixed(0)}%, CI width: ${ci_width.toFixed(2)}). ` +
        `Investigate sources of non-determinism and improve consistency.`;
      reports.push({
        case_name,
        category: case_name,
        fault_kind: {
          type: "flaky",
          mean_rate: success_rate,
          ci_width,
        },
        sample_errors,
        n_failures,
        n_trials,
        suggested_task_description: suggested,
      });
      continue;
    }

    // 4. New capability
    if (baseline === null && regressionSuite !== null && success_rate >= 0.8) {
      reports.push(
        newCapabilityFault(
          case_name,
          case_name,
          `New eval case '${case_name}' achieving ${(success_rate * 100).toFixed(0)}% success — baseline not yet recorded`,
          success_rate,
          n_failures,
          n_trials,
        ),
      );
    }
  }

  // Sort by priority descending.
  reports.sort((a, b) => faultReportPriority(b) - faultReportPriority(a));
  return reports;
}
