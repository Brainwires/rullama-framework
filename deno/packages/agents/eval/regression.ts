/**
 * Regression testing infrastructure for CI integration.
 *
 * {@link RegressionSuite} compares current {@link SuiteResult} success rates
 * against stored per-category baselines. If any category drops more than
 * {@link RegressionConfig.max_regression} below its baseline, the check
 * fails — enabling CI pipelines to gate on evaluation regressions.
 *
 * Equivalent to Rust's `brainwires_agents::eval::regression` module.
 */

import type { EvaluationStats } from "./trial.ts";
import { evaluationStatsFromTrials, type TrialResult } from "./trial.ts";
import type { SuiteResult } from "./suite.ts";

// ─── Baseline ──────────────────────────────────────────────────────────────

/** Per-category success-rate baseline stored for regression comparison. */
export interface CategoryBaseline {
  /** Category label. */
  category: string;
  /** Baseline success rate in [0, 1]. */
  baseline_success_rate: number;
  /** Unix timestamp (seconds) when this baseline was recorded. */
  measured_at_unix: number;
  /** Number of trials used to compute this baseline. */
  n_trials: number;
}

/** Create a new baseline from measured stats. */
export function newCategoryBaseline(
  category: string,
  stats: EvaluationStats,
): CategoryBaseline {
  return {
    category,
    baseline_success_rate: stats.success_rate,
    measured_at_unix: Math.floor(Date.now() / 1000),
    n_trials: stats.n_trials,
  };
}

// ─── Configuration ─────────────────────────────────────────────────────────

/** Configuration for the regression checker. */
export interface RegressionConfig {
  /** Maximum tolerated regression below baseline in [0, 1]. Default: 0.05 (5%). */
  max_regression: number;
  /**
   * Minimum number of trials required for a category to be checked.
   * Categories with fewer trials are skipped. Default: 30.
   */
  min_trials: number;
}

export function defaultRegressionConfig(): RegressionConfig {
  return { max_regression: 0.05, min_trials: 30 };
}

// ─── Per-category result ───────────────────────────────────────────────────

export interface CategoryRegressionResult {
  category: string;
  current_success_rate: number;
  baseline_success_rate: number;
  /** `baseline - current` (positive = regression, negative = improvement). */
  regression: number;
  passed: boolean;
  reason: string | null;
}

// ─── Aggregate result ──────────────────────────────────────────────────────

export interface RegressionResult {
  /** true when all checked categories passed. */
  passed: boolean;
  category_results: CategoryRegressionResult[];
}

/** Whether all categories passed (suitable for CI gate). */
export function isCiPassing(r: RegressionResult): boolean {
  return r.passed;
}

/** Categories that failed the regression threshold. */
export function failingCategoryResults(
  r: RegressionResult,
): CategoryRegressionResult[] {
  return r.category_results.filter((c) => !c.passed);
}

/** Categories with improvements (negative regression). */
export function improvedCategoryResults(
  r: RegressionResult,
): CategoryRegressionResult[] {
  return r.category_results.filter((c) => c.regression < 0.0);
}

// ─── RegressionSuite ───────────────────────────────────────────────────────

/**
 * Compares evaluation suite results against stored per-category baselines.
 *
 * Fails the check if any category's success rate drops more than
 * {@link RegressionConfig.max_regression} below its baseline.
 */
export class RegressionSuite {
  config: RegressionConfig;
  readonly baselines: Map<string, CategoryBaseline>;

  constructor(config: RegressionConfig = defaultRegressionConfig()) {
    this.config = { ...config };
    this.baselines = new Map();
  }

  /** Create with default config. */
  static new(): RegressionSuite {
    return new RegressionSuite();
  }

  /** Manually register a baseline for a category (chainable). */
  with_baseline(baseline: CategoryBaseline): this {
    this.baselines.set(baseline.category, baseline);
    return this;
  }

  /** Register a baseline from an EvaluationStats object. */
  addBaseline(category: string, stats: EvaluationStats): void {
    this.baselines.set(category, newCategoryBaseline(category, stats));
  }

  /** Record baselines for ALL categories present in the suite result. */
  recordBaselines(suiteResult: SuiteResult): void {
    const catStats = RegressionSuite.aggregateByCategory(suiteResult);
    for (const [category, stats] of catStats) {
      this.addBaseline(category, stats);
    }
  }

  /**
   * Aggregate per-case stats into per-category stats.
   *
   * SuiteResult only keys results by case name, so we fall back to treating
   * each case name as its own category (same as Rust). Callers wanting true
   * category aggregation should use {@link addBaseline} directly.
   */
  private static aggregateByCategory(
    suiteResult: SuiteResult,
  ): Map<string, EvaluationStats> {
    const catTrials = new Map<string, TrialResult[]>();
    for (const [caseName, trials] of suiteResult.case_results) {
      const key = caseName;
      const bucket = catTrials.get(key);
      if (bucket) bucket.push(...trials);
      else catTrials.set(key, [...trials]);
    }

    const out = new Map<string, EvaluationStats>();
    for (const [cat, trials] of catTrials) {
      const stats = evaluationStatsFromTrials(trials);
      if (stats) out.set(cat, stats);
    }
    return out;
  }

  /** Serialize baselines to a pretty-printed JSON string. */
  baselinesToJson(): string {
    const list = [...this.baselines.values()];
    return JSON.stringify(list, null, 2);
  }

  /** True when a baseline has been recorded for the category. */
  hasBaseline(category: string): boolean {
    return this.baselines.has(category);
  }

  /** Retrieve the stored baseline for a category, or null if absent. */
  getBaseline(category: string): CategoryBaseline | null {
    return this.baselines.get(category) ?? null;
  }

  /** Load baselines from a JSON string (produced by {@link baselinesToJson}). */
  static loadBaselinesFromJson(json: string): RegressionSuite {
    const list = JSON.parse(json) as CategoryBaseline[];
    const s = new RegressionSuite();
    for (const b of list) s.baselines.set(b.category, b);
    return s;
  }

  /**
   * Run the regression check against a completed {@link SuiteResult}.
   *
   * For each category with a stored baseline:
   * - Skip if `current_n_trials < min_trials`.
   * - Fail if `baseline_rate - current_rate > max_regression`.
   */
  check(suiteResult: SuiteResult): RegressionResult {
    const currentStats = RegressionSuite.aggregateByCategory(suiteResult);
    const results: CategoryRegressionResult[] = [];
    let allPassed = true;

    for (const [category, baseline] of this.baselines) {
      const current = currentStats.get(category);
      if (!current) continue; // absent from current run
      if (current.n_trials < this.config.min_trials) continue; // not enough data

      const regression = baseline.baseline_success_rate - current.success_rate;
      const passed = regression <= this.config.max_regression;
      const reason = passed ? null : `category '${category}' dropped ${(regression * 100).toFixed(1)}% ` +
        `(from ${(baseline.baseline_success_rate * 100).toFixed(1)}% to ${(current.success_rate * 100).toFixed(1)}%), ` +
        `limit is ${(this.config.max_regression * 100).toFixed(1)}%`;

      if (!passed) allPassed = false;

      results.push({
        category,
        current_success_rate: current.success_rate,
        baseline_success_rate: baseline.baseline_success_rate,
        regression,
        passed,
        reason,
      });
    }

    results.sort((a, b) => a.category.localeCompare(b.category));
    return { passed: allPassed, category_results: results };
  }
}
