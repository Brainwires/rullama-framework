/**
 * Evaluation suite — N-trial Monte Carlo runner.
 *
 * {@link EvaluationSuite} runs each registered {@link EvaluationCase} N times,
 * collects {@link TrialResult}s, and computes {@link EvaluationStats} for
 * every case.
 *
 * Equivalent to Rust's `brainwires_agents::eval::suite` module.
 */

import type { EvaluationCase } from "./case.ts";
import {
  type EvaluationStats,
  evaluationStatsFromTrials,
  type TrialResult,
  trialFailure,
} from "./trial.ts";

// ─── Suite result ──────────────────────────────────────────────────────────

/** Aggregated results for all cases in a suite run. */
export interface SuiteResult {
  /** Raw trial results keyed by case name. */
  case_results: Map<string, TrialResult[]>;
  /** Summary statistics keyed by case name. */
  stats: Map<string, EvaluationStats>;
}

/** Overall success rate across *all* cases and trials. */
export function overallSuccessRate(sr: SuiteResult): number {
  let total = 0;
  let successes = 0;
  for (const trials of sr.case_results.values()) {
    total += trials.length;
    for (const t of trials) if (t.success) successes += 1;
  }
  return total === 0 ? 0.0 : successes / total;
}

/** Returns all cases whose success rate is strictly below `threshold`. */
export function failingCases(sr: SuiteResult, threshold: number): string[] {
  const out: string[] = [];
  for (const [name, s] of sr.stats.entries()) {
    if (s.success_rate < threshold) out.push(name);
  }
  return out;
}

// ─── Suite configuration ───────────────────────────────────────────────────

/** Configuration for {@link EvaluationSuite}. */
export interface SuiteConfig {
  /** Number of times each case is run. Minimum 1. */
  n_trials: number;
  /**
   * Maximum number of trials to execute concurrently per case.
   * `1` means sequential execution (deterministic ordering).
   */
  max_parallel: number;
  /**
   * If `true`, a single trial error (not a test failure, but a hard JS
   * error) is treated as a test failure rather than propagating to the
   * caller.
   */
  catch_errors_as_failures: boolean;
}

export function defaultSuiteConfig(): SuiteConfig {
  return { n_trials: 10, max_parallel: 1, catch_errors_as_failures: true };
}

// ─── Suite ─────────────────────────────────────────────────────────────────

/** N-trial Monte Carlo evaluation runner. */
export class EvaluationSuite {
  readonly config: SuiteConfig;

  /** Create a suite that runs each case `n_trials` times sequentially. */
  constructor(n_trials: number) {
    this.config = {
      ...defaultSuiteConfig(),
      n_trials: Math.max(1, n_trials),
    };
  }

  /** Override the full configuration. */
  static withConfig(config: SuiteConfig): EvaluationSuite {
    const s = new EvaluationSuite(config.n_trials);
    (s as { config: SuiteConfig }).config = {
      ...config,
      n_trials: Math.max(1, config.n_trials),
    };
    return s;
  }

  /** Run `n_trials` for a single case and return the raw results. */
  async runCase(c: EvaluationCase): Promise<TrialResult[]> {
    const results: TrialResult[] = [];

    if (this.config.max_parallel <= 1) {
      for (let trial_id = 0; trial_id < this.config.n_trials; trial_id++) {
        try {
          results.push(await c.run(trial_id));
        } catch (e) {
          results.push(this.resolveError(e, trial_id));
        }
      }
    } else {
      // Bounded parallel execution via a simple semaphore pattern.
      const pending: Promise<TrialResult>[] = [];
      let inFlight = 0;
      const waiters: Array<() => void> = [];

      const acquire = async () => {
        while (inFlight >= this.config.max_parallel) {
          await new Promise<void>((resolve) => waiters.push(resolve));
        }
        inFlight += 1;
      };
      const release = () => {
        inFlight -= 1;
        const w = waiters.shift();
        if (w) w();
      };

      for (let trial_id = 0; trial_id < this.config.n_trials; trial_id++) {
        await acquire();
        const id = trial_id;
        pending.push(
          (async () => {
            try {
              return await c.run(id);
            } catch (e) {
              return this.resolveError(e, id);
            } finally {
              release();
            }
          })(),
        );
      }

      const settled = await Promise.all(pending);
      results.push(...settled);
      results.sort((a, b) => a.trial_id - b.trial_id);
    }

    return results;
  }

  /** Run the full suite: execute each case N times and return aggregated results. */
  async runSuite(cases: readonly EvaluationCase[]): Promise<SuiteResult> {
    const case_results = new Map<string, TrialResult[]>();
    const stats = new Map<string, EvaluationStats>();

    for (const c of cases) {
      const results = await this.runCase(c);
      const caseStats = evaluationStatsFromTrials(results);
      if (!caseStats) {
        throw new Error("case must have at least one trial");
      }
      const name = c.name();
      case_results.set(name, results);
      stats.set(name, caseStats);
    }

    return { case_results, stats };
  }

  private resolveError(e: unknown, trial_id: number): TrialResult {
    const msg = e instanceof Error ? e.message : String(e);
    if (this.config.catch_errors_as_failures) {
      return trialFailure(trial_id, 0, msg);
    }
    return trialFailure(trial_id, 0, `Trial errored: ${msg}`);
  }
}
