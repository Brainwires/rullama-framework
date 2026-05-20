/**
 * The {@link EvaluationCase} interface — the unit of evaluation.
 *
 * Implement this interface for any scenario you want to evaluate N times.
 *
 * Equivalent to Rust's `brainwires_agents::eval::case` module.
 */

import { type TrialResult, trialFailure, trialSuccess } from "./trial.ts";

/**
 * A single evaluation scenario.
 *
 * Implement this interface and pass instances to {@link EvaluationSuite} to
 * run N independent trials and compute statistics.
 */
export interface EvaluationCase {
  /** Short identifier used in reports and log output. */
  name(): string;
  /**
   * Category label for grouping (e.g. `"smoke"`, `"adversarial"`,
   * `"budget_stress"`).
   */
  category(): string;
  /**
   * Execute one trial and return its result.
   *
   * The implementation is responsible for measuring wall-clock duration and
   * encoding it in the returned {@link TrialResult}.
   */
  run(trial_id: number): Promise<TrialResult>;
}

/**
 * A minimal no-op evaluation case useful for unit-testing the evaluation
 * infrastructure itself.
 */
export class AlwaysPassCase implements EvaluationCase {
  /** Short identifier for this case. */
  private name_: string;
  /** Category label for grouping. */
  private category_: string;
  /** Simulated duration in milliseconds returned by each trial. */
  duration_ms: number;

  constructor(name: string) {
    this.name_ = name;
    this.category_ = "test";
    this.duration_ms = 0;
  }

  /** Set the simulated duration in milliseconds for each trial. */
  with_duration(ms: number): this {
    this.duration_ms = ms;
    return this;
  }

  name(): string {
    return this.name_;
  }
  category(): string {
    return this.category_;
  }
  run(trial_id: number): Promise<TrialResult> {
    return Promise.resolve(trialSuccess(trial_id, this.duration_ms));
  }
}

/** A no-op evaluation case that always fails — useful for testing failure paths. */
export class AlwaysFailCase implements EvaluationCase {
  private name_: string;
  private category_: string;
  /** Error message returned by each trial. */
  readonly error_msg: string;

  constructor(name: string, error: string) {
    this.name_ = name;
    this.category_ = "test";
    this.error_msg = error;
  }

  name(): string {
    return this.name_;
  }
  category(): string {
    return this.category_;
  }
  run(trial_id: number): Promise<TrialResult> {
    return Promise.resolve(trialFailure(trial_id, 0, this.error_msg));
  }
}

/** A case that succeeds with a configurable probability (for testing statistics). */
export class StochasticCase implements EvaluationCase {
  private name_: string;
  /** Probability of success per trial (0.0-1.0). */
  readonly success_rate: number;

  constructor(name: string, success_rate: number) {
    this.name_ = name;
    this.success_rate = Math.max(0.0, Math.min(1.0, success_rate));
  }

  name(): string {
    return this.name_;
  }
  category(): string {
    return "stochastic";
  }
  run(trial_id: number): Promise<TrialResult> {
    // Deterministic per trial_id so tests are reproducible.
    // Uses a simple LCG hash: seed = trial_id * prime + offset (u64 wrapping),
    // mapped to [0, 1). BigInt + explicit 64-bit mask to mirror Rust.
    const MASK = (1n << 64n) - 1n;
    const tid = BigInt(trial_id);
    const mul = 6364136223846793005n;
    const add = 1442695040888963407n;
    const seed = (((tid * mul) & MASK) + add) & MASK;
    const norm = Number(seed) / Number(MASK);
    if (norm < this.success_rate) {
      return Promise.resolve(trialSuccess(trial_id, 1));
    }
    return Promise.resolve(trialFailure(trial_id, 1, "stochastic failure"));
  }
}
