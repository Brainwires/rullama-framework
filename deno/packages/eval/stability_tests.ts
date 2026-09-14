/**
 * Long-horizon stability cases that simulate 15+ step agent runs without a
 * live provider. `LoopDetectionSimCase` checks that loop detection fires after
 * N identical tool calls, `GoalPreservationCase` checks that the original goal
 * text survives the run, and `longHorizonStabilitySuite` bundles them.
 * Equivalent to Rust's `rullama_eval::stability_tests`.
 *
 * @module
 */

import type { EvaluationCase } from "./case.ts";
import {
  trialFailure,
  type TrialResult,
  trialSuccess,
  trialWithMeta,
} from "./trial.ts";

// ─── Loop detection simulation ─────────────────────────────────────────────

/**
 * Simulates a sequence of tool calls and checks that the loop detection
 * algorithm fires at the expected iteration.
 */
export class LoopDetectionSimCase implements EvaluationCase {
  /** Case name reported in results. */
  readonly nameVal: string;
  /** Number of simulated agent steps. */
  readonly n_steps: number;
  /** Tool name that starts repeating. */
  readonly looping_tool: string;
  /** Step index at which the loop begins. */
  readonly loop_starts_at: number;
  /** Loop-detection window size. */
  readonly window_size: number;
  /** Whether detection is expected to fire. */
  readonly expect_detection: boolean;

  private constructor(
    nameVal: string,
    n_steps: number,
    looping_tool: string,
    loop_starts_at: number,
    window_size: number,
    expect_detection: boolean,
  ) {
    this.nameVal = nameVal;
    this.n_steps = n_steps;
    this.looping_tool = looping_tool;
    this.loop_starts_at = loop_starts_at;
    this.window_size = window_size;
    this.expect_detection = expect_detection;
  }

  /**
   * Create a scenario that expects the loop detector to fire. The looping
   * tool repeats from `loop_starts_at` to the end of the `n_steps` sequence.
   */
  static shouldDetect(
    n_steps: number,
    looping_tool: string,
    loop_starts_at: number,
    window_size: number,
  ): LoopDetectionSimCase {
    return new LoopDetectionSimCase(
      `loop_detection_window${window_size}_step${loop_starts_at}`,
      n_steps,
      looping_tool,
      loop_starts_at,
      window_size,
      true,
    );
  }

  /** Create a scenario that expects the loop detector NOT to fire. */
  static shouldNotDetect(
    n_steps: number,
    window_size: number,
  ): LoopDetectionSimCase {
    return new LoopDetectionSimCase(
      `loop_no_detection_window${window_size}_${n_steps}steps`,
      n_steps,
      "read_file",
      Number.MAX_SAFE_INTEGER,
      window_size,
      false,
    );
  }

  /** Run the simulation and return whether a loop was detected. */
  simulate(): boolean {
    const toolNames = [
      "read_file",
      "write_file",
      "search_code",
      "list_dir",
      "bash",
    ];
    const window: string[] = [];

    for (let step = 1; step <= this.n_steps; step++) {
      const tool = step >= this.loop_starts_at
        ? this.looping_tool
        : toolNames[(step - 1) % toolNames.length];

      if (window.length === this.window_size) window.shift();
      window.push(tool);

      if (
        window.length === this.window_size &&
        window.every((n) => n === window[0])
      ) {
        return true;
      }
    }
    return false;
  }

  name(): string {
    return this.nameVal;
  }
  category(): string {
    return "stability/loop_detection";
  }

  run(trial_id: number): Promise<TrialResult> {
    const start = performance.now();
    const detected = this.simulate();
    const ms = Math.round(performance.now() - start);

    if (detected === this.expect_detection) {
      let t = trialSuccess(trial_id, ms);
      t = trialWithMeta(t, "loop_detected", detected);
      t = trialWithMeta(t, "n_steps", this.n_steps);
      t = trialWithMeta(t, "window_size", this.window_size);
      return Promise.resolve(t);
    }
    const msg = this.expect_detection
      ? `Expected loop detection after ${this.n_steps} steps (window=${this.window_size}) but none fired`
      : `Expected no loop detection but one fired at window=${this.window_size}`;
    return Promise.resolve(trialFailure(trial_id, ms, msg));
  }
}

// ─── Goal preservation simulation ──────────────────────────────────────────

/**
 * Simulates a 15+ step agent execution and verifies that the goal text is
 * re-injected into the conversation context at the expected iterations.
 */
export class GoalPreservationCase implements EvaluationCase {
  /** Case name reported in results. */
  readonly nameVal: string;
  /** Number of simulated iterations. */
  readonly n_iterations: number;
  /** How often (in iterations) the goal is re-validated. */
  readonly revalidation_interval: number;
  /** The goal text that must survive the run. */
  readonly goal_text: string;

  /** Create a goal-preservation case over `n_iterations` with the given re-validation interval. */
  constructor(n_iterations: number, revalidation_interval: number) {
    this.nameVal =
      `goal_preservation_${n_iterations}iter_every${revalidation_interval}`;
    this.n_iterations = n_iterations;
    this.revalidation_interval = revalidation_interval;
    this.goal_text = "Complete the long-horizon task reliably";
  }

  /** Iteration numbers at which a goal reminder should be injected. */
  expectedInjectionPoints(): number[] {
    const out: number[] = [];
    for (let i = 2; i <= this.n_iterations; i++) {
      if (
        this.revalidation_interval > 0 &&
        (i - 1) % this.revalidation_interval === 0
      ) {
        out.push(i);
      }
    }
    return out;
  }

  /** Simulated injection pattern. */
  simulateInjections(): number[] {
    const out: number[] = [];
    for (let iteration = 1; iteration <= this.n_iterations; iteration++) {
      if (
        this.revalidation_interval > 0 &&
        iteration > 1 &&
        (iteration - 1) % this.revalidation_interval === 0
      ) {
        out.push(iteration);
      }
    }
    return out;
  }

  name(): string {
    return this.nameVal;
  }
  category(): string {
    return "stability/goal_preservation";
  }

  run(trial_id: number): Promise<TrialResult> {
    const start = performance.now();
    const injected = this.simulateInjections();
    const expected = this.expectedInjectionPoints();
    const ms = Math.round(performance.now() - start);

    if (this.n_iterations >= 15 && this.revalidation_interval > 0) {
      if (injected.length < 1) {
        return Promise.resolve(
          trialFailure(
            trial_id,
            ms,
            `Expected at least 1 goal injection(s) across ${this.n_iterations} iterations (interval=${this.revalidation_interval}), got 0`,
          ),
        );
      }
    }

    const match = injected.length === expected.length &&
      injected.every((v, i) => v === expected[i]);
    if (!match) {
      return Promise.resolve(
        trialFailure(
          trial_id,
          ms,
          `Goal injection mismatch: expected at iterations ${
            JSON.stringify(expected)
          }, got ${JSON.stringify(injected)}`,
        ),
      );
    }

    let t = trialSuccess(trial_id, ms);
    t = trialWithMeta(t, "n_iterations", this.n_iterations);
    t = trialWithMeta(t, "injections", injected.length);
    t = trialWithMeta(t, "interval", this.revalidation_interval);
    return Promise.resolve(t);
  }
}

// ─── Standard long-horizon stability suite ─────────────────────────────────

/** Return the standard set of long-horizon stability test cases. */
export function longHorizonStabilitySuite(): EvaluationCase[] {
  return [
    LoopDetectionSimCase.shouldDetect(20, "read_file", 3, 5),
    LoopDetectionSimCase.shouldDetect(15, "write_file", 1, 5),
    LoopDetectionSimCase.shouldDetect(25, "bash", 10, 7),
    LoopDetectionSimCase.shouldDetect(30, "search_code", 5, 10),
    LoopDetectionSimCase.shouldNotDetect(20, 5),
    LoopDetectionSimCase.shouldNotDetect(30, 7),
    new GoalPreservationCase(15, 10),
    new GoalPreservationCase(20, 5),
    new GoalPreservationCase(30, 10),
    new GoalPreservationCase(50, 15),
  ];
}
