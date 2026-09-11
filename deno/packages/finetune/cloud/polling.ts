/**
 * Exponential-backoff job poller — wait for a FineTuneProvider job to finish.
 *
 * Equivalent to Rust's `rullama_training::cloud::polling` module.
 */

import { TrainingError } from "../error.ts";
import type { TrainingJobId, TrainingJobStatus } from "../types.ts";
import { isTerminal } from "../types.ts";
import type { FineTuneProvider } from "./types.ts";

/** Tunable parameters for the poller. */
export interface JobPollerConfig {
  /** First sleep after the initial check, in ms. */
  initial_interval_ms: number;
  /** Upper bound on a single sleep, in ms. */
  max_interval_ms: number;
  /** Multiplier applied on every retry (e.g. 1.5). */
  backoff_multiplier: number;
  /** Give up after this many ms of wall-clock time. null = never. */
  timeout_ms: number | null;
  /** Optional status callback fired on every poll (useful for logging). */
  on_status?: (s: TrainingJobStatus) => void;
}

/** Sensible defaults — 15s → 5min cap, 1.5× backoff, 4-hour ceiling. */
export function defaultPollerConfig(): JobPollerConfig {
  return {
    initial_interval_ms: 15_000,
    max_interval_ms: 300_000,
    backoff_multiplier: 1.5,
    timeout_ms: 4 * 60 * 60 * 1000,
  };
}

/** Poll `provider` until `job_id` reaches a terminal state (or timeout). */
export class JobPoller {
  /** Backoff/timeout settings this poller runs with. */
  readonly config: JobPollerConfig;

  /**
   * Create a poller.
   *
   * @param config Backoff/timeout settings; defaults to {@link defaultPollerConfig}.
   */
  constructor(config: JobPollerConfig = defaultPollerConfig()) {
    this.config = config;
  }

  /**
   * Repeatedly call `provider.getJobStatus(job_id)` — invoking
   * `config.on_status` each time — sleeping between calls with exponential
   * backoff, until the status is terminal.
   *
   * @returns The terminal status (`succeeded`, `failed`, or `cancelled`).
   * @throws `TrainingError` (`timeout`) once `config.timeout_ms` of wall-clock
   *   time has elapsed without a terminal status.
   */
  async poll(
    provider: FineTuneProvider,
    job_id: TrainingJobId,
  ): Promise<TrainingJobStatus> {
    const started = Date.now();
    let interval = this.config.initial_interval_ms;

    while (true) {
      const status = await provider.getJobStatus(job_id);
      this.config.on_status?.(status);

      if (isTerminal(status)) return status;

      if (
        this.config.timeout_ms !== null &&
        Date.now() - started >= this.config.timeout_ms
      ) {
        // Returning the last non-terminal status hid the timeout from callers.
        throw TrainingError.timeout(
          `job ${job_id.value} still ${status.status} after ${this.config.timeout_ms} ms`,
        );
      }

      await new Promise((resolve) => setTimeout(resolve, interval));
      interval = Math.min(
        this.config.max_interval_ms,
        Math.floor(interval * this.config.backoff_multiplier),
      );
    }
  }
}
