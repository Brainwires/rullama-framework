/**
 * Evaluation trial results and statistical analysis.
 *
 * A *trial* is one execution of an {@link EvaluationCase}. Run N trials and
 * summarise with {@link evaluationStatsFromTrials} which reports the success
 * rate together with a Wilson-score 95% confidence interval.
 *
 * Equivalent to Rust's `brainwires_agents::eval::trial` module.
 */

// ─── Trial result ──────────────────────────────────────────────────────────

/** Result produced by a single trial run. */
export interface TrialResult {
  /** Sequential index of this trial (0-based). */
  trial_id: number;
  /** Whether the trial succeeded. */
  success: boolean;
  /** Wall-clock duration of the trial in milliseconds. */
  duration_ms: number;
  /** Error message when `success === false`. */
  error: string | null;
  /**
   * Arbitrary key-value metadata emitted by the case (e.g. iteration count,
   * token usage, tool names used).
   */
  metadata: Record<string, unknown>;
}

/** Create a successful trial result. */
export function trialSuccess(
  trial_id: number,
  duration_ms: number,
): TrialResult {
  return {
    trial_id,
    success: true,
    duration_ms,
    error: null,
    metadata: {},
  };
}

/** Create a failed trial result. */
export function trialFailure(
  trial_id: number,
  duration_ms: number,
  error: string,
): TrialResult {
  return {
    trial_id,
    success: false,
    duration_ms,
    error,
    metadata: {},
  };
}

/** Attach an arbitrary metadata value (returns a new {@link TrialResult}). */
export function trialWithMeta(
  trial: TrialResult,
  key: string,
  value: unknown,
): TrialResult {
  return {
    ...trial,
    metadata: { ...trial.metadata, [key]: value },
  };
}

// ─── Confidence interval ───────────────────────────────────────────────────

/** A symmetric 95% confidence interval around a proportion. */
export interface ConfidenceInterval95 {
  /** Lower bound (clipped to 0). */
  lower: number;
  /** Upper bound (clipped to 1). */
  upper: number;
}

/**
 * Compute a Wilson-score 95% confidence interval.
 *
 * The Wilson interval is preferred over the naïve Wald interval because it
 * behaves well at the extremes (p = 0 or p = 1) and for small N.
 *
 * Formula: `(p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`
 * where `z = 1.96` for 95% confidence.
 */
export function wilsonInterval(
  successes: number,
  n: number,
): ConfidenceInterval95 {
  if (n === 0) return { lower: 0.0, upper: 1.0 };

  const Z = 1.96; // 95% two-tailed
  const p = successes / n;
  const nf = n;
  const z2 = Z * Z;

  const centre = p + z2 / (2.0 * nf);
  const margin = Z * Math.sqrt(p * (1.0 - p) / nf + z2 / (4.0 * nf * nf));
  const denom = 1.0 + z2 / nf;

  const clamp = (x: number) => Math.max(0.0, Math.min(1.0, x));
  return {
    lower: clamp((centre - margin) / denom),
    upper: clamp((centre + margin) / denom),
  };
}

// ─── Summary statistics ────────────────────────────────────────────────────

/** Aggregate statistics for a set of {@link TrialResult}s from the same case. */
export interface EvaluationStats {
  /** Total number of trials executed. */
  n_trials: number;
  /** Number of trials that succeeded. */
  successes: number;
  /** `successes / n_trials` (0.0 when n_trials == 0). */
  success_rate: number;
  /** Wilson-score 95% confidence interval around `success_rate`. */
  confidence_interval_95: ConfidenceInterval95;
  /** Mean trial duration across all trials in milliseconds. */
  mean_duration_ms: number;
  /** Median (P50) trial duration in milliseconds. */
  p50_duration_ms: number;
  /** 95th-percentile trial duration in milliseconds. */
  p95_duration_ms: number;
}

/**
 * Compute statistics from an array of trial results.
 *
 * Returns `null` if `results` is empty.
 */
export function evaluationStatsFromTrials(
  results: readonly TrialResult[],
): EvaluationStats | null {
  const n = results.length;
  if (n === 0) return null;

  const successes = results.filter((r) => r.success).length;
  const success_rate = successes / n;
  const ci = wilsonInterval(successes, n);

  const durations: number[] = results.map((r) => r.duration_ms);
  durations.sort((a, b) => a - b);

  const mean_duration_ms = durations.reduce((s, d) => s + d, 0) / n;
  const p50_duration_ms = percentile(durations, 50.0);
  const p95_duration_ms = percentile(durations, 95.0);

  return {
    n_trials: n,
    successes,
    success_rate,
    confidence_interval_95: ci,
    mean_duration_ms,
    p50_duration_ms,
    p95_duration_ms,
  };
}

/** Compute the p-th percentile of a sorted array (linear interpolation). */
export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0.0;
  if (sorted.length === 1) return sorted[0];
  const rank = p / 100.0 * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  const frac = rank - lower;
  return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}
