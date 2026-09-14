/**
 * Per-cluster sampling-temperature optimization. `TemperatureOptimizer`
 * starts from the paper's defaults (low temperature for logical tasks, high for
 * linguistic ones), records outcomes per cluster and temperature with
 * `TemperaturePerformance`, and returns the best-scoring temperature once
 * enough samples exist.
 * Equivalent to Rust's `rullama_prompting::temperature`.
 *
 * @module
 */

import type { TaskCluster } from "./cluster.ts";

// ---------------------------------------------------------------------------
// TemperaturePerformance
// ---------------------------------------------------------------------------

/** Tracks performance metrics for a specific temperature setting. */
export interface TemperaturePerformance {
  /** Success rate (0.0-1.0) using EMA. */
  successRate: number;
  /** Average quality score (0.0-1.0) using EMA. */
  avgQuality: number;
  /** Number of samples collected. */
  sampleCount: number;
  /** Last updated timestamp (unix ms). */
  lastUpdated: number;
}

/** Create a new TemperaturePerformance with neutral defaults. */
export function createTemperaturePerformance(): TemperaturePerformance {
  return {
    successRate: 0.5,
    avgQuality: 0.5,
    sampleCount: 0,
    lastUpdated: Date.now(),
  };
}

/** Update metrics with a new outcome using EMA (alpha = 0.3). */
export function updateTemperaturePerformance(
  perf: TemperaturePerformance,
  success: boolean,
  quality: number,
): void {
  const alpha = 0.3;
  perf.successRate = alpha * (success ? 1.0 : 0.0) +
    (1.0 - alpha) * perf.successRate;
  perf.avgQuality = alpha * quality + (1.0 - alpha) * perf.avgQuality;
  perf.sampleCount += 1;
  perf.lastUpdated = Date.now();
}

/** Combined score for ranking (60% success rate, 40% quality). */
export function temperatureScore(perf: TemperaturePerformance): number {
  return 0.6 * perf.successRate + 0.4 * perf.avgQuality;
}

// ---------------------------------------------------------------------------
// TemperatureOptimizer
// ---------------------------------------------------------------------------

/** Default candidate temperatures from the paper. */
const DEFAULT_CANDIDATES: readonly number[] = [
  0.0,
  0.2,
  0.4,
  0.6,
  0.8,
  1.0,
  1.3,
];

/** Convert a temperature float to an integer key (multiply by 10). */
function tempToKey(temp: number): number {
  return Math.round(temp * 10);
}

/** Composite key for the performance map. */
function perfKey(clusterId: string, tempInt: number): string {
  return `${clusterId}:${tempInt}`;
}

/** Manages adaptive temperature selection per task cluster. */
export class TemperatureOptimizer {
  /** Maps "clusterId:tempInt" -> performance stats. */
  private readonly performanceMap = new Map<string, TemperaturePerformance>();
  /** Candidate temperatures to test. */
  private readonly candidates: readonly number[];
  /** Minimum samples before trusting a temperature setting. */
  private readonly minSamples: number;

  /** Create an optimizer; `options` set the candidate temperatures and minimum samples. */
  constructor(options?: {
    candidates?: number[];
    minSamples?: number;
  }) {
    this.candidates = options?.candidates ?? DEFAULT_CANDIDATES;
    this.minSamples = options?.minSamples ?? 5;
  }

  /**
   * Get optimal temperature for a cluster.
   *
   * Selection order:
   * 1. Local learned temperature (if enough samples)
   * 2. Default heuristic based on cluster characteristics
   */
  getOptimalTemperature(cluster: TaskCluster): number {
    const local = this.getLocalOptimal(cluster.id);
    if (local !== undefined) return local;
    return this.getDefaultTemperature(cluster);
  }

  /** Get locally learned optimal temperature, or undefined if not enough data. */
  getLocalOptimal(clusterId: string): number | undefined {
    let bestTemp: number | undefined;
    let bestScore = -Infinity;

    for (const temp of this.candidates) {
      const key = perfKey(clusterId, tempToKey(temp));
      const perf = this.performanceMap.get(key);
      if (perf && perf.sampleCount >= this.minSamples) {
        const score = temperatureScore(perf);
        if (score > bestScore) {
          bestScore = score;
          bestTemp = temp;
        }
      }
    }

    return bestTemp;
  }

  /**
   * Get default temperature based on cluster description heuristics.
   *
   * Based on the paper's findings:
   * - Logic/reasoning: 0.0
   * - Creative/linguistic: 1.3
   * - Numerical/calculation: 0.2
   * - Code/programming: 0.6
   * - Default: 0.7
   */
  getDefaultTemperature(cluster: TaskCluster): number {
    const desc = cluster.description.toLowerCase();

    if (
      desc.includes("logic") || desc.includes("boolean") ||
      desc.includes("reasoning") || desc.includes("puzzle") ||
      desc.includes("deduction")
    ) {
      return 0.0;
    }

    if (
      desc.includes("creative") || desc.includes("linguistic") ||
      desc.includes("story") || desc.includes("writing") ||
      desc.includes("generation")
    ) {
      return 1.3;
    }

    if (
      desc.includes("numerical") || desc.includes("calculation") ||
      desc.includes("math") || desc.includes("arithmetic")
    ) {
      return 0.2;
    }

    if (
      desc.includes("code") || desc.includes("programming") ||
      desc.includes("implementation") || desc.includes("algorithm")
    ) {
      return 0.6;
    }

    return 0.7;
  }

  /** Record outcome for a temperature setting. */
  recordOutcome(
    clusterId: string,
    temperature: number,
    success: boolean,
    quality: number,
  ): void {
    const key = perfKey(clusterId, tempToKey(temperature));
    let perf = this.performanceMap.get(key);
    if (!perf) {
      perf = createTemperaturePerformance();
      this.performanceMap.set(key, perf);
    }
    updateTemperaturePerformance(perf, success, quality);
  }

  /** Get performance for a specific cluster and temperature. */
  getPerformance(
    clusterId: string,
    temperature: number,
  ): TemperaturePerformance | undefined {
    return this.performanceMap.get(perfKey(clusterId, tempToKey(temperature)));
  }

  /** Get all performance entries. */
  getAllPerformance(): ReadonlyMap<string, TemperaturePerformance> {
    return this.performanceMap;
  }

  /** Get the candidate temperature list. */
  getCandidates(): readonly number[] {
    return this.candidates;
  }

  /** Get the minimum samples threshold. */
  getMinSamples(): number {
    return this.minSamples;
  }
}
