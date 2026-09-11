/**
 * MDAP Voter - First-to-ahead-by-k voting system
 *
 * Implements Algorithm 2 from the MAKER paper for error correction through consensus.
 * The voting continues until one option has at least k more votes than any other option.
 *
 * Enhanced with:
 * - Early stopping (RASC paper: arxiv:2408.17017)
 * - Confidence-weighted voting (CISC paper: arxiv:2502.06233v1)
 * - Borda count alternative (Ranked voting paper: arxiv:2505.10772)
 */

import type {
  EarlyStoppingConfig,
  RedFlagResult,
  ResponseMetadata,
  SampledResponse,
  VoteResult,
  VotingMethod,
} from "./types.ts";
import { MdapError } from "./types.ts";
import { defaultEarlyStopping } from "./planner.ts";

// ---------------------------------------------------------------------------
// RedFlagValidator interface (duck-typed for flexibility)
// ---------------------------------------------------------------------------

/** Interface for red-flag validators. */
export interface RedFlagValidator {
  /** Decide whether a raw sample is acceptable; invalid results carry a reason and severity. */
  validate(response: string, metadata: ResponseMetadata): RedFlagResult;
}

// ---------------------------------------------------------------------------
// FirstToAheadByKVoter
// ---------------------------------------------------------------------------

/**
 * First-to-ahead-by-k voter implementing Algorithm 2 from the MAKER paper.
 *
 * The algorithm continues sampling until one candidate has at least k more
 * votes than any other candidate: `V[y] >= k + max(V[v] for v != y)`
 */
export class FirstToAheadByKVoter {
  private readonly _k: number;
  private readonly _maxSamples: number;
  private readonly batchSize: number;
  private readonly earlyStopping: EarlyStoppingConfig;
  private readonly _votingMethod: VotingMethod;
  private readonly useConfidenceWeights: boolean;

  /**
   * Create a voter.
   *
   * @param options.k - Required lead over the runner-up (must be >= 1, else throws).
   * @param options.maxSamples - Sample cap before `max_samples_exceeded` (clamped to >= 1).
   * @param options.batchSize - Samples drawn concurrently per round (default 4).
   * @param options.earlyStopping - RASC early-stop settings (default `defaultEarlyStopping()`).
   * @param options.votingMethod - Winner rule (default `"first_to_ahead_by_k"`).
   * @param options.useConfidenceWeights - Also accumulate confidence-weighted tallies (default false).
   */
  constructor(options: {
    k: number;
    maxSamples: number;
    batchSize?: number;
    earlyStopping?: EarlyStoppingConfig;
    votingMethod?: VotingMethod;
    useConfidenceWeights?: boolean;
  }) {
    if (options.k < 1) throw new Error("k must be >= 1");
    this._k = options.k;
    this._maxSamples = Math.max(1, options.maxSamples);
    this.batchSize = options.batchSize ?? 4;
    this.earlyStopping = options.earlyStopping ?? defaultEarlyStopping();
    this._votingMethod = options.votingMethod ?? "first_to_ahead_by_k";
    this.useConfidenceWeights = options.useConfidenceWeights ?? false;
  }

  /** Create a basic voter with k and maxSamples. */
  static create(k: number, maxSamples: number): FirstToAheadByKVoter {
    return new FirstToAheadByKVoter({ k, maxSamples });
  }

  /** Create with early stopping. */
  static withEarlyStopping(
    k: number,
    maxSamples: number,
    earlyStopping: EarlyStoppingConfig,
  ): FirstToAheadByKVoter {
    return new FirstToAheadByKVoter({ k, maxSamples, earlyStopping });
  }

  /** Create with confidence-weighted voting. */
  static withConfidenceWeighting(
    k: number,
    maxSamples: number,
  ): FirstToAheadByKVoter {
    return new FirstToAheadByKVoter({
      k,
      maxSamples,
      votingMethod: "confidence_weighted",
      useConfidenceWeights: true,
    });
  }

  /** Create with Borda count voting. */
  static withBordaCount(
    k: number,
    maxSamples: number,
  ): FirstToAheadByKVoter {
    return new FirstToAheadByKVoter({
      k,
      maxSamples,
      votingMethod: "borda_count",
      useConfidenceWeights: true,
    });
  }

  /** The required lead over the runner-up. */
  get k(): number {
    return this._k;
  }

  /** The sample cap before voting fails. */
  get maxSamples(): number {
    return this._maxSamples;
  }

  /** The winner rule this voter applies. */
  get votingMethod(): VotingMethod {
    return this._votingMethod;
  }

  /**
   * Execute voting until a winner emerges or max samples reached.
   *
   * @param sampler - A function that samples a response from the model
   * @param redFlagValidator - Validator for checking red flags
   * @param keyExtractor - Function to extract a comparable key from a value
   */
  async vote<T>(
    sampler: () => Promise<SampledResponse<T>>,
    redFlagValidator: RedFlagValidator,
    keyExtractor: (value: T) => string,
  ): Promise<VoteResult<T>> {
    const votes = new Map<string, { count: number; value: T }>();
    const weightedVotes = new Map<string, number>();
    let totalSamples = 0;
    let redFlagged = 0;
    const redFlagReasons: string[] = [];

    while (true) {
      if (totalSamples >= this._maxSamples) {
        const voteMap: Record<string, number> = {};
        for (const [k, { count }] of votes) voteMap[k] = count;
        throw new MdapError({
          type: "voting",
          details: {
            kind: "max_samples_exceeded",
            message:
              `Maximum samples exceeded: ${totalSamples} samples taken, no consensus reached`,
            samples: totalSamples,
            votes: voteMap,
          },
        });
      }

      // Sample a batch
      const remaining = this._maxSamples - totalSamples;
      const batchCount = Math.min(this.batchSize, remaining);
      const samples = await this.sampleBatch(sampler, batchCount);

      if (samples.length === 0 && totalSamples === 0) {
        throw new MdapError({
          type: "voting",
          details: {
            kind: "no_valid_responses",
            message: `No valid responses received after ${batchCount} attempts`,
            attempts: batchCount,
          },
        });
      }

      for (const sample of samples) {
        totalSamples++;

        const result = redFlagValidator.validate(
          sample.rawResponse,
          sample.metadata,
        );

        if (result.valid) {
          const key = keyExtractor(sample.value);
          const entry = votes.get(key) ?? { count: 0, value: sample.value };
          entry.count++;
          votes.set(key, entry);

          // Track weighted votes
          if (
            this.useConfidenceWeights ||
            this._votingMethod === "borda_count" ||
            this._votingMethod === "confidence_weighted"
          ) {
            weightedVotes.set(
              key,
              (weightedVotes.get(key) ?? 0) + sample.confidence,
            );
          }

          // Check early stopping
          if (this.earlyStopping.enabled) {
            const earlyWinner = this.checkEarlyStop(votes);
            if (earlyWinner) {
              return this.buildResult(
                earlyWinner.value,
                earlyWinner.key,
                votes,
                weightedVotes,
                totalSamples,
                redFlagged,
                redFlagReasons,
                true,
              );
            }
          }

          // Check winner based on voting method
          const winnerResult = this.checkWinnerByMethod(
            votes,
            weightedVotes,
          );
          if (winnerResult) {
            return this.buildResult(
              winnerResult.value,
              winnerResult.key,
              votes,
              weightedVotes,
              totalSamples,
              redFlagged,
              redFlagReasons,
              false,
            );
          }
        } else {
          redFlagged++;
          redFlagReasons.push(
            result.valid === false ? this.formatReason(result.reason) : "",
          );
        }
      }

      // Check if all samples have been red-flagged
      const validVotes = Array.from(votes.values()).reduce(
        (s, e) => s + e.count,
        0,
      );
      if (validVotes === 0 && totalSamples >= this._k * 3) {
        throw new MdapError({
          type: "voting",
          details: {
            kind: "all_samples_red_flagged",
            message:
              `All samples were red-flagged: ${redFlagged}/${totalSamples} samples invalid`,
            redFlagged,
            total: totalSamples,
          },
        });
      }

      // Loss-of-hope check
      if (
        this.earlyStopping.lossOfHopeEnabled &&
        this.checkLossOfHope(votes, totalSamples)
      ) {
        // Return current leader
        let leader: { key: string; value: T; count: number } | null = null;
        for (const [key, { count, value }] of votes) {
          if (!leader || count > leader.count) {
            leader = { key, value, count };
          }
        }
        if (leader) {
          return this.buildResult(
            leader.value,
            leader.key,
            votes,
            weightedVotes,
            totalSamples,
            redFlagged,
            redFlagReasons,
            true,
          );
        }
      }
    }
  }

  /**
   * Simple vote with default string key extraction via JSON.stringify.
   */
  // deno-lint-ignore require-await
  async voteSimple<T>(
    sampler: () => Promise<SampledResponse<T>>,
    redFlagValidator: RedFlagValidator,
  ): Promise<VoteResult<T>> {
    return this.vote(sampler, redFlagValidator, (v) => JSON.stringify(v));
  }

  // -- Private helpers --

  /** Run `count` sampler calls concurrently, dropping any that reject. */
  private async sampleBatch<T>(
    sampler: () => Promise<SampledResponse<T>>,
    count: number,
  ): Promise<SampledResponse<T>[]> {
    const promises: Promise<SampledResponse<T> | null>[] = [];
    for (let i = 0; i < count; i++) {
      promises.push(
        sampler().catch(() => null),
      );
    }
    const settled = await Promise.all(promises);
    return settled.filter((s): s is SampledResponse<T> => s !== null);
  }

  /** Dispatch the winner check to the rule selected by `votingMethod`. */
  private checkWinnerByMethod<T>(
    votes: Map<string, { count: number; value: T }>,
    weightedVotes: Map<string, number>,
  ): { key: string; value: T } | null {
    switch (this._votingMethod) {
      case "borda_count":
        return this.checkBordaWinner(votes, weightedVotes);
      case "confidence_weighted":
        return this.checkWeightedWinner(votes, weightedVotes);
      case "first_to_ahead_by_k":
      default:
        return this.checkWinner(votes);
    }
  }

  /** Original: V[y] >= k + max(V[v] for v != y) */
  private checkWinner<T>(
    votes: Map<string, { count: number; value: T }>,
  ): { key: string; value: T } | null {
    for (const [candidateKey, { count, value }] of votes) {
      let maxOther = 0;
      for (const [k, { count: c }] of votes) {
        if (k !== candidateKey && c > maxOther) maxOther = c;
      }
      if (count >= this._k + maxOther) {
        return { key: candidateKey, value };
      }
    }
    return null;
  }

  /** Borda count: winner determined by weighted confidence scores. */
  private checkBordaWinner<T>(
    votes: Map<string, { count: number; value: T }>,
    weightedVotes: Map<string, number>,
  ): { key: string; value: T } | null {
    if (votes.size === 0 || weightedVotes.size === 0) return null;

    const totalWeight = Array.from(weightedVotes.values()).reduce(
      (a, b) => a + b,
      0,
    );
    if (totalWeight < 0.001) return null;

    let leaderKey: string | null = null;
    let leaderWeight = 0;
    for (const [key, weight] of weightedVotes) {
      if (weight > leaderWeight) {
        leaderKey = key;
        leaderWeight = weight;
      }
    }
    if (!leaderKey) return null;

    let secondWeight = 0;
    for (const [key, weight] of weightedVotes) {
      if (key !== leaderKey && weight > secondWeight) secondWeight = weight;
    }

    const margin = this._k * 0.25;
    if (leaderWeight >= secondWeight + margin) {
      const entry = votes.get(leaderKey);
      if (entry) return { key: leaderKey, value: entry.value };
    }
    return null;
  }

  /** Confidence-weighted: k-ahead margin using weighted votes. */
  private checkWeightedWinner<T>(
    votes: Map<string, { count: number; value: T }>,
    weightedVotes: Map<string, number>,
  ): { key: string; value: T } | null {
    if (votes.size === 0) return null;
    if (weightedVotes.size === 0) return this.checkWinner(votes);

    for (const [candidateKey, { value }] of votes) {
      const candidateWeight = weightedVotes.get(candidateKey) ?? 0;
      let maxOtherWeight = 0;
      for (const [k, w] of weightedVotes) {
        if (k !== candidateKey && w > maxOtherWeight) maxOtherWeight = w;
      }
      const kMargin = this._k * 0.5;
      if (candidateWeight >= kMargin + maxOtherWeight) {
        return { key: candidateKey, value };
      }
    }
    return null;
  }

  /** Check early stopping (RASC paper). */
  private checkEarlyStop<T>(
    votes: Map<string, { count: number; value: T }>,
  ): { key: string; value: T } | null {
    const total = Array.from(votes.values()).reduce(
      (s, e) => s + e.count,
      0,
    );
    if (total < this.earlyStopping.minVotes) return null;

    // Get leader
    let leaderKey: string | null = null;
    let leaderCount = 0;
    let leaderValue: T | null = null;
    for (const [key, { count, value }] of votes) {
      if (count > leaderCount) {
        leaderKey = key;
        leaderCount = count;
        leaderValue = value;
      }
    }
    if (!leaderKey || !leaderValue) return null;

    const confidence = leaderCount / total;

    // 1. Simple confidence threshold
    if (confidence >= this.earlyStopping.minConfidence) {
      return { key: leaderKey, value: leaderValue };
    }

    // 2. Variance-based stopping
    if (total >= 5) {
      const variance = this.calculateVoteVariance(votes, total);
      if (
        variance < this.earlyStopping.maxVarianceThreshold &&
        confidence >= 0.6
      ) {
        return { key: leaderKey, value: leaderValue };
      }
    }

    return null;
  }

  /** Check loss-of-hope condition. */
  private checkLossOfHope<T>(
    votes: Map<string, { count: number; value: T }>,
    totalSamples: number,
  ): boolean {
    if (!this.earlyStopping.lossOfHopeEnabled) return false;

    const remaining = this._maxSamples - totalSamples;
    if (remaining <= 0) return true;

    const counts = Array.from(votes.values())
      .map((e) => e.count)
      .sort((a, b) => b - a);
    if (counts.length < 2) return false;

    const leader = counts[0];
    const runnerUp = counts[1];
    const votesNeeded = leader + this._k - runnerUp;

    return votesNeeded > remaining;
  }

  /** Calculate variance of vote distribution. */
  private calculateVoteVariance<T>(
    votes: Map<string, { count: number; value: T }>,
    total: number,
  ): number {
    if (votes.size === 0 || total === 0) return 1.0;
    const mean = total / votes.size;
    let sumSqDiff = 0;
    for (const { count } of votes.values()) {
      const diff = count - mean;
      sumSqDiff += diff * diff;
    }
    const variance = sumSqDiff / votes.size;
    return Math.sqrt(variance / (total * total));
  }

  /** `winnerVotes / totalVotes`, or 0 when there are no votes. */
  private calculateConfidence(winnerVotes: number, totalVotes: number): number {
    if (totalVotes === 0) return 0;
    return winnerVotes / totalVotes;
  }

  /** Assemble the `VoteResult`, adding `weightedConfidence` when weights were tracked. */
  private buildResult<T>(
    winner: T,
    winnerKey: string,
    votes: Map<string, { count: number; value: T }>,
    weightedVotes: Map<string, number>,
    totalSamples: number,
    redFlagged: number,
    redFlagReasons: string[],
    earlyStopped: boolean,
  ): VoteResult<T> {
    const voteDistribution: Record<string, number> = {};
    for (const [k, { count }] of votes) voteDistribution[k] = count;

    const winnerVotes = votes.get(winnerKey)?.count ?? 0;
    const totalVotes = Array.from(votes.values()).reduce(
      (s, e) => s + e.count,
      0,
    );

    let weightedConfidence: number | undefined;
    if (
      this.useConfidenceWeights ||
      this._votingMethod === "borda_count" ||
      this._votingMethod === "confidence_weighted"
    ) {
      const totalWeight = Array.from(weightedVotes.values()).reduce(
        (a, b) => a + b,
        0,
      );
      const winnerWeight = weightedVotes.get(winnerKey) ?? 0;
      weightedConfidence = winnerWeight / Math.max(0.001, totalWeight);
    }

    return {
      winner,
      winnerVotes,
      totalVotes,
      totalSamples,
      redFlaggedCount: redFlagged,
      voteDistribution,
      confidence: this.calculateConfidence(winnerVotes, totalVotes),
      redFlagReasons,
      earlyStopped,
      weightedConfidence,
      votingMethod: this._votingMethod,
    };
  }

  /** Render a `RedFlagReason` as a compact `Kind(detail)` string for `redFlagReasons`. */
  private formatReason(reason: import("./types.ts").RedFlagReason): string {
    switch (reason.kind) {
      case "response_too_long":
        return `ResponseTooLong(${reason.tokens}/${reason.limit})`;
      case "response_too_short":
        return `ResponseTooShort(${reason.length}/${reason.minimum})`;
      case "invalid_format":
        return `InvalidFormat(expected: ${reason.expected})`;
      case "self_correction_detected":
        return `SelfCorrectionDetected(${reason.pattern})`;
      case "confused_reasoning":
        return `ConfusedReasoning(${reason.pattern})`;
      case "parse_error":
        return `ParseError(${reason.message})`;
      case "empty_response":
        return "EmptyResponse";
      case "too_many_empty_lines":
        return `TooManyEmptyLines(${reason.ratio.toFixed(2)})`;
      case "invalid_json":
        return `InvalidJson(${reason.message})`;
      case "missing_field":
        return `MissingField(${reason.field})`;
      case "truncated":
        return `Truncated(${reason.reason})`;
    }
  }
}

// ---------------------------------------------------------------------------
// VoterBuilder
// ---------------------------------------------------------------------------

/** Builder for FirstToAheadByKVoter. */
export class VoterBuilder {
  private _k = 3;
  private _maxSamples = 50;
  private _batchSize = 4;
  private _earlyStopping = defaultEarlyStopping();
  private _votingMethod: VotingMethod = "first_to_ahead_by_k";
  private _useConfidenceWeights = false;

  /** Set the required lead k (default 3; clamped to >= 1 at build). */
  k(k: number): this {
    this._k = k;
    return this;
  }

  /** Set the sample cap (default 50; clamped to >= 1 at build). */
  maxSamples(maxSamples: number): this {
    this._maxSamples = maxSamples;
    return this;
  }

  /** Set the concurrent samples per round (default 4; clamped to >= 1 at build). */
  batchSize(size: number): this {
    this._batchSize = size;
    return this;
  }

  /** Set the early-stopping config (default `defaultEarlyStopping()`). */
  earlyStopping(config: EarlyStoppingConfig): this {
    this._earlyStopping = config;
    return this;
  }

  /** Set the winner rule (default `"first_to_ahead_by_k"`). */
  votingMethod(method: VotingMethod): this {
    this._votingMethod = method;
    return this;
  }

  /**
   * Toggle confidence-weighted tallies; enabling it also switches a
   * `first_to_ahead_by_k` method to `confidence_weighted`.
   */
  confidenceWeighted(enabled: boolean): this {
    this._useConfidenceWeights = enabled;
    if (enabled && this._votingMethod === "first_to_ahead_by_k") {
      this._votingMethod = "confidence_weighted";
    }
    return this;
  }

  /** Construct the voter, clamping k, maxSamples and batchSize to >= 1. */
  build(): FirstToAheadByKVoter {
    return new FirstToAheadByKVoter({
      k: Math.max(1, this._k),
      maxSamples: Math.max(1, this._maxSamples),
      batchSize: Math.max(1, this._batchSize),
      earlyStopping: this._earlyStopping,
      votingMethod: this._votingMethod,
      useConfidenceWeights: this._useConfidenceWeights,
    });
  }
}
