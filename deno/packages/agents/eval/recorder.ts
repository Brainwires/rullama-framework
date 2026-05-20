/**
 * Tool call sequence recording and diff.
 *
 * {@link ToolSequenceRecorder} captures the ordered sequence of tool calls
 * made during an agent run. Attach it to an agent's pre-execution hook and
 * call {@link ToolSequenceRecorder.diffAgainst} at the end of a trial to
 * verify behavioural correctness.
 *
 * Equivalent to Rust's `brainwires_agents::eval::recorder` module.
 */

// ─── Tool call record ──────────────────────────────────────────────────────

/** A single recorded tool call. */
export interface ToolCallRecord {
  /** Name of the tool that was invoked. */
  name: string;
  /**
   * A short fingerprint of the tool's input arguments (first 16 hex chars of
   * a hash). Used for lightweight argument comparison without storing the
   * full payload.
   */
  args_fingerprint: string;
  /** Wall-clock timestamp of the call in milliseconds since Unix epoch. */
  timestamp_ms: number;
}

function makeToolCallRecord(name: string, args: unknown): ToolCallRecord {
  return {
    name,
    args_fingerprint: fingerprintJson(args),
    timestamp_ms: Date.now(),
  };
}

/**
 * Simple FNV-style 64-bit hash truncated to 16 hex chars. Computed over the
 * JSON string form of `v` so semantically identical inputs fingerprint equal.
 */
function fingerprintJson(v: unknown): string {
  const s = JSON.stringify(v);
  // FNV-1a 64-bit via BigInt
  const FNV_OFFSET = 0xcbf29ce484222325n;
  const FNV_PRIME = 0x100000001b3n;
  const MASK = (1n << 64n) - 1n;
  let hash = FNV_OFFSET;
  for (let i = 0; i < s.length; i++) {
    hash = (hash ^ BigInt(s.charCodeAt(i))) & MASK;
    hash = (hash * FNV_PRIME) & MASK;
  }
  return hash.toString(16).padStart(16, "0");
}

// ─── Sequence diff ─────────────────────────────────────────────────────────

/** Result of comparing the recorded tool sequence against an expected sequence. */
export interface SequenceDiff {
  /** The expected tool names (in order). */
  expected: string[];
  /** The actual tool names recorded (in order). */
  actual: string[];
  /** Edit distance between the two sequences (Levenshtein). */
  edit_distance: number;
  /**
   * Similarity in [0, 1]: `1.0 − edit_distance / max(len_expected, len_actual)`.
   * `1.0` means an exact match; `0.0` means maximally different.
   */
  similarity: number;
}

/** Compute the diff between `expected` and `actual` name sequences. */
export function computeSequenceDiff(
  expected: readonly string[],
  actual: readonly string[],
): SequenceDiff {
  const ed = levenshtein(expected, actual);
  const maxLen = Math.max(expected.length, actual.length);
  const similarity = maxLen === 0 ? 1.0 : 1.0 - ed / maxLen;
  return {
    expected: [...expected],
    actual: [...actual],
    edit_distance: ed,
    similarity,
  };
}

/** Returns true if actual exactly matches expected. */
export function isExactMatch(diff: SequenceDiff): boolean {
  return diff.edit_distance === 0;
}

/** Compute Levenshtein edit distance between two string sequences. */
export function levenshtein(
  a: readonly string[],
  b: readonly string[],
): number {
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from(
    { length: n + 1 },
    () => new Array<number>(m + 1).fill(0),
  );
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }
  return dp[n][m];
}

// ─── Recorder ──────────────────────────────────────────────────────────────

/** Recorder for tool call sequences. */
export class ToolSequenceRecorder {
  private calls_: ToolCallRecord[] = [];

  /** Record a tool call. */
  record(name: string, args: unknown): void {
    this.calls_.push(makeToolCallRecord(name, args));
  }

  /** Return a snapshot of all recorded calls in insertion order. */
  calls(): ToolCallRecord[] {
    return [...this.calls_];
  }

  /** Return only the tool names in insertion order. */
  callNames(): string[] {
    return this.calls_.map((c) => c.name);
  }

  /** Diff the recorded sequence against an expected list of tool names. */
  diffAgainst(expected: readonly string[]): SequenceDiff {
    return computeSequenceDiff(expected, this.callNames());
  }

  /** Clear all recorded calls. */
  reset(): void {
    this.calls_.length = 0;
  }

  /** Number of recorded calls. */
  get length(): number {
    return this.calls_.length;
  }

  /** True if no calls have been recorded. */
  isEmpty(): boolean {
    return this.calls_.length === 0;
  }
}

/** Factory for a fresh {@link ToolCallRecord} (exposed for tests). */
export const _toolCallRecord = makeToolCallRecord;
