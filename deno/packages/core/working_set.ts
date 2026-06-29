/** Maximum number of files in the working set by default. */
export const DEFAULT_MAX_FILES = 15;

/** Maximum total tokens in working set by default. */
export const DEFAULT_MAX_TOKENS = 100_000;

/** A file entry in the working set.
 * Equivalent to Rust's `WorkingSetEntry` in rullama-core. */
export interface WorkingSetEntry {
  path: string;
  tokens: number;
  access_count: number;
  last_access_turn: number;
  added_at_turn: number;
  pinned: boolean;
  label?: string;
}

/** Working set configuration.
 * Equivalent to Rust's `WorkingSetConfig` in rullama-core. */
export interface WorkingSetConfig {
  max_files: number;
  max_tokens: number;
  stale_after_turns: number;
  auto_evict: boolean;
}

/** Default WorkingSetConfig. */
export function defaultWorkingSetConfig(): WorkingSetConfig {
  return {
    max_files: DEFAULT_MAX_FILES,
    max_tokens: DEFAULT_MAX_TOKENS,
    stale_after_turns: 10,
    auto_evict: true,
  };
}

/** Manages the set of files currently in the agent's context.
 * Equivalent to Rust's `WorkingSet` in rullama-core. */
export class WorkingSet {
  private entries: Map<string, WorkingSetEntry> = new Map();
  private config: WorkingSetConfig;
  private _currentTurn = 0;
  private _lastEviction?: string;

  constructor(config?: WorkingSetConfig) {
    this.config = config ?? defaultWorkingSetConfig();
  }

  /** Advance to the next turn, triggering stale eviction if enabled. */
  nextTurn(): void {
    this._currentTurn += 1;
    if (this.config.auto_evict) this.evictStale();
  }

  /** Returns the current turn number. */
  get currentTurn(): number {
    return this._currentTurn;
  }

  /** Add a file to the working set, evicting LRU entries if needed. */
  add(path: string, tokens: number): string | undefined {
    if (this.entries.has(path)) {
      const entry = this.entries.get(path)!;
      entry.access_count += 1;
      entry.last_access_turn = this._currentTurn;
      return undefined;
    }
    const evictionReason = this.maybeEvict(tokens);
    this.entries.set(path, {
      path,
      tokens,
      access_count: 1,
      last_access_turn: this._currentTurn,
      added_at_turn: this._currentTurn,
      pinned: false,
    });
    return evictionReason;
  }

  /** Add a file with a label. */
  addLabeled(path: string, tokens: number, label: string): string | undefined {
    if (this.entries.has(path)) {
      const entry = this.entries.get(path)!;
      entry.access_count += 1;
      entry.last_access_turn = this._currentTurn;
      entry.label = label;
      return undefined;
    }
    const evictionReason = this.maybeEvict(tokens);
    this.entries.set(path, {
      path,
      tokens,
      access_count: 1,
      last_access_turn: this._currentTurn,
      added_at_turn: this._currentTurn,
      pinned: false,
      label,
    });
    return evictionReason;
  }

  /** Add a pinned file immune to eviction. */
  addPinned(path: string, tokens: number, label?: string): void {
    if (this.entries.has(path)) {
      const entry = this.entries.get(path)!;
      entry.pinned = true;
      entry.access_count += 1;
      entry.last_access_turn = this._currentTurn;
      if (label !== undefined) entry.label = label;
      return;
    }
    this.entries.set(path, {
      path,
      tokens,
      access_count: 1,
      last_access_turn: this._currentTurn,
      added_at_turn: this._currentTurn,
      pinned: true,
      label,
    });
  }

  /** Touch a file to update its access count and turn. */
  touch(path: string): boolean {
    const entry = this.entries.get(path);
    if (entry) {
      entry.access_count += 1;
      entry.last_access_turn = this._currentTurn;
      return true;
    }
    return false;
  }

  /** Remove a file from the working set. */
  remove(path: string): boolean {
    return this.entries.delete(path);
  }

  /** Pin a file to prevent eviction. */
  pin(path: string): boolean {
    const entry = this.entries.get(path);
    if (entry) {
      entry.pinned = true;
      return true;
    }
    return false;
  }

  /** Unpin a file. */
  unpin(path: string): boolean {
    const entry = this.entries.get(path);
    if (entry) {
      entry.pinned = false;
      return true;
    }
    return false;
  }

  /** Clear the working set. */
  clear(keepPinned = false): void {
    if (keepPinned) {
      for (const [key, entry] of this.entries) {
        if (!entry.pinned) this.entries.delete(key);
      }
    } else {
      this.entries.clear();
    }
    this._lastEviction = undefined;
  }

  /** Iterate over all entries. */
  allEntries(): WorkingSetEntry[] {
    return [...this.entries.values()];
  }

  /** Get an entry by path. */
  get(path: string): WorkingSetEntry | undefined {
    return this.entries.get(path);
  }

  /** Check if a path is in the working set. */
  contains(path: string): boolean {
    return this.entries.has(path);
  }

  /** Returns the number of entries. */
  get length(): number {
    return this.entries.size;
  }

  /** Returns true if the working set is empty. */
  isEmpty(): boolean {
    return this.entries.size === 0;
  }

  /** Returns the total estimated token count. */
  totalTokens(): number {
    let sum = 0;
    for (const e of this.entries.values()) sum += e.tokens;
    return sum;
  }

  /** Returns the last eviction message. */
  get lastEviction(): string | undefined {
    return this._lastEviction;
  }

  /** Returns all file paths. */
  filePaths(): string[] {
    return [...this.entries.values()].map((e) => e.path);
  }

  private evictStale(): void {
    const threshold = Math.max(
      0,
      this._currentTurn - this.config.stale_after_turns,
    );
    const before = this.entries.size;
    for (const [key, entry] of this.entries) {
      if (!entry.pinned && entry.last_access_turn <= threshold) {
        this.entries.delete(key);
      }
    }
    const evicted = before - this.entries.size;
    if (evicted > 0) {
      this._lastEviction = `Evicted ${evicted} stale file(s)`;
    }
  }

  private maybeEvict(newTokens: number): string | undefined {
    const evictedFiles: string[] = [];
    while (this.entries.size >= this.config.max_files) {
      const key = this.findLruCandidate();
      if (!key) break;
      const entry = this.entries.get(key);
      if (entry) evictedFiles.push(entry.path);
      this.entries.delete(key!);
    }
    while (this.totalTokens() + newTokens > this.config.max_tokens) {
      const key = this.findLruCandidate();
      if (!key) break;
      const entry = this.entries.get(key);
      if (entry) evictedFiles.push(entry.path);
      this.entries.delete(key!);
    }
    if (evictedFiles.length === 0) return undefined;
    const reason = `Evicted: ${evictedFiles.join(", ")}`;
    this._lastEviction = reason;
    return reason;
  }

  private findLruCandidate(): string | undefined {
    let bestKey: string | undefined;
    let bestTurn = Infinity;
    let bestCount = Infinity;
    for (const [key, entry] of this.entries) {
      if (entry.pinned) continue;
      if (
        entry.last_access_turn < bestTurn ||
        (entry.last_access_turn === bestTurn && entry.access_count < bestCount)
      ) {
        bestKey = key;
        bestTurn = entry.last_access_turn;
        bestCount = entry.access_count;
      }
    }
    return bestKey;
  }
}

/** Estimate tokens for a string (rough: ~4 chars per token). */
export function estimateTokens(content: string): number {
  return Math.ceil(content.length / 4);
}

/** Estimate tokens for a file by size. */
export function estimateTokensFromSize(bytes: number): number {
  return Math.ceil(bytes / 4);
}
