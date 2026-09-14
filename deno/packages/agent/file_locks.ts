/**
 * File locking system for multi-agent coordination.
 *
 * Provides a mechanism for agents to "checkout" files, preventing concurrent
 * modifications and ensuring consistency across background task agents.
 * Uses in-memory Map (single-process).
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Lock types
// ---------------------------------------------------------------------------

/** Type of file lock. */
export type LockType = "read" | "write";

/** Information about a held lock. */
export interface LockInfo {
  /** ID of the agent holding the lock. */
  agentId: string;
  /** Type of lock. */
  lockType: LockType;
  /** When the lock was acquired (epoch ms). */
  acquiredAt: number;
  /** Optional timeout in ms for auto-release. */
  timeoutMs?: number;
}

/** Check if the lock has expired. */
export function isLockExpired(info: LockInfo): boolean {
  if (info.timeoutMs == null) return false;
  return Date.now() - info.acquiredAt > info.timeoutMs;
}

/** Get remaining time before timeout (ms), or undefined if no timeout. */
export function lockTimeRemaining(info: LockInfo): number | undefined {
  if (info.timeoutMs == null) return undefined;
  const elapsed = Date.now() - info.acquiredAt;
  return Math.max(0, info.timeoutMs - elapsed);
}

// ---------------------------------------------------------------------------
// Internal lock state
// ---------------------------------------------------------------------------

interface FileLockState {
  writeLock: LockInfo | null;
  readLocks: LockInfo[];
}

// ---------------------------------------------------------------------------
// Lock stats
// ---------------------------------------------------------------------------

/** Statistics about current locks. */
export interface LockStats {
  /** Number of files with locks. */
  totalFiles: number;
  /** Number of write locks. */
  totalWriteLocks: number;
  /** Number of read locks. */
  totalReadLocks: number;
}

// ---------------------------------------------------------------------------
// LockGuard (RAII-ish pattern via explicit release)
// ---------------------------------------------------------------------------

/** Guard that can be used to track and release a lock. */
export interface LockGuard {
  /** Agent that holds the lock this guard releases. */
  readonly agentId: string;
  /** File path the lock applies to. */
  readonly path: string;
  /** Whether the guarded lock is a `"read"` or `"write"` lock. */
  readonly lockType: LockType;
  /** Release the lock. */
  release(): void;
}

// ---------------------------------------------------------------------------
// File lock manager
// ---------------------------------------------------------------------------

const DEFAULT_LOCK_TIMEOUT_MS = 300_000; // 5 minutes

/**
 * Manages file locks across multiple agents.
 *
 * Supports read/write locks with deadlock detection, timeout-based
 * expiration, and waiting with polling.
 */
export class FileLockManager {
  private locks = new Map<string, FileLockState>();
  private defaultTimeoutMs: number | undefined;
  /** agent_id -> set of paths the agent is waiting for (for deadlock detection). */
  private waiting = new Map<string, Set<string>>();

  /**
   * Create a lock manager.
   * @param options `timeoutMs` sets the default auto-expiry applied to every
   * lock acquired without an explicit timeout (default 300 000 ms = 5 minutes);
   * `noTimeout: true` disables the default expiry entirely.
   */
  constructor(options?: { timeoutMs?: number; noTimeout?: boolean }) {
    if (options?.noTimeout) {
      this.defaultTimeoutMs = undefined;
    } else {
      this.defaultTimeoutMs = options?.timeoutMs ?? DEFAULT_LOCK_TIMEOUT_MS;
    }
  }

  // ---- Acquire ----

  /** Acquire a lock on a file. Returns a LockGuard. */
  acquireLock(
    agentId: string,
    path: string,
    lockType: LockType,
    timeoutMs?: number,
  ): LockGuard {
    const effectiveTimeout = timeoutMs ?? this.defaultTimeoutMs;
    this.cleanupExpired();

    const state = this.getOrCreateState(path);

    if (lockType === "read") {
      if (state.writeLock && state.writeLock.agentId !== agentId) {
        throw new Error(
          `File ${path} is write-locked by agent ${state.writeLock.agentId}`,
        );
      }
      state.readLocks.push({
        agentId,
        lockType: "read",
        acquiredAt: Date.now(),
        timeoutMs: effectiveTimeout,
      });
    } else {
      // Write lock
      if (state.writeLock) {
        if (state.writeLock.agentId !== agentId) {
          throw new Error(
            `File ${path} is already write-locked by agent ${state.writeLock.agentId}`,
          );
        }
        // Same agent already has write lock -- return guard
        return this.createGuard(agentId, path, lockType);
      }

      const otherReaders = state.readLocks.filter(
        (l) => l.agentId !== agentId,
      );
      if (otherReaders.length > 0) {
        const agents = otherReaders.map((l) => l.agentId);
        throw new Error(
          `File ${path} has read locks from agents: ${JSON.stringify(agents)}`,
        );
      }

      state.writeLock = {
        agentId,
        lockType: "write",
        acquiredAt: Date.now(),
        timeoutMs: effectiveTimeout,
      };
    }

    return this.createGuard(agentId, path, lockType);
  }

  /**
   * Acquire a lock with waiting and timeout.
   *
   * Polls until the lock becomes available, the wait timeout expires,
   * or a deadlock is detected.
   */
  async acquireWithWait(
    agentId: string,
    path: string,
    lockType: LockType,
    waitTimeoutMs: number,
    pollIntervalMs = 50,
  ): Promise<LockGuard> {
    const deadline = Date.now() + waitTimeoutMs;

    while (true) {
      if (this.wouldDeadlock(agentId, path)) {
        throw new Error(
          `Deadlock detected: agent ${agentId} waiting for ${path} would create circular dependency`,
        );
      }

      try {
        const guard = this.acquireLock(agentId, path, lockType);
        this.stopWaiting(agentId, path);
        return guard;
      } catch {
        if (Date.now() >= deadline) {
          this.stopWaiting(agentId, path);
          throw new Error(
            `Lock acquisition timeout after ${waitTimeoutMs}ms on ${path}`,
          );
        }
        this.startWaiting(agentId, path);
        this.cleanupExpired();
        await new Promise((r) => setTimeout(r, pollIntervalMs));
      }
    }
  }

  // ---- Release ----

  /** Release a specific lock. */
  releaseLock(agentId: string, path: string, lockType: LockType): void {
    const state = this.locks.get(path);
    if (!state) throw new Error(`No locks found for ${path}`);

    if (lockType === "read") {
      const idx = state.readLocks.findIndex((l) => l.agentId === agentId);
      if (idx === -1) {
        throw new Error(`No read lock found for agent ${agentId} on ${path}`);
      }
      state.readLocks.splice(idx, 1);
    } else {
      if (!state.writeLock) {
        throw new Error(`No write lock found on ${path}`);
      }
      if (state.writeLock.agentId !== agentId) {
        throw new Error(
          `Write lock on ${path} belongs to agent ${state.writeLock.agentId}, not ${agentId}`,
        );
      }
      state.writeLock = null;
    }

    if (!state.writeLock && state.readLocks.length === 0) {
      this.locks.delete(path);
    }
  }

  /** Release all locks held by an agent. Returns count of released locks. */
  releaseAllLocks(agentId: string): number {
    let released = 0;
    for (const [path, state] of this.locks) {
      if (state.writeLock?.agentId === agentId) {
        state.writeLock = null;
        released++;
      }
      const before = state.readLocks.length;
      state.readLocks = state.readLocks.filter((l) => l.agentId !== agentId);
      released += before - state.readLocks.length;

      if (!state.writeLock && state.readLocks.length === 0) {
        this.locks.delete(path);
      }
    }
    return released;
  }

  // ---- Query ----

  /** Check if a file is locked. Returns the most relevant lock info. */
  checkLock(path: string): LockInfo | undefined {
    const state = this.locks.get(path);
    if (!state) return undefined;
    return state.writeLock ?? state.readLocks[0] ?? undefined;
  }

  /** Check if a file is locked by a specific agent. */
  isLockedBy(path: string, agentId: string): boolean {
    const state = this.locks.get(path);
    if (!state) return false;
    if (state.writeLock?.agentId === agentId) return true;
    return state.readLocks.some((l) => l.agentId === agentId);
  }

  /** Check if a file can be locked with a specific type by an agent. */
  canAcquire(path: string, agentId: string, lockType: LockType): boolean {
    const state = this.locks.get(path);
    if (!state) return true;

    if (lockType === "read") {
      if (state.writeLock && state.writeLock.agentId !== agentId) return false;
      return true;
    }
    // Write
    if (state.writeLock && state.writeLock.agentId !== agentId) return false;
    return !state.readLocks.some((l) => l.agentId !== agentId);
  }

  /** Force release all locks on a path. */
  forceRelease(path: string): void {
    if (!this.locks.delete(path)) {
      throw new Error(`No locks found for ${path}`);
    }
  }

  /** Get all currently held locks. */
  listLocks(): Array<{ path: string; info: LockInfo }> {
    const result: Array<{ path: string; info: LockInfo }> = [];
    for (const [path, state] of this.locks) {
      if (state.writeLock) result.push({ path, info: state.writeLock });
      for (const rl of state.readLocks) result.push({ path, info: rl });
    }
    return result;
  }

  /** Get locks held by a specific agent. */
  locksForAgent(agentId: string): Array<{ path: string; info: LockInfo }> {
    const result: Array<{ path: string; info: LockInfo }> = [];
    for (const [path, state] of this.locks) {
      if (state.writeLock?.agentId === agentId) {
        result.push({ path, info: state.writeLock });
      }
      for (const rl of state.readLocks) {
        if (rl.agentId === agentId) result.push({ path, info: rl });
      }
    }
    return result;
  }

  /** Get statistics about current locks. */
  stats(): LockStats {
    let totalFiles = 0;
    let totalWriteLocks = 0;
    let totalReadLocks = 0;
    for (const state of this.locks.values()) {
      totalFiles++;
      if (state.writeLock) totalWriteLocks++;
      totalReadLocks += state.readLocks.length;
    }
    return { totalFiles, totalWriteLocks, totalReadLocks };
  }

  // ---- Expiration ----

  /** Clean up expired locks. Returns count of cleaned locks. */
  cleanupExpired(): number {
    let cleaned = 0;
    for (const [path, state] of this.locks) {
      if (state.writeLock && isLockExpired(state.writeLock)) {
        state.writeLock = null;
        cleaned++;
      }
      const before = state.readLocks.length;
      state.readLocks = state.readLocks.filter((l) => !isLockExpired(l));
      cleaned += before - state.readLocks.length;

      if (!state.writeLock && state.readLocks.length === 0) {
        this.locks.delete(path);
      }
    }
    return cleaned;
  }

  // ---- Deadlock detection ----

  /** Clear all waiting entries for an agent. */
  clearWaiting(agentId: string): void {
    this.waiting.delete(agentId);
  }

  /** Get all agents currently waiting for locks. */
  getWaitingAgents(): Map<string, string[]> {
    const result = new Map<string, string[]>();
    for (const [agentId, paths] of this.waiting) {
      result.set(agentId, [...paths]);
    }
    return result;
  }

  // ---- Private helpers ----

  /** Return the lock state for `path`, inserting an empty one if the path is untracked. */
  private getOrCreateState(path: string): FileLockState {
    let state = this.locks.get(path);
    if (!state) {
      state = { writeLock: null, readLocks: [] };
      this.locks.set(path, state);
    }
    return state;
  }

  /** Build a `LockGuard` whose `release()` calls `releaseLock` and swallows any error. */
  private createGuard(
    agentId: string,
    path: string,
    lockType: LockType,
  ): LockGuard {
    return {
      agentId,
      path,
      lockType,
      release: () => {
        try {
          this.releaseLock(agentId, path, lockType);
        } catch {
          // Ignore errors on release (lock may already be released)
        }
      },
    };
  }

  /** Record that `agentId` is blocked waiting on `path` (feeds the deadlock wait-for graph). */
  private startWaiting(agentId: string, path: string): void {
    let paths = this.waiting.get(agentId);
    if (!paths) {
      paths = new Set();
      this.waiting.set(agentId, paths);
    }
    paths.add(path);
  }

  /** Drop the wait-for entry for `agentId` on `path`, removing the agent once it waits on nothing. */
  private stopWaiting(agentId: string, path: string): void {
    const paths = this.waiting.get(agentId);
    if (paths) {
      paths.delete(path);
      if (paths.size === 0) this.waiting.delete(agentId);
    }
  }

  /**
   * Check if acquiring a lock would cause a deadlock.
   * Uses DFS cycle detection in the wait-for graph.
   */
  private wouldDeadlock(agentId: string, targetPath: string): boolean {
    const state = this.locks.get(targetPath);
    if (!state) return false;

    const holders = new Set<string>();
    if (state.writeLock) holders.add(state.writeLock.agentId);
    for (const rl of state.readLocks) holders.add(rl.agentId);

    if (holders.has(agentId)) return false; // Already hold it

    const visited = new Set<string>();
    const stack = [...holders];

    while (stack.length > 0) {
      const current = stack.pop()!;
      if (current === agentId) return true; // Cycle detected

      if (visited.has(current)) continue;
      visited.add(current);

      const waitingFor = this.waiting.get(current);
      if (!waitingFor) continue;

      for (const waitPath of waitingFor) {
        const waitState = this.locks.get(waitPath);
        if (!waitState) continue;
        if (waitState.writeLock && !visited.has(waitState.writeLock.agentId)) {
          stack.push(waitState.writeLock.agentId);
        }
        for (const rl of waitState.readLocks) {
          if (!visited.has(rl.agentId)) stack.push(rl.agentId);
        }
      }
    }

    return false;
  }
}
