/**
 * @module remote/command_queue
 *
 * Priority command queue with deadline tracking and retry logic.
 * Equivalent to Rust's `rullama-network::remote::command_queue`.
 */

import type { BackendCommand, PrioritizedCommand } from "./protocol.ts";
import { PRIORITY_ORDER } from "./protocol.ts";

const DEFAULT_QUEUE_MAX_DEPTH = 1000;

// ============================================================================
// Queue Entry
// ============================================================================

/** Entry in the priority queue. */
export class QueueEntry {
  /** The prioritized command. */
  readonly command: PrioritizedCommand;
  /** When the command was enqueued (ms since epoch). */
  readonly enqueuedAt: number;
  /** Deadline timestamp (ms since epoch), if set. */
  readonly deadline: number | undefined;
  /** Current retry attempt (0-based). */
  retryAttempt: number;
  /** Sequence number for FIFO within same priority. */
  sequence: number;

  /**
   * Wrap a command for queueing; stamps `enqueuedAt = now` and resolves
   * `command.deadline_ms` (relative) into an absolute `deadline`.
   *
   * @param command The prioritized command.
   * @param sequence Monotonic sequence number used to keep FIFO order among
   *   entries of equal priority.
   */
  constructor(command: PrioritizedCommand, sequence: number) {
    this.command = command;
    this.enqueuedAt = Date.now();
    this.deadline = command.deadline_ms !== undefined
      ? Date.now() + command.deadline_ms
      : undefined;
    this.retryAttempt = 0;
    this.sequence = sequence;
  }

  /** Check if the command has expired. */
  isExpired(): boolean {
    return this.deadline !== undefined && Date.now() > this.deadline;
  }

  /** Get time until deadline in milliseconds (undefined if no deadline or already expired). */
  timeUntilDeadline(): number | undefined {
    if (this.deadline === undefined) return undefined;
    const remaining = this.deadline - Date.now();
    return remaining > 0 ? remaining : undefined;
  }

  /** Calculate next retry delay in milliseconds, or undefined if max retries exceeded. */
  nextRetryDelay(): number | undefined {
    const policy = this.command.retry_policy;
    if (!policy || this.retryAttempt >= policy.max_attempts) return undefined;
    const delayMs = policy.initial_delay_ms *
      Math.pow(policy.backoff_multiplier, this.retryAttempt);
    return delayMs;
  }

  /** Increment retry attempt. */
  incrementRetry(): void {
    this.retryAttempt++;
  }

  /** Check if command should retry. */
  shouldRetry(): boolean {
    const policy = this.command.retry_policy;
    return policy !== undefined && this.retryAttempt < policy.max_attempts;
  }
}

// ============================================================================
// Queue Error
// ============================================================================

/** Queue errors. */
export class QueueError extends Error {
  /**
   * Create a queue error.
   *
   * @param kind `"QueueFull"` (depth limit hit by a non-critical command) or
   *   `"MaxRetriesExceeded"` (requeue attempted past the retry policy).
   * @param message Human-readable detail.
   */
  constructor(
    readonly kind: "QueueFull" | "MaxRetriesExceeded",
    message: string,
  ) {
    super(message);
    this.name = "QueueError";
  }

  /** Error thrown by {@link CommandQueue.enqueue} when the queue is at `maxDepth`. */
  static queueFull(): QueueError {
    return new QueueError("QueueFull", "Queue is full");
  }

  /** Error thrown by {@link CommandQueue.requeueForRetry} when the entry's retry policy is exhausted. */
  static maxRetriesExceeded(): QueueError {
    return new QueueError("MaxRetriesExceeded", "Maximum retries exceeded");
  }
}

// ============================================================================
// Queue Statistics
// ============================================================================

/** Queue statistics. */
export interface QueueStats {
  /** Number of entries currently queued (sum of the four priority buckets). */
  total: number;
  /** Entries with `"critical"` priority. */
  critical: number;
  /** Entries with `"high"` priority. */
  high: number;
  /** Entries with `"normal"` priority. */
  normal: number;
  /** Entries with `"low"` priority. */
  low: number;
}

// ============================================================================
// Command Queue
// ============================================================================

/**
 * Compare two queue entries for priority ordering.
 * Returns negative if `a` should come first, positive if `b` should come first.
 */
function compareEntries(a: QueueEntry, b: QueueEntry): number {
  const aPri = PRIORITY_ORDER[a.command.priority];
  const bPri = PRIORITY_ORDER[b.command.priority];
  if (aPri !== bPri) return aPri - bPri; // lower value = higher priority
  return a.sequence - b.sequence; // FIFO within same priority
}

/**
 * Priority command queue.
 *
 * Commands are dequeued in priority order (critical first, then high, normal, low).
 * Within the same priority level, commands are dequeued in FIFO order.
 */
export class CommandQueue {
  private entries: QueueEntry[] = [];
  private sequenceCounter = 0;
  private readonly maxDepth: number;

  /**
   * Create an empty queue.
   *
   * @param maxDepth Maximum number of queued entries before non-critical
   *   enqueues throw {@link QueueError.queueFull} (default 1000). Critical
   *   commands always bypass the limit.
   */
  constructor(maxDepth: number = DEFAULT_QUEUE_MAX_DEPTH) {
    this.maxDepth = maxDepth;
  }

  /** Enqueue a command with priority. */
  enqueue(command: PrioritizedCommand): void {
    if (this.entries.length >= this.maxDepth) {
      // Critical commands bypass the limit
      if (command.priority !== "critical") {
        throw QueueError.queueFull();
      }
    }

    const entry = new QueueEntry(command, this.sequenceCounter);
    this.sequenceCounter++;
    this.entries.push(entry);
    // Keep sorted by priority (insertion sort is fine for typical queue sizes)
    this.entries.sort(compareEntries);
  }

  /** Enqueue a simple command (normal priority, no retry). */
  enqueueSimple(command: BackendCommand): void {
    this.enqueue({
      command,
      priority: "normal",
    });
  }

  /** Dequeue the highest priority command. Returns undefined if empty. */
  dequeue(): QueueEntry | undefined {
    this.removeExpired();
    return this.entries.shift();
  }

  /** Peek at the highest priority command without removing it. */
  peek(): QueueEntry | undefined {
    this.removeExpired();
    return this.entries[0];
  }

  /** Get current queue depth. */
  get length(): number {
    return this.entries.length;
  }

  /** Check if queue is empty. */
  get isEmpty(): boolean {
    return this.entries.length === 0;
  }

  /** Remove expired entries. */
  private removeExpired(): void {
    this.entries = this.entries.filter((entry) => !entry.isExpired());
  }

  /** Re-enqueue a command for retry. Throws if max retries exceeded. */
  requeueForRetry(entry: QueueEntry): void {
    if (!entry.shouldRetry()) {
      throw QueueError.maxRetriesExceeded();
    }
    entry.incrementRetry();
    entry.sequence = this.sequenceCounter;
    this.sequenceCounter++;
    this.entries.push(entry);
    this.entries.sort(compareEntries);
  }

  /** Get queue statistics. */
  stats(): QueueStats {
    let critical = 0, high = 0, normal = 0, low = 0;
    for (const entry of this.entries) {
      switch (entry.command.priority) {
        case "critical":
          critical++;
          break;
        case "high":
          high++;
          break;
        case "normal":
          normal++;
          break;
        case "low":
          low++;
          break;
      }
    }
    return { total: this.entries.length, critical, high, normal, low };
  }
}
