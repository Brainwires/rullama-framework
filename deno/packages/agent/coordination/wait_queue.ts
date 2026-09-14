/**
 * Wait queue implementation for resource coordination.
 *
 * Manages agents waiting for locked resources with priority ordering
 * and notification when resources become available. Uses Promise +
 * EventTarget patterns for the Deno/browser runtime.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Events emitted by the wait queue. */
export type WaitQueueEvent =
  | {
    type: "registered";
    agentId: string;
    resourceKey: string;
    position: number;
    priority: number;
  }
  | {
    type: "position_changed";
    agentId: string;
    resourceKey: string;
    oldPosition: number;
    newPosition: number;
  }
  | {
    type: "ready";
    agentId: string;
    resourceKey: string;
    waitDurationMs: number;
  }
  | {
    type: "removed";
    agentId: string;
    resourceKey: string;
    reason: RemovalReason;
  }
  | {
    type: "queue_empty";
    resourceKey: string;
  };

/** Reason for removal from queue. */
export type RemovalReason =
  | "cancelled"
  | "acquired"
  | "timeout"
  | "resource_unavailable";

/** Information about a waiter in the queue. */
export interface WaiterInfo {
  /** Waiting agent. */
  agentId: string;
  /** Index in the queue (0 = next to be served). */
  position: number;
  /** Priority given at registration (lower number = served sooner). */
  priority: number;
  /** Whole seconds elapsed since registration. */
  waitingSinceSecs: number;
  /** Whether the agent asked to acquire automatically on reaching the front. */
  autoAcquire: boolean;
}

/** Status of a wait queue for a resource. */
export interface QueueStatus {
  /** Resource key the queue belongs to. */
  resourceKey: string;
  /** Number of agents waiting. */
  queueLength: number;
  /** Waiters in queue order. */
  waiters: WaiterInfo[];
  /** Mean historical wait for this resource in ms, or null with no history. */
  estimatedWaitMs: number | null;
}

/** Handle returned when registering in wait queue. */
export interface WaitQueueHandle {
  /** Promise that resolves when agent reaches front of queue. */
  ready: Promise<void>;
  /** Initial position in queue (0 = front). */
  initialPosition: number;
  /** Resource being waited for. */
  resourceKey: string;
  /** Agent ID. */
  agentId: string;
  /** Cancel waiting and remove from queue. Returns true if was removed. */
  cancel: () => boolean;
}

// ---------------------------------------------------------------------------
// Internal entry
// ---------------------------------------------------------------------------

interface WaitEntry {
  agentId: string;
  priority: number;
  registeredAt: number;
  autoAcquire: boolean;
  resolve: () => void;
}

// ---------------------------------------------------------------------------
// WaitQueue
// ---------------------------------------------------------------------------

/** Priority-ordered wait queue for resource locks. */
export class WaitQueue {
  private queues = new Map<string, WaitEntry[]>();
  private waitHistory = new Map<string, number[]>();
  private maxHistoryEntries: number;
  private eventTarget = new EventTarget();

  /**
   * Create an empty wait queue.
   * @param maxHistoryEntries Wait durations kept per resource for `estimateWait` (default 100).
   */
  constructor(maxHistoryEntries = 100) {
    this.maxHistoryEntries = maxHistoryEntries;
  }

  /** Subscribe to queue events. Returns an abort function. */
  subscribe(listener: (event: WaitQueueEvent) => void): () => void {
    const handler = (e: Event) => {
      listener((e as CustomEvent<WaitQueueEvent>).detail);
    };
    this.eventTarget.addEventListener("queue-event", handler);
    return () => this.eventTarget.removeEventListener("queue-event", handler);
  }

  /**
   * Register interest in a resource.
   *
   * Returns a handle with a promise that resolves when the agent reaches
   * the front of the queue.
   */
  register(
    resourceKey: string,
    agentId: string,
    priority: number,
    autoAcquire: boolean,
  ): WaitQueueHandle {
    let resolve!: () => void;
    const ready = new Promise<void>((r) => {
      resolve = r;
    });

    const entry: WaitEntry = {
      agentId,
      priority,
      registeredAt: Date.now(),
      autoAcquire,
      resolve,
    };

    let queue = this.queues.get(resourceKey);
    if (!queue) {
      queue = [];
      this.queues.set(resourceKey, queue);
    }

    // Find insertion position based on priority (lower number = higher priority)
    const insertPos = queue.findIndex((e) => e.priority > priority);
    const pos = insertPos === -1 ? queue.length : insertPos;
    queue.splice(pos, 0, entry);

    // Notify agents whose position changed
    for (let i = pos + 1; i < queue.length; i++) {
      this.emit({
        type: "position_changed",
        agentId: queue[i].agentId,
        resourceKey,
        oldPosition: i - 1,
        newPosition: i,
      });
    }

    this.emit({
      type: "registered",
      agentId,
      resourceKey,
      position: pos,
      priority,
    });

    return {
      ready,
      initialPosition: pos,
      resourceKey,
      agentId,
      cancel: () => this.cancel(resourceKey, agentId),
    };
  }

  /** Remove an agent from the queue. Returns true if removed. */
  cancel(resourceKey: string, agentId: string): boolean {
    const queue = this.queues.get(resourceKey);
    if (!queue) return false;

    const pos = queue.findIndex((e) => e.agentId === agentId);
    if (pos === -1) return false;

    queue.splice(pos, 1);

    // Notify agents whose position changed
    for (let i = pos; i < queue.length; i++) {
      this.emit({
        type: "position_changed",
        agentId: queue[i].agentId,
        resourceKey,
        oldPosition: i + 1,
        newPosition: i,
      });
    }

    this.emit({ type: "removed", agentId, resourceKey, reason: "cancelled" });

    if (queue.length === 0) {
      this.queues.delete(resourceKey);
      this.emit({ type: "queue_empty", resourceKey });
    }

    return true;
  }

  /**
   * Notify that a resource was released.
   * Returns the agent_id of the next waiter (if any) who should acquire.
   */
  notifyReleased(resourceKey: string): string | null {
    const queue = this.queues.get(resourceKey);
    if (!queue || queue.length === 0) return null;

    const entry = queue.shift()!;
    const waitDuration = Date.now() - entry.registeredAt;

    // Record wait time for estimation
    let times = this.waitHistory.get(resourceKey);
    if (!times) {
      times = [];
      this.waitHistory.set(resourceKey, times);
    }
    times.push(waitDuration);
    if (times.length > this.maxHistoryEntries) times.shift();

    // Notify the waiter
    entry.resolve();

    this.emit({
      type: "ready",
      agentId: entry.agentId,
      resourceKey,
      waitDurationMs: waitDuration,
    });

    // Update positions for remaining waiters
    for (let i = 0; i < queue.length; i++) {
      this.emit({
        type: "position_changed",
        agentId: queue[i].agentId,
        resourceKey,
        oldPosition: i + 1,
        newPosition: i,
      });
    }

    if (queue.length === 0) {
      this.queues.delete(resourceKey);
      this.emit({ type: "queue_empty", resourceKey });
    }

    return entry.agentId;
  }

  /** Get queue length for a resource. */
  queueLength(resourceKey: string): number {
    return this.queues.get(resourceKey)?.length ?? 0;
  }

  /** Get position of agent in queue (0 = front). */
  position(resourceKey: string, agentId: string): number | null {
    const queue = this.queues.get(resourceKey);
    if (!queue) return null;
    const pos = queue.findIndex((e) => e.agentId === agentId);
    return pos === -1 ? null : pos;
  }

  /** Estimate wait time based on historical data (ms). */
  estimateWait(resourceKey: string): number | null {
    const times = this.waitHistory.get(resourceKey);
    if (!times || times.length === 0) return null;
    return times.reduce((a, b) => a + b, 0) / times.length;
  }

  /** Estimate wait time for a specific position (ms). */
  estimateWaitAtPosition(resourceKey: string, position: number): number | null {
    const base = this.estimateWait(resourceKey);
    if (base == null) return null;
    return base * (position + 1);
  }

  /** Get detailed status of a queue. */
  getQueueStatus(resourceKey: string): QueueStatus | null {
    const queue = this.queues.get(resourceKey);
    if (!queue) return null;

    const now = Date.now();
    return {
      resourceKey,
      queueLength: queue.length,
      waiters: queue.map((e, i) => ({
        agentId: e.agentId,
        position: i,
        priority: e.priority,
        waitingSinceSecs: Math.floor((now - e.registeredAt) / 1000),
        autoAcquire: e.autoAcquire,
      })),
      estimatedWaitMs: this.estimateWait(resourceKey),
    };
  }

  /** Get all active queues. */
  listQueues(): string[] {
    return [...this.queues.keys()];
  }

  /** Check if an agent is waiting for any resource. */
  isWaiting(agentId: string): boolean {
    for (const queue of this.queues.values()) {
      if (queue.some((e) => e.agentId === agentId)) return true;
    }
    return false;
  }

  /** Get all resources an agent is waiting for. */
  waitingFor(agentId: string): string[] {
    const result: string[] = [];
    for (const [key, queue] of this.queues) {
      if (queue.some((e) => e.agentId === agentId)) result.push(key);
    }
    return result;
  }

  /** Record a completed wait time (for external tracking). */
  recordWaitTime(resourceKey: string, durationMs: number): void {
    let times = this.waitHistory.get(resourceKey);
    if (!times) {
      times = [];
      this.waitHistory.set(resourceKey, times);
    }
    times.push(durationMs);
    if (times.length > this.maxHistoryEntries) times.shift();
  }

  /** Get the next waiter without removing them (peek). */
  peekNext(resourceKey: string): WaiterInfo | null {
    const queue = this.queues.get(resourceKey);
    if (!queue || queue.length === 0) return null;
    const e = queue[0];
    return {
      agentId: e.agentId,
      position: 0,
      priority: e.priority,
      waitingSinceSecs: Math.floor((Date.now() - e.registeredAt) / 1000),
      autoAcquire: e.autoAcquire,
    };
  }

  /** Check if agent should auto-acquire (is at front and has autoAcquire set). */
  shouldAutoAcquire(resourceKey: string, agentId: string): boolean {
    const queue = this.queues.get(resourceKey);
    if (!queue || queue.length === 0) return false;
    return queue[0].agentId === agentId && queue[0].autoAcquire;
  }

  // ── Private ─────────────────────────────────────────────────────────────

  /** Dispatch `event` to subscribers as a `queue-event` CustomEvent. */
  private emit(event: WaitQueueEvent): void {
    this.eventTarget.dispatchEvent(
      new CustomEvent("queue-event", { detail: event }),
    );
  }
}

/** Generate a resource key for a given operation type and scope. */
export function resourceKey(operationType: string, scope: string): string {
  return `${operationType}:${scope}`;
}

/** Generate a resource key for a file. */
export function fileResourceKey(path: string): string {
  return `file:${path}`;
}
