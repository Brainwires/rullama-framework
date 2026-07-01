/**
 * Priority-based task queue for agent scheduling.
 *
 * Provides a queue with priority levels (Urgent, High, Normal, Low)
 * for scheduling tasks across worker agents.
 *
 * @module
 */

import type { Task, TaskPriority, TaskStatus } from "@rullama/core";

// ---------------------------------------------------------------------------
// Queued task
// ---------------------------------------------------------------------------

/** A queued task with priority and metadata. */
export interface QueuedTask {
  /** The underlying task. */
  task: Task;
  /** Priority level. */
  priority: TaskPriority;
  /** When the task was queued (epoch ms). */
  queuedAt: number;
  /** Worker ID if assigned. */
  assignedTo?: string;
}

/** Create a new queued task. */
function createQueuedTask(task: Task, priority: TaskPriority): QueuedTask {
  return { task, priority, queuedAt: Date.now() };
}

// ---------------------------------------------------------------------------
// Priority order helper
// ---------------------------------------------------------------------------

const _PRIORITY_ORDER: Record<TaskPriority, number> = {
  urgent: 0,
  high: 1,
  normal: 2,
  low: 3,
};

// ---------------------------------------------------------------------------
// TaskQueue
// ---------------------------------------------------------------------------

/**
 * Priority queue for tasks.
 *
 * Tasks are dequeued in priority order: urgent > high > normal > low.
 * Within the same priority, FIFO order is maintained.
 */
export class TaskQueue {
  private queues: Record<TaskPriority, QueuedTask[]> = {
    urgent: [],
    high: [],
    normal: [],
    low: [],
  };
  readonly maxSize: number;

  constructor(maxSize = 100) {
    this.maxSize = maxSize;
  }

  /** Add a task to the queue. Throws if the queue is full. */
  enqueue(task: Task, priority: TaskPriority): void {
    if (this.size() >= this.maxSize) {
      throw new Error(`Task queue is full (max: ${this.maxSize})`);
    }
    this.queues[priority].push(createQueuedTask(task, priority));
  }

  /** Dequeue the highest priority task. */
  dequeue(): QueuedTask | undefined {
    for (const p of ["urgent", "high", "normal", "low"] as TaskPriority[]) {
      const item = this.queues[p].shift();
      if (item) return item;
    }
    return undefined;
  }

  /** Dequeue a task and assign it to a worker. */
  dequeueAndAssign(workerId: string): QueuedTask | undefined {
    const item = this.dequeue();
    if (item) item.assignedTo = workerId;
    return item;
  }

  /** Peek at the next task without removing it. */
  peek(): QueuedTask | undefined {
    for (const p of ["urgent", "high", "normal", "low"] as TaskPriority[]) {
      if (this.queues[p].length > 0) return this.queues[p][0];
    }
    return undefined;
  }

  /** Get the total number of tasks in the queue. */
  size(): number {
    return (
      this.queues.urgent.length +
      this.queues.high.length +
      this.queues.normal.length +
      this.queues.low.length
    );
  }

  /** Get the number of tasks at each priority level. */
  sizeByPriority(): {
    urgent: number;
    high: number;
    normal: number;
    low: number;
  } {
    return {
      urgent: this.queues.urgent.length,
      high: this.queues.high.length,
      normal: this.queues.normal.length,
      low: this.queues.low.length,
    };
  }

  /** Check if the queue is empty. */
  isEmpty(): boolean {
    return this.size() === 0;
  }

  /** Check if the queue is full. */
  isFull(): boolean {
    return this.size() >= this.maxSize;
  }

  /** Clear all tasks from the queue. */
  clear(): void {
    this.queues.urgent = [];
    this.queues.high = [];
    this.queues.normal = [];
    this.queues.low = [];
  }

  /** Get all tasks (for inspection/debugging). */
  allTasks(): QueuedTask[] {
    return [
      ...this.queues.urgent,
      ...this.queues.high,
      ...this.queues.normal,
      ...this.queues.low,
    ];
  }

  /** Find tasks by status. */
  findByStatus(status: TaskStatus): QueuedTask[] {
    return this.allTasks().filter((qt) => qt.task.status === status);
  }

  /** Remove a specific task by ID. Returns the removed task or undefined. */
  removeById(taskId: string): QueuedTask | undefined {
    for (const p of ["urgent", "high", "normal", "low"] as TaskPriority[]) {
      const idx = this.queues[p].findIndex((qt) => qt.task.id === taskId);
      if (idx !== -1) {
        return this.queues[p].splice(idx, 1)[0];
      }
    }
    return undefined;
  }
}
