/**
 * TaskManager - Hierarchical task decomposition and dependency tracking.
 *
 * Manages a tree of tasks with parent-child relationships, dependency
 * tracking, and lifecycle management (pending -> in_progress -> completed).
 *
 * @module
 */

import { Task, type TaskPriority } from "@rullama/core";

// ---------------------------------------------------------------------------
// Task node (internal)
// ---------------------------------------------------------------------------

interface TaskNode {
  task: Task;
  parentId?: string;
  childIds: string[];
  dependencyIds: string[];
  startedAt?: number;
  completedAt?: number;
  summary?: string;
  skipReason?: string;
  failureReason?: string;
}

// ---------------------------------------------------------------------------
// Task stats
// ---------------------------------------------------------------------------

/** Aggregate statistics about managed tasks. */
export interface TaskStats {
  total: number;
  pending: number;
  inProgress: number;
  completed: number;
  failed: number;
  skipped: number;
  blocked: number;
}

/** Time statistics about managed tasks. */
export interface TimeStats {
  totalDurationSecs: number;
  averageDurationSecs?: number;
}

// ---------------------------------------------------------------------------
// TaskManager
// ---------------------------------------------------------------------------

/**
 * Manages hierarchical tasks with dependency awareness.
 */
export class TaskManager {
  private tasks = new Map<string, TaskNode>();
  private nextId = 1;

  /** Create a new task. Returns the task ID. */
  createTask(
    description: string,
    parentId?: string,
    _priority?: TaskPriority,
  ): string {
    const id = `task-${this.nextId++}`;
    const task = new Task(id, description);

    if (parentId && !this.tasks.has(parentId)) {
      throw new Error(`Parent task ${parentId} does not exist`);
    }

    const node: TaskNode = {
      task,
      parentId,
      childIds: [],
      dependencyIds: [],
    };

    this.tasks.set(id, node);

    if (parentId) {
      const parent = this.tasks.get(parentId)!;
      parent.childIds.push(id);
    }

    return id;
  }

  /** Add a dependency between tasks. */
  addDependency(taskId: string, dependsOn: string): void {
    const node = this.tasks.get(taskId);
    if (!node) throw new Error(`Task ${taskId} does not exist`);
    if (!this.tasks.has(dependsOn)) {
      throw new Error(`Dependency task ${dependsOn} does not exist`);
    }
    if (!node.dependencyIds.includes(dependsOn)) {
      node.dependencyIds.push(dependsOn);
    }
  }

  /** Check if a task can start. Returns Ok(true), Ok(false), or Err(blockingTaskIds). */
  canStart(taskId: string): { ready: boolean; blockedBy?: string[] } {
    const node = this.tasks.get(taskId);
    if (!node) throw new Error(`Task ${taskId} does not exist`);

    if (
      node.task.status === "completed" ||
      node.task.status === "inprogress"
    ) {
      return { ready: false };
    }

    const blocking: string[] = [];
    for (const depId of node.dependencyIds) {
      const dep = this.tasks.get(depId);
      if (dep && dep.task.status !== "completed") {
        blocking.push(depId);
      }
    }

    if (blocking.length > 0) return { ready: false, blockedBy: blocking };
    return { ready: true };
  }

  /** Start a task. */
  startTask(taskId: string): void {
    const node = this.tasks.get(taskId);
    if (!node) throw new Error(`Task ${taskId} does not exist`);
    node.task.status = "inprogress";
    node.startedAt = Date.now();
  }

  /** Complete a task. */
  completeTask(taskId: string, summary: string): void {
    const node = this.tasks.get(taskId);
    if (!node) throw new Error(`Task ${taskId} does not exist`);
    node.task.status = "completed";
    node.completedAt = Date.now();
    node.summary = summary;
  }

  /** Fail a task. */
  failTask(taskId: string, reason: string): void {
    const node = this.tasks.get(taskId);
    if (!node) throw new Error(`Task ${taskId} does not exist`);
    node.task.status = "failed";
    node.completedAt = Date.now();
    node.failureReason = reason;
  }

  /** Skip a task. */
  skipTask(taskId: string, reason?: string): void {
    const node = this.tasks.get(taskId);
    if (!node) throw new Error(`Task ${taskId} does not exist`);
    node.task.status = "completed"; // Mark as completed so dependents can proceed
    node.completedAt = Date.now();
    node.skipReason = reason ?? "Skipped";
  }

  /** Get a task by ID. */
  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId)?.task;
  }

  /** Get all tasks. */
  getAllTasks(): Task[] {
    return [...this.tasks.values()].map((n) => n.task);
  }

  /** Get tasks that are ready to start (no blocking dependencies). */
  getReadyTasks(): Task[] {
    const ready: Task[] = [];
    for (const node of this.tasks.values()) {
      if (node.task.status !== "pending") continue;
      const result = this.canStart(node.task.id);
      if (result.ready) ready.push(node.task);
    }
    return ready;
  }

  /** Get aggregate statistics. */
  getStats(): TaskStats {
    let pending = 0;
    let inProgress = 0;
    let completed = 0;
    let failed = 0;
    let skipped = 0;
    let blocked = 0;

    for (const node of this.tasks.values()) {
      switch (node.task.status) {
        case "pending": {
          const result = this.canStart(node.task.id);
          if (result.blockedBy && result.blockedBy.length > 0) {
            blocked++;
          } else {
            pending++;
          }
          break;
        }
        case "inprogress":
          inProgress++;
          break;
        case "completed":
          if (node.skipReason) {
            skipped++;
          } else {
            completed++;
          }
          break;
        case "failed":
          failed++;
          break;
      }
    }

    return {
      total: this.tasks.size,
      pending,
      inProgress,
      completed,
      failed,
      skipped,
      blocked,
    };
  }

  /** Get time statistics. */
  getTimeStats(): TimeStats {
    let totalDuration = 0;
    let count = 0;

    for (const node of this.tasks.values()) {
      if (node.startedAt && node.completedAt) {
        totalDuration += (node.completedAt - node.startedAt) / 1000;
        count++;
      }
    }

    return {
      totalDurationSecs: Math.round(totalDuration),
      averageDurationSecs: count > 0
        ? Math.round(totalDuration / count)
        : undefined,
    };
  }

  /** Estimate remaining time based on average task duration. */
  estimateRemainingTime(): number | undefined {
    const stats = this.getStats();
    const timeStats = this.getTimeStats();

    if (!timeStats.averageDurationSecs) return undefined;

    const remaining = stats.pending + stats.inProgress + stats.blocked;
    return remaining * timeStats.averageDurationSecs;
  }
}

/** Format a duration in seconds as a human-readable string. */
export function formatDurationSecs(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}
