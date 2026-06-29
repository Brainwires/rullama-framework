/** Task status.
 * Equivalent to Rust's `TaskStatus` in rullama-core. */
export type TaskStatus =
  | "pending"
  | "inprogress"
  | "completed"
  | "failed"
  | "blocked"
  | "skipped";

/** Task priority levels.
 * Equivalent to Rust's `TaskPriority` in rullama-core. */
export type TaskPriority = "low" | "normal" | "high" | "urgent";

/** Priority numeric values for ordering. */
export const TASK_PRIORITY_VALUES: Record<TaskPriority, number> = {
  low: 0,
  normal: 1,
  high: 2,
  urgent: 3,
};

function nowTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

/** A task being executed by an agent (supports tree structure).
 * Equivalent to Rust's `Task` in rullama-core. */
export class Task {
  id: string;
  description: string;
  status: TaskStatus;
  plan_id?: string;
  parent_id?: string;
  children: string[];
  depends_on: string[];
  priority: TaskPriority;
  assigned_to?: string;
  iterations: number;
  summary?: string;
  created_at: number;
  updated_at: number;
  started_at?: number;
  completed_at?: number;

  constructor(id: string, description: string) {
    const now = nowTimestamp();
    this.id = id;
    this.description = description;
    this.status = "pending";
    this.children = [];
    this.depends_on = [];
    this.priority = "normal";
    this.iterations = 0;
    this.created_at = now;
    this.updated_at = now;
  }

  /** Create a new task associated with a plan. */
  static newForPlan(id: string, description: string, planId: string): Task {
    const task = new Task(id, description);
    task.plan_id = planId;
    return task;
  }

  /** Create a new subtask. */
  static newSubtask(id: string, description: string, parentId: string): Task {
    const task = new Task(id, description);
    task.parent_id = parentId;
    return task;
  }

  /** Mark task as in progress. */
  start(): void {
    const now = nowTimestamp();
    this.status = "inprogress";
    this.started_at = now;
    this.updated_at = now;
  }

  /** Mark task as completed. */
  complete(summary: string): void {
    const now = nowTimestamp();
    this.status = "completed";
    this.summary = summary;
    this.completed_at = now;
    this.updated_at = now;
  }

  /** Get task duration in seconds (if started and completed). */
  durationSecs(): number | undefined {
    if (this.started_at !== undefined && this.completed_at !== undefined) {
      return this.completed_at - this.started_at;
    }
    return undefined;
  }

  /** Get elapsed time since task started (in seconds). */
  elapsedSecs(): number | undefined {
    if (this.started_at !== undefined) {
      return nowTimestamp() - this.started_at;
    }
    return undefined;
  }

  /** Mark task as failed. */
  fail(error: string): void {
    this.status = "failed";
    this.summary = error;
    this.updated_at = nowTimestamp();
  }

  /** Mark task as blocked. */
  block(): void {
    this.status = "blocked";
    this.updated_at = nowTimestamp();
  }

  /** Mark task as skipped. */
  skip(reason?: string): void {
    const now = nowTimestamp();
    this.status = "skipped";
    if (reason !== undefined) this.summary = reason;
    this.completed_at = now;
    this.updated_at = now;
  }

  /** Increment iterations. */
  incrementIteration(): void {
    this.iterations += 1;
    this.updated_at = nowTimestamp();
  }

  /** Add a child task ID. */
  addChild(childId: string): void {
    if (!this.children.includes(childId)) {
      this.children.push(childId);
      this.updated_at = nowTimestamp();
    }
  }

  /** Add a dependency. */
  addDependency(taskId: string): void {
    if (!this.depends_on.includes(taskId)) {
      this.depends_on.push(taskId);
      this.updated_at = nowTimestamp();
    }
  }

  /** Check if task has any dependencies. */
  hasDependencies(): boolean {
    return this.depends_on.length > 0;
  }

  /** Check if task has children. */
  hasChildren(): boolean {
    return this.children.length > 0;
  }

  /** Check if task is a root task (no parent). */
  isRoot(): boolean {
    return this.parent_id === undefined;
  }

  /** Set priority. */
  setPriority(priority: TaskPriority): void {
    this.priority = priority;
    this.updated_at = nowTimestamp();
  }
}

/** Agent response after processing.
 * Equivalent to Rust's `AgentResponse` in rullama-core. */
export interface AgentResponse {
  message: string;
  is_complete: boolean;
  tasks: Task[];
  iterations: number;
}
