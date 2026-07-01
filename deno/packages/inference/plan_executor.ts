/**
 * Plan Executor Agent - Executes plans by orchestrating task execution.
 *
 * Runs through a plan's tasks, respecting dependencies and approval modes.
 *
 * @module
 */

import type { PlanMetadata, Task } from "@rullama/core";

import type { TaskManager } from "@rullama/agent";
import { formatDurationSecs } from "@rullama/agent";

// ---------------------------------------------------------------------------
// Approval mode
// ---------------------------------------------------------------------------

/** Approval mode for plan execution. */
export type ExecutionApprovalMode = "suggest" | "auto_edit" | "full_auto";

/** Parse an approval mode string. */
export function parseExecutionApprovalMode(
  s: string,
): ExecutionApprovalMode {
  switch (s.toLowerCase()) {
    case "suggest":
      return "suggest";
    case "auto-edit":
    case "autoedit":
    case "auto_edit":
      return "auto_edit";
    case "full-auto":
    case "fullauto":
    case "full_auto":
    case "auto":
      return "full_auto";
    default:
      throw new Error(`Unknown approval mode: ${s}`);
  }
}

// ---------------------------------------------------------------------------
// Plan execution status
// ---------------------------------------------------------------------------

/** Status of plan execution. */
export type PlanExecutionStatus =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "waiting_for_approval"; task: string }
  | { kind: "paused" }
  | { kind: "completed" }
  | { kind: "failed"; error: string };

/** Format a PlanExecutionStatus as a display string. */
export function formatPlanExecutionStatus(
  status: PlanExecutionStatus,
): string {
  switch (status.kind) {
    case "idle":
      return "Idle";
    case "running":
      return "Running";
    case "waiting_for_approval":
      return `Waiting for approval: ${status.task}`;
    case "paused":
      return "Paused";
    case "completed":
      return "Completed";
    case "failed":
      return `Failed: ${status.error}`;
  }
}

// ---------------------------------------------------------------------------
// Plan execution config
// ---------------------------------------------------------------------------

/** Configuration for plan execution. */
export interface PlanExecutionConfig {
  /** Approval mode. Default: "full_auto". */
  approvalMode: ExecutionApprovalMode;
  /** Maximum iterations per task. Default: 15. */
  maxIterationsPerTask: number;
  /** Whether to auto-start next task after completion. Default: true. */
  autoAdvance: boolean;
  /** Stop on first error. Default: true. */
  stopOnError: boolean;
}

/** Create a default PlanExecutionConfig. */
export function defaultPlanExecutionConfig(): PlanExecutionConfig {
  return {
    approvalMode: "full_auto",
    maxIterationsPerTask: 15,
    autoAdvance: true,
    stopOnError: true,
  };
}

// ---------------------------------------------------------------------------
// Execution progress
// ---------------------------------------------------------------------------

/** Execution progress information. */
export interface ExecutionProgress {
  totalTasks: number;
  completedTasks: number;
  inProgressTasks: number;
  pendingTasks: number;
  blockedTasks: number;
  skippedTasks: number;
  failedTasks: number;
  totalDurationSecs: number;
  averageTaskDurationSecs?: number;
  estimatedRemainingSecs?: number;
}

// ---------------------------------------------------------------------------
// Plan executor agent
// ---------------------------------------------------------------------------

/** Plan Executor Agent - coordinates execution of a plan's tasks. */
export class PlanExecutorAgent {
  private plan: PlanMetadata;
  private taskManager: TaskManager;
  private config: PlanExecutionConfig;
  private _status: PlanExecutionStatus = { kind: "idle" };
  private currentTaskId: string | undefined;

  constructor(
    plan: PlanMetadata,
    taskManager: TaskManager,
    config?: Partial<PlanExecutionConfig>,
  ) {
    this.plan = plan;
    this.taskManager = taskManager;
    this.config = { ...defaultPlanExecutionConfig(), ...config };
  }

  /** Get the plan. */
  getPlan(): PlanMetadata {
    return this.plan;
  }

  /** Get the execution status. */
  get status(): PlanExecutionStatus {
    return this._status;
  }

  /** Get the current task ID. */
  getCurrentTaskId(): string | undefined {
    return this.currentTaskId;
  }

  /** Get the approval mode. */
  get approvalMode(): ExecutionApprovalMode {
    return this.config.approvalMode;
  }

  /** Set the approval mode. */
  set approvalMode(mode: ExecutionApprovalMode) {
    this.config.approvalMode = mode;
  }

  /** Check if a task needs approval based on current mode. */
  needsApproval(_task: Task): boolean {
    return this.config.approvalMode === "suggest";
  }

  /** Get the next task to execute. */
  getNextTask(): Task | undefined {
    const ready = this.taskManager.getReadyTasks();
    return ready[0];
  }

  /** Start executing a specific task. */
  startTask(taskId: string): void {
    const result = this.taskManager.canStart(taskId);
    if (!result.ready) {
      if (result.blockedBy) {
        throw new Error(
          `Task '${taskId}' is blocked by: ${result.blockedBy.join(", ")}`,
        );
      }
      throw new Error(`Task '${taskId}' cannot be started`);
    }

    this.taskManager.startTask(taskId);
    this.currentTaskId = taskId;
    this._status = { kind: "running" };
  }

  /** Complete the current task. Returns the next task if auto-advance. */
  completeCurrentTask(summary: string): Task | undefined {
    if (!this.currentTaskId) return undefined;

    this.taskManager.completeTask(this.currentTaskId, summary);
    this.currentTaskId = undefined;

    const stats = this.taskManager.getStats();
    if (stats.completed === stats.total) {
      this._status = { kind: "completed" };
    }

    if (this.config.autoAdvance) {
      return this.getNextTask();
    }
    return undefined;
  }

  /** Skip the current task. Returns the next task if auto-advance. */
  skipCurrentTask(reason?: string): Task | undefined {
    if (!this.currentTaskId) return undefined;

    this.taskManager.skipTask(this.currentTaskId, reason);
    this.currentTaskId = undefined;

    if (this.config.autoAdvance) {
      return this.getNextTask();
    }
    return undefined;
  }

  /** Fail the current task. */
  failCurrentTask(error: string): void {
    if (!this.currentTaskId) return;

    this.taskManager.failTask(this.currentTaskId, error);
    this.currentTaskId = undefined;

    if (this.config.stopOnError) {
      this._status = { kind: "failed", error };
    }
  }

  /** Pause execution. */
  pause(): void {
    this._status = { kind: "paused" };
  }

  /** Resume execution. Returns the current or next task. */
  resume(): Task | undefined {
    this._status = { kind: "running" };
    if (this.currentTaskId) {
      return this.taskManager.getTask(this.currentTaskId);
    }
    return this.getNextTask();
  }

  /** Request approval for a task (in Suggest mode). */
  requestApproval(task: Task): void {
    this._status = {
      kind: "waiting_for_approval",
      task: task.description,
    };
  }

  /** Approve and start a task. */
  approveAndStart(taskId: string): void {
    this.startTask(taskId);
  }

  /** Get execution progress. */
  getProgress(): ExecutionProgress {
    const stats = this.taskManager.getStats();
    const timeStats = this.taskManager.getTimeStats();
    const estimatedRemaining = this.taskManager.estimateRemainingTime();

    return {
      totalTasks: stats.total,
      completedTasks: stats.completed,
      inProgressTasks: stats.inProgress,
      pendingTasks: stats.pending,
      blockedTasks: stats.blocked,
      skippedTasks: stats.skipped,
      failedTasks: stats.failed,
      totalDurationSecs: timeStats.totalDurationSecs,
      averageTaskDurationSecs: timeStats.averageDurationSecs,
      estimatedRemainingSecs: estimatedRemaining,
    };
  }

  /** Format progress as a string. */
  formatProgress(): string {
    const progress = this.getProgress();
    const statusStr = formatPlanExecutionStatus(this._status);

    let output = `Plan Execution Status: ${statusStr}\n`;
    output +=
      `Progress: ${progress.completedTasks}/${progress.totalTasks} tasks completed\n`;

    if (progress.inProgressTasks > 0) {
      output += `  In Progress: ${progress.inProgressTasks}\n`;
    }
    if (progress.blockedTasks > 0) {
      output += `  Blocked: ${progress.blockedTasks}\n`;
    }
    if (progress.skippedTasks > 0) {
      output += `  Skipped: ${progress.skippedTasks}\n`;
    }
    if (progress.failedTasks > 0) {
      output += `  Failed: ${progress.failedTasks}\n`;
    }
    if (progress.totalDurationSecs > 0) {
      output += `Time: ${
        formatDurationSecs(progress.totalDurationSecs)
      } elapsed`;
      if (progress.estimatedRemainingSecs != null) {
        output += `, ~${
          formatDurationSecs(progress.estimatedRemainingSecs)
        } remaining`;
      }
      output += "\n";
    }

    return output;
  }
}
