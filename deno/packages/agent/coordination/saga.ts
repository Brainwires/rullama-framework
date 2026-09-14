/**
 * Saga-Style Compensating Transactions.
 *
 * Implements the Saga pattern for multi-step operations. When an operation
 * fails mid-way, compensation actions are executed in reverse order to undo
 * completed sub-operations.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Operation types
// ---------------------------------------------------------------------------

/** Types of saga operations for categorization. */
export type SagaOperationType =
  | "file_write"
  | "file_edit"
  | "file_delete"
  | "git_stage"
  | "git_unstage"
  | "git_commit"
  | "git_branch_create"
  | "git_branch_delete"
  | "build"
  | "test"
  | "generic";

/** Returns true if this operation type can be compensated. */
export function isCompensable(opType: SagaOperationType): boolean {
  switch (opType) {
    case "file_write":
    case "file_edit":
    case "file_delete":
    case "git_stage":
    case "git_unstage":
    case "git_commit":
    case "git_branch_create":
    case "git_branch_delete":
      return true;
    case "build":
    case "test":
    case "generic":
      return false;
  }
}

// ---------------------------------------------------------------------------
// Operation result
// ---------------------------------------------------------------------------

/** Result of an operation, needed for compensation. */
export interface OperationResult {
  /** Unique identifier for this operation. */
  operationId: string;
  /** Whether the operation succeeded. */
  success: boolean;
  /** State captured before operation (for rollback). */
  checkpoint?: Checkpoint;
  /** Metadata needed for compensation. */
  compensationData: unknown;
  /** Output from the operation. */
  output?: string;
}

/** Create a successful operation result. */
export function successResult(
  operationId: string,
  compensationData: unknown = null,
): OperationResult {
  return { operationId, success: true, compensationData };
}

/** Create a failed operation result. */
export function failureResult(operationId: string): OperationResult {
  return { operationId, success: false, compensationData: null };
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/** Snapshot of a file's state for restoration. */
export interface FileState {
  /** Path of the captured file. */
  path: string;
  /** Hash of the file content at capture time. */
  contentHash: string;
  /** Full original content, when it was captured for restoration. */
  originalContent?: string;
}

/** Snapshot of git state for restoration. */
export interface GitCheckpoint {
  /** Commit hash HEAD pointed at when the checkpoint was taken. */
  headCommit: string;
  /** Paths that were staged in the index. */
  stagedFiles: string[];
  /** Branch that was checked out. */
  branch: string;
}

/** Checkpoint for state restoration. */
export interface Checkpoint {
  /** Caller-supplied checkpoint identifier. */
  id: string;
  /** When the checkpoint was created (epoch ms). */
  timestamp: number;
  /** File snapshots captured in this checkpoint (empty from `createCheckpoint`). */
  fileStates: FileState[];
  /** Git snapshot, if the checkpoint covers repository state. */
  gitState?: GitCheckpoint;
}

/** Create a new checkpoint. */
export function createCheckpoint(id: string): Checkpoint {
  return { id, timestamp: Date.now(), fileStates: [] };
}

// ---------------------------------------------------------------------------
// Compensable operation interface
// ---------------------------------------------------------------------------

/** A compensable operation that can be undone. */
export interface CompensableOperation {
  /** Execute the forward operation. */
  execute(): Promise<OperationResult>;
  /** Compensate (undo) the operation. */
  compensate(result: OperationResult): Promise<void>;
  /** Get operation description for logging. */
  description(): string;
  /** Get the operation type. */
  operationType(): SagaOperationType;
}

// ---------------------------------------------------------------------------
// Saga status
// ---------------------------------------------------------------------------

/** Current status of a saga execution. */
export type SagaStatus =
  | "running"
  | "completed"
  | "failed"
  | "compensating"
  | "compensated";

// ---------------------------------------------------------------------------
// Compensation report
// ---------------------------------------------------------------------------

/** Outcome of a compensation attempt. */
export type CompensationOutcome = "success" | "failed" | "skipped";

/** Status of a single compensation action. */
export interface CompensationStatus {
  /** The operation's `description()` at the time it was compensated. */
  description: string;
  /** Whether the compensation ran, threw, or was skipped. */
  status: CompensationOutcome;
  /** Stringified error for `"failed"`, or the skip reason for `"skipped"`. */
  error?: string;
}

/** Report of compensation actions. */
export class CompensationReport {
  /** ID of the saga that was compensated. */
  readonly sagaId: string;
  /** One entry per compensated operation, in the order they were undone (reverse of execution). */
  readonly operations: CompensationStatus[] = [];
  /** When the report was created (epoch ms). */
  readonly startedAt: number;
  /** Set by `markCompleted` (epoch ms); undefined while compensation is in flight. */
  completedAt?: number;

  /** Start an empty report for `sagaId`, stamping `startedAt` with the current time. */
  constructor(sagaId: string) {
    this.sagaId = sagaId;
    this.startedAt = Date.now();
  }

  /** Append a `"success"` entry for an operation whose `compensate()` resolved. */
  addSuccess(description: string): void {
    this.operations.push({ description, status: "success" });
  }

  /** Append a `"failed"` entry with the error text from a rejected `compensate()`. */
  addFailure(description: string, error: string): void {
    this.operations.push({ description, status: "failed", error });
  }

  /** Append a `"skipped"` entry; `reason` is stored in the `error` field. */
  addSkipped(description: string, reason: string): void {
    this.operations.push({ description, status: "skipped", error: reason });
  }

  /** Returns true if all compensations succeeded or were skipped. */
  allSuccessful(): boolean {
    return this.operations.every(
      (s) => s.status === "success" || s.status === "skipped",
    );
  }

  /** Generate a human-readable summary. */
  summary(): string {
    const successful = this.operations.filter(
      (s) => s.status === "success",
    ).length;
    const failed = this.operations.filter(
      (s) => s.status === "failed",
    ).length;
    const skipped = this.operations.filter(
      (s) => s.status === "skipped",
    ).length;
    return `${successful} successful, ${failed} failed, ${skipped} skipped (total: ${this.operations.length})`;
  }

  /** Stamp `completedAt` with the current time. */
  markCompleted(): void {
    this.completedAt = Date.now();
  }
}

// ---------------------------------------------------------------------------
// Saga executor
// ---------------------------------------------------------------------------

/** Saga executor that manages compensating transactions. */
export class SagaExecutor {
  /** Generated as `saga-<agentId>-<startedAt>`. */
  readonly sagaId: string;
  /** Agent that owns this saga. */
  readonly agentId: string;
  /** Human-readable description supplied at construction. */
  readonly description: string;
  /** When the saga was created (epoch ms). */
  readonly startedAt: number;
  private completedOps: Array<{
    op: CompensableOperation;
    result: OperationResult;
  }> = [];
  private _status: SagaStatus = "running";
  private compensationHooks: Array<
    (summary: string, allSuccessful: boolean) => void
  > = [];

  /** Create a saga in the `"running"` state with no completed operations. */
  constructor(agentId: string, description: string) {
    this.agentId = agentId;
    this.description = description;
    this.startedAt = Date.now();
    this.sagaId = `saga-${agentId}-${Date.now()}`;
  }

  /** Get current status. */
  get status(): SagaStatus {
    return this._status;
  }

  /** Get number of completed operations. */
  operationCount(): number {
    return this.completedOps.length;
  }

  /** Execute an operation within the saga. */
  async executeStep(op: CompensableOperation): Promise<OperationResult> {
    if (this._status !== "running") {
      throw new Error("Cannot execute step: saga is not running");
    }

    const result = await op.execute();

    if (result.success) {
      this.completedOps.push({ op, result });
    } else {
      this._status = "failed";
    }

    return result;
  }

  /** Mark the saga as successfully completed. */
  complete(): void {
    this._status = "completed";
  }

  /** Mark the saga as failed. */
  fail(): void {
    this._status = "failed";
  }

  /** Compensate all completed operations in reverse order. */
  async compensateAll(): Promise<CompensationReport> {
    this._status = "compensating";
    const report = new CompensationReport(this.sagaId);

    while (this.completedOps.length > 0) {
      const { op, result } = this.completedOps.pop()!;

      if (!isCompensable(op.operationType())) {
        report.addSkipped(
          op.description(),
          "Non-compensable operation type",
        );
        continue;
      }

      try {
        await op.compensate(result);
        report.addSuccess(op.description());
      } catch (e) {
        report.addFailure(op.description(), String(e));
        // Continue compensating even if one fails
      }
    }

    this._status = "compensated";

    const summaryText = report.summary();
    const allOk = report.allSuccessful();
    for (const hook of this.compensationHooks) {
      try {
        hook(summaryText, allOk);
      } catch { /* ignore */ }
    }

    return report;
  }

  /** Add a hook called after compensation. */
  onCompensation(
    hook: (summary: string, allSuccessful: boolean) => void,
  ): void {
    this.compensationHooks.push(hook);
  }

  /** Get descriptions of all completed operations. */
  getOperationDescriptions(): string[] {
    return this.completedOps.map(({ op }) => op.description());
  }
}
