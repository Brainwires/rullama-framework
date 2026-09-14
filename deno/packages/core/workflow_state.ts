/**
 * Persistent workflow state for crash-safe agent retry.
 *
 * When an agent crashes or is killed mid-execution, naïvely re-running it
 * from scratch risks duplicating already-completed side effects (file writes,
 * API calls, database inserts). {@link WorkflowCheckpoint} records which tool
 * calls have already been executed so that a re-started agent can skip them.
 *
 * {@link FsWorkflowStateStore} persists checkpoints as JSON files under
 * `~/.rullama/workflow/<task_id>.json` using an atomic write (write to a
 * temp file, then rename).
 *
 * Equivalent to Rust's `rullama_core::workflow_state` module.
 */

import { dirname, join } from "@std/path";

// ─── Data types ────────────────────────────────────────────────────────────

/** Record of a single side effect that has been durably applied. */
export interface SideEffectRecord {
  /** The `tool_use_id` of the call that produced this side effect. */
  tool_use_id: string;
  /** Name of the tool that was called. */
  tool_name: string;
  /** Primary target of the operation (file path, URL, etc.), if applicable. */
  target: string | null;
  /** Unix timestamp (seconds) when the side effect was applied. */
  completed_at: number;
  /** Whether this side effect can be undone / is safe to retry. */
  reversible: boolean;
}

/** Create a new SideEffectRecord for a completed tool call. */
export function newSideEffectRecord(
  tool_use_id: string,
  tool_name: string,
  target: string | null,
  reversible: boolean,
): SideEffectRecord {
  return {
    tool_use_id,
    tool_name,
    target,
    completed_at: Math.floor(Date.now() / 1000),
    reversible,
  };
}

/** Snapshot of an agent's execution progress that survives process restarts. */
export interface WorkflowCheckpoint {
  /** Task the checkpoint belongs to; also the store key. */
  task_id: string;
  /** Agent executing the task. */
  agent_id: string;
  /** Number of steps completed so far. */
  step_index: number;
  /** `tool_use_id` values for calls that have already been executed. */
  completed_tool_ids: string[];
  /** Side effects applied so far, in execution order. */
  side_effects_log: SideEffectRecord[];
  /** Unix timestamp (seconds) of the last update. */
  updated_at: number;
}

/** Create a fresh checkpoint for the given task/agent pair. */
export function newCheckpoint(
  task_id: string,
  agent_id: string,
): WorkflowCheckpoint {
  return {
    task_id,
    agent_id,
    step_index: 0,
    completed_tool_ids: [],
    side_effects_log: [],
    updated_at: Math.floor(Date.now() / 1000),
  };
}

/** Return true if the given tool call has already been completed. */
export function isCompleted(
  cp: WorkflowCheckpoint,
  tool_use_id: string,
): boolean {
  return cp.completed_tool_ids.includes(tool_use_id);
}

// ─── Store interface ───────────────────────────────────────────────────────

/** Persistence backend for workflow checkpoints. */
export interface WorkflowStateStore {
  /** Persist `cp`, replacing any checkpoint stored for the same task. */
  saveCheckpoint(cp: WorkflowCheckpoint): Promise<void>;
  /** Load the checkpoint for `task_id`, or null if none exists. */
  loadCheckpoint(task_id: string): Promise<WorkflowCheckpoint | null>;
  /**
   * Record a completed tool call: add `tool_use_id` to the completed set,
   * append `effect`, and advance the step index (creating the checkpoint if needed).
   */
  markStepComplete(
    task_id: string,
    tool_use_id: string,
    effect: SideEffectRecord,
  ): Promise<void>;
  /** Remove the checkpoint for `task_id`; a no-op if none exists. */
  deleteCheckpoint(task_id: string): Promise<void>;
}

// ─── In-memory implementation ──────────────────────────────────────────────

/** In-memory workflow state store for tests — no filesystem I/O. */
export class InMemoryWorkflowStateStore implements WorkflowStateStore {
  private checkpoints = new Map<string, WorkflowCheckpoint>();

  /** Store a deep copy of `cp` under its task ID. */
  saveCheckpoint(cp: WorkflowCheckpoint): Promise<void> {
    this.checkpoints.set(cp.task_id, structuredClone(cp));
    return Promise.resolve();
  }

  /** Return a deep copy of the stored checkpoint, or null if none exists. */
  loadCheckpoint(task_id: string): Promise<WorkflowCheckpoint | null> {
    const cp = this.checkpoints.get(task_id);
    return Promise.resolve(cp ? structuredClone(cp) : null);
  }

  /**
   * Record a completed tool call in the stored checkpoint, creating one with
   * agent ID `"unknown"` if the task has none yet.
   */
  markStepComplete(
    task_id: string,
    tool_use_id: string,
    effect: SideEffectRecord,
  ): Promise<void> {
    let cp = this.checkpoints.get(task_id);
    if (!cp) {
      cp = newCheckpoint(task_id, "unknown");
      this.checkpoints.set(task_id, cp);
    }
    if (!cp.completed_tool_ids.includes(tool_use_id)) {
      cp.completed_tool_ids.push(tool_use_id);
    }
    cp.side_effects_log.push(effect);
    cp.step_index += 1;
    cp.updated_at = Math.floor(Date.now() / 1000);
    return Promise.resolve();
  }

  /** Drop the checkpoint for `task_id`, if any. */
  deleteCheckpoint(task_id: string): Promise<void> {
    this.checkpoints.delete(task_id);
    return Promise.resolve();
  }
}

// ─── Filesystem implementation ─────────────────────────────────────────────

/** Resolve the default checkpoint directory `~/.rullama/workflow/`. */
export function defaultWorkflowStatePath(): string {
  const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE");
  if (!home) {
    throw new Error("cannot determine home directory");
  }
  return join(home, ".rullama", "workflow");
}

/** Sanitise `task_id` so it's safe as a filename. */
function sanitizeTaskId(task_id: string): string {
  return [...task_id]
    .map((c) => (/[A-Za-z0-9_-]/.test(c) ? c : "_"))
    .join("");
}

/**
 * Stores workflow checkpoints as JSON files. Writes are atomic: the file is
 * written to a `.tmp` path and then renamed.
 */
export class FsWorkflowStateStore implements WorkflowStateStore {
  /** Directory the `<task_id>.json` checkpoint files live in. */
  readonly dir: string;

  /** Use `dir` for checkpoint files, creating it (recursively) if missing. */
  constructor(dir: string) {
    this.dir = dir;
    Deno.mkdirSync(dir, { recursive: true });
  }

  /** Create a store using `~/.rullama/workflow/`, creating dirs as needed. */
  static withDefaultPath(): FsWorkflowStateStore {
    return new FsWorkflowStateStore(defaultWorkflowStatePath());
  }

  /** Path of the checkpoint file for `task_id` (ID sanitised to a safe filename). */
  private checkpointPath(task_id: string): string {
    return join(this.dir, `${sanitizeTaskId(task_id)}.json`);
  }

  /** Write `cp` as pretty-printed JSON via a `.tmp` file and an atomic rename. */
  async saveCheckpoint(cp: WorkflowCheckpoint): Promise<void> {
    const path = this.checkpointPath(cp.task_id);
    const tmp = `${path}.tmp`;
    await Deno.mkdir(dirname(path), { recursive: true });
    await Deno.writeTextFile(tmp, JSON.stringify(cp, null, 2));
    await Deno.rename(tmp, path);
  }

  /** Read and parse the checkpoint file; null when the file does not exist. */
  async loadCheckpoint(task_id: string): Promise<WorkflowCheckpoint | null> {
    const path = this.checkpointPath(task_id);
    try {
      const json = await Deno.readTextFile(path);
      return JSON.parse(json) as WorkflowCheckpoint;
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return null;
      throw e;
    }
  }

  /**
   * Load (or create, with agent ID `"unknown"`) the checkpoint, record the
   * completed tool call, and write it back atomically.
   */
  async markStepComplete(
    task_id: string,
    tool_use_id: string,
    effect: SideEffectRecord,
  ): Promise<void> {
    const cp = (await this.loadCheckpoint(task_id)) ??
      newCheckpoint(task_id, "unknown");
    if (!cp.completed_tool_ids.includes(tool_use_id)) {
      cp.completed_tool_ids.push(tool_use_id);
    }
    cp.side_effects_log.push(effect);
    cp.step_index += 1;
    cp.updated_at = Math.floor(Date.now() / 1000);
    await this.saveCheckpoint(cp);
  }

  /** Delete the checkpoint file; a missing file is not an error. */
  async deleteCheckpoint(task_id: string): Promise<void> {
    const path = this.checkpointPath(task_id);
    try {
      await Deno.remove(path);
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return;
      throw e;
    }
  }
}
