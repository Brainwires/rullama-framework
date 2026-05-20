/**
 * Persistent workflow state for crash-safe agent retry.
 *
 * When an agent crashes or is killed mid-execution, naïvely re-running it
 * from scratch risks duplicating already-completed side effects (file writes,
 * API calls, database inserts). {@link WorkflowCheckpoint} records which tool
 * calls have already been executed so that a re-started agent can skip them.
 *
 * {@link FsWorkflowStateStore} persists checkpoints as JSON files under
 * `~/.brainwires/workflow/<task_id>.json` using an atomic write (write to a
 * temp file, then rename).
 *
 * Equivalent to Rust's `brainwires_core::workflow_state` module.
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
  task_id: string;
  agent_id: string;
  step_index: number;
  /** `tool_use_id` values for calls that have already been executed. */
  completed_tool_ids: string[];
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
  saveCheckpoint(cp: WorkflowCheckpoint): Promise<void>;
  loadCheckpoint(task_id: string): Promise<WorkflowCheckpoint | null>;
  markStepComplete(
    task_id: string,
    tool_use_id: string,
    effect: SideEffectRecord,
  ): Promise<void>;
  deleteCheckpoint(task_id: string): Promise<void>;
}

// ─── In-memory implementation ──────────────────────────────────────────────

/** In-memory workflow state store for tests — no filesystem I/O. */
export class InMemoryWorkflowStateStore implements WorkflowStateStore {
  private checkpoints = new Map<string, WorkflowCheckpoint>();

  saveCheckpoint(cp: WorkflowCheckpoint): Promise<void> {
    this.checkpoints.set(cp.task_id, structuredClone(cp));
    return Promise.resolve();
  }

  loadCheckpoint(task_id: string): Promise<WorkflowCheckpoint | null> {
    const cp = this.checkpoints.get(task_id);
    return Promise.resolve(cp ? structuredClone(cp) : null);
  }

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

  deleteCheckpoint(task_id: string): Promise<void> {
    this.checkpoints.delete(task_id);
    return Promise.resolve();
  }
}

// ─── Filesystem implementation ─────────────────────────────────────────────

/** Resolve the default checkpoint directory `~/.brainwires/workflow/`. */
export function defaultWorkflowStatePath(): string {
  const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE");
  if (!home) {
    throw new Error("cannot determine home directory");
  }
  return join(home, ".brainwires", "workflow");
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
  readonly dir: string;

  constructor(dir: string) {
    this.dir = dir;
    Deno.mkdirSync(dir, { recursive: true });
  }

  /** Create a store using `~/.brainwires/workflow/`, creating dirs as needed. */
  static withDefaultPath(): FsWorkflowStateStore {
    return new FsWorkflowStateStore(defaultWorkflowStatePath());
  }

  private checkpointPath(task_id: string): string {
    return join(this.dir, `${sanitizeTaskId(task_id)}.json`);
  }

  async saveCheckpoint(cp: WorkflowCheckpoint): Promise<void> {
    const path = this.checkpointPath(cp.task_id);
    const tmp = `${path}.tmp`;
    await Deno.mkdir(dirname(path), { recursive: true });
    await Deno.writeTextFile(tmp, JSON.stringify(cp, null, 2));
    await Deno.rename(tmp, path);
  }

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
