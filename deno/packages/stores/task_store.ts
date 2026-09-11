/**
 * Task Store -- persists tasks and agent state via a backend-agnostic storage layer.
 *
 * Equivalent to Rust's `stores/task_store.rs` in rullama-storage.
 * @module
 */

import type { Task, TaskPriority, TaskStatus } from "@rullama/core";
import type { StorageBackend } from "@rullama/storage";
import {
  type FieldDef,
  FieldTypes,
  fieldValueAsI32,
  fieldValueAsI64,
  fieldValueAsStr,
  FieldValues,
  Filters,
  optionalField,
  type Record,
  recordGet,
  requiredField,
} from "@rullama/storage";

const TASK_TABLE = "tasks";
const AGENT_STATE_TABLE = "agent_states";

// -- TaskMetadata -----------------------------------------------------------

/** Metadata for storing tasks (flat serialization form). */
export interface TaskMetadata {
  /** Unique task identifier (`Task.id`); the key `get` and `delete` use. */
  taskId: string;
  /** Id of the conversation the task belongs to. */
  conversationId: string;
  /** Id of the plan the task was created from, if any. */
  planId?: string;
  /** What the task is meant to accomplish. */
  description: string;
  /**
   * Lifecycle state as a plain string, one of `TaskStatus`: `"pending"`,
   * `"inprogress"`, `"completed"`, `"failed"`, `"blocked"` or `"skipped"`.
   * An empty string reads back as `"pending"` via {@link metadataToTask}.
   */
  status: string;
  /** Id of the parent task in the task tree, if this is a subtask. */
  parentId?: string;
  /** JSON-encoded array of child task ids (e.g. `"[]"`). */
  children: string;
  /** JSON-encoded array of ids of tasks that must finish before this one may start. */
  dependsOn: string;
  /**
   * Priority as a plain string, one of `TaskPriority`: `"low"`, `"normal"`,
   * `"high"` or `"urgent"`. An empty string reads back as `"normal"`.
   */
  priority: string;
  /** Id of the agent the task is assigned to, if any. */
  assignedTo?: string;
  /** Number of agent iterations spent on the task so far. */
  iterations: number;
  /** Summary of the outcome, once one has been written. */
  summary?: string;
  /** Creation time as a Unix timestamp in seconds. */
  createdAt: number;
  /** Last-modification time as a Unix timestamp in seconds. */
  updatedAt: number;
  /** Time work on the task began, as a Unix timestamp in seconds, if it has started. */
  startedAt?: number;
  /** Time the task reached a terminal state, as a Unix timestamp in seconds, if it has. */
  completedAt?: number;
}

/** Convert a Task to TaskMetadata. */
export function taskToMetadata(
  task: Task,
  conversationId: string,
): TaskMetadata {
  return {
    taskId: task.id,
    conversationId,
    planId: task.plan_id,
    description: task.description,
    status: task.status,
    parentId: task.parent_id,
    children: JSON.stringify(task.children),
    dependsOn: JSON.stringify(task.depends_on),
    priority: task.priority,
    assignedTo: task.assigned_to,
    iterations: task.iterations,
    summary: task.summary,
    createdAt: task.created_at,
    updatedAt: task.updated_at,
    startedAt: task.started_at,
    completedAt: task.completed_at,
  };
}

/** Convert TaskMetadata back to a Task-like object. */
export function metadataToTask(m: TaskMetadata): Task {
  // Build a plain object matching Task shape
  return Object.assign(Object.create(null), {
    id: m.taskId,
    description: m.description,
    status: (m.status || "pending") as TaskStatus,
    plan_id: m.planId,
    parent_id: m.parentId,
    children: tryParseJsonArray(m.children),
    depends_on: tryParseJsonArray(m.dependsOn),
    priority: (m.priority || "normal") as TaskPriority,
    assigned_to: m.assignedTo,
    iterations: m.iterations,
    summary: m.summary,
    created_at: m.createdAt,
    updated_at: m.updatedAt,
    started_at: m.startedAt,
    completed_at: m.completedAt,
  }) as Task;
}

function tryParseJsonArray(json: string): string[] {
  try {
    return JSON.parse(json) as string[];
  } catch {
    return [];
  }
}

function tasksFieldDefs(): FieldDef[] {
  return [
    requiredField("task_id", FieldTypes.Utf8),
    requiredField("conversation_id", FieldTypes.Utf8),
    optionalField("plan_id", FieldTypes.Utf8),
    requiredField("description", FieldTypes.Utf8),
    requiredField("status", FieldTypes.Utf8),
    optionalField("parent_id", FieldTypes.Utf8),
    requiredField("children", FieldTypes.Utf8),
    requiredField("depends_on", FieldTypes.Utf8),
    requiredField("priority", FieldTypes.Utf8),
    optionalField("assigned_to", FieldTypes.Utf8),
    requiredField("iterations", FieldTypes.Int32),
    optionalField("summary", FieldTypes.Utf8),
    requiredField("created_at", FieldTypes.Int64),
    requiredField("updated_at", FieldTypes.Int64),
    optionalField("started_at", FieldTypes.Int64),
    optionalField("completed_at", FieldTypes.Int64),
  ];
}

function taskToRecord(m: TaskMetadata): Record {
  return [
    ["task_id", FieldValues.Utf8(m.taskId)],
    ["conversation_id", FieldValues.Utf8(m.conversationId)],
    ["plan_id", FieldValues.Utf8(m.planId ?? null)],
    ["description", FieldValues.Utf8(m.description)],
    ["status", FieldValues.Utf8(m.status)],
    ["parent_id", FieldValues.Utf8(m.parentId ?? null)],
    ["children", FieldValues.Utf8(m.children)],
    ["depends_on", FieldValues.Utf8(m.dependsOn)],
    ["priority", FieldValues.Utf8(m.priority)],
    ["assigned_to", FieldValues.Utf8(m.assignedTo ?? null)],
    ["iterations", FieldValues.Int32(m.iterations)],
    ["summary", FieldValues.Utf8(m.summary ?? null)],
    ["created_at", FieldValues.Int64(m.createdAt)],
    ["updated_at", FieldValues.Int64(m.updatedAt)],
    ["started_at", FieldValues.Int64(m.startedAt ?? null)],
    ["completed_at", FieldValues.Int64(m.completedAt ?? null)],
  ];
}

function taskFromRecord(r: Record): TaskMetadata {
  return {
    taskId: fieldValueAsStr(recordGet(r, "task_id")!)!,
    conversationId: fieldValueAsStr(recordGet(r, "conversation_id")!)!,
    planId: recordGet(r, "plan_id")
      ? fieldValueAsStr(recordGet(r, "plan_id")!)
      : undefined,
    description: fieldValueAsStr(recordGet(r, "description")!)!,
    status: fieldValueAsStr(recordGet(r, "status")!)!,
    parentId: recordGet(r, "parent_id")
      ? fieldValueAsStr(recordGet(r, "parent_id")!)
      : undefined,
    children: fieldValueAsStr(recordGet(r, "children")!) ?? "[]",
    dependsOn: fieldValueAsStr(recordGet(r, "depends_on")!) ?? "[]",
    priority: fieldValueAsStr(recordGet(r, "priority")!) ?? "normal",
    assignedTo: recordGet(r, "assigned_to")
      ? fieldValueAsStr(recordGet(r, "assigned_to")!)
      : undefined,
    iterations: fieldValueAsI32(recordGet(r, "iterations")!) ?? 0,
    summary: recordGet(r, "summary")
      ? fieldValueAsStr(recordGet(r, "summary")!)
      : undefined,
    createdAt: fieldValueAsI64(recordGet(r, "created_at")!)!,
    updatedAt: fieldValueAsI64(recordGet(r, "updated_at")!)!,
    startedAt: recordGet(r, "started_at")
      ? fieldValueAsI64(recordGet(r, "started_at")!)
      : undefined,
    completedAt: recordGet(r, "completed_at")
      ? fieldValueAsI64(recordGet(r, "completed_at")!)
      : undefined,
  };
}

// -- AgentStateMetadata -----------------------------------------------------

/** Metadata for storing agent state. */
export interface AgentStateMetadata {
  /** Unique id of the agent instance whose state this is; the key `get` and `delete` use. */
  agentId: string;
  /** Id of the task the agent is working on (`getByTask` looks up by it). */
  taskId: string;
  /** Id of the conversation the agent is running in. */
  conversationId: string;
  /** Agent lifecycle status as a free-form string chosen by the caller (stored verbatim, not validated). */
  status: string;
  /** Iteration counter the agent had reached when the state was saved. */
  iteration: number;
  /** The agent's execution context, JSON-serialized by the caller (stored opaquely). */
  contextJson: string;
  /** Time the state was first created, as a Unix timestamp in seconds. */
  createdAt: number;
  /** Time the state was last saved, as a Unix timestamp in seconds. */
  updatedAt: number;
}

function agentStatesFieldDefs(): FieldDef[] {
  return [
    requiredField("agent_id", FieldTypes.Utf8),
    requiredField("task_id", FieldTypes.Utf8),
    requiredField("conversation_id", FieldTypes.Utf8),
    requiredField("status", FieldTypes.Utf8),
    requiredField("iteration", FieldTypes.Int32),
    requiredField("context_json", FieldTypes.Utf8),
    requiredField("created_at", FieldTypes.Int64),
    requiredField("updated_at", FieldTypes.Int64),
  ];
}

function stateToRecord(s: AgentStateMetadata): Record {
  return [
    ["agent_id", FieldValues.Utf8(s.agentId)],
    ["task_id", FieldValues.Utf8(s.taskId)],
    ["conversation_id", FieldValues.Utf8(s.conversationId)],
    ["status", FieldValues.Utf8(s.status)],
    ["iteration", FieldValues.Int32(s.iteration)],
    ["context_json", FieldValues.Utf8(s.contextJson)],
    ["created_at", FieldValues.Int64(s.createdAt)],
    ["updated_at", FieldValues.Int64(s.updatedAt)],
  ];
}

function stateFromRecord(r: Record): AgentStateMetadata {
  return {
    agentId: fieldValueAsStr(recordGet(r, "agent_id")!)!,
    taskId: fieldValueAsStr(recordGet(r, "task_id")!)!,
    conversationId: fieldValueAsStr(recordGet(r, "conversation_id")!)!,
    status: fieldValueAsStr(recordGet(r, "status")!)!,
    iteration: fieldValueAsI32(recordGet(r, "iteration")!) ?? 0,
    contextJson: fieldValueAsStr(recordGet(r, "context_json")!)!,
    createdAt: fieldValueAsI64(recordGet(r, "created_at")!)!,
    updatedAt: fieldValueAsI64(recordGet(r, "updated_at")!)!,
  };
}

// -- TaskStore --------------------------------------------------------------

/** Interface for task store operations. */
export interface TaskStoreI {
  /** Create the backing `tasks` table if it does not already exist. */
  ensureTable(): Promise<void>;
  /** Insert the task, replacing any stored task with the same `taskId`. */
  save(task: TaskMetadata): Promise<void>;
  /** Look up a task by `taskId`; `undefined` when none has that id. */
  get(taskId: string): Promise<TaskMetadata | undefined>;
  /** Every task whose `conversationId` matches, in backend order (no sorting applied). */
  getByConversation(conversationId: string): Promise<TaskMetadata[]>;
  /** Every task whose `planId` matches, in backend order (no sorting applied). */
  getByPlan(planId: string): Promise<TaskMetadata[]>;
  /** Delete one task by `taskId` (no-op when absent). */
  delete(taskId: string): Promise<void>;
  /** Delete every task belonging to a conversation. */
  deleteByConversation(conversationId: string): Promise<void>;
  /** Delete every task created from a plan. */
  deleteByPlan(planId: string): Promise<void>;
}

/** Store for managing tasks. */
export class TaskStore implements TaskStoreI {
  /**
   * Create a store over `backend`; call {@link ensureTable} before use.
   * @param backend Storage backend that holds the `tasks` table.
   */
  constructor(private readonly backend: StorageBackend) {}

  /** Create the `tasks` table on the backend if it is missing. */
  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(TASK_TABLE, tasksFieldDefs());
  }

  /** Delete any stored row with the same `taskId` (errors ignored), then insert `task`. */
  async save(task: TaskMetadata): Promise<void> {
    // Delete existing task with same ID first
    try {
      await this.delete(task.taskId);
    } catch { /* ignore */ }
    await this.backend.insert(TASK_TABLE, [taskToRecord(task)]);
  }

  /** Query the backend for the single record whose `task_id` matches. */
  async get(taskId: string): Promise<TaskMetadata | undefined> {
    const filter = Filters.Eq("task_id", FieldValues.Utf8(taskId));
    const records = await this.backend.query(TASK_TABLE, filter, 1);
    return records.length > 0 ? taskFromRecord(records[0]) : undefined;
  }

  /** Query every record whose `conversation_id` matches. */
  async getByConversation(conversationId: string): Promise<TaskMetadata[]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TASK_TABLE, filter);
    return records.map(taskFromRecord);
  }

  /** Query every record whose `plan_id` matches. */
  async getByPlan(planId: string): Promise<TaskMetadata[]> {
    const filter = Filters.Eq("plan_id", FieldValues.Utf8(planId));
    const records = await this.backend.query(TASK_TABLE, filter);
    return records.map(taskFromRecord);
  }

  /** Delete the record whose `task_id` matches. */
  async delete(taskId: string): Promise<void> {
    const filter = Filters.Eq("task_id", FieldValues.Utf8(taskId));
    await this.backend.delete(TASK_TABLE, filter);
  }

  /** Delete every record whose `conversation_id` matches. */
  async deleteByConversation(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TASK_TABLE, filter);
  }

  /** Delete every record whose `plan_id` matches. */
  async deleteByPlan(planId: string): Promise<void> {
    const filter = Filters.Eq("plan_id", FieldValues.Utf8(planId));
    await this.backend.delete(TASK_TABLE, filter);
  }
}

/** In-memory task store for testing. */
export class InMemoryTaskStore implements TaskStoreI {
  private tasks: Map<string, TaskMetadata> = new Map();

  /** No-op: there is no table to create in memory. */
  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  /** Store a shallow copy of `task` under its `taskId`, replacing any existing entry. */
  async save(task: TaskMetadata): Promise<void> {
    this.tasks.set(task.taskId, { ...task });
    await Promise.resolve();
  }

  /** Return the stored task for `taskId`, if any. */
  async get(taskId: string): Promise<TaskMetadata | undefined> {
    return await Promise.resolve(this.tasks.get(taskId));
  }

  /** Stored tasks whose `conversationId` matches, in insertion order. */
  async getByConversation(conversationId: string): Promise<TaskMetadata[]> {
    return await Promise.resolve(
      [...this.tasks.values()].filter((t) =>
        t.conversationId === conversationId
      ),
    );
  }

  /** Stored tasks whose `planId` matches, in insertion order. */
  async getByPlan(planId: string): Promise<TaskMetadata[]> {
    return await Promise.resolve(
      [...this.tasks.values()].filter((t) => t.planId === planId),
    );
  }

  /** Remove the task from the map (no-op when absent). */
  async delete(taskId: string): Promise<void> {
    this.tasks.delete(taskId);
    await Promise.resolve();
  }

  /** Remove every stored task whose `conversationId` matches. */
  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, t] of this.tasks) {
      if (t.conversationId === conversationId) this.tasks.delete(id);
    }
    await Promise.resolve();
  }

  /** Remove every stored task whose `planId` matches. */
  async deleteByPlan(planId: string): Promise<void> {
    for (const [id, t] of this.tasks) {
      if (t.planId === planId) this.tasks.delete(id);
    }
    await Promise.resolve();
  }
}

// -- AgentStateStore --------------------------------------------------------

/** Interface for agent state store operations. */
export interface AgentStateStoreI {
  /** Create the backing `agent_states` table if it does not already exist. */
  ensureTable(): Promise<void>;
  /** Insert the state, replacing any stored state with the same `agentId`. */
  save(state: AgentStateMetadata): Promise<void>;
  /** Look up an agent's state by `agentId`; `undefined` when none has that id. */
  get(agentId: string): Promise<AgentStateMetadata | undefined>;
  /** Every stored state whose `conversationId` matches, in backend order. */
  getByConversation(conversationId: string): Promise<AgentStateMetadata[]>;
  /** The first stored state whose `taskId` matches; `undefined` when there is none. */
  getByTask(taskId: string): Promise<AgentStateMetadata | undefined>;
  /** Delete one agent's state by `agentId` (no-op when absent). */
  delete(agentId: string): Promise<void>;
  /** Delete every stored state belonging to a conversation. */
  deleteByConversation(conversationId: string): Promise<void>;
}

/** Store for managing agent state persistence. */
export class AgentStateStore implements AgentStateStoreI {
  /**
   * Create a store over `backend`; call {@link ensureTable} before use.
   * @param backend Storage backend that holds the `agent_states` table.
   */
  constructor(private readonly backend: StorageBackend) {}

  /** Create the `agent_states` table on the backend if it is missing. */
  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(AGENT_STATE_TABLE, agentStatesFieldDefs());
  }

  /** Delete any stored row with the same `agentId` (errors ignored), then insert `state`. */
  async save(state: AgentStateMetadata): Promise<void> {
    try {
      await this.delete(state.agentId);
    } catch { /* ignore */ }
    await this.backend.insert(AGENT_STATE_TABLE, [stateToRecord(state)]);
  }

  /** Query the backend for the single record whose `agent_id` matches. */
  async get(agentId: string): Promise<AgentStateMetadata | undefined> {
    const filter = Filters.Eq("agent_id", FieldValues.Utf8(agentId));
    const records = await this.backend.query(AGENT_STATE_TABLE, filter, 1);
    return records.length > 0 ? stateFromRecord(records[0]) : undefined;
  }

  /** Query every record whose `conversation_id` matches. */
  async getByConversation(
    conversationId: string,
  ): Promise<AgentStateMetadata[]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(AGENT_STATE_TABLE, filter);
    return records.map(stateFromRecord);
  }

  /** Query the backend for the first record whose `task_id` matches (limit 1). */
  async getByTask(taskId: string): Promise<AgentStateMetadata | undefined> {
    const filter = Filters.Eq("task_id", FieldValues.Utf8(taskId));
    const records = await this.backend.query(AGENT_STATE_TABLE, filter, 1);
    return records.length > 0 ? stateFromRecord(records[0]) : undefined;
  }

  /** Delete the record whose `agent_id` matches. */
  async delete(agentId: string): Promise<void> {
    const filter = Filters.Eq("agent_id", FieldValues.Utf8(agentId));
    await this.backend.delete(AGENT_STATE_TABLE, filter);
  }

  /** Delete every record whose `conversation_id` matches. */
  async deleteByConversation(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(AGENT_STATE_TABLE, filter);
  }
}

/** In-memory agent state store for testing. */
export class InMemoryAgentStateStore implements AgentStateStoreI {
  private states: Map<string, AgentStateMetadata> = new Map();

  /** No-op: there is no table to create in memory. */
  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  /** Store a shallow copy of `state` under its `agentId`, replacing any existing entry. */
  async save(state: AgentStateMetadata): Promise<void> {
    this.states.set(state.agentId, { ...state });
    await Promise.resolve();
  }

  /** Return the stored state for `agentId`, if any. */
  async get(agentId: string): Promise<AgentStateMetadata | undefined> {
    return await Promise.resolve(this.states.get(agentId));
  }

  /** Stored states whose `conversationId` matches, in insertion order. */
  async getByConversation(
    conversationId: string,
  ): Promise<AgentStateMetadata[]> {
    return await Promise.resolve(
      [...this.states.values()].filter((s) =>
        s.conversationId === conversationId
      ),
    );
  }

  /** The first stored state (insertion order) whose `taskId` matches, if any. */
  async getByTask(taskId: string): Promise<AgentStateMetadata | undefined> {
    return await Promise.resolve(
      [...this.states.values()].find((s) => s.taskId === taskId),
    );
  }

  /** Remove the state from the map (no-op when absent). */
  async delete(agentId: string): Promise<void> {
    this.states.delete(agentId);
    await Promise.resolve();
  }

  /** Remove every stored state whose `conversationId` matches. */
  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, s] of this.states) {
      if (s.conversationId === conversationId) this.states.delete(id);
    }
    await Promise.resolve();
  }
}
