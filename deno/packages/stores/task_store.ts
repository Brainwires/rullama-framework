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
  taskId: string;
  conversationId: string;
  planId?: string;
  description: string;
  status: string;
  parentId?: string;
  children: string; // JSON array
  dependsOn: string; // JSON array
  priority: string;
  assignedTo?: string;
  iterations: number;
  summary?: string;
  createdAt: number;
  updatedAt: number;
  startedAt?: number;
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
  agentId: string;
  taskId: string;
  conversationId: string;
  status: string;
  iteration: number;
  contextJson: string;
  createdAt: number;
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
  ensureTable(): Promise<void>;
  save(task: TaskMetadata): Promise<void>;
  get(taskId: string): Promise<TaskMetadata | undefined>;
  getByConversation(conversationId: string): Promise<TaskMetadata[]>;
  getByPlan(planId: string): Promise<TaskMetadata[]>;
  delete(taskId: string): Promise<void>;
  deleteByConversation(conversationId: string): Promise<void>;
  deleteByPlan(planId: string): Promise<void>;
}

/** Store for managing tasks. */
export class TaskStore implements TaskStoreI {
  constructor(private readonly backend: StorageBackend) {}

  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(TASK_TABLE, tasksFieldDefs());
  }

  async save(task: TaskMetadata): Promise<void> {
    // Delete existing task with same ID first
    try {
      await this.delete(task.taskId);
    } catch { /* ignore */ }
    await this.backend.insert(TASK_TABLE, [taskToRecord(task)]);
  }

  async get(taskId: string): Promise<TaskMetadata | undefined> {
    const filter = Filters.Eq("task_id", FieldValues.Utf8(taskId));
    const records = await this.backend.query(TASK_TABLE, filter, 1);
    return records.length > 0 ? taskFromRecord(records[0]) : undefined;
  }

  async getByConversation(conversationId: string): Promise<TaskMetadata[]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TASK_TABLE, filter);
    return records.map(taskFromRecord);
  }

  async getByPlan(planId: string): Promise<TaskMetadata[]> {
    const filter = Filters.Eq("plan_id", FieldValues.Utf8(planId));
    const records = await this.backend.query(TASK_TABLE, filter);
    return records.map(taskFromRecord);
  }

  async delete(taskId: string): Promise<void> {
    const filter = Filters.Eq("task_id", FieldValues.Utf8(taskId));
    await this.backend.delete(TASK_TABLE, filter);
  }

  async deleteByConversation(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TASK_TABLE, filter);
  }

  async deleteByPlan(planId: string): Promise<void> {
    const filter = Filters.Eq("plan_id", FieldValues.Utf8(planId));
    await this.backend.delete(TASK_TABLE, filter);
  }
}

/** In-memory task store for testing. */
export class InMemoryTaskStore implements TaskStoreI {
  private tasks: Map<string, TaskMetadata> = new Map();

  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  async save(task: TaskMetadata): Promise<void> {
    this.tasks.set(task.taskId, { ...task });
    await Promise.resolve();
  }

  async get(taskId: string): Promise<TaskMetadata | undefined> {
    return await Promise.resolve(this.tasks.get(taskId));
  }

  async getByConversation(conversationId: string): Promise<TaskMetadata[]> {
    return await Promise.resolve(
      [...this.tasks.values()].filter((t) =>
        t.conversationId === conversationId
      ),
    );
  }

  async getByPlan(planId: string): Promise<TaskMetadata[]> {
    return await Promise.resolve(
      [...this.tasks.values()].filter((t) => t.planId === planId),
    );
  }

  async delete(taskId: string): Promise<void> {
    this.tasks.delete(taskId);
    await Promise.resolve();
  }

  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, t] of this.tasks) {
      if (t.conversationId === conversationId) this.tasks.delete(id);
    }
    await Promise.resolve();
  }

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
  ensureTable(): Promise<void>;
  save(state: AgentStateMetadata): Promise<void>;
  get(agentId: string): Promise<AgentStateMetadata | undefined>;
  getByConversation(conversationId: string): Promise<AgentStateMetadata[]>;
  getByTask(taskId: string): Promise<AgentStateMetadata | undefined>;
  delete(agentId: string): Promise<void>;
  deleteByConversation(conversationId: string): Promise<void>;
}

/** Store for managing agent state persistence. */
export class AgentStateStore implements AgentStateStoreI {
  constructor(private readonly backend: StorageBackend) {}

  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(AGENT_STATE_TABLE, agentStatesFieldDefs());
  }

  async save(state: AgentStateMetadata): Promise<void> {
    try {
      await this.delete(state.agentId);
    } catch { /* ignore */ }
    await this.backend.insert(AGENT_STATE_TABLE, [stateToRecord(state)]);
  }

  async get(agentId: string): Promise<AgentStateMetadata | undefined> {
    const filter = Filters.Eq("agent_id", FieldValues.Utf8(agentId));
    const records = await this.backend.query(AGENT_STATE_TABLE, filter, 1);
    return records.length > 0 ? stateFromRecord(records[0]) : undefined;
  }

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

  async getByTask(taskId: string): Promise<AgentStateMetadata | undefined> {
    const filter = Filters.Eq("task_id", FieldValues.Utf8(taskId));
    const records = await this.backend.query(AGENT_STATE_TABLE, filter, 1);
    return records.length > 0 ? stateFromRecord(records[0]) : undefined;
  }

  async delete(agentId: string): Promise<void> {
    const filter = Filters.Eq("agent_id", FieldValues.Utf8(agentId));
    await this.backend.delete(AGENT_STATE_TABLE, filter);
  }

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

  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  async save(state: AgentStateMetadata): Promise<void> {
    this.states.set(state.agentId, { ...state });
    await Promise.resolve();
  }

  async get(agentId: string): Promise<AgentStateMetadata | undefined> {
    return await Promise.resolve(this.states.get(agentId));
  }

  async getByConversation(
    conversationId: string,
  ): Promise<AgentStateMetadata[]> {
    return await Promise.resolve(
      [...this.states.values()].filter((s) =>
        s.conversationId === conversationId
      ),
    );
  }

  async getByTask(taskId: string): Promise<AgentStateMetadata | undefined> {
    return await Promise.resolve(
      [...this.states.values()].find((s) => s.taskId === taskId),
    );
  }

  async delete(agentId: string): Promise<void> {
    this.states.delete(agentId);
    await Promise.resolve();
  }

  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, s] of this.states) {
      if (s.conversationId === conversationId) this.states.delete(id);
    }
    await Promise.resolve();
  }
}
