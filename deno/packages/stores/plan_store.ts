/**
 * Plan Store -- persists execution plans with conversation association.
 *
 * Equivalent to Rust's `stores/plan_store.rs` in rullama-storage.
 * @module
 */

import type { PlanMetadata, PlanStatus } from "@rullama/core";
import { parsePlanStatus } from "@rullama/core";
import type { EmbeddingProvider } from "@rullama/core";
import type { StorageBackend } from "@rullama/storage";
import {
  type FieldDef,
  FieldTypes,
  fieldValueAsBool,
  fieldValueAsI32,
  fieldValueAsI64,
  fieldValueAsStr,
  fieldValueAsVector,
  FieldValues,
  Filters,
  optionalField,
  type Record,
  recordGet,
  requiredField,
} from "@rullama/storage";

const TABLE_NAME = "plans";

function plansFieldDefs(embeddingDim: number): FieldDef[] {
  return [
    requiredField("plan_id", FieldTypes.Utf8),
    requiredField("conversation_id", FieldTypes.Utf8),
    requiredField("title", FieldTypes.Utf8),
    requiredField("task_description", FieldTypes.Utf8),
    requiredField("plan_content", FieldTypes.Utf8),
    optionalField("model_id", FieldTypes.Utf8),
    requiredField("status", FieldTypes.Utf8),
    requiredField("executed", FieldTypes.Boolean),
    requiredField("iterations_used", FieldTypes.Int32),
    requiredField("created_at", FieldTypes.Int64),
    requiredField("updated_at", FieldTypes.Int64),
    optionalField("file_path", FieldTypes.Utf8),
    optionalField("parent_plan_id", FieldTypes.Utf8),
    optionalField("child_plan_ids", FieldTypes.Utf8),
    optionalField("branch_name", FieldTypes.Utf8),
    requiredField("merged", FieldTypes.Boolean),
    requiredField("depth", FieldTypes.Int32),
    optionalField("embedding", FieldTypes.Vector(embeddingDim)),
  ];
}

function toRecord(plan: PlanMetadata): Record {
  const childPlanIdsJson = JSON.stringify(plan.child_plan_ids);
  return [
    ["plan_id", FieldValues.Utf8(plan.plan_id)],
    ["conversation_id", FieldValues.Utf8(plan.conversation_id)],
    ["title", FieldValues.Utf8(plan.title)],
    ["task_description", FieldValues.Utf8(plan.task_description)],
    ["plan_content", FieldValues.Utf8(plan.plan_content)],
    ["model_id", FieldValues.Utf8(plan.model_id ?? null)],
    ["status", FieldValues.Utf8(plan.status)],
    ["executed", FieldValues.Boolean(plan.executed)],
    ["iterations_used", FieldValues.Int32(plan.iterations_used)],
    ["created_at", FieldValues.Int64(plan.created_at)],
    ["updated_at", FieldValues.Int64(plan.updated_at)],
    ["file_path", FieldValues.Utf8(plan.file_path ?? null)],
    ["parent_plan_id", FieldValues.Utf8(plan.parent_plan_id ?? null)],
    ["child_plan_ids", FieldValues.Utf8(childPlanIdsJson)],
    ["branch_name", FieldValues.Utf8(plan.branch_name ?? null)],
    ["merged", FieldValues.Boolean(plan.merged)],
    ["depth", FieldValues.Int32(plan.depth)],
    ["embedding", FieldValues.Vector(plan.embedding ?? [])],
  ];
}

function fromRecord(r: Record): PlanMetadata {
  const statusStr = fieldValueAsStr(recordGet(r, "status")!) ?? "draft";
  const status: PlanStatus = parsePlanStatus(statusStr) ?? "draft";

  const childPlanIds: string[] = (() => {
    const json = recordGet(r, "child_plan_ids");
    if (!json) return [];
    const str = fieldValueAsStr(json);
    if (!str) return [];
    try {
      return JSON.parse(str) as string[];
    } catch {
      return [];
    }
  })();

  const embeddingFv = recordGet(r, "embedding");
  const embedding = embeddingFv ? fieldValueAsVector(embeddingFv) : undefined;

  // Build a PlanMetadata-compatible object
  const result = Object.create(null) as PlanMetadata;
  result.plan_id = fieldValueAsStr(recordGet(r, "plan_id")!)!;
  result.conversation_id = fieldValueAsStr(recordGet(r, "conversation_id")!)!;
  result.title = fieldValueAsStr(recordGet(r, "title")!)!;
  result.task_description = fieldValueAsStr(recordGet(r, "task_description")!)!;
  result.plan_content = fieldValueAsStr(recordGet(r, "plan_content")!)!;
  result.model_id = recordGet(r, "model_id")
    ? fieldValueAsStr(recordGet(r, "model_id")!)
    : undefined;
  result.status = status;
  result.executed = fieldValueAsBool(recordGet(r, "executed")!) ?? false;
  result.iterations_used = fieldValueAsI32(recordGet(r, "iterations_used")!) ??
    0;
  result.created_at = fieldValueAsI64(recordGet(r, "created_at")!)!;
  result.updated_at = fieldValueAsI64(recordGet(r, "updated_at")!)!;
  result.file_path = recordGet(r, "file_path")
    ? fieldValueAsStr(recordGet(r, "file_path")!)
    : undefined;
  result.embedding = embedding;
  result.parent_plan_id = recordGet(r, "parent_plan_id")
    ? fieldValueAsStr(recordGet(r, "parent_plan_id")!)
    : undefined;
  result.child_plan_ids = childPlanIds;
  result.branch_name = recordGet(r, "branch_name")
    ? fieldValueAsStr(recordGet(r, "branch_name")!)
    : undefined;
  result.merged = fieldValueAsBool(recordGet(r, "merged")!) ?? false;
  result.depth = fieldValueAsI32(recordGet(r, "depth")!) ?? 0;
  return result;
}

/** Interface for plan store operations. */
export interface PlanStoreI {
  ensureTable(): Promise<void>;
  save(plan: PlanMetadata): Promise<void>;
  get(planId: string): Promise<PlanMetadata | undefined>;
  getByConversation(conversationId: string): Promise<PlanMetadata[]>;
  listRecent(limit: number): Promise<PlanMetadata[]>;
  delete(planId: string): Promise<void>;
  deleteByConversation(conversationId: string): Promise<void>;
}

/** Store for managing execution plans. */
export class PlanStore implements PlanStoreI {
  constructor(
    private readonly backend: StorageBackend,
    private readonly embeddings: EmbeddingProvider,
  ) {}

  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(
      TABLE_NAME,
      plansFieldDefs(this.embeddings.dimension),
    );
  }

  async save(plan: PlanMetadata): Promise<void> {
    try {
      await this.delete(plan.plan_id);
    } catch { /* ignore */ }

    // Generate embedding if not already present
    if (!plan.embedding) {
      const text = `${plan.title} ${plan.task_description}`;
      plan.embedding = await this.embeddings.embed(text);
    }

    await this.backend.insert(TABLE_NAME, [toRecord(plan)]);
  }

  async get(planId: string): Promise<PlanMetadata | undefined> {
    const filter = Filters.Eq("plan_id", FieldValues.Utf8(planId));
    const records = await this.backend.query(TABLE_NAME, filter, 1);
    return records.length > 0 ? fromRecord(records[0]) : undefined;
  }

  async getByConversation(conversationId: string): Promise<PlanMetadata[]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TABLE_NAME, filter);
    const plans = records.map(fromRecord);
    plans.sort((a, b) => b.created_at - a.created_at);
    return plans;
  }

  async listRecent(limit: number): Promise<PlanMetadata[]> {
    const records = await this.backend.query(TABLE_NAME, undefined, limit * 2);
    const plans = records.map(fromRecord);
    plans.sort((a, b) => b.created_at - a.created_at);
    return plans.slice(0, limit);
  }

  async delete(planId: string): Promise<void> {
    const filter = Filters.Eq("plan_id", FieldValues.Utf8(planId));
    await this.backend.delete(TABLE_NAME, filter);
  }

  async deleteByConversation(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);
  }

  /** Search plans by semantic similarity. */
  async search(query: string, limit: number): Promise<PlanMetadata[]> {
    const queryEmbedding = await this.embeddings.embed(query);
    const scored = await this.backend.vectorSearch(
      TABLE_NAME,
      "embedding",
      queryEmbedding,
      limit,
    );
    return scored.map((sr) => fromRecord(sr.record));
  }
}

/** In-memory plan store for testing. */
export class InMemoryPlanStore implements PlanStoreI {
  private plans: Map<string, PlanMetadata> = new Map();

  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  async save(plan: PlanMetadata): Promise<void> {
    this.plans.set(plan.plan_id, plan);
    await Promise.resolve();
  }

  async get(planId: string): Promise<PlanMetadata | undefined> {
    return await Promise.resolve(this.plans.get(planId));
  }

  async getByConversation(conversationId: string): Promise<PlanMetadata[]> {
    const plans = [...this.plans.values()]
      .filter((p) => p.conversation_id === conversationId);
    plans.sort((a, b) => b.created_at - a.created_at);
    return await Promise.resolve(plans);
  }

  async listRecent(limit: number): Promise<PlanMetadata[]> {
    const plans = [...this.plans.values()];
    plans.sort((a, b) => b.created_at - a.created_at);
    return await Promise.resolve(plans.slice(0, limit));
  }

  async delete(planId: string): Promise<void> {
    this.plans.delete(planId);
    await Promise.resolve();
  }

  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, p] of this.plans) {
      if (p.conversation_id === conversationId) this.plans.delete(id);
    }
    await Promise.resolve();
  }
}
