/**
 * Conversation Store -- persists conversation metadata.
 *
 * Equivalent to Rust's `stores/conversation_store.rs` in rullama-storage.
 * @module
 */

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

const TABLE_NAME = "conversations";

/** Metadata for a conversation. */
export interface ConversationMetadata {
  conversationId: string;
  title?: string;
  modelId?: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
}

function tableSchema(): FieldDef[] {
  return [
    requiredField("conversation_id", FieldTypes.Utf8),
    optionalField("title", FieldTypes.Utf8),
    optionalField("model_id", FieldTypes.Utf8),
    requiredField("created_at", FieldTypes.Int64),
    requiredField("updated_at", FieldTypes.Int64),
    requiredField("message_count", FieldTypes.Int32),
  ];
}

function toRecord(m: ConversationMetadata): Record {
  return [
    ["conversation_id", FieldValues.Utf8(m.conversationId)],
    ["title", FieldValues.Utf8(m.title ?? null)],
    ["model_id", FieldValues.Utf8(m.modelId ?? null)],
    ["created_at", FieldValues.Int64(m.createdAt)],
    ["updated_at", FieldValues.Int64(m.updatedAt)],
    ["message_count", FieldValues.Int32(m.messageCount)],
  ];
}

function fromRecord(r: Record): ConversationMetadata {
  const conversationId = recordGet(r, "conversation_id");
  const createdAt = recordGet(r, "created_at");
  const updatedAt = recordGet(r, "updated_at");
  const messageCount = recordGet(r, "message_count");

  if (!conversationId || !createdAt || !updatedAt || !messageCount) {
    throw new Error("Missing required fields in conversation record");
  }

  return {
    conversationId: fieldValueAsStr(conversationId)!,
    title: recordGet(r, "title")
      ? fieldValueAsStr(recordGet(r, "title")!)
      : undefined,
    modelId: recordGet(r, "model_id")
      ? fieldValueAsStr(recordGet(r, "model_id")!)
      : undefined,
    createdAt: fieldValueAsI64(createdAt)!,
    updatedAt: fieldValueAsI64(updatedAt)!,
    messageCount: fieldValueAsI32(messageCount)!,
  };
}

/** Interface for conversation store operations. */
export interface ConversationStoreI {
  ensureTable(): Promise<void>;
  create(
    conversationId: string,
    title?: string,
    modelId?: string,
    messageCount?: number,
  ): Promise<ConversationMetadata>;
  get(conversationId: string): Promise<ConversationMetadata | undefined>;
  list(limit?: number): Promise<ConversationMetadata[]>;
  update(
    conversationId: string,
    title?: string,
    messageCount?: number,
  ): Promise<void>;
  delete(conversationId: string): Promise<void>;
}

/** Store for managing conversations. */
export class ConversationStore implements ConversationStoreI {
  constructor(private readonly backend: StorageBackend) {}

  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(TABLE_NAME, tableSchema());
  }

  async create(
    conversationId: string,
    title?: string,
    modelId?: string,
    messageCount?: number,
  ): Promise<ConversationMetadata> {
    // Check if conversation already exists - if so, just update
    const existing = await this.get(conversationId);
    if (existing) {
      await this.update(conversationId, title ?? existing.title, messageCount);
      const updated = await this.get(conversationId);
      if (!updated) throw new Error("Conversation should exist after update");
      return updated;
    }

    const now = Math.floor(Date.now() / 1000);
    const metadata: ConversationMetadata = {
      conversationId,
      title,
      modelId,
      createdAt: now,
      updatedAt: now,
      messageCount: messageCount ?? 0,
    };

    await this.backend.insert(TABLE_NAME, [toRecord(metadata)]);
    return metadata;
  }

  async get(conversationId: string): Promise<ConversationMetadata | undefined> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TABLE_NAME, filter, 1);
    return records.length > 0 ? fromRecord(records[0]) : undefined;
  }

  async list(limit?: number): Promise<ConversationMetadata[]> {
    const records = await this.backend.query(TABLE_NAME);
    let conversations = records.map(fromRecord);
    conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    if (limit !== undefined) {
      conversations = conversations.slice(0, limit);
    }
    return conversations;
  }

  async update(
    conversationId: string,
    title?: string,
    messageCount?: number,
  ): Promise<void> {
    const current = await this.get(conversationId);
    if (!current) throw new Error("Conversation not found");

    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);

    const updated: ConversationMetadata = {
      conversationId,
      title: title ?? current.title,
      modelId: current.modelId,
      createdAt: current.createdAt,
      updatedAt: Math.floor(Date.now() / 1000),
      messageCount: messageCount ?? current.messageCount,
    };

    await this.backend.insert(TABLE_NAME, [toRecord(updated)]);
  }

  async delete(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);
  }
}

/** In-memory conversation store for testing. */
export class InMemoryConversationStore implements ConversationStoreI {
  private conversations: Map<string, ConversationMetadata> = new Map();

  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  async create(
    conversationId: string,
    title?: string,
    modelId?: string,
    messageCount?: number,
  ): Promise<ConversationMetadata> {
    const existing = this.conversations.get(conversationId);
    if (existing) {
      await this.update(conversationId, title ?? existing.title, messageCount);
      return this.conversations.get(conversationId)!;
    }

    const now = Math.floor(Date.now() / 1000);
    const metadata: ConversationMetadata = {
      conversationId,
      title,
      modelId,
      createdAt: now,
      updatedAt: now,
      messageCount: messageCount ?? 0,
    };
    this.conversations.set(conversationId, metadata);
    return await Promise.resolve(metadata);
  }

  async get(conversationId: string): Promise<ConversationMetadata | undefined> {
    return await Promise.resolve(this.conversations.get(conversationId));
  }

  async list(limit?: number): Promise<ConversationMetadata[]> {
    let result = [...this.conversations.values()];
    result.sort((a, b) => b.updatedAt - a.updatedAt);
    if (limit !== undefined) result = result.slice(0, limit);
    return await Promise.resolve(result);
  }

  async update(
    conversationId: string,
    title?: string,
    messageCount?: number,
  ): Promise<void> {
    const current = this.conversations.get(conversationId);
    if (!current) throw new Error("Conversation not found");

    this.conversations.set(conversationId, {
      ...current,
      title: title ?? current.title,
      messageCount: messageCount ?? current.messageCount,
      updatedAt: Math.floor(Date.now() / 1000),
    });
    await Promise.resolve();
  }

  async delete(conversationId: string): Promise<void> {
    this.conversations.delete(conversationId);
    await Promise.resolve();
  }
}
